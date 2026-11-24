from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from bsro.model import build_model, load_checkpoint, load_config, residual_name, stem_names

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def window_array(chunk_size: int, fade_size: int) -> torch.Tensor:
    window = torch.ones(chunk_size)
    if fade_size > 0:
        window[:fade_size] = torch.linspace(0.0, 1.0, fade_size)
        window[-fade_size:] = torch.linspace(1.0, 0.0, fade_size)
    return window


@torch.no_grad()
def separate_track(
    model: torch.nn.Module,
    mix: torch.Tensor,
    chunk_size: int,
    num_overlap: int,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Overlap-add chunked inference. mix: (channels, time) float32. Returns (stems, channels, time)."""
    if mix.ndim != 2:
        raise ValueError(f"expected mix of shape (channels, time), got {tuple(mix.shape)}")
    if num_overlap < 1 or batch_size < 1:
        raise ValueError("num_overlap and batch_size must be >= 1")

    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length = mix.shape[-1]
    window = window_array(chunk_size, fade_size).to(device)

    mix = mix.to(device)
    if length > 2 * border and border > 0:
        mix = F.pad(mix, (border, border), mode="reflect")
    padded_length = mix.shape[-1]

    result = None
    counter = torch.zeros(padded_length, dtype=torch.float32, device=device)

    batch, locations = [], []
    start = 0
    while start < padded_length:
        part = mix[:, start : start + chunk_size]
        part_len = part.shape[-1]
        pad_mode = "reflect" if part_len > chunk_size // 2 else "constant"
        part = F.pad(part, (0, chunk_size - part_len), mode=pad_mode)
        batch.append(part)
        locations.append((start, part_len))
        start += step

        if len(batch) < batch_size and start < padded_length:
            continue

        x = torch.stack(batch)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x)
        out = out.float()
        if out.ndim == 3:
            out = out.unsqueeze(1)  # single-stem models return (batch, channels, time)
        if result is None:
            result = torch.zeros((out.shape[1],) + tuple(mix.shape), dtype=torch.float32, device=device)

        for j, (seg_start, seg_len) in enumerate(locations):
            w = window
            first, last = seg_start == 0, seg_start + step >= padded_length
            if fade_size and (first or last):
                w = window.clone()
                if first:
                    w[:fade_size] = 1.0
                if last:
                    w[-fade_size:] = 1.0
            result[..., seg_start : seg_start + seg_len] += out[j, ..., :seg_len] * w[:seg_len]
            counter[seg_start : seg_start + seg_len] += w[:seg_len]
        batch.clear()
        locations.clear()

    estimated = result / counter.clamp_min(1e-8)
    if length > 2 * border and border > 0:
        estimated = estimated[..., border:-border]
    return estimated[..., :length].cpu()


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T)  # (channels, time)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        raise ValueError(f"{path}: expected mono or stereo, got {audio.shape[0]} channels")
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio


def list_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"no audio files with extensions {sorted(AUDIO_EXTENSIONS)} in {path}")
    return files


def git_revision() -> str | None:
    try:
        root = Path(__file__).resolve().parent.parent
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        return rev.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Separate music into stems with a pretrained BS-RoFormer.")
    parser.add_argument("--config", required=True, help="model config yaml (see configs/)")
    parser.add_argument("--checkpoint", required=True, help="checkpoint .ckpt matching the config")
    parser.add_argument("--input", required=True, help="audio file or directory of audio files")
    parser.add_argument("--output", required=True, help="output directory; stems go to <output>/<track>/<stem>.wav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="chunks per forward pass (default: config inference.batch_size)")
    parser.add_argument("--overlap", type=int, default=None, help="chunk overlap factor (default: config inference.num_overlap)")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16", help="autocast dtype on CUDA (ignored on CPU)")
    parser.add_argument("--limit", type=int, default=None, help="process at most N tracks (smoke tests)")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    config = load_config(args.config)
    sample_rate = config["audio"]["sample_rate"]
    chunk_size = config["inference"].get("chunk_size", config["audio"]["chunk_size"])
    num_overlap = args.overlap or config["inference"]["num_overlap"]
    batch_size = args.batch_size or config["inference"]["batch_size"]
    amp_dtype = None
    if device.type == "cuda" and args.amp != "none":
        amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16

    model = load_checkpoint(build_model(config), args.checkpoint).to(device).eval()
    names = stem_names(config)
    residual = residual_name(config)

    inputs = list_inputs(Path(args.input))
    if args.limit is not None:
        inputs = inputs[: args.limit]
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    records = []
    for path in inputs:
        t0 = time.time()
        mix = load_audio(path, sample_rate)
        stems = separate_track(model, mix, chunk_size, num_overlap, batch_size, device, amp_dtype)
        if stems.shape[0] != len(names):
            raise RuntimeError(f"model returned {stems.shape[0]} stems but config names {names}")
        track_dir = out_root / path.stem
        track_dir.mkdir(exist_ok=True)
        written = {}
        for name, stem in zip(names, stems):
            sf.write(str(track_dir / f"{name}.wav"), stem.T.numpy(), sample_rate, subtype="FLOAT")
            written[name] = str(track_dir / f"{name}.wav")
        if residual is not None:
            sf.write(str(track_dir / f"{residual}.wav"), (mix - stems[0]).T.numpy(), sample_rate, subtype="FLOAT")
            written[residual] = str(track_dir / f"{residual}.wav")
        elapsed = time.time() - t0
        records.append({"input": str(path), "duration_s": mix.shape[-1] / sample_rate, "seconds": round(elapsed, 2), "stems": written})
        print(f"{path.name}: {mix.shape[-1] / sample_rate:.1f}s audio in {elapsed:.1f}s -> {track_dir}", flush=True)

    manifest = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "code_revision": git_revision(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": args.amp if amp_dtype is not None else "none",
        "torch": torch.__version__,
        "chunk_size": chunk_size,
        "num_overlap": num_overlap,
        "batch_size": batch_size,
        "tracks": records,
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
