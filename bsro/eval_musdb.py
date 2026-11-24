"""Ground-truth evaluation on MUSDB18-HQ test (SDR per MSST definition, plus SI-SDR)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from bsro.metrics import sdr, si_sdr
from bsro.model import build_model, load_checkpoint, load_config, residual_name, stem_names
from bsro.separate import separate_track


def read(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != 44100:
        raise ValueError(f"{path}: MUSDB18-HQ must be 44.1 kHz, got {sr}")
    return data.T


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a BS-RoFormer checkpoint on MUSDB18-HQ test.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--musdb", required=True, help="MUSDB18-HQ root containing test/<song>/{mixture,vocals,drums,bass,other}.wav")
    p.add_argument("--out", required=True, help="output directory for results csv/json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N songs (smoke test)")
    args = p.parse_args(argv)

    test_dir = Path(args.musdb) / "test"
    songs = sorted(d for d in test_dir.iterdir() if (d / "mixture.wav").exists())
    if not songs:
        raise FileNotFoundError(f"no songs with mixture.wav under {test_dir}")
    if args.limit:
        songs = songs[: args.limit]

    device = torch.device(args.device)
    config = load_config(args.config)
    chunk_size = config["inference"].get("chunk_size", config["audio"]["chunk_size"])
    num_overlap = args.overlap or config["inference"]["num_overlap"]
    batch_size = args.batch_size or config["inference"]["batch_size"]
    amp_dtype = None
    if device.type == "cuda" and args.amp != "none":
        amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    model = load_checkpoint(build_model(config), args.checkpoint).to(device).eval()
    names = stem_names(config)
    residual = residual_name(config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for song in songs:
        t0 = time.time()
        mix = read(song / "mixture.wav")
        est = separate_track(model, torch.from_numpy(mix), chunk_size, num_overlap, batch_size, device, amp_dtype).numpy()
        estimates = dict(zip(names, est))
        if residual is not None:
            # 2-stem model: "other" is everything but vocals, so the reference is mixture minus vocals
            estimates[residual] = mix - est[0]
        row = {"song": song.name}
        for name, estimate in estimates.items():
            if residual is not None and name == residual:
                reference = mix - read(song / "vocals.wav")
            else:
                reference = read(song / f"{name}.wav")
            n = min(reference.shape[-1], estimate.shape[-1])
            row[f"sdr_{name}"] = round(sdr(reference[..., :n], estimate[..., :n]), 3)
            row[f"sisdr_{name}"] = round(si_sdr(reference[..., :n], estimate[..., :n]), 3)
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"{song.name}: " + " ".join(f"{k}={v}" for k, v in row.items() if k != "song"), flush=True)

    metric_keys = [k for k in rows[0] if k.startswith(("sdr_", "sisdr_"))]
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "songs": len(rows),
        "device": str(device),
        "amp": args.amp if amp_dtype is not None else "none",
        "num_overlap": num_overlap,
        "mean": {k: round(float(np.mean([r[k] for r in rows])), 3) for k in metric_keys},
        "median": {k: round(float(np.median([r[k] for r in rows])), 3) for k in metric_keys},
    }
    sdr_keys = [k for k in metric_keys if k.startswith("sdr_")]
    summary["mean"]["sdr_avg_stems"] = round(float(np.mean([summary["mean"][k] for k in sdr_keys])), 3)
    with open(out_dir / "per_song.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
