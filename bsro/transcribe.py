"""Transcribe separated vocal stems with Whisper (lyrics with segment timestamps)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import whisper

WHISPER_SR = 16000


def load_mono_16k(path: Path) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
    if sr != WHISPER_SR:
        audio = torchaudio.functional.resample(audio, sr, WHISPER_SR)
    return audio[0]


def screened_out(run: Path, stem: str) -> set[str]:
    """Tracks the quality screen marked instrumental (nothing to transcribe)."""
    path = run / "screen.csv"
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {r["track"] for r in csv.DictReader(f) if r["decision"] == "instrumental"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Transcribe <run>/<track>/<stem>.wav with Whisper.")
    p.add_argument("--run", required=True, help="output dir of bsro.separate")
    p.add_argument("--stem", default="vocals")
    p.add_argument("--model", default="large-v3", help="whisper model name (tiny, base, small, medium, large-v3, turbo)")
    p.add_argument("--language", default=None, help="ISO code to force a language; default auto-detect per track")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="output dir (default: <run>/lyrics_<model>)")
    p.add_argument("--all", action="store_true", help="also transcribe tracks the screen marked instrumental")
    p.add_argument("--word-timestamps", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    run = Path(args.run)
    tracks = sorted(d for d in run.iterdir() if d.is_dir() and (d / f"{args.stem}.wav").exists())
    if not tracks:
        raise FileNotFoundError(f"no <track>/{args.stem}.wav under {run}")
    skip = set() if args.all else screened_out(run, args.stem)
    if args.limit:
        tracks = tracks[: args.limit]

    out = Path(args.out) if args.out else run / f"lyrics_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = whisper.load_model(args.model, device=device)
    fp16 = device.type == "cuda"

    rows = []
    for track in tracks:
        name = track.name
        if name in skip:
            rows.append({"track": name, "status": "skipped_instrumental", "language": "", "text": ""})
            print(f"{name}: skipped (screen says instrumental)", flush=True)
            continue
        t0 = time.time()
        audio = load_mono_16k(track / f"{args.stem}.wav")
        result = model.transcribe(audio.numpy(), language=args.language, fp16=fp16, word_timestamps=args.word_timestamps, verbose=None)
        segments = [
            {
                "start": round(s["start"], 2),
                "end": round(s["end"], 2),
                "text": s["text"].strip(),
                "avg_logprob": round(s["avg_logprob"], 3),
                "no_speech_prob": round(s["no_speech_prob"], 3),
                **({"words": [{"word": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2)} for w in s.get("words", [])]} if args.word_timestamps else {}),
            }
            for s in result["segments"]
        ]
        text = " ".join(s["text"] for s in segments)
        record = {
            "track": name,
            "stem": str(track / f"{args.stem}.wav"),
            "model": args.model,
            "language": result["language"],
            "duration_s": round(audio.shape[-1] / WHISPER_SR, 2),
            "seconds": round(time.time() - t0, 1),
            "text": text,
            "segments": segments,
        }
        with open(out / f"{name}.json", "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        rows.append({"track": name, "status": "ok", "language": result["language"], "text": text})
        print(f"{name} [{result['language']}] {record['seconds']}s: {text[:120]}", flush=True)

    with open(out / "lyrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track", "status", "language", "text"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "summary.json", "w") as f:
        json.dump(
            {"run": str(run.resolve()), "stem": args.stem, "model": args.model, "whisper_version": whisper.__version__ if hasattr(whisper, "__version__") else None,
             "device": str(device), "language": args.language or "auto", "tracks": len(rows), "transcribed": sum(r["status"] == "ok" for r in rows)},
            f, indent=2,
        )
    print(f"wrote {out / 'lyrics.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
