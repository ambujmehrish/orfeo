"""Reference-free quality screen for separated vocals (ORFEO pipeline: agreement, level, bleed)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from bsro.metrics import (
    active_frames,
    db,
    energy,
    energy_in_frames,
    frame_rms,
    low_frequency_fraction,
    si_sdr,
)


def read(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T, sr


def load_mix(track_name: str, input_dir: Path, sample_rate: int) -> np.ndarray:
    matches = [p for p in input_dir.iterdir() if p.stem == track_name]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one input file named {track_name}.* in {input_dir}, found {len(matches)}")
    from bsro.separate import load_audio

    return load_audio(matches[0], sample_rate).numpy()


def track_metrics(
    mix: np.ndarray,
    vocals: np.ndarray,
    sample_rate: int,
    reference_vocals: np.ndarray | None,
    frame_s: float,
    active_db: float,
    low_cutoff_hz: float,
) -> dict:
    n = min(mix.shape[-1], vocals.shape[-1])
    mix, vocals = mix[..., :n], vocals[..., :n]
    acc = mix - vocals
    frame = int(frame_s * sample_rate)
    active = active_frames(vocals, mix, frame, active_db)

    out = {
        "duration_s": round(n / sample_rate, 2),
        # level
        "vocal_level_db": round(db(energy(vocals) / max(energy(mix), 1e-10)), 2),
        "acc_level_db": round(db(energy(acc) / max(energy(mix), 1e-10)), 2),
        "vocal_active_fraction": round(float(active.mean()) if active.size else 0.0, 3),
        # bleed (single-model proxies)
        "vocal_low_freq_fraction_db": round(db(low_frequency_fraction(vocals, sample_rate, low_cutoff_hz)), 2),
        "vocal_energy_in_inactive_db": round(
            db(energy_in_frames(vocals, ~active, frame) / max(energy_in_frames(mix, ~active, frame), 1e-10)), 2
        )
        if (~active).any()
        else None,
    }

    if reference_vocals is not None:
        ref = reference_vocals[..., :n]
        ref_active = active_frames(ref, mix, frame, active_db)
        union = (active | ref_active).sum()
        # agreement between the two separators
        out["agreement_sisdr_db"] = round(min(si_sdr(ref, vocals), si_sdr(vocals, ref)), 2)
        out["agreement_active_iou"] = round(float((active & ref_active).sum() / union) if union else 1.0, 3)
        # bleed: energy this model leaves in the vocal stem where the other model hears no vocals
        if (~ref_active).any():
            out["bleed_cross_db"] = round(
                db(energy_in_frames(vocals, ~ref_active, frame) / max(energy_in_frames(mix, ~ref_active, frame), 1e-10)), 2
            )
        else:
            out["bleed_cross_db"] = None
    return out


def decide(m: dict, args: argparse.Namespace) -> tuple[str, list[str]]:
    reasons = []
    if m["vocal_active_fraction"] < args.min_active_fraction:
        return "instrumental", ["vocal_active_fraction below threshold"]
    if m["vocal_level_db"] > args.max_vocal_level_db:
        reasons.append("vocal stem nearly equals the mix (separator passed the mix through)")
    if m["vocal_low_freq_fraction_db"] > args.max_low_freq_db:
        reasons.append("low-frequency bleed in vocals")
    if m.get("agreement_sisdr_db") is not None and m["agreement_sisdr_db"] < args.min_agreement_db:
        reasons.append("separators disagree")
    if m.get("bleed_cross_db") is not None and m["bleed_cross_db"] > args.max_bleed_db:
        reasons.append("accompaniment bleed where reference hears no vocals")
    return ("reject" if reasons else "accept"), reasons


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Screen separated vocals without ground truth.")
    p.add_argument("--run", required=True, help="output dir of bsro.separate for the model under test")
    p.add_argument("--inputs", required=True, help="directory holding the original mixes")
    p.add_argument("--reference-run", default=None, help="output dir of a second separator, for agreement/bleed")
    p.add_argument("--stem", default="vocals")
    p.add_argument("--out", default=None, help="CSV path (default: <run>/screen.csv)")
    p.add_argument("--frame-s", type=float, default=0.1)
    p.add_argument("--active-db", type=float, default=-25.0, help="vocal/mix frame ratio above which vocals count as present")
    p.add_argument("--low-cutoff-hz", type=float, default=100.0)
    p.add_argument("--min-active-fraction", type=float, default=0.05)
    p.add_argument("--max-vocal-level-db", type=float, default=-0.5)
    p.add_argument("--max-low-freq-db", type=float, default=-20.0)
    p.add_argument("--min-agreement-db", type=float, default=8.0)
    p.add_argument("--max-bleed-db", type=float, default=-20.0)
    args = p.parse_args(argv)

    run, inputs = Path(args.run), Path(args.inputs)
    ref_run = Path(args.reference_run) if args.reference_run else None
    tracks = sorted(d for d in run.iterdir() if d.is_dir() and (d / f"{args.stem}.wav").exists())
    if not tracks:
        raise FileNotFoundError(f"no <track>/{args.stem}.wav under {run}")

    rows = []
    for track_dir in tracks:
        vocals, sr = read(track_dir / f"{args.stem}.wav")
        mix = load_mix(track_dir.name, inputs, sr)
        ref = None
        if ref_run is not None:
            ref_path = ref_run / track_dir.name / f"{args.stem}.wav"
            if not ref_path.exists():
                raise FileNotFoundError(f"reference run lacks {ref_path}")
            ref, ref_sr = read(ref_path)
            if ref_sr != sr:
                raise ValueError(f"sample rate mismatch for {track_dir.name}: {sr} vs {ref_sr}")
        m = track_metrics(mix, vocals, sr, ref, args.frame_s, args.active_db, args.low_cutoff_hz)
        decision, reasons = decide(m, args)
        rows.append({"track": track_dir.name, **m, "decision": decision, "reasons": "; ".join(reasons)})
        print(f"{track_dir.name}: {decision} " + " ".join(f"{k}={v}" for k, v in m.items()), flush=True)

    out = Path(args.out) if args.out else run / "screen.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "tracks": len(rows),
        "accept": sum(r["decision"] == "accept" for r in rows),
        "reject": sum(r["decision"] == "reject" for r in rows),
        "instrumental": sum(r["decision"] == "instrumental" for r in rows),
        "thresholds": {k: v for k, v in vars(args).items() if k.startswith(("min_", "max_", "active_", "low_", "frame_"))},
    }
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
