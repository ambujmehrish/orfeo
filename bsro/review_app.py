"""Gradio app to listen to original mixes and separated stems from two runs, with screen metrics charts."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import gradio as gr

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}
MAX_STEMS = 4

# metric key, label, threshold key in screen.json (None = informational)
CHARTS = [
    ("agreement_sisdr_db", "Agreement between separators, SI-SDR (dB)", "min_agreement_db"),
    ("vocal_active_fraction", "Fraction of frames with vocals", "min_active_fraction"),
    ("vocal_level_db", "Vocal level relative to mix (dB)", "max_vocal_level_db"),
    ("bleed_cross_db", "Accompaniment left in vocals where the reference hears none (dB)", "max_bleed_db"),
    ("vocal_low_freq_fraction_db", "Vocal energy below 100 Hz (dB)", "max_low_freq_db"),
    ("agreement_active_iou", "Agreement on where vocals are (IoU)", None),
]

CSS = """
.bsro-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 14px; }
.bsro-chart { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 12px 4px; background: var(--background-fill-primary); }
.bsro-chart h4 { margin: 0 0 2px; font-size: 13px; font-weight: 600; color: var(--body-text-color); }
.bsro-chart .legend { margin: 0 0 4px; font-size: 11px; color: var(--body-text-color-subdued); }
.bsro-chart svg { width: 100%; height: auto; display: block; }
.bsro-chart text { font-family: inherit; font-size: 11px; fill: var(--body-text-color-subdued); }
.bsro-chart text.value { fill: var(--body-text-color); }
.bsro-chart .bar { fill: #2a78d6; }
.bsro-chart .bar:hover { opacity: 0.8; }
.bsro-chart .grid { stroke: var(--border-color-primary); stroke-width: 1; }
.bsro-chart .thr { stroke: var(--body-text-color); stroke-width: 1.5; stroke-dasharray: 4 3; }
.bsro-tiles { display: flex; flex-wrap: wrap; gap: 10px; }
.bsro-tile { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 8px 14px; min-width: 120px; background: var(--background-fill-primary); }
.bsro-tile .n { font-size: 24px; font-weight: 600; color: var(--body-text-color); }
.bsro-tile .l { font-size: 12px; color: var(--body-text-color-subdued); }
.bsro-badge { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px; border: 1px solid var(--border-color-primary); font-weight: 600; font-size: 13px; }
.bsro-badge::before { content: ""; width: 9px; height: 9px; border-radius: 50%; background: #898781; }
.bsro-badge.accept::before { background: #0ca30c; } .bsro-badge.reject::before { background: #d03b3b; } .bsro-badge.instrumental::before { background: #fab219; }
"""


def fmt(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def read_screen(path: Path) -> tuple[list[dict], dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("track", "decision", "reasons"):
                r[k] = float(v) if v not in ("", "None") else None
    summary = path.with_suffix(".json")
    thresholds = json.load(open(summary))["thresholds"] if summary.exists() else {}
    return rows, thresholds


def collect_tracks(inputs: Path, runs: dict[str, Path]) -> list[dict]:
    tracks = []
    for mix in sorted(p for p in inputs.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS):
        stems = {}
        for label, run_dir in runs.items():
            stem_dir = run_dir / mix.stem
            found = sorted(stem_dir.glob("*.wav")) if stem_dir.is_dir() else []
            stems[label] = sorted(found, key=lambda p: (p.stem != "vocals", p.stem))
        tracks.append({"name": mix.stem, "mix": mix, "stems": stems})
    return tracks


def bar_chart_svg(rows: list[tuple[str, float]], threshold: float | None) -> str:
    vals = [v for _, v in rows] + ([threshold] if threshold is not None else [])
    lo, hi = min(0.0, *vals), max(0.0, *vals)
    pad = (hi - lo) * 0.12 or 1.0
    lo -= pad if lo < 0 else 0
    hi += pad if hi > 0 else 0
    width, left, right, row_h, top = 560, 190, 16, 22, 8
    height = top + len(rows) * row_h + 24

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (width - left - right)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        parts.append(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{height - 22}"/>')
        parts.append(f'<text x="{x(v):.1f}" y="{height - 6}" text-anchor="middle">{fmt(v)}</text>')
    parts.append(f'<line class="grid" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top}" y2="{height - 22}"/>')
    for i, (name, v) in enumerate(rows):
        y, h = top + i * row_h + 4, row_h - 8
        x0, w = min(x(0), x(v)), max(1.0, abs(x(v) - x(0)))
        parts.append(f'<text x="{left - 8}" y="{y + h - 3}" text-anchor="end">{html.escape(name)}</text>')
        parts.append(f'<rect class="bar" x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="3"><title>{html.escape(name)}: {fmt(v)}</title></rect>')
        tx, anchor = (x(v) + 4, "start") if v >= 0 else (x(v) - 4, "end")
        parts.append(f'<text class="value" x="{tx:.1f}" y="{y + h - 3}" text-anchor="{anchor}">{fmt(v)}</text>')
    if threshold is not None:
        parts.append(f'<line class="thr" x1="{x(threshold):.1f}" x2="{x(threshold):.1f}" y1="{top}" y2="{height - 22}"><title>threshold {fmt(threshold)}</title></line>')
    parts.append("</svg>")
    return "".join(parts)


def charts_html(rows: list[dict], thresholds: dict) -> str:
    blocks = []
    for key, label, thr_key in CHARTS:
        data = [(r["track"], r[key]) for r in rows if r.get(key) is not None]
        if not data:
            continue
        thr = thresholds.get(thr_key) if thr_key else None
        legend = f"dashed line: threshold {thr_key.replace('_', ' ')} = {fmt(thr)}" if thr is not None else "informational, no threshold"
        blocks.append(f'<div class="bsro-chart"><h4>{html.escape(label)}</h4><p class="legend">{html.escape(legend)}</p>{bar_chart_svg(data, thr)}</div>')
    return '<div class="bsro-charts">' + "".join(blocks) + "</div>"


def tiles_html(rows: list[dict]) -> str:
    count = lambda d: sum(r["decision"] == d for r in rows)
    minutes = sum(r.get("duration_s") or 0 for r in rows) / 60
    items = [("Tracks", len(rows)), ("Accepted", count("accept")), ("Rejected", count("reject")), ("Instrumental", count("instrumental")), ("Minutes of audio", f"{minutes:.1f}")]
    return '<div class="bsro-tiles">' + "".join(f'<div class="bsro-tile"><div class="n">{n}</div><div class="l">{l}</div></div>' for l, n in items) + "</div>"


def build_app(inputs: Path, runs: dict[str, Path], rows: list[dict], thresholds: dict, title: str) -> gr.Blocks:
    tracks = collect_tracks(inputs, runs)
    by_name = {r["track"]: r for r in rows}
    labels = list(runs)

    def choice_label(t: dict) -> str:
        sc = by_name.get(t["name"])
        return f"{t['name']}  [{sc['decision']}]" if sc else t["name"]

    choices = [choice_label(t) for t in tracks]
    by_choice = {choice_label(t): t for t in tracks}

    def select(choice: str):
        t = by_choice[choice]
        sc = by_name.get(t["name"])
        if sc:
            reasons = html.escape(sc.get("reasons") or "")
            metrics = " · ".join(f"{k.replace('_', ' ')} **{fmt(v)}**" for k, v in sc.items() if k not in ("track", "decision", "reasons"))
            header = f'<span class="bsro-badge {sc["decision"]}">{sc["decision"]}</span> <span style="color:var(--body-text-color-subdued)">{reasons}</span>'
        else:
            header, metrics = "not screened", ""
        updates = [gr.update(value=header), gr.update(value=metrics), gr.update(value=str(t["mix"]), label=f"mix · {t['name']}")]
        for label in labels:
            stems = t["stems"][label]
            for i in range(MAX_STEMS):
                if i < len(stems):
                    updates.append(gr.update(value=str(stems[i]), label=f"{label} · {stems[i].stem}", visible=True))
                else:
                    updates.append(gr.update(value=None, visible=False))
        return updates

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"## {title}")
        gr.Markdown(f'Separators: **{" vs ".join(labels)}**. Model under test is "{labels[0]}"; agreement and cross-bleed are measured against "{labels[1] if len(labels) > 1 else "none"}".')
        if rows:
            gr.HTML(tiles_html(rows))
        with gr.Tab("Listen"):
            selector = gr.Dropdown(choices=choices, value=choices[0] if choices else None, label="Track")
            header = gr.HTML()
            metrics = gr.Markdown()
            mix_audio = gr.Audio(label="mix", type="filepath", interactive=False)
            stem_audios = []
            with gr.Row():
                for label in labels:
                    with gr.Column():
                        gr.Markdown(f"**{label}**")
                        for _ in range(MAX_STEMS):
                            stem_audios.append(gr.Audio(type="filepath", interactive=False, visible=False))
            outputs = [header, metrics, mix_audio, *stem_audios]
            selector.change(select, selector, outputs)
            app.load(select, selector, outputs)
        with gr.Tab("Charts"):
            gr.Markdown("Dashed line = accept/reject threshold. Hover a bar for the exact value.")
            gr.HTML(charts_html(rows, thresholds) if rows else "no screen.csv found")
        with gr.Tab("Table"):
            if rows:
                keys = list(rows[0].keys())
                gr.Dataframe(value=[[r[k] for k in keys] for r in rows], headers=keys, interactive=False, wrap=True)
            gr.Markdown("thresholds: " + ", ".join(f"{k} = {v}" for k, v in thresholds.items()))
    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Listen to separations and view screen metrics.")
    p.add_argument("--inputs", required=True, help="directory with the original mixes")
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR", help="separation run; the first is the model under test")
    p.add_argument("--screen", default=None, help="screen.csv from bsro.screen (default: <first run>/screen.csv)")
    p.add_argument("--title", default="BS-RoFormer separation review")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--auth", default=None, metavar="USER:PASSWORD", help="require a login (use when exposing the app publicly)")
    args = p.parse_args(argv)

    runs: dict[str, Path] = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run expects LABEL=DIR, got {spec}")
        label, path = spec.split("=", 1)
        runs[label] = Path(path).resolve()
        if not runs[label].is_dir():
            raise FileNotFoundError(f"run directory not found: {path}")
    inputs = Path(args.inputs).resolve()
    if not inputs.is_dir():
        raise FileNotFoundError(f"inputs directory not found: {inputs}")
    screen = Path(args.screen) if args.screen else next(iter(runs.values())) / "screen.csv"
    if screen.exists():
        rows, thresholds = read_screen(screen)
    else:
        print(f"warning: {screen} not found; metrics tabs will be empty", file=sys.stderr)
        rows, thresholds = [], {}

    auth = None
    if args.auth:
        if ":" not in args.auth:
            raise ValueError("--auth expects USER:PASSWORD")
        auth = tuple(args.auth.split(":", 1))

    app = build_app(inputs, runs, rows, thresholds, args.title)
    app.launch(server_name=args.host, server_port=args.port, share=args.share, auth=auth, css=CSS, allowed_paths=[str(inputs), *map(str, runs.values())])
    return 0


if __name__ == "__main__":
    sys.exit(main())
