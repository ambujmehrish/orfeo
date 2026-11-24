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

# key, title, plain-language description, threshold key in screen.json (None = informational), good direction
CHARTS = [
    ("vocal_active_fraction", "How much of the song has vocals",
     "Share of 100 ms frames where the vocal stem is within 25 dB of the mix. Near 0 means no singing was found and the track is tagged instrumental.",
     "min_active_fraction", "higher"),
    ("vocal_level_db", "How much of the mix was removed",
     "Loudness of the vocal stem relative to the full mix. A value near 0 dB would mean the separator returned the whole mix as vocals. Lower is better.",
     "max_vocal_level_db", "lower"),
    ("vocal_low_freq_fraction_db", "Bass leaking into the vocals",
     "Share of vocal-stem energy below 100 Hz, where a voice has almost none. Bass and kick drum leaking in push this up. More negative is cleaner.",
     "max_low_freq_db", "lower"),
    ("vocal_energy_in_inactive_db", "Noise between vocal phrases",
     "Energy left in the vocal stem during its own quiet parts, relative to the mix there. More negative means cleaner silence when nobody sings.",
     None, "lower"),
    ("agreement_sisdr_db", "Do two separators agree on the vocal?",
     "Similarity between this run's vocal stem and the reference run's, as SI-SDR. Higher means both models extracted the same signal, so the result is more trustworthy.",
     "min_agreement_db", "higher"),
    ("bleed_cross_db", "Accompaniment left where the reference hears no vocals",
     "Energy in this run's vocal stem during frames where the reference separator finds no singing. More negative is cleaner.",
     "max_bleed_db", "lower"),
    ("agreement_active_iou", "Do two separators agree on when singing happens?",
     "Overlap (IoU) between the frames each separator marks as vocal. 1 means identical timing.",
     None, "higher"),
]

CSS = """
.bsro-charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 14px; }
.bsro-rules { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 14px; margin: 0 0 14px; font-size: 13px; background: var(--background-fill-primary); }
.bsro-rules ul { margin: 4px 0 0 18px; padding: 0; } .bsro-rules li { margin: 2px 0; }
.bsro-chart { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 12px 4px; background: var(--background-fill-primary); }
.bsro-chart h4 { margin: 0 0 2px; font-size: 13px; font-weight: 600; color: var(--body-text-color); }
.bsro-chart .legend { margin: 0 0 4px; font-size: 11px; color: var(--body-text-color-subdued); }
.bsro-chart svg { width: 100%; height: auto; display: block; }
.bsro-chart text { font-family: inherit; fill: var(--body-text-color-subdued); }
.bsro-chart text.value { fill: var(--body-text-color); }
.bsro-chart .bar { fill: #2a78d6; }
.bsro-chart .bar.fail { fill: #d03b3b; }
.bsro-chart .bar.na { fill: #898781; opacity: 0.55; }
.bsro-chart .pass-zone { fill: #2a78d6; opacity: 0.07; }
.bsro-chart text.note { font-size: 10px; fill: var(--body-text-color-subdued); }
.bsro-chart text.thr-label { font-size: 10px; font-weight: 600; fill: var(--body-text-color); }
.bsro-chart .desc { margin: 0 0 8px; font-size: 12px; color: var(--body-text-color-subdued); line-height: 1.4; }
.bsro-chart .key { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11px; color: var(--body-text-color-subdued); margin: 4px 0 2px; }
.bsro-chart .key span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.bsro-chart .key .k-ok::before { background: #2a78d6; } .bsro-chart .key .k-fail::before { background: #d03b3b; } .bsro-chart .key .k-na::before { background: #898781; opacity: 0.55; }
.bsro-chart text { font-size: 12px; }
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


def find_lyrics_dir(run_dir: Path) -> Path | None:
    dirs = sorted(d for d in run_dir.glob("lyrics_*") if (d / "lyrics.csv").exists())
    if not dirs:
        return None
    # prefer the largest whisper model if several transcriptions exist
    order = ["large-v3", "large", "large-v2", "turbo", "large-v3-turbo", "medium", "small", "base", "tiny"]
    return min(dirs, key=lambda d: next((i for i, m in enumerate(order) if d.name == f"lyrics_{m}"), len(order)))


def load_lyrics(lyrics_dir: Path | None, track: str) -> tuple[str, str, str]:
    """Returns (header, text, timed segments) for a track, or a note when there is no transcript."""
    if lyrics_dir is None:
        return "no transcription found", "", ""
    path = lyrics_dir / f"{track}.json"
    if not path.exists():
        return f"no transcript for this track in {lyrics_dir.name} (skipped as instrumental, or not run)", "", ""
    rec = json.load(open(path))
    header = f"Whisper {rec.get('model', '?')} · language: {rec.get('language', '?')} · {len(rec.get('segments', []))} segments"
    lines = [f"[{seg['start']:6.1f} – {seg['end']:6.1f}]  {seg['text']}" + (f"   (no-speech p={seg['no_speech_prob']:.2f})" if seg.get("no_speech_prob", 0) > 0.5 else "") for seg in rec.get("segments", [])]
    return header, rec.get("text", ""), "\n".join(lines)


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


def bar_chart_svg(rows: list[dict], threshold: float | None, good: str) -> str:
    """rows: {name, value|None, instrumental: bool}. Bars failing the threshold are red; instrumental tracks are gray."""
    vals = [r["value"] for r in rows if r["value"] is not None] + ([threshold] if threshold is not None else [])
    lo, hi = min(0.0, *vals), max(0.0, *vals)
    pad = (hi - lo) * 0.14 or 1.0
    lo -= pad if lo < 0 else 0
    hi += pad if hi > 0 else 0
    width, left, right, row_h, top = 620, 200, 70, 26, 26
    height = top + len(rows) * row_h + 30

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (width - left - right)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    if threshold is not None:
        x0, x1 = (x(threshold), x(hi)) if good == "higher" else (x(lo), x(threshold))
        parts.append(f'<rect class="pass-zone" x="{x0:.1f}" y="{top}" width="{max(0.0, x1 - x0):.1f}" height="{len(rows) * row_h}"/>')
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        parts.append(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{top + len(rows) * row_h}"/>')
        parts.append(f'<text x="{x(v):.1f}" y="{height - 14}" text-anchor="middle">{fmt(v)}</text>')
    parts.append(f'<line class="grid" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top}" y2="{top + len(rows) * row_h}"/>')
    arrow = "better →" if good == "higher" else "← better"
    parts.append(f'<text class="note" x="{width - right}" y="{height - 2}" text-anchor="end">{arrow}</text>')
    for i, r in enumerate(rows):
        y, h = top + i * row_h + 5, row_h - 10
        parts.append(f'<text x="{left - 8}" y="{y + h - 3}" text-anchor="end">{html.escape(r["name"])}</text>')
        v = r["value"]
        if r["instrumental"] or v is None:
            note = r.get("note") or "instrumental, not scored"
            parts.append(f'<rect class="bar na" x="{x(0):.1f}" y="{y}" width="6" height="{h}" rx="2"/>')
            if x(0) > (left + width - right) / 2:
                parts.append(f'<text class="note" x="{x(0) - 6:.1f}" y="{y + h - 3}" text-anchor="end">{html.escape(note)}</text>')
            else:
                parts.append(f'<text class="note" x="{x(0) + 10:.1f}" y="{y + h - 3}">{html.escape(note)}</text>')
            continue
        fails = threshold is not None and ((v < threshold) if good == "higher" else (v > threshold))
        x0, w = min(x(0), x(v)), max(1.0, abs(x(v) - x(0)))
        parts.append(f'<rect class="bar{" fail" if fails else ""}" x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="3"><title>{html.escape(r["name"])}: {fmt(v)}{" (fails threshold)" if fails else ""}</title></rect>')
        tx, anchor = (x(v) + 5, "start") if v >= 0 else (x(v) - 5, "end")
        label = fmt(v) + ("  ✕" if fails else "")
        parts.append(f'<text class="value" x="{tx:.1f}" y="{y + h - 3}" text-anchor="{anchor}">{label}</text>')
    if threshold is not None:
        parts.append(f'<line class="thr" x1="{x(threshold):.1f}" x2="{x(threshold):.1f}" y1="{top - 4}" y2="{top + len(rows) * row_h}"/>')
        word = "at least" if good == "higher" else "at most"
        tx = x(threshold)
        anchor = "end" if tx > width - 110 else ("start" if tx < left + 110 else "middle")
        parts.append(f'<text class="thr-label" x="{tx:.1f}" y="{top - 8}" text-anchor="{anchor}">accept if {word} {fmt(threshold)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def charts_html(rows: list[dict], thresholds: dict) -> str:
    blocks = []
    for key, title, desc, thr_key, good in CHARTS:
        data = [{"name": r["track"], "value": r.get(key), "instrumental": r["decision"] == "instrumental"} for r in rows]
        if not any(d["value"] is not None and not d["instrumental"] for d in data):
            continue
        if key == "vocal_active_fraction":
            for d in data:
                if d["instrumental"]:
                    d["note"] = f"{fmt(d['value'])}: below threshold, tagged instrumental"
        thr = thresholds.get(thr_key) if thr_key else None
        key_html = '<div class="key"><span class="k-ok">within threshold</span><span class="k-fail">fails threshold</span><span class="k-na">instrumental, not scored</span></div>' if thr is not None else '<div class="key"><span class="k-ok">informational, no threshold</span><span class="k-na">instrumental, not scored</span></div>'
        blocks.append(f'<div class="bsro-chart"><h4>{html.escape(title)}</h4><p class="desc">{html.escape(desc)}</p>{key_html}{bar_chart_svg(data, thr, good)}</div>')
    return '<div class="bsro-charts">' + "".join(blocks) + "</div>"


def rules_html(thresholds: dict, has_reference: bool) -> str:
    t = thresholds
    items = [
        f"<b>instrumental</b> if less than {t.get('min_active_fraction', 0.05):.0%} of the song has vocals (nothing to separate, not a failure)",
        f"<b>reject</b> if the vocal stem is louder than {fmt(t.get('max_vocal_level_db', -0.5))} dB relative to the mix (separator passed the mix through)",
        f"<b>reject</b> if more than {fmt(t.get('max_low_freq_db', -20))} dB of the vocal energy sits below 100 Hz (bass leaking in)",
    ]
    if has_reference:
        items += [
            f"<b>reject</b> if the two separators agree below {fmt(t.get('min_agreement_db', 8))} dB SI-SDR",
            f"<b>reject</b> if more than {fmt(t.get('max_bleed_db', -20))} dB of accompaniment remains where the reference hears no vocals",
        ]
    items.append("<b>accept</b> otherwise")
    return '<div class="bsro-rules">How a track is scored (thresholds are initial settings, recorded in screen.json):<ul>' + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>"


def tiles_html(rows: list[dict]) -> str:
    count = lambda d: sum(r["decision"] == d for r in rows)
    minutes = sum(r.get("duration_s") or 0 for r in rows) / 60
    items = [("Tracks", len(rows)), ("Accepted", count("accept")), ("Rejected", count("reject")), ("Instrumental", count("instrumental")), ("Minutes of audio", f"{minutes:.1f}")]
    return '<div class="bsro-tiles">' + "".join(f'<div class="bsro-tile"><div class="n">{n}</div><div class="l">{l}</div></div>' for l, n in items) + "</div>"


def build_app(inputs: Path, runs: dict[str, Path], rows: list[dict], thresholds: dict, title: str, lyrics_dir: Path | None) -> gr.Blocks:
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
        lyr_header, lyr_text, lyr_segments = load_lyrics(lyrics_dir, t["name"])
        updates += [gr.update(value=lyr_text, label=f"Lyrics · {lyr_header}"), gr.update(value=lyr_segments)]
        return updates

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"## {title}")
        if len(labels) > 1:
            gr.Markdown(f'Separators: **{" vs ".join(labels)}**. Model under test is "{labels[0]}"; agreement and cross-bleed are measured against "{labels[1]}".')
        else:
            gr.Markdown(f'Separator: **{labels[0]}**. Agreement charts appear only if screen.csv was computed with a reference run.')
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
            lyrics_text = gr.Textbox(label="Lyrics", lines=6, max_lines=20, interactive=False)
            with gr.Accordion("Timed segments", open=False):
                lyrics_segments = gr.Textbox(label="start – end  text", lines=10, max_lines=60, interactive=False)
            outputs = [header, metrics, mix_audio, *stem_audios, lyrics_text, lyrics_segments]
            selector.change(select, selector, outputs)
            app.load(select, selector, outputs)
        with gr.Tab("Charts"):
            gr.Markdown("One chart per screen metric, one bar per track, same track order in every chart. Blue bars pass, red bars fail that chart's threshold, the tinted area is the passing zone. Hover a bar for the exact value.")
            has_reference = any(r.get("agreement_sisdr_db") is not None for r in rows)
            gr.HTML((rules_html(thresholds, has_reference) + charts_html(rows, thresholds)) if rows else "no screen.csv found")
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
    p.add_argument("--lyrics", default=None, help="transcription dir from bsro.transcribe (default: newest <first run>/lyrics_* if present)")
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

    lyrics_dir = Path(args.lyrics).resolve() if args.lyrics else find_lyrics_dir(next(iter(runs.values())))
    if args.lyrics and not lyrics_dir.is_dir():
        raise FileNotFoundError(f"lyrics directory not found: {lyrics_dir}")
    print(f"lyrics: {lyrics_dir or 'none found'}")

    app = build_app(inputs, runs, rows, thresholds, args.title, lyrics_dir)
    app.launch(server_name=args.host, server_port=args.port, share=args.share, auth=auth, css=CSS, allowed_paths=[str(inputs), *map(str, runs.values())])
    return 0


if __name__ == "__main__":
    sys.exit(main())
