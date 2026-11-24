# orfeo

Music source separation experiments with [BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) (lucidrains implementation, pinned to 0.4.1, the last release whose architecture matches the public pretrained checkpoints).

## Setup (H200 server)

```bash
bash scripts/setup_env.sh bsroformer            # fresh conda env: python 3.11, torch 2.6.0 cu124
BSRO_WEIGHTS_DIR=/path/to/weights bash scripts/download_weights.sh all
```

Checkpoints (trained with ZFTurbo's MSST code, loaded here into lucidrains' `BSRoformer`):

| model | stems | config | checkpoint |
|---|---|---|---|
| 4-stem (MUSDB18 SDR 9.66) | drums, bass, other, vocals | `configs/config_bs_roformer_384_8_2_485100.yaml` | `model_bs_roformer_ep_17_sdr_9.6568.ckpt` |
| vocals, viperx (SDR 12.98) | vocals + other (residual) | `configs/model_bs_roformer_ep_317_sdr_12.9755.yaml` | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` |

## Separate

```bash
conda activate bsroformer
python -m bsro.separate \
    --config configs/config_bs_roformer_384_8_2_485100.yaml \
    --checkpoint "$BSRO_WEIGHTS_DIR/model_bs_roformer_ep_17_sdr_9.6568.ckpt" \
    --input /path/to/song.wav_or_dir --output outputs/run1
```

Stems are written to `outputs/run1/<track>/<stem>.wav` (44.1 kHz float wav) plus `manifest.json` with checkpoint, code revision, device and chunking settings.

SLURM (one H200 on `gpus`): `BSRO_MODEL=4stem sbatch slurm/separate.sbatch` separates everything in `data/`; outputs go to `$HOME/work/orfeo/outputs/<jobid>/`.

## Tests

```bash
python -m pytest -q tests
```

## Evaluate separation quality

**Reference-free screen** (the ORFEO quality screen: level, bleed, and agreement between two separators). Run both models on the same inputs, then:

```bash
python -m bsro.screen --run $HOME/work/orfeo/outputs/<vocals_job> --reference-run $HOME/work/orfeo/outputs/<4stem_job> --inputs data
```

Writes `screen.csv` (per-track metrics and accept/reject/instrumental) and `screen.json` (counts and thresholds) into the run directory.

**Ground truth** on MUSDB18-HQ test (SDR as reported by MSST, plus SI-SDR), to confirm the lucidrains loading and chunked inference match the published numbers (4-stem: avg 9.65; vocals 11.08):

```bash
BSRO_MUSDB=$HOME/work/orfeo/musdb18hq BSRO_MODEL=4stem sbatch slurm/eval_musdb.sbatch
```

## Listen and inspect

Gradio app with the original mix, the stems of both runs per track, the screen charts and the metrics table:

```bash
python -m bsro.review_app --inputs data --run vocals=$HOME/work/orfeo/outputs/1508415 --run 4stem=$HOME/work/orfeo/outputs/1508406
```

Open it from your laptop through an SSH tunnel (`ssh -N -L 7860:127.0.0.1:7860 <user>@vhpc01`, then http://localhost:7860), or add `--share` for a temporary public gradio.live link.

## Lyrics from the vocal stems

Whisper on the separated vocals (tracks the screen marked instrumental are skipped):

```bash
BSRO_RUN=$HOME/work/orfeo/outputs/1508415 sbatch slurm/transcribe.sbatch
```

Writes `<run>/lyrics_large-v3/<track>.json` (language, text, timed segments with confidence) and `lyrics.csv`. Set `BSRO_WHISPER=turbo` for a faster model or `BSRO_LANGUAGE=en` to force a language.
