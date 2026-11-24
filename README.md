# orfeo

Vocal separation for the ORFEO pipeline with [BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) (lucidrains implementation, pinned to 0.4.1, the last release whose architecture matches the public pretrained checkpoint). Accompaniment is mix − vocals.

Checkpoint: viperx `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (trained with ZFTurbo's MSST code; vocals SDR 12.98), config `configs/model_bs_roformer_ep_317_sdr_12.9755.yaml`.

## Setup (vhpc)

```bash
bash scripts/setup_env.sh bsroformer            # fresh conda env: python 3.11, torch 2.6.0 cu124
BSRO_WEIGHTS_DIR=$PWD/weights bash scripts/download_weights.sh
python -m pytest -q tests
```

## Separate

Everything in `data/` on one GPU (defaults to an H200 on `gpus`; any 24 GB+ GPU works):

```bash
sbatch slurm/separate.sbatch
sbatch --partition=gpulongjobs --qos=gpulongjobs --gres=gpu:a40:1 slurm/separate.sbatch
```

Stems land in `$HOME/work/orfeo/outputs/<jobid>/<track>/{vocals,other}.wav` (44.1 kHz float wav) with `manifest.json` (checkpoint, code revision, device, chunking). Direct use:

```bash
python -m bsro.separate --config configs/model_bs_roformer_ep_317_sdr_12.9755.yaml --checkpoint weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt --input data --output outputs/run1
```

## Quality screen (no ground truth)

Level, activity and bleed checks on the vocal stem; agreement metrics appear only if a second run is given with `--reference-run`:

```bash
python -m bsro.screen --run $HOME/work/orfeo/outputs/<jobid> --inputs data
```

Writes `screen.csv` (per-track metrics, accept/reject/instrumental) and `screen.json` (counts, thresholds) into the run directory.

## Ground truth on MUSDB18-HQ

SDR as reported by MSST plus SI-SDR on the 50 test songs, to confirm the lucidrains loading and chunked inference reproduce the published numbers:

```bash
BSRO_MUSDB=$HOME/work/orfeo/musdb18hq sbatch slurm/eval_musdb.sbatch
```

## Listen and inspect

```bash
python -m bsro.review_app --inputs data --run vocals=$HOME/work/orfeo/outputs/<jobid>
```

Then tunnel: `ssh -R 80:127.0.0.1:7860 serveo.net` (public link) or `ssh -N -L 7860:127.0.0.1:7860 <user>@vhpc01` (http://localhost:7860).

## Lyrics from the vocal stems

```bash
BSRO_RUN=$HOME/work/orfeo/outputs/<jobid> sbatch slurm/transcribe.sbatch
```

Whisper large-v3 on each vocal stem (instrumental tracks skipped). Writes `<run>/lyrics_large-v3/<track>.json` and `lyrics.csv`. `BSRO_WHISPER=turbo` for a faster model, `BSRO_LANGUAGE=en` to force a language.
