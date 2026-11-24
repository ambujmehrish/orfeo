#!/bin/bash
# MUSDB18-HQ test SDR for a BS-RoFormer checkpoint on ONE H200.
#
#   BSRO_MUSDB=$HOME/work/orfeo/musdb18hq sbatch slurm/eval_musdb.sbatch
#   BSRO_MUSDB=... BSRO_LIMIT=3 sbatch slurm/eval_musdb.sbatch   # smoke test
#SBATCH --job-name=bsro-musdb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${BSRO_REPO_DIR:-$SLURM_SUBMIT_DIR}"
WEIGHTS_DIR="${BSRO_WEIGHTS_DIR:-$REPO_DIR/weights}"
MUSDB="${BSRO_MUSDB:?set BSRO_MUSDB to the MUSDB18-HQ root (contains test/)}"
OUTPUT="${BSRO_OUTPUT:-$HOME/work/orfeo/eval/musdb_$SLURM_JOB_ID}"

CONFIG="$REPO_DIR/configs/model_bs_roformer_ep_317_sdr_12.9755.yaml"
CKPT="$WEIGHTS_DIR/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT" >&2; exit 1; }
[ -d "$MUSDB/test" ] || { echo "missing $MUSDB/test" >&2; exit 1; }

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "${BSRO_ENV:-bsroformer}"
cd "$REPO_DIR"
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m bsro.eval_musdb \
    --config "$CONFIG" --checkpoint "$CKPT" --musdb "$MUSDB" --out "$OUTPUT" \
    --device cuda --batch-size "${BSRO_BATCH_SIZE:-8}" ${BSRO_LIMIT:+--limit "$BSRO_LIMIT"}
echo "### done -> $OUTPUT"
