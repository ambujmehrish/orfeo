#!/bin/bash
# Whisper transcription of separated vocals on ONE GPU.
#
#   BSRO_RUN=$HOME/work/orfeo/outputs/1508415 sbatch slurm/transcribe.sbatch
#   BSRO_RUN=... BSRO_WHISPER=turbo sbatch --partition=gpulongjobs --qos=gpulongjobs --gres=gpu:a40:1 slurm/transcribe.sbatch
#SBATCH --job-name=bsro-lyrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${BSRO_REPO_DIR:-$SLURM_SUBMIT_DIR}"
RUN="${BSRO_RUN:?set BSRO_RUN to a bsro.separate output dir}"
MODEL="${BSRO_WHISPER:-large-v3}"
[ -d "$RUN" ] || { echo "missing run dir $RUN" >&2; exit 1; }

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "${BSRO_ENV:-bsroformer}"
cd "$REPO_DIR"
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m bsro.transcribe --run "$RUN" --stem "${BSRO_STEM:-vocals}" --model "$MODEL" --device cuda ${BSRO_LANGUAGE:+--language "$BSRO_LANGUAGE"} ${BSRO_EXTRA_ARGS:-}
echo "### done -> $RUN/lyrics_$MODEL"
