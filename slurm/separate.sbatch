#!/usr/bin/env bash
# BS-RoFormer separation on one H200.
# Submit: sbatch slurm/separate.sbatch
# H200 nodes on vhpc: partition "gpus" (vhpc13-15) or "gpulongjobs" (vhpc11-12,25-28).
#SBATCH --job-name=bsro-separate
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${BSRO_REPO_DIR:-$SLURM_SUBMIT_DIR}"
WEIGHTS_DIR="${BSRO_WEIGHTS_DIR:-$REPO_DIR/weights}"
INPUT="${BSRO_INPUT:?set BSRO_INPUT to an audio file or directory}"
OUTPUT="${BSRO_OUTPUT:-$REPO_DIR/outputs/$SLURM_JOB_ID}"
MODEL="${BSRO_MODEL:-4stem}"   # 4stem | vocals

case "$MODEL" in
    4stem)
        CONFIG="$REPO_DIR/configs/config_bs_roformer_384_8_2_485100.yaml"
        CKPT="$WEIGHTS_DIR/model_bs_roformer_ep_17_sdr_9.6568.ckpt" ;;
    vocals)
        CONFIG="$REPO_DIR/configs/model_bs_roformer_ep_317_sdr_12.9755.yaml"
        CKPT="$WEIGHTS_DIR/model_bs_roformer_ep_317_sdr_12.9755.ckpt" ;;
    *) echo "unknown BSRO_MODEL=$MODEL" >&2; exit 1 ;;
esac

eval "$(conda shell.bash hook)"
conda activate "${BSRO_ENV:-bsroformer}"
cd "$REPO_DIR"
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m bsro.separate \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --device cuda \
    --batch-size "${BSRO_BATCH_SIZE:-8}" \
    ${BSRO_EXTRA_ARGS:-}
