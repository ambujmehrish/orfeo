#!/bin/bash
# BS-RoFormer separation on ONE H200 (vhpc13/14/15, gpu:h200:1 each).
#
#   BSRO_INPUT=data sbatch slurm/separate.sbatch
#   sbatch --qos=gpulongjobs slurm/separate.sbatch        # if 'normal' is full
#   sbatch --partition=gpulongjobs --qos=gpulongjobs --gres=gpu:a40:1 slurm/separate.sbatch   # any free GPU works
#
# Env vars: BSRO_INPUT (file or dir, default: data),
# BSRO_WEIGHTS_DIR (default: <repo>/weights), BSRO_OUTPUT (default: $HOME/work/orfeo/outputs/<jobid>),
# BSRO_ENV (conda env, default: bsroformer), BSRO_BATCH_SIZE (chunks per forward, default: 8).
#SBATCH --job-name=bsro-separate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:h200:1
# one H200 only; plain gpu:1 could land on an RTX6000/A40 node
#SBATCH --cpus-per-task=4
# 4 CPUs fits both QoS caps (normal: 120 per group, gpulongjobs: 10 per user)
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${BSRO_REPO_DIR:-$SLURM_SUBMIT_DIR}"
WEIGHTS_DIR="${BSRO_WEIGHTS_DIR:-$REPO_DIR/weights}"
INPUT="${BSRO_INPUT:-$REPO_DIR/data}"
OUTPUT="${BSRO_OUTPUT:-$HOME/work/orfeo/outputs/$SLURM_JOB_ID}"

CONFIG="$REPO_DIR/configs/model_bs_roformer_ep_317_sdr_12.9755.yaml"
CKPT="$WEIGHTS_DIR/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT; run scripts/download_weights.sh" >&2; exit 1; }
[ -e "$INPUT" ] || { echo "missing input $INPUT" >&2; exit 1; }

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "${BSRO_ENV:-bsroformer}"
cd "$REPO_DIR"
mkdir -p logs "$OUTPUT"

echo "job $SLURM_JOB_ID on $(hostname) | input=$INPUT | output=$OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))"

python -m bsro.separate \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --device cuda \
    --batch-size "${BSRO_BATCH_SIZE:-8}" \
    ${BSRO_EXTRA_ARGS:-}

echo "### done -> $OUTPUT"
