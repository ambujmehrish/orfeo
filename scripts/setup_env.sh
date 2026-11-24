#!/usr/bin/env bash
# Creates a fresh conda env for BS-RoFormer inference on the H200 node.
# Usage: bash scripts/setup_env.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-bsroformer}"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.6.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

pip install --upgrade pip
pip install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" --index-url "$TORCH_INDEX"
pip install -r "$REPO_DIR/requirements.txt"
pip install pytest

python - <<'PY'
import torch, bs_roformer
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
echo "env '$ENV_NAME' ready"
