#!/usr/bin/env bash
# Downloads the pretrained BS-RoFormer vocals checkpoint (viperx, trained with MSST) and verifies sha256.
# Usage: BSRO_WEIGHTS_DIR=/path/to/weights bash scripts/download_weights.sh
set -euo pipefail

WEIGHTS_DIR="${BSRO_WEIGHTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/weights}"
mkdir -p "$WEIGHTS_DIR"

fetch() {
    local name="$1" url="$2" sha="$3"
    local dest="$WEIGHTS_DIR/$name"
    if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --quiet; then
        echo "ok (cached): $dest"
        return
    fi
    echo "downloading $name"
    curl -L --fail --retry 3 --retry-delay 5 -o "$dest.part" "$url"
    echo "$sha  $dest.part" | sha256sum -c --quiet
    mv "$dest.part" "$dest"
    echo "ok: $dest"
}

# vocals/other, viperx, SDR vocals 12.98. Config: configs/model_bs_roformer_ep_317_sdr_12.9755.yaml
fetch model_bs_roformer_ep_317_sdr_12.9755.ckpt \
    https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
    5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa
