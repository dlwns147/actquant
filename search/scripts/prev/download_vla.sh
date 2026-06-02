#!/usr/bin/env bash
# Download base VLA models for the model-merging study into /SSD/huggingface/{org}/{name}.
# Reuses ../download_hf.py (snapshot_download). Run from /NAS/SJ/actquant/search:
#     bash scripts/download_vla.sh
#
# Research roles (see merging study):
#   - SmolVLA       : small (450M), cheap to iterate / scaling-law ablations
#   - GR00T N1.5    : VLM backbone frozen during FT -> isolates merge to the action head
#   - pi0 / pi0-FAST: action representation comparison (flow-matching vs FAST tokens)
#   - OpenVLA       : literature baseline (7B)
#
# NOTE: nvidia/GR00T-N1.5-3B (and possibly the pi0 repos) are gated -- accept the
# license on the HF model page first, then authenticate, e.g.:
#     export HF_TOKEN=hf_xxx        # or: huggingface-cli login
# pi0-FAST repo id uses a hyphen in the official from_pretrained example
# ("lerobot/pi0fast-base"); if download 404s, try the underscore form "lerobot/pi0fast_base".

set -euo pipefail

# org/name pairs to download
MODELS=(
  "openvla/openvla-7b"
  "lerobot/smolvla_base"
  "nvidia/GR00T-N1.5-3B"
  "lerobot/pi0_base"
  "lerobot/pi0fast-base"
)

HF_TOKEN=hf
huggingface-cli login --token ${HF_TOKEN}
for repo in "${MODELS[@]}"; do
  MODEL_PATH="${repo%%/*}"   # org
  MODEL_NAME="${repo#*/}"    # name
  echo "==> Downloading ${repo} -> /SSD/huggingface/${repo}"
  python download_hf.py --model_path "${MODEL_PATH}" --model_name "${MODEL_NAME}"
done

echo "All downloads complete."
