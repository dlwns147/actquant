#!/usr/bin/env bash
# Run the ifeval_pp and APPS custom tasks with Llama 3.1 8B Instruct.
#
# Usage (from anywhere):
#   bash run_llama31_8b.sh                # full run
#   LIMIT=20 bash run_llama31_8b.sh       # quick smoke run (20 docs/task)
#   TASKS=ifeval_pp bash run_llama31_8b.sh
#   CUDA_VISIBLE_DEVICES=2,3 bash run_llama31_8b.sh
set -euo pipefail

# This package lives at /NAS/SJ/actquant/search/tests/custom_tasks
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct}"
TASKS="${TASKS:-ifeval_pp,apps}"
LIMIT_ARG=""
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG="--limit ${LIMIT}"
fi
OUT="${OUT:-${SCRIPT_DIR}/results}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUT}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True" \
  --tasks "${TASKS}" \
  --include_path "${SCRIPT_DIR}" \
  --apply_chat_template \
  --batch_size auto \
  --output_path "${OUT}" \
  --log_samples \
  ${LIMIT_ARG}

echo "Done. Results written under ${OUT}/"
