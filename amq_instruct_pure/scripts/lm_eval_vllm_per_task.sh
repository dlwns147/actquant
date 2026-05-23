#!/bin/bash

# Usage: bash scripts/lm_eval_vllm_per_task.sh <DEVICES> [TASK]
#   ex) bash scripts/lm_eval_vllm_per_task.sh 0 gsm8k
#   ex) bash scripts/lm_eval_vllm_per_task.sh 0,1 ifeval
#   ex) bash scripts/lm_eval_vllm_per_task.sh 0 mbpp
#   인자 없이 실행하면 gsm8k, ifeval, mbpp 를 순차적으로 모두 실행합니다.

DEVICES=${1}
SELECTED_TASK=${2}

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct

INCLUDE_PATH=/NAS/SJ/actquant/search/tests/custom_tasks
BATCH_SIZE=auto
OUT_ROOT=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/results/lm_eval

# LIMIT=10

run_task () {
    local TASK=$1
    local OUT="${OUT_ROOT}/${TASK}"

    echo "=========================================="
    echo "[lm_eval] TASK=${TASK}  DEVICES=${DEVICES}"
    echo "          OUT=${OUT}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${DEVICES}" python lm_eval_vllm.py \
      --model "${MODEL_PATH}/${MODEL_NAME}" \
      --task "${TASK}" \
      --include_path "${INCLUDE_PATH}" \
      --batch_size "${BATCH_SIZE}" \
      --output_path "${OUT}" \
      --log_samples
      # "${LIMIT_ARG[@]}"
}

if [ -z "${SELECTED_TASK}" ]; then
    for TASK in gsm8k ifeval mbpp; do
        run_task "${TASK}"
    done
else
    run_task "${SELECTED_TASK}"
fi
