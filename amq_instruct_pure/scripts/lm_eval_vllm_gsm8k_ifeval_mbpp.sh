#!/bin/bash

DEVICES=${1}
MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct

# gsm8k, ifeval, mbpp 동시 측정 (콤마로 구분)
TASK=gsm8k,ifeval,mbpp

INCLUDE_PATH=/NAS/SJ/actquant/search/tests/custom_tasks
BATCH_SIZE=auto
OUT=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/results/lm_eval/gsm8k_ifeval_mbpp

# LIMIT=10

CUDA_VISIBLE_DEVICES="${DEVICES}" python lm_eval_vllm.py \
  --model "${MODEL_PATH}/${MODEL_NAME}" \
  --task "${TASK}" \
  --include_path "${INCLUDE_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUT}" \
  --log_samples
  # "${LIMIT_ARG[@]}"
