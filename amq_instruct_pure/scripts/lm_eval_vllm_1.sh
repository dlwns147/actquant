DEVICES=${1}
MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
# TASK=ifeval_pp,apps
# OUT=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/results/lm_eval/ifeval_apps
# TASK=ifeval_pp
# OUT=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/results/lm_eval/ifeval
TASK=apps
OUT=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/results/lm_eval/apps

INCLUDE_PATH=/NAS/SJ/actquant/search/tests/custom_tasks
BATCH_SIZE=auto

# LIMIT=10

CUDA_VISIBLE_DEVICES="${DEVICES}" python lm_eval_vllm.py \
  --model "${MODEL_PATH}/${MODEL_NAME}" \
  --task "${TASK}" \
  --include_path "${INCLUDE_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUT}" \
  --log_samples
  # "${LIMIT_ARG[@]}"

