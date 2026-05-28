#!/bin/bash

TODAY=`date +%y%m%d%H%M`

## GPU Args
CUDA_VISIBLE_DEVICES=${1}

## Model Args
# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-3.1-8B-Instruct
# CONFIG=amq/configs/llama.json

MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
MODEL_NAME=Qwen2.5-14B-Instruct
CONFIG=amq/configs/qwen2.json

# MODEL_PATH=/SSD/huggingface/google
# MODEL_NAME=gemma-3-12b-it
# CONFIG=amq/configs/gemma.json

## Baselines: fp16, AWQ w3gs128, w3gs-1, w4gs-1 (w_bit=16 -> no quant)
W_BITS=(16 3 3 4)
GROUP_SIZES=(-1 128 -1 -1)
# W_BITS=(16)
# GROUP_SIZES=(-1)
# W_BITS=(3)
# GROUP_SIZES=(-1)
# W_BITS=(3)
# GROUP_SIZES=(4)
# W_BITS=(4)
# GROUP_SIZES=(-1)

## lm_eval Args
LM_EVAL_TASK=("gsm8k_cot" "ifeval" "mbpp")
LM_EVAL_BATCH_SIZE=auto
SEED=0

## Persistent output (lm_eval results + amq_base.json)
SAVE_PATH=amq/results/baseline/${TODAY}_${MODEL_NAME}_baselines

## Ephemeral AWQ checkpoints (one subdir per baseline).
## Set KEEP_TMP=1 to keep them; set --keep_quantized to keep individual baselines.
TMP_ROOT=$(mktemp -d -t amq_base_XXXXXX)
MODEL_SAVE_PATH=${TMP_ROOT}/awq_models
if [[ "${KEEP_TMP}" != "1" ]]; then
    trap 'rm -rf "${TMP_ROOT}"' EXIT
fi
echo "[amq_base] ephemeral TMP_ROOT=${TMP_ROOT}"
echo "[amq_base] persistent SAVE_PATH=${SAVE_PATH}"

GPU_ID=${CUDA_VISIBLE_DEVICES}

args=(
    --gpu_id ${GPU_ID}
    --save_path ${SAVE_PATH}
    --model_save_path ${MODEL_SAVE_PATH}
    --model_path ${MODEL_PATH}
    --model_name ${MODEL_NAME}
    --config ${CONFIG}
    --method awq
    --w_bit ${W_BITS[@]}
    --group_size ${GROUP_SIZES[@]}
    --seed ${SEED}
    --lm_eval_task ${LM_EVAL_TASK[@]}
    --lm_eval_batch_size ${LM_EVAL_BATCH_SIZE}
)
echo ${args[@]}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python amq/amq_base.py ${args[@]}
