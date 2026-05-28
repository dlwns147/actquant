#!/bin/bash

TODAY=`date +%y%m%d%H%M`

## GPU Args
# Pass a comma-separated GPU list (e.g. "0,1"). AWQ + lm_eval run in this
# process; multi-GPU is only used by the (unused-here) gen-eval pool.
CUDA_VISIBLE_DEVICES=${1}
TARGET_BITS=${2}
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

## Model Args
# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# CONFIG=amq/configs/qwen2.json

MODEL_PATH=/SSD/huggingface/google
MODEL_NAME=gemma-3-12b-it
CONFIG=amq/configs/gemma.json

## Quantization / Selection Args
METHOD=awq
GROUP_SIZE=128
NUM_OF_CANDIDATES=1
TARGET_BITS_OFFSET=0.005

LOAD=/NAS/SJ/actquant/amq_instruct_pure/amq/results/search/2605250014_gemma-3-12b-it_dataset_wikitext2_nsamples_128_seed_0_Loss_JSD_all_random_plain/iter_200.stats

## Post-search (lm_eval) Args
SEED=0
LM_EVAL_TASK=("gsm8k_cot" "ifeval" "mbpp")
LM_EVAL_BATCH_SIZE=auto

## Base Args
# Ephemeral dirs so the AWQ pseudo-quant checkpoint isn't kept around.
# vLLM (lm_eval backend) loads from a path, so we still need *a* path —
# this one lives in a mktemp dir and is wiped on EXIT.
# Set KEEP_TMP=1 to keep the saved AWQ model for follow-up analysis (PPL etc).
TMP_ROOT=$(mktemp -d -t amq_gen_XXXXXX)
SAVE_PATH=${TMP_ROOT}/run
MODEL_SAVE_PATH=${TMP_ROOT}/awq_model
if [[ "${KEEP_TMP}" != "1" ]]; then
    trap 'rm -rf "${TMP_ROOT}"' EXIT
fi
echo "[amq_quantization_gen] ephemeral TMP_ROOT=${TMP_ROOT}"

GPU_ID=${CUDA_VISIBLE_DEVICES}

args=(
    --gpu_id ${GPU_ID}
    --save_path ${SAVE_PATH}
    --model_save_path ${MODEL_SAVE_PATH}
    --model_path ${MODEL_PATH}
    --model_name ${MODEL_NAME}
    --config ${CONFIG}
    --method ${METHOD}
    --group_size ${GROUP_SIZE}
    --num_of_candidates ${NUM_OF_CANDIDATES}
    --target_bits ${TARGET_BITS}
    --target_bits_offset ${TARGET_BITS_OFFSET}
    --load ${LOAD}
    --seed ${SEED}
    --lm_eval_task ${LM_EVAL_TASK[@]}
    --lm_eval_batch_size ${LM_EVAL_BATCH_SIZE}
)
echo ${args[@]}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python amq/amq_quantization_gen.py ${args[@]}
