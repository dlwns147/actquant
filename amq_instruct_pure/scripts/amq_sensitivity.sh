#!/bin/bash

CUDA_VISIBLE_DEVICES=${1}

## Model Args
# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-3.1-8B-Instruct
# CONFIG=amq/configs/llama.json
# DTYPE=float16

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# # MODEL_NAME=Qwen2.5-14B-Instruct
# CONFIG=amq/configs/qwen2.json
# DTYPE=float16

MODEL_PATH=/SSD/huggingface/google
MODEL_NAME=gemma-3-12b-it
# MODEL_NAME=gemma-3-27b-it
CONFIG=amq/configs/gemma.json
# DTYPE=float16
DTYPE=bfloat16

GPU_ID=${CUDA_VISIBLE_DEVICES}

QUANTIZATION_PROXY_PATHS=("/SSD/hqq/${MODEL_NAME}_2bit_128gs_1axis_${DTYPE}" \
"/SSD/hqq/${MODEL_NAME}_3bit_128gs_1axis_${DTYPE}" \
"/SSD/hqq/${MODEL_NAME}_4bit_128gs_1axis_${DTYPE}")

## Data Args
DATASET=wikitext2
SEED=0
N_SAMPLE=128
SEQLEN=2048

## Output Args
SAVE_PATH=amq/sensitivity/${MODEL_NAME}_dataset_${DATASET}_n_sample_${N_SAMPLE}_seqlen_${SEQLEN}

## Main Args
args=(
    --gpu_id ${GPU_ID} \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --quantization_proxy_paths ${QUANTIZATION_PROXY_PATHS[@]}
    --dataset ${DATASET} \
    --seed ${SEED} \
    --n_sample ${N_SAMPLE} \
    --seqlen ${SEQLEN} \
    --config ${CONFIG} \
    --save_path ${SAVE_PATH}
)

echo ${args[@]}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python amq/amq_sensitivity.py "${args[@]}"