#!/bin/bash
# all_random gsm8k n=64 search. Args: <gpu> <seed>

TODAY=`date +%y%m%d%H%M`

CUDA_VISIBLE_DEVICES=${1}
N_PROC=1
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
CONFIG=amq/configs/llama.json

QUANTIZATION_PROXY_FAKE_PATHS=("/SSD/Woo/hqq/Llama-3.1-8B-Instruct_2bit_128gs_1axis_fake" "/SSD/Woo/hqq/Llama-3.1-8B-Instruct_3bit_128gs_1axis_fake" "/SSD/Woo/hqq/Llama-3.1-8B-Instruct_4bit_128gs_1axis_fake")
QUANTIZATION_PROXY_PATHS=("/SSD/Woo/hqq/Llama-3.1-8B-Instruct_2bit_128gs_1axis" "/SSD/Woo/hqq/Llama-3.1-8B-Instruct_3bit_128gs_1axis" "/SSD/Woo/hqq/Llama-3.1-8B-Instruct_4bit_128gs_1axis")

SENSITIVITY_THRESHOLD=2.0
SENSITIVITY_DATASETS=wikitext2
SENSITIVITY_N_SAMPLE=128
SENSITIVITY_SEQLEN=2048

PREDICTOR=rbf_gpu
ITERATIONS=200
N_DOE=250
N_ITER=50
MAX_VALUE=1.0

SUBSET_POP_SIZE=100
GA_POP_SIZE=200
CROSSOVER_PROB=0.9
MUT_PROB=0.1

SAVE_ITER=1
RESULT_FILE=results.txt

DATASETS=("gsm8k")
DATASETS_STR=$(IFS=_; echo "${DATASETS[*]}")
N_SAMPLE=64
BATCH_SIZE=1
SEED=${2}

JSD_TP=1

SAVE_PATH=amq/results/search/${TODAY}_${MODEL_NAME}_dataset_${DATASETS_STR}_nsamples_${N_SAMPLE}_seed_${SEED}_all_random
GPU_ID=${CUDA_VISIBLE_DEVICES}

args=(
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --config ${CONFIG} \
    --quantization_proxy_paths ${QUANTIZATION_PROXY_PATHS[@]}
    --quantization_proxy_fake_paths ${QUANTIZATION_PROXY_FAKE_PATHS[@]}
    --gpu_id ${GPU_ID} \
    --sensitivity_threshold ${SENSITIVITY_THRESHOLD} \
    --sensitivity_datasets ${SENSITIVITY_DATASETS} \
    --sensitivity_n_sample ${SENSITIVITY_N_SAMPLE} \
    --sensitivity_seqlen ${SENSITIVITY_SEQLEN} \
    --predictor ${PREDICTOR} \
    --iterations ${ITERATIONS} \
    --n_doe ${N_DOE} \
    --n_iter ${N_ITER} \
    --max_value ${MAX_VALUE} \
    --subset_pop_size ${SUBSET_POP_SIZE} \
    --ga_pop_size ${GA_POP_SIZE} \
    --crossover_prob ${CROSSOVER_PROB} \
    --mut_prob ${MUT_PROB} \
    --save_iter ${SAVE_ITER} \
    --save_path ${SAVE_PATH} \
    --result_file ${RESULT_FILE} \
    --jsd_tp ${JSD_TP} \
    --datasets ${DATASETS[@]} \
    --n_sample ${N_SAMPLE} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED}
)

echo ${args[@]}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python amq/amq_search.py ${args[@]}
