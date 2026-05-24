#!/bin/bash

TODAY=`date +%y%m%d%H%M`

## GPU Args
CUDA_VISIBLE_DEVICES=${1}
N_PROC=1
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

LLAMA_31_8B_INSTRUCT=Llama-3.1-8B-Instruct
QWEN_25_7B_INSTRUCT=Qwen2.5-7B-Instruct
QWEN_25_14B_INSTRUCT=Qwen2.5-14B-Instruct
GEMMA_3_12B_INSTRUCT=gemma-3-12b-it
GEMMA_3_27B_INSTRUCT=gemma-3-27b-it

## Model Args
# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=${LLAMA_31_8B_INSTRUCT}
# CONFIG=/NAS/SJ/actquant/amq_instruct_pure/amq/configs/llama.json
# DTYPE=float16

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=${QWEN_25_7B_INSTRUCT}
# # MODEL_NAME=${QWEN_25_14B_INSTRUCT}
# CONFIG=/NAS/SJ/actquant/amq_instruct_pure/amq/configs/qwen2.json
# DTYPE=float16

MODEL_PATH=/SSD/huggingface/google
MODEL_NAME=${GEMMA_3_12B_INSTRUCT}
# MODEL_NAME=${GEMMA_3_27B_INSTRUCT}
CONFIG=/NAS/SJ/actquant/amq_instruct_pure/amq/configs/gemma.json
DTYPE=bfloat16

# QUANTIZATION_PROXY_FAKE_PATHS=("/SSD/hqq/Llama-3.1-8B-Instruct_2bit_128gs_1axis_fake" \
# "/SSD/hqq/Llama-3.1-8B-Instruct_3bit_128gs_1axis_fake" \
# "/SSD/hqq/Llama-3.1-8B-Instruct_4bit_128gs_1axis_fake")
QUANTIZATION_PROXY_PATHS=("/SSD/hqq/${MODEL_NAME}_2bit_128gs_1axis_${DTYPE}" \
"/SSD/hqq/${MODEL_NAME}_3bit_128gs_1axis_${DTYPE}" \
"/SSD/hqq/${MODEL_NAME}_4bit_128gs_1axis_${DTYPE}")

## Search Args
SENSITIVITY_THRESHOLD=2.0
SENSITIVITY_DATASETS=wikitext2
SENSITIVITY_N_SAMPLE=128
SENSITIVITY_SEQLEN=2048

# PREDICTOR=rbf
PREDICTOR=rbf_gpu
# ITERATIONS=200
# N_DOE=250
N_ITER=50
MAX_VALUE=1.0

if [ "${MODEL_NAME}" == "${QWEN_25_7B_INSTRUCT}" ]; then
    N_DOE=200
    ITERATIONS=200
elif [ "${MODEL_NAME}" == "${LLAMA_31_8B_INSTRUCT}" ]; then
    N_DOE=250
    ITERATIONS=200
# elif [ "${MODEL_NAME}" == "${LLAMA_2_13B}" ]; then
#     N_DOE=300
#     ITERATIONS=200
elif [ "${MODEL_NAME}" == "${QWEN_25_14B_INSTRUCT}" ] || [ "${MODEL_NAME}" == "${GEMMA_3_12B_INSTRUCT}" ]; then
    N_DOE=350
    ITERATIONS=200
# elif [ "${MODEL_NAME}" == "${QWEN_25_32B}" ]; then
#     N_DOE=450
#     ITERATIONS=250
# elif [ "${MODEL_NAME}" == "${LLAMA_2_70B}" ] || [ "${MODEL_NAME}" == "${LLAMA_31_70B}" ] || [ "${MODEL_NAME}" == "${QWEN_25_72B}" ]; then
#     N_DOE=600
#     ITERATIONS=250
else
	echo "Invalid model name: ${MODEL_NAME}"
	exit
fi



SUBSET_POP_SIZE=100
GA_POP_SIZE=200
CROSSOVER_PROB=0.9
MUT_PROB=0.1

SAVE_ITER=1
RESULT_FILE=results.txt

## Data Args
# DATASETS=("gsm8k" "livebench")
# DATASETS=("gsm8k")
# DATASETS=("gsm8k_cot_train" "ifeval_train" "mbpp_train" "wikitext2")
# DATASETS=("gsm8k_cot_train" "wikitext2")
DATASETS=("wikitext2")
DATASETS_STR=$(IFS=_; echo "${DATASETS[*]}")
# Per-dataset sample size, matched positionally to DATASETS.
# Examples:
#   DATASETS=("gsm8k")              N_SAMPLE=(5)
#   DATASETS=("gsm8k" "livebench")  N_SAMPLE=(5 4)
# N_SAMPLE=(5 4)
# N_SAMPLE=(16 4)
# N_SAMPLE=(16)
N_SAMPLE=(128)
# N_SAMPLE=(16 16 16)
N_SAMPLE_STR=$(IFS=_; echo "${N_SAMPLE[*]}")
# BATCH_SIZE=8
BATCH_SIZE=1
# SEED=42
# SEED=${2}
SEED=0

# STRATIFIED=${3}
STRATIFIED=" "

## Sample-loader (only used when STRATIFIED=JSD).
##   LOADER: plain | disco
LOADER=${4:-plain}

## Worker Args
JSD_TP=3

## Base Args
SAVE_PATH=amq/results/search/${TODAY}_${MODEL_NAME}_dataset_${DATASETS_STR}_nsamples_${N_SAMPLE_STR}_seed_${SEED}_Loss_JSD

GPU_ID=${CUDA_VISIBLE_DEVICES}

if [ "$STRATIFIED" == "JSD" ]; then
    stratified="--jsd_stratified"
    SAVE_PATH=${SAVE_PATH}_jsd_stratified
else
    stratified=""
    SAVE_PATH=${SAVE_PATH}_all_random
fi

loader_args=(--loader ${LOADER})
case "$LOADER" in
    plain)          SAVE_PATH=${SAVE_PATH}_plain ;;
    disco)          SAVE_PATH=${SAVE_PATH}_disco ;;
esac

args=(
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --config ${CONFIG} \
    --quantization_proxy_paths ${QUANTIZATION_PROXY_PATHS[@]}
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
    --n_sample ${N_SAMPLE[@]} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    ${stratified} \
    ${loader_args[@]}
)
# --quantization_proxy_fake_paths ${QUANTIZATION_PROXY_FAKE_PATHS[@]}

echo ${args[@]}

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} amq/amq_search.py ${args[@]}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python amq/amq_search.py ${args[@]}
