#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

# if [ $# -ne 2 ]; then
#     echo "Usage: $0 <model_name> $1 <benchmark_name>"
#     exit 1
# fi




# Root Directories
GPUS=${1} # GPU size for tensor_parallel.
ROOT_DIR="/NAS/SJ/RULER/benchmark_root" # the path that stores generated task samples and model predictions.
MODEL_DIR="/SSD/huggingface/meta-llama" # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization


# # Model and Tokenizer
TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"
SEQ_LENGTHS=(
    # 131072
    65536
    # 32768
    # 16384
    # 8192
    # 4096
)

MODEL_PATH="${MODEL_DIR}/Llama-3.1-8B-Instruct"
MODEL_TEMPLATE_TYPE="meta-llama3"

MODEL_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct"
MODEL_TEMPLATE_TYPE="qwen2.5"

MODEL_PATH="${MODEL_DIR}/Mistral-7B-Instruct-v0.3"
MODEL_TEMPLATE_TYPE="mistral-v0.3"
MODEL_FRAMEWORK="hf"

TOKENIZER_PATH=${MODEL_PATH}
TOKENIZER_TYPE="hf"

export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}


# Benchmark and Tasks
# source config_tasks.sh
# # BENCHMARK=${2}
# BENCHMARK=synthetic
# declare -n TASKS=$BENCHMARK
# if [ -z "${TASKS}" ]; then
#     echo "Benchmark: ${BENCHMARK} is not supported"
#     exit 1
# fi


# NUM_SAMPLES=500
NUM_SAMPLES=50
# NUM_SAMPLES=10
REMOVE_NEWLINE_TAB=false
STOP_WORDS=""

if [ -z "${STOP_WORDS}" ]; then
    STOP_WORDS=""
else
    STOP_WORDS="--stop_words \"${STOP_WORDS}\""
fi

if [ "${REMOVE_NEWLINE_TAB}" = false ]; then
    REMOVE_NEWLINE_TAB=""
else
    REMOVE_NEWLINE_TAB="--remove_newline_tab"
fi


# task name in `synthetic.yaml`
synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

hard_synthetic=(
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

BENCHMARK=synthetic
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${NUM_SAMPLES}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${TASKS[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPUS} python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}
        
        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=${GPUS} python pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_PATH} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            --batch_size ${BATCH_SIZE} \
            ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done
    
    CUDA_VISIBLE_DEVICES=${GPUS} python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done

echo "Total time spent on call_api: $total_time seconds"
