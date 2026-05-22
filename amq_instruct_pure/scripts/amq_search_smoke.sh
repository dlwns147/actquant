#!/bin/bash
# Smoke test: just verify the JSD-stratified DISCO pipeline wires up with
# the newly extracted ifeval_pp_train / apps_train data. Tiny iteration
# counts and only A+B groups (set via DISCO_GROUPS_INCLUDE).
#
# Usage: bash scripts/amq_search_smoke.sh [GPUs]  (default: 0,1)
set -e
REPO_DIR=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure
cd ${REPO_DIR}

GPUS=${1:-0,1}
TODAY=$(date +%y%m%d%H%M)

# A,B groups only (env var read by DISCO/sample_loader_disco.py)
export DISCO_GROUPS_INCLUDE=A,B

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
CONFIG=${REPO_DIR}/amq/configs/llama.json

QUANTIZATION_PROXY_PATHS=( \
  /SSD/hqq/Llama-3.1-8B-Instruct_2bit_128gs_1axis_float16 \
  /SSD/hqq/Llama-3.1-8B-Instruct_3bit_128gs_1axis_float16 \
  /SSD/hqq/Llama-3.1-8B-Instruct_4bit_128gs_1axis_float16 \
)

DATASETS=("ifeval_pp_train" "apps_train")
N_SAMPLE=(4 4)
DATASETS_STR=$(IFS=_; echo "${DATASETS[*]}")
N_SAMPLE_STR=$(IFS=_; echo "${N_SAMPLE[*]}")

SAVE_PATH=amq/results/search/SMOKE_${TODAY}_${MODEL_NAME}_${DATASETS_STR}_n_${N_SAMPLE_STR}_AB

args=(
    --model_path ${MODEL_PATH}
    --model_name ${MODEL_NAME}
    --config ${CONFIG}
    --quantization_proxy_paths ${QUANTIZATION_PROXY_PATHS[@]}
    --gpu_id ${GPUS}
    --sensitivity_threshold 2.0
    --sensitivity_datasets wikitext2
    --sensitivity_n_sample 128           # match existing cache (no recompute)
    --sensitivity_seqlen 2048
    --predictor rbf_gpu
    --iterations 3                       # smoke: was 200
    --n_doe 256                          # min 224 for rbf_gpu (1+dim)
    --n_iter 2                           # smoke: was 50
    --max_value 1.0
    --subset_pop_size 16                 # smoke: was 100
    --ga_pop_size 32                     # smoke: was 200
    --crossover_prob 0.9
    --mut_prob 0.1
    --save_iter 1
    --save_path ${SAVE_PATH}
    --result_file results.txt
    --jsd_tp 1
    --datasets ${DATASETS[@]}
    --n_sample ${N_SAMPLE[@]}
    --batch_size 4
    --seed 42
    --jsd_stratified
    --loader disco
)

echo "DISCO_GROUPS_INCLUDE=${DISCO_GROUPS_INCLUDE}"
echo "${args[@]}"
CUDA_VISIBLE_DEVICES=${GPUS} python amq/amq_search.py "${args[@]}"
