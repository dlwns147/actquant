#!/bin/bash

TODAY=`date +%y%m%d%H%M`

## GPU Args
# Pass a comma-separated GPU list (e.g. "0,1"). AWQ + lm_eval run in this
# process; multi-GPU is only used by the (unused-here) gen-eval pool.
CUDA_VISIBLE_DEVICES=${1}
TARGET_BITS=${2}
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

## Model Args
MODEL_PATH=/SSD/huggingface/Qwen
MODEL_NAME=Qwen2.5-7B-Instruct
CONFIG=amq/configs/qwen2.json
# (debug) LOAD below is a Llama-3.1-8B SMOKE stats (32 blocks);
# Qwen2.5-7B-Instruct has 28 blocks, so AWQ apply will likely fail
# on size mismatch. Kept on purpose for pipeline debugging per user.

## Quantization / Selection Args
METHOD=awq
GROUP_SIZE=128
NUM_OF_CANDIDATES=1
TARGET_BITS_OFFSET=0.005

# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2604292238_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_0_stratified/iter_200.stats
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2604281531_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_42_stratified/iter_200.stats
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2604300605_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_100_stratified/iter_200.stats

# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605091734_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_42_jsd_stratified/iter_200.stats

# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2604292238_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_0_all_random/iter_200.stats
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2604300725_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_100_all_random/iter_200.stats

# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605030855_Llama-3.1-8B-Instruct_dataset_gsm8k_livebench_nsamples_16_seed_0_stratified/iter_200.stats
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605011244_Llama-3.1-8B-Instruct_dataset_gsm8k_livebench_nsamples_16_seed_42_stratified/iter_200.stats
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605030855_Llama-3.1-8B-Instruct_dataset_gsm8k_livebench_nsamples_16_seed_100_stratified/iter_200.stats

LOAD=/NAS/SJ/actquant/amq_instruct_pure/amq/results/search/SMOKE_2605201309_Llama-3.1-8B-Instruct_ifeval_pp_train_apps_train_n_4_4_AB/iter_3.stats

# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_original/amq/results/search/2605020232_Llama-3.1-8B-Instruct_dataset_wikitext2/iter_200.stats

## Post-search (lm_eval) Args
SEED=0
LM_EVAL_TASK=("gsm8k_cot" "ifeval" "mbpp")
# LM_EVAL_TASK=( "gsm8k_cot" )
LM_EVAL_BATCH_SIZE=auto

## Eval Args (kept for shared parser)
# EVAL_DATASET=("gsm8k")
# EVAL_SEQLEN=2048
# EVAL_SEED=0

## Base Args
# Ephemeral dirs so the AWQ pseudo-quant checkpoint isn't kept around.
# vLLM (lm_eval backend) loads from a path, so we still need *a* path —
# this one lives in a mktemp dir and is wiped on EXIT.
TMP_ROOT=$(mktemp -d -t amq_gen_XXXXXX)
SAVE_PATH=${TMP_ROOT}/run
MODEL_SAVE_PATH=${TMP_ROOT}/awq_model
trap 'rm -rf "${TMP_ROOT}"' EXIT
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
