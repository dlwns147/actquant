#!/bin/bash

TODAY=`date +%y%m%d%H%M`

## GPU Args
# Pass a comma-separated GPU list (e.g. "0,1"). AWQ + lm_eval run in this
# process; multi-GPU is only used by the (unused-here) gen-eval pool.
CUDA_VISIBLE_DEVICES=${1}
TARGET_BITS=${2}
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

## Model Args
MODEL_PATH=meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
CONFIG=amq/configs/llama.json

## Quantization / Selection Args
METHOD=awq
GROUP_SIZE=128
NUM_OF_CANDIDATES=1
TARGET_BITS_OFFSET=0.005

# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605161015_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_100_Loss_JSD_SPEC_jsd_stratified_DISCO_w2w3w4_dynamic/iter_200.stats
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605160029_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_100_Loss_JSD_SPEC_delta_hv_01_jsd_stratified/iter_200.stats
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605151031_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_100_Loss_JSD_SPEC_delta_hv_001_jsd_stratified/iter_200.stats
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605161058_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_100_Loss_JSD_SPEC_jsd_stratified_DISCO_w3_fixed/iter_200.stats

## Post-search (lm_eval) Args
SEED=0
# LM_EVAL_TASK=("gsm8k_cot" "ifeval")
LM_EVAL_TASK=("gsm8k_cot" "ifeval_test" "mbpp_test")
# LM_EVAL_TASK=( "gsm8k_cot" )
LM_EVAL_BATCH_SIZE=auto

## Eval Args (kept for shared parser)
EVAL_DATASET=("gsm8k")
EVAL_SEQLEN=2048
EVAL_SEED=0

## Base Args
MODEL_SAVE_PATH=/SSD/Woo/actquant/poc/benchmark_proxy/amq/results/quantization/${MODEL_NAME}_gen_${METHOD}_target_bits_${TARGET_BITS}_${TODAY}_100

# SAVE_PATH=amq/results/quantization/train_gsm8k/jsd_stratified/loss_jsd_spec_100.0002/jsd_stratified_seed_0_eval_gsm8k_ifeval/${MODEL_NAME}_gen_${METHOD}_target_bits_${TARGET_BITS}_${TODAY}
# SAVE_PATH=amq/results/quantization/train_gsm8k/jsd_stratified/loss_jsd_disco_w2w3w4_dynamic/jsd_stratified_disco_w2w3w4_dynamic_seed_100_eval_gsm8k_ifeval/${MODEL_NAME}_gen_${METHOD}_target_bits_${TARGET_BITS}_${TODAY}
# SAVE_PATH=amq/results/quantization/train_gsm8k/jsd_stratified/loss_jsd_disco_w3_fixed/jsd_stratified_eval_gsm8k_ifeval_final_lambda_seed_100/${MODEL_NAME}_gen_${METHOD}_target_bits_${TARGET_BITS}_${TODAY}

# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605162256_Llama-3.1-8B-Instruct_dataset_gsm8k_nsamples_16_seed_100_Loss_JSD_SPEC_jsd_stratified_disco/iter_200.stats
# SAVE_PATH=amq/results/quantization/train_gsm8k/jsd_stratified/loss_jsd_spec_0.005_disco_uniform/jsd_stratified_eval_gsm8k_ifeval_final_lambda_seed_100/${MODEL_NAME}_gen_${METHOD}_target_bits_${TARGET_BITS}_${TODAY}
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605180342_Llama-3.1-8B-Instruct_dataset_gsm8k_cot_train_ifeval_train_mbpp_train_nsamples_16_16_16_seed_100_Loss_JSD_SPEC_jsd_stratified_disco_lamoff/iter_200.stats
# SAVE_PATH=amq/results/quantization/train_gsm8k_ifeval_mbpp/jsd_spec_005/jsd_spec_005_eval_gsm8k_ifeval_mbpp_mmlu_pro_seed_100/${MODEL_NAME}_gen_${METHOD}_target_bits_${TARGET_BITS}_${TODAY}

# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_original/amq/results/search/2605020232_Llama-3.1-8B-Instruct_dataset_wikitext2_seed_0
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_original/amq/results/search/2605110301_Llama-3.1-8B-Instruct_dataset_wikitext2_seed_42
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_original/amq/results/search/2605110301_Llama-3.1-8B-Instruct_dataset_wikitext2_seed_100/iter_200.stats

# SAVE_PATH=amq/results/quantization/train_wikitext2/eval_gsm8k_ifeval_test_mbpp_test_seed_100/${MODEL_NAME}_gen_${METHOD}_target_bits_${TARGET_BITS}_${TODAY}

# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605172150_Llama-3.1-8B-Instruct_dataset_gsm8k_cot_train_ifeval_train_mbpp_train_nsamples_16_16_16_seed_0_Loss_JSD_SPEC_jsd_stratified_disco_lamoff/iter_200.stats
# LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605180050_Llama-3.1-8B-Instruct_dataset_gsm8k_cot_train_ifeval_train_mbpp_train_nsamples_16_16_16_seed_42_Loss_JSD_SPEC_jsd_stratified_disco_lamoff/iter_200.stats
LOAD=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure/amq/results/search/2605180342_Llama-3.1-8B-Instruct_dataset_gsm8k_cot_train_ifeval_train_mbpp_train_nsamples_16_16_16_seed_100_Loss_JSD_SPEC_jsd_stratified_disco_lamoff/iter_200.stats
# SAVE_PATH=amq/results/quantization/train_gsm8k_ifeval_mbpp/jsd_disco/jsd_disco_eval_gsm8k_ifeval_mbpp_mmlu_pro_seed_0/${MODEL_NAME}_gen_${METHOD}_target_bits_${TARGET_BITS}_${TODAY}
# SAVE_PATH=amq/results/quantization/train_gsm8k_ifeval_mbpp/jsd_disco/jsd_disco_eval_gsm8k_ifeval_mbpp_mmlu_pro_seed_42/${MODEL_NAME}_gen_${METHOD}_target_bits_${TARGET_BITS}_${TODAY}
SAVE_PATH=amq/results/quantization/train_gsm8k_ifeval_mbpp/jsd_disco/jsd_disco_eval_gsm8k_ifeval_mbpp_mmlu_pro_seed_100/${MODEL_NAME}_gen_${METHOD}_target_bits_${TARGET_BITS}_${TODAY}

GPU_ID=${CUDA_VISIBLE_DEVICES}

args=(
    --gpu_id ${GPU_ID}
    --model_save_path ${MODEL_SAVE_PATH}
    --save_path ${SAVE_PATH}
    --model_path ${MODEL_PATH}
    --model_name ${MODEL_NAME}
    --config ${CONFIG}
    --method ${METHOD}
    --group_size ${GROUP_SIZE}
    --num_of_candidates ${NUM_OF_CANDIDATES}
    --target_bits ${TARGET_BITS}
    --target_bits_offset ${TARGET_BITS_OFFSET}
    --load ${LOAD}
    --eval_dataset ${EVAL_DATASET[@]}
    --eval_seqlen ${EVAL_SEQLEN}
    --eval_seed ${EVAL_SEED}
    --seed ${SEED}
    --lm_eval_task ${LM_EVAL_TASK[@]}
    --lm_eval_batch_size ${LM_EVAL_BATCH_SIZE}
)

echo ${args[@]}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python amq/amq_quantization_gen.py ${args[@]}
