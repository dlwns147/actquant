#!/usr/bin/env bash
# Correlation analysis: LongPPL key-token JSD vs all-token JSD on quant configs.
# Usage: bash scripts/corr_keyppl.sh 0,1,2,3
set -e

DEVICES=${1:-0,1,2,3}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

W_METHOD=hqq
KV_METHOD=kivi
W_BITS="2 3 4"
W_GROUP_SIZE=128
AXIS=1
K_GROUP_SIZE=128
V_GROUP_SIZE=128
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token
RESIDUAL_LENGTH=128

QMODEL_PATHS_LIST=()
for B in ${W_BITS}; do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

# ----- Calibration (gov_report) -----
CALIB_DATASET=gov_report
N_SAMPLE=32
SEQLEN=8192
MIN_SEQLEN=8192

# ----- Key token (LongPPL) -----
TRUNC_LEN=512
SLIDING_WINDOW=128
ALPHA=2
BETA=-2

# ----- RULER -----
RULER_TASKS="niah_single_1 ruler_vt"
RULER_N_PER_TASK=30
RULER_LENGTH=8192

# Strided forward exercises KIVI per-step KV quant during prefill.
STRIDE=512

# ----- Output -----
SAVE_ROOT=analysis/corr_keyppl/runs/${TODAY}_${MODEL_NAME}_w${W_BITS// /}_${KV_METHOD}_n${N_SAMPLE}_seq${SEQLEN}
mkdir -p ${SAVE_ROOT}

# 1) Build candidates.
CAND=${SAVE_ROOT}/candidates.json
python analysis/corr_keyppl/build_candidates.py \
    --n_block 32 \
    --w_bits ${W_BITS} \
    --kv_bits 2 3 4 \
    --kv_gs ${K_GROUP_SIZE} \
    --n 30 \
    --seed 0 \
    --out ${CAND}

# 2) Run correlation driver.
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=1 --num_machines=1 --main_process_port=${PORT_NUM} \
    analysis/corr_keyppl/run_corr.py \
    --gpu_id ${DEVICES} \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --config ${CONFIG} \
    --quant_model_paths ${QMODEL_PATHS} \
    --w_bits ${W_BITS} \
    --w_method ${W_METHOD} \
    --kv_method ${KV_METHOD} \
    --w_group_size ${W_GROUP_SIZE} \
    --k_group_size ${K_GROUP_SIZE} \
    --v_group_size ${V_GROUP_SIZE} \
    --k_quant_scheme ${K_QUANT_SCHEME} \
    --v_quant_scheme ${V_QUANT_SCHEME} \
    --residual_length ${RESIDUAL_LENGTH} \
    --dtype ${DTYPE} \
    --calib_dataset ${CALIB_DATASET} \
    --n_sample ${N_SAMPLE} \
    --seqlen ${SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --trunc_len ${TRUNC_LEN} \
    --sliding_window ${SLIDING_WINDOW} \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --ruler_tasks ${RULER_TASKS} \
    --ruler_n_per_task ${RULER_N_PER_TASK} \
    --ruler_length ${RULER_LENGTH} \
    --stride ${STRIDE} \
    --candidates ${CAND} \
    --save ${SAVE_ROOT} \
    --seed 0 \
    2>&1 | tee ${SAVE_ROOT}/run.log

# 3) Aggregate.
python analysis/corr_keyppl/aggregate.py \
    --results ${SAVE_ROOT}/results.csv \
    --save ${SAVE_ROOT}

echo "Outputs at ${SAVE_ROOT}"
