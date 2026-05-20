#!/usr/bin/env bash
# Usage: bash scripts/correlation_sample.sh <DEVICES>
# Stage 1 of the correlation harness: build the joint combo space from per-axis
# search archives (--*_expr) and random-sample N architectures into archs.csv.
# Pairs with scripts/correlation_eval.sh (stage 2, per-row evaluation).

DEVICES=${1:-0}
TODAY=$(date +%y%m%d%H%M)
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# ── Model / config (must match the per-axis archives below) ──
MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# DTYPE=float16
# CONFIG=config/qwen2.json

W_METHOD=hqq
W_METHOD_TEXT=hqq
W_BITS="2 3 4"
AXIS=1
W_GROUP_SIZE=128

KV_METHOD=kivi
K_BITS="2 4"
V_BITS="2 4"
K_GROUP_SIZE=("128" "128")
V_GROUP_SIZE=("128" "128")

RESIDUAL_LENGTH=128
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token

QMODEL_PATHS_LIST=()
for B in ${W_BITS}; do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

SEED=0
N_TOKEN=16384
# Number of architectures (= rows in archs.csv). DO NOT confuse with the
# per-loader `n_sample` (calibration data examples per metric), which is
# set inside GROUPS in correlation.py.
N_ARCHS=50

# ── per-axis search archives (Llama-3.1-8B-Instruct, same as scripts/sample_surrogate.sh) ──
W_EXPR=save/search/think/2605112032_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_200.stats
KV_EXPR=save/search/think/2605112033_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x2_128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_100.stats
KVDIM_EXPR=save/search/think/2605112036_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats

for VAR_NAME in W_EXPR KV_EXPR KVDIM_EXPR; do
    VAR_VALUE="${!VAR_NAME}"
    if [ -n "${VAR_VALUE}" ] && [[ "${VAR_VALUE}" != *"${MODEL_NAME}"* ]]; then
        echo "ERROR: ${VAR_NAME} does not contain MODEL_NAME (${MODEL_NAME}): ${VAR_VALUE}"
        exit 1
    fi
done

# ── (optional) comp_obj pre-filter; leave empty to sample from the full PF combo ──
# COMP_OBJ=(wbits kvbits kvdim)
# COMP_OBJ_VAL=(3 3.25 102)
# COMP_OBJ_THRESHOLD_LIST=(0.5 0.5 64.0)
COMP_OBJ=()
COMP_OBJ_VAL=()
COMP_OBJ_THRESHOLD_LIST=()

MIN_COMP_OBJ_LIST=()
MAX_COMP_OBJ_LIST=()
for IDX in "${!COMP_OBJ[@]}"; do
    MIN_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} - ${COMP_OBJ_THRESHOLD_LIST[$IDX]}" | bc) )
    MAX_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} + ${COMP_OBJ_THRESHOLD_LIST[$IDX]}" | bc) )
done
COMP_OBJ_STR=$(IFS=" " ; echo "${COMP_OBJ[*]}")
MIN_COMP_OBJ=$(IFS=" " ; echo "${MIN_COMP_OBJ_LIST[*]}")
MAX_COMP_OBJ=$(IFS=" " ; echo "${MAX_COMP_OBJ_LIST[*]}")

# ── output dir (eval/aggregate read archs.csv from here) ──
SAVE=save/correlation/${TODAY}_${MODEL_NAME}_${W_METHOD_TEXT}_${KV_METHOD}_n${N_ARCHS}_s${SEED}
echo "OUTPUT -> ${SAVE}/archs.csv"

ARGS="--mode sample \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--w_method ${W_METHOD} \
--kv_method ${KV_METHOD} \
--config ${CONFIG} \
--dtype ${DTYPE} \
--w_bits ${W_BITS} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
--n_token ${N_TOKEN} \
--seed ${SEED} \
--n_archs ${N_ARCHS} \
--save ${SAVE} \
--expr_front"

for g in "${K_GROUP_SIZE[@]}"; do ARGS+=" --k_group_size ${g} "; done
for g in "${V_GROUP_SIZE[@]}"; do ARGS+=" --v_group_size ${g} "; done

[ "${W_METHOD}" = "hqq" ] && ARGS+=" --quant_model_paths ${QMODEL_PATHS} "
[ -n "${W_EXPR}" ]     && ARGS+=" --w_expr ${W_EXPR}"
[ -n "${KV_EXPR}" ]    && ARGS+=" --kv_expr ${KV_EXPR}"
[ -n "${KVDIM_EXPR}" ] && ARGS+=" --kvdim_expr ${KVDIM_EXPR}"

if [ -n "${COMP_OBJ_STR}" ]; then
    ARGS+=" --comp_obj ${COMP_OBJ_STR} --comp_obj_min ${MIN_COMP_OBJ} --comp_obj_max ${MAX_COMP_OBJ}"
fi

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} \
    --num_machines=1 --main_process_port=${PORT_NUM} \
    correlation.py ${ARGS}
