#!/usr/bin/env bash
# Shared config for the random / GA / AL-EI surrogate-comparison study.
# Sourced by every script in scripts/surrogate/. Keep model / quant
# / expr-archive settings here; per-script knobs live in the caller.
#
# Convention: every script `cd`s to the search/ root before launching
# python so relative paths (config/, save/, …) resolve identically.

# ── Model / config ─────────────────────────────────────────────────────────
MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# DTYPE=float16
# CONFIG=config/qwen2.json

# ── Quantisation ───────────────────────────────────────────────────────────
W_METHOD=hqq
W_METHOD_TEXT=hqq
# W_METHOD=awq
# W_METHOD_TEXT=awq
W_BITS="2 3 4"
AXIS=1
W_GROUP_SIZE=128

KV_METHOD=kivi
KV_METHOD_TEXT=kivi_think
K_BITS="2 3 4"
V_BITS="2 3 4"
K_GROUP_SIZE=("32 64 128" "32 64 128" "128")
V_GROUP_SIZE=("32 64 128" "32 64 128" "128")

RESIDUAL_LENGTH=128
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token

K_PRUNING_DIM="0 16 32 48 64"
V_PRUNING_DIM="0 16 32 48 64"

QMODEL_PATHS_LIST=()
for B in ${W_BITS}; do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

# ── Calibration data / metric (eval mode only) ─────────────────────────────
DATASETS="wikitext2"
METRIC=loss
LOSS_FUNC=jsd
N_SAMPLE=128
SEQLEN=2048
MIN_SEQLEN=2048
DATA_BATCH_SIZE=1
STRIDE=128
PREFILL_PROMPT=True
LAST_TOKENS=512

USE_KEY_TOKEN=False
TRUNC_LEN=256
SLIDING_WINDOW=64
ALPHA=2
BETA=-2
KEY_TOKEN_PATH=

# ── Per-axis search archives (must contain MODEL_NAME) ─────────────────────
W_EXPR=save/search/think/2605112032_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_200.stats
KV_EXPR=save/search/think/2605112033_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x2_128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_100.stats
KVDIM_EXPR=save/search/think/2605112036_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats

for VAR_NAME in W_EXPR KV_EXPR KVDIM_EXPR; do
    VAR_VALUE="${!VAR_NAME}"
    if [ -n "${VAR_VALUE}" ] && [[ "${VAR_VALUE}" != *"${MODEL_NAME}"* ]]; then
        echo "ERROR: ${VAR_NAME} does not contain MODEL_NAME (${MODEL_NAME})"
        exit 1
    fi
done

# ── Comparison study knobs ─────────────────────────────────────────────────
SEED=0
N_TOKEN=16384

# Shared quantile warm-start (round 0, all 3 methods use the same anchors).
QUANTILE_SAMPLE="metric_w#0.01,0.5,0.99 metric_kv#0.01,0.5,0.99 metric_kvdim#0.01,0.5,0.99"

# GA-based sampling (when --method ga).
SAMPLING_METHOD=coverage_nsga2_combined
COVERAGE_COORD=rank
COVERAGE_PER_AXIS_AGG=max
COVERAGE_PARETO_SELECT=knee

# AL EI surrogate (mean predictor; σ via brute-force LOOCV residual q0.95).
SURROGATE=ard_gp
ARD_KERNEL=matern32
GP_N_RESTARTS=10

# Per-method extras (random / ga / al_ei total = same).
N_EXTRAS=31
# AL only: batch K per round, R rounds. K × R should equal N_EXTRAS for parity.
AL_BATCH=8
AL_ROUNDS=4

# Validation hold-out (uniform random over feasible set, disjoint seed).
N_VAL=100
VAL_SEED=1000

# SLURM concurrency (override per-call as needed).
SLURM_ARRAY_CONCURRENCY=4

# Optional comp_obj pre-filter (leave empty for full PF combo).
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

# ── Common arg builders (sourced into ARGS by each script) ─────────────────
# build_model_args  → emits the model / quant / expr / comp_obj flags
build_model_args() {
    local A=""
    A+=" --model_path ${MODEL_PATH}"
    A+=" --model_name ${MODEL_NAME}"
    A+=" --config ${CONFIG}"
    A+=" --dtype ${DTYPE}"
    A+=" --seed ${SEED}"
    A+=" --w_method ${W_METHOD}"
    A+=" --kv_method ${KV_METHOD}"
    A+=" --w_bits ${W_BITS}"
    A+=" --k_bits ${K_BITS}"
    A+=" --v_bits ${V_BITS}"
    A+=" --w_group_size ${W_GROUP_SIZE}"
    A+=" --residual_length ${RESIDUAL_LENGTH}"
    A+=" --k_quant_scheme ${K_QUANT_SCHEME}"
    A+=" --v_quant_scheme ${V_QUANT_SCHEME}"
    A+=" --n_token ${N_TOKEN}"
    A+=" --expr_front"
    for g in "${K_GROUP_SIZE[@]}"; do A+=" --k_group_size ${g}"; done
    for g in "${V_GROUP_SIZE[@]}"; do A+=" --v_group_size ${g}"; done
    [ -n "${K_PRUNING_DIM}" ] && A+=" --k_pruning_dim ${K_PRUNING_DIM}"
    [ -n "${V_PRUNING_DIM}" ] && A+=" --v_pruning_dim ${V_PRUNING_DIM}"
    [ "${W_METHOD}" = "hqq" ] && A+=" --quant_model_paths ${QMODEL_PATHS}"
    [ -n "${W_EXPR}" ]     && A+=" --w_expr ${W_EXPR}"
    [ -n "${KV_EXPR}" ]    && A+=" --kv_expr ${KV_EXPR}"
    [ -n "${KVDIM_EXPR}" ] && A+=" --kvdim_expr ${KVDIM_EXPR}"
    if [ -n "${COMP_OBJ_STR}" ]; then
        A+=" --comp_obj ${COMP_OBJ_STR}"
        A+=" --comp_obj_min ${MIN_COMP_OBJ}"
        A+=" --comp_obj_max ${MAX_COMP_OBJ}"
    fi
    echo "${A}"
}

# build_eval_args  → emits calibration / data / loss flags for --mode eval
build_eval_args() {
    local A=""
    A+=" --datasets ${DATASETS}"
    A+=" --metric ${METRIC}"
    A+=" --loss_func ${LOSS_FUNC}"
    A+=" --n_sample ${N_SAMPLE}"
    A+=" --seqlen ${SEQLEN}"
    A+=" --min_seqlen ${MIN_SEQLEN}"
    A+=" --data_batch_size ${DATA_BATCH_SIZE}"
    [ ${STRIDE} -gt 0 ] && A+=" --stride ${STRIDE}"
    [ "${PREFILL_PROMPT}" = "True" ] && A+=" --prefill_prompt --last_tokens ${LAST_TOKENS}"
    if [ "${USE_KEY_TOKEN}" = "True" ]; then
        A+=" --use_key_token --trunc_len ${TRUNC_LEN} --sliding_window ${SLIDING_WINDOW}"
        A+=" --alpha ${ALPHA} --beta ${BETA}"
        [ -n "${KEY_TOKEN_PATH}" ] && A+=" --key_token_path ${KEY_TOKEN_PATH}"
    fi
    echo "${A}"
}

# build_sample_args  → emits sampling-strategy flags for --mode sample
build_sample_args() {
    local A=""
    A+=" --sampling_method ${SAMPLING_METHOD}"
    A+=" --coverage_coord ${COVERAGE_COORD}"
    A+=" --coverage_per_axis_agg ${COVERAGE_PER_AXIS_AGG}"
    A+=" --coverage_pareto_select ${COVERAGE_PARETO_SELECT}"
    A+=" --surrogate ${SURROGATE}"
    A+=" --ard_kernel ${ARD_KERNEL}"
    A+=" --gp_n_restarts ${GP_N_RESTARTS}"
    echo "${A}"
}
