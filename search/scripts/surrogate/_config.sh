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
CONFIG=config/llama.json
# 2-axis (W, eff_kv) study: eff_kv search archive exists only for Llama.

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# CONFIG=config/qwen2.json
# HQQ banks on disk are bfloat16 (no float16 build exists); QMODEL_PATHS embeds
# ${DTYPE} in the dir name, so this MUST be bfloat16 for the hqq path to resolve.
DTYPE=bfloat16

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
# Protocol matched to the save/think eff_kvbits archive (pp512, stride 32,
# 128 sample, 2048 seq, 0 token). NOTE: the W archive (2606070017) is stride
# 128, the eff_kv archive (2606091743) is stride 32 — eval here uses stride 32
# (the more answer-sensitive eff_kv axis); the W-axis per-axis X is the only
# slightly cross-protocol piece (all methods share it → comparison still fair).
N_SAMPLE=128
SEQLEN=2048
MIN_SEQLEN=2048
DATA_BATCH_SIZE=1
STRIDE=32
PREFILL_PROMPT=True
LAST_TOKENS=512

USE_KEY_TOKEN=False
TRUNC_LEN=256
SLIDING_WINDOW=64
ALPHA=2
BETA=-2
KEY_TOKEN_PATH=

# ── Per-axis search archives (must contain MODEL_NAME) ─────────────────────
# 2-axis (W, eff_kv): eff_kv collapses KV-bits+KV-dim into ONE eff_kvbits axis.
# Both from save/search/think at the SAME protocol (pp512, stride 32, 128 sample,
# 2048 seq, 0 token) → per-axis X and eval y are the same flavour. eff_kv search
# varied KV bits+gs+prune (W fixed 4); W search varied W bits (KV fixed 4).
W_EXPR=save/search/think/2605290210_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_32stride_pp512/iter_200.stats
KV_EXPR=
KVDIM_EXPR=
EFF_KV_EXPR=save/search/think/2606091743_Llama-3.1-8B-Instruct_eff_kvbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w4kv234bits_w128kv3264128x3gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0.1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_32stride_pp512/iter_200.stats

for VAR_NAME in W_EXPR KV_EXPR KVDIM_EXPR EFF_KV_EXPR; do
    VAR_VALUE="${!VAR_NAME}"
    if [ -n "${VAR_VALUE}" ] && [[ "${VAR_VALUE}" != *"${MODEL_NAME}"* ]]; then
        echo "ERROR: ${VAR_NAME} does not contain MODEL_NAME (${MODEL_NAME})"
        exit 1
    fi
done

# ── Comparison study knobs ─────────────────────────────────────────────────
SEED=0
N_TOKEN=0

# Shared quantile warm-start (round 0 = AL initial values; all methods share it).
QUANTILE_SAMPLE="metric_w#0.01,0.5,0.99 metric_eff_kv#0.01,0.5,0.99"

# GA-based sampling (when --method ga).
SAMPLING_METHOD=coverage_nsga2_combined
COVERAGE_COORD=rank
COVERAGE_PER_AXIS_AGG=max
COVERAGE_PARETO_SELECT=knee

# Surrogate for the aggregate validation fit (and AL-EI mean / σ_conf LOOCV).
# RECIPE: rbf (tps kernel) — maximin × rbf-tps is the TOP combo (multi-seed
# R² 0.9694). Pair with the maximin coverage sampler (run_compare_local … maximin).
# sqrty_ard_gp is the robust alternative. ⚠️ NEVER pair rbf-tps with an
# uncertainty acquisition (ACQ=alm/imse/qbc/rank) — interpolant extrapolates
# wildly on clustered picks (R² went negative); rbf-tps is for maximin/coverage only.
SURROGATE=rbf
RBF_KERNEL=tps
ARD_KERNEL=matern32
GP_N_RESTARTS=10

# AL acquisition selector (used when METHOD=al, or the al_<acq> shorthand in
# sample.sh). One of: ei ucb (objective B = lowest-JSD arch in band) |
# alm imse maximin qbc rank (objective A = global surrogate-ranking accuracy).
# imse needs SURROGATE=ard_gp; maximin/qbc/rank also work with rbf/tps.
ACQ=imse
AL_UCB_KAPPA=2.0
AL_QBC_B=20
AL_POOL_CAP=8000
AL_DIVERSE=True
# Improvement knobs: AL_CAND front = Pareto-frontier-focused candidate pool (1);
# AL_TRANSFORM sqrt = stabilise heavy-tailed JSD for the acquisition surrogate (2).
# (ga_imse method = GA-coverage + front/sqrt IMSE-refine = improvements 1+2+3.)
AL_CAND=random
AL_TRANSFORM=none

# Per-method extras (random / ga / al_ei total = same).
N_EXTRAS=31
# AL only: batch K per round, R rounds. K × R should equal N_EXTRAS for parity.
AL_BATCH=8
AL_ROUNDS=4

# Validation hold-out (uniform random over feasible set, disjoint seed).
N_VAL=100
VAL_SEED=1000
# Region-validation knobs (sample.sh validation_band / validation_front +
# aggregate region split). VAL_BAND_OBJ empty = auto (first non-w axis comp,
# e.g. eff_kvbits). The random pool is bulk-weighted (pool density follows the
# per-axis-front product), so band/front pools add equal-per-band precision +
# the argmin-in-band ("Pareto-front combo") locus post_search selects from.
VAL_BANDS=6
VAL_BAND_OBJ=
VAL_FRONT_WBANDS=4

# SLURM concurrency (override per-call as needed).
SLURM_ARRAY_CONCURRENCY=4

# Optional comp_obj pre-filter (empty = full PF combo pool).
# HARD_BAND=1 env → aggressive low-bit deployment band eff_kvbits ∈ [2.0, 2.7]
# (llama_effkv_hard study: in-band SAMPLING pool). Leave unset for the normal
# global pool — the deployment protocol is GLOBAL-train → band-local validate,
# so only the in-band-training upper-bound study sets this.
if [ -n "${HARD_BAND:-}" ]; then
    COMP_OBJ=(eff_kvbits)
    COMP_OBJ_VAL=(2.35)
    COMP_OBJ_THRESHOLD_LIST=(0.35)
else
    COMP_OBJ=()
    COMP_OBJ_VAL=()
    COMP_OBJ_THRESHOLD_LIST=()
fi

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
    [ -n "${W_EXPR}" ]      && A+=" --w_expr ${W_EXPR}"
    [ -n "${KV_EXPR}" ]     && A+=" --kv_expr ${KV_EXPR}"
    [ -n "${KVDIM_EXPR}" ]  && A+=" --kvdim_expr ${KVDIM_EXPR}"
    [ -n "${EFF_KV_EXPR}" ] && A+=" --eff_kv_expr ${EFF_KV_EXPR}"
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
    A+=" --rbf_kernel ${RBF_KERNEL}"
    A+=" --ard_kernel ${ARD_KERNEL}"
    A+=" --gp_n_restarts ${GP_N_RESTARTS}"
    A+=" --al_ucb_kappa ${AL_UCB_KAPPA}"
    A+=" --al_qbc_B ${AL_QBC_B}"
    A+=" --al_pool_cap ${AL_POOL_CAP}"
    A+=" --al_cand ${AL_CAND}"
    A+=" --al_transform ${AL_TRANSFORM}"
    [ "${AL_DIVERSE}" = "True" ] && A+=" --al_diverse"
    echo "${A}"
}
