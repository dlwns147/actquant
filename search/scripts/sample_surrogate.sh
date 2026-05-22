DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# ── Stage 1: sample architectures + write surrogate-training results.csv ──
# Output results.csv feeds scripts/post_search.sh (--sample_path).

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# DTYPE=float16
# CONFIG=config/qwen2.json

USE_KEY_TOKEN=False

# W_METHOD=hqq
# W_METHOD_TEXT=hqq
W_METHOD=awq
W_METHOD_TEXT=awq
W_BITS="2 3 4"
AXIS=1
W_GROUP_SIZE=128

KV_METHOD="kivi"
K_BITS="2 4"
K_BITS_TEXT="24"
K_GROUP_SIZE=("128" "128")
V_BITS="2 4"
V_BITS_TEXT="24"
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

DATASETS="wikitext2"
DATASETS_TEXT="wikitext2"
METRIC="loss"
LOSS_FUNC="jsd"

N_SAMPLE=128
SEQLEN=2048
MIN_SEQLEN=2048
DATA_BATCH_SIZE=1

STRIDE=128
PREFILL_PROMPT=True
LAST_TOKENS=512

TRUNC_LEN=256
SLIDING_WINDOW=64
ALPHA=2
BETA=-2

# ── per-axis search archives ──
# Qwen2.5-7B-Instruct
# W_EXPR=save/search/think/2605112127_Qwen2.5-7B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_200.stats
# KV_EXPR=save/search/think/2605112126_Qwen2.5-7B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x2_128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats
# KVDIM_EXPR=save/search/think/2605112128_Qwen2.5-7B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats

# Llama-3.1-8B-Instruct (uncomment + set MODEL_* / CONFIG above)
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

# ── sampling design ──
RANDOM_SAMPLE=23
QUANTILE_SAMPLE="metric_w#0.01,0.5,0.99 metric_kv#0.01,0.5,0.99 metric_kvdim#0.01,0.5,0.99"
# SAMPLING_METHOD=coverage_nsga2_marginal
SAMPLING_METHOD=coverage_nsga2_combined   # 2-obj (cov_rad, std_max)
COVERAGE_COORD=rank
COVERAGE_PER_AXIS_AGG=max
COVERAGE_PARETO_SELECT=knee

# W_SCALE=1
# KV_SCALE=1
# KVDIM_SCALE=1
# EFF_KVDIM_SCALE=1

SAVE=save/result/sample/${TODAY}_${MODEL_NAME}_${W_METHOD_TEXT}_${KV_METHOD}_${DATASETS_TEXT}_sample_${SEED}seed
[ -n "${W_EXPR}" ]      && SAVE+="_w_expr"
[ -n "${KV_EXPR}" ]     && SAVE+="_kv_expr"
[ -n "${KVDIM_EXPR}" ]  && SAVE+="_kvdim_expr"
[ -n "${EFF_KV_EXPR}" ] && SAVE+="_eff_kv_expr"
if [ -n "${QUANTILE_SAMPLE}" ]; then
    QS_TEXT=""
    for entry in ${QUANTILE_SAMPLE}; do
        metric="${entry%%#*}"; quantiles="${entry#*#}"
        short_q="${quantiles//0./}"; short_q="${short_q//,/}"
        QS_TEXT+="_${metric%bits}${short_q}"
    done
    SAVE+="_qs${QS_TEXT}"
fi
if [ -n "${RANDOM_SAMPLE}" ] && [ "${RANDOM_SAMPLE}" -gt 0 ]; then
    SAVE+="_rs${RANDOM_SAMPLE}"
    case "${SAMPLING_METHOD}" in
        random)                  SAVE+="_r" ;;
        coverage_nsga2_joint)    SAVE+="_j${COVERAGE_COORD:0:1}" ;;
        coverage_nsga2_marginal) SAVE+="_m${COVERAGE_COORD:0:1}${COVERAGE_PER_AXIS_AGG:0:1}" ;;
        coverage_nsga2_combined) SAVE+="_c${COVERAGE_COORD:0:1}${COVERAGE_PARETO_SELECT:0:1}" ;;
    esac
fi
echo "RESULTS CSV -> ${SAVE}/results.csv"

ARGS="--gpu_id ${DEVICES} \
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
--expr_front \
--metric ${METRIC} \
--loss_func ${LOSS_FUNC} \
--seed ${SEED} \
--save ${SAVE} \
--datasets ${DATASETS} \
--seqlen ${SEQLEN} \
--min_seqlen ${MIN_SEQLEN} \
--n_sample ${N_SAMPLE} \
--data_batch_size ${DATA_BATCH_SIZE} \
--random_sample ${RANDOM_SAMPLE} \
--quantile_sample ${QUANTILE_SAMPLE} \
--sampling_method ${SAMPLING_METHOD} \
--coverage_coord ${COVERAGE_COORD} \
--coverage_per_axis_agg ${COVERAGE_PER_AXIS_AGG} \
--coverage_pareto_select ${COVERAGE_PARETO_SELECT}"

for g in "${K_GROUP_SIZE[@]}"; do ARGS+=" --k_group_size ${g} "; done
for g in "${V_GROUP_SIZE[@]}"; do ARGS+=" --v_group_size ${g} "; done

if [ ${USE_KEY_TOKEN} == 'True' ]; then
    ARGS+=" --use_key_token --trunc_len ${TRUNC_LEN} --sliding_window ${SLIDING_WINDOW} --alpha ${ALPHA} --beta ${BETA} --key_token_path ${KEY_TOKEN_PATH} "
fi
[ ${STRIDE} -gt 0 ] && ARGS+=" --stride ${STRIDE} "
[ ${PREFILL_PROMPT} == 'True' ] && ARGS+=" --prefill_prompt --last_tokens ${LAST_TOKENS} "
[ ${W_METHOD} == "hqq" ] && ARGS+=" --quant_model_paths ${QMODEL_PATHS} "
[ -n "${W_EXPR}" ]      && ARGS+=" --w_expr ${W_EXPR}"
[ -n "${KV_EXPR}" ]     && ARGS+=" --kv_expr ${KV_EXPR}"
[ -n "${KVDIM_EXPR}" ]  && ARGS+=" --kvdim_expr ${KVDIM_EXPR}"
[ -n "${EFF_KV_EXPR}" ] && ARGS+=" --eff_kv_expr ${EFF_KV_EXPR}"

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} sample_surrogate.py ${ARGS}

# --w_scale ${W_SCALE} \
# --kv_scale ${KV_SCALE} \
# --kvdim_scale ${KVDIM_SCALE} \
# --eff_kv_scale ${EFF_KVDIM_SCALE} \