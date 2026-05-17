DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# ── Stage 2: fit surrogate from results.csv → pick final arch → benchmark ──
# Pairs with scripts/sample_surrogate.sh (stage 1). SAMPLE_PATH must
# point at a results.csv it wrote; the expr archives / model / config MUST
# match the ones used in stage 1.

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# DTYPE=float16
# CONFIG=config/qwen2.json

USE_KEY_TOKEN=False

W_METHOD=hqq
W_METHOD_TEXT=hqq
# W_METHOD=awq
# W_METHOD_TEXT=awq
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

# ── COMP_OBJ range (the deployment budget) ──
# COMP_OBJ=(memory)
# COMP_OBJ_VAL=(6271016960)
# # COMP_OBJ_THRESHOLD_LIST=($(echo "scale=3; (${COMP_OBJ_VAL[0]} * 0.05)" | bc))
# COMP_OBJ_THRESHOLD_LIST=($(echo "scale=3; (${COMP_OBJ_VAL[0]} * 0.0001)" | bc))

COMP_OBJ=(wbits kvbits kvdim)
COMP_OBJ_VAL=(3 3.25 102)
# COMP_OBJ_THRESHOLD_LIST=($(echo "scale=3; (${COMP_OBJ_VAL[0]} * 0.05)" | bc))
COMP_OBJ_THRESHOLD_LIST=(0.005 0.005 0.05)

N_TOKEN=16384

MIN_COMP_OBJ_LIST=()
MAX_COMP_OBJ_LIST=()
for IDX in "${!COMP_OBJ[@]}"; do
    MIN_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} - ${COMP_OBJ_THRESHOLD_LIST[$IDX]}" | bc) )
    MAX_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} + ${COMP_OBJ_THRESHOLD_LIST[$IDX]}" | bc) )
done
COMP_OBJ_TEXT=$(IFS="_" ; echo "${COMP_OBJ[*]}")
COMP_OBJ=$(IFS=" " ; echo "${COMP_OBJ[*]}")
MIN_COMP_OBJ=$(IFS=" " ; echo "${MIN_COMP_OBJ_LIST[*]}")
MAX_COMP_OBJ=$(IFS=" " ; echo "${MAX_COMP_OBJ_LIST[*]}")
MIN_COMP_OBJ_TEXT=$(IFS="_" ; echo "${MIN_COMP_OBJ_LIST[*]}")
MAX_COMP_OBJ_TEXT=$(IFS="_" ; echo "${MAX_COMP_OBJ_LIST[*]}")

PREFER="metric#0.0"
N=1

DATASETS="wikitext2"
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

# ── per-axis search archives — MUST match stage 1 ──
# Llama-3.1-8B-Instruct
W_EXPR=save/search/think/2605112032_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_200.stats
KV_EXPR=save/search/think/2605112033_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x2_128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_100.stats
KVDIM_EXPR=save/search/think/2605112036_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats
SAMPLE_PATH=save/result/260513/2605132157_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_qs_metric_w05595_metric_kv05595_metric_kvdim05595_rs23/results.csv

# W_EXPR=save/search/think/2605112127_Qwen2.5-7B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_200.stats
# KV_EXPR=save/search/think/2605112126_Qwen2.5-7B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x2_128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats
# KVDIM_EXPR=save/search/think/2605112128_Qwen2.5-7B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats
# SAMPLE_PATH=save/result/260513/2605130706_Qwen2.5-7B-Instruct_memory_5957466112.00_6584567808.00_hqq_kivi_wikitext2_1_kv_scale_w_expr_kvdim_expr_qs_metric_w159_metric_kvdim159_rs41/results.csv

for VAR_NAME in W_EXPR KV_EXPR KVDIM_EXPR SAMPLE_PATH; do
    VAR_VALUE="${!VAR_NAME}"
    if [ -n "${VAR_VALUE}" ] && [[ "${VAR_VALUE}" != *"${MODEL_NAME}"* ]]; then
        echo "ERROR: ${VAR_NAME} does not contain MODEL_NAME (${MODEL_NAME}): ${VAR_VALUE}"
        exit 1
    fi
done

SURROGATE=rbf
SURROGATE=ard_gp
RBF_KERNEL=tps

# TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq openbookqa social_iqa"
# TASKS="coqa gsm8k truthfulqa"
TASKS="coqa truthfulqa gsm8k"
LM_EVAL_BATCH_SIZE=1

LONGBENCH_CONFIG=utils/longbench_config
MINILONGBENCH_RESULT_PATH=save/minilongbench/${TODAY}_${MODEL_NAME}_${W_METHOD_TEXT}_${KV_METHOD}_k${K_BITS_TEXT}bits_v${V_BITS_TEXT}bits_r${RESIDUAL_LENGTH}
PASS_KEY_FILE=/NAS/SJ/actquant/search/passkey_examples.jsonl

# RULER_TASK="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery ruler_vt ruler_cwe ruler_fwe ruler_qa_squad ruler_qa_hotpot"
RULER_TASK="niah_single_1"
RULER_YAML_PATH=utils/ruler_utils
# RULER_LENGTH=4096
RULER_LENGTH=16384
# RULER_LENGTH=65536
# RULER_LENGTH=128000
# RULER_LENGTH=131072
RULER_SAMPLE=5
# RULER_SAMPLE=50
RULER_BATCH_SIZE=1
RULER_RESULT_PATH=save/ruler/${TODAY}_${MODEL_NAME}_our_${W_METHOD_TEXT}_${KV_METHOD}_${COMP_OBJ_TEXT}_${MIN_COMP_OBJ_TEXT}_${MAX_COMP_OBJ_TEXT}_k${K_BITS_TEXT}bits_k${K_GROUP_SIZE_TEXT}gs_${K_QUANT_SCHEME}_v${V_BITS_TEXT}bits_v${V_GROUP_SIZE_TEXT}gs_${V_QUANT_SCHEME}_r${RESIDUAL_LENGTH}_ruler_${RULER_LENGTH}len_${RULER_SAMPLE}sample_${RULER_BATCH_SIZE}bs_${SEED}seed


SAVE=save/post_search/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${MIN_COMP_OBJ_TEXT}_${MAX_COMP_OBJ_TEXT}_${W_METHOD_TEXT}_${KV_METHOD}_${SURROGATE}

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
-n ${N} \
--save ${SAVE} \
--sample_path ${SAMPLE_PATH} \
--surrogate ${SURROGATE} \
--rbf_kernel ${RBF_KERNEL} \
--comp_obj ${COMP_OBJ} \
--comp_obj_min ${MIN_COMP_OBJ} \
--comp_obj_max ${MAX_COMP_OBJ}"

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

# ARGS+=" --datasets ${DATASETS} --seqlen ${SEQLEN} --min_seqlen ${MIN_SEQLEN} --n_sample ${N_SAMPLE} --data_batch_size ${DATA_BATCH_SIZE}"
# ARGS+=" --zeroshot --tasks ${TASKS} --lm_eval_batch_size ${LM_EVAL_BATCH_SIZE}"
# ARGS+=" --longbench --longbench_result_path ${LONGBENCH_RESULT_PATH} --longbench_config ${LONGBENCH_CONFIG}"
# ARGS+=" --minilongbench --minilongbench_result_path ${MINILONGBENCH_RESULT_PATH} --longbench_config ${LONGBENCH_CONFIG}"
ARGS+=" --ruler --ruler_task ${RULER_TASK} --ruler_yaml_path ${RULER_YAML_PATH} --ruler_result_path ${RULER_RESULT_PATH} --ruler_batch_size ${RULER_BATCH_SIZE} --ruler_sample ${RULER_SAMPLE} --ruler_length ${RULER_LENGTH}"
# ARGS+=" --pass_key_file ${PASS_KEY_FILE}"


N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} post_search.py ${ARGS}

# W_SCALE=1
# KV_SCALE=1
# KVDIM_SCALE=1
# EFF_KVDIM_SCALE=1
# --w_scale ${W_SCALE} \
# --kv_scale ${KV_SCALE} \
# --kvdim_scale ${KVDIM_SCALE} \
# --eff_kv_scale ${EFF_KVDIM_SCALE} \

# --prefer ${PREFER}
