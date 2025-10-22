DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# # MODEL_NAME=Qwen2.5-7B
# # MODEL_NAME=Qwen2.5-14B
# # MODEL_NAME=Qwen2.5-32B
# # MODEL_NAME=Qwen2.5-72B
# # MODEL_NAME=Qwen2.5-7B-Instruct
# MODEL_NAME=Qwen2.5-14B-Instruct
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# # MODEL_NAME=Mistral-7B-v0.3
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/mistral.json

# W_METHOD="hqq layer_prune"
# W_METHOD_TEXT="hqq_layer_prune"

W_METHOD=hqq
W_METHOD_TEXT=hqq
# W_METHOD=awq
# W_METHOD_TEXT=awq
# W_METHOD="awq layer_prune"
# W_METHOD_TEXT=awq_layer_prune
# W_METHOD=fp16
# W_METHOD_TEXT=fp16

W_BITS="2 3 4"
W_BITS_TEXT="234"
# W_BITS="2 4"
# W_BITS_TEXT="24"
# W_BITS="16"
# W_BITS_TEXT="16"
AXIS=1
W_GROUP_SIZE=128
QSCALE=false

# KV_METHOD="hqq"
KV_METHOD="kivi"

K_BITS="2 4"
K_BITS_TEXT="24"
K_GROUP_SIZE=("128" "128")
K_GROUP_SIZE_TEXT=128x2

V_BITS="2 4"
V_BITS_TEXT="24"
V_GROUP_SIZE=("128" "128")
V_GROUP_SIZE_TEXT=128x2

RESIDUAL_LENGTH=128
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token

QMODEL_PATHS_LIST=()
for B in ${W_BITS}
do
    # QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
# QMODEL_PATHS=( "/SSD/hqq/${MODEL_NAME}_2bit_64gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_3bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_4bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}")
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

N_OUTLIER=32
OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth

# COMP_OBJ=(wbits kvbits)
# COMP_OBJ_TEXT="wkv"
# COMP_OBJ_VAL=(3 3.25)
# # COMP_OBJ_VAL=(3 4.25)
# # COMP_OBJ_VAL=(4.25 3.25)
# COMP_OBJ_THRESHOLD=0.005

# COMP_OBJ=(kvbits)
# COMP_OBJ_VAL=(3.0)
# COMP_OBJ_THRESHOLD=0.005

# N_TOKEN=1024

# COMP_OBJ=(memory)

# COMP_OBJ_VAL=(5862072320)
# # COMP_OBJ_VAL=(5666250752)
# # COMP_OBJ_VAL=(5649473536) # LLama 3.1 8B
# # # COMP_OBJ_VAL=(5006434304) # LLama 3.1 8B
# # # COMP_OBJ_VAL=(4989657088) # LLama 3.1 8B
# # # COMP_OBJ_VAL=(4793835520) # LLama 3.1 8B
# # # COMP_OBJ_VAL=(4777058304) # LLama 3.1 8B
# # COMP_OBJ_VAL=(4134019072) # LLama 3.1 8B
# N_TOKEN=1024

COMP_OBJ=(kvbits memory)
COMP_OBJ_VAL=(2.25 8264884224)
N_TOKEN=131072

# COMP_OBJ_VAL=(10194001920)
# COMP_OBJ_VAL=(9534185472)
# COMP_OBJ_VAL=(9321586688)
# COMP_OBJ_VAL=(8661770240)
# COMP_OBJ_VAL=(8259117056)
# COMP_OBJ_VAL=(8046518272)
# COMP_OBJ_VAL=(7386701824)
# COMP_OBJ_VAL=(7174103040)
# # N_TOKEN=1024


# # COMP_OBJ_VAL=(25170550784)
# # COMP_OBJ_VAL=(42137821184)
# # COMP_OBJ_VAL=(24957952000)
# # COMP_OBJ_VAL=(41478004736)
# # COMP_OBJ_VAL=(24298135552)
# # COMP_OBJ_VAL=(41265405952)
# COMP_OBJ_VAL=(24085536768)
# # COMP_OBJ_VAL=(40605589504)
# N_TOKEN=1048576

COMP_OBJ_THRESHOLD=$(echo "scale=3; (${COMP_OBJ_VAL[0]} * 0.001)" | bc)
COMP_OBJ_THRESHOLD_LIST=(0.005 $(echo "scale=3; (${COMP_OBJ_VAL[0]} * 0.001)" | bc))

# PREFER="metric#0.0 ${TARGET_COMP_OBJ}#${TARGET_COMP_OBJ_VAL}"

PREFER_LIST=("metric#0.0")
MIN_COMP_OBJ_LIST=()
MAX_COMP_OBJ_LIST=()

for IDX in "${!COMP_OBJ[@]}"
do
    PREFER_LIST+=( "${COMP_OBJ[$IDX]}#${COMP_OBJ_VAL[$IDX]}" )
    # if [ "${COMP_OBJ[$IDX]}" == "memory" ]; then
    #     MIN_COMP_OBJ_LIST+=( 0 )
    # elif [[ "${COMP_OBJ[$IDX]}" == "kvbits" || "${COMP_OBJ[$IDX]}" == "wbits" ]]; then
    #     MIN_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} - $COMP_OBJ_THRESHOLD" | bc) )
    # fi
    # MIN_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} - $COMP_OBJ_THRESHOLD" | bc) )
    # MAX_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} + $COMP_OBJ_THRESHOLD" | bc) )
    # MIN_COMP_OBJ_LIST+=( 1 )
    # MAX_COMP_OBJ_LIST+=( 1e99 )
    MIN_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} - ${COMP_OBJ_THRESHOLD_LIST[$IDX]}" | bc) )
    MAX_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} + ${COMP_OBJ_THRESHOLD_LIST[$IDX]}" | bc) )
done

COMP_OBJ_TEXT=$(IFS="_" ; echo "${COMP_OBJ[*]}")
COMP_OBJ=$(IFS=" " ; echo "${COMP_OBJ[*]}")
COMP_OBJ_VAL=$(IFS=" " ; echo "${COMP_OBJ_VAL[*]}")
PREFER=$(IFS=" " ; echo "${PREFER_LIST[*]}")
MIN_COMP_OBJ=$(IFS=" " ; echo "${MIN_COMP_OBJ_LIST[*]}")
MAX_COMP_OBJ=$(IFS=" " ; echo "${MAX_COMP_OBJ_LIST[*]}")
MIN_COMP_OBJ_TEXT=$(IFS="_" ; echo "${MIN_COMP_OBJ_LIST[*]}")
MAX_COMP_OBJ_TEXT=$(IFS="_" ; echo "${MAX_COMP_OBJ_LIST[*]}")

# DATASETS="wikitext2 c4"
# DATASETS_TEXT="wikitext2_c4"
# METRIC="ppl"
# LOSS_FUNC="cross_entropy"

DATASETS="wikitext2"
DATASETS_TEXT="wikitext2"
METRIC="loss"
LOSS_FUNC="jsd"

# LOSS_FUNC="cross_entropy"

# TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq openbookqa social_iqa"
# TASKS="coqa gsm8k truthfulqa"
TASKS="coqa truthfulqa gsm8k"

LM_EVAL_BATCH_SIZE=32

# KV_SCALE=1
# KV_SCALE=0.95
KV_SCALE=0.9
# KV_SCALE=0.85
# KV_SCALE=0.8

# TRUNC_LEN=512
TRUNC_LEN=256
# SLIDING_WINDOW=128
SLIDING_WINDOW=64

ALPHA=2
BETA=-2

# EXPR_FOLDER=save/search/quant

W_EXPR=save/search/quant/2508271327_Llama-3.1-8B-Instruct_w_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234k4v4bits_w128kvgs_128res_len_k_channel_v_token_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0min_0token_rbf_256trunc_64sw/iter_200.stats
KV_EXPR=save/search/quant/2508271349_Llama-3.1-8B-Instruct_kv_loss_w_hqq_kv_kivi_iter_100_n_iter_30_w4k234v234bits_w128k3264128x3v3264128x3gs_128res_len_k_channel_v_token_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0min_0token_rbf_256trunc_64sw/iter_100.stats

LONG_BENCH_RESULT_PATH=save/long_bench/${TODAY}_${MODEL_NAME}_our_${W_METHOD_TEXT}_${KV_METHOD}_${COMP_OBJ_TEXT}_${MIN_COMP_OBJ_TEXT}_${MAX_COMP_OBJ_TEXT}_k${K_BITS_TEXT}bits_k${K_GROUP_SIZE_TEXT}gs_${K_QUANT_SCHEME}_v${V_BITS_TEXT}bits_v${V_GROUP_SIZE_TEXT}gs_${V_QUANT_SCHEME}_r${RESIDUAL_LENGTH}
LONG_BENCH_CONFIG=utils/long_bench_config
LONG_BENCH_TASK=""

PASS_KEY_FILE=/NAS/SJ/actquant/search/passkey_examples.jsonl

# N=1
N=10

RANDOM_SAMPLE=1000
SAVE=save/result/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${MIN_COMP_OBJ_TEXT}_${MAX_COMP_OBJ_TEXT}_${W_METHOD_TEXT}_${KV_METHOD}_${DATASETS_TEXT}_${KV_SCALE}_kv_scale_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
# SAVE=save/result/${TODAY}_${MODEL_NAME}_random_sample_${W_METHOD_TEXT}_${KV_METHOD}_${RANDOM_SAMPLE}_sample_${SEED}seed_${KV_SCALE}_kv_scale_${DATASETS_TEXT}_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
# SAVE=save/result/${TODAY}_${MODEL_NAME}_random_sample_${W_METHOD_TEXT}_${KV_METHOD}_${RANDOM_SAMPLE}_sample_${SEED}seed_${KV_SCALE}_kv_scale_sqrt_${DATASETS_TEXT}_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta


ARGS="--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--w_method ${W_METHOD} \
--kv_method ${KV_METHOD} \
--config ${CONFIG} \
--dtype ${DTYPE} \
--comp_obj ${COMP_OBJ} \
--comp_obj_min ${MIN_COMP_OBJ} \
--comp_obj_max ${MAX_COMP_OBJ} \
--w_bits ${W_BITS} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
--n_token ${N_TOKEN} \
--debug \
--w_expr ${W_EXPR} \
--kv_expr ${KV_EXPR} \
--expr_front \
--datasets ${DATASETS} \
--metric ${METRIC} \
--loss_func ${LOSS_FUNC} \
--random_sample ${RANDOM_SAMPLE} \
--save ${SAVE} \
--quant_model_paths ${QMODEL_PATHS} \
--kv_scale ${KV_SCALE} 
"
# --sqrt \
# --use_key_token \
# --trunc_len ${TRUNC_LEN} \
# --sliding_window ${SLIDING_WINDOW} \
# --alpha ${ALPHA} \
# --beta ${BETA}
# --prefer ${PREFER} \
# -n ${N}
# --zeroshot \
# --tasks ${TASKS} \
# --lm_eval_batch_size ${LM_EVAL_BATCH_SIZE} \
# --long_bench \
# --long_bench_result_path ${LONG_BENCH_RESULT_PATH} \
# --long_bench_config ${LONG_BENCH_CONFIG}

for g in "${K_GROUP_SIZE[@]}"
do
    ARGS+=" --k_group_size ${g} "
done

for g in "${V_GROUP_SIZE[@]}"
do
    ARGS+="--v_group_size ${g} "
done

# -n ${N} \
N_PROC=1
# N_PROC=2
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} post_search_split.py ${ARGS}
