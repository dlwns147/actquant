DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# # MODEL_NAME=Qwen2.5-7B-Instruct
# # MODEL_NAME=Qwen2.5-14B-Instruct
# MODEL_NAME=Qwen2.5-72B-Instruct
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/mistral.json

USE_KEY_TOKEN=True
# USE_KEY_TOKEN=False

# W_METHOD="hqq layer_prune"
# W_METHOD_TEXT="hqq_layer_prune"
W_METHOD="hqq"
W_METHOD_TEXT="hqq"

AXIS=1
QSCALE=false
QZERO=false

# KV_METHOD="hqq"
KV_METHOD="kivi"

K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token

# RESIDUAL_LENGTH=0
RESIDUAL_LENGTH=128

W_BITS="2 3 4"
W_BITS_TEXT="234"
W_GROUP_SIZE=128

KV_BITS="2 3 4"
KV_BITS_TEXT="234"
KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
KV_GROUP_SIZE_TEXT=3264128x35
N_TOKEN=0

W_PARETO_PATH=save/search/quant/2601141301_Llama-3.1-8B-Instruct_w_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0min_0token_rbf/iter_200.stats
KV_PARETO_PATH=save/search/quant/2601141301_Llama-3.1-8B-Instruct_kv_loss_w_hqq_kv_kivi_iter_100_n_iter_30_w4kv234bits_w128kv3264128x3gs_128res_len_k_channel_v_token_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0min_0token_rbf/iter_100.stats

# N_DOE=100
N_DOE=500
# N_DOE=2000
ITER=100
N_ITER=50

QMODEL_PATHS_LIST=()
for B in ${W_BITS}
do
    # QMODEL_PATHS+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")


# OUTLIER_BITS="2 3"
# OUTLIER_TEXT=23

OUTLIER_BITS="2 3 4"
OUTLIER_TEXT=234

N_OUTLIER=32
OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth

LOSS_FUNC=cross_entropy
# LOSS_FUNC=jsd

# PREDICTOR=mlp
PREDICTOR=rbf

# DATASET=wikitext2
# # DATASET=c4
# N_SAMPLE=32
# # N_SAMPLE=128
# SEQLEN=2048
# # SEQLEN=4096
# DATA_BATCH_SIZE=1
# MIN_SEQLEN=0

DATASET=gov_report
# N_SAMPLE=4
N_SAMPLE=8
# N_SAMPLE=16
# N_SAMPLE=32
# N_SAMPLE=64
# SEQLEN=2048
# MIN_SEQLEN=2048
SEQLEN=8192
MIN_SEQLEN=8192
# SEQLEN=16384
# MIN_SEQLEN=16384
DATA_BATCH_SIZE=1
# MIN_SEQLEN=0


# GA_POP_SIZE=200
# GA_POP_SIZE=100

METRIC=loss
# METRIC=gsm8k
# METRIC=gsm8k_cot

if [ ${LOSS_FUNC} == 'cross_entropy' ]; then   
    MAX_VALUE=20
elif [ ${LOSS_FUNC} == 'jsd' ]; then
    MAX_VALUE=0.7
fi
# MAX_VALUE=1
MUT_PROB=0.1
CROSSOVER_PROB=0.9

SAVE_ITER=1
# SAVE_ITER=10


# TRUNC_LEN=4096
# SLIDING_WINDOW=1024
TRUNC_LEN=1024
SLIDING_WINDOW=1024
# TRUNC_LEN=512
# SLIDING_WINDOW=128
# TRUNC_LEN=128
# SLIDING_WINDOW=128
# SLIDING_WINDOW=64
# TRUNC_LEN=32
# SLIDING_WINDOW=32

# ALPHA=2
# BETA=-2
ALPHA=1
BETA=-1
# KEY_TOKEN_PATH=key_token/${MODEL_NAME}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}min_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
KEY_TOKEN_PATH=key_token/Qwen2.5-72B-Instruct_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}min_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta

# SENSITIVITY_RESULT_PATH=csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_${DATASET}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}minseq_${LOSS_FUNC}/loss
# SENSITIVITY_RESULT_PATH=csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_2048seqlen_0minseq_${LOSS_FUNC}/loss
SENSITIVITY_RESULT_PATH=csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_2048seqlen_0minseq_jsd/loss
# SENSITIVITY_RESULT_PATH=csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_${SEQLEN}seqlen_${MIN_SEQLEN}minseq_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta_${LOSS_FUNC}/loss
# SENSITIVITY_RESULT_PATH=csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24kv24bits_w128kv128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_${DATASET}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}minseq_${LOSS_FUNC}/loss


# if [ ${USE_KEY_TOKEN} == 'True' ]; then
#     SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${DATA_BATCH_SIZE}bs_${N_SAMPLE}sample_${SEQLEN}seq_${MIN_SEQLEN}min_${N_TOKEN}token_${PREDICTOR}_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
# else
#     SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${DATA_BATCH_SIZE}bs_${N_SAMPLE}sample_${SEQLEN}seq_${MIN_SEQLEN}min_${N_TOKEN}token_${PREDICTOR} 
if [ ${USE_KEY_TOKEN} == 'True' ]; then
    SAVE=save/search/second/${TODAY}_${MODEL_NAME}_${METRIC}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_${LOSS_FUNC}_${DATASET}_${DATA_BATCH_SIZE}bs_${N_SAMPLE}sample_${SEQLEN}seq_${MIN_SEQLEN}min_${N_TOKEN}token_${PREDICTOR}_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
else
    SAVE=save/search/second/${TODAY}_${MODEL_NAME}_${METRIC}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_${LOSS_FUNC}_${DATASET}_${DATA_BATCH_SIZE}bs_${N_SAMPLE}sample_${SEQLEN}seq_${MIN_SEQLEN}min_${N_TOKEN}token_${PREDICTOR} 
fi

N_PROC=1

ARGS="--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--quant_model_paths ${QMODEL_PATHS} \
--w_method ${W_METHOD} \
--kv_method ${KV_METHOD} \
--w_bits ${W_BITS} \
--k_bits ${KV_BITS} \
--v_bits ${KV_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--w_pareto_path ${W_PARETO_PATH} \
--kv_pareto_path ${KV_PARETO_PATH} \
--n_token ${N_TOKEN} \
--residual_length ${RESIDUAL_LENGTH} \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
--quant_kv_output \
--sensitivity_result_path ${SENSITIVITY_RESULT_PATH} \
--predictor ${PREDICTOR} \
--save ${SAVE} \
--iterations ${ITER} \
--n_doe ${N_DOE} \
--n_iter ${N_ITER} \
--metric ${METRIC} \
--config ${CONFIG} \
--debug \
--max_value ${MAX_VALUE} \
--loss_func ${LOSS_FUNC} \
--n_sample ${N_SAMPLE} \
--data_batch_size ${DATA_BATCH_SIZE} \
--seqlen ${SEQLEN} \
--min_seqlen ${MIN_SEQLEN} \
--dataset ${DATASET} \
--save_iter ${SAVE_ITER}"
# --ga_pop_size ${GA_POP_SIZE} \
# --mut_prob ${MUT_PROB} \
# --crossover_prob ${CROSSOVER_PROB} \

if [ ${USE_KEY_TOKEN} == 'True' ]; then
    ARGS+=" --use_key_token \
    --trunc_len ${TRUNC_LEN} \
    --sliding_window ${SLIDING_WINDOW} \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --key_token_path ${KEY_TOKEN_PATH}"
fi

for g in "${KV_GROUP_SIZE[@]}"
do
    ARGS+=" --k_group_size ${g} "
done

for g in "${KV_GROUP_SIZE[@]}"
do
    ARGS+="--v_group_size ${g} "
done

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} second_search.py \
${ARGS}
