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

# W_METHOD="hqq layer_prune"
# W_METHOD_TEXT="hqq_layer_prune"
W_METHOD="hqq"
W_METHOD_TEXT="hqq"
# W_BITS="4"
# W_BITS_TEXT="4"
# W_GROUP_SIZE=128
W_BITS="2 3 4"
W_BITS_TEXT="234"
W_GROUP_SIZE=128

# W_METHOD="fp16"
# W_METHOD_TEXT="fp16"
# W_BITS=16
# W_BITS_TEXT=16
# W_GROUP_SIZE=-1

AXIS=1
QSCALE=false
QZERO=false


# KV_METHOD="hqq"
KV_METHOD="kivi"
# KV_BITS=4
# KV_BITS_TEXT=4
# KV_GROUP_SIZE=("128")
# KV_GROUP_SIZE_TEXT=128
# KV_BITS="2 4"
# KV_BITS_TEXT="24"
# KV_GROUP_SIZE=("32 64 128" "32 64 128")
# KV_GROUP_SIZE_TEXT=3264128x2
# KV_BITS="2 4 8"
# KV_BITS_TEXT="248"
# KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
# KV_GROUP_SIZE_TEXT=3264128x3
KV_BITS="2 3 4"
KV_BITS_TEXT="234"
# # # KV_GROUP_SIZE=("128" "128" "128")
# # # KV_GROUP_SIZE_TEXT=128x3
# # KV_GROUP_SIZE=("64 128" "64 128" "64 128")
# # KV_GROUP_SIZE_TEXT=64128x3
KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
KV_GROUP_SIZE_TEXT=3264128x3
# KV_BITS="4 8"
# KV_BITS_TEXT="48"
# KV_GROUP_SIZE=("32" "32")
# KV_GROUP_SIZE_TEXT=32x2

K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token

# RESIDUAL_LENGTH=0
RESIDUAL_LENGTH=128

# COMP_OBJ="wbits kvbits"
# COMP_OBJ_TEXT=wkv
# COMP_OBJ_MIN="${W_BITS:0:1} ${K_BITS:0:1}"
# COMP_OBJ_MIN_TEXT=${W_BITS:0:1}${K_BITS:0:1}
# COMP_OBJ_MAX="5 5"
# COMP_OBJ_MAX_TEXT=55

# COMP_OBJ=wbits
# COMP_OBJ_TEXT=w
# COMP_OBJ_MIN=${W_BITS:0:1}
# COMP_OBJ_MIN_TEXT=${W_BITS:0:1}
# COMP_OBJ_MAX=5
# COMP_OBJ_MAX_TEXT=5

# COMP_OBJ=kvbits
# COMP_OBJ_TEXT=kv
# COMP_OBJ_MIN=${KV_BITS:0:1}
# COMP_OBJ_MIN_TEXT=${KV_BITS:0:1}
# COMP_OBJ_MAX=5
# COMP_OBJ_MAX_TEXT=5

# N_TOKEN=0

COMP_OBJ=memory
COMP_OBJ_TEXT=memory
COMP_OBJ_MIN=1
COMP_OBJ_MIN_TEXT=1
COMP_OBJ_MAX=1e99
COMP_OBJ_MAX_TEXT=1e99

# N_TOKEN=1024
# N_TOKEN=16384
N_TOKEN=131072
# N_TOKEN=1048576

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
# PREDICTOR=gp

# DATASET=wikitext2
# # DATASET=c4
# # N_SAMPLE=32
# N_SAMPLE=128
# # SEQLEN=2048
# # N_SAMPLE=256
# SEQLEN=4096
# # SEQLEN=8192
# DATA_BATCH_SIZE=1
# MIN_SEQLEN=0

DATASET=gov_report
N_SAMPLE=4
# N_SAMPLE=8
# SEQLEN=2048
# SEQLEN=8192
# MIN_SEQLEN=8192
SEQLEN=16384
MIN_SEQLEN=16384
DATA_BATCH_SIZE=1
# MIN_SEQLEN=0

# DATASET=gsm8k
# N_SAMPLE=32
# SEQLEN=256
# DATA_BATCH_SIZE=8
# MIN_SEQLEN=0
# # MIN_SEQLEN=192

N_DOE=600
# N_DOE=500
ITER=200
N_ITER=50

# N_DOE=200
# ITER=100
# N_ITER=30

GA_POP_SIZE=200
# GA_POP_SIZE=100

METRIC=loss
# METRIC=gsm8k
# METRIC=gsm8k_cot

# MAX_VALUE=5
MAX_VALUE=0.7
# MAX_VALUE=1
MUT_PROB=0.1
CROSSOVER_PROB=0.9

SAVE_ITER=1
# SAVE_ITER=10

# TRUNC_LEN=4096
SLIDING_WINDOW=1024
TRUNC_LEN=1024
# SLIDING_WINDOW=256
# TRUNC_LEN=512
# SLIDING_WINDOW=128
# TRUNC_LEN=128
# SLIDING_WINDOW=32
# TRUNC_LEN=64
# SLIDING_WINDOW=32
# TRUNC_LEN=32
# SLIDING_WINDOW=32

# ALPHA=2
# BETA=-2
ALPHA=1
BETA=-1

# KEY_TOKEN_LOAD_PATH=key_token/${MODEL_NAME}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}min_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
# KEY_TOKEN_LOAD_PATH=key_token/Qwen2.5-72B-Instruct_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}min_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
KEY_TOKEN_PATH=key_token/Qwen2.5-72B-Instruct_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}min_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
# KEY_TOKEN_PATH=/NAS/SJ/actquant/search/key_token/Qwen2.5-72B-Instruct_8sample_16384seqlen_16384min_1024trunc_1024sw_1alpha_-1beta

# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_hqq_w24k24v24bits_w${W_GROUP_SIZE}k${K_GROUP_SIZE_TEXT}v${V_GROUP_SIZE_TEXT}group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_${LOSS_FUNC}/loss
# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_hqq_w24k24v24bits_w${W_GROUP_SIZE}k128v128group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_${LOSS_FUNC}/loss
# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_hqq_w24k24v24bits_w${W_GROUP_SIZE}k128v128group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_jsd/loss
# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_hqq_w24k24v24bits_w${W_GROUP_SIZE}k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_gsm8k_32sample_256seqlen_0minseq_jsd/loss
# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_${DATASET}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}minseq_${LOSS_FUNC}/loss
# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_2048seqlen_0minseq_${LOSS_FUNC}/loss
SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_2048seqlen_0minseq_jsd/loss
# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_${SEQLEN}seqlen_${MIN_SEQLEN}minseq_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta_${LOSS_FUNC}/loss
# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_w${W_BITS_TEXT}k24v24bits_w${W_GROUP_SIZE}k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_${SEQLEN}seqlen_${MIN_SEQLEN}minseq_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta_${LOSS_FUNC}/loss


# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${DATA_BATCH_SIZE}bs_${N_SAMPLE}sample_${SEQLEN}seq_${MIN_SEQLEN}min_${N_TOKEN}token_${PREDICTOR} 
SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${DATA_BATCH_SIZE}bs_${N_SAMPLE}sample_${SEQLEN}seq_${MIN_SEQLEN}min_${N_TOKEN}token_${PREDICTOR}_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_${W_METHOD_TEXT}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}k${K_BITS_TEXT}v${V_BITS_TEXT}bits_w${W_GROUP_SIZE}k${K_GROUP_SIZE}v${KV_GROUP_SIZE}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_${PREDICTOR}_${OUTLIER_TEXT}

# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${W_METHOD_TEXT}_iter_${ITER}_w${Q_BITS_TEXT}k${K_BITS}v${V_BITS}bits_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_${PREDICTOR}
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${W_METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_${PREDICTOR}_outlier_${OUTLIER_TEXT}
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${W_METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_${PREDICTOR}_outlier_${OUTLIER_TEXT}_mixed
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${W_METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_lp_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${DATASET}_${N_SAMPLE}sample_2_64
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${W_METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_lp_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${DATASET}_${N_SAMPLE}sample
# SAVE=save/search/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${W_METHOD_TEXT}_iter_${ITER}_${GA_ALGORITHM}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_mut_${MUT_PROB}_layer_prune_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_linear_group

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
--comp_obj ${COMP_OBJ} \
--comp_obj_min ${COMP_OBJ_MIN} \
--comp_obj_max ${COMP_OBJ_MAX} \
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
--ga_pop_size ${GA_POP_SIZE} \
--config ${CONFIG} \
--debug \
--max_value ${MAX_VALUE} \
--mut_prob ${MUT_PROB} \
--crossover_prob ${CROSSOVER_PROB} \
--loss_func ${LOSS_FUNC} \
--n_sample ${N_SAMPLE} \
--data_batch_size ${DATA_BATCH_SIZE} \
--seqlen ${SEQLEN} \
--min_seqlen ${MIN_SEQLEN} \
--dataset ${DATASET} \
--save_iter ${SAVE_ITER} \
--use_key_token \
--trunc_len ${TRUNC_LEN} \
--sliding_window ${SLIDING_WINDOW} \
--alpha ${ALPHA} \
--beta ${BETA} \
--key_token_path ${KEY_TOKEN_PATH}"








for g in "${KV_GROUP_SIZE[@]}"
do
    ARGS+=" --k_group_size ${g} "
done

for g in "${KV_GROUP_SIZE[@]}"
do
    ARGS+="--v_group_size ${g} "
done

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search.py \
${ARGS}

# # LM_EVAL_BATCH_SIZE=4
# # LM_EVAL_BATCH_SIZE=16
# LM_EVAL_BATCH_SIZE=32

# # NUM_FEWSHOT=0
# NUM_FEWSHOT=5

# # LIMIT=40
# # LIMIT=50
# LIMIT=100

# # VERBOSITY='FATAL'
# VERBOSITY='INFO'


# --limit ${LIMIT} \
# --lm_eval_batch_size ${LM_EVAL_BATCH_SIZE} \
# --verbosity ${VERBOSITY} \
# --num_fewshot ${NUM_FEWSHOT} \

# --gpu_id ${DEVICES} \
# --model_path ${MODEL_PATH} \
# --model_name ${MODEL_NAME} \
# --quant_model_paths ${QMODEL_PATHS} \
# --w_bits ${W_BITS} \
# --k_bits ${K_BITS} \
# --v_bits ${V_BITS} \
# --w_group_size ${W_GROUP_SIZE} \
# --k_group_size ${K_GROUP_SIZE[0]} \
# --k_group_size ${K_GROUP_SIZE[1]} \
# --v_group_size ${V_GROUP_SIZE[0]} \
# --v_group_size ${V_GROUP_SIZE[1]} \
# --comp_obj ${COMP_OBJ} \
# --comp_obj_min ${COMP_OBJ_MIN} \
# --comp_obj_max ${COMP_OBJ_MAX} \
# --sensitivity_result_path ${SENSITIVITY_RESULT_PATH} \
# --residual_length ${RESIDUAL_LENGTH} \
# --quant_kv_output \
# --k_quant_scheme ${K_QUANT_SCHEME} \
# --v_quant_scheme ${V_QUANT_SCHEME} \
# --use_flash \
# --predictor ${PREDICTOR} \
# --save ${SAVE} \
# --iterations ${ITER} \
# --n_doe ${N_DOE} \
# --n_iter ${N_ITER} \
# --metric ${METRIC} \
# --ga_pop_size ${GA_POP_SIZE} \
# --config ${CONFIG} \
# --debug \
# --max_value ${MAX_VALUE} \
# --mut_prob ${MUT_PROB} \
# --crossover_prob ${CROSSOVER_PROB} \
# --loss_func ${LOSS_FUNC} \
# --n_sample ${N_SAMPLE} \
# --dataset ${DATASET} \
# --save_iter ${SAVE_ITER} \


# --pass_w_linear ${PASS_W_LINEAR} \
# --pass_k_layer ${PASS_K_LAYER} \
# --pass_v_layer ${PASS_V_LAYER} \

# --only_outlier_bits
# --base_outlier_bits ${OUTLIER_BITS} \
# --outlier_path ${OUTLIER_PATH} \
# --n_outlier ${N_OUTLIER}
# --layer_prune_range ${LAYER_PRUNE_RANGE_SMALL} ${LAYER_PRUNE_RANGE_LARGE} \
# --base_outlier_bits ${OUTLIER_BITS} \
# --outlier_path ${OUTLIER_PATH} \
# --n_outlier ${N_OUTLIER}
# --latency_table_file ${LATENCY_TABLE}

# --base_outlier_bits ${OUTLIER_BITS} \
# --outlier_path ${OUTLIER_PATH} \
# --n_outlier ${N_OUTLIER} \
# --only_outlier_bits

# --use_linear_group
# --resume ${RESUME} \



# COMP_OBJ="wbits kbits vbits"
# COMP_OBJ_TEXT=wkv

# COMP_OBJ_MIN="${W_BITS:0:1} ${K_BITS:0:1} ${V_BITS:0:1}"
# COMP_OBJ_MIN_TEXT="${W_BITS:0:1}${K_BITS:0:1}${V_BITS:0:1}"

# COMP_OBJ_MAX="${W_BITS:(-1)} ${K_BITS:(-1)} ${V_BITS:(-1)}"
# COMP_OBJ_MAX_TEXT=${W_BITS:(-1)}${K_BITS:(-1)}${V_BITS:(-1)}

# COMP_OBJ_MAX="5 5 5"
# COMP_OBJ_MAX_TEXT="555"

# --iqr_threshold ${IQR_THRESHOLD} \


# V_BITS=4
# V_BITS_TEXT=4
# V_GROUP_SIZE=("128")
# V_GROUP_SIZE_TEXT=128
# V_BITS="2 4"
# V_BITS_TEXT="24"
# V_GROUP_SIZE=("32 64 128" "32 64 128")
# V_GROUP_SIZE_TEXT=3264128x2
# V_BITS="2 4 8"
# V_BITS_TEXT="248"
# V_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
# V_GROUP_SIZE_TEXT=3264128x3
# V_BITS="2 3 4"
# V_BITS_TEXT="234"
# V_GROUP_SIZE=("128" "128" "128")
# # V_GROUP_SIZE_TEXT=128x3
# V_GROUP_SIZE=("64 128" "64 128" "64 128")
# V_GROUP_SIZE_TEXT=64128x3
# V_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
# V_GROUP_SIZE_TEXT=3264128x3