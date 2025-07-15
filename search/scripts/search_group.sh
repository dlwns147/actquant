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

# METHOD="hqq layer_prune"
# METHOD_TEXT="hqq_layer_prune"
METHOD="hqq"
METHOD_TEXT="hqq"

# W_BITS="2 3 4"
# W_BITS_TEXT="234"
# W_BITS="4"
# W_BITS_TEXT="4"
W_BITS=16
W_BITS_TEXT=16
AXIS=1
W_GROUP_SIZE=128
QSCALE=false
QZERO=false

# K_BITS="2 4"
# K_BITS_TEXT="24"
# K_BITS=4
# K_BITS_TEXT=4
# K_GROUP_SIZE=128
# K_GROUP_SIZE=("32 64 128" "128")
# K_GROUP_SIZE_TEXT=3264128128
K_BITS="2 3 4"
K_BITS_TEXT="234"
K_GROUP_SIZE=("128" "128" "128")
K_GROUP_SIZE_TEXT=128128128

# V_BITS="2 4"
# V_BITS_TEXT="24"
# V_BITS=4
# V_BITS_TEXT=4
# V_GROUP_SIZE=128
# V_GROUP_SIZE=("32 64 128" "128")
# V_GROUP_SIZE_TEXT=3264128128
V_BITS="2 3 4"
V_BITS_TEXT="234"
V_GROUP_SIZE=("128" "128" "128")
V_GROUP_SIZE_TEXT=128128128

K_QUANT_PER=channel
V_QUANT_PER=token

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

COMP_OBJ=kvbits
COMP_OBJ_TEXT=kv

COMP_OBJ_MIN=${K_BITS:0:1}
COMP_OBJ_MIN_TEXT=${K_BITS:0:1}

COMP_OBJ_MAX=5
COMP_OBJ_MAX_TEXT=5

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

# LOSS_FUNC=cross_entropy
LOSS_FUNC=jsd

# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_hqq_w24k24v24bits_w${W_GROUP_SIZE}k${K_GROUP_SIZE_TEXT}v${V_GROUP_SIZE_TEXT}group_size_1axis_k_${K_QUANT_PER}_v_${V_QUANT_PER}_wikitext2_128sample_${LOSS_FUNC}/loss
# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_hqq_w24k24v24bits_w${W_GROUP_SIZE}k128v128group_size_1axis_k_${K_QUANT_PER}_v_${V_QUANT_PER}_wikitext2_128sample_${LOSS_FUNC}/loss
SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_hqq_w24k24v24bits_w${W_GROUP_SIZE}k128v128group_size_1axis_k_${K_QUANT_PER}_v_${V_QUANT_PER}_wikitext2_128sample_jsd/loss


# PREDICTOR=mlp
PREDICTOR=rbf
# PREDICTOR=gp

DATASET=wikitext2
# DATASET=c4
# DATASET=gsm8k

# N_SAMPLE=8
# N_SAMPLE=16
N_SAMPLE=32
# N_SAMPLE=64
# N_SAMPLE=128

DATA_BATCH_SIZE=1
# DATA_BATCH_SIZE=128

# N_DOE=100
# N_DOE=300

# N_DOE=250
# ITER=200
# N_ITER=50

# N_DOE=150
# ITER=100
# N_ITER=50

N_DOE=100
ITER=100
N_ITER=30

# N_DOE=400
# ITER=200

GA_POP_SIZE=200
# GA_POP_SIZE=100

METRIC=loss
# METRIC=gsm8k
# METRIC=gsm8k_cot

MAX_VALUE=5
# MAX_VALUE=1
MUT_PROB=0.1
CROSSOVER_PROB=0.9

SAVE_ITER=1
# SAVE_ITER=10

# LM_EVAL_BATCH_SIZE=4
# LM_EVAL_BATCH_SIZE=16
LM_EVAL_BATCH_SIZE=32

# NUM_FEWSHOT=0
NUM_FEWSHOT=5

# LIMIT=40
# LIMIT=50
LIMIT=100

# VERBOSITY='FATAL'
VERBOSITY='INFO'

SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}k${K_BITS_TEXT}v${V_BITS_TEXT}bits_w${W_GROUP_SIZE}k${K_GROUP_SIZE_TEXT}v${V_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_PER}_v_${V_QUANT_PER}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_${PREDICTOR}
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}k${K_BITS_TEXT}v${V_BITS_TEXT}bits_w${W_GROUP_SIZE}k${K_GROUP_SIZE_TEXT}v${V_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_PER}_v_${V_QUANT_PER}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_${PREDICTOR}_${LM_EVAL_BATCH_SIZE}bs_${LIMIT}limit_${NUM_FEWSHOT}few
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}k${K_BITS_TEXT}v${V_BITS_TEXT}bits_w${W_GROUP_SIZE}k${K_GROUP_SIZE}v${V_GROUP_SIZE}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_PER}_v_${V_QUANT_PER}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_${PREDICTOR}_${OUTLIER_TEXT}

# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_w${Q_BITS_TEXT}k${K_BITS}v${V_BITS}bits_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_${PREDICTOR}
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_${PREDICTOR}_outlier_${OUTLIER_TEXT}
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${N_SAMPLE}sample_${PREDICTOR}_outlier_${OUTLIER_TEXT}_mixed
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_lp_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${DATASET}_${N_SAMPLE}sample_2_64
# SAVE=save/search/quant/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_lp_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_${DATASET}_${N_SAMPLE}sample
# SAVE=save/search/${TODAY}_${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD_TEXT}_iter_${ITER}_${GA_ALGORITHM}_${Q_BITS_TEXT}_obj_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}_mut_${MUT_PROB}_layer_prune_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}_linear_group

N_PROC=1

ARGS="--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--quant_model_paths ${QMODEL_PATHS} \
--w_bits ${W_BITS} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--comp_obj ${COMP_OBJ} \
--comp_obj_min ${COMP_OBJ_MIN} \
--comp_obj_max ${COMP_OBJ_MAX} \
--sensitivity_result_path ${SENSITIVITY_RESULT_PATH} \
--residual_length ${RESIDUAL_LENGTH} \
--k_quant_per ${K_QUANT_PER} \
--v_quant_per ${V_QUANT_PER} \
--use_flash \
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
--dataset ${DATASET} \
--save_iter ${SAVE_ITER} \
--limit ${LIMIT} \
--lm_eval_batch_size ${LM_EVAL_BATCH_SIZE} \
--verbosity ${VERBOSITY} \
--num_fewshot ${NUM_FEWSHOT}"
for g in "${K_GROUP_SIZE[@]}"
do
    ARGS+=" --k_group_size ${g} "
done

for g in "${V_GROUP_SIZE[@]}"
do
    ARGS+="--v_group_size ${g} "
done

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search_group.py \
${ARGS}

# --quant_kv_output \
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
# --k_quant_per ${K_QUANT_PER} \
# --v_quant_per ${V_QUANT_PER} \
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

# --method ${METHOD} \

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
