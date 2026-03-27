DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# DTYPE=float16
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# DTYPE=float16
# CONFIG=config/mistral.json


# COMP_OBJ=wbits
# COMP_OBJ=kvdim
# COMP_OBJ=kvbits
COMP_OBJ=eff_kvbits
# COMP_OBJ=memory

# USE_KEY_TOKEN=True
USE_KEY_TOKEN=False

# N_TOKEN=1024
N_TOKEN=16384
# N_TOKEN=32768

W_METHOD="hqq"
W_METHOD_TEXT="hqq"

AXIS=1
DTYPE=float16

KV_METHOD="kivi"
KV_METHOD_TEXT="kivi"
# KV_METHOD="hqq"
# KV_METHOD_TEXT="hqq"

K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token

RESIDUAL_LENGTH=128
# RESIDUAL_LENGTH=0

if [ ${COMP_OBJ} == 'wbits' ]; then
    W_BITS="2 3 4"
    W_BITS_TEXT="234"
    W_GROUP_SIZE=128

    KV_BITS="4"
    KV_BITS_TEXT="4"
    KV_GROUP_SIZE=("128")
    KV_GROUP_SIZE_TEXT=128
    
    K_PRUNING_DIM="0"
    # V_PRUNING_DIM="0 16 32 48 64"
    V_PRUNING_DIM="0"
    K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
    V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

    COMP_OBJ_TEXT=wbits
    COMP_OBJ_MIN=${W_BITS:0:1}
    COMP_OBJ_MIN_TEXT=${W_BITS:0:1}
    COMP_OBJ_MAX=5
    COMP_OBJ_MAX_TEXT=5
    N_TOKEN=0

    N_DOE=400
    ITER=200
    N_ITER=50

elif [ ${COMP_OBJ} == 'kvbits' ]; then
    W_BITS="4"
    W_BITS_TEXT="4"
    W_GROUP_SIZE=128

    KV_BITS="2 3 4"
    KV_BITS_TEXT="234"
    KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
    KV_GROUP_SIZE_TEXT=3264128x3

    K_PRUNING_DIM="0"
    # V_PRUNING_DIM="0 16 32 48 64"
    V_PRUNING_DIM="0"
    K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
    V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

    COMP_OBJ_TEXT=kvbits
    COMP_OBJ_MIN=1
    COMP_OBJ_MIN_TEXT=1
    COMP_OBJ_MAX=5
    COMP_OBJ_MAX_TEXT=5
    N_TOKEN=0

    N_DOE=400
    ITER=150
    N_ITER=30

elif [ ${COMP_OBJ} == 'kvdim' ]; then
    KV_METHOD="think"
    KV_METHOD_TEXT="think"
    W_BITS="4"
    W_BITS_TEXT="4"
    W_GROUP_SIZE=128

    KV_BITS="4"
    KV_BITS_TEXT="4"
    KV_GROUP_SIZE=("128")
    KV_GROUP_SIZE_TEXT=128
    
    K_PRUNING_DIM="0 16 32 48 64"
    # V_PRUNING_DIM="0 16 32 48 64"
    V_PRUNING_DIM="0"
    K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
    V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

    COMP_OBJ_TEXT=kvdim
    COMP_OBJ_MIN=0      # remained_dim >= 64  (prune <= 50% of head_dim=128)
    COMP_OBJ_MIN_TEXT=0
    COMP_OBJ_MAX=128     # remained_dim <= 128 (no pruning)
    COMP_OBJ_MAX_TEXT=128
    N_TOKEN=0

    N_DOE=200
    ITER=150
    N_ITER=30

elif [ ${COMP_OBJ} == 'eff_kvbits' ]; then
    W_BITS="4"
    W_BITS_TEXT="4"
    W_GROUP_SIZE=128

    KV_BITS="2 3 4"
    KV_BITS_TEXT="234"
    KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
    KV_GROUP_SIZE_TEXT=3264128x3

    K_PRUNING_DIM="0 16 32 48 64"
    V_PRUNING_DIM="0"
    K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
    V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

    COMP_OBJ_TEXT=eff_kvbits
    COMP_OBJ_MIN=0.1
    COMP_OBJ_MIN_TEXT=0.1
    COMP_OBJ_MAX=5
    COMP_OBJ_MAX_TEXT=5

    N_DOE=600
    ITER=200
    N_ITER=50

elif [ ${COMP_OBJ} == 'memory' ]; then
    W_BITS="2 3 4"
    W_BITS_TEXT="234"
    W_GROUP_SIZE=128

    KV_BITS="2 3 4"
    KV_BITS_TEXT="234"
    KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
    KV_GROUP_SIZE_TEXT=3264128x3
    
    K_PRUNING_DIM="0 16 32 48 64"
    # V_PRUNING_DIM="0 16 32 48 64"
    V_PRUNING_DIM="0"
    K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
    V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

    COMP_OBJ_TEXT=memory
    COMP_OBJ_MIN=1
    COMP_OBJ_MIN_TEXT=1
    COMP_OBJ_MAX=1e99
    COMP_OBJ_MAX_TEXT=1e99

    N_DOE=600
    ITER=200
    N_ITER=50

fi

QMODEL_PATHS_LIST=()
for B in ${W_BITS}
do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")


LOSS_FUNC=jsd
# LOSS_FUNC=cross_entropy

PREDICTOR=rbf
# PREDICTOR=mlp
# PREDICTOR=gp

DATASET=wikitext2
# N_SAMPLE=128
N_SAMPLE=32
SEQLEN=2048
DATA_BATCH_SIZE=1
MIN_SEQLEN=0

# STRIDE=0
# STRIDE=128
STRIDE=512
# STRIDE=1024

GA_POP_SIZE=200

METRIC=loss
# METRIC=ppl

if [ ${LOSS_FUNC} == 'cross_entropy' ]; then
    MAX_VALUE=5
elif [ ${LOSS_FUNC} == 'jsd' ]; then
    MAX_VALUE=0.7
fi


MUT_PROB=0.1
CROSSOVER_PROB=0.9
SAVE_ITER=10

# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_2048seqlen_0minseq_jsd/loss
SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_w_hqq_kv_kivi_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_2048seqlen_0minseq_jsd/loss

SAVE=save/search/think/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_kdim${K_PRUNING_DIM_TEXT}_vdim${V_PRUNING_DIM_TEXT}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${DATA_BATCH_SIZE}bs_${N_SAMPLE}sample_${SEQLEN}seq_${N_TOKEN}token_${PREDICTOR}_${STRIDE}stride

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
--k_pruning_dim ${K_PRUNING_DIM} \
--v_pruning_dim ${V_PRUNING_DIM} \
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
--save_iter ${SAVE_ITER}"

# --sensitivity_result_path ${SENSITIVITY_RESULT_PATH} \
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

if [ ${STRIDE} -gt 0 ]; then
    ARGS+=" --stride ${STRIDE} "
else
    ARGS+=" --quant_kv_output "
fi

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search_think.py \
${ARGS}

# --stride ${STRIDE}
# --quant_kv_output \