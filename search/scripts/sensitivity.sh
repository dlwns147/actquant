DEVICES=${1}
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-3.1-8B-Instruct
CONFIG=config/llama.json
DTYPE=float16

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# # MODEL_NAME=Qwen2.5-14B-Instruct
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# # MODEL_NAME=Mistral-7B-v0.3
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/mistral.json

METHOD="hqq"

W_BITS="2 4"
W_BITS_TEXT="24"
AXIS=1
W_GROUP_SIZE=128
# W_GROUP_SIZE=("128" "128")

K_BITS="2 4"
K_BITS_TEXT="24"
K_GROUP_SIZE=("128" "128")
K_GROUP_SIZE_TEXT=128x2

V_BITS="2 4"
V_BITS_TEXT="24"
V_GROUP_SIZE=("128" "128")
V_GROUP_SIZE_TEXT=128x2

QMODEL_PATHS_LIST=()
for B in ${W_BITS}
do
    # QMODEL_PATHS+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

# LOSS_FUNC=cross_entropy
LOSS_FUNC=jsd

# DATASET=wikitext2
# # DATASET=c4
# N_SAMPLE=128
# SEQLEN=2048
# DATA_BATCH_SIZE=1

DATASET=gsm8k
N_SAMPLE=32
SEQLEN=256
DATA_BATCH_SIZE=8
MIN_SEQLEN=0
# MIN_SEQLEN=192

RESIDUAL_LENGTH=0
K_QUANT_PER=channel
# K_QUANT_PER=token
V_QUANT_PER=token

RESULT_PATH=csv/sensitivity/${MODEL_NAME}_${METHOD}_w${W_BITS_TEXT}k${K_BITS_TEXT}v${V_BITS_TEXT}bits_w${W_GROUP_SIZE}k${K_GROUP_SIZE_TEXT}v${V_GROUP_SIZE_TEXT}group_size_${AXIS}axis_k_${K_QUANT_PER}_v_${V_QUANT_PER}_${DATASET}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}minseq_${LOSS_FUNC}

TARGET="w k v"
# TARGET="k v"

N_PROC=1
ARGS="--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--method ${METHOD} \
--target ${TARGET} \
--quant_model_paths ${QMODEL_PATHS} \
--w_bits ${W_BITS} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--quant_kv_output \
--k_quant_per ${K_QUANT_PER} \
--v_quant_per ${V_QUANT_PER} \
--n_sample ${N_SAMPLE} \
--data_batch_size ${DATA_BATCH_SIZE} \
--seqlen ${SEQLEN} \
--min_seqlen ${MIN_SEQLEN} \
--result_path ${RESULT_PATH} \
--config ${CONFIG} \
--loss_func ${LOSS_FUNC} \
--dataset ${DATASET} \
--use_flash"

# --eval_ppl
# --outlier_bits ${OUTLIER_BITS} \
# --outlier_path ${OUTLIER_PATH} \

for g in "${K_GROUP_SIZE[@]}"
do
    ARGS+=" --k_group_size ${g} "
done

for g in "${V_GROUP_SIZE[@]}"
do
    ARGS+="--v_group_size ${g} "
done

# CUDA_VISIBLE_DEVICES=${DEVICES} python sensitivity.py \
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} sensitivity.py \
${ARGS}
