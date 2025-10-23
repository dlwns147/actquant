DEVICES=${1}
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
CONFIG=config/llama.json
DTYPE=float16
# DTYPE=bfloat16

# MODEL_PATH=/SSD/huggingface/Qwen
# # MODEL_NAME=Qwen2.5-7B-Instruct
# MODEL_NAME=Qwen2.5-14B-Instruct
# CONFIG=config/qwen2.json
# DTYPE=float16
# # DTYPE=bfloat16

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# CONFIG=config/mistral.json
# DTYPE=float16
# # # DTYPE=bfloat16

W_METHOD="hqq"
W_METHOD_TEXT="hqq"
W_BITS="2 4"
W_BITS_TEXT="24"
W_GROUP_SIZE=128

# W_METHOD="fp16"
# W_METHOD_TEXT="fp16"
# W_BITS="16"
# W_BITS_TEXT="16"
# W_GROUP_SIZE=-1

AXIS=1
# W_GROUP_SIZE=("128" "128")

# KV_METHOD="hqq"
KV_METHOD="kivi"

KV_BITS="2 4"
KV_BITS_TEXT="24"
KV_GROUP_SIZE=("128" "128")
KV_GROUP_SIZE_TEXT=128x2

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
# MIN_SEQLEN=0

DATASET=gov_report
N_SAMPLE=4
SEQLEN=2048
# SEQLEN=8192
DATA_BATCH_SIZE=1
MIN_SEQLEN=0

# DATASET=gsm8k
# N_SAMPLE=32
# SEQLEN=256
# # DATA_BATCH_SIZE=1
# DATA_BATCH_SIZE=8
# MIN_SEQLEN=0
# # MIN_SEQLEN=192

RESIDUAL_LENGTH=0
K_QUANT_SCHEME=channel
# K_QUANT_SCHEME=token
V_QUANT_SCHEME=token

# TRUNC_LEN=512
TRUNC_LEN=256
# SLIDING_WINDOW=128
SLIDING_WINDOW=64

ALPHA=2
BETA=-2

RESULT_PATH=csv/sensitivity/${MODEL_NAME}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}group_size_${AXIS}axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_${DATASET}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}minseq_${LOSS_FUNC}
# RESULT_PATH=csv/sensitivity/${MODEL_NAME}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}group_size_${AXIS}axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_${DATASET}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}minseq_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta_${LOSS_FUNC}


TARGET="w k v"
# TARGET="k v"

ARGS="--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--target ${TARGET} \
--quant_model_paths ${QMODEL_PATHS} \
--w_method ${W_METHOD} \
--kv_method ${KV_METHOD} \
--w_bits ${W_BITS} \
--k_bits ${KV_BITS} \
--v_bits ${KV_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--quant_kv_output \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
--n_sample ${N_SAMPLE} \
--data_batch_size ${DATA_BATCH_SIZE} \
--seqlen ${SEQLEN} \
--min_seqlen ${MIN_SEQLEN} \
--result_path ${RESULT_PATH} \
--config ${CONFIG} \
--loss_func ${LOSS_FUNC} \
--dataset ${DATASET} \
--use_key_token \
--trunc_len ${TRUNC_LEN} \
--sliding_window ${SLIDING_WINDOW} \
--alpha ${ALPHA} \
--beta ${BETA}
"
#  \
# --use_key_token \
# --trunc_len ${TRUNC_LEN} \
# --sliding_window ${SLIDING_WINDOW} \
# --alpha ${ALPHA} \
# --beta ${BETA}

# --eval_ppl
# --outlier_bits ${OUTLIER_BITS} \
# --outlier_path ${OUTLIER_PATH} \

for g in "${KV_GROUP_SIZE[@]}"
do
    ARGS+=" --k_group_size ${g} "
done

for g in "${KV_GROUP_SIZE[@]}"
do
    ARGS+="--v_group_size ${g} "
done

N_PROC=1
# CUDA_VISIBLE_DEVICES=${DEVICES} python sensitivity.py \
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} sensitivity.py \
${ARGS}
