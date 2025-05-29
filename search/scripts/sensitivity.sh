DEVICES=${1}
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json
DTYPE=float16

# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-3.1-8B
# MODEL_NAME=Llama-3.1-8B-Instruct
# DTYPE=bfloat16

# MODEL=facebook/opt-6.7b
# MODEL=facebook/opt-13b
# MODEL=facebook/opt-30b
# MODEL=facebook/opt-66b

METHOD="hqq"

W_BITS="2 4"
W_BITS_TEXT="24"
AXIS=1
W_GROUP_SIZE=128

K_BITS="2 4"
K_BITS_TEXT="24"
K_GROUP_SIZE=128

V_BITS="2 4"
V_BITS_TEXT="24"
V_GROUP_SIZE=128

QMODEL_PATHS_LIST=()
for B in ${W_BITS}
do
    # QMODEL_PATHS+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

DATASET=wikitext2
# DATASET=c4

# LOSS_FUNC=cross_entropy
LOSS_FUNC=jsd

N_SAMPLE=128

RESIDUAL_LENGTH=0
K_QUANT_PER=channel
# K_QUANT_PER=token
V_QUANT_PER=token

RESULT_PATH=csv/sensitivity/${MODEL_NAME}_${METHOD}_w${W_BITS_TEXT}k${K_BITS_TEXT}v${V_BITS_TEXT}bits_w${W_GROUP_SIZE}k${K_GROUP_SIZE}v${V_GROUP_SIZE}group_size_${AXIS}axis_k_${K_QUANT_PER}_v_${V_QUANT_PER}_${DATASET}_${N_SAMPLE}sample_${LOSS_FUNC}

TARGET="w k v"
# TARGET="k v"

N_PROC=1
# CUDA_VISIBLE_DEVICES=${DEVICES} python sensitivity.py \
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} sensitivity.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--method ${METHOD} \
--target ${TARGET} \
--quant_model_paths ${QMODEL_PATHS} \
--w_bits ${W_BITS} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--k_group_size ${K_GROUP_SIZE} \
--v_group_size ${V_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--quant_kv_output \
--k_quant_per ${K_QUANT_PER} \
--v_quant_per ${V_QUANT_PER} \
--n_sample ${N_SAMPLE} \
--result_path ${RESULT_PATH} \
--config ${CONFIG} \
--loss_func ${LOSS_FUNC} \
--dataset ${DATASET} \
--use_flash
# --eval_ppl
# --outlier_bits ${OUTLIER_BITS} \
# --outlier_path ${OUTLIER_PATH} \
