DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

Q_BITS="2 3 4"
Q_BITS_TEXT="234"

METHOD=awq
# METHOD=gptq

COMP_OBJ="bits"
COMP_OBJ_TEXT=bits


# TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq openbookqa social_iqa"
TASKS="coqa gsm8k truthfulqa"
# TASKS="coqa truthfulqa"

N=1
DATASETS="wikitext2 c4"

# GROUP_SIZE=128
# GROUP_SIZE=-1


# TARGET_COMP_OBJ=bits
# TARGET_BITS_LIST=(2 3 4)

W_BITS=16
W_GROUP_SIZE=128

K_BITS=2

V_BITS=2

# KV_GROUP_SIZE=32
# KV_GROUP_SIZE=64
KV_GROUP_SIZE=128


RESIDUAL_LENGTH=128
K_QUANT_PER=channel
V_QUANT_PER=token

# SAVE=save/result/${TODAY}_${MODEL_NAME}_${COMP_OBJ}_${METHOD}_${BITS}
SAVE=save/result/${TODAY}_test

N_PROC=1

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} awqgptq.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--w_bits ${W_BITS} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--k_group_size ${KV_GROUP_SIZE} \
--v_group_size ${KV_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--k_quant_per ${K_QUANT_PER} \
--v_quant_per ${V_QUANT_PER} \
--use_flash \
-n ${N} \
--save ${SAVE} \
--datasets ${DATASETS} \
--zeroshot \
--tasks ${TASKS} \
--clip_asym

# --method ${METHOD} \
# --group_size ${GROUP_SIZE} \

# --zeroshot_batch_size ${BATCH_SIZE} \