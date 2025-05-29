DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-7b-chat-hf
# MODEL_NAME=Llama-2-13b-chat-hf
# MODEL_NAME=Llama-3.1-8B
MODEL_NAME=Llama-3.1-8B-Instruct


# K_BITS=4
# V_BITS=4

# K_BITS=2
# V_BITS=4

# K_BITS=4
# V_BITS=2

K_BITS=2
V_BITS=2

GROUP_SIZE=128
RESIDUAL_LENGTH=128
# RESIDUAL_LENGTH=0

# GROUP_SIZE=32
# RESIDUAL_LENGTH=32


CUDA_VISIBLE_DEVICES=${DEVICES} python pred_long_bench.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--group_size ${GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH}
# --e


# --tasks ${TASKS}