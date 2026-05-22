DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# MODEL_PATH=/SSD/huggingface/meta-llama
# # MODEL_NAME=Llama-2-7b-hf
# # MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-3.1-8B-Instruct

MODEL_PATH=/SSD/huggingface/Qwen
MODEL_NAME=Qwen2.5-7B
# MODEL_NAME=Qwen2.5-14B
# MODEL_NAME=Qwen2.5-32B
# MODEL_NAME=Qwen2.5-72B
# MODEL_NAME=Qwen2.5-7B-Instruct
# MODEL_NAME=Qwen2.5-14B-Instruct

# MODEL_PATH=/SSD/huggingface/mistralai
# # MODEL_NAME=Mistral-7B-v0.3
# MODEL_NAME=Mistral-7B-Instruct-v0.3

# K_BITS=4
# V_BITS=4

# K_BITS=2
# V_BITS=4

# K_BITS=4
# V_BITS=2

K_BITS=2
V_BITS=2

# GROUP_SIZE=128
# RESIDUAL_LENGTH=128
# RESIDUAL_LENGTH=0


GROUP_SIZE=32
RESIDUAL_LENGTH=32


CUDA_VISIBLE_DEVICES=${DEVICES} python long_context_example.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--group_size ${GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH}
# --use_flash
# --eval_ppl \

