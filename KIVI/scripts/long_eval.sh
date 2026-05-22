
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

# MODEL=Llama-3.1-8B-Instruct_131072_2bits_group128_residual128
MODEL=Llama-3.1-8B-Instruct_131072_4bits_group128_residual128

python eval_long_bench.py \
--model ${MODEL} \
# --e


# --tasks ${TASKS}