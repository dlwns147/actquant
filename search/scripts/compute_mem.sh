MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# # MODEL_NAME=Qwen2.5-7B-Instruct
# MODEL_NAME=Qwen2.5-14B-Instruct
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/mistral.json

# # W_BITS=2
# W_BITS=3
# # W_BITS=4
# W_GROUP_SIZE=128
# # W_GROUP_SIZE=-1


# # KV_BITS=2
# KV_BITS=4
# # KV_BITS=8

# K_BITS=2
# V_BITS=2
# K_BITS=4
# V_BITS=4
# K_BITS=8
# V_BITS=8

# KV_GROUP_SIZE=32
# KV_GROUP_SIZE=64
# KV_GROUP_SIZE=128

# K_GROUP_SIZE=32
# K_GROUP_SIZE=64
# K_GROUP_SIZE=128

# V_GROUP_SIZE=32
# V_GROUP_SIZE=64
# V_GROUP_SIZE=128

RESIDUAL_LENGTH=128

# N_TOKEN=1024
# N_TOKEN=1048576

# N_TOKEN_LIST=(1024 131072 1048576)
N_TOKEN_LIST=(16384)

# W_BITS_LIST=(3)
# W_GROUP_SIZE=128
W_BITS_LIST=(4 3)
W_GROUP_SIZE=-1

KV_BITS_LIST=(4 3 2)
KV_GROUP_SIZE=128

for W_BITS in ${W_BITS_LIST[@]}
do
    for KV_BITS in ${KV_BITS_LIST[@]}
    do
        for N_TOKEN in ${N_TOKEN_LIST[@]}
        do
            python compute_mem.py \
            --model_path ${MODEL_PATH} \
            --model_name ${MODEL_NAME} \
            --config ${CONFIG} \
            --w_bits ${W_BITS} \
            --w_group_size ${W_GROUP_SIZE} \
            --residual_length ${RESIDUAL_LENGTH} \
            --k_bits ${KV_BITS} \
            --v_bits ${KV_BITS} \
            --k_group_size ${KV_GROUP_SIZE} \
            --v_group_size ${KV_GROUP_SIZE} \
            --n_token ${N_TOKEN}
        done
    done
done