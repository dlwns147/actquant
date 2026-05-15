TODAY=`date +%y%m%d%H%M`

# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-3.1-8B-Instruct
# DTYPE=float16
# CONFIG=config/llama.json

MODEL_PATH=/SSD/huggingface/Qwen
MODEL_NAME=Qwen2.5-7B-Instruct
# MODEL_NAME=Qwen2.5-14B-Instruct
# DTYPE=bfloat16
DTYPE=float16
CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/mistral.json

RESIDUAL_LENGTH=128

# N_TOKEN=1024
# N_TOKEN=1048576

N_TOKEN_LIST=(16384 32768 65536 131072)

W_BITS_LIST=(3 3 4)
W_GROUP_SIZE_LIST=(-1 128 128)

KV_BITS_LIST=(2 4)
KV_GROUP_SIZE_LIST=(128)

KV_DIM_LIST=(48 0)

SAVE=csv/mem
CSV_FILE=${TODAY}_${MODEL_NAME}.csv

python compute_mem.py \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --config ${CONFIG} \
    --w_bits ${W_BITS_LIST[@]} \
    --w_group_size ${W_GROUP_SIZE_LIST[@]} \
    --residual_length ${RESIDUAL_LENGTH} \
    --kv_bits ${KV_BITS_LIST[@]} \
    --kv_group_size ${KV_GROUP_SIZE_LIST[@]} \
    --kv_dim ${KV_DIM_LIST[@]} \
    --n_token ${N_TOKEN_LIST[@]} \
    --save ${SAVE} \
    --csv_file ${CSV_FILE}