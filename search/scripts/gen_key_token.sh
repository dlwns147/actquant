DEVICES=${1}
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-3.1-8B-Instruct
# CONFIG=config/llama.json
# DTYPE=float16
# DTYPE=bfloat16

MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# MODEL_NAME=Qwen2.5-14B-Instruct
MODEL_NAME=Qwen2.5-72B-Instruct
CONFIG=config/qwen2.json
DTYPE=float16
# DTYPE=bfloat16

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# CONFIG=config/mistral.json
# DTYPE=float16
# # # DTYPE=bfloat16

# DATASET=wikitext2
# # DATASET=c4
# N_SAMPLE=128
# SEQLEN=2048
# DATA_BATCH_SIZE=1
# MIN_SEQLEN=0

DATASET=gov_report
N_SAMPLE=4
# N_SAMPLE=8
# SEQLEN=2048
SEQLEN=8192
DATA_BATCH_SIZE=1
MIN_SEQLEN=8192
# SEQLEN=16384
# DATA_BATCH_SIZE=1
# MIN_SEQLEN=16384

# DATASET=gsm8k
# N_SAMPLE=32
# SEQLEN=256
# # DATA_BATCH_SIZE=1
# DATA_BATCH_SIZE=8
# MIN_SEQLEN=0
# # MIN_SEQLEN=192

# TRUNC_LEN=4096
# SLIDING_WINDOW=1024
# TRUNC_LEN=1024
# SLIDING_WINDOW=256
TRUNC_LEN=512
SLIDING_WINDOW=128
# TRUNC_LEN=128
# SLIDING_WINDOW=32
# TRUNC_LEN=64
# SLIDING_WINDOW=32
# TRUNC_LEN=32
# SLIDING_WINDOW=32

# ALPHA=2
# BETA=-2
ALPHA=1
BETA=-1

KEY_TOKEN_SAVE_PATH=key_token/${MODEL_NAME}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}min_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
# KEY_TOKEN_LOAD_PATH=key_token/${MODEL_NAME}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}min_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta
# KEY_TOKEN_LOAD_PATH=key_token/Qwen2.5-72B-Instruct_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}min_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta

CUDA_VISIBLE_DEVICES=${DEVICES} python gen_key_token.py \
    --gpu_id ${DEVICES} \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --config ${CONFIG} \
    --dataset ${DATASET} \
    --n_sample ${N_SAMPLE} \
    --seqlen ${SEQLEN} \
    --data_batch_size ${DATA_BATCH_SIZE} \
    --min_seqlen ${MIN_SEQLEN} \
    --trunc_len ${TRUNC_LEN} \
    --sliding_window ${SLIDING_WINDOW} \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --save_path ${KEY_TOKEN_SAVE_PATH} \
    --verbosity