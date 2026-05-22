DEVICES=${1}

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-3.1-8B

# BATCH_SIZE=1
# BATCH_SIZE=2
# BATCH_SIZE=4
BATCH_SIZE=8
# BATCH_SIZE=16
# BATCH_SIZE=32
# BATCH_SIZE=64

PROMPT_LEN=64
# PROMPT_LEN=2048
# PROMPT_LEN=4096
# PROMPT_LEN=131072

# GEN_LEN=0
# GEN_LEN=4
GEN_LEN=16
# GEN_LEN=128
# GEN_LEN=256


# ATTN=sdpa
ATTN=flash_attention_2

# MODEL_TYPE=int4
# MODEL_TYPE=fp16
MODEL_TYPE=hf

OUTPUT_PATH=memory
# FILE_NAME=${MODEL_NAME}_${MODEL_TYPE}_bs${BATCH_SIZE}_pl${PROMPT_LEN}_gl${GEN_LEN}_attn${ATTN}

CUDA_VISIBLE_DEVICES=${DEVICES} python e2e/memory.py \
--model_config ${MODEL_PATH}/${MODEL_NAME} \
--batch_size ${BATCH_SIZE} \
--prefill_seq_len ${PROMPT_LEN} \
--decode_steps ${GEN_LEN} \
--attn_implementation ${ATTN} \
--model_type ${MODEL_TYPE} \
--output_path ${OUTPUT_PATH}


