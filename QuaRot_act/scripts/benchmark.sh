DEVICES=${1}

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf

# BATCH_SIZE=1
# BATCH_SIZE=2
BATCH_SIZE=4
# BATCH_SIZE=8
# BATCH_SIZE=32
# BATCH_SIZE=64

# PROMPT_LEN=64
PROMPT_LEN=4096

# GEN_LEN=64
# GEN_LEN=128
GEN_LEN=256
# GEN_LEN=0


# ATTN=sdpa
ATTN=flash_attention_2

NUM_WARMUP_STEPS=1
NUM_BENCH_STEPS=1

CUDA_VISIBLE_DEVICES=${DEVICES} python e2e/benchmark.py \
--model_config ${MODEL_PATH}/${MODEL_NAME} \
--batch_size ${BATCH_SIZE} \
--prefill_seq_len ${PROMPT_LEN} \
--decode_steps ${GEN_LEN} \
--num_warmup_steps ${NUM_WARMUP_STEPS} \
--num_bench_steps ${NUM_BENCH_STEPS} \
--attn_implementation ${ATTN} \
--use_hf
# --use_cuda_graph \

