DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-3.1-8B-Instruct


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

DATASET="wikitext2 c4"

K_QUANT_PER=channel
# K_QUANT_PER=token

# V_QUANT_PER=channel
V_QUANT_PER=token

BATCH_SIZE=1
# BATCH_SIZE=4
# BATCH_SIZE=16
# BATCH_SIZE=32
# BATCH_SIZE=64

# TASKS="piqa winogrande hellaswag arc_challenge arc_easy"
# TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq social_iqa openbookqa"
TASKS="gsm8k truthfulqa coqa"

CUDA_VISIBLE_DEVICES=${DEVICES} python eval.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--group_size ${GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--datasets ${DATASET} \
--zeroshot \
--zeroshot_batch_size ${BATCH_SIZE} \
--tasks ${TASKS} \
--use_flash

# --eval_ppl \
# --quant_kv_output \
# --k_quant_per ${K_QUANT_PER} \
# --v_quant_per ${V_QUANT_PER} \
