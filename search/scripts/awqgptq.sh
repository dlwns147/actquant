DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# MODEL_PATH=/SSD/huggingface/meta-llama
# # MODEL_NAME=Llama-2-7b-hf
# # MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-3.1-8B-Instruct
# DTYPE=float16
# CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# # MODEL_NAME=Qwen2.5-7B
# # MODEL_NAME=Qwen2.5-14B
# # MODEL_NAME=Qwen2.5-32B
# # MODEL_NAME=Qwen2.5-72B
# # MODEL_NAME=Qwen2.5-7B-Instruct
# # MODEL_NAME=Qwen2.5-14B-Instruct
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/qwen2.json

MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-v0.3
MODEL_NAME=Mistral-7B-Instruct-v0.3
# DTYPE=bfloat16
DTYPE=float16
CONFIG=config/mistral.json

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

KV_GROUP_SIZE=32
# KV_GROUP_SIZE=64
# KV_GROUP_SIZE=128


RESIDUAL_LENGTH=128
K_QUANT_PER=channel
V_QUANT_PER=token

# SAVE=save/result/${TODAY}_${MODEL_NAME}_${COMP_OBJ}_${METHOD}_${BITS}
SAVE=save/result/${TODAY}_test

LONG_BENCH_RESULT_PATH=save/long_bench/${TODAY}_${MODEL_NAME}_pure_${METHOD}_${COMP_OBJ_TEXT}_w${W_BITS}bits_w${W_GROUP_SIZE}gs_k${K_BITS}bits_k${KV_GROUP_SIZE}gs_${K_QUANT_PER}_v${V_BITS}bits_v${KV_GROUP_SIZE}gs_${V_QUANT_PER}_r${RESIDUAL_LENGTH}
LONG_BENCH_CONFIG=utils/long_bench_config

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
--long_bench \
--long_bench_result_path ${LONG_BENCH_RESULT_PATH} \
--long_bench_config ${LONG_BENCH_CONFIG} \
--clip_asym

# --datasets ${DATASETS} \
# --zeroshot \
# --tasks ${TASKS} \

# --method ${METHOD} \
# --group_size ${GROUP_SIZE} \

# --zeroshot_batch_size ${BATCH_SIZE} \
