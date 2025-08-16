DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# # MODEL_NAME=Qwen2.5-14B-Instruct
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/mistral.json

# W_METHOD="hqq layer_prune"
# W_METHOD_TEXT="hqq_layer_prune"

# W_METHOD=hqq
# W_METHOD_TEXT=hqq
W_METHOD=awq
W_METHOD_TEXT=awq
# W_METHOD="awq layer_prune"
# W_METHOD_TEXT=awq_layer_prune
# W_METHOD=fp16
# W_METHOD_TEXT=fp16

COMP_OBJ="bits"
COMP_OBJ_TEXT=bits

# TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq openbookqa social_iqa"
TASKS="coqa gsm8k truthfulqa"
# TASKS="coqa truthfulqa"
# TASKS="truthfulqa"
# TASKS="coqa"
# TASKS="gsm8k"
# TASKS="gsm8k_cot"

N=1
DATASETS="wikitext2 c4"

# GROUP_SIZE=128
# GROUP_SIZE=-1


# TARGET_COMP_OBJ=bits
# TARGET_BITS_LIST=(2 3 4)

# W_BITS=16
W_BITS=4
W_GROUP_SIZE=128

# KV_METHOD="hqq"
KV_METHOD="kivi"

# K_BITS=2
# V_BITS=2
K_BITS=4
V_BITS=4
# K_BITS=8
# V_BITS=8
# K_BITS=16
# V_BITS=16

# KV_GROUP_SIZE=0
# KV_GROUP_SIZE=32
# KV_GROUP_SIZE=64
KV_GROUP_SIZE=128


RESIDUAL_LENGTH=128
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token

# LM_EVAL_BATCH_SIZE=1
# LM_EVAL_BATCH_SIZE=4
# LM_EVAL_BATCH_SIZE=16
LM_EVAL_BATCH_SIZE=32

# NUM_FEWSHOT=0
# NUM_FEWSHOT=5


# SAVE=save/result/${TODAY}_${MODEL_NAME}_${COMP_OBJ}_${METHOD}_${BITS}
SAVE=save/result/${TODAY}_test

LONG_BENCH_RESULT_PATH=save/long_bench/${TODAY}_${MODEL_NAME}_pure_${W_METHOD_TEXT}_${KV_METHOD}_${COMP_OBJ_TEXT}_w${W_BITS}bits_w${W_GROUP_SIZE}gs_k${K_BITS}bits_k${KV_GROUP_SIZE}gs_${K_QUANT_SCHEME}_v${V_BITS}bits_v${KV_GROUP_SIZE}gs_${V_QUANT_SCHEME}_r${RESIDUAL_LENGTH}
LONG_BENCH_CONFIG=utils/long_bench_config

PASS_KEY_FILE=/NAS/SJ/actquant/search/passkey_examples.jsonl

N_PROC=1

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} awqgptq.py \
--gpu_id ${DEVICES} \
--w_method ${W_METHOD} \
--kv_method ${KV_METHOD} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--dtype ${DTYPE} \
--config ${CONFIG} \
--w_bits ${W_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--k_group_size ${KV_GROUP_SIZE} \
--v_group_size ${KV_GROUP_SIZE} \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
-n ${N} \
--save ${SAVE} \
--clip_asym \
--datasets ${DATASETS} \
--zeroshot \
--tasks ${TASKS} \
--lm_eval_batch_size ${LM_EVAL_BATCH_SIZE} \
# --long_bench \
# --long_bench_result_path ${LONG_BENCH_RESULT_PATH} \
# --long_bench_config ${LONG_BENCH_CONFIG} \
# --pass_key_file ${PASS_KEY_FILE} \
# --num_fewshot ${NUM_FEWSHOT}
# --use_flash \
# --quant_kv_output \


# --zeroshot_batch_size ${LM_EVAL_BATCH_SIZE} \

