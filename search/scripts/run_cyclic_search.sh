#!/usr/bin/env bash
# Cyclic Pareto Frontier Search (CPFS)
# Usage: bash scripts/run_cyclic_search.sh <GPU_ID>

DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# DTYPE=float16
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# DTYPE=float16
# CONFIG=config/mistral.json

# ---- Quantization method ----
W_METHOD="hqq"
W_METHOD_TEXT="hqq"
AXIS=1

KV_METHOD="kivi"
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token
RESIDUAL_LENGTH=128

# ---- Bit choices ----
W_BITS="2 3 4"
W_BITS_TEXT="234"
W_GROUP_SIZE=128

KV_BITS="2 3 4"
KV_BITS_TEXT="234"
KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
KV_GROUP_SIZE_TEXT=3264128x3

# ---- Joint compression objectives ----
# Both wbits and kvbits are required for CyclicSearch
COMP_OBJ="wbits kvbits"
COMP_OBJ_MIN="2 2"
COMP_OBJ_MAX="5 5"
COMP_OBJ_TEXT="wbits_kvbits"

# ---- CPFS-specific hyperparams ----
N_CYCLES=3          # progressive threshold decays over this many cycles
MAX_CONTEXTS=5      # max unique Pareto contexts per phase iteration
PHASES="kv w"       # alternate KV-phase then W-phase
THRESHOLD_SCHEDULE=cosine

# ---- Surrogate / search budget ----
N_DOE=400
ITER=120            # 120 iters × 2 phases = 60 per phase ≈ 30 per cycle
N_ITER=50
GA_POP_SIZE=200
SUBSET_POP_SIZE=100

# ---- Predictor ----
PREDICTOR=rbf

# ---- Evaluation ----
LOSS_FUNC=jsd
MAX_VALUE=0.7
MUT_PROB=0.1
CROSSOVER_PROB=0.9
SAVE_ITER=10

DATASET=wikitext2
N_SAMPLE=128
SEQLEN=2048
MIN_SEQLEN=0
DATA_BATCH_SIZE=1

METRIC=loss

# ---- Quant model paths ----
QMODEL_PATHS_LIST=()
for B in ${W_BITS}
do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

# ---- Save path ----
SAVE=save/search/cyclic/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${DATA_BATCH_SIZE}bs_${N_SAMPLE}sample_${SEQLEN}seq_${N_CYCLES}cycles_${MAX_CONTEXTS}ctx_${THRESHOLD_SCHEDULE}_${PREDICTOR}

N_PROC=1

ARGS="--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--quant_model_paths ${QMODEL_PATHS} \
--w_method ${W_METHOD} \
--kv_method ${KV_METHOD} \
--w_bits ${W_BITS} \
--k_bits ${KV_BITS} \
--v_bits ${KV_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--comp_obj ${COMP_OBJ} \
--comp_obj_min ${COMP_OBJ_MIN} \
--comp_obj_max ${COMP_OBJ_MAX} \
--n_token 0 \
--residual_length ${RESIDUAL_LENGTH} \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
--quant_kv_output \
--predictor ${PREDICTOR} \
--save ${SAVE} \
--iterations ${ITER} \
--n_doe ${N_DOE} \
--n_iter ${N_ITER} \
--metric ${METRIC} \
--ga_pop_size ${GA_POP_SIZE} \
--subset_pop_size ${SUBSET_POP_SIZE} \
--config ${CONFIG} \
--debug \
--max_value ${MAX_VALUE} \
--loss_func ${LOSS_FUNC} \
--mut_prob ${MUT_PROB} \
--crossover_prob ${CROSSOVER_PROB} \
--n_sample ${N_SAMPLE} \
--data_batch_size ${DATA_BATCH_SIZE} \
--seqlen ${SEQLEN} \
--min_seqlen ${MIN_SEQLEN} \
--dataset ${DATASET} \
--save_iter ${SAVE_ITER} \
--n_cycles ${N_CYCLES} \
--max_contexts ${MAX_CONTEXTS} \
--phases ${PHASES} \
--threshold_schedule ${THRESHOLD_SCHEDULE}"

for g in "${KV_GROUP_SIZE[@]}"
do
    ARGS+=" --k_group_size ${g} "
done

for g in "${KV_GROUP_SIZE[@]}"
do
    ARGS+="--v_group_size ${g} "
done

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} cyclic_search.py \
${ARGS}
