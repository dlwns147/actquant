#!/bin/bash
# Parameterized search_think.py launcher for COMP_OBJ=kvbits.
# Search space: KV bits in {2,3,4} with group_sizes {32,64,128} per layer
# (no kvdim pruning). The "right end" is kvbits=4 (highest fidelity / cost).
#
# Usage:
#   bash search_think_param_kvbits.sh <gpu_id> <last_tokens> <stride> [<n_sample> [<total_seq> [<resume_path>]]]
# Defaults: n_sample=32, total_seq=2048.
# Constraint: total_seq > last_tokens (else eval degenerates - utils/eval.py:242).

DEVICES=${1}
LAST_TOKENS=${2}
STRIDE=${3}
N_SAMPLE=${4:-32}
TOTAL_SEQ=${5:-2048}
RESUME=${6:-}

if [ -z "$DEVICES" ] || [ -z "$LAST_TOKENS" ] || [ -z "$STRIDE" ]; then
  echo "Usage: $0 <gpu_id> <last_tokens> <stride> [<n_sample> [<total_seq> [<resume_path>]]]" >&2
  exit 2
fi
if [ "$TOTAL_SEQ" -le "$LAST_TOKENS" ]; then
  echo "ERROR: total_seq ($TOTAL_SEQ) must be > last_tokens ($LAST_TOKENS)" >&2
  exit 3
fi

TODAY=$(date +%y%m%d%H%M)
PORT_NUM=$(( ( RANDOM % 10000 ) + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

COMP_OBJ=kvbits
N_TOKEN=0

W_METHOD="hqq"; W_METHOD_TEXT="hqq"
AXIS=1
KV_METHOD="kivi"; KV_METHOD_TEXT="kivi"
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token
RESIDUAL_LENGTH=128

W_BITS="4"; W_BITS_TEXT="4"; W_GROUP_SIZE=128
KV_BITS="2 3 4"; KV_BITS_TEXT="234"
KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128"); KV_GROUP_SIZE_TEXT=3264128x3
K_PRUNING_DIM="0"; V_PRUNING_DIM="0"
K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

COMP_OBJ_TEXT=kvbits
COMP_OBJ_MIN=1; COMP_OBJ_MIN_TEXT=1
COMP_OBJ_MAX=5; COMP_OBJ_MAX_TEXT=5

N_DOE=400; ITER=150; N_ITER=30

QMODEL_PATHS_LIST=()
for B in ${W_BITS}; do
  QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

LOSS_FUNC=jsd; PREDICTOR=rbf; DATASET=wikitext2
DATA_BATCH_SIZE=1; MIN_SEQLEN=0; GA_POP_SIZE=200
METRIC=loss; MUT_PROB=0.1; CROSSOVER_PROB=0.9; SAVE_ITER=10
MAX_VALUE=0.7

PREFILL_PROMPT=True
SEQLEN=$(( ${TOTAL_SEQ} - ${LAST_TOKENS} ))
PP_TAG="_pp${LAST_TOKENS}"

SAVE=save/search/think/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${METRIC}_w_${W_METHOD_TEXT}_kv_${KV_METHOD}_iter_${ITER}_n_iter_${N_ITER}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}bits_w${W_GROUP_SIZE}kv${KV_GROUP_SIZE_TEXT}gs_${RESIDUAL_LENGTH}res_len_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_kdim${K_PRUNING_DIM_TEXT}_vdim${V_PRUNING_DIM_TEXT}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_${LOSS_FUNC}_co_${CROSSOVER_PROB}_mut_${MUT_PROB}_${DATASET}_${DATA_BATCH_SIZE}bs_${N_SAMPLE}sample_${SEQLEN}seq_${N_TOKEN}token_${PREDICTOR}_${STRIDE}stride${PP_TAG}

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
--n_token ${N_TOKEN} \
--residual_length ${RESIDUAL_LENGTH} \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
--k_pruning_dim ${K_PRUNING_DIM} \
--v_pruning_dim ${V_PRUNING_DIM} \
--predictor ${PREDICTOR} \
--save ${SAVE} \
--iterations ${ITER} \
--n_doe ${N_DOE} \
--n_iter ${N_ITER} \
--metric ${METRIC} \
--ga_pop_size ${GA_POP_SIZE} \
--config ${CONFIG} \
--debug \
--max_value ${MAX_VALUE} \
--mut_prob ${MUT_PROB} \
--crossover_prob ${CROSSOVER_PROB} \
--loss_func ${LOSS_FUNC} \
--n_sample ${N_SAMPLE} \
--data_batch_size ${DATA_BATCH_SIZE} \
--seqlen ${SEQLEN} \
--min_seqlen ${MIN_SEQLEN} \
--dataset ${DATASET} \
--save_iter ${SAVE_ITER}"

for g in "${KV_GROUP_SIZE[@]}"; do ARGS+=" --k_group_size ${g} "; done
for g in "${KV_GROUP_SIZE[@]}"; do ARGS+="--v_group_size ${g} "; done
if [ ${STRIDE} -gt 0 ]; then ARGS+=" --stride ${STRIDE} "; else ARGS+=" --quant_kv_output "; fi
if [ ${PREFILL_PROMPT} == 'True' ]; then ARGS+=" --prefill_prompt --last_tokens ${LAST_TOKENS} "; fi
if [ -n "${RESUME}" ]; then ARGS+=" --resume ${RESUME} "; fi

echo "==== launch kvbits: gpu=${DEVICES} ans=${LAST_TOKENS} stride=${STRIDE} nsamp=${N_SAMPLE} total=${TOTAL_SEQ} ===="
echo "SAVE=${SAVE}"

cd /NAS/SJ/actquant/search
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search_think.py ${ARGS}
