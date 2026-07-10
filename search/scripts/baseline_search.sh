DEVICES=${1:-0}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# Single-stage joint (loss × wbits × eff_kvbits) NSGA-III baseline (baseline_search.py).
# Baseline for the two-stage method (search.sh per-axis → second_search.sh joint):
# one NSGA-III search over the full joint space; same evaluator/surrogate/protocol as search.sh.

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=bfloat16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# CONFIG=config/qwen2.json

# weights
W_METHOD=hqq
W_BITS="2 3 4"; W_BITS_TEXT="234"
W_GROUP_SIZE=128; AXIS=1

# USE_QEFT=False
USE_QEFT=True
if [ "${USE_QEFT}" == "True" ]; then
    N_QEFT_COLUMN="0 32 64 96 128"
    BASE_OUTLIER_BITS="2 3"
    N_OUTLIER=128
    QEFT_RANK_TEXT=32_64_96_128
    QEFT_OUTLIER_DATASET=wikitext2
fi

# KV bits + group size + ThinK prune
KV_METHOD="kivi think"
KV_BITS="2 3 4"; KV_BITS_TEXT="234"
KV_GROUP_SIZE=("32 64 128" "32 64 128" "128")   # per KV bit-width
KV_GROUP_SIZE_TEXT=3264128x2_128
K_PRUNING_DIM="0 16 32 48 64"
V_PRUNING_DIM="0 16 32 48 64"

K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token
RESIDUAL_LENGTH=128
ATTN_SINK=8

# 3-D joint front. Bounds generous so DOE anchors stay feasible (wbits max > uniform-4-bit
# ~4.06; eff_kvbits min < pruned corner); also serve as the NSGA-III comp constraints.
COMP_OBJ="wbits eff_kvbits"
COMP_OBJ_MIN="2 0.1"
COMP_OBJ_MAX="5 5"
N_TOKEN=0

W_METHOD_TEXT=${W_METHOD}
KV_METHOD_TEXT=${KV_METHOD// /_}

QMODEL_PATHS_LIST=()
for B in ${W_BITS}; do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

# search loop / NSGA-III
SURROGATE=rbf
GA_POP_SIZE=200          # >= das-dennis 3-obj/12-part = 91 ref dirs
REF_PARTITIONS=12
N_DOE=500
ITERATIONS=200
N_ITER=50
SEED=0
SAVE_ITER=1
ANCHOR_LEVELS=3          # full anchor grid explodes; thin each axis to min/mid/max
# data-parallel EVALUATION: N_PROC ranks each replicate the model and evaluate a shard of the
# calibration batches (search() is multi-process safe → no duplicated output/file races).
# For N_PROC>1 set DEVICES=0,1,2,3 (one GPU/rank); needs #calibration batches >= N_PROC.
N_PROC=1

# post-NSGA-III candidate down-select (as in second_search.py): maximin/grid/hybrid/moo
CAND_EVEN=moo
MOO_ALGO=nsga3
MOO_GAP_STD=False
MOO_OBJS=axis_gap

# measurement protocol (matches search.sh)
LOSS_FUNC=jsd
METRIC=loss
MAX_VALUE=0.7
DATASET=wikitext2
N_SAMPLE=128
SEQLEN=2048
STRIDE=128
PREFILL_PROMPT=True
LAST_TOKENS=512
MUT_PROB=0.1
CROSSOVER_PROB=0.9

SINK_TAG=""; [ ${ATTN_SINK} -ne 0 ] && SINK_TAG="_sk${ATTN_SINK}"
PP_TAG="";   [ "${PREFILL_PROMPT}" == "True" ] && PP_TAG="_pp${LAST_TOKENS}"
QEFT_TAG=""; [ -n "${N_QEFT_COLUMN}" ] && QEFT_TAG="_qc$(echo ${N_QEFT_COLUMN} | sed 's/ /-/g')_ob$(echo ${BASE_OUTLIER_BITS} | sed 's/ //g')"
DS_TAG="_${CAND_EVEN}"; [ "${CAND_EVEN}" == "moo" ] && [ "${MOO_OBJS}" == "axis_gap" ] && DS_TAG="_moo_axg"
SAVE=save/baseline_search/${TODAY}_${MODEL_NAME}_baseline_joint_${W_METHOD_TEXT}${QEFT_TAG}_${KV_METHOD_TEXT}${SINK_TAG}_${SURROGATE}_nsga3p${REF_PARTITIONS}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}_gs${KV_GROUP_SIZE_TEXT}_doe${N_DOE}_it${ITERATIONS}n${N_ITER}p${GA_POP_SIZE}_st${STRIDE}${PP_TAG}${DS_TAG}_s${SEED}

echo "BASELINE-SEARCH -> ${SAVE}"

ARGS="--config ${CONFIG} \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--dtype ${DTYPE} \
--w_method ${W_METHOD} \
--kv_method ${KV_METHOD} \
--quant_model_paths ${QMODEL_PATHS} \
--w_bits ${W_BITS} \
--k_bits ${KV_BITS} \
--v_bits ${KV_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--k_pruning_dim ${K_PRUNING_DIM} \
--v_pruning_dim ${V_PRUNING_DIM} \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
--residual_length ${RESIDUAL_LENGTH} \
--attn_sink ${ATTN_SINK} \
--comp_obj ${COMP_OBJ} \
--comp_obj_min ${COMP_OBJ_MIN} \
--comp_obj_max ${COMP_OBJ_MAX} \
--n_token ${N_TOKEN} \
--predictor ${SURROGATE} \
--ga_pop_size ${GA_POP_SIZE} \
--ref_partitions ${REF_PARTITIONS} \
--n_doe ${N_DOE} \
--iterations ${ITERATIONS} \
--n_iter ${N_ITER} \
--anchor_levels ${ANCHOR_LEVELS} \
--metric ${METRIC} \
--loss_func ${LOSS_FUNC} \
--max_value ${MAX_VALUE} \
--mut_prob ${MUT_PROB} \
--crossover_prob ${CROSSOVER_PROB} \
--cand_even ${CAND_EVEN} \
--moo_algo ${MOO_ALGO} \
--moo_objs ${MOO_OBJS} \
--dataset ${DATASET} \
--n_sample ${N_SAMPLE} \
--seqlen ${SEQLEN} \
--seed ${SEED} \
--save_iter ${SAVE_ITER} \
--debug \
--save ${SAVE}"

for g in "${KV_GROUP_SIZE[@]}"; do ARGS+=" --k_group_size ${g} "; done
for g in "${KV_GROUP_SIZE[@]}"; do ARGS+=" --v_group_size ${g} "; done

[ ${STRIDE} -gt 0 ] && ARGS+=" --stride ${STRIDE} "
[ "${PREFILL_PROMPT}" == 'True' ] && ARGS+=" --prefill_prompt --last_tokens ${LAST_TOKENS} "
[ "${MOO_GAP_STD}" == 'True' ] && ARGS+=" --moo_gap_std "

if [ -n "${N_QEFT_COLUMN}" ]; then
    OUTLIER_PATH=/NAS/SJ/actquant/search/outlier/${MODEL_NAME}/w16_r${QEFT_RANK_TEXT}_${QEFT_OUTLIER_DATASET}/outlier.pth
    ARGS+=" --n_qeft_column ${N_QEFT_COLUMN} \
    --base_outlier_bits ${BASE_OUTLIER_BITS} \
    --outlier_path ${OUTLIER_PATH} \
    --n_outlier ${N_OUTLIER} "
fi

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 \
    --main_process_port=${PORT_NUM} baseline_search.py ${ARGS}
