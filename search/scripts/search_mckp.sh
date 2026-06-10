DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# ── Stage-1 per-method search by MEASURED DP-MCKP (drop-in alt to scripts/search.sh).
#    Same LlamaEvaluator / protocol; measures per-module marginal JSD + frontier
#    archs (real values), writes iter_mckp.stats (post_search-consumable).
#    Pick the axis via COMP_OBJ exactly like search.sh.

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=bfloat16
CONFIG=config/llama.json

COMP_OBJ=wbits
# COMP_OBJ=kvbits
# COMP_OBJ=kvdim

W_METHOD="hqq"
AXIS=1
KV_METHOD="kivi"
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token
RESIDUAL_LENGTH=128
# Attention-sink (KVSink/KIVI-K2V2*): keep first S KV tokens FP. 0=off. NOT a
# searched axis — a fixed global primitive like RESIDUAL_LENGTH. Recommend 8 when on.
ATTN_SINK=0

if [ ${COMP_OBJ} == 'wbits' ]; then
    W_BITS="2 3 4";  W_GROUP_SIZE=128
    KV_BITS="4";     KV_GROUP_SIZE=("128")
    K_PRUNING_DIM="0"; V_PRUNING_DIM="0"
    COMP_OBJ_MIN=${W_BITS:0:1}; COMP_OBJ_MAX=5; N_TOKEN=0
elif [ ${COMP_OBJ} == 'kvbits' ]; then
    W_BITS="4";      W_GROUP_SIZE=128
    KV_BITS="2 3 4"; KV_GROUP_SIZE=("64 128" "64 128" "128")
    K_PRUNING_DIM="0"; V_PRUNING_DIM="0"
    COMP_OBJ_MIN=1;  COMP_OBJ_MAX=5; N_TOKEN=0
elif [ ${COMP_OBJ} == 'kvdim' ]; then
    KV_METHOD="think"
    W_BITS="4";      W_GROUP_SIZE=128
    KV_BITS="4";     KV_GROUP_SIZE=("128")
    K_PRUNING_DIM="0 16 32 48 64"; V_PRUNING_DIM="0 16 32 48 64"
    COMP_OBJ_MIN=0;  COMP_OBJ_MAX=128; N_TOKEN=0
fi

QMODEL_PATHS_LIST=()
for B in ${W_BITS}; do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

LOSS_FUNC=jsd
DATASET=wikitext2
N_SAMPLE=128
SEQLEN=2048
MIN_SEQLEN=0
DATA_BATCH_SIZE=1
STRIDE=128
PREFILL_PROMPT=True
LAST_TOKENS=512
METRIC=loss

# ── QEFT outlier-column axis on the HQQ backend (COMP_OBJ=wbits only) ─────────
# QEFT_OUTLIER=1 turns each weight linear's option into a (w_bits, n_outlier)
# tuple: the HQQ bank at w_bits is loaded and n_outlier FP16 columns are
# OVERWRITTEN on top (insert_fp16_channel_hqq). This is the CHEAP weight-swap
# proxy for QEFT-direct quant (no per-arch GPTQ/AWQ requant), so the per-module
# marginal sweep stays HQQ-fast. Outlier indices/values come from the multi-rank
# dict {key:{n_out:[cols]}} from extract_outidx.py (its ranks must cover the
# N_QEFT_COLUMN values). The wbits cost folds the FP16 columns in automatically.
# NOTE: HQQ uniform 2-bit gs128 ~collapses, so QEFT-direct (w_method qeft) is
# better in the aggressive low-bit corner; this HQQ proxy is for fast search.
QEFT_OUTLIER=0
N_QEFT_COLUMN="0 32 64 96 128"
BASE_OUTLIER_BITS="2 3 4"
OUTLIER_RANK=128   # extract_outidx.py target_rank max; must be >= max(N_QEFT_COLUMN)
OUTLIER_PATH=/NAS/SJ/actquant/outlier/${MODEL_NAME}/w16_r32_64_96_128_${DATASET}/outlier.pth
QEFT_TAG=""
[ ${QEFT_OUTLIER} -eq 1 ] && QEFT_TAG="_qeftout${N_QEFT_COLUMN// /-}"

# MCKP knobs
# 0 = MEASURE the ENTIRE DP-MCKP Pareto frontier (no subsampling). The resulting
# iter_mckp.stats is directly consumable by post_search.py as a per-axis archive,
# e.g.  --w_expr ${SAVE}/iter_mckp.stats  (or --kv_expr / --kvdim_expr per COMP_OBJ).
MCKP_FRONT_POINTS=0

# Abbreviated attention-sink tag (e.g. _sk8), only when on so sink=0 names stay comparable.
SINK_TAG=""
[ ${ATTN_SINK} -ne 0 ] && SINK_TAG="_sk${ATTN_SINK}"
SAVE=save/search/mckp/${TODAY}_${MODEL_NAME}_${COMP_OBJ}_${METRIC}_w_${W_METHOD}${QEFT_TAG}_kv_${KV_METHOD}_w${W_BITS// /}kv${KV_BITS// /}bits_${RESIDUAL_LENGTH}res${SINK_TAG}_obj_${COMP_OBJ_MIN}_${COMP_OBJ_MAX}_${LOSS_FUNC}_${DATASET}_${N_SAMPLE}sample_${SEQLEN}seq_${STRIDE}stride_pp${LAST_TOKENS}

ARGS="--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--dtype ${DTYPE} \
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
--attn_sink ${ATTN_SINK} \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
--k_pruning_dim ${K_PRUNING_DIM} \
--v_pruning_dim ${V_PRUNING_DIM} \
--save ${SAVE} \
--metric ${METRIC} \
--config ${CONFIG} \
--loss_func ${LOSS_FUNC} \
--n_sample ${N_SAMPLE} \
--data_batch_size ${DATA_BATCH_SIZE} \
--seqlen ${SEQLEN} \
--min_seqlen ${MIN_SEQLEN} \
--dataset ${DATASET} \
--mckp_front_points ${MCKP_FRONT_POINTS}"

for g in "${KV_GROUP_SIZE[@]}"; do ARGS+=" --k_group_size ${g} "; done
for g in "${KV_GROUP_SIZE[@]}"; do ARGS+=" --v_group_size ${g} "; done

if [ ${STRIDE} -gt 0 ]; then ARGS+=" --stride ${STRIDE} "; else ARGS+=" --quant_kv_output "; fi
if [ ${PREFILL_PROMPT} == 'True' ]; then ARGS+=" --prefill_prompt --last_tokens ${LAST_TOKENS} "; fi

# HQQ-based QEFT outlier-column search axis (cheap insert-on-top proxy).
if [ ${QEFT_OUTLIER} -eq 1 ]; then
    ARGS+=" --n_qeft_column ${N_QEFT_COLUMN} --base_outlier_bits ${BASE_OUTLIER_BITS} --outlier_path ${OUTLIER_PATH} --n_outlier ${OUTLIER_RANK} "
fi

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=1 --num_machines=1 \
    --main_process_port=${PORT_NUM} search_mckp.py ${ARGS}
