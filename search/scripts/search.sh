DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=bfloat16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# DTYPE=bfloat16
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# DTYPE=bfloat16
# CONFIG=config/mistral.json


# ── Named calibration metric (the ONE registry, utils/metric_specs.py) ─────
# METRIC_TASK=<name> takes the ENTIRE measurement protocol from the registry --
# the same names correlation.py --metrics and post_search.py --metric_tasks
# resolve -- so this archive and the correlation table that judges it are the
# same measurement by construction instead of by retyping stride / last_tokens /
# n_sample correctly in two scripts. It overrides DATASET / USE_CHAT_TEMPLATE /
# N_SAMPLE / SEQLEN / MIN_SEQLEN / METRIC / LOSS_FUNC / STRIDE / PREFILL_PROMPT /
# LAST_TOKENS / SCORE / USE_KEY_TOKEN (+ the key-token protocol and archive
# path) BELOW, so the SAVE-dir tags are derived from the registry too -- a
# hand-set run and its METRIC_TASK twin land in the SAME directory name.
# The resolution happens HERE, not in search.py: search.py keeps taking raw knobs
# and knows nothing about metric names, so the measurement and the SAVE-dir tags
# both come from this one lookup. A knob set by hand below is OVERRIDDEN by the
# name (and the override is printed) — the two are alternatives, not layers.
# The name + its spec hash are written to <SAVE>/metric_task.json.
# Empty = the hand-set knobs below (unchanged legacy behaviour).
# `python -m utils.metric_specs` lists every valid name.
METRIC_TASK=""
# METRIC_TASK=wt2_jsd_pp512_s128_chat     # == the hand-set defaults below
# METRIC_TASK=wt2_jsd_pp128_s32_chat
# METRIC_TASK=gov_jsd_pp512_s32_chat
# BASE key-token archive dir; the kt_eval-<evaluator>_tgt-<model> root is
# DERIVED from the metric name (its _l8b/_q7b/_q72b suffix), so this is only
# used when METRIC_TASK names a key-token metric.
KEY_TOKEN_BASE=key_token

# COMP_OBJ=wbits
# COMP_OBJ=kvbits
# COMP_OBJ=kvdim
COMP_OBJ=eff_kvbits
# COMP_OBJ=memory

# Key-token (LongPPL) weighting: score only the tokens an EVALUATOR model finds
# long-range-dependent. The archive stores CHARACTER intervals, so one evaluator
# serves every target -- but the archive itself is per (target, corpus, seqlen)
# because the loader text is target-tokenized.
# USE_KEY_TOKEN=True
USE_KEY_TOKEN=False
KEY_TOKEN_EVALUATOR=${MODEL_NAME}      # any Instruct model; Qwen2.5-72B-Instruct is the shipped one
TRUNC_LEN=512
SLIDING_WINDOW=128
ALPHA=1
BETA=-1

# USE_QEFT=True
USE_QEFT=False

# N_TOKEN=1024
N_TOKEN=16384
# N_TOKEN=32768

W_METHOD="hqq"
W_METHOD_TEXT="hqq"

AXIS=1

KV_METHOD="kivi"
KV_METHOD_TEXT="kivi"
# KV_METHOD="hqq"
# KV_METHOD_TEXT="hqq"

K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token

RESIDUAL_LENGTH=128
# RESIDUAL_LENGTH=0

# ATTN_SINK=0
ATTN_SINK=8

# DOE anchor grid thinning: initialize() seeds anchors from the cartesian
# product of per-axis options (w x k x v x kdim x vdim), which explodes past
# N_DOE for eff_kvbits/memory (e.g. 9x9x5x5=2025 > 600 -> no random samples
# left). ANCHOR_LEVELS=N keeps only N evenly spaced options per axis
# (3 = min/mid/max; list ends are always kept); 0 = full grid (legacy).
ANCHOR_LEVELS=0

if [ ${COMP_OBJ} == 'wbits' ]; then
    if [ ${USE_QEFT} == 'True' ]; then
        N_QEFT_COLUMN="0 32 64 96 128"   # per-layer outlier-column options
        BASE_OUTLIER_BITS="2 3"        # which W bit-widths get the outlier ladder
        N_OUTLIER=128                    # only to satisfy search.py's outlier-arg assert
        QEFT_RANK_TEXT=32_64_96_128      # non-zero ranks → outlier-dict dirname
    fi

    W_BITS="2 3 4"
    W_BITS_TEXT="234"
    W_GROUP_SIZE=128

    KV_BITS="4"
    KV_BITS_TEXT="4"
    KV_GROUP_SIZE=("128")
    KV_GROUP_SIZE_TEXT=128
    
    K_PRUNING_DIM="0"
    # V_PRUNING_DIM="0 16 32 48 64"
    V_PRUNING_DIM="0"
    K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
    V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

    COMP_OBJ_TEXT=wbits
    COMP_OBJ_MIN=${W_BITS:0:1}
    COMP_OBJ_MIN_TEXT=${W_BITS:0:1}
    COMP_OBJ_MAX=5
    COMP_OBJ_MAX_TEXT=5
    N_TOKEN=0

    N_DOE=400
    ITER=200
    N_ITER=50

elif [ ${COMP_OBJ} == 'kvbits' ]; then
    KV_METHOD="kivi"
    KV_METHOD_TEXT="kivi"
    W_BITS="4"
    W_BITS_TEXT="4"
    W_GROUP_SIZE=128

    KV_BITS="2 3 4"
    KV_BITS_TEXT="234"
    # KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
    # KV_GROUP_SIZE_TEXT=3264128x3
    KV_GROUP_SIZE=("32 64 128" "32 64 128" "128")
    KV_GROUP_SIZE_TEXT=3264128x2_128
    # KV_GROUP_SIZE=("64 128" "64 128" "128")
    # KV_GROUP_SIZE_TEXT=64128x2_128

    K_PRUNING_DIM="0"
    # V_PRUNING_DIM="0 16 32 48 64"
    V_PRUNING_DIM="0"
    K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
    V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

    COMP_OBJ_TEXT=kvbits
    COMP_OBJ_MIN=1
    COMP_OBJ_MIN_TEXT=1
    COMP_OBJ_MAX=5
    COMP_OBJ_MAX_TEXT=5
    N_TOKEN=0

    N_DOE=400
    ITER=150
    N_ITER=30

elif [ ${COMP_OBJ} == 'kvdim' ]; then
    KV_METHOD="think"
    KV_METHOD_TEXT="think"
    W_BITS="4"
    W_BITS_TEXT="4"
    W_GROUP_SIZE=128

    KV_BITS="4"
    KV_BITS_TEXT="4"
    KV_GROUP_SIZE=("128")
    KV_GROUP_SIZE_TEXT=128
    
    K_PRUNING_DIM="0 16 32 48 64"
    V_PRUNING_DIM="0 16 32 48 64"
    # V_PRUNING_DIM="0"
    K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
    V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

    COMP_OBJ_TEXT=kvdim
    COMP_OBJ_MIN=0      # remained_dim >= 64  (prune <= 50% of head_dim=128)
    COMP_OBJ_MIN_TEXT=0
    COMP_OBJ_MAX=128     # remained_dim <= 128 (no pruning)
    COMP_OBJ_MAX_TEXT=128
    N_TOKEN=0

    N_DOE=200
    ITER=150
    N_ITER=30

elif [ ${COMP_OBJ} == 'eff_kvbits' ]; then
    KV_METHOD="kivi think"
    KV_METHOD_TEXT="kivi_think"
    W_BITS="4"
    W_BITS_TEXT="4"
    W_GROUP_SIZE=128

    KV_BITS="2 3 4"
    KV_BITS_TEXT="234"
    # KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
    # KV_GROUP_SIZE_TEXT=3264128x3
    KV_GROUP_SIZE=("32 64 128" "32 64 128" "128")
    KV_GROUP_SIZE_TEXT=3264128x2_128

    K_PRUNING_DIM="0 16 32 48 64"
    V_PRUNING_DIM="0 16 32 48 64"
    # V_PRUNING_DIM="0"
    K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
    V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

    COMP_OBJ_TEXT=eff_kvbits
    COMP_OBJ_MIN=0.1
    COMP_OBJ_MIN_TEXT=0.1
    COMP_OBJ_MAX=5
    COMP_OBJ_MAX_TEXT=5

    # full anchor grid = k7 x v7 x kdim5 x vdim5 = 1225 > N_DOE; thin to 3^4 = 81
    ANCHOR_LEVELS=3

    N_DOE=300
    ITER=200
    N_ITER=50

elif [ ${COMP_OBJ} == 'memory' ]; then
    W_BITS="2 3 4"
    W_BITS_TEXT="234"
    W_GROUP_SIZE=128

    KV_BITS="2 3 4"
    KV_BITS_TEXT="234"
    KV_GROUP_SIZE=("32 64 128" "32 64 128" "32 64 128")
    KV_GROUP_SIZE_TEXT=3264128x3
    
    K_PRUNING_DIM="0 16 32 48 64"
    # V_PRUNING_DIM="0 16 32 48 64"
    V_PRUNING_DIM="0"
    K_PRUNING_DIM_TEXT=$(echo ${K_PRUNING_DIM} | sed 's/ /_/g')
    V_PRUNING_DIM_TEXT=$(echo ${V_PRUNING_DIM} | sed 's/ /_/g')

    COMP_OBJ_TEXT=memory
    COMP_OBJ_MIN=1
    COMP_OBJ_MIN_TEXT=1
    COMP_OBJ_MAX=1e99
    COMP_OBJ_MAX_TEXT=1e99

    # full anchor grid = w3 x k9 x v9 x kdim5 = 1215 > N_DOE; thin to 3^4 = 81
    ANCHOR_LEVELS=3

    N_DOE=600
    ITER=200
    N_ITER=50

fi

QMODEL_PATHS_LIST=()
for B in ${W_BITS}
do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")


LOSS_FUNC=jsd
# LOSS_FUNC=cross_entropy

PREDICTOR=rbf
# PREDICTOR=mlp
# PREDICTOR=gp

# Chat-templated calibration data: `chat:<corpus>` wraps every sample in ONE
# user turn (BOS + role header + document + assistant header), matching the
# format deployment feeds an Instruct model (RULER/LongBench both
# apply_chat_template). SEQLEN/MIN_SEQLEN stay TOTAL lengths -- the document
# budget shrinks by the per-model affix overhead (Llama-3.1 35 tokens,
# Qwen2.5 29, Gemma-3 9, Mistral-v0.3 4). Loss/train side only (no PPL loader).
# NOTE: this REDEFINES the objective, so archives produced with it are not
# comparable to the pre-existing raw-wikitext2 ones. Measured on a 130-arch DOE
# (eff_kvbits, Llama-3.1-8B): chat/raw JSD ratio 1.031, within-band Spearman
# 0.983-0.995, top-10 overlap 8/10 -- i.e. near-monotone with raw-JSD on that
# axis. Set to False to reproduce the older archives.
USE_CHAT_TEMPLATE=True
# USE_CHAT_TEMPLATE=False

DATASET=wikitext2
N_SAMPLE=128
# N_SAMPLE=32
# N_SAMPLE=64
# SEQLEN=1024
# SEQLEN=1536
SEQLEN=2048
DATA_BATCH_SIZE=1
MIN_SEQLEN=0

# DATASET=gov_report
# # N_SAMPLE=4
# N_SAMPLE=8
# # N_SAMPLE=16
# # N_SAMPLE=32
# # N_SAMPLE=64
# # SEQLEN=2048
# # MIN_SEQLEN=2048
# SEQLEN=8192
# MIN_SEQLEN=8192
# # SEQLEN=16384
# # MIN_SEQLEN=16384
# DATA_BATCH_SIZE=1
# # MIN_SEQLEN=0

# STRIDE=0
# STRIDE=32
# STRIDE=64
STRIDE=128
# STRIDE=512
# STRIDE=1024

# PREFILL_PROMPT=False
# LAST_TOKENS=0
PREFILL_PROMPT=True

# What enters the loss. LAST_TOKENS is the ANSWER WINDOW either way: it sets the
# prefill/answer split and, under chat:, where the assistant header lands.
#   last : score only that window (the historical behaviour)
#   full : score every position, split unchanged
# Key tokens are ANDed with the window and are sparse, so a small window
# intersects few or none of them (MEASURED: 0 of ~6 per document on 3 of 4
# models at seqlen 512 / last_tokens 128) and eval_loss then scores NOTHING ->
# use 'full' whenever USE_KEY_TOKEN=True.
SCORE=last
# SCORE=full
# LAST_TOKENS=128
# LAST_TOKENS=256
LAST_TOKENS=512
# LAST_TOKENS=1024

# SEQLEN=$(( ${SEQLEN} + ${LAST_TOKENS} ))
# SEQLEN=$(( ${SEQLEN} - ${LAST_TOKENS} ))

GA_POP_SIZE=200

METRIC=loss
# METRIC=ppl

# METRIC_TASK -> overwrite every knob the registry owns (see the block at the
# top). Done HERE: after the hand-set knobs so it wins, before MAX_VALUE and the
# dir tags so they follow the resolved protocol. A bad name / an unmeasurable
# task (needle_*, gsm8k_unpad_pp) / a missing key-token archive aborts with the
# valid list or the gen_key_token command, before any GPU work.
source "$(dirname "${BASH_SOURCE[0]}")/metric_task.sh"
metric_task_apply "${METRIC_TASK}" "${MODEL_NAME}" "${KEY_TOKEN_BASE}"

# PPL is unbounded, so the cross_entropy clamp (5) would flatten every arch onto
# the cap -- max_value only exists to keep a NaN/diverged eval out of the
# archive. Checked before LOSS_FUNC because a PPL task carries cross_entropy.
if [ ${METRIC} == 'ppl' ]; then
    MAX_VALUE=1e4
elif [ ${LOSS_FUNC} == 'cross_entropy' ]; then
    MAX_VALUE=5
elif [ ${LOSS_FUNC} == 'jsd' ]; then
    MAX_VALUE=0.7
fi


MUT_PROB=0.1
CROSSOVER_PROB=0.9
SAVE_ITER=10

# SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_w_hqq_kv_${KV_METHOD}_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_2048seqlen_0minseq_jsd/loss
SENSITIVITY_RESULT_PATH=/NAS/SJ/actquant/search/csv/sensitivity/${MODEL_NAME}_w_hqq_kv_kivi_w24k24v24bits_w128k128x2v128x2group_size_1axis_k_${K_QUANT_SCHEME}_v_${V_QUANT_SCHEME}_wikitext2_128sample_2048seqlen_0minseq_jsd/loss

# The tag has to carry SCORE: _pp512 with score=last and with score=full are
# different metrics (one scores 512 positions, the other the whole sequence)
# and would otherwise collide in the same save dir.
PP_TAG=""
if [ ${PREFILL_PROMPT} == 'True' ]; then
    PP_TAG="_pp${LAST_TOKENS}"
    [ ${SCORE} == 'full' ] && PP_TAG="${PP_TAG}full"
elif [ -n "${METRIC_TASK}" ] && [ ${LAST_TOKENS} -gt 0 ]; then
    # registry tasks that score only a window with NO prefill/answer split
    # (e.g. gov_jsd_lt128): not _pp, but the window must still be in the name.
    PP_TAG="_lt${LAST_TOKENS}"
    [ ${SCORE} == 'full' ] && PP_TAG="${PP_TAG}full"
fi

KT_TAG=""
[ ${USE_KEY_TOKEN} == 'True' ] && KT_TAG="_kt"

# Compress an arithmetic int list ("0 16 32 48 64" -> "0to64x16"); else '_'-join.
compress_dim() {
    local a=($1) n=${#a[@]} i step
    if [ $n -ge 3 ]; then
        step=$(( a[1] - a[0] ))
        if [ $step -gt 0 ]; then
            for (( i=1; i<n; i++ )); do
                [ $(( a[i] - a[i-1] )) -ne $step ] && { echo "${1// /_}"; return; }
            done
            echo "${a[0]}to${a[n-1]}x${step}"; return
        fi
    fi
    echo "${1// /_}"
}
K_PRUNING_DIM_C=$(compress_dim "${K_PRUNING_DIM}")
V_PRUNING_DIM_C=$(compress_dim "${V_PRUNING_DIM}")

SINK_TAG=""
if [ ${ATTN_SINK} -ne 0 ]; then
    SINK_TAG="_sk${ATTN_SINK}"
fi

QEFT_TAG=""
if [ -n "${N_QEFT_COLUMN}" ]; then
    QEFT_TAG="_qc$(echo ${N_QEFT_COLUMN} | sed 's/ /-/g')_ob$(echo ${BASE_OUTLIER_BITS} | sed 's/ //g')"
fi

# One layout, parameterised by the answer window: chat: puts the assistant
# header at seqlen - LAST_TOKENS, so the scored tail is generated in assistant
# position. LAST_TOKENS=0 leaves the tail empty -> the header simply trails the
# document (the old "wrapper"). No separate knob: the incoherent combination
# (wrapper layout WITH a scored tail, which would sit in the USER turn) is not
# representable. The split is visible in the dir name as _pp<LAST_TOKENS>.
CHAT_TAG=""
if [ ${USE_CHAT_TEMPLATE} == 'True' ]; then
    DATASET="chat:${DATASET}"; CHAT_TAG="_ct"
fi

# The key-token archive is computed on the RAW document. Under a chat layout the
# document is SEQLEN minus the model's affix overhead, so the archive must have
# been generated at THAT length -- ask utils.data for it rather than hardcoding
# a per-model constant.
if [ ${USE_KEY_TOKEN} == 'True' ] && [ -n "${METRIC_TASK}" ]; then
    # KEY_TOKEN_PATH (and trunc/sw/alpha/beta) came from the registry, which
    # derived the kt_eval-*_tgt-* root from the metric name and already verified
    # the archive is there -- nothing to rebuild or re-check here.
    echo "[search.sh] key-token archive: ${KEY_TOKEN_PATH}"
elif [ ${USE_KEY_TOKEN} == 'True' ]; then
    # Archive layout: `raw`, or `chat-a<answer window>` -- under a chat layout
    # the sample text depends on where the assistant header falls, so each
    # window is a different input with its own archive. The evaluator and the
    # TARGET are in the root name; the per-corpus protocol is the subdirectory.
    KT_LAYOUT=raw
    KT_SEQLEN=${SEQLEN}
    KT_ANS=0
    if [ ${USE_CHAT_TEMPLATE} == 'True' ]; then
        KT_ANS=${LAST_TOKENS}
        KT_LAYOUT=chat-a${KT_ANS}
    fi
    KT_MIN_SEQLEN=${MIN_SEQLEN}
    KEY_TOKEN_PATH=key_token/kt_eval-${KEY_TOKEN_EVALUATOR}_tgt-${MODEL_NAME}_${KT_LAYOUT}
    KT_SUBDIR=${DATASET##chat*:}_${N_SAMPLE}sample_${KT_SEQLEN}seqlen_${KT_MIN_SEQLEN}min_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta_s0
    if [ ! -d "${KEY_TOKEN_PATH}/${KT_SUBDIR}" ] && [ ! -d "${KEY_TOKEN_PATH}/${DATASET##chat*:}" ]; then
        echo "[search.sh] key-token archive missing: ${KEY_TOKEN_PATH}/${KT_SUBDIR}"
        echo "  generate it (gen_key_token.py resolves the chat body budget and the"
        echo "  answer window itself when you pass --metrics; here it is spelled out):"
        echo "    python gen_key_token.py --config ${CONFIG} --dtype ${DTYPE} \\"
        echo "      --model_path ${MODEL_PATH} --model_name ${KEY_TOKEN_EVALUATOR} \\"
        echo "      --target_model_path ${MODEL_PATH} --target_model ${MODEL_NAME} \\"
        echo "      --dataset ${DATASET} --train --n_sample ${N_SAMPLE} \\"
        echo "      --seqlen ${KT_SEQLEN} --min_seqlen ${KT_MIN_SEQLEN} --answer_tokens ${KT_ANS} \\"
        echo "      --trunc_len ${TRUNC_LEN} --sliding_window ${SLIDING_WINDOW} --alpha ${ALPHA} --beta ${BETA} \\"
        echo "      --save_path ${KEY_TOKEN_PATH}"
        exit 1
    fi
fi

source "$(dirname "${BASH_SOURCE[0]}")/metric_tag.sh"
MTAG=$(metric_tag_from_knobs "${DATASET##chat*:}" "${LOSS_FUNC}" "${METRIC:-loss}" \
                             "${N_SAMPLE}" "${SEQLEN}" "${MIN_SEQLEN:-0}")

SAVE=save/search/think/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${KV_METHOD_TEXT}${QEFT_TAG}${SINK_TAG}_w${W_BITS_TEXT}kv${KV_BITS_TEXT}_gs${KV_GROUP_SIZE_TEXT}_r${RESIDUAL_LENGTH}_kd${K_PRUNING_DIM_C}_vd${V_PRUNING_DIM_C}_obj_${COMP_OBJ_MIN_TEXT}_${COMP_OBJ_MAX_TEXT}_st${STRIDE}${PP_TAG}${MTAG}${CHAT_TAG}${KT_TAG}

N_PROC=1

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
--predictor ${PREDICTOR} \
--save ${SAVE} \
--iterations ${ITER} \
--n_doe ${N_DOE} \
--n_iter ${N_ITER} \
--anchor_levels ${ANCHOR_LEVELS} \
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

metric_task_stamp "${SAVE}"

# QEFT outlier-column axis (wbits search): searchable per-layer FP16 outlier
# counts + the multi-rank outlier dict from extract_outidx.py. The dataset/ranks
# in the path must match how extract_outidx.sh was run. (w_method stays hqq.)
if [ -n "${N_QEFT_COLUMN}" ]; then
    OUTLIER_PATH=/NAS/SJ/actquant/search/outlier/${MODEL_NAME}/w16_r${QEFT_RANK_TEXT}_${DATASET}/outlier.pth
    ARGS+=" --n_qeft_column ${N_QEFT_COLUMN} \
    --base_outlier_bits ${BASE_OUTLIER_BITS} \
    --outlier_path ${OUTLIER_PATH} \
    --n_outlier ${N_OUTLIER} "
fi

# --sensitivity_result_path ${SENSITIVITY_RESULT_PATH} \
if [ ${USE_KEY_TOKEN} == 'True' ]; then
    ARGS+=" --use_key_token \
    --trunc_len ${TRUNC_LEN} \
    --sliding_window ${SLIDING_WINDOW} \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --key_token_path ${KEY_TOKEN_PATH}"
fi

for g in "${KV_GROUP_SIZE[@]}"
do
    ARGS+=" --k_group_size ${g} "
done

for g in "${KV_GROUP_SIZE[@]}"
do
    ARGS+="--v_group_size ${g} "
done

if [ ${STRIDE} -gt 0 ]; then
    ARGS+=" --stride ${STRIDE} "
else
    ARGS+=" --quant_kv_output "
fi

if [ ${PREFILL_PROMPT} != 'True' ] && [ -n "${METRIC_TASK}" ] && [ ${LAST_TOKENS} -gt 0 ]; then
    # see PP_TAG above: a scored window with no prefill. The legacy path ties
    # --last_tokens to --prefill_prompt (PREFILL_PROMPT=False expects
    # LAST_TOKENS=0), so this is added only under METRIC_TASK.
    ARGS+=" --last_tokens ${LAST_TOKENS} --score ${SCORE} "
fi

if [ ${PREFILL_PROMPT} == 'True' ]; then
    if [ ${USE_KEY_TOKEN} == 'True' ] && [ ${SCORE} == 'last' ]; then
        echo "[search.sh] WARNING: USE_KEY_TOKEN=True with SCORE=last — key tokens are"
        echo "  ANDed with the last ${LAST_TOKENS} positions (measured coverage 4-20%, and 0 on"
        echo "  some documents, which drops them from the average). Set SCORE=full to score"
        echo "  every key token while keeping the prefill/answer split."
        [ -n "${METRIC_TASK}" ] && echo "  (SCORE came from METRIC_TASK=${METRIC_TASK};" \
            "the registry picked 'last' for it deliberately — changing SCORE makes this" \
            "run a DIFFERENT measurement than that name, so clear METRIC_TASK if you do.)"
    fi
    ARGS+=" --prefill_prompt --last_tokens ${LAST_TOKENS} --score ${SCORE} "
fi

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} search.py \
${ARGS}

# --stride ${STRIDE}
# --quant_kv_output \