DEVICES=${1:-0}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# ── 2nd-stage joint W×eff_kvbits search (second_search_new.py), SAMPLER=psi:
#    full-pool prediction → PSI (predicted staircase improvement: how much of the measured
#    record staircase over the (wbits, eff_kv) budget plane a cell would remove) → cost-aware
#    greedy, so FEW W builds get MANY, UNEQUAL KV cells; leftovers go to KV-extreme skeleton
#    + even coverage inside the opened rows.
#    BUDGETS ONLY (this file has exactly one non-budget knob, SAMPLER):
#      DOE       = N_DOE archs over DOE_BUILDS distinct W allocations (evenly spaced across
#                  the wbits range), i.e. N_DOE/DOE_BUILDS KV cells per W — 1 is legal and is
#                  the no-multi-KV shape. KV within a row is random unless DOE_KV_EXTREMES
#                  pins the two pool extremes. The four budget-box corners are always
#                  measured. COMPANION_KV does NOT govern the DOE.
#      per iter  = up to N_BUILDS W builds and N_BUILDS × COMPANION_KV KV cells; per-row
#                  counts are emergent, and the greedy stops early when no cell still beats
#                  the record (the shortfall is logged as "[short of N]").
#      SAMPLER   = psi (default) | front (the previous sampler, A/B baseline — it starves on
#                  real pools) | product (uniform block-product CONTROL ARM, cost-matched:
#                  any PSI gain must beat it, compared in GPU-HOURS, not evals).
#    Everything else that used to be a flag is a constant in the class or derived from the
#    run (e.g. the W-build cost is measured, not configured). U / ΔU are logged per iteration
#    so a saturated run is visible; the run always completes ITERATIONS.

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=bfloat16
CONFIG=config/llama.json

W_EXPR=save/search/think/2606302046_Llama-3.1-8B-Instruct_wbits_kivi_sk8_w234kv4_gs128_r128_kd0_vd0_obj_2_5_st128_pp512/iter_200.stats
EFF_KV_EXPR=save/search/think/2606302047_Llama-3.1-8B-Instruct_eff_kvbits_kivi_think_sk8_w4kv234_gs3264128x2_128_r128_kd0to64x16_vd0to64x16_obj_0.1_5_st128_pp512/iter_200.stats

for VAR_NAME in W_EXPR EFF_KV_EXPR; do
    VAR_VALUE="${!VAR_NAME}"
    if [ -n "${VAR_VALUE}" ] && [[ "${VAR_VALUE}" != *"${MODEL_NAME}"* ]]; then
        echo "ERROR: ${VAR_NAME} does not contain MODEL_NAME (${MODEL_NAME}): ${VAR_VALUE}"; exit 1
    fi
done

# W_METHOD=awq
W_METHOD=hqq

W_BITS="2 3 4"; W_GROUP_SIZE=128; AXIS=1
KV_METHOD="kivi think"

SURROGATE=rbf
SURROGATE_KERNEL=tps
SURROGATE_INPUT=genome
if [ "${W_METHOD}" == "awq" ]; then   # awq branch mirrors second_search.sh (2608 input A/B)
    W_METHOD=hqq                      # AWQ-setting pipeline for fast-eval debugging
    # SURROGATE=sqrty_ard_gp            # plstyp input + 2 stage-1-y warm-start features
    # SURROGATE_KERNEL=""

    SURROGATE=rbf
    SURROGATE_KERNEL=tps
    SURROGATE_INPUT=plstyp

    N_DOE=1000                  # TOTAL DOE archs
    DOE_BUILDS=25               # distinct W allocations in the DOE -> 1000/25 = 40 KV cells each
    ITERATIONS=15               # search iterations
    SAVE_ITER=1
    COMPANION_KV=40             # avg KV cells per build (total/iter = N_BUILDS × this; time knob)
    N_BUILDS=5                  # max W builds per iteration (the row cap)
fi
W_METHOD_TEXT=${W_METHOD}
KV_METHOD_TEXT=${KV_METHOD// /_}
QMODEL_PATHS=""
if [ "${W_METHOD}" == "hqq" ]; then   # pre-quantized banks are HQQ-only
    QMODEL_PATHS_LIST=()
    for B in ${W_BITS}; do
        QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
    done
    QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

    N_DOE=1000                  # TOTAL DOE archs
    DOE_BUILDS=100              # distinct W allocations in the DOE -> 1000/100 = 10 KV cells each
    ITERATIONS=200               # search iterations
    SAVE_ITER=10                 # dump iter_<it>.stats + iter_<it>.png (via --debug) every SAVE_ITER iters (and the last)
    COMPANION_KV=10             # avg KV cells per build (total/iter = N_BUILDS × this; time knob)
    N_BUILDS=5                  # max W builds per iteration (the row cap)
fi

SURR_TAG=${SURROGATE}; [ "${SURROGATE_INPUT}" != "genome" ] && SURR_TAG+=${SURROGATE_INPUT}
[ -n "${SURROGATE_KERNEL}" ] && SURR_TAG+=${SURROGATE_KERNEL}   # e.g. rbftps / sqrty_ard_gpplstyprq
SAMPLER=psi                 # psi | front | product   (the ONLY non-budget knob)
DOE_KV_EXTREMES=False       # pin both pool-extreme KV blocks in every DOE W row (else random
                            # KV per row). Ignored when N_DOE/DOE_BUILDS is 1. The four
                            # budget-box corners are measured either way.



SEED=0

DEBUG=True                  # save per-iter figures: iter_<it>.png (save_viz) + iter_<it>_nd.png
FRONT_EPS_REL=0.05          # ε-band width (the pool IS the search space)
WORKER_RECYCLE=32

ATTN_SINK=8
N_TOKEN=0

# DOE_RESULTS: curated pre-measured DOE dir (READ-ONLY; *_it<N>_* always ignored)
DOE_RESULTS=""
RESUME=""

LOSS_FUNC=jsd
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
SEQLEN=2048
MIN_SEQLEN=0
RESIDUAL_LENGTH=128
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token
STRIDE=32
PREFILL_PROMPT=True
LAST_TOKENS=128

# ── Named calibration metric (the ONE registry, utils/metric_specs.py) ─────
# METRIC_TASK=<name> takes the measurement protocol from the registry -- the same names
# correlation.py --metrics, post_search.py --metric_tasks and search.sh's own
# METRIC_TASK resolve -- so a 2nd-stage archive states which registered metric it optimised, and is
# comparable with the 1st-stage archives and the correlation table by construction
# instead of by retyping DATASET/STRIDE/LAST_TOKENS identically in every script.
# The resolution happens in the SHELL: second_search_new.py takes raw knobs and knows nothing about
# metric names, so this one lookup feeds both the arg list and the SAVE-dir tags. A
# knob set by hand above is OVERRIDDEN by the name and the override is printed -- the
# two are alternatives, not layers. Name + spec hash -> <SAVE>/metric_task.json.
# This entry point measures a LOSS objective with no key-token weighting, so PPL and
# key-token metrics are refused (--loss_only) -- use correlation.py / post_search.py for
# those. Empty = the hand-set knobs above (unchanged legacy behaviour).
# `python -m utils.metric_specs` lists every valid name.
METRIC_TASK=""
# METRIC_TASK=wt2_jsd_pp128_s32_chat      # == the hand-set knobs above
source "$(dirname "${BASH_SOURCE[0]}")/metric_task.sh"
metric_task_apply "${METRIC_TASK}" "${MODEL_NAME}" "" --loss_only

SINK_TAG=""; [ ${ATTN_SINK} -ne 0 ] && SINK_TAG="_sk${ATTN_SINK}"
PP_TAG="";   [ "${PREFILL_PROMPT}" == "True" ] && PP_TAG="_pp${LAST_TOKENS}"
# registry tasks that score a window with NO prefill/answer split (e.g.
# gov_jsd_lt128): not _pp, but the window still belongs in the dir name.
[ "${PREFILL_PROMPT}" != "True" ] && [ -n "${METRIC_TASK}" ] && [ ${LAST_TOKENS} -gt 0 ] && PP_TAG="_lt${LAST_TOKENS}"

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

source "$(dirname "${BASH_SOURCE[0]}")/metric_tag.sh"
MTAG=$(metric_tag_from_knobs "${DATASET##chat*:}" "${LOSS_FUNC}" "loss" \
                             "${N_SAMPLE}" "${SEQLEN}" "${MIN_SEQLEN:-0}")

# sampler in the dir name only when it is NOT the default, so psi runs keep the existing
# naming and an A/B arm can never overwrite them (SINK_TAG/PP_TAG pattern).
SMP_TAG=""; [ "${SAMPLER}" != "psi" ] && SMP_TAG="_${SAMPLER}"

SAVE=save/second_search_new/${TODAY}_${MODEL_NAME}_ndf${SMP_TAG}_${W_METHOD_TEXT}_${KV_METHOD_TEXT}_${SURR_TAG}_doe${DOE_BUILDS}x${N_DOE}_it${ITERATIONS}b${N_BUILDS}_ckv${COMPANION_KV}_eps${FRONT_EPS_REL}_st${STRIDE}${PP_TAG}${SINK_TAG}${MTAG}${CHAT_TAG}_s${SEED}

echo "SECOND-SEARCH-NEW (${SAMPLER}) -> ${SAVE}"

ARGS="--config ${CONFIG} \
--model_name ${MODEL_NAME} \
--w_expr ${W_EXPR} \
--eff_kv_expr ${EFF_KV_EXPR} \
--surrogate ${SURROGATE} \
--surrogate_input ${SURROGATE_INPUT} \
--sampler ${SAMPLER} \
--n_doe ${N_DOE} \
--doe_builds ${DOE_BUILDS} \
--iterations ${ITERATIONS} \
--n_iter ${N_BUILDS} \
--companion_kv ${COMPANION_KV} \
--attn_sink ${ATTN_SINK} \
--n_token ${N_TOKEN} \
--front_eps_rel ${FRONT_EPS_REL} \
--seed ${SEED} \
--save_iter ${SAVE_ITER} \
--save ${SAVE}"
[ -n "${SURROGATE_KERNEL}" ] && ARGS+=" --surrogate_kernel ${SURROGATE_KERNEL}"
[ "${DEBUG}" == "True" ] && ARGS+=" --debug"
[ "${DOE_KV_EXTREMES}" == "True" ] && ARGS+=" --doe_kv_extremes"

[ -n "${QMODEL_PATHS}" ] && ARGS+=" --quant_model_paths ${QMODEL_PATHS}"
GPU_ID=${DEVICES}
if [ "${W_METHOD}" == "awq" ]; then
    GPU_ID=${DEVICES%%,*}                                     # main process; workers own the rest
    EVAL_WORKERS=$(echo ${DEVICES} | awk -F',' '{print NF}')  # one worker per DEVICES entry
    ARGS+=" --eval_workers ${EVAL_WORKERS} --worker_gpus ${DEVICES} --worker_recycle ${WORKER_RECYCLE}"
    [ -n "${DOE_RESULTS}" ] && ARGS+=" --doe_results ${DOE_RESULTS}"
fi

ARGS+=" --gpu_id ${GPU_ID} \
--model_path ${MODEL_PATH} \
--dtype ${DTYPE} \
--w_method ${W_METHOD} \
--kv_method ${KV_METHOD} \
--w_bits ${W_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
--dataset ${DATASET} \
--n_sample ${N_SAMPLE} \
--seqlen ${SEQLEN} \
--min_seqlen ${MIN_SEQLEN:-0} \
--loss_func ${LOSS_FUNC}"

if [ ${STRIDE} -gt 0 ]; then
    ARGS+=" --stride ${STRIDE} "
fi
if [ ${PREFILL_PROMPT} == 'True' ]; then
    ARGS+=" --prefill_prompt --last_tokens ${LAST_TOKENS} "
elif [ -n "${METRIC_TASK}" ] && [ ${LAST_TOKENS} -gt 0 ]; then
    # see PP_TAG: a scored window with no prefill. The legacy path ties --last_tokens to
    # --prefill_prompt, so this is added only under METRIC_TASK.
    ARGS+=" --last_tokens ${LAST_TOKENS} "
fi
if [ -n "${METRIC_TASK}" ]; then
    # --score is not part of this script's legacy arg list (it only ever ran the default
    # 'last'), so it is passed explicitly when a name is what set it.
    ARGS+=" --score ${SCORE} "
    metric_task_stamp "${SAVE}"
fi
[ -n "${RESUME}" ] && ARGS+=" --resume ${RESUME}"

if [ "${W_METHOD}" == "awq" ]; then
    python -u second_search_new.py ${ARGS}
else
    CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=1 --num_machines=1 \
        --main_process_port=${PORT_NUM} second_search_new.py ${ARGS}
fi
