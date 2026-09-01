DEVICES=${1:-0}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# ── predicted-new-front 2nd-stage search (second_search_new.py): full-pool prediction →
#    predicted new Pareto front (pure dominance) → W picks by union W-gap filling →
#    KV cells = extremes + plane-gap spread over the picked rows' front cells.
#    Budgets only: ITERATIONS × N_BUILDS W builds; N_BUILDS × COMPANION_KV KV cells/iter
#    (COMPANION_KV = per-model time knob; per-row counts are emergent).

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

# W_METHOD=awq enters the AWQ-setting branch (plstyp+sqrty_ard_gp surrogate, worker pool);
# the flip inside it switches ONLY the eval method to hqq for fast-eval debugging of the
# AWQ configuration. Comment the flip out for a real AWQ run; W_METHOD=hqq here for the
# plain production-hqq (rbf+genome) run.
W_METHOD=awq
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
fi

SURR_TAG=${SURROGATE}; [ "${SURROGATE_INPUT}" != "genome" ] && SURR_TAG+=${SURROGATE_INPUT}
[ -n "${SURROGATE_KERNEL}" ] && SURR_TAG+=${SURROGATE_KERNEL}   # e.g. rbftps / sqrty_ard_gpplstyprq
DOE_BUILDS=25               # random DOE W builds, each × (2 KV extremes + random cells)
ITERATIONS=15               # search iterations
N_BUILDS=5                  # W builds per iteration (union W-gap filling)
COMPANION_KV=40             # avg KV cells per build (total/iter = N_BUILDS × this; time knob)
SEED=0
SAVE_ITER=1
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

SINK_TAG=""; [ ${ATTN_SINK} -ne 0 ] && SINK_TAG="_sk${ATTN_SINK}"
PP_TAG="";   [ "${PREFILL_PROMPT}" == "True" ] && PP_TAG="_pp${LAST_TOKENS}"

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

SAVE=save/second_search_new/${TODAY}_${MODEL_NAME}_ndf_${W_METHOD_TEXT}_${KV_METHOD_TEXT}_${SURR_TAG}_doe${DOE_BUILDS}_it${ITERATIONS}b${N_BUILDS}_ckv${COMPANION_KV}_eps${FRONT_EPS_REL}_st${STRIDE}${PP_TAG}${SINK_TAG}${MTAG}${CHAT_TAG}_s${SEED}

echo "SECOND-SEARCH-NEW (claim) -> ${SAVE}"

ARGS="--config ${CONFIG} \
--model_name ${MODEL_NAME} \
--w_expr ${W_EXPR} \
--eff_kv_expr ${EFF_KV_EXPR} \
--surrogate ${SURROGATE} \
--surrogate_input ${SURROGATE_INPUT} \
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
fi
[ -n "${RESUME}" ] && ARGS+=" --resume ${RESUME}"

if [ "${W_METHOD}" == "awq" ]; then
    python -u second_search_new.py ${ARGS}
else
    CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=1 --num_machines=1 \
        --main_process_port=${PORT_NUM} second_search_new.py ${ARGS}
fi
