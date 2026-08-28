DEVICES=${1:-0}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=bfloat16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# CONFIG=config/mistral.json

# W_EXPR=save/search/think/2606070017_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_128stride_pp512
# EFF_KV_EXPR=save/search/think/2606181423_Llama-3.1-8B-Instruct_eff_kvbits_kivi_sk8_w4kv234_gs3264128x2_128_r128_kd0-64x5_vd0-64x5_obj_0.1_5_st128_pp512
W_EXPR=save/search/think/2606302046_Llama-3.1-8B-Instruct_wbits_kivi_sk8_w234kv4_gs128_r128_kd0_vd0_obj_2_5_st128_pp512/iter_200.stats
EFF_KV_EXPR=save/search/think/2606302047_Llama-3.1-8B-Instruct_eff_kvbits_kivi_think_sk8_w4kv234_gs3264128x2_128_r128_kd0to64x16_vd0to64x16_obj_0.1_5_st128_pp512/iter_200.stats

for VAR_NAME in W_EXPR EFF_KV_EXPR; do
    VAR_VALUE="${!VAR_NAME}"
    if [ -n "${VAR_VALUE}" ] && [[ "${VAR_VALUE}" != *"${MODEL_NAME}"* ]]; then
        echo "ERROR: ${VAR_NAME} does not contain MODEL_NAME (${MODEL_NAME}): ${VAR_VALUE}"; exit 1
    fi
done

# W_METHOD=hqq
W_METHOD=awq
W_BITS="2 3 4"; W_GROUP_SIZE=128; AXIS=1
KV_METHOD="kivi think"
W_METHOD_TEXT=${W_METHOD}
KV_METHOD_TEXT=${KV_METHOD// /_}
QMODEL_PATHS=""
if [ "${W_METHOD}" == "hqq" ]; then   # pre-quantized banks are HQQ-only (AWQ quantizes per-arch in sample())
    QMODEL_PATHS_LIST=()
    for B in ${W_BITS}; do
        QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
    done
    QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")
fi

SURROGATE=rbf      # arch-input predictor: rbf (needs N_DOE > #active genes ~360) / gp / ard_gp / carts
POP=200             # NSGA-III pop (>= das-dennis 3-obj/12-part = 91 ref dirs; 200 as in search.sh)
N_DOE=500          # DOE measured archs (>= #active genes for rbf)
ITERATIONS=200       # search iterations (fit <-> measure)
N_ITER=50          # candidates measured per iteration
SEED=0
SAVE_ITER=10       # dump iter_<it>.stats + iter_<it>.png (via --debug) every SAVE_ITER iters (and the last)
N_PROC=1           # data-parallel eval ranks (search() is multi-process safe). For N_PROC>1 set
SURROGATE_INPUT=genome
WORKER_RECYCLE=32

# DOE_RESULTS: curated clean DOE dir (read READ-ONLY except a one-time create if empty;
# iteration-cache files *_it<N>_* are always ignored). Empty = measure a fresh DOE.
DOE_RESULTS=""
# DOE_RESULTS=save/awq_second_search/2607240358_Llama-3.1-8B-Instruct_awq_premeasured_kivi_think_n128_st128_pp512_sk8_wikitext2/

if [ "${W_METHOD}" == "awq" ]; then
    #W_METHOD=hqq # AWQ-setting pipeline for fast-eval debugging
    SURROGATE_INPUT=plstyp
    # sqrty_ard_gp (2608): sqrt-target head; with the one-hot 8y plstyp embedding
    # cell-mean 0.414->0.646 @DOE100 vs plain ard_gp (tests/awq_alloc_flip/adoption_check3.py)
    SURROGATE=sqrty_ard_gp
    N_DOE=100
    ITERATIONS=15
    N_ITER=20
    SAVE_ITER=1
fi

ATTN_SINK=8
N_TOKEN=0

GRID_SEED=True

# COMPANION_KV=0       # WAFE: extra geometry-diverse KV archs attached per W-anchor at the subset
COMPANION_KV=10       # WAFE: extra geometry-diverse KV archs attached per W-anchor at the subset
COMPANION_METHOD=2d   # KV placement: 2d (DEFAULT) = predicted-Pareto filter (loss,wbits,eff_kv)

FRONT_EPS_REL=0.05
# FRONT_EPS_REL=0
DIV_K=0           # structural-diversity blocks/axis (maximin; richest crossover -- dominant for hv)

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
# gov_report/gsm8k 전용 필터(wikitext2/c4 로더는 무시). gov_report로 바꿀 때는
# DATASET=gov_report N_SAMPLE=8 SEQLEN=8192 MIN_SEQLEN=8192 처럼 같이 올려야 한다.
MIN_SEQLEN=0
RESIDUAL_LENGTH=128
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token
STRIDE=128
PREFILL_PROMPT=True
LAST_TOKENS=512

SINK_TAG=""; [ ${ATTN_SINK} -ne 0 ] && SINK_TAG="_sk${ATTN_SINK}"
QEFT_TAG=""; [ "${USE_QEFT}" == "True" ] && QEFT_TAG="_qc${QEFT_RANK_TEXT}"
PP_TAG="";   [ "${PREFILL_PROMPT}" == "True" ] && PP_TAG="_pp${LAST_TOKENS}"
CAND_TAG=subset                                                 # down-select = subset (fixed)
[ "${GRID_SEED}" == "True" ] && CAND_TAG+=-st                   # staircase supply seeds on
[ "${COMPANION_KV}" -gt 0 ] && CAND_TAG+="-ckv${COMPANION_KV}${COMPANION_METHOD:0:3}"  # +companion KV sweep
SURROGATE_KERNEL=""   # kernel for the active SURROGATE (rbf: cubic/tps/linear · ard_gp: matern32/52/rbf/rq); ""=model default
SURR_TAG=${SURROGATE}; [ "${SURROGATE_INPUT}" != "genome" ] && SURR_TAG+=${SURROGATE_INPUT}  # e.g. rbfplstyp
[ -n "${SURROGATE_KERNEL}" ] && SURR_TAG+=${SURROGATE_KERNEL}   # e.g. rbfplstyptps

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
MTAG=$(metric_tag_from_knobs "${DATASET##chat*:}" "${LOSS_FUNC}" "${METRIC:-loss}" \
                             "${N_SAMPLE}" "${SEQLEN}" "${MIN_SEQLEN:-0}")

SAVE=save/second_search/${TODAY}_${MODEL_NAME}_joint_${W_METHOD_TEXT}${QEFT_TAG}_${KV_METHOD_TEXT}_${SURR_TAG}_doe${N_DOE}_it${ITERATIONS}n${N_ITER}p${POP}_${CAND_TAG}_eps${FRONT_EPS_REL}_dk${DIV_K}_st${STRIDE}${PP_TAG}${SINK_TAG}${MTAG}${CHAT_TAG}_s${SEED}

echo "SECOND-SEARCH -> ${SAVE}"

ARGS="--config ${CONFIG} \
--model_name ${MODEL_NAME} \
--w_expr ${W_EXPR} \
--eff_kv_expr ${EFF_KV_EXPR} \
--surrogate ${SURROGATE} \
--pop ${POP} \
--n_doe ${N_DOE} \
--iterations ${ITERATIONS} \
--n_iter ${N_ITER} \
--attn_sink ${ATTN_SINK} \
--n_token ${N_TOKEN} \
--front_eps_rel ${FRONT_EPS_REL} \
--div_k ${DIV_K} \
--seed ${SEED} \
--save_iter ${SAVE_ITER} \
--debug \
--save ${SAVE}"

[ "${GRID_SEED}" == "True" ] && ARGS+=" --grid_seed"
[ "${COMPANION_KV}" -gt 0 ] && ARGS+=" --companion_kv ${COMPANION_KV} --companion_method ${COMPANION_METHOD}"

ARGS+=" --surrogate_input ${SURROGATE_INPUT}"
[ -n "${SURROGATE_KERNEL}" ] && ARGS+=" --surrogate_kernel ${SURROGATE_KERNEL}"
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

if [ "${W_METHOD}" == "awq" ]; then
    [ -n "${RESUME}" ] && ARGS+=" --resume ${RESUME}"
    python -u second_search.py ${ARGS}
else
    CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 \
        --main_process_port=${PORT_NUM} second_search.py ${ARGS}
fi
