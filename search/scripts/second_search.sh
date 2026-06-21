DEVICES=${1:-0}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# ── 2nd-stage JOINT (W × eff_kvbits) NAS — HQQ-based, NSGA-III (second_search.py).
#    genome = (W-block, KV-block) assembled from the 1st-stage per-axis Pareto fronts.
#    CROSSOVER unit = whole axis block (W or KV swap; additive W⊥KV).
#    MUTATION = 1st-stage-weighted (per-gene lever strength, eps-floor → coverage kept;
#               direction free → NON-monotone allocations stay explorable).
#    PREDICTOR = arch-input (genome), like 1st-stage search.py (frozen genes dropped).
#    Objectives = (loss, wbits, eff_kvbits).
#
#    MODE=proxy : CPU smoke — loss = 1st-stage front additive interpolation (no GPU/model).
#    MODE=hqq   : real HQQ measurement (needs LlamaEvaluator wired into second_search.py).

# MODE=proxy
MODE=hqq

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=bfloat16
CONFIG=config/llama.json

# ── alternative models (second_search is model-general: options + budget box auto-derive
#    from the archives — verified on Qwen2.5-7B). Swap W_EXPR/EFF_KV_EXPR to match.
#    DTYPE stays bfloat16 (the HQQ banks are bfloat16 on disk).
# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# CONFIG=config/mistral.json

# ── 1st-stage per-axis frontier archives (a dir → latest iter_N.stats, or an explicit .stats) ──
W_EXPR=save/search/think/2606070017_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_128stride_pp512
EFF_KV_EXPR=save/search/think/2606181423_Llama-3.1-8B-Instruct_eff_kvbits_kivi_sk8_w4kv234_gs3264128x2_128_r128_kd0-64x5_vd0-64x5_obj_0.1_5_st128_pp512

for VAR_NAME in W_EXPR EFF_KV_EXPR; do
    VAR_VALUE="${!VAR_NAME}"
    if [ -n "${VAR_VALUE}" ] && [[ "${VAR_VALUE}" != *"${MODEL_NAME}"* ]]; then
        echo "ERROR: ${VAR_NAME} does not contain MODEL_NAME (${MODEL_NAME}): ${VAR_VALUE}"; exit 1
    fi
done

# ── HQQ banks (MODE=hqq only) ──
W_METHOD=hqq
W_BITS="2 3 4"; W_GROUP_SIZE=128; AXIS=1
KV_METHOD="kivi think"
QMODEL_PATHS_LIST=()
for B in ${W_BITS}; do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

# ── search hyperparameters (loop mirrors search.py: DOE → fit → next → eval → iter_N.stats) ──
SURROGATE=rbf      # arch-input predictor: rbf (needs N_DOE > #active genes ≈ 360) / gp / ard_gp / carts
POP=200             # NSGA-III pop (≥ das-dennis 3-obj/12-part = 91 ref dirs; 200 as in search.sh)
N_DOE=500          # DOE measured archs (≥ #active genes for rbf)
ITERATIONS=200       # search iterations (fit ↔ measure)
N_ITER=50          # candidates measured per iteration
SEED=0
SAVE_ITER=10       # dump iter_<it>.stats + iter_<it>.png (via --debug) every SAVE_ITER iters (and the last)

# comp_obj budget box: AUTO-derived from the input archives' achievable comp range
# (second_search.py reads it off the W / eff_kvbits pools). To NARROW an axis below
# the input range, pass --comp_obj_min/--comp_obj_max explicitly in ARGS.
ATTN_SINK=8
N_TOKEN=0

# ── building-block selector from 1st-stage archives (FINALIZED via search sweep).
#    Pool = adaptive-ε band → structural-diversity. NO window/bin thinning: a sweep showed
#    thinning HURTS hv (band→div 0.265 > window→div 0.237 > hardbin→div 0.20) — maximin
#    diversity wants the FULL band as material; pre-thinning throws away what it would pick.
FRONT_EPS_REL=0.3   # adaptive ε band = front_jsd·(1+rel): scale-free, auto-wider in the corner
DIV_K=200           # structural-diversity blocks/axis (maximin; richest crossover — dominant for hv)

# ── hqq-mode eval primitives (named like search.sh; ignored in MODE=proxy). These MUST match
#    the 1st-stage search-time measurement so the surrogate trains on the same JSD definition.
LOSS_FUNC=jsd
DATASET=wikitext2
N_SAMPLE=128
SEQLEN=2048
RESIDUAL_LENGTH=128
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token
STRIDE=128            # >0 = stride-aware (use_cache) eval; 0 = single forward pass
PREFILL_PROMPT=True  # prefill the prompt + stride only the answer span (matches real-decode KV)
LAST_TOKENS=512      # loss on the last N tokens only (answer-phase JSD; needs PREFILL_PROMPT)

SAVE=save/second_search/${TODAY}_${MODEL_NAME}_joint_${MODE}_${SURROGATE}_doe${N_DOE}_it${ITERATIONS}n${N_ITER}_sk${ATTN_SINK}_s${SEED}
echo "SECOND-SEARCH -> ${SAVE}"

ARGS="--config ${CONFIG} \
--model_name ${MODEL_NAME} \
--w_expr ${W_EXPR} \
--eff_kv_expr ${EFF_KV_EXPR} \
--mode ${MODE} \
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

if [ ${MODE} == "hqq" ]; then
    ARGS+=" --gpu_id ${DEVICES} --model_path ${MODEL_PATH} --dtype ${DTYPE} --w_method ${W_METHOD} \
--kv_method ${KV_METHOD} --quant_model_paths ${QMODEL_PATHS} \
--w_bits ${W_BITS} --w_group_size ${W_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} --k_quant_scheme ${K_QUANT_SCHEME} --v_quant_scheme ${V_QUANT_SCHEME} \
--dataset ${DATASET} --n_sample ${N_SAMPLE} --seqlen ${SEQLEN} --loss_func ${LOSS_FUNC}"

    if [ ${STRIDE} -gt 0 ]; then
        ARGS+=" --stride ${STRIDE} "
    fi
    if [ ${PREFILL_PROMPT} == 'True' ]; then
        ARGS+=" --prefill_prompt --last_tokens ${LAST_TOKENS} "
    fi

    CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=1 --num_machines=1 \
        --main_process_port=${PORT_NUM} second_search.py ${ARGS}
else
    # proxy mode is pure-CPU (no model / GPU)
    python second_search.py ${ARGS}
fi
