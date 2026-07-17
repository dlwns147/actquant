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

W_EXPR=save/search/think/2606070017_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_128stride_pp512
EFF_KV_EXPR=save/search/think/2606181423_Llama-3.1-8B-Instruct_eff_kvbits_kivi_sk8_w4kv234_gs3264128x2_128_r128_kd0-64x5_vd0-64x5_obj_0.1_5_st128_pp512

for VAR_NAME in W_EXPR EFF_KV_EXPR; do
    VAR_VALUE="${!VAR_NAME}"
    if [ -n "${VAR_VALUE}" ] && [[ "${VAR_VALUE}" != *"${MODEL_NAME}"* ]]; then
        echo "ERROR: ${VAR_NAME} does not contain MODEL_NAME (${MODEL_NAME}): ${VAR_VALUE}"; exit 1
    fi
done

W_METHOD=hqq       # hqq = in-process eval (accelerate DP) | awq = AWQ-native arch-parallel eval pool
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

SURROGATE=rbf      # arch-input predictor: rbf (needs N_DOE > #active genes ≈ 360) / gp / ard_gp / carts
POP=200             # NSGA-III pop (≥ das-dennis 3-obj/12-part = 91 ref dirs; 200 as in search.sh)
N_DOE=500          # DOE measured archs (≥ #active genes for rbf)
ITERATIONS=200       # search iterations (fit ↔ measure)
N_ITER=50          # candidates measured per iteration
SEED=0
SAVE_ITER=10       # dump iter_<it>.stats + iter_<it>.png (via --debug) every SAVE_ITER iters (and the last)
N_PROC=1           # data-parallel eval ranks (search() is multi-process safe). For N_PROC>1 set
                   # DEVICES=0,1,... (one GPU/rank); needs #calibration batches >= N_PROC.

# ── AWQ-native mode (W_METHOD=awq) ────────────────────────────────────────────
# Per-arch AWQ build+eval ≈ 8.5 min → whole archs are farmed to a persistent
# per-GPU worker pool (utils/awq_pool.py). accelerate DP would DUPLICATE the
# build on every rank, so this mode launches plain `python -u` (num_processes=1
# is enforced); workers pin their own GPUs from DEVICES (absolute ids).
SURROGATE_INPUT=genome  # genome (legacy) / feat / cv. AWQ archives are SMALL (~500):
                        # genome-rbf collapses there (OOS rho 0.14 @N=88) → awq mode
                        # switches to cv (per-iter 5-fold winner, logs [surrogate_cv])
WORKER_RECYCLE=32       # recycle each worker after this many archs (run_awq inter-build leak)
SEED_RESULTS=""         # pre-measured result dirs (*specs*.json + *results*.jsonl) appended
                        # to the DOE archive. Protocol (stride/pp/last_tokens/sink/n_sample/
                        # dataset) AND W_METHOD of those measurements must match this run.
if [ "${W_METHOD}" == "awq" ]; then
    SURROGATE_INPUT=cv
    SEED_RESULTS="tests/awq_alloc_flip"          # 88 AWQ-measured archs (pilot + round-0)
    N_DOE=200; ITERATIONS=10; N_ITER=25          # 450 evals ≈ 16h on 4 GPUs @ ~509s/arch
fi

ATTN_SINK=8
N_TOKEN=0

# down-select = subset selector, mutation = band-conditional (P1), seeds = staircase (P2);
# losing variants (maximin/grid/hybrid/moo, --al_frac, global-draw/block-seed arms) were
# removed 2026-07 after the offline evidence + 2-seed A/B pilots.
GRID_SEED=True       # True = inject staircase even-supply genomes per box cell each iter


FRONT_EPS_REL=0.3   # adaptive ε band = front_jsd·(1+rel): scale-free, auto-wider in the corner
DIV_K=200           # structural-diversity blocks/axis (maximin; richest crossover — dominant for hv)

LOSS_FUNC=jsd
DATASET=wikitext2
N_SAMPLE=128
SEQLEN=2048
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
SURR_TAG=${SURROGATE}; [ "${SURROGATE_INPUT}" != "genome" ] && SURR_TAG+=${SURROGATE_INPUT}  # e.g. rbfcv
SAVE=save/second_search/${TODAY}_${MODEL_NAME}_joint_${W_METHOD_TEXT}${QEFT_TAG}_${KV_METHOD_TEXT}_${SURR_TAG}_doe${N_DOE}_it${ITERATIONS}n${N_ITER}p${POP}_${CAND_TAG}_eps${FRONT_EPS_REL}_dk${DIV_K}_st${STRIDE}${PP_TAG}${SINK_TAG}_s${SEED}

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

ARGS+=" --surrogate_input ${SURROGATE_INPUT}"
[ -n "${QMODEL_PATHS}" ] && ARGS+=" --quant_model_paths ${QMODEL_PATHS}"
GPU_ID=${DEVICES}
if [ "${W_METHOD}" == "awq" ]; then
    GPU_ID=${DEVICES%%,*}                                     # main process; workers own the rest
    EVAL_WORKERS=$(echo ${DEVICES} | awk -F',' '{print NF}')  # one worker per DEVICES entry
    ARGS+=" --eval_workers ${EVAL_WORKERS} --worker_gpus ${DEVICES} --worker_recycle ${WORKER_RECYCLE}"
    [ -n "${SEED_RESULTS}" ] && ARGS+=" --seed_results ${SEED_RESULTS}"
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
--loss_func ${LOSS_FUNC}"

if [ ${STRIDE} -gt 0 ]; then
    ARGS+=" --stride ${STRIDE} "
fi
if [ ${PREFILL_PROMPT} == 'True' ]; then
    ARGS+=" --prefill_prompt --last_tokens ${LAST_TOKENS} "
fi

if [ "${W_METHOD}" == "awq" ]; then
    # pool mode: plain unbuffered python (pool enforces num_processes=1); workers pin their
    # own GPUs from --worker_gpus. A full awq run is ~16h — launch DETACHED so it survives
    # the launching session:
    #   cd /NAS/SJ/actquant/search && setsid nohup bash scripts/second_search.sh 0,1,2,3 \
    #       > awq_run.log 2>&1 < /dev/null &
    # Resume after a kill: add RESUME=<save_dir>/iter_N.stats (archive+seeds restored from it).
    [ -n "${RESUME}" ] && ARGS+=" --resume ${RESUME}"
    python -u second_search.py ${ARGS}
else
    CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 \
        --main_process_port=${PORT_NUM} second_search.py ${ARGS}
fi

