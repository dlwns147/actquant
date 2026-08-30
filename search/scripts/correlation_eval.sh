DEVICES=${1:-0}
IDX=${2:?"need IDX (row index in archs.csv)"}
SAVE_ARG=${3:-}
METRICS=${4:-all}

TODAY=$(date +%y%m%d%H%M)
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# auto-discover newest correlation save dir if not supplied
if [ -z "${SAVE_ARG}" ]; then
    SAVE_ARG=$(ls -dt save/correlation/*/ 2>/dev/null | head -1 | sed 's:/*$::')
    if [ -z "${SAVE_ARG}" ]; then
        echo "ERROR: no save/correlation/* dir found; run scripts/correlation_sample.sh first or pass SAVE_DIR explicitly."
        exit 1
    fi
    echo "[auto] SAVE_DIR=${SAVE_ARG}"
fi
SAVE=${SAVE_ARG}
if [ ! -f "${SAVE}/archs.csv" ]; then
    echo "ERROR: ${SAVE}/archs.csv missing — run scripts/correlation_sample.sh first."
    exit 1
fi

# ── Model / quant config (MUST match the run that produced archs.csv) ──
MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=bfloat16
CONFIG=config/llama.json

W_METHOD=hqq
W_METHOD_TEXT=hqq
W_BITS="2 3 4"
AXIS=1
W_GROUP_SIZE=128

KV_METHOD=kivi
K_BITS="2 4"
K_BITS_TEXT="24"
V_BITS="2 4"
V_BITS_TEXT="24"
K_GROUP_SIZE=("128" "128")
V_GROUP_SIZE=("128" "128")

RESIDUAL_LENGTH=128
# Attention-sink (KVSink): keep first S KV tokens FP. 0=off. Match the eval config.
ATTN_SINK=0
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token

# ThinK pruning_dim options (# of head_dim channels to prune; 0 = no prune).
# Each arch's per-layer assignment comes from the kvdim_expr archive →
# archs.csv → arch['p']; this list is just the scalar fallback for layers
# without a per-arch override. Match the values used at sampling/search time.
K_PRUNING_DIM="0 16 32 48 64"
V_PRUNING_DIM="0 16 32 48 64"

QMODEL_PATHS_LIST=()
for B in ${W_BITS}; do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

# The bank directory name ends in DTYPE. On this box Llama-3.1-8B-Instruct is
# stored as _bfloat16 ONLY (Llama-2 has _float16), so a DTYPE/bank mismatch used
# to surface deep inside the HQQ loader; fail here with the path instead.
if [ "${W_METHOD}" = "hqq" ]; then
    for P in "${QMODEL_PATHS_LIST[@]}"; do
        if [ ! -d "${P}" ]; then
            echo "ERROR: HQQ bank not found: ${P}"
            echo "       DTYPE=${DTYPE} decides the suffix; available banks:"
            ls -d /SSD/hqq/${MODEL_NAME}_* 2>/dev/null | sed 's/^/         /'
            exit 1
        fi
    done
fi

SEED=0
N_TOKEN=16384

# Optional: aggregate/evaluate a SPECIFIC measurement folder instead of the one
# derived from the config below (see the header). Leave empty for the default.
MEASURE_DIR=""

# ── key tokens ─────────────────────────────────────────────────────────────
# The DIRECTORY the archives live in, not a single archive: correlation.py
# derives the root per metric as
#   <dir>/kt_eval-<evaluator>_tgt-<target>_<layout>/<corpus>_<protocol>_s<seed>
# with the evaluator read from the metric NAME (_q72b / _q7b / _l8b), so one
# setting serves every key-token metric and none can be paired with the wrong
# evaluator's archive. Set to '' to SKIP them all — correlation.py drops those
# metrics, prints which ones, and measures the rest. A path that IS set but
# holds no matching archive stays an error (a wrong answer, not a missing one).
KEY_TOKEN_PATH=key_token

# ── needle_nll knobs (kept small: 8 prompts × 2048 ctx ≈ 16k tokens, ~3s) ──
# NEEDLE_TASK: harder than niah_single_1 — multikey_2 uses a haystack of
# look-alike distractor needles. Other valid: niah_single_{1,2,3},
# niah_multikey_{1,3}.
NEEDLE_TASK=niah_multikey_2
NEEDLE_N_SAMPLE=8
NEEDLE_SEQLEN=2048

# ── LongBench / RULER artefact paths ──
LONGBENCH_CONFIG=utils/longbench_config
LONGBENCH_RESULT_PATH=${SAVE}/longbench_${IDX}
LONGBENCH_E_RESULT_PATH=${SAVE}/longbench_e_${IDX}

RULER_TASK="niah_single_1"
RULER_YAML_PATH=utils/ruler_utils
# RULER context length(s) — INDEPENDENT of N_TOKEN (which is only the
# memory-accounting token count for get_net_info). Space-separated values run
# one full RULER_SAMPLE sweep per length; scores are then reported per length
# as <task>_len<L> and raw artefacts go to <RULER_RESULT_PATH>/len<L>/.
#   e.g. RULER_LENGTH="4096 8192 16384"   (cost scales linearly with lengths)
RULER_LENGTH="16384"
RULER_SAMPLE=5
RULER_BATCH_SIZE=1
# Per generated RULER token, store the top-K next-token candidates (raw logits →
# log-probs) + the chosen token's log-prob + the top1−top2 margin, so a wrong
# answer can be read as "how wrong". 0 = off. MEASURED cost (niah_single_1
# @4096): peak GPU memory unchanged (17.00GB both), no wall-clock penalty, and
# 306 → 1338 bytes per row (~153 B per generated token; a 13-task × 50-sample
# sweep adds ~0.7 MB). LongBench is NOT wired for this yet.
RULER_TOPK_LOGITS=5
# Same record for LongBench generations. OFF by default here because this
# harness sweeps many IDX. MEASURED on a real prediction dir (2200 examples,
# 225,455 generated tokens = 66.5% of the max_gen cap):
#   multi_news 14.2MB | lcc 5.0 | repobench-p 5.0 | samsum 3.7 | qmsum 3.0
#   trec 2.0 | triviaqa 1.0 | qasper 0.8   ->  34.5 MB per pass at k=5
# i.e. ~6.9 GB over a 200-idx sweep, vs 0.7 MB for the whole RULER sweep.
# Transient GPU add is max_gen x vocab x 2B = 131 MB at max_gen 512.
# post_search.py defaults this ON (it benchmarks one final arch).
LONGBENCH_TOPK_LOGITS=0
# Compact tag for the result dir: "4096 8192" → "4096-8192".
RULER_LEN_TAG=$(echo "${RULER_LENGTH}" | tr -s ' ' '-')
RULER_RESULT_PATH=${SAVE}/ruler_${IDX}_len${RULER_LEN_TAG}_s${RULER_SAMPLE}

ARGS="--mode eval \
--gpu_id ${DEVICES} \
--idx ${IDX} \
--save ${SAVE} \
--metrics ${METRICS} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--w_method ${W_METHOD} \
--kv_method ${KV_METHOD} \
--config ${CONFIG} \
--dtype ${DTYPE} \
--w_bits ${W_BITS} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--attn_sink ${ATTN_SINK} \
--k_quant_scheme ${K_QUANT_SCHEME} \
--v_quant_scheme ${V_QUANT_SCHEME} \
--n_token ${N_TOKEN} \
--seed ${SEED} \
--needle_task ${NEEDLE_TASK} \
--needle_n_sample ${NEEDLE_N_SAMPLE} \
--needle_seqlen ${NEEDLE_SEQLEN}"

[ -n "${MEASURE_DIR}" ] && ARGS+=" --measure_dir ${MEASURE_DIR}"

for g in "${K_GROUP_SIZE[@]}"; do ARGS+=" --k_group_size ${g} "; done
for g in "${V_GROUP_SIZE[@]}"; do ARGS+=" --v_group_size ${g} "; done

[ -n "${K_PRUNING_DIM}" ] && ARGS+=" --k_pruning_dim ${K_PRUNING_DIM}"
[ -n "${V_PRUNING_DIM}" ] && ARGS+=" --v_pruning_dim ${V_PRUNING_DIM}"

[ "${W_METHOD}" = "hqq" ] && ARGS+=" --quant_model_paths ${QMODEL_PATHS} "
[ -n "${KEY_TOKEN_PATH}" ] && ARGS+=" --key_token_path ${KEY_TOKEN_PATH}"

# ── Benchmark toggles ──
# Set to 1 to run the benchmark. Path/param args are always passed (they're
# harmless when the corresponding --ruler / --longbench / --longbench_e
# toggle is off), so just flip these switches.
RUN_RULER=0
RUN_LONGBENCH=0
RUN_LONGBENCH_E=0

# Per-example GENERATIONS are always written (no knob):
#   RULER     → <RULER_RESULT_PATH>[/len<L>]/per_example_s${SEED}.jsonl
#               (per seed — the seed decides which samples are generated)
#   LongBench → <LONGBENCH_RESULT_PATH>/pred[_e]/<dataset>.jsonl
# Each row carries the generation, the references, the per-example score, token
# counts, input_sha256, provenance (run_id / arch_sha8 / idx) and — for RULER —
# the per-token top-K candidates (RULER_TOPK_LOGITS below), so a row can
# never be misattributed even if files are merged later. The prompts themselves
# are NOT stored: they are regenerable from (dataset, _id) / (seed, task,
# sample_index) and the hash proves the regenerated prompt matches.
# Each artefact directory also gets a meta.json (config it was produced under);
# re-running the SAME config refreshes it in place, a DIFFERENT config moves the
# old directory to <parent>/archive/<ts>/ instead of interleaving files.

# LongBench / LongBench-E params (always passed; only used if toggled on)
ARGS+=" --longbench_config ${LONGBENCH_CONFIG} --longbench_result_path ${LONGBENCH_RESULT_PATH} --longbench_e_result_path ${LONGBENCH_E_RESULT_PATH} --longbench_topk_logits ${LONGBENCH_TOPK_LOGITS}"
# RULER params (always passed; only used if toggled on)
ARGS+=" --ruler_task ${RULER_TASK} --ruler_yaml_path ${RULER_YAML_PATH} --ruler_length ${RULER_LENGTH} --ruler_sample ${RULER_SAMPLE} --ruler_batch_size ${RULER_BATCH_SIZE} --ruler_result_path ${RULER_RESULT_PATH} --topk_logits ${RULER_TOPK_LOGITS}"

[ "${RUN_RULER}"        = "1" ] && ARGS+=" --ruler"
[ "${RUN_LONGBENCH}"    = "1" ] && ARGS+=" --longbench"
[ "${RUN_LONGBENCH_E}"  = "1" ] && ARGS+=" --longbench_e"

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} \
    --num_machines=1 --main_process_port=${PORT_NUM} \
    correlation.py ${ARGS}
