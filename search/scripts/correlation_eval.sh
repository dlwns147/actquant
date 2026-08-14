#!/usr/bin/env bash
# Usage: bash scripts/correlation_eval.sh <DEVICES> <IDX> [<SAVE_DIR>] [<METRICS>]
#   DEVICES    e.g. "0" or "0,1"
#   IDX        row index in archs.csv to evaluate
#   SAVE_DIR   correlation save dir (the one stage 1 wrote);
#              if omitted, picks the newest save/correlation/* directory.
#   METRICS    space- or comma-separated CALIBRATION metric keys (default
#              "all" = all PPL/loss metrics). Benchmarks (ruler/longbench/
#              longbench_e) are NOT valid here — toggle them with the
#              RUN_RULER / RUN_LONGBENCH / RUN_LONGBENCH_E variables below.
#              Calibration keys: c4_ppl wt2_jsd wt2_jsd_s512 wt2_jsd_pp512_s128
#                  wt2_jsd_lt128 needle_nll needle_nll_s512 needle_nll_pp512_s128
#                  gsm8k_jsd gov_jsd gov_jsd_s512 gov_jsd_pp512_s128 gov_jsd_lt128
#                  gsm8k_jsd_pp_s128 gov_jsd_kt gov_jsd_kt_s512
#              Decode-tight answer window (stride 8 over the last 32 tokens):
#                  wt2_jsd_pp32_s8 gov_jsd_pp32_s8
#                  needle_nll_pp32_s8 needle_jsd_pp32_s8
#              Longer-context PPL (own groups, no FP-teacher pass):
#                  wt2_ppl_sl8192 c4_ppl_sl8192
#                  wt2_ppl_sl8192_s512   (8k windows through the real KV cache)
#              Long-DOCUMENT PPL — 128 docs, selection seed pinned to 0, each
#              corpus at 2048 (domain control) and 8192:
#                  gov_ppl_sl2048 gov_ppl (+ _s512 _pp512_s128)
#                  nqa_ppl_sl2048 nqa_ppl (+ _s512)              narrativeqa
#                  qmsum_ppl_sl2048 qmsum_ppl (+ _s512)
#              (LongBench grades qmsum → qmsum_ppl* are REPORTING metrics; the
#               correlation is still measured, just listed in
#               correlation_contamination.txt as not-a-prediction-claim)
#              Answer-phase PPL: EVERY corpus above also has <base>_pp128_s32
#              (e.g. nqa_ppl_pp128_s32, wt2_ppl_sl8192_pp128_s32) — prefill +
#              32-token-chunk scoring over the last 128 tokens. The full-window
#              base task stays; the 512-token _s32 variant was dropped.
#              Long-doc JSD at the 128-doc default: gov_jsd_pp128_s32_n128_sl8192
#              COST per long-doc task = n_sample x seqlen forward tokens
#              (0.26M @2048 … 1.05M @8192); an answer-phase variant costs the
#              same as its base (the prefill dominates). For reference the
#              benchmarks this harness correlates against measure, per idx:
#              RULER 28.6 min / LongBench 150.1 min / LongBench-E 282.7 min.
#              The authoritative list is utils/metric_specs.py::METRIC_TASKS;
#              `--metrics all` runs every one of them.
#
# Run this once per IDX (parallelise across GPUs by launching multiple
# instances). Results go to SAVE_DIR/m_<config sha8>/result_<IDX>.json — ONE
# FOLDER PER MEASUREMENT CONFIG (model + w/kv method + bits + group sizes +
# residual + sink + quant schemes + seed), with the full config in that
# folder's meta.json. archs.csv stays at SAVE_DIR and is shared, so the same
# arch pool can be measured under several configs without them ever mixing:
# aggregate reads ONE folder, and rows inside it are comparable by
# construction. Benchmark artefact paths follow into the same folder.
# A SAVE_DIR that already holds result_*.json at its root keeps using the root
# (existing runs are untouched). bash scripts/correlation_aggregate.sh merges
# one folder into correlation.csv.
#
# Each measured value also stores a spec hash of its DEFINITION (the registry
# group+task): edit a group and the affected metric is re-measured on the next
# run instead of keeping a number that no longer means the same thing.

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
DTYPE=float16
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

# ── gov_jsd_kt key-token archive (set to '' to skip gov_jsd_kt) ──
KEY_TOKEN_PATH=key_token/Qwen2.5-72B-Instruct_gov_report_test_8sample_8192seqlen_8192min_256trunc_64sw_1alpha_-1beta

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
# counts, input_sha256 AND provenance (run_id / arch_sha8 / idx) so a row can
# never be misattributed even if files are merged later. The prompts themselves
# are NOT stored: they are regenerable from (dataset, _id) / (seed, task,
# sample_index) and the hash proves the regenerated prompt matches.
# Each artefact directory also gets a meta.json (config it was produced under);
# re-running the SAME config refreshes it in place, a DIFFERENT config moves the
# old directory to <parent>/archive/<ts>/ instead of interleaving files.

# LongBench / LongBench-E params (always passed; only used if toggled on)
ARGS+=" --longbench_config ${LONGBENCH_CONFIG} --longbench_result_path ${LONGBENCH_RESULT_PATH} --longbench_e_result_path ${LONGBENCH_E_RESULT_PATH}"
# RULER params (always passed; only used if toggled on)
ARGS+=" --ruler_task ${RULER_TASK} --ruler_yaml_path ${RULER_YAML_PATH} --ruler_length ${RULER_LENGTH} --ruler_sample ${RULER_SAMPLE} --ruler_batch_size ${RULER_BATCH_SIZE} --ruler_result_path ${RULER_RESULT_PATH}"

[ "${RUN_RULER}"        = "1" ] && ARGS+=" --ruler"
[ "${RUN_LONGBENCH}"    = "1" ] && ARGS+=" --longbench"
[ "${RUN_LONGBENCH_E}"  = "1" ] && ARGS+=" --longbench_e"

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} \
    --num_machines=1 --main_process_port=${PORT_NUM} \
    correlation.py ${ARGS}
