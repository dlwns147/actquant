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
#
# Run this once per IDX (parallelise across GPUs by launching multiple
# instances). result_<IDX>.json is written into SAVE_DIR; bash
# scripts/correlation_aggregate.sh merges everything into correlation.csv.

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

SEED=0
N_TOKEN=16384

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
RULER_LENGTH=${N_TOKEN}
RULER_SAMPLE=5
RULER_BATCH_SIZE=1
RULER_RESULT_PATH=${SAVE}/ruler_${IDX}_len${RULER_LENGTH}_s${RULER_SAMPLE}

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
