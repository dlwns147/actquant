DEVICES=${1}
# METRICS can also come from argv: bash scripts/gen_key_token.sh 0 "gov_jsd_kt_q7b …"
METRICS_ARG=${2:-}
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# ── what the archive is FOR ────────────────────────────────────────────────
# Set METRICS to correlation.py metric names and the protocol (corpus,
# n_sample, seqlen, min_seqlen, trunc/sliding_window, alpha/beta, seed) is read
# from utils/metric_specs.py instead of the knobs below -- the protocol has to
# match EXACTLY or utils/loss.py rejects the archive by text hash.
# Metrics that differ only in last_tokens / stride / prefill_prompt / score
# share ONE archive (those are eval-time knobs; MEASURED: the same archive
# serves last_tokens 1024/512/256/128 and both score modes), so list them
# together. Mixing two corpora is an error -- run the script once per protocol.
# A chat: group is resolved to the DOCUMENT it wraps: dataset becomes
# chatdoc:<corpus> and seqlen becomes the body budget (total - the TARGET's chat
# affix overhead, 35 on Llama-3.1), so it needs TARGET_MODEL set. The chat
# archive is filed under the same corpus directory as the raw one, so give it
# its OWN root (KEY_TOKEN_ROOT below) -- a run reads one or the other.
# METRICS="gov_jsd_kt gov_jsd_kt_s512 gov_jsd_kt_pp512_s128"
# METRICS="wt2_jsd_kt_pp512_s128"
# METRICS="gov_jsd_kt_pp512_s128_chat"   # chat needs an answer window -> _chat-a512
# METRICS="wt2_jsd_kt_pp512_s128_chat"
METRICS="${METRICS_ARG}"

# correlation.py takes ONE --key_token_path and appends the corpus name, so all
# corpora used in a run must live under the SAME root. With METRICS set the
# derived name below cannot describe several corpora at once -> use a root.
KEY_TOKEN_ROOT=""

# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-3.1-8B-Instruct
# CONFIG=config/llama.json
# DTYPE=float16
# DTYPE=bfloat16

MODEL_PATH=/SSD/huggingface/Qwen
MODEL_NAME=Qwen2.5-7B-Instruct
# MODEL_NAME=Qwen2.5-14B-Instruct
# MODEL_NAME=Qwen2.5-72B-Instruct
CONFIG=config/qwen2.json
DTYPE=float16
# DTYPE=bfloat16

# ── TARGET: the model the archive will be MEASURED with ────────────────────
# The evaluator only decides WHICH tokens are key (it re-tokenizes the text).
# The target owns the loader -- which documents clear min_seqlen, where seqlen
# truncates, the decode and the offset_mapping the character intervals map onto.
# Leaving this empty makes the archive follow the EVALUATOR's tokenization: the
# shipped gov_report archives were cut 2-6% shorter than the Llama loader cuts
# and 19% of their intervals landed off a Llama token boundary. Always set it.
TARGET_MODEL_PATH=/SSD/huggingface/meta-llama
TARGET_MODEL=Llama-3.1-8B-Instruct

SEED=0
# Reuse slices already staged in <save_path>.partial from an interrupted run
# (they are written atomically, so an existing one is complete).
RESUME=True

# MODEL_PATH=/SSD/huggingface/mistralai
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# CONFIG=config/mistral.json
# DTYPE=float16
# # # DTYPE=bfloat16

# DATASET=wikitext2
# # DATASET=c4
# N_SAMPLE=128
# SEQLEN=2048
# DATA_BATCH_SIZE=1
# MIN_SEQLEN=0

DATASET=gov_report
# N_SAMPLE=4
N_SAMPLE=8
# N_SAMPLE=16
# N_SAMPLE=50
# SEQLEN=2048
# DATA_BATCH_SIZE=1
# MIN_SEQLEN=2048
# SEQLEN=4096
# DATA_BATCH_SIZE=1
# MIN_SEQLEN=4096
SEQLEN=8192
DATA_BATCH_SIZE=1
MIN_SEQLEN=8192
# SEQLEN=16384
# DATA_BATCH_SIZE=1
# MIN_SEQLEN=16384

# DATASET=gsm8k
# N_SAMPLE=32
# SEQLEN=256
# # DATA_BATCH_SIZE=1
# DATA_BATCH_SIZE=8
# MIN_SEQLEN=0
# # MIN_SEQLEN=192

# TRUNC_LEN=4096
# SLIDING_WINDOW=1024
# TRUNC_LEN=1024
# SLIDING_WINDOW=256
# TRUNC_LEN=512
# SLIDING_WINDOW=128
TRUNC_LEN=256
SLIDING_WINDOW=64
# TRUNC_LEN=128
# SLIDING_WINDOW=128
# TRUNC_LEN=32
# SLIDING_WINDOW=32

# ALPHA=2
# BETA=-2
ALPHA=1
BETA=-1

# SPLIT='train'
SPLIT='test'

# ── where it goes ──────────────────────────────────────────────────────────
# The archive is per (evaluator, TARGET, corpus, n_sample, seqlen, min_seqlen,
# seed, trunc, sliding_window, alpha, beta). The directory name records most of
# it and meta.json records ALL of it (and is what actually gets checked).
if [ -n "${METRICS}" ]; then
    # Self-describing: who judged (eval-), whose loader/tokenizer/template the
    # intervals are indexed against (tgt-), and the input layout (raw /
    # chat-a<answer window>) -- the three things that make two archives
    # incompatible. The per-corpus protocol lives in <root>/<corpus>/meta.json,
    # which is what actually gets checked.
    KEY_TOKEN_SAVE_PATH=${KEY_TOKEN_ROOT:-key_token/kt_eval-${MODEL_NAME}_tgt-${TARGET_MODEL}}
else
    KEY_TOKEN_SAVE_PATH=${KEY_TOKEN_ROOT:-key_token/${MODEL_NAME}_${DATASET}_${SPLIT}_${N_SAMPLE}sample_${SEQLEN}seqlen_${MIN_SEQLEN}min_${TRUNC_LEN}trunc_${SLIDING_WINDOW}sw_${ALPHA}alpha_${BETA}beta}
fi

ARGS="--gpu_id ${DEVICES} \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --config ${CONFIG} \
    --dtype ${DTYPE} \
    --seed ${SEED} \
    --save_path ${KEY_TOKEN_SAVE_PATH} \
    --verbosity"

if [ -n "${TARGET_MODEL}" ]; then
    ARGS+=" --target_model_path ${TARGET_MODEL_PATH} --target_model ${TARGET_MODEL}"
else
    echo "[gen_key_token.sh] WARNING: TARGET_MODEL is empty — the archive will follow"
    echo "  the EVALUATOR's tokenization and will only 'prefix'-match the model you"
    echo "  measure with. Set TARGET_MODEL unless you know you want that."
fi

[ "${RESUME}" == "True" ] && ARGS+=" --resume"

if [ -n "${METRICS}" ]; then
    # protocol comes from utils/metric_specs.py; --train is forced there
    ARGS+=" --metrics ${METRICS}"
else
    ARGS+=" --dataset ${DATASET} \
    --n_sample ${N_SAMPLE} \
    --seqlen ${SEQLEN} \
    --data_batch_size ${DATA_BATCH_SIZE} \
    --min_seqlen ${MIN_SEQLEN} \
    --trunc_len ${TRUNC_LEN} \
    --sliding_window ${SLIDING_WINDOW} \
    --alpha ${ALPHA} \
    --beta ${BETA}"
    [ ${SPLIT} == 'train' ] && ARGS+=" --train"
fi

echo "[gen_key_token.sh] archive root : ${KEY_TOKEN_SAVE_PATH}"
if [ -n "${METRICS}" ]; then
  echo "  NOTE with --metrics the group appends its layout suffix (_raw / _chat-a<N>),"
  echo "  so pass the BASE root above to correlation.py --key_token_path."
fi
echo "[gen_key_token.sh] pass it to correlation.py as --key_token_path (the corpus"
echo "  name is appended per dataset), or to search.sh as KEY_TOKEN_PATH."

CUDA_VISIBLE_DEVICES=${DEVICES} python gen_key_token.py ${ARGS}
