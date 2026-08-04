DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# ── Stage 2: fit surrogate from results.csv → pick final arch → benchmark ──
# Pairs with scripts/sample_surrogate.sh (stage 1). SAMPLE_PATH must
# point at a results.csv it wrote; the expr archives / model / config MUST
# match the ones used in stage 1.

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
# DTYPE=float16
DTYPE=bfloat16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# MODEL_NAME=Qwen2.5-7B-Instruct
# DTYPE=float16
# CONFIG=config/qwen2.json

USE_KEY_TOKEN=False

W_METHOD=hqq
W_METHOD_TEXT=hqq
# W_METHOD=awq
# W_METHOD_TEXT=awq
# ── AWQ-based QEFT (searchable per-layer FP16 outlier columns) ──
# Needs --outlier_path = the multi-rank dict from extract_outidx.py; its
# ranks must cover every n_outlier the arch selects (0/32/64/96/128). The
# w arch entries are (bits, n_outlier) tuples (from a QEFT-space archive).
# W_METHOD=awq_qeft
# W_METHOD_TEXT=awq_qeft
W_BITS="2 3 4"
AXIS=1
W_GROUP_SIZE=128

KV_METHOD="kivi think"
KV_METHOD_TEXT="kivi_think"
K_BITS="2 4"
K_BITS_TEXT="24"
K_GROUP_SIZE=("128" "128")
V_BITS="2 4"
V_BITS_TEXT="24"
V_GROUP_SIZE=("128" "128")

RESIDUAL_LENGTH=128
# ATTN_SINK=0
ATTN_SINK=8
K_QUANT_SCHEME=channel
V_QUANT_SCHEME=token

QMODEL_PATHS_LIST=()
for B in ${W_BITS}; do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

SEED=0

# ── COMP_OBJ range (the deployment budget) ──
# NOTE (second_expr): the pool is DISCRETE (10500 archs), so a tiny ±band can
# match 0 archs (the per-axis path enumerates a continuous space, so it doesn't).
# At target 5.316e9, n_token 16384: ×0.00001→0 archs, ×0.001→55, ×0.005→234.
COMP_OBJ=(memory)
COMP_OBJ_VAL=(5315764224)

# COMP_OBJ=4969209856
# COMP_OBJ=6141255680
# BIN=12
# COMP_OBJ=( $(linspace "${COMP_OBJ_MIN}" "${COMP_OBJ_MAX}" "${BIN}") )
# N_TOKEN=16384

# COMP_OBJ_THRESHOLD_LIST=($(echo "scale=3; (${COMP_OBJ_VAL[0]} * 0.05)" | bc))
COMP_OBJ_THRESHOLD_LIST=($(echo "scale=3; (${COMP_OBJ_VAL[0]} * 0.005)" | bc))
# COMP_OBJ_THRESHOLD_LIST=($(echo "scale=3; (${COMP_OBJ_VAL[0]} * 0.00001)" | bc))

# COMP_OBJ=(wbits kvbits kvdim)
# COMP_OBJ_VAL=(3 3.25 102)
# # COMP_OBJ_THRESHOLD_LIST=($(echo "scale=3; (${COMP_OBJ_VAL[0]} * 0.05)" | bc))
# COMP_OBJ_THRESHOLD_LIST=(0.005 0.005 0.05)

# ── 2nd-stage JOINT (second_search) budget box: W × eff_kvbits ──
# Use with SECOND_EXPR below. VAL = box center, min/max = VAL ∓ THRESHOLD.
# COMP_OBJ=(wbits eff_kvbits)
# COMP_OBJ_VAL=(2.5 2.5)
# COMP_OBJ_THRESHOLD_LIST=(0.5 0.5)   # → wbits[2,3] eff_kvbits[2,3]

N_TOKEN=16384

MIN_COMP_OBJ_LIST=()
MAX_COMP_OBJ_LIST=()
for IDX in "${!COMP_OBJ[@]}"; do
    MIN_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} - ${COMP_OBJ_THRESHOLD_LIST[$IDX]}" | bc) )
    MAX_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} + ${COMP_OBJ_THRESHOLD_LIST[$IDX]}" | bc) )
done
COMP_OBJ_TEXT=$(IFS="_" ; echo "${COMP_OBJ[*]}")
COMP_OBJ=$(IFS=" " ; echo "${COMP_OBJ[*]}")
MIN_COMP_OBJ=$(IFS=" " ; echo "${MIN_COMP_OBJ_LIST[*]}")
MAX_COMP_OBJ=$(IFS=" " ; echo "${MAX_COMP_OBJ_LIST[*]}")
MIN_COMP_OBJ_TEXT=$(IFS="_" ; echo "${MIN_COMP_OBJ_LIST[*]}")
MAX_COMP_OBJ_TEXT=$(IFS="_" ; echo "${MAX_COMP_OBJ_LIST[*]}")

PREFER="metric#0.0"
# RECIPE: top-k verify — JSD-screen the predicted top-VERIFY_TOPK in the band,
# benchmark only the measured-best N. k=5 recovers the true band-best 96-100%
# (top-1 alone 50-73%; in-band tau ~0.5-0.7 near-ties) at (k-N) extra JSD evals.
# N keeps its original meaning = number of final architectures to benchmark.
N=1
SELECT_MEASURED_BEST=False   # second_expr: loss already measured → no re-screen
VERIFY_TOPK=5

# ── (A) 이름으로 재기 — utils/metric_specs.py 레지스트리 ──
# 이름 하나가 데이터 프로토콜(dataset/seqlen/n_sample/답변창)과 포워드 프로토콜
# (stride/prefill)을 모두 고정한다 → correlation.py의 같은 이름과 정의가 동일.
# 같은 group을 쓰는 태스크끼리는 FP-teacher 패스를 공유한다.
#   wt2_jsd_pp512_s32  (A_pp    : wikitext2 2048tok n128, 답변창512, s32)
#   wt2_jsd_pp128_s32  (A_lt128 : wikitext2 2048tok n128, 답변창128, s32)
#   gov_jsd_pp512_s128 (B_pp    : gov_report 8196tok n8, 답변창512, s128)
#   gov_jsd_pp128_s32  (B_lt128 : gov_report 8196tok n8, 답변창128, s32)
#   c4_ppl / wt2_ppl   (A       : 전체창 단일 forward PPL)
# 전체 목록: python post_search.py --help 의 --metric_tasks 참고.
# ⚠️ needle_*/gsm8k_jsd_pp_* 는 전용 로더가 필요해서 correlation.py에서만 된다.
# ⚠️ teacher logits(CPU) 크기 = n_sample × 답변창 × vocab × 2B. 요청한 태스크가
#    실제로 쓰는 것만 만든다(PPL 전용이면 0). gov 계열은 합쳐서 ~1.3GB지만
#    wt2_jsd_pp512_* 를 추가하면 +16.8GB, 답변창 없는 wt2_jsd는 +67GB.
METRIC_TASKS="gov_jsd_pp512_s128 gov_jsd_pp128_s32 wt2_ppl c4_ppl"
# METRIC_TASKS="gov_jsd_pp512_s128 gov_jsd_pp128_s32 wt2_jsd_pp512_s32 wt2_jsd_pp128_s32 wt2_ppl c4_ppl"
# METRIC_TASKS=""   # 비우면 아래 (B) knob 방식으로 동작

# ── (B) knob으로 재기 — 레지스트리에 없는 임시 조합용 ──
# METRIC_TASKS가 비어 있을 때만 쓰인다. 첫 항목=선택/verify 기준.
METRIC="loss ppl"
LOSS_FUNC="jsd"
# METRIC="ppl"
# LOSS_FUNC="cross_entropy"

# ── metric별 데이터/포워드 프로토콜 (공유 knob 없음, 각자 다 지정) ──
# 'loss'(JSD)는 train_loaders, 'ppl'은 test_loaders를 읽는다. METRIC에 넣은
# 지표는 해당 쪽 DATASETS가 반드시 있어야 한다(없으면 즉시 에러).
# STRIDE/LAST_TOKENS 는 0 = 끄기(단일 forward / 전체 시퀀스 채점).
# MIN_SEQLEN은 gov_report/gsm8k에만 적용(wikitext2/c4 로더는 무시).
# N_SAMPLE은 train split·gov_report에만 적용(wikitext2/c4 TEST split은 전량 사용).
# ⚠️ LOSS_LAST_TOKENS 는 FP-teacher dense_logits 마스킹까지 결정한다.
#    gov_report 8x8196을 전체 위치로 저장하면 ~16.8GB VRAM → 512 유지 권장(~1GB).
LOSS_DATASETS="gov_report"
LOSS_SEQLEN=8196
LOSS_MIN_SEQLEN=8192
LOSS_N_SAMPLE=8
LOSS_DATA_BATCH_SIZE=1
LOSS_STRIDE=128
LOSS_PREFILL_PROMPT=True
LOSS_LAST_TOKENS=512

PPL_DATASETS="wikitext2 c4"
PPL_SEQLEN=2048
PPL_MIN_SEQLEN=0
PPL_N_SAMPLE=128
PPL_DATA_BATCH_SIZE=1
PPL_STRIDE=0
PPL_PREFILL_PROMPT=False
PPL_LAST_TOKENS=0

# LOGIT_DATASET = teacher logit(=JSD 측정)을 저장할 LOSS_DATASETS의 부분집합.
# 비워두면 LOSS_DATASETS 전체. LOSS쪽에 없는 이름을 주면 즉시 에러(예전엔 JSD가
# 조용히 0개 측정됐음).
LOGIT_DATASET=""

# teacher logits 보관 위치. cpu = 필요한 시퀀스만 GPU로 올림(기본).
# 크기 = LOSS_N_SAMPLE × LOSS_LAST_TOKENS × vocab × 2B
#   gov n8·lt512 ≈ 1.0GB / wt2 n128·lt512 ≈ 16.8GB / wt2 n128·lt128 ≈ 4.2GB
DENSE_LOGITS_DEVICE=cpu

TRUNC_LEN=256
SLIDING_WINDOW=64
ALPHA=2
BETA=-2

# ── per-axis search archives — MUST match stage 1 ──
# W_EXPR=save/search/think/2605112032_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_200.stats
# KV_EXPR=save/search/think/2605112033_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x2_128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_100.stats
# KVDIM_EXPR=save/search/think/2605112036_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats
# SAMPLE_PATH=save/result/260513/2605132157_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_qs_metric_w05595_metric_kv05595_metric_kvdim05595_rs23/results.csv

# ── 2nd-stage JOINT search (second_search.py) ──
SECOND_EXPR=save/second_search/2606202032_Llama-3.1-8B-Instruct_joint_hqq_rbf_doe500_it200n50_sk8_s0/iter_200.stats

for VAR_NAME in W_EXPR KV_EXPR KVDIM_EXPR SAMPLE_PATH SECOND_EXPR; do
    VAR_VALUE="${!VAR_NAME}"
    if [ -n "${VAR_VALUE}" ] && [[ "${VAR_VALUE}" != *"${MODEL_NAME}"* ]]; then
        echo "ERROR: ${VAR_NAME} does not contain MODEL_NAME (${MODEL_NAME}): ${VAR_VALUE}"
        exit 1
    fi
done

SURROGATE=ard_gp
SURROGATE=sqrty_ard_gp
# RECIPE: rbf (tps kernel) — maximin × rbf-tps was the TOP combo (multi-seed
# R² 0.9694, best worst-bin), pair with maximin coverage sampling
# (sample_surrogate.sh SAMPLING_METHOD=maximin). sqrty_ard_gp is the robust
# alternative. ⚠️ NEVER pair rbf-tps with uncertainty-AL (interpolant has no
# noise reg → clustered picks make it extrapolate wildly, R² went negative).
SURROGATE=rbf
RBF_KERNEL=tps
# device for the pure-PyTorch rbf / ard_gp surrogate:
# auto (cuda if visible else cpu) | cpu | cuda | cuda:N
SURROGATE_DEVICE=auto

# TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq openbookqa social_iqa"
# TASKS="coqa gsm8k truthfulqa"
TASKS="coqa truthfulqa gsm8k"
LM_EVAL_BATCH_SIZE=1

LONGBENCH_CONFIG=utils/longbench_config
# Abbreviated attention-sink tag (e.g. _sk8), only when sink is on so sink=0
# result-dir names stay comparable. MUST match the search-time ATTN_SINK.
SINK_TAG=""
[ ${ATTN_SINK} -ne 0 ] && SINK_TAG="_sk${ATTN_SINK}"
LONGBENCH_RESULT_PATH=save/longbench/${TODAY}_${MODEL_NAME}_${W_METHOD_TEXT}_${KV_METHOD_TEXT}_k${K_BITS_TEXT}bits_v${V_BITS_TEXT}bits_r${RESIDUAL_LENGTH}${SINK_TAG}
MINILONGBENCH_RESULT_PATH=save/minilongbench/${TODAY}_${MODEL_NAME}_${W_METHOD_TEXT}_${KV_METHOD_TEXT}_k${K_BITS_TEXT}bits_v${V_BITS_TEXT}bits_r${RESIDUAL_LENGTH}${SINK_TAG}
PASS_KEY_FILE=/NAS/SJ/actquant/search/passkey_examples.jsonl

# RULER_TASK="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery ruler_vt ruler_cwe ruler_fwe ruler_qa_squad ruler_qa_hotpot"
# RULER_TASK="niah_single_1"
RULER_TASK="ruler_qa_squad ruler_qa_hotpot"
RULER_YAML_PATH=utils/ruler_utils
# RULER_LENGTH=4096
RULER_LENGTH=16384
# RULER_LENGTH=65536
# RULER_LENGTH=128000
# RULER_LENGTH=131072
# RULER_SAMPLE=5
RULER_SAMPLE=50
RULER_BATCH_SIZE=1
RULER_RESULT_PATH=save/ruler/${TODAY}_${MODEL_NAME}_our_${W_METHOD_TEXT}_${KV_METHOD_TEXT}_${COMP_OBJ_TEXT}_${MIN_COMP_OBJ_TEXT}_${MAX_COMP_OBJ_TEXT}_k${K_BITS_TEXT}bits_k${K_GROUP_SIZE_TEXT}gs_${K_QUANT_SCHEME}_v${V_BITS_TEXT}bits_v${V_GROUP_SIZE_TEXT}gs_${V_QUANT_SCHEME}_r${RESIDUAL_LENGTH}${SINK_TAG}_ruler_${RULER_LENGTH}len_${RULER_SAMPLE}sample_${RULER_BATCH_SIZE}bs_${SEED}seed


# 잰 지표 태그: METRIC_TASKS가 있으면 그 이름에서, 없으면 loss쪽 knob에서.
# 정확한 목록은 어차피 results.csv의 metric 열에 남는다.
source "$(dirname "${BASH_SOURCE[0]}")/metric_tag.sh"
if [ -n "${METRIC_TASKS}" ]; then
    MTAG=$(metric_tag_from_tasks "${METRIC_TASKS}")
else
    MTAG=$(metric_tag_from_knobs "${LOSS_DATASETS}" "${LOSS_FUNC}" loss \
                                 "${LOSS_N_SAMPLE}" "${LOSS_SEQLEN}" "${LOSS_MIN_SEQLEN:-0}")
    # 이름 방식과 달리 knob 방식은 stride/답변창이 dir 어디에도 없다 → 여기서 붙인다
    # (search.sh의 _st<STRIDE>_pp<LAST_TOKENS> 관례. loss쪽 = 선택 목적함수 기준)
    [ "${LOSS_STRIDE:-0}" -gt 0 ] && MTAG="${MTAG}_st${LOSS_STRIDE}"
    [ "${LOSS_PREFILL_PROMPT}" == "True" ] && MTAG="${MTAG}_pp${LOSS_LAST_TOKENS}"
fi

SAVE=save/post_search/${TODAY}_${MODEL_NAME}_${COMP_OBJ_TEXT}_${MIN_COMP_OBJ_TEXT}_${MAX_COMP_OBJ_TEXT}_${W_METHOD_TEXT}_${KV_METHOD_TEXT}_${SURROGATE}${SINK_TAG}${MTAG}

ARGS="--gpu_id ${DEVICES} \
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
--metric ${METRIC} \
--loss_func ${LOSS_FUNC} \
--seed ${SEED} \
-n ${N} \
--save ${SAVE} \
--comp_obj ${COMP_OBJ} \
--comp_obj_min ${MIN_COMP_OBJ} \
--comp_obj_max ${MAX_COMP_OBJ}"


# --expr_front \

for g in "${K_GROUP_SIZE[@]}"; do ARGS+=" --k_group_size ${g} "; done
for g in "${V_GROUP_SIZE[@]}"; do ARGS+=" --v_group_size ${g} "; done

if [ ${USE_KEY_TOKEN} == 'True' ]; then
    ARGS+=" --use_key_token --trunc_len ${TRUNC_LEN} --sliding_window ${SLIDING_WINDOW} --alpha ${ALPHA} --beta ${BETA} --key_token_path ${KEY_TOKEN_PATH} "
fi
[ ${W_METHOD} == "hqq" ] && ARGS+=" --quant_model_paths ${QMODEL_PATHS} "
# QEFT / AWQ-QEFT: the multi-rank outlier dict from extract_outidx.py. Its
# ranks (e.g. r32_64_96_128) must cover every n_outlier the arch selects and
# OUTLIER_DATASET must match how extract_outidx.sh was run (it names the dir —
# it is NOT the loss/ppl measurement dataset).
OUTLIER_DATASET=wikitext2
if [ "${W_METHOD}" == "qeft" ] || [ "${W_METHOD}" == "awq_qeft" ]; then
    OUTLIER_PATH=/NAS/SJ/actquant/search/outlier/${MODEL_NAME}/w16_r32_64_96_128_${OUTLIER_DATASET}/outlier.pth
    ARGS+=" --outlier_path ${OUTLIER_PATH} "
fi
if [ -n "${SECOND_EXPR}" ]; then
    # joint path: archive already holds assembled joint archs with measured JSD,
    # so the per-axis expr archives + surrogate are skipped entirely.
    ARGS+=" --second_expr ${SECOND_EXPR}"
else
    [ -n "${W_EXPR}" ]      && ARGS+=" --w_expr ${W_EXPR}"
    [ -n "${KV_EXPR}" ]     && ARGS+=" --kv_expr ${KV_EXPR}"
    [ -n "${KVDIM_EXPR}" ]  && ARGS+=" --kvdim_expr ${KVDIM_EXPR}"
    [ -n "${EFF_KV_EXPR}" ] && ARGS+=" --eff_kv_expr ${EFF_KV_EXPR}"
    [ -n "${SAMPLE_PATH}" ] && ARGS+=" --sample_path ${SAMPLE_PATH} --surrogate ${SURROGATE} --rbf_kernel ${RBF_KERNEL} --surrogate_device ${SURROGATE_DEVICE}"
fi
[ "${SELECT_MEASURED_BEST:-False}" = "True" ] && ARGS+=" --select_measured_best --verify_topk ${VERIFY_TOPK:-5}"

[ -n "${LOGIT_DATASET}" ] && ARGS+=" --logit_dataset ${LOGIT_DATASET}"
# (A) 이름 방식이 지정되면 그게 우선 — 아래 loss_*/ppl_* knob은 무시된다
[ -n "${METRIC_TASKS}" ] && ARGS+=" --metric_tasks ${METRIC_TASKS}"
[ -n "${DENSE_LOGITS_DEVICE}" ] && ARGS+=" --dense_logits_device ${DENSE_LOGITS_DEVICE}"
# metric별 프로토콜: loss_* / ppl_* 를 그대로 넘긴다(빈 값은 생략 = argparse 기본값)
for SIDE in loss ppl; do
    UP=$(echo ${SIDE} | tr '[:lower:]' '[:upper:]')
    for KEY in DATASETS SEQLEN MIN_SEQLEN N_SAMPLE DATA_BATCH_SIZE STRIDE LAST_TOKENS; do
        VAR_NAME="${UP}_${KEY}"
        VAR_VALUE="${!VAR_NAME}"
        LOW=$(echo ${KEY} | tr '[:upper:]' '[:lower:]')
        [ -n "${VAR_VALUE}" ] && ARGS+=" --${SIDE}_${LOW} ${VAR_VALUE}"
    done
    PP_NAME="${UP}_PREFILL_PROMPT"
    PP_VALUE="${!PP_NAME}"
    [ "${PP_VALUE}" == "True" ]  && ARGS+=" --${SIDE}_prefill_prompt"
    [ "${PP_VALUE}" == "False" ] && ARGS+=" --no-${SIDE}_prefill_prompt"
done
# ARGS+=" --zeroshot --tasks ${TASKS} --lm_eval_batch_size ${LM_EVAL_BATCH_SIZE}"
# ARGS+=" --longbench --longbench_result_path ${LONGBENCH_RESULT_PATH} --longbench_config ${LONGBENCH_CONFIG} --longbench_e "
# ARGS+=" --minilongbench --minilongbench_result_path ${MINILONGBENCH_RESULT_PATH} --longbench_config ${LONGBENCH_CONFIG}"
# ARGS+=" --ruler --ruler_task ${RULER_TASK} --ruler_yaml_path ${RULER_YAML_PATH} --ruler_result_path ${RULER_RESULT_PATH} --ruler_batch_size ${RULER_BATCH_SIZE} --ruler_sample ${RULER_SAMPLE} --ruler_length ${RULER_LENGTH}"
# ARGS+=" --pass_key_file ${PASS_KEY_FILE}"


N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} post_search.py ${ARGS}

# W_SCALE=1
# KV_SCALE=1
# KVDIM_SCALE=1
# EFF_KVDIM_SCALE=1
# --w_scale ${W_SCALE} \
# --kv_scale ${KV_SCALE} \
# --kvdim_scale ${KVDIM_SCALE} \
# --eff_kv_scale ${EFF_KVDIM_SCALE} \

# --prefer ${PREFER}

# Llama-3.1-8B-Instruct
# W_EXPR=save/search/think/2605112032_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_200.stats
# KV_EXPR=save/search/think/2605112033_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x2_128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_100.stats
# KVDIM_EXPR=save/search/think/2605112036_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats
# SAMPLE_PATH=save/result/260513/2605132157_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_qs_metric_w05595_metric_kv05595_metric_kvdim05595_rs23/results.csv

# W_EXPR=save/search/think/2605112127_Qwen2.5-7B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_200.stats
# KV_EXPR=save/search/think/2605112126_Qwen2.5-7B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x2_128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats
# KVDIM_EXPR=save/search/think/2605112128_Qwen2.5-7B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats
# SAMPLE_PATH=save/result/260513/2605130706_Qwen2.5-7B-Instruct_memory_5957466112.00_6584567808.00_hqq_kivi_wikitext2_1_kv_scale_w_expr_kvdim_expr_qs_metric_w159_metric_kvdim159_rs41/results.csv