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

W_METHOD=hqq
W_BITS="2 3 4"; W_GROUP_SIZE=128; AXIS=1
KV_METHOD="kivi think"
W_METHOD_TEXT=${W_METHOD}
KV_METHOD_TEXT=${KV_METHOD// /_}
QMODEL_PATHS_LIST=()
for B in ${W_BITS}; do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

SURROGATE=rbf      # arch-input predictor: rbf (needs N_DOE > #active genes ≈ 360) / gp / ard_gp / carts
POP=200             # NSGA-III pop (≥ das-dennis 3-obj/12-part = 91 ref dirs; 200 as in search.sh)
N_DOE=500          # DOE measured archs (≥ #active genes for rbf)
ITERATIONS=200       # search iterations (fit ↔ measure)
N_ITER=50          # candidates measured per iteration
SEED=0
SAVE_ITER=10       # dump iter_<it>.stats + iter_<it>.png (via --debug) every SAVE_ITER iters (and the last)

ATTN_SINK=8
N_TOKEN=0

CAND_EVEN=moo        # maximin / grid / hybrid / moo
MOO_GAP_STD=True     # moo 3rd objective ON: (pred-loss × cov_rad × gap-std) knee.
GRID_SEED=True       # True = inject per-cell block-product seeds each iter
SEED_POOL=full       # seed block source: full (FULL ε-band pools W~9k/KV~3k ∪ archive parts;
MOO_ALGO=nsga3       # moo solver: nsga3 (recommended) / nsga2
EVEN_FRAC=0.5        # hybrid only: fraction of K on grid-even coverage


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
CAND_TAG=${CAND_EVEN}                                           # maximin/grid/hybrid/moo
[ "${CAND_EVEN}" == "moo" ]    && CAND_TAG=moo${MOO_ALGO#nsga}  # moo3/moo2 (=nsga ver.)
[ "${CAND_EVEN}" == "moo" ] && [ "${MOO_GAP_STD}" == "True" ] && CAND_TAG+=gs  # +gap-std obj
[ "${CAND_EVEN}" == "hybrid" ] && CAND_TAG=hyb${EVEN_FRAC}
[ "${GRID_SEED}" == "True" ]   && CAND_TAG+=-g${SEED_POOL:0:2}  # -gfu/-gar/-gfi = full/archive/first
SAVE=save/second_search/${TODAY}_${MODEL_NAME}_joint_${W_METHOD_TEXT}${QEFT_TAG}_${KV_METHOD_TEXT}_${SURROGATE}_doe${N_DOE}_it${ITERATIONS}n${N_ITER}p${POP}_${CAND_TAG}_eps${FRONT_EPS_REL}_dk${DIV_K}_st${STRIDE}${PP_TAG}${SINK_TAG}_s${SEED}

echo "SECOND-SEARCH -> ${SAVE}"

ARGS="--config ${CONFIG} \
--model_name ${MODEL_NAME} \
--w_expr ${W_EXPR} \
--eff_kv_expr ${EFF_KV_EXPR} \
--surrogate ${SURROGATE} \
--cand_even ${CAND_EVEN} \
--seed_pool ${SEED_POOL} \
--moo_algo ${MOO_ALGO} \
--even_frac ${EVEN_FRAC} \
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
[ "${MOO_GAP_STD}" == "True" ] && ARGS+=" --moo_gap_std"

ARGS+=" --gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--dtype ${DTYPE} \
--w_method ${W_METHOD} \
--kv_method ${KV_METHOD} \
--quant_model_paths ${QMODEL_PATHS} \
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

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=1 --num_machines=1 \
    --main_process_port=${PORT_NUM} second_search.py ${ARGS}

