DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=float16
CONFIG=config/llama.json

# MODEL_PATH=/SSD/huggingface/Qwen
# # MODEL_NAME=Qwen2.5-7B
# # MODEL_NAME=Qwen2.5-14B
# # MODEL_NAME=Qwen2.5-32B
# # MODEL_NAME=Qwen2.5-72B
# # MODEL_NAME=Qwen2.5-7B-Instruct
# MODEL_NAME=Qwen2.5-14B-Instruct
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/qwen2.json

# MODEL_PATH=/SSD/huggingface/mistralai
# # MODEL_NAME=Mistral-7B-v0.3
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# # DTYPE=bfloat16
# DTYPE=float16
# CONFIG=config/mistral.json

# METHOD="hqq layer_prune"
# METHOD_TEXT="hqq_layer_prune"

# METHOD=hqq
# METHOD_TEXT=hqq
METHOD=awq
METHOD_TEXT=awq
# METHOD="awq layer_prune"
# METHOD_TEXT=awq_layer_prune
# METHOD=fp16
# METHOD_TEXT=fp16

W_BITS="2 3 4"
W_BITS_TEXT="234"
# W_BITS="2 4"
# W_BITS_TEXT="24"
W_BITS="16"
W_BITS_TEXT="16"
AXIS=1
W_GROUP_SIZE=128
QSCALE=false

K_BITS="2 4"
K_BITS_TEXT="24"
# K_GROUP_SIZE=128
K_GROUP_SIZE=("32 64 128" "32 64 128")
K_GROUP_SIZE_TEXT=3264128x2

V_BITS="2 4"
V_BITS_TEXT="24"
V_GROUP_SIZE=128
V_GROUP_SIZE=("32 64 128" "32 64 128")
V_GROUP_SIZE_TEXT=3264128x2

RESIDUAL_LENGTH=128
K_QUANT_PER=channel
V_QUANT_PER=token

QMODEL_PATHS_LIST=()
for B in ${W_BITS}
do
    # QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_${DTYPE}" )
done
# QMODEL_PATHS=( "/SSD/hqq/${MODEL_NAME}_2bit_64gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_3bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_4bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}")
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

N_OUTLIER=32
OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth

# COMP_OBJ=(wbits kvbits)
# COMP_OBJ_TEXT="wkv"
# COMP_OBJ_VAL=(2.5 2.5)

# COMP_OBJ=(kvbits)
# COMP_OBJ_VAL=(3.0)
# COMP_OBJ_THRESHOLD=0.005

COMP_OBJ=(memory)

# COMP_OBJ_VAL=(5878849536)
# COMP_OBJ_VAL=(5862072320)
# N_TOKEN=1024

# COMP_OBJ_VAL=(42350419968)
COMP_OBJ_VAL=(25170550784)
N_TOKEN=1048576
COMP_OBJ_THRESHOLD=$(echo "scale=3; (${COMP_OBJ_VAL[0]} * 0.001)" | bc)

# PREFER="metric#0.0 ${TARGET_COMP_OBJ}#${TARGET_COMP_OBJ_VAL}"

PREFER_LIST=("metric#0.0")
MIN_COMP_OBJ_LIST=()
MAX_COMP_OBJ_LIST=()

for IDX in "${!COMP_OBJ[@]}"
do
    PREFER_LIST+=( "${COMP_OBJ[$IDX]}#${COMP_OBJ_VAL[$IDX]}" )
    MIN_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} - $COMP_OBJ_THRESHOLD" | bc) )
    MAX_COMP_OBJ_LIST+=( $(echo "scale=3; ${COMP_OBJ_VAL[$IDX]} + $COMP_OBJ_THRESHOLD" | bc) )
done

COMP_OBJ=$(IFS=" " ; echo "${COMP_OBJ[*]}")
COMP_OBJ_VAL=$(IFS=" " ; echo "${COMP_OBJ_VAL[*]}")
PREFER=$(IFS=" " ; echo "${PREFER_LIST[*]}")
MIN_COMP_OBJ=$(IFS=" " ; echo "${MIN_COMP_OBJ_LIST[*]}")
MAX_COMP_OBJ=$(IFS=" " ; echo "${MAX_COMP_OBJ_LIST[*]}")
MIN_COMP_OBJ_TEXT=$(IFS="_" ; echo "${MIN_COMP_OBJ_LIST[*]}")
MAX_COMP_OBJ_TEXT=$(IFS="_" ; echo "${MAX_COMP_OBJ_LIST[*]}")

DATASETS="wikitext2 c4"
# TASKS="piqa winogrande hellaswag arc_challenge arc_easy lambada_openai boolq openbookqa social_iqa"
TASKS="coqa gsm8k truthfulqa"
# TASKS="coqa truthfulqa"

LM_EVAL_BATCH_SIZE=32

EXPR_FOLDER=save/search/quant

# EXPR_FILE=2507191514_Llama-3.1-8B-Instruct_memory_loss_hqq_iter_200_n_iter_50_w234k24v24bits_w128k3264128x2v3264128x2gs_128res_len_k_channel_v_token_obj_1_1e99_jsd_co_0.9_mut_0.1_wikitext2_1bs_32sample_2048seq_0minseq_1024token_rbf/iter_200.stats
EXPR_FILE=2507191514_Llama-3.1-8B-Instruct_memory_loss_hqq_iter_200_n_iter_50_w234k24v24bits_w128k3264128x2v3264128x2gs_128res_len_k_channel_v_token_obj_1_1e99_jsd_co_0.9_mut_0.1_wikitext2_1bs_32sample_2048seq_0minseq_1048576token_rbf/iter_200.stats
# EXPR_FILE=2506030600_Llama-3.1-8B-Instruct_kv_loss_hqq_iter_50_n_iter_50_w16k24v24bits_w128k3264128128v3264128128gs_0res_len_k_channel_v_token_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_32sample_rbf/iter_27.stats

# EXPR_FILE=2505290559_Llama-2-13b-hf_kv_loss_hqq_iter_100_n_iter_50_w16k24v24bits_w128k128v128group_size_0res_len_k_channel_v_token_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_128sample_rbf/iter_50.stats
# EXPR_FILE=2505290559_Llama-2-13b-hf_kv_loss_hqq_iter_100_n_iter_50_w16k24v24bits_w128k128v128group_size_0res_len_k_channel_v_token_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_128sample_rbf/iter_80.stats

# EXPR_FILE=2505281228_Llama-2-7b-hf_kv_loss_hqq_iter_100_n_iter_50_w16k24v24bits_w128k128v128group_size_0res_len_k_channel_v_token_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_128sample_rbf/iter_50.stats
# EXPR_FILE=2505281228_Llama-2-7b-hf_kv_loss_hqq_iter_100_n_iter_50_w16k24v24bits_w128k128v128group_size_0res_len_k_channel_v_token_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_128sample_rbf/iter_80.stats

# EXPR_FILE=2505111531_Llama-2-7b-hf_wkv_loss_hqq_iter_200_w234k24v24bits_w128k128v128group_size_0res_len_k_channel_v_token_obj_22_55_jsd_co_0.9_mut_0.1_wikitext2_32sample_rbf/iter_200.stats
# EXPR_FILE=2505111134_Llama-2-7b-hf_wkv_loss_hqq_iter_200_w234k24v24bits_w128k128v128group_size_0res_len_k_channel_v_token_obj_22_55_jsd_co_0.9_mut_0.1_wikitext2_32sample_rbf/iter_200.stats

# EXPR_FILE=2502101608_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4.1_jsd_co_0.9_mut_0.1_wikitext2_32sample_rbf_outlier_234_mixed/iter_200.stats
# EXPR_FILE=2502012035_Llama-2-7b-hf_bits_loss_hqq_layer_prune_iter_300_234_obj_1.99_4_jsd_co_0.9_mut_0.1_wikitext2_32sample_lp_0.001_1.0/iter_300.stats
# EXPR_FILE=2501231721_Llama-2-13b-hf_bits_loss_hqq_iter_400_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample/iter_400.stats
# EXPR_FILE=2501231719_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample/iter_300.stats
# EXPR_FILE=2501231756_Llama-2-7b-hf_bits_loss_hqq_iter_300_234_obj_2_4_jsd_co_0.9_mut_0.1_wikitext2_128sample_outlier/iter_300.stats
# EXPR_FILE=2411211754_Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.05_layer_prune_1.0_1.0/iter_299.stats

# SAVE=save/result/${TODAY}_${MODEL_NAME}_${COMP_OBJ}_${MIN_COMP_OBJ}_${MAX_COMP_OBJ}
# LONG_BENCH_RESULT_PATH=save/long_bench/${TODAY}_${MODEL_NAME}_our_${METHOD}_${COMP_OBJ_TEXT}_${MIN_COMP_OBJ_TEXT}_${MAX_COMP_OBJ_TEXT}_k${K_BITS_TEXT}bits_k${K_GROUP_SIZE}gs_${K_QUANT_PER}_v${V_BITS_TEXT}bits_v${V_GROUP_SIZE}gs_${V_QUANT_PER}_r${RESIDUAL_LENGTH}
LONG_BENCH_RESULT_PATH=save/long_bench/${TODAY}_${MODEL_NAME}_our_${METHOD}_${COMP_OBJ_TEXT}_${MIN_COMP_OBJ_TEXT}_${MAX_COMP_OBJ_TEXT}_k${K_BITS_TEXT}bits_k${K_GROUP_SIZE_TEXT}gs_${K_QUANT_PER}_v${V_BITS_TEXT}bits_v${V_GROUP_SIZE_TEXT}gs_${V_QUANT_PER}_r${RESIDUAL_LENGTH}
LONG_BENCH_CONFIG=utils/long_bench_config
LONG_BENCH_TASK=""

PASS_KEY_FILE=/NAS/SJ/actquant/search/passkey_examples.jsonl

N=1

N_PROC=1
# N_PROC=2
ARGS="--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--dtype ${DTYPE} \
--comp_obj ${COMP_OBJ} \
--comp_obj_min ${MIN_COMP_OBJ} \
--comp_obj_max ${MAX_COMP_OBJ} \
--w_bits ${W_BITS} \
--k_bits ${K_BITS} \
--v_bits ${V_BITS} \
--w_group_size ${W_GROUP_SIZE} \
--residual_length ${RESIDUAL_LENGTH} \
--k_quant_per ${K_QUANT_PER} \
--v_quant_per ${V_QUANT_PER} \
--use_flash \
--n_token ${N_TOKEN} \
-n ${N} \
--debug \
--expr ${EXPR_FOLDER}/${EXPR_FILE} \
--prefer ${PREFER} \
--datasets ${DATASETS} \
--zeroshot \
--tasks ${TASKS} \
--lm_eval_batch_size ${LM_EVAL_BATCH_SIZE}"
for g in "${K_GROUP_SIZE[@]}"
do
    ARGS+=" --k_group_size ${g} "
done

for g in "${V_GROUP_SIZE[@]}"
do
    ARGS+="--v_group_size ${g} "
done

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} post_search.py ${ARGS}

# --long_bench \
# --long_bench_result_path ${LONG_BENCH_RESULT_PATH} \
# --long_bench_config ${LONG_BENCH_CONFIG}

# --pass_key_file ${PASS_KEY_FILE} \
# --long_bench_e

# --k_group_size ${K_GROUP_SIZE} \
# --v_group_size ${V_GROUP_SIZE} \

# --method ${METHOD} \
# --quant_model_paths ${QMODEL_PATHS} \

# --save ${SAVE} \
# --latency_table_file ${LATENCY_TABLE}
# --outlier_path ${OUTLIER_PATH} \
# --only_front \


    # --greedy_search_result_path ${GREEDY_SEARCH}
# GREEDY_SEARCH=''
# GREEDY_SEARCH=csv/greedy_search/Llama-2-7b-hf_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv
# EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_layer_prune_iter_300_nsga2_2_4_obj_2_4_mut_0.1_layer_prune_0.95_1.0_2410311536/iter_299.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_awq_iter_300_nsga2_2_4_0.01_2410211524/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_2_4_mut_prob_0.1_2410101147/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_2_4_mut_prob_0.2_2410101159/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_2_4_mut_prob_0.02_2410101352/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_gptq_iter_300_nsga2_2_4_2410070911/iter_270.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_owq_iter_300_nsga2_2.1_4.1_2410071301/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_owq_iter_300_nsga2_2.01_4.01_2410071302/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_hqq_iter_300_nsga2_2_4_2410071303/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_300_nsga2_2_4_2410051059/iter_300.stats
# EXPR_FILE=Llama-2-7b-hf_bits_loss_iter_200_nsga2_2_4_2410051103/iter_200.stats
# TARGET_BITS_RANGE="${MIN_BITS} ${MAX_BITS}"
# QMODEL_PATHS=("/SSD/awq/${MODEL_NAME}_w2_g64_fake_${SCALE_BITS}bit_128gs_awq.pt" "/SSD/awq/${MODEL_NAME}_w3_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt" "/SSD/awq/${MODEL_NAME}_w4_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt")


# METHOD=awq
# METHOD_TEXT=awq
# GROUP_SIZE=128
# SCALE_BITS=2
# # SCALE_BITS=3

# QMODEL_PATHS=()
# for B in ${Q_BITS}
# do
#     # QMODEL_PATHS+=( "/SSD/awq/${MODEL_NAME}_w${B}_g${GROUP_SIZE}_fake_${SCALE_BITS}bit_awq.pt" )
#     QMODEL_PATHS+=( "/SSD/awq/${MODEL_NAME}_w${B}_g${GROUP_SIZE}_fake_${SCALE_BITS}scale_asym.pt" )
# done