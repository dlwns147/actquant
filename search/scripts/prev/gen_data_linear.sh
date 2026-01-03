DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
CONFIG=config/llama.json

# N_DATA=250
# N_DATA=500
N_DATA=1000
N_SAMPLE=128

LOSS_FUNC=jsd

METHOD=hqq
METHOD_TEXT=hqq

Q_BITS="2 4"
Q_BITS_TEXT="2_4"
AXIS=1
GROUP_SIZE=128
QSCALE=false
QZERO=false
PASS_LIST="0.self_attn.v_proj 1.self_attn.v_proj 1.mlp.down_proj 31.mlp.down_proj"

QMODEL_PATHS_LIST=()
for B in ${Q_BITS}
do
    # QMODEL_PATHS+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" )
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_float16" )
done
# QMODEL_PATHS=( "/SSD/hqq/${MODEL_NAME}_2bit_64gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_3bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}" "/SSD/hqq/${MODEL_NAME}_4bit_${GROUP_SIZE}gs_${AXIS}axis_qscale_${QSCALE}_qzero_${QZERO}")
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")

SEC_OBJ=bits
SEC_OBJ_RANGE_SMALL=${Q_BITS:0:1} 
SEC_OBJ_RANGE_LARGE=${Q_BITS:(-1)}
SEC_OBJ_RANGE="${SEC_OBJ_RANGE_SMALL} ${SEC_OBJ_RANGE_LARGE}"
# NSAMPLE=32
NSAMPLE=128

LOSS_FILE=data/${MODEL_NAME}_${METHOD_TEXT}_loss_${N_DATA}_${Q_BITS_TEXT}_${SEC_OBJ}_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_pass_${NSAMPLE}sample_test.json
# POOL=data/Llama-2-7b-hf_hqq_loss_1000_2_4_bits_2_4.json
# POOL=/NAS/SJ/nsgaquant/data/Llama-2-7b-hf_hqq_loss_1000_2_4_bits_2_4_pass_train.json
# PPL_FILE=data/${MODEL_NAME}_${METHOD_TEXT}_ppl_${N_DATA}_range_bits_${Q_BITS_TEXT}_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_axis_${AXIS}_scale_${QSCALE}_qz_${QZERO}_lp_range_${LAYER_PRUNE_RANGE_SMALL}_${LAYER_PRUNE_RANGE_LARGE}.json

# LOSS_FILE=data/${MODEL_NAME}_${METHOD}_loss_${N_DATA}_range_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.json
# PPL_FILE=data/${MODEL_NAME}_${METHOD}_ppl_${N_DATA}_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_axis_${SMALL_AXIS}_lb_${LARGE_WBITS}_lgs_${LARGE_GROUP_SIZE}_lqs_${LARGE_QSCALE}_lqz_${LARGE_QZERO}_sb_${SMALL_WBITS}_sgs_${SMALL_GROUP_SIZE}_sqs_${SMALL_QSCALE}_sqz_${SMALL_QZERO}.json

PASS_LINEAR_LIST="0.self_attn.v_proj 1.self_attn.v_proj 1.mlp.down_proj 31.mlp.down_proj" # Llama-2-7b

SEC_OBJ=bits
SEC_OBJ_RANGE_SMALL=${Q_BITS:0:1} 
SEC_OBJ_RANGE_LARGE=${Q_BITS:(-1)}
SEC_OBJ_RANGE="${SEC_OBJ_RANGE_SMALL} ${SEC_OBJ_RANGE_LARGE}"

MAX_VALUE=5
N_PROC=2

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} gen_data_linear.py \
--gpu_id ${DEVICES} \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--quant_model_paths ${QMODEL_PATHS} \
--quant_model_bits ${Q_BITS} \
--loss_json_file ${LOSS_FILE} \
--pass_linear_list ${PASS_LIST} \
--n_data ${N_DATA} \
--n_sample ${N_SAMPLE} \
--sec_obj ${SEC_OBJ} \
--sec_obj_range ${SEC_OBJ_RANGE} \
--method ${METHOD} \
--max_value ${MAX_VALUE} \
--loss_func ${LOSS_FUNC}
# --pool ${POOL} \

# --layer_prune_range ${LAYER_PRUNE_RANGE} \
# --use_linear_group
# --ppl_json_file ${PPL_FILE} \
# --pass_linear_list ${PASS_LIST} \
# --large_model_path ${LARGE_MODEL_PATH} \
# --large_model_bits ${LARGE_WBITS} \
# --small_model_path ${SMALL_MODEL_PATH} \
# --small_model_bits ${SMALL_WBITS} \