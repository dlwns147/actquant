DEVICES=${1}
TODAY=`date +%y%m%d%H%M`

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
CONFIG=config/llama.json

N_DATA=64
N_SAMPLE=128

PASS_LIST="0.self_attn.q_proj 0.mlp.up_proj 1.self_attn.q_proj 1.mlp.up_proj 31.mlp.up_proj"
SEC_OBJ_RANGE_SMALL=0.5
SEC_OBJ_RANGE_LARGE=1

LOSS_FUNC='cross_entropy'
# LOSS_FUNC='jsd'

LOSS_FILE=data/${MODEL_NAME}_layer_prune_loss_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}.json
PPL_FILE=data/${MODEL_NAME}_layer_prune_ppl_${SEC_OBJ_RANGE_SMALL}_${SEC_OBJ_RANGE_LARGE}_${LOSS_FUNC}.json

SEC_OBJ_RANGE="${SEC_OBJ_RANGE_SMALL} ${SEC_OBJ_RANGE_LARGE}"

# MAX_VALUE=5
MAX_VALUE=12

CUDA_VISIBLE_DEVICES=${DEVICES} python gen_data_layer.py \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--pass_linear_list ${PASS_LIST} \
--loss_json_file ${LOSS_FILE} \
--ppl_json_file ${PPL_FILE} \
--n_data ${N_DATA} \
--n_sample ${N_SAMPLE} \
--sec_obj_range ${SEC_OBJ_RANGE} \
--max_value ${MAX_VALUE} \
--loss_func ${LOSS_FUNC}