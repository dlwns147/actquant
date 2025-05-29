DEVICES=${1}
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

# METHOD="hqq layer_prune"
# METHOD_TEXT="hqq_layer_prune"
METHOD="hqq"
METHOD_TEXT="hqq"

Q_BITS="2 3 4"

AXIS=1
GROUP_SIZE=128

QMODEL_PATHS_LIST=()
for B in ${Q_BITS}
do
    QMODEL_PATHS_LIST+=( "/SSD/hqq/${MODEL_NAME}_${B}bit_${GROUP_SIZE}gs_${AXIS}axis_float16" )
done
QMODEL_PATHS=$(IFS=" " ; echo "${QMODEL_PATHS_LIST[*]}")
BACKEND="gptq gptq gptq"
# BACKEND="bitblas gptq bitblas"

DATASET="wikitext2"

N_OUTLIER=32
OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth

PROMPT_LENGTH=64
GEN_LEN=1024


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${DEVICES} python generation_speed.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--quant_model_paths ${QMODEL_PATHS} \
--quant_model_bits ${Q_BITS} \
--backend ${BACKEND} \
--prompt_length ${PROMPT_LENGTH} \
--gen_length ${GEN_LEN} \
--use_ft \
--only_gemv 
# --dataset ${DATASET}

