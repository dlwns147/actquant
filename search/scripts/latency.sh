DEVICES=${1}

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf

CONFIG=config/llama.json

METHOD=layer_prune
Q_BITS=16
ENV=rtx6000ada
# ENV=a100
LATENCY_TABLE=latency_table/${MODEL_NAME}_${ENV}_token.json

NSAMPLE=128

# LAYERS=25
# LAST_LAYERS=(19.self_attn 28.self_attn 18.self_attn)
# LAST_LAYER=19.self_attn
# LAST_LAYER=28.self_attn
# LAST_LAYER=18.self_attn

LAYERS=40
LAST_LAYERS=(22.self_attn 7.mlp)
# LAST_LAYER=22.self_attn
# LAST_LAYER=7.mlp

# LAYERS=32
# LAST_LAYERS=(32.self_attn 31.mlp 26.mlp)
# LAST_LAYERS=(32.self_attn)
# LAST_LAYER=32.self_attn
# LAST_LAYER=31.mlp
# LAST_LAYER=26.mlp


# LAYERS=50
# LAST_LAYERS=(15.self_attn 25.mlp)
# # LAST_LAYER=15.self_attn
# LAST_LAYER=25.mlp


GREEDY_RESULTS=/NAS/SJ/nsgaquant/csv/finercut/${MODEL_NAME}_loss_${NSAMPLE}_${LAYERS}layers_js.csv

for LAST_LAYER in ${LAST_LAYERS[*]}
do
    CUDA_VISIBLE_DEVICES=${DEVICES} python -m latency \
    --gpu_id ${DEVICES} \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --quant_model_bits ${Q_BITS} \
    --config ${CONFIG} \
    --method ${METHOD} \
    --latency_table_file ${LATENCY_TABLE} \
    --greedy_result_path ${GREEDY_RESULTS} \
    --last_layer ${LAST_LAYER}
done
# --iteration ${ITERATION} \
# --batch_size ${BATCH_SIZE} \

# --backend ${BACKEND} \
# CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
