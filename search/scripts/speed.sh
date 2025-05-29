DEVICES=${1}


MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_NAME=Llama-2-7b-hf
MODEL_NAME=Llama-2-13b-hf
CUDA_VISIBLE_DEVICES=${DEVICES} python speed.py \
${MODEL_PATH}/${MODEL_NAME} \
--use_ft