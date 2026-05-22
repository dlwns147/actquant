DEVICES=${1}

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf
# MODEL_NAME=Meta-Llama-3-8B
DATASET=wikitext2
# DATASET=c4

OUTPUT_DIR=/NAS/SJ/actquant/outlier/${MODEL_NAME}
# TARGET_RANK=32
TARGET_RANK="4 16 64 256"
WBITS=16


CUDA_VISIBLE_DEVICES=${DEVICES} python extract_outidx_multi.py \
${MODEL_PATH}/${MODEL_NAME} \
${DATASET} \
--no_frob_norm \
--wbits ${WBITS} \
--target_rank ${TARGET_RANK} \
--output_dir ${OUTPUT_DIR}