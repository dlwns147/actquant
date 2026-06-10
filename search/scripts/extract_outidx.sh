DEVICES=${1}

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf
# MODEL_NAME=Meta-Llama-3-8B
DATASET=wikitext2
# DATASET=c4

OUTPUT_DIR=/NAS/SJ/actquant/outlier/${MODEL_NAME}
WBITS=16

# One or more outlier-column counts → dict output {key: {n_out: [idx]}}
TARGET_RANK="32 64 96 128"
# TARGET_RANK="32"


CUDA_VISIBLE_DEVICES=${DEVICES} python extract_outidx.py \
${MODEL_PATH}/${MODEL_NAME} \
${DATASET} \
--no_frob_norm \
--wbits ${WBITS} \
--target_rank ${TARGET_RANK} \
--output_dir ${OUTPUT_DIR}
