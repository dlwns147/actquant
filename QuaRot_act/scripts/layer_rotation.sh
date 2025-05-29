DEVICES=${1}

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
N_LAYER=32

# MODEL_NAME=Llama-2-13b-hf

# DATASET=wikitext2
DATASET=c4

# WBITS=2
WBITS=4

# ABITS=8
# ABITS=4
ABITS=16

# VBITS=4
VBITS=16

# KBITS=4
KBITS=16

WGROUP=128
# WGROUP=-1

AGROUP=128
# AGROUP=-1

# VGROUP=128
VGROUP=-1

# KGROUP=128
KGROUP=-1

PPL_BATCH_SIZE=1

CUDA_VISIBLE_DEVICES=${DEVICES} python fake_quant/layer_rotation.py \
--model ${MODEL_PATH}/${MODEL_NAME} \
--rotate \
--w_bits ${WBITS} \
--a_bits ${ABITS} \
--v_bits ${VBITS} \
--k_bits ${KBITS} \
--cal_dataset ${DATASET} \
--a_groupsize ${AGROUP} \
--w_groupsize ${WGROUP} \
--v_groupsize ${VGROUP} \
--k_groupsize ${KGROUP} \
--bsz ${PPL_BATCH_SIZE} \
--n_layer ${N_LAYER}