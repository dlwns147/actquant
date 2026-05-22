DEVICES=${1}

MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf


# WBITS=2
WBITS=4
# WBITS=16

# ABITS=8
ABITS=4
# ABITS=16

VBITS=4
# VBITS=16

KBITS=4
# KBITS=16

# WGROUP=128
WGROUP=-1

# AGROUP=128
AGROUP=-1

# VGROUP=128
VGROUP=-1

KGROUP=128
# KGROUP=-1

PPL_BATCH_SIZE=1
# PPL_BATCH_SIZE=32
# PPL_BATCH_SIZE=64
# PPL_BATCH_SIZE=128

LOSS_FUNC=jsd
# LOSS_FUNC=cross_entropy

# GPTQ_DATASET=wikitext2
GPTQ_DATASET=c4

LOSS_DATASET=wikitext2

# SENSITIVITY_MODULE=a
SENSITIVITY_MODULE=v
# SENSITIVITY_MODULE=k

LOSS_FILE=csv/${MODEL_NAME}_linear_${SENSITIVITY_MODULE}_sensitivity_loss_w${WBITS}a${ABITS}v${VBITS}k${KBITS}bits_w${WGROUP}a${AGROUP}v${VGROUP}k${KGROUP}gs_${PPL_BATCH_SIZE}bs_${GPTQ_DATASET}_${LOSS_DATASET}_${LOSS_FUNC}.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_a_sensitivity_loss_w${WBITS}a${ABITS}v${VBITS}k${KBITS}bits_w${WGROUP}a${AGROUP}v${VGROUP}k${KGROUP}gs_${PPL_BATCH_SIZE}bs_${GPTQ_DATASET}_${LOSS_DATASET}_${LOSS_FUNC}.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_k_sensitivity_loss_w${WBITS}a${ABITS}v${VBITS}k${KBITS}bits_w${WGROUP}a${AGROUP}v${VGROUP}k${KGROUP}gs_${PPL_BATCH_SIZE}bs_${GPTQ_DATASET}_${LOSS_DATASET}_${LOSS_FUNC}.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_v_sensitivity_loss_w${WBITS}a${ABITS}v${VBITS}k${KBITS}bits_w${WGROUP}a${AGROUP}v${VGROUP}k${KGROUP}gs_${PPL_BATCH_SIZE}bs_${GPTQ_DATASET}_${LOSS_DATASET}_${LOSS_FUNC}.csv

# PPL_FILE=csv/${MODEL_NAME}_linear_k_sensitivity_ppl_w${WBITS}a${ABITS}v${VBITS}k${KBITS}bits_w${WGROUP}a${AGROUP}v${VGROUP}k${KGROUP}gs_${PPL_BATCH_SIZE}bs_${LOSS_FUNC}.csv
MIN_BITS=2


CUDA_VISIBLE_DEVICES=${DEVICES} python fake_quant/linear_sensitivity.py \
--model ${MODEL_PATH}/${MODEL_NAME} \
--rotate \
--w_bits ${WBITS} \
--a_bits ${ABITS} \
--v_bits ${VBITS} \
--k_bits ${KBITS} \
--a_groupsize ${AGROUP} \
--w_groupsize ${WGROUP} \
--v_groupsize ${VGROUP} \
--k_groupsize ${KGROUP} \
--bsz ${PPL_BATCH_SIZE} \
--sensitivity_file ${LOSS_FILE} \
--min_bits ${MIN_BITS} \
--w_clip \
--w_asym \
--cal_dataset ${GPTQ_DATASET} \
--loss_dataset ${LOSS_DATASET} \
--loss_func ${LOSS_FUNC} \
--sensitivity_module ${SENSITIVITY_MODULE}
# --eval_layerwise \


# LOSS_FILE=csv/${MODEL_NAME}_test.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_act_sensitivity_w${WGROUP}a${AGROUP}v${VGROUP}k${KGROUP}_group_test.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_act_sensitivity_loss_w${WBITS}a${ABITS}v${VBITS}k${KBITS}bits_w${WGROUP}a${AGROUP}v${VGROUP}k${KGROUP}gs_${PPL_BATCH_SIZE}bs_${LOSS_FUNC}.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_act_sensitivity_w${WGROUP}a${AGROUP}v${VGROUP}k${KGROUP}_group_${PPL_BATCH_SIZE}_bs.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_k_sensitivity_w${WGROUP}a${AGROUP}v${VGROUP}k${KGROUP}_group.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_v_sensitivity_w${WGROUP}a${AGROUP}v${VGROUP}k${KGROUP}_group.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_act_sensitivity_test.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_k_sensitivity.csv
# LOSS_FILE=csv/${MODEL_NAME}_linear_v_sensitivity.csv
