DEVICES=${1}

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf

# MODEL=facebook/opt-6.7b
# MODEL=facebook/opt-13b
# MODEL=facebook/opt-30b
# MODEL=facebook/opt-66b

CONFIG=config/llama.json
N_SAMPLE=128

LOSS_CSV_FILE=csv/sensitivity/${MODEL_NAME}_layer_prune_loss.csv
PPL_CSV_FILE=csv/sensitivity/${MODEL_NAME}_layer_prune_ppl.csv

# CUDA_VISIBLE_DEVICES=${DEVICES} python -m layer_sensitivity \
# --model_name ${MODEL_PATH}/${MODEL_NAME} \
# --n_sample ${N_SAMPLE} \
# --loss_csv_file ${LOSS_CSV_FILE} \
# --ppl_csv_file ${PPL_CSV_FILE} \
# --config ${CONFIG}
# --eval_ppl \
# --eval_zeroshot \

CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=2 layer_sensitivity.py \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--n_sample ${N_SAMPLE} \
--loss_csv_file ${LOSS_CSV_FILE} \
--ppl_csv_file ${PPL_CSV_FILE} \
--config ${CONFIG}