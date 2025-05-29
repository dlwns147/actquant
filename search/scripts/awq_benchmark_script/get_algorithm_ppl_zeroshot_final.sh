DEVICES=2
TODAY=`date +%y%m%d%H%M`
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

# MODEL_PATH=/SSD/huggingface/meta-llama
# MODEL_PATH=/SSD/.cache/
# MODEL_NAME=Llama-2-7b-hf
MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
CONFIG=config/llama.json

METHOD=awq

SEQLEN=2048
N_SAMPLE=128

OUTLIER_PATH=/NAS/SJ/nsgaquant/outlier/${MODEL_NAME}/w16_r${N_OUTLIER}/outlier.pth

# DATASETS=( "wikitext2" "c4" )
DATASETS=( "wikitext2" )

# OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/${MODEL_NAME}-${METHOD}.csv
GROUP_SIZE=128
TARGET_BITS=( 2 3 4 )

# ARCH_PATH=/NAS/Woo/Automation/autoopt/archs/HQQ_woPrior_random_linear_wINT3_meta-llama_Llama-2-7b-hf.json
# ARCH_PATH=/NAS/Woo/Automation/autoopt/archs/HQQ_woPrior_random_linear_wINT3_meta-llama_Llama-2-13b-hf.json
ARCH_PATH=/NAS/Woo/Automation/autoopt/archs/final_search_7b_candidate.json
# ARCH_PATH=/NAS/Woo/Automation/autoopt/archs/final_search_13b_candidate.json

# OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/group_size_${GROUP_SIZE}/${MODEL_NAME}-${METHOD}.csv
OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/awq_final_search_sample/n_samples_128/${MODEL_NAME}-${METHOD}.csv
# OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/awq_final_search_sample/${MODEL_NAME}-${METHOD}.csv

N_PROC=1
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} get_algorithm_ppl_zeroshot.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--config ${CONFIG} \
--gpu_id ${DEVICES} \
--method ${METHOD} \
--seqlen ${SEQLEN} \
--n_sample ${N_SAMPLE} \
--eval_datasets ${DATASETS[@]} \
--output_path ${OUTPUT_PATH} \
--target_bits ${TARGET_BITS[@]} \
--group_size ${GROUP_SIZE} \
--arch_path ${ARCH_PATH}
# --half
# --do_clip_sym

# --zeroshot \

# --do_prune \
# --do_owq \
# OUTPUT_PATH=/NAS/Woo/Automation/autoopt/result/get_algorithm_ppl_zeroshot/group_size_${GROUP_SIZE}/${MODEL_NAME}-${METHOD}_sym.csv

# N_PROC=1
# CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} get_algorithm_ppl_zeroshot.py \
# --model_path ${MODEL_PATH} \
# --model_name ${MODEL_NAME} \
# --config ${CONFIG} \
# --gpu_id ${DEVICES} \
# --method ${METHOD} \
# --seqlen ${SEQLEN} \
# --n_sample ${N_SAMPLE} \
# --eval_datasets ${DATASETS[@]} \
# --zeroshot \
# --output_path ${OUTPUT_PATH} \
# --target_bits ${TARGET_BITS[@]} \
# --group_size ${GROUP_SIZE} \

# --do_prune \
# --do_owq \
# --output_path ${OUTPUT_PATH} \