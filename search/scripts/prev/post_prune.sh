DEVICES=${1}
TODAY=`date +%y%m%d%H%M`

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
CONFIG=config/llama.json

OBJ=sparsity
PREFER="metric#2.0 sparsity#0.8"
EXPR_FOLDER=save/search

METHOD=layer_prune

# EXPR_FILE=Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_n_iter_32_nsga2_0.5_1_jsd_0.05_pntx_2410301802/iter_63.stats
EXPR_FILE=Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_n_iter_32_nsga2_0.5_1_jsd_0.05_2410301746/iter_63.stats
# EXPR_FILE=Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_n_iter_32_nsga2_0.5_1_jsd_0.2_2410301638/iter_63.stats
# EXPR_FILE=Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_n_iter_32_nsga2_0.5_1_jsd_0.1_2410291429/iter_63.stats
# EXPR_FILE=Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_n_iter_32_nsga2_0.5_1_0.1_2410281403/iter_63.stats
# EXPR_FILE=Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_nsga2_0.5_1_0.1_2410281059/iter_63.stats
# EXPR_FILE=Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_nsga2_0.5_1_0.1_2410281057/iter_63.stats
# EXPR_FILE=Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_nsga2_0.7_1_0.05_2410280839/iter_63.stats
# EXPR_FILE=Llama-2-7b-hf_sparsity_loss_layer_prune_iter_64_nsga2_0.7_1_0.02_2410280801/iter_32.stats

MIN_OBJ=0.78
MAX_OBJ=0.81
# TARGET_BITS_RANGE="${MIN_BITS} ${MAX_BITS}"
SAVE=save/result/${TODAY}_${METHOD}_${MIN_OBJ}_${MAX_OBJ}
N=1
DATASETS=wikitext2

CUDA_VISIBLE_DEVICES=${DEVICES} python post_prune.py \
    --model_name ${MODEL_PATH}/${MODEL_NAME} \
    --config ${CONFIG} \
    --sec_obj ${OBJ} \
    -n ${N} \
    --save ${SAVE} \
    --expr ${EXPR_FOLDER}/${EXPR_FILE} \
    --datasets ${DATASETS} \
    --only_front True \
    --target_obj_range ${MIN_OBJ} ${MAX_OBJ} \
    --method ${METHOD}
    # --prefer ${PREFER} \
