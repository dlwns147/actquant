DEVICES=${1}
TODAY=`date +%y%m%d%H%M`

MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
CONFIG=config/llama.json
# PASS_LIST="1.self_attn.v_proj 1.mlp.down_proj 31.mlp.down_proj"

PASS_LIST="0.self_attn 0.mlp 1.self_attn 1.mlp 31.mlp"

MIN_SEC_OBJ=${SMALL_WBITS}
MAX_SEC_OBJ=${LARGE_WBITS}

PREDICTOR=mlp
OBJ=sparsity
DEBUG=True

N_DOE=64
N_ITER=64
ITER=64
GA_POP_SIZE=200
SUBSET_POP_SIZE=200
METRIC=loss

GA_ALGORITHM=nsga2
NAN_VALUE=5
MUT_PROB=0.1

MIN_SEC_OBJ=0.5
MAX_SEC_OBJ=1

METHOD='layer_prune'
LOSS_FUNC='jsd'
N_SAMPLE=64

SAVE=save/search/${MODEL_NAME}_${OBJ}_${METRIC}_${METHOD}_iter_${ITER}_n_iter_${N_ITER}_${GA_ALGORITHM}_${MIN_SEC_OBJ}_${MAX_SEC_OBJ}_${LOSS_FUNC}_mut_prob_${MUT_PROB}_ns_${N_SAMPLE}_${TODAY}
# RESUME=/NAS/SJ/nsgaquant/save/search/Llama-2-7b-hf_bits_loss_iter_300_nsga2_2_4_2410051059/iter_100.stats

CUDA_VISIBLE_DEVICES=${DEVICES} python prune.py \
--model_name ${MODEL_PATH}/${MODEL_NAME} \
--sec_obj ${OBJ} \
--predictor ${PREDICTOR} \
--save ${SAVE} \
--iterations ${ITER} \
--n_doe ${N_DOE} \
--n_iter ${N_ITER} \
--metric ${METRIC} \
--ga_pop_size ${GA_POP_SIZE} \
--subset_pop_size ${SUBSET_POP_SIZE} \
--config ${CONFIG} \
--debug ${DEBUG} \
--sec_obj_range ${MIN_SEC_OBJ} ${MAX_SEC_OBJ} \
--ga_algorithm ${GA_ALGORITHM} \
--method ${METHOD} \
--pass_layer_list ${PASS_LIST} \
--nan_value ${NAN_VALUE} \
--mut_prob ${MUT_PROB} \
--loss_func ${LOSS_FUNC} \
--n_sample ${N_SAMPLE}


# --sec_n_doe ${SEC_N_DOE} \
# --sec_n_iter ${SEC_N_ITER} \
# --sec_iter ${SEC_ITER} \
# --sec_metric ${SEC_METRIC} \
# --sec_ga_pop_size ${SEC_GA_POP_SIZE} \
# --resume ${RESUME}


# MIN_SEC_OBJ=2.19
# MAX_SEC_OBJ=2.21

# MIN_SEC_OBJ=2.39
# MAX_SEC_OBJ=2.41

# MIN_SEC_OBJ=2.59
# MAX_SEC_OBJ=2.61

# MIN_SEC_OBJ=2.79
# MAX_SEC_OBJ=2.81

# MIN_SEC_OBJ=2.99
# MAX_SEC_OBJ=3.01

# MIN_SEC_OBJ=3.19
# MAX_SEC_OBJ=3.21

# MIN_SEC_OBJ=3.39
# MAX_SEC_OBJ=3.41

# MIN_SEC_OBJ=3.59
# MAX_SEC_OBJ=3.61

# MIN_SEC_OBJ=3.79
# MAX_SEC_OBJ=3.81

# MIN_SEC_OBJ=2.495
# MAX_SEC_OBJ=2.505

# MIN_SEC_OBJ=2.745
# MAX_SEC_OBJ=2.755

# MIN_SEC_OBJ=2.995
# MAX_SEC_OBJ=3.005

# MIN_SEC_OBJ=3.495
# MAX_SEC_OBJ=3.505

# PREDICTOR_DATA=data/Llama-2-7b-hf_loss_1000_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json
# PREDICTOR_DATA=data/Llama-2-7b-hf_loss_300_range_2.9_3.1_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json
# PREDICTOR_DATA=data/Llama-2-7b-hf_loss_300_range_2_4_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.json

# SEC_N_DOE=50
# SEC_N_ITER=20
# SEC_ITER=20
# SEC_METRIC=ppl
# SEC_GA_POP_SIZE=50

# SEC_N_DOE=50
# SEC_N_ITER=20
# SEC_ITER=30
# SEC_METRIC=ppl
# SEC_GA_POP_SIZE=50
