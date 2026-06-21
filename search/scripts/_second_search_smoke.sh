#!/usr/bin/env bash
# SMOKE launcher (debug second_search.py + L2 freeze end-to-end). Written via Bash to dodge
# the Cursor-held inode on scripts/second_search.sh. NEW interface (no --mode), tiny params.
DEVICES=${1:-1}
PORT_NUM=$(( ( RANDOM % 10000 ) + 20000 ))
MODEL_PATH=/SSD/huggingface/meta-llama
MODEL_NAME=Llama-3.1-8B-Instruct
DTYPE=bfloat16
CONFIG=config/llama.json
W_EXPR="save/search/think/2606070017_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_128stride_pp512"
EFF_KV_EXPR="save/search/think/2606181423_Llama-3.1-8B-Instruct_eff_kvbits_kivi_sk8_w4kv234_gs3264128x2_128_r128_kd0-64x5_vd0-64x5_obj_0.1_5_st128_pp512"
W_GROUP_SIZE=128; AXIS=1
QMODEL_PATHS=""
for B in 2 3 4; do QMODEL_PATHS+=" /SSD/hqq/${MODEL_NAME}_${B}bit_${W_GROUP_SIZE}gs_${AXIS}axis_${DTYPE}"; done
SAVE=save/second_search/_smoke_$(date +%y%m%d%H%M)_sk8_af095
echo "SMOKE -> ${SAVE}"
CUDA_VISIBLE_DEVICES=${DEVICES} accelerate launch --num_processes=1 --num_machines=1 --main_process_port=${PORT_NUM} \
  second_search.py --config ${CONFIG} --model_name ${MODEL_NAME} \
  --w_expr "${W_EXPR}" --eff_kv_expr "${EFF_KV_EXPR}" \
  --surrogate carts --pop 100 --n_doe 24 --iterations 1 --n_iter 6 \
  --attn_sink 8 --n_token 0 --front_eps_rel 0.3 --div_k 200 --agree_frac 0.95 \
  --seed 0 --save_iter 1 --debug --save ${SAVE} \
  --gpu_id ${DEVICES} --model_path ${MODEL_PATH} --dtype ${DTYPE} \
  --w_method hqq --kv_method kivi think --quant_model_paths ${QMODEL_PATHS} \
  --w_bits 2 3 4 --w_group_size ${W_GROUP_SIZE} \
  --residual_length 128 --k_quant_scheme channel --v_quant_scheme token \
  --dataset wikitext2 --n_sample 32 --seqlen 2048 --loss_func jsd \
  --stride 128 --prefill_prompt --last_tokens 512
