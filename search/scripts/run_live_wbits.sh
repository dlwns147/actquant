#!/usr/bin/env bash
# Live wbits NAS search on the CURRENT transformers 4.57.6 env (ported via shims +
# sdpa attn + kivi_gemv on PYTHONPATH). Args: $1=GPU $2=loss_func(jsd|forward_kl) $3=SAVE
set -u
cd /NAS/SJ/actquant/search
export PYTHONPATH=/NAS/SJ/actquant/search/quant/kivi_utils:${PYTHONPATH:-}
G=$1; LF=$2; SAVE=$3; DSET=${4:-wikitext2}
B=/SSD/hqq/Llama-3.1-8B-Instruct
MAXV=0.7   # JSD bounded by ln2; forward_kl is UNbounded -> wider surrogate cap
if [ "$LF" = "forward_kl" ]; then MAXV=5.0; fi
CUDA_VISIBLE_DEVICES=$G python -u search.py \
  --gpu_id 0 --model_path /SSD/huggingface/meta-llama --model_name Llama-3.1-8B-Instruct --dtype bfloat16 \
  --config config/llama.json \
  --quant_model_paths ${B}_2bit_128gs_1axis_bfloat16 ${B}_3bit_128gs_1axis_bfloat16 ${B}_4bit_128gs_1axis_bfloat16 \
  --w_method hqq --kv_method kivi --w_bits 2 3 4 --k_bits 4 --v_bits 4 --w_group_size 128 \
  --k_group_size 128 --v_group_size 128 --k_quant_scheme channel --v_quant_scheme token \
  --k_pruning_dim 0 --v_pruning_dim 0 \
  --comp_obj wbits --comp_obj_min 2 --comp_obj_max 5 --n_token 0 \
  --residual_length 128 --attn_sink 0 \
  --predictor rbf --metric loss --loss_func $LF \
  --iterations 60 --n_doe 256 --n_iter 30 --ga_pop_size 200 --max_value $MAXV \
  --mut_prob 0.1 --crossover_prob 0.9 \
  --n_sample 128 --data_batch_size 1 --seqlen 2048 --min_seqlen 0 --dataset $DSET \
  --stride 128 --prefill_prompt --last_tokens 512 --save_iter 5 \
  --save $SAVE
echo "LIVE_SEARCH_DONE $LF $(date)"
