
## Measure Linear Sensitivity 
> scripts/linear_sensitivity.sh

```
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --num_machines=1 --main_process_port=12345 
linear_sensitivity.py \
--gpu_id 0 \ # Use the same string with CUDA_VISIBLE_DEVICES
--model_path /SSD/huggingface/meta-llama \
--model_name Llama2-7b-hf \
--method hqq \
--quant_model_paths "2bit_model_path 3bit_model_path 4bit_model_path" \
--quant_model_bits "2 3 4" \
--n_sample 128 \
--loss_csv_file path/to/save/loss \
--ppl_csv_file path/to/save/ppl \
--config config/llama.json \
--loss_func jsd
```

## Search
> scripts/search.sh
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --num_machines=1 --main_process_port=12345 search.py \
--gpu_id 0 \ # Use the same string with CUDA_VISIBLE_DEVICES
--model_path /SSD/huggingface/meta-llama \
--model_name Llama2-7b-hf \
--method hqq \
--quant_model_paths "2bit_model_path 3bit_model_path 4bit_model_path" \
--quant_model_bits "2 3 4" \
--sec_obj bits \
--predictor mlp \
--save path/to/save \
--iterations 300 \
--n_doe 250 \
--n_iter 50 \
--metric loss \
--ga_pop_size 200 \
--config config/llama.json \
--debug \
--sec_obj_range 2 4 \
--ga_algorithm nsga2 \
--max_value 5 \
--mut_prob 0.05 \
--pass_linear_list "0.self_attn.v_proj 1.self_attn.v_proj 1.mlp.down_proj 31.mlp.down_proj" \
--layer_prune_range 1 1 \
--loss_func jsd \
# --use_linear_group
```

## Post Search
> scripts/post_search.sh
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --num_machines=1 --main_process_port=12345 post_search.py \
--model_path /SSD/huggingface/meta-llama \
--model_name Llama2-7b-hf \
--config config/llama.json \
--method hqq
--quant_model_paths "2bit_model_path 3bit_model_path 4bit_model_path" \
--quant_model_bits "2 3 4" \
--sec_obj bits \
-n 5 \
--save path/to/save \
--debug \
--expr path/to/search/results \
--prefer "metric#0.0 bits#3.0" \
--datasets wikitext2 \
--target_bits_range 2.995 3.005 \
```