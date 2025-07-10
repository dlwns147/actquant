# LLaMA model with KIVI
import torch
import os
from models.KIVICache import KIVICacheConfig, KIVIDynamicCache
from transformers.cache_utils import DynamicCache, QuantoQuantizedCache, HQQQuantizedCache, QuantizedCacheConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datasets import load_dataset
import numpy as np
import json
import gc
from copy import deepcopy

from models.llama_kivi_4_50_3_generation import convert_generation_kivi
from models.llama_kivi_4_50_3 import convert_model_kivi

BITS = 4
K_BITS = BITS
V_BITS = BITS
GROUP_SIZE = 128
RESIDUAL_LENGTH = 128
BATCH_SIZE = 1

quanto_quantized_cache_config = QuantizedCacheConfig(backend="quanto", nbits=BITS, axis_key=0, axis_value=0, q_group_size=GROUP_SIZE, residual_length=RESIDUAL_LENGTH, compute_dtype=torch.float16, device="cuda")
hqq_quantized_cache_config = QuantizedCacheConfig(backend="HQQ", nbits=BITS, axis_key=0, axis_value=0, q_group_size=GROUP_SIZE, residual_length=RESIDUAL_LENGTH, compute_dtype=torch.float16, device="cuda")
# per_layer_config = {layer_idx: {'nbits_key': K_BITS, 'nbits_value': V_BITS} for layer_idx in range(model.config.num_hidden_layers)}
kivi_cache_config = KIVICacheConfig(nbits_key=K_BITS, 
                                    nbits_value=V_BITS, 
                                    q_group_size=GROUP_SIZE, 
                                    residual_length=RESIDUAL_LENGTH,
                                    per_layer_quant=False)

# model_name_or_path = 'meta-llama/Llama-2-7b-hf'
model_name_or_path = 'meta-llama/Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16,
                                             _attn_implementation="flash_attention_2", device_map="auto").cuda().eval()
model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)

# --- Load GSM8K dataset and create a prompt ---
dataset = load_dataset('gsm8k', 'main')

prompt = ''
for i in range(1000):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

inputs = inputs[:, : 64000]

print(f"Prompt length: {inputs.shape[1]} tokens")
print(f"model: {model_name_or_path}")
print('===============================================')

max_new_tokens = 1024

print(f"Generating {max_new_tokens} tokens...")

results = {}

# for past_key_values in [DynamicCache(), QuantoQuantizedCache(quanto_quantized_cache_config), HQQQuantizedCache(hqq_quantized_cache_config), KIVIDynamicCache(kivi_cache_config)]:
for past_key_values in [HQQQuantizedCache(hqq_quantized_cache_config), KIVIDynamicCache(kivi_cache_config)]:
    torch.cuda.synchronize()
    memory_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    if isinstance(past_key_values, KIVIDynamicCache):
        print("Replacing model & attention forward method using KIVI.")
        convert_model_kivi(model, kivi_cache_config)
        convert_generation_kivi(kivi_cache_config)

    with torch.inference_mode():
        # Prefill
        outputs = model(inputs, past_key_values=past_key_values, use_cache=True)
        print('prefill done')

        # Generation
        time_list = []
        for i in range(max_new_tokens):
            token = outputs.logits[:, -1].max(1)[1].unsqueeze(1)
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = model(token, past_key_values=past_key_values, use_cache=True)
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            time_list.append(end_time - start_time)

    torch.cuda.synchronize()
    memory_after = torch.cuda.memory_allocated()
    memory_used_gb = (memory_after - memory_before) / 1024 / 1024 / 1024
    elapsed_time = 1 / np.median(time_list)

    print(f"Cache type: {type(past_key_values).__name__}")
    print(f"Memory used by generate: {memory_used_gb:.2f} GB")
    print(f"Tokens per second: {elapsed_time:.2f}")

    results[type(past_key_values).__name__] = {
        'memory_used_gb': memory_used_gb,
        'tokens_per_second': elapsed_time
    }

    del past_key_values 
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

for key, value in results.items():
    print(f"Cache type: {key}")
    print(f"Memory used by generate: {value['memory_used_gb']:.2f} GB")
    print(f"Tokens per second: {value['tokens_per_second']:.2f}")

def pass_key(model, tokenizer, past_key_values, pass_key_file='/NAS/SJ/actquant/common_code/passkey_examples.jsonl'):
    print( "-----------------------------------" )

    original_past_key_values = deepcopy(past_key_values)

    for line in open(pass_key_file, "r"):
        torch.cuda.reset_max_memory_allocated() 

        past_key_values = deepcopy(original_past_key_values)

        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"] + prompt_postfix
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        print( "-----------------------------------" )
        print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
        print( "Passkey target:", example["target"] )

        tokens = model.generate(input_ids, past_key_values=past_key_values, use_cache=True, max_new_tokens=len(example["target"]))
        answer = prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
        answer = answer.replace("\n", "\\n")
        answer= f"[ {answer} ]"
        
        peak_memory = torch.cuda.max_memory_allocated()
        print( answer )
        print(f"Mem: {peak_memory / 1024 / 1024 / 1024:.3f} GB")
        print( "-----------------------------------\n" )

        del past_key_values
        gc.collect()
        torch.cuda.empty_cache()