# LLaMA model with KIVI
import torch
import os
from models.llama_kivi_original import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
import time
from datasets import load_dataset
import gc
# from copy import deepcopy

# from models.cache import KIVIDynamicCache

K_BITS = 3
V_BITS = 3
GROUP_SIZE = 128
RESIDUAL_LENGTH = 128
BATCH_SIZE = 1
# PATH_TO_YOUR_SAVE_DIR = '/SSD/.cache'

# model_name_or_path = 'meta-llama/Llama-2-7b-hf'
model_name_or_path = 'meta-llama/Llama-3.1-8B-Instruct'
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_bits = K_BITS # current support 2/4 bit for KV Cache
config.v_bits = V_BITS # current support 2/4 bit for KV Cache
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH # the number of recent fp16 tokens
config.use_flash = True
# CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

if K_BITS < 16 and V_BITS < 16:
    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        # cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
else:
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        # cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

# past_key_values = KIVIDynamicCache()

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True)

model.cuda().eval()

model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]

# --- Load GSM8K dataset and create a prompt ---
dataset = load_dataset('gsm8k', 'main')

prompt = ''
for i in range(100):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
print(f"Prompt length: {inputs['input_ids'].shape[1]} tokens")
print(f"model: {model_name_or_path}")
print('======')

max_new_tokens = 1024

# --- Warm-up run ---
# past_key_values_copy = deepcopy(past_key_values)

print("Warming up...")
with torch.no_grad():
    _ = model.generate(**inputs, use_cache=True, max_new_tokens=10)
print("Warm-up finished.")

# del past_key_values_copy

gc.collect()
torch.cuda.empty_cache()


# --- Measurement ---
print(f"Generating {max_new_tokens} tokens...")
torch.cuda.synchronize()

memory_before = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    start_time = time.time()
    outputs = model.generate(**inputs, use_cache=True, max_new_tokens=max_new_tokens, return_dict_in_generate=True)
    # outputs = model.generate(**inputs, use_cache=True, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    end_time = time.time()

max_gpu_memory_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
print(f"Max GPU memory (peak): {max_gpu_memory_gb * 1024:.2f} MB")
memory_after = torch.cuda.memory_allocated()
memory_used_gb = (memory_after - memory_before) / 1024 / 1024 / 1024

# import code; code.interact("cache speed line 119", local=dict(globals(), **locals()))

elapsed_time = end_time - start_time
num_generated_tokens = outputs.sequences.shape[1] - inputs['input_ids'].shape[1]

if elapsed_time > 0:
    tokens_per_sec = num_generated_tokens / elapsed_time
else:
    tokens_per_sec = float('inf')

print(f"Generated {num_generated_tokens} tokens in {elapsed_time:.4f} seconds.")
print(f"Elapsed time: {elapsed_time:.4f} seconds")
print(f"Tokens per second: {tokens_per_sec:.2f}")
print(f"Max GPU memory (peak): {max_gpu_memory_gb:.2f} GB")
print(f"Memory used by generate: {memory_used_gb:.2f} GB")
