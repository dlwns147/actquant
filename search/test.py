import torch
from hqq.models.hf.base import AutoHQQHFModel

from utils.data import get_loader
from utils.eval import get_tokenizer
from utils.func import load_hqq_model, get_hfmodel
from utils.loss import compute_longppl, get_key_token_list
from model.replace import replace_model

from datasets import load_dataset

# model_path = 'meta-llama'
# model_name = 'Llama-3.1-8B-Instruct'
# model_path = 'mistralai'
# model_name = 'Mistral-7B-Instruct-v0.3'
model_path = 'Qwen'
model_name = 'Qwen2.5-7B'
model_id = f'/SSD/huggingface/{model_path}/{model_name}'

# eval_model_path = 'meta-llama'
# eval_model_name = 'Llama-3.1-8B-Instruct'
eval_model_path = 'Qwen'
eval_model_name = 'Qwen2.5-14B'
eval_model_id = f'/SSD/huggingface/{model_path}/{model_name}'

hqq_path = f'/SSD/hqq/{model_name}_4bit_128gs_1axis_float16'
dataset = 'wikitext2'

# n_sample = 32
n_sample = 128

seqlen = 2048
# seqlen = 8192
# seqlen = 8960
batch_size = 1
seed = 0
alpha = 2
beta = -2

# trunc_len = 4096
# trunc_len = 1024
# trunc_len = 512
trunc_len = 256
# trunc_len = 128

# sliding_window = 1024
# sliding_window = 512
# sliding_window = 256
# sliding_window = 128
sliding_window = 64
# sliding_window = 32

tokenizer = get_tokenizer(model_id, use_fast=True)
eval_tokenizer = get_tokenizer(eval_model_id, use_fast=True)

loader = get_loader('wikitext2', n_sample=n_sample, train=True, seed=seed, seqlen=seqlen, batch_size=batch_size, tokenizer=tokenizer)
text_list = []
for input_ids_batch, _, _ in loader:
    for input_ids in input_ids_batch:
        text_list.append(tokenizer.decode(input_ids))
# import pdb; pdb.set_trace()  

# traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
# traindata = traindata.shuffle(seed=seed)
# text = "\n\n".join(traindata[:n_sample]['text'])

# model = AutoHQQHFModel.from_quantized(hqq_path, device_map='cuda')
model = get_hfmodel(model_id, dtype=torch.float16)
eval_model = get_hfmodel(eval_model_id, dtype=torch.float16)

print(get_key_token_list(model=eval_model, tokenizer=eval_tokenizer, loader=loader, trunc_len=trunc_len, sliding_window=sliding_window, alpha=alpha, beta=beta))

print(compute_longppl(text_list, model=model, evaluator_model=eval_model, tokenizer=tokenizer, evaluator_tokenizer=eval_tokenizer, trunc_len=trunc_len, sliding_window=sliding_window, alpha=alpha, beta=beta))
print(f'Mem: {(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024):.3f} GB')
print(f'trunc_len: {trunc_len}, sliding_window: {sliding_window}')

# loader = get_loader(dataset, n_sample=n_sample, train=True, seqlen=seqlen, batch_size=batch_size, model=model, tokenizer=tokenizer)
