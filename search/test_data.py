import torch
from hqq.models.hf.base import AutoHQQHFModel

from utils.data import get_loader
from utils.eval import get_tokenizer
from utils.func import load_hqq_model, get_hfmodel
from model.replace import replace_model


# model_path = 'meta-llama'
# model_name = 'Llama-3.1-8B-Instruct'
model_path = 'mistralai'
model_name = 'Mistral-7B-Instruct-v0.3'

model_id = f'/SSD/huggingface/{model_path}/{model_name}'
hqq_path = f'/SSD/hqq/{model_name}_4bit_128gs_1axis_float16'
model = AutoHQQHFModel.from_quantized(hqq_path, device_map='cuda')
# model = get_hfmodel(model_id, dtype=torch.float16)
n_block = 32
model.config.k_bits = [4] * n_block
model.config.v_bits = [4] * n_block
model.config.k_group_size = [128] * n_block
model.config.v_group_size = [128] * n_block

model.config.use_flash = True
model.config.residual_length = 128
model.config.quant_kv_output = True
model.config.k_quant_per = 'channel'
model.config.v_quant_per = 'token'
model = replace_model(model, model.config)
tokenizer = get_tokenizer(model_id)
gsm8k_loader = get_loader('gsm8k', n_sample=32, train=True, seed=0, seqlen=256, batch_size=8, tokenizer=tokenizer)
wikitext2_loader = get_loader('wikitext2', n_sample=128, train=True, seed=0, seqlen=2048, batch_size=1, tokenizer=tokenizer)

# import pdb; pdb.set_trace()
for inputs, attention_mask, labels in gsm8k_loader:
    logits = model(inputs.cuda()).logits
    import pdb; pdb.set_trace()
    break
print(f'gsm8k logits: {logits[0]}, {logits.shape}')
# import pdb; pdb.set_trace()
for inputs, attention_mask, labels in wikitext2_loader:
    logits = model(inputs.cuda()).logits
    break
print(f'wikitext2 logits: {logits[0]}, {logits.shape}')