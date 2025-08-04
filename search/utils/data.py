# Import necessary modules
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer
from torch.utils.data import DataLoader, TensorDataset

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_tokenizer(model, cache_dir=None):
    # if "llama" in model.lower():
    #     tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False, cache_dir=cache_dir)
    #     # fix for transformer 4.28.0.dev0 compatibility
    #     if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
    #         try:
    #             tokenizer.bos_token_id = 1
    #             tokenizer.eos_token_id = 2
    #         except AttributeError:
    #             pass
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir=cache_dir)
    return tokenizer

def get_wikitext2(tokenizer, seqlen=2048, batch_size=1, cache_dir=None):
    
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=cache_dir)

    # # trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    # testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt').input_ids
    # n_sample = testenc.numel() // seqlen
    # testenc = testenc[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    # return DataLoader(testenc, batch_size=batch_size, drop_last=False)

    tokenized = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
    n_sample = input_ids.numel() // seqlen
    input_ids = input_ids[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    attention_mask = attention_mask[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    return DataLoader(TensorDataset(input_ids, attention_mask, input_ids), batch_size=batch_size, drop_last=False)

def get_c4(tokenizer, seqlen=2048, batch_size=1, cache_dir=None):
   
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', cache_dir=cache_dir)

    # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # valenc = valenc.input_ids[:, :(256 * seqlen)]
    # n_sample = valenc.numel() // seqlen
    # valenc = valenc[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    # return DataLoader(valenc, batch_size=batch_size, drop_last=False)

    tokenized = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    input_ids, attention_mask = tokenized['input_ids'][:, :(256 * seqlen)], tokenized['attention_mask'][:, :(256 * seqlen)]
    n_sample = input_ids.numel() // seqlen
    input_ids = input_ids[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    attention_mask = attention_mask[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    return DataLoader(TensorDataset(input_ids, attention_mask, input_ids), batch_size=batch_size, drop_last=False)

def get_wikitext2_trainenc(seed, n_sample, tokenizer, batch_size=1, seqlen=2048, cache_dir=None):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
    traindata = traindata.shuffle(seed=seed)
    
    # trainenc = tokenizer("\n\n".join(traindata[:n_sample]['text']), return_tensors='pt').input_ids
    # n_sample = trainenc.numel() // seqlen
    # trainenc = trainenc[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    # return DataLoader(trainenc, batch_size=batch_size)

    tokenized = tokenizer("\n\n".join(traindata[:n_sample]['text']), return_tensors='pt')
    input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
    n_sample = input_ids.numel() // seqlen
    input_ids = input_ids[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    attention_mask = attention_mask[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    return DataLoader(TensorDataset(input_ids, attention_mask, input_ids), batch_size=batch_size)


def get_c4_trainenc(seed, n_sample, tokenizer, batch_size=1, seqlen=2048, cache_dir=None):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', cache_dir=cache_dir
    )
    traindata = traindata.shuffle(seed=seed)
    
    # trainenc = tokenizer(' '.join(traindata[:n_sample]['text']), return_tensors='pt').input_ids
    # n_sample = trainenc.numel() // seqlen
    # trainenc = trainenc[:, :n_sample * seqlen].reshape(n_sample, seqlen)    
    # return DataLoader(trainenc, batch_size=batch_size, drop_last=True)

    tokenized = tokenizer(' '.join(traindata[:n_sample]['text']), return_tensors='pt')
    input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
    n_sample = input_ids.numel() // seqlen
    input_ids = input_ids[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    attention_mask = attention_mask[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    return DataLoader(TensorDataset(input_ids, attention_mask, input_ids), batch_size=batch_size, drop_last=True)
    


def get_gsm8k_trainenc(seed, n_sample, tokenizer, batch_size=1, seqlen=2048, min_seqlen=0, cache_dir=None, ignore_index=-100):
    traindata = load_dataset('gsm8k', 'main', split='train', cache_dir=cache_dir)
    traindata = traindata.shuffle(seed=seed)    
    count = 0
    data_list = []
    for data in traindata:
        prompt = f"Question: {data['question']}\nAnswer: "
        # prompt = f"Q: {data['question']}\nA: "
        # prompt = f"Q: {data['question']}\nA: Let's think step by step. "
        target = data['answer'].replace('\n', ' ')
        
        tokenized = tokenizer(prompt + target, return_tensors='pt')
        input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
        len_prompt_target = input_ids.shape[-1]
        len_prompt = len(tokenizer(prompt)["input_ids"])
        # print(f'prompt|{prompt}, target|{target}')
        # print(f'{prompt + target}')
        print(f'count: {count}, len_prompt_target: {len_prompt_target}, len_prompt: {len_prompt}, len_target: {len_prompt_target - len_prompt}')
        if len_prompt_target > seqlen or len_prompt_target < min_seqlen:
            continue
        input_ids = torch.column_stack([input_ids, torch.zeros((1, seqlen - len_prompt_target), dtype=int)])
        attention_mask = torch.column_stack([attention_mask, torch.zeros((1, seqlen - len_prompt_target), dtype=int)])
        labels = input_ids.detach().clone()
        labels[0, :len_prompt] = ignore_index
        labels[0, len_prompt_target:] = ignore_index
        data_list.append([input_ids, attention_mask, labels])
        count += 1
        if count == n_sample:
            break
    if count < n_sample:
        raise NotImplementedError(f"'seqlen' is too small to generate a calibration dataset, calibration dataset size: {count}, target n_sample: {n_sample}")
    input_ids_dataset = torch.concat([x[0] for x in data_list], dim=0)
    attention_mask_dataset = torch.concat([x[1] for x in data_list], dim=0)
    labels_dataset = torch.concat([x[2] for x in data_list], dim=0)
    
    return DataLoader(TensorDataset(input_ids_dataset, attention_mask_dataset, labels_dataset), batch_size=batch_size)

def get_trainloaders(name, n_sample=128, seed=0, seqlen=2048, model='', batch_size=1, cache_dir=None):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)
    if 'c4' in name:
        return get_c4_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)
    if 'gsm8k' in name:
        return get_gsm8k_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)

def get_loader(name, n_sample=128, train=True, seed=0, seqlen=2048, min_seqlen=0, batch_size=1, tokenizer=None, model='', cache_dir=None):
    if tokenizer is None:
        tokenizer = get_tokenizer(model, cache_dir=cache_dir)
    if train:
        if 'wikitext2' in name:
            return get_wikitext2_trainenc(seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, tokenizer=tokenizer, cache_dir=cache_dir)
        if 'c4' in name:
            return get_c4_trainenc(seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, tokenizer=tokenizer, cache_dir=cache_dir)
        if 'gsm8k' in name:
            return get_gsm8k_trainenc(seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, min_seqlen=min_seqlen, tokenizer=tokenizer, cache_dir=cache_dir)
    else:
        if 'wikitext2' in name:
            return get_wikitext2(tokenizer=tokenizer, batch_size=batch_size, seqlen=seqlen, cache_dir=cache_dir)
        if 'c4' in name:
            return get_c4(tokenizer=tokenizer, batch_size=batch_size, seqlen=seqlen, cache_dir=cache_dir)
        if 'gsm8k' in name:
            return None
