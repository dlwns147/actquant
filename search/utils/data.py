# Import necessary modules
import json
import torch
import os
import glob
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, LlamaTokenizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from huggingface_hub import snapshot_download


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_tokenizer(model, use_fast=False, cache_dir=None, **kwargs):
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
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast, cache_dir=cache_dir)
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
    # Offline-safe c4 validation load: mmap arrow shards directly from cache,
    # bypassing the config-hash lookup. Cache layout (see dataset_info.json):
    #   ~/.cache/huggingface/datasets/allenai___c4/default-c7bc8b0aefc5e48f/
    #     0.0.0/<rev-hash>/c4-validation.arrow
    _base = cache_dir or os.path.expanduser('~/.cache/huggingface/datasets')
    _arrow_glob = os.path.join(_base, 'allenai___c4', 'default-*', '0.0.0', '*', 'c4-validation*.arrow')
    _arrow_files = sorted(glob.glob(_arrow_glob))
    if _arrow_files:
        valdata = concatenate_datasets([HFDataset.from_file(f) for f in _arrow_files])
    else:
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
    # Offline-safe c4 load: read arrow shards directly from the cache, bypassing
    # `datasets.load_dataset`'s config-hash lookup which never resolves to the
    # pre-built cache config (`default-b04fc8a0b8562884`) across datasets-lib
    # versions / data_files-spec shapes. Pure mmap, no network.
    _base = cache_dir or os.path.expanduser('~/.cache/huggingface/datasets')
    _arrow_glob = os.path.join(_base, 'allenai___c4', 'default-*', '0.0.0', '*', 'c4-train-*.arrow')
    _arrow_files = sorted(glob.glob(_arrow_glob))
    if _arrow_files:
        traindata = concatenate_datasets([HFDataset.from_file(f) for f in _arrow_files])
    else:
        # Fallback: online resolution (only if cache miss).
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train', cache_dir=cache_dir,
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


def get_gov_report(seed, n_sample, tokenizer, batch_size=1, seqlen=2048, split='train', min_seqlen=0, cache_dir=None):
    traindata = load_dataset('launch/gov_report', 'plain_text', split=split, cache_dir=cache_dir)
    
    # Shuffle and flatten indices
    traindata = traindata.shuffle(seed=seed)
    traindata = traindata.flatten_indices()
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Collect samples that meet min_seqlen requirement
    data_list = []
    for data in traindata:
        document = data['document']
        # Tokenize the document
        tokenized = tokenizer(document, add_special_tokens=False, return_tensors='pt', truncation=False)
        tokenized_length = tokenized['input_ids'].shape[1]
        
        # Filter by min_seqlen
        if tokenized_length < min_seqlen:
            continue
        
        # Truncate to seqlen
        tokenized = tokenizer(document, add_special_tokens=False, padding=True, truncation=True, max_length=seqlen, return_tensors='pt')
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        
        data_list.append([input_ids, attention_mask, input_ids])
        
        # Stop when we have n_sample samples
        if len(data_list) >= n_sample:
            break
    
    if len(data_list) < n_sample:
        raise ValueError(f"Could not find enough samples with min_seqlen={min_seqlen}. Found {len(data_list)}, required {n_sample}")
    
    tokenizer.pad_token = None
    # Concatenate all samples
    input_ids_dataset = torch.concat([x[0] for x in data_list], dim=0)
    attention_mask_dataset = torch.concat([x[1] for x in data_list], dim=0)
    labels_dataset = torch.concat([x[2] for x in data_list], dim=0)
    
    
    return DataLoader(TensorDataset(input_ids_dataset, attention_mask_dataset, labels_dataset), batch_size=batch_size)


# MiniLongBench (LongBench format): each example has input, context, answers (list), length, dataset, language, all_classes, _id.
# Prompt templates from LongBench (dataset2prompt.json). See: https://github.com/MilkThink-Lab/MiniLongBench


def _load_minilongbench_prompt_templates():
    path = os.path.join(os.path.dirname(__file__), "longbench_config", "dataset2prompt.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_minilongbench(tokenizer, cache_dir=None, require_answer=True, ignore_index=-100):
    """
    Load all MiniLongBench sub-datasets and build LLM-style examples.

    Uses LongBench prompt templates from utils/longbench_config/dataset2prompt.json.
    - require_answer=False: each example is prompt only (input_ids, attention_mask, labels=input_ids).
    - require_answer=True: prompt + answer; labels use ignore_index on the prompt part so loss is only on answer tokens.
    No shuffle, batch_size=1, no padding, no seqlen/min_seqlen.
    """
    root = snapshot_download(repo_id="linggm/MiniLongBench", repo_type="dataset", cache_dir=cache_dir)
    data_dir = os.path.join(root, "data")
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No *.jsonl under {data_dir}")

    dataset2prompt = _load_minilongbench_prompt_templates()
    all_examples = []
    for fp in files:
        sub_name = os.path.splitext(os.path.basename(fp))[0]
        if sub_name not in dataset2prompt:
            continue
        prompt_format = dataset2prompt[sub_name]
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                context = item["context"]
                input_str = item.get("input", "")
                prompt = prompt_format.format(context=context, input=input_str)
                if require_answer:
                    answers = item.get("answers") or []
                    target = answers[0] if answers else ""
                    text = prompt + target
                else:
                    text = prompt
                tokenized = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
                input_ids = tokenized["input_ids"][0]
                attention_mask = tokenized["attention_mask"][0]
                labels = input_ids.clone()
                if require_answer and target:
                    len_prompt = len(tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0])
                    labels[:len_prompt] = ignore_index
                all_examples.append((input_ids, attention_mask, labels))

    # dataset = Dataset(all_examples)
    return DataLoader(all_examples, batch_size=1)
    # class _MiniLongBenchDataset(Dataset):
    #     def __init__(self, examples):
    #         self.examples = examples

    #     def __len__(self):
    #         return len(self.examples)

    #     def __getitem__(self, i):
    #         return self.examples[i]

    # dataset = _MiniLongBenchDataset(all_examples)
    # return DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False)

def get_task3(tokenizer, n_sample=129, ignore_index=-100, max_len=2048):
    """Downstream-aware calibration: prompt+answer from FP16-correct gsm8k_cot/ifeval/mbpp
    samples (the proxy-study sample set). labels=ignore_index on the prompt so the
    search objective (JSD/forward_kl) is measured ONLY on the answer tokens — i.e. the
    search calibrates on the 3 downstream tasks rather than wikitext2. ~n_sample/3 per task."""
    import json as _json, glob as _glob
    SAMP = "/NAS/SJ/actquant/poc/benchmark_proxy/analyse_metric/samples/Llama-3.1-8B-Instruct"
    TASKS = {"gsm8k_cot": ("exact_match", "strict-match"),
             "ifeval": ("prompt_level_strict_acc", None),
             "mbpp": ("pass_at_1", None)}
    per = max(1, n_sample // len(TASKS))
    ex = []
    for task, (field, filt) in TASKS.items():
        fs = sorted(_glob.glob(f"{SAMP}/samples_{task}_*.jsonl"))
        if not fs:
            continue
        n = 0
        for line in open(fs[-1]):
            r = _json.loads(line)
            if filt is not None and r.get("filter") != filt:
                continue
            if not bool(r.get(field)):
                continue
            prompt = r["arguments"]["gen_args_0"]["arg_0"]
            ans = r["resps"][0][0]
            if not isinstance(ans, str) or not ans:
                continue
            ids = tokenizer(prompt + ans, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if ids.numel() < 2 or ids.numel() > max_len:
                continue
            plen = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].numel()
            labels = ids.clone()
            labels[:plen] = ignore_index
            if (labels != ignore_index).sum() == 0:
                continue
            am = torch.ones_like(ids)
            ex.append((ids, am, labels))
            n += 1
            if n >= per:
                break
    print(f"[get_task3] {len(ex)} calibration seqs (prompt+answer, answer-only loss) over {len(TASKS)} tasks")
    return DataLoader(ex, batch_size=1)


def get_trainloaders(name, n_sample=128, seed=0, seqlen=2048, model='', batch_size=1, cache_dir=None):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)
    if 'c4' in name:
        return get_c4_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)
    if 'gsm8k' in name:
        return get_gsm8k_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)

def get_loader(name, n_sample=128, train=True, seed=0, seqlen=2048, min_seqlen=0, batch_size=1, tokenizer=None, model='', cache_dir=None, sub_dataset=None, require_answer=False):
    if tokenizer is None:
        tokenizer = get_tokenizer(model, cache_dir=cache_dir)
    if "minilongbench" in name:
        return get_minilongbench(tokenizer=tokenizer, cache_dir=cache_dir, require_answer=require_answer)
    if "task3" in name:  # downstream-aware calibration (gsm8k+ifeval+mbpp answer tokens)
        return get_task3(tokenizer=tokenizer, n_sample=n_sample)
    if train:
        if 'wikitext2' in name:
            return get_wikitext2_trainenc(seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, tokenizer=tokenizer, cache_dir=cache_dir)
        if 'c4' in name:
            return get_c4_trainenc(seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, tokenizer=tokenizer, cache_dir=cache_dir)
        if 'gsm8k' in name:
            return get_gsm8k_trainenc(seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, min_seqlen=min_seqlen, tokenizer=tokenizer, cache_dir=cache_dir)
        if 'gov_report' in name:
            # return get_gov_report(seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, tokenizer=tokenizer, split='train', min_seqlen=min_seqlen, cache_dir=cache_dir)
            return get_gov_report(seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, tokenizer=tokenizer, split='test', min_seqlen=min_seqlen, cache_dir=cache_dir)
    else:
        if 'wikitext2' in name:
            return get_wikitext2(tokenizer=tokenizer, batch_size=batch_size, seqlen=seqlen, cache_dir=cache_dir)
        if 'c4' in name:
            return get_c4(tokenizer=tokenizer, batch_size=batch_size, seqlen=seqlen, cache_dir=cache_dir)
        if 'gsm8k' in name:
            return None
        if 'gov_report' in name:
            return get_gov_report(seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, tokenizer=tokenizer, split='test', min_seqlen=min_seqlen, cache_dir=cache_dir)
