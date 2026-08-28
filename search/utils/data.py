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

def get_wikitext2_trainenc(seed, n_sample, tokenizer, batch_size=1, seqlen=2048, cache_dir=None,
                           add_special_tokens=True):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
    traindata = traindata.shuffle(seed=seed)
    
    # trainenc = tokenizer("\n\n".join(traindata[:n_sample]['text']), return_tensors='pt').input_ids
    # n_sample = trainenc.numel() // seqlen
    # trainenc = trainenc[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    # return DataLoader(trainenc, batch_size=batch_size)

    # add_special_tokens=False is what the chat path needs: the BOS then comes
    # from the chat template's own prefix instead of the stream (otherwise
    # sample 0 would carry TWO).
    tokenized = tokenizer("\n\n".join(traindata[:n_sample]['text']), return_tensors='pt',
                          add_special_tokens=add_special_tokens)
    input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
    n_sample = input_ids.numel() // seqlen
    input_ids = input_ids[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    attention_mask = attention_mask[:, :n_sample * seqlen].reshape(n_sample, seqlen)
    return DataLoader(TensorDataset(input_ids, attention_mask, input_ids), batch_size=batch_size)


def get_c4_trainenc(seed, n_sample, tokenizer, batch_size=1, seqlen=2048, cache_dir=None,
                    add_special_tokens=True):
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

    tokenized = tokenizer(' '.join(traindata[:n_sample]['text']), return_tensors='pt',
                          add_special_tokens=add_special_tokens)
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
    
    # Floor is max(min_seqlen, seqlen): kept documents are truncated to `seqlen`
    # and torch.concat'ed, so anything SHORTER than seqlen breaks the concat
    # ("Expected size {seqlen} but got size {len}").
    length_floor = max(int(min_seqlen), int(seqlen))
    data_list = []
    for data in traindata:
        document = data['document']
        # Tokenize the document
        tokenized = tokenizer(document, add_special_tokens=False, return_tensors='pt', truncation=False)
        tokenized_length = tokenized['input_ids'].shape[1]

        # Filter by the length floor
        if tokenized_length < length_floor:
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
        raise ValueError(
            f"Could not find enough gov_report documents of >= {length_floor} tokens "
            f"(= max(min_seqlen={min_seqlen}, seqlen={seqlen})). Found {len(data_list)}, "
            f"required n_sample={n_sample}. Lower --seqlen/--n_sample, or use a split "
            f"with longer documents.")
    
    tokenizer.pad_token = None
    # Concatenate all samples
    input_ids_dataset = torch.concat([x[0] for x in data_list], dim=0)
    attention_mask_dataset = torch.concat([x[1] for x in data_list], dim=0)
    labels_dataset = torch.concat([x[2] for x in data_list], dim=0)
    
    
    return DataLoader(TensorDataset(input_ids_dataset, attention_mask_dataset, labels_dataset), batch_size=batch_size)


# LongBench subsets usable as a LONG-DOCUMENT PPL corpus. Only subsets whose
# `context` is ONE coherent document qualify: the point of a long window is
# long-range DEPENDENCY, and a context built by concatenating unrelated
# passages (hotpotqa / 2wikimqa / musique / passage_*) or few-shot examples
# (trec / triviaqa / samsum) gives length without dependency — i.e. the same
# defect as the c4 document-join, so it measures nothing wikitext2 doesn't.
# Measured context lengths (Llama-3.1 tokenizer, first 80 test docs):
#   narrativeqa  median 31,284 tok, 77/80 >= 8192, 56/80 >= 16384  (books/scripts)
#   qmsum        median 12,934 tok, 67/80 >= 8192                  (meeting transcripts)
# Registering a subset here as a metric corpus is fine even when LongBench
# GRADES it (qmsum): nothing is fit to the text, so it serves as a long-context
# REPORTING metric beside benchmark accuracy. It just cannot double as evidence
# that the metric PREDICTS that benchmark — correlation.py flags those cells.
# tests/audit_corpus_contamination.py prints which subsets are graded.
LONGBENCH_PPL_SUBSETS = ('narrativeqa', 'qmsum', 'gov_report', 'multifieldqa_en',
                         'qasper', 'multi_news', 'lcc', 'repobench-p')


# Vendored LongBench rows: utils/longbench_data/<config>.jsonl — the files out
# of THUDM/LongBench data.zip's data/, committed FLAT in the repo (a `data/`
# subdir would be swallowed by search/.gitignore's `data/` rule, like
# minilongbench_data/data already is) so offline containers read them through
# the repo mount instead of a pre-baked HF datasets cache. Selection is
# unchanged vs the hub path: the hub script materialises the SAME jsonl line by
# line, and .shuffle(seed)'s permutation depends only on (seed, num_rows).
LONGBENCH_LOCAL_DIR = os.path.join(os.path.dirname(__file__), 'longbench_data')


def load_longbench_split(config, cache_dir=None):
    """LongBench test rows, local-first.

    The vendored jsonl when present, else the THUDM/LongBench hub script — the
    hub fallback needs network or an already-prepared HF cache (script datasets
    re-download data.zip on a cache miss even offline → OfflineModeIsEnabled;
    the script API is gone entirely in datasets>=3.0), so vendor the file
    rather than relying on it."""
    local = os.path.join(LONGBENCH_LOCAL_DIR, f'{config}.jsonl')
    if os.path.isfile(local):
        return load_dataset('json', data_files=local, split='train')
    try:
        return load_dataset('THUDM/LongBench', config, split='test',
                            cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"LongBench/{config}: no vendored file at {local} and the hub "
            f"fallback failed ({e!r}). Run `python utils/fetch_offline_data.py "
            f"--only longbench` once on a machine with internet.") from e


def get_longbench_ppl(subset, seed, n_sample, tokenizer, batch_size=1, seqlen=2048,
                      min_seqlen=0, cache_dir=None):
    """LongBench `context` documents as a fixed-length PPL/loss corpus.

    Same contract as get_gov_report (shuffle by seed → keep docs of at least
    max(min_seqlen, seqlen) tokens → truncate to seqlen → stack), so it drops
    into get_loader / LlamaEvaluator unchanged.

    The RAW `context` is used, NOT the LongBench prompt template: the template
    wraps every document in identical instruction boilerplate, which is
    near-zero-entropy filler that would dilute the PPL of the document itself.
    (utils/longbench.py still uses the template for the BENCHMARK — that path
    is unaffected.)
    """
    if subset not in LONGBENCH_PPL_SUBSETS:
        raise ValueError(f"LongBench subset '{subset}' is not registered as a PPL "
                         f"corpus. Registered: {list(LONGBENCH_PPL_SUBSETS)}.")
    data = load_longbench_split(subset, cache_dir=cache_dir)
    data = data.shuffle(seed=seed).flatten_indices()

    tokenizer.pad_token = tokenizer.eos_token
    length_floor = max(int(min_seqlen), int(seqlen))
    data_list, scanned = [], 0
    for row in data:
        scanned += 1
        document = row['context']
        if len(tokenizer(document, add_special_tokens=False)['input_ids']) < length_floor:
            continue
        tokenized = tokenizer(document, add_special_tokens=False, padding=True,
                              truncation=True, max_length=seqlen, return_tensors='pt')
        data_list.append([tokenized['input_ids'], tokenized['attention_mask'],
                          tokenized['input_ids']])
        if len(data_list) >= n_sample:
            break

    if len(data_list) < n_sample:
        raise ValueError(
            f"LongBench/{subset}: only {len(data_list)} of {scanned} scanned documents "
            f"reach {length_floor} tokens (= max(min_seqlen={min_seqlen}, "
            f"seqlen={seqlen})), need n_sample={n_sample}. Lower the seqlen or "
            f"n_sample, or pick a longer subset (narrativeqa is the longest).")

    tokenizer.pad_token = None
    return DataLoader(TensorDataset(
        torch.concat([x[0] for x in data_list], dim=0),
        torch.concat([x[1] for x in data_list], dim=0),
        torch.concat([x[2] for x in data_list], dim=0)), batch_size=batch_size)


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
    # Local-first: utils/minilongbench_data holds the fetched data files
    # (utils/minilongbench.py already reads it; populate via
    # `python utils/fetch_offline_data.py --only minilongbench`); hub fallback.
    root = os.path.join(os.path.dirname(__file__), "minilongbench_data")
    if not os.path.isdir(os.path.join(root, "data")):
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


# ════════════════════════════════════════════════════════════════════════════
# Chat-templated calibration data ('chat:<corpus>')
# ════════════════════════════════════════════════════════════════════════════
# Deployment feeds Instruct models a CHAT-formatted prompt (RULER / LongBench
# both apply_chat_template now), while calibration JSD has always been measured
# on raw continuation text. `chat:<corpus>` builds the same corpora inside a
# user turn so the two can be compared:
#
#     [ pre ][            document            ][ post ]
#       ^ template prefix (BOS + role header)    ^ assistant header
#
# CONVENTION: `seqlen` / `min_seqlen` are the TOTAL sequence length, affixes
# INCLUDED — the same thing they mean for every other corpus and for the KV /
# memory accounting. The document budget is therefore
#     seqlen - len(pre) - len(post)
# which shrinks by a model-specific amount (measured: Mistral-v0.3 4 tokens,
# Gemma-3 9, Qwen2.5 29, Llama-3.1 35 — Llama and Qwen auto-inject a system
# block, Llama's containing a date string).
#
# The sample is assembled in TOKEN space, never by re-tokenizing decoded text:
# the document keeps the base loader's exact ids, which is what key-token
# character offsets index (see utils/loss._loader_offsets on decode drift).
# The cost is a <=1-token BPE boundary artefact vs a single-shot
# apply_chat_template when the document starts with whitespace (the affix's
# trailing space/newline does not merge into the first document token);
# `chat_affix_report` measures it. It is fixed for a given (model, corpus), so
# it shifts no architecture ranking.
CHAT_PREFIX = 'chat:'
# The chat sample is ONE layout, parameterised by the answer window: the
# assistant header lands at seqlen - answer_tokens, so the scored tail is
# produced in ASSISTANT position. answer_tokens=0 leaves the tail empty and the
# header simply trails the document — the degenerate case, not a second mode.
#
#   answer_tokens=0 : [ pre ][            document            ][ post ]
#   answer_tokens=A : [ pre ][   context   ][ post ][   tail   ]
#                      prefill ─────────────────────┘  ^scored (A tokens)
#
# With --prefill_prompt --last_tokens A the prefill covers exactly
# pre+context+post (prompt_len = seqlen - A), so the prefill/answer split
# COINCIDES with the turn boundary and the scored tokens are produced in
# assistant position — which is what the answer-phase protocol claims to
# measure. `context` and `tail` are a CONTIGUOUS slice of one document (the
# header is inserted between them), so the tail is still a genuine continuation.
# CAVEAT: the tail is corpus text, not a real assistant reply — realistic in
# position and KV state, still out-of-distribution in content.
# `chatdoc:` is the DOCUMENT a chat sample wraps, with no affixes — what a
# key-token archive for a chat corpus must be built on. It is NOT the same as the
# raw corpus for the stream corpora: get_chat_loader builds their base with
# add_special_tokens=False (the BOS comes from the template), which shifts every
# chunk boundary by one token. Generating the archive from plain `wikitext2`
# therefore yields documents that differ from the ones the chat loader produces
# (MEASURED: 9292 vs 9291 chars on slice 1) and the manifest check rejects it.
CHATDOC_PREFIX = 'chatdoc:'
_CHAT_MARK = '\u0001DOC\u0001'
_CHAT_AFFIX_CACHE = {}


def chat_affixes(tokenizer, instruction=''):
    """(pre_ids, post_ids): the chat wrapper around a document, in TOKEN space.

    Derived by splitting the `tokenize=False` template on a marker and encoding
    each side with add_special_tokens=False, so `pre` carries the model's own
    BOS exactly once and `post` is the assistant header the answer would follow.
    Raises for a tokenizer with no chat template (base models)."""
    if getattr(tokenizer, 'chat_template', None) in (None, ''):
        raise ValueError(
            f"tokenizer {getattr(tokenizer, 'name_or_path', '?')} has no chat_template: "
            f"'chat:<corpus>' needs an Instruct/chat model. Use the raw corpus instead.")
    key = (getattr(tokenizer, 'name_or_path', id(tokenizer)), instruction)
    if key in _CHAT_AFFIX_CACHE:
        return _CHAT_AFFIX_CACHE[key]
    content = (instruction + '\n\n' + _CHAT_MARK) if instruction else _CHAT_MARK
    s = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                      tokenize=False, add_generation_prompt=True)
    if _CHAT_MARK not in s:
        raise ValueError("chat template dropped the document marker — cannot split it")
    pre_s, post_s = s.split(_CHAT_MARK)
    pre = tokenizer(pre_s, add_special_tokens=False)['input_ids']
    post = tokenizer(post_s, add_special_tokens=False)['input_ids']
    if not pre or not post:
        raise ValueError(f"empty chat affix (pre={len(pre)}, post={len(post)})")
    bos = getattr(tokenizer, 'bos_token_id', None)
    if bos is not None:
        n_bos = sum(1 for t in pre if t == bos) + sum(1 for t in post if t == bos)
        if n_bos > 1:
            raise ValueError(f"chat affixes carry {n_bos} BOS tokens — expected at most 1")
    _CHAT_AFFIX_CACHE[key] = (pre, post)
    return pre, post


def chat_overhead(tokenizer, instruction=''):
    pre, post = chat_affixes(tokenizer, instruction)
    return len(pre) + len(post)


def chat_answer_spans(tokenizer, seqlen, answer_tokens, instruction=''):
    """[(start, end), (start, end)] — the two token ranges the DOCUMENT occupies
    in a chat sample WITH an answer window: the context before the assistant header and the
    scored tail after it. Also the prefill boundary: `seqlen - answer_tokens`
    is the end of `post`."""
    pre, post = chat_affixes(tokenizer, instruction)
    a = int(answer_tokens)
    ctx_end = int(seqlen) - a - len(post)
    return [(len(pre), ctx_end), (int(seqlen) - a, int(seqlen))]


def chat_doc_span(tokenizer, seqlen, instruction=''):
    """(start, end) token indices of the DOCUMENT body inside a chat sample.

    Constant across samples for the continuation corpora (the document fills its
    budget exactly), which is what key-token consumption needs: build the offset
    mapping over this slice only, then shift the returned indices by `start`."""
    pre, post = chat_affixes(tokenizer, instruction)
    return len(pre), int(seqlen) - len(post)


def chat_affix_report(tokenizer, probe='The quick brown fox jumps over the lazy dog.',
                      instruction=''):
    """Diagnostic: how far the token-space assembly is from a single-shot
    apply_chat_template on the same document text. Returns a dict; `delta` is
    the (built - canonical) token count, expected in {-1, 0, 1}."""
    pre, post = chat_affixes(tokenizer, instruction)
    doc = tokenizer(probe, add_special_tokens=False)['input_ids']
    built = list(pre) + list(doc) + list(post)
    content = (instruction + '\n\n' + probe) if instruction else probe
    canon = list(tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                               tokenize=True, add_generation_prompt=True))
    return dict(pre=len(pre), post=len(post), overhead=len(pre) + len(post),
                built=len(built), canon=len(canon), delta=len(built) - len(canon),
                exact=built == canon)


# Corpora whose base loader emits ONE contiguous document per sample with no
# special tokens — these wrap directly. wikitext2 / c4 are streams (handled by
# the add_special_tokens=False branch); gsm8k is a prompt+answer layout whose
# chat form puts the assistant header BETWEEN them, so it is not a wrapper case.
_CHAT_DOC_CORPORA = ('gov_report', 'longbench:')
_CHAT_STREAM_CORPORA = ('wikitext2', 'c4')


def chat_spans_for(name, tokenizer, seqlen, answer_tokens=None, instruction=''):
    """DOCUMENT token range(s) for a corpus name, or None when it is not a chat
    corpus. This is what `get_key_token_list(doc_spans=...)` wants:
      chat:<c>     -> [(len(pre), seqlen-len(post))]
      chat:<c> + answer_tokens=A -> [(len(pre), ctx_end), (seqlen-A, seqlen)]
    """
    if name.startswith((CHAT_PREFIX, CHATDOC_PREFIX)):
        # the document is SPLIT only when there is a tail; with no answer window
        # it is one contiguous range (the wrapper).
        if answer_tokens:
            return chat_answer_spans(tokenizer, seqlen, answer_tokens, instruction)
        return [chat_doc_span(tokenizer, seqlen, instruction)]
    return None


def key_token_corpus(name):
    """The BASE corpus a key-token archive is filed under: the archive is built
    on the RAW document, so `chat:`/`chatdoc:` must be stripped before joining
    the path (else it looks for `<key_token_path>/chat:wikitext2`)."""
    for p in (CHATDOC_PREFIX, CHAT_PREFIX):
        if name.startswith(p):
            return name[len(p):]
    return name


def key_token_dirname(dataset, n_sample, seqlen, min_seqlen, trunc_len,
                      sliding_window, alpha, beta, seed=0):
    """Directory a key-token archive lives in, protocol spelled out.

    The ROOT already says who judged (eval-), whose loader/tokenizer/template
    the intervals are indexed against (tgt-) and the input layout (raw /
    chat-a<N>). What it cannot say is the per-CORPUS protocol, because one root
    holds several corpora with different ones -- so it goes here. meta.json is
    still the authority (it is what gets checked); this only makes a mismatch
    visible in the path instead of only in an exception.
    """
    return (f"{key_token_corpus(dataset)}_{int(n_sample)}sample_{int(seqlen)}seqlen"
            f"_{int(min_seqlen)}min_{int(trunc_len)}trunc_{int(sliding_window)}sw"
            f"_{alpha}alpha_{beta}beta_s{int(seed)}")


def key_token_dir(root, dataset, n_sample, seqlen, min_seqlen, trunc_len,
                  sliding_window, alpha, beta, seed=0):
    """`root`/<protocol dir>, falling back to the bare corpus name.

    Archives written before the protocol went into the directory name are filed
    under `<root>/<corpus>`; they stay loadable (their meta.json still pins the
    protocol). New ones get the explicit name.
    """
    import os
    explicit = os.path.join(root, key_token_dirname(
        dataset, n_sample, seqlen, min_seqlen, trunc_len, sliding_window,
        alpha, beta, seed))
    if os.path.isdir(explicit):
        return explicit
    legacy = os.path.join(root, key_token_corpus(dataset))
    if os.path.isdir(legacy):
        return legacy
    return explicit          # not there: name the EXPECTED one in the error


def get_chat_loader(name, seed=0, n_sample=128, tokenizer=None, batch_size=1,
                    seqlen=2048, min_seqlen=0, train=True, cache_dir=None,
                    instruction='', model='', answer_tokens=None):
    """`chat:<corpus>` — the corpus inside one chat turn.

    seqlen / min_seqlen are TOTAL lengths (affixes included); the base corpus is
    built at the reduced document budget and the affixes are concatenated in
    token space. Every returned sample is exactly `seqlen` tokens.

    `answer_tokens` (= the run's answer window) decides the layout: with A > 0 the
    document is split so the last `answer_tokens` sit AFTER the assistant header.
    The base-corpus call is identical either way — only the assembly differs,
    since len(context) + len(tail) == the same document budget.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model, cache_dir=cache_dir)
    doc_only = name.startswith(CHATDOC_PREFIX)
    for _p in (CHATDOC_PREFIX, CHAT_PREFIX):
        if name.startswith(_p):
            base = name[len(_p):]
            break
    else:
        base = name
    if not train:
        # LlamaEvaluator.__init__ builds BOTH loader sides unconditionally, so a
        # hard error here made `--dataset chat:<corpus>` crash at construction
        # even for a pure `--metric loss` run. The chat wrapper is a LOSS-side
        # (train) construction — PPL over template boilerplate is not a metric
        # anyone wants — so the test side is None, exactly as gsm8k already does.
        # `--metric ppl` on a chat: dataset is therefore unsupported, not silently
        # substituted with the raw corpus.
        print(f"[chat] '{name}': no test-side loader (loss/train side only); "
              f"measure PPL on the raw corpus instead.")
        return None
    pre, post = chat_affixes(tokenizer, instruction)
    ov = 0 if doc_only else len(pre) + len(post)
    inner = int(seqlen) - ov
    if inner <= 0:
        raise ValueError(f"seqlen={seqlen} is shorter than the chat overhead ({ov} tokens)")
    inner_min = max(0, int(min_seqlen) - ov) if min_seqlen else 0
    # ONE layout, parameterised by where the assistant header falls:
    #     pre + context + post + tail        len(tail) = answer_tokens
    # answer_tokens = 0 leaves the tail EMPTY, which is exactly the wrapper
    # (pre + document + post) -- so "wrapper" is not a separate mode, it is the
    # degenerate case, and the incoherent combination (wrapper layout WITH an
    # answer window, i.e. a scored tail sitting in the USER turn) becomes
    # unrepresentable.
    n_ans = int(answer_tokens or 0)
    if n_ans < 0 or n_ans >= inner:
        raise ValueError(
            f"'{name}' needs 0 <= answer_tokens < {inner} (the document budget at "
            f"seqlen={seqlen}); got {answer_tokens}. It is the run's answer window: "
            f"the tail placed after the assistant header (0 = no tail).")

    if any(base.startswith(c) for c in _CHAT_STREAM_CORPORA):
        fn = get_wikitext2_trainenc if 'wikitext2' in base else get_c4_trainenc
        base_loader = fn(seed=seed, n_sample=n_sample, batch_size=1, seqlen=inner,
                         tokenizer=tokenizer, cache_dir=cache_dir,
                         add_special_tokens=False)
    elif any(base.startswith(c) for c in _CHAT_DOC_CORPORA):
        base_loader = get_loader(base, n_sample=n_sample, train=train, seed=seed,
                                 seqlen=inner, min_seqlen=inner_min, batch_size=1,
                                 tokenizer=tokenizer, model=model, cache_dir=cache_dir)
    else:
        raise ValueError(
            f"'{name}': no chat wrapper for corpus '{base}'. Supported: "
            f"{_CHAT_STREAM_CORPORA + _CHAT_DOC_CORPORA} (gsm8k needs the "
            f"prompt/answer split layout, not a wrapper).")

    if doc_only:
        return base_loader          # the bare document, at the caller's seqlen
    pre_t = torch.tensor(pre, dtype=torch.long)
    post_t = torch.tensor(post, dtype=torch.long)
    ones_pre = torch.ones(len(pre), dtype=torch.long)
    ones_post = torch.ones(len(post), dtype=torch.long)
    bos = getattr(tokenizer, 'bos_token_id', None)
    ids_l, am_l, lab_l = [], [], []
    for doc_ids, doc_am, doc_lab in base_loader:
        for r in range(doc_ids.shape[0]):
            d = doc_ids[r]
            if d.numel() != inner:
                raise ValueError(f"'{name}': base corpus returned {d.numel()} tokens, "
                                 f"expected the document budget {inner}")
            if bos is not None and int(d[0]) == bos:
                raise ValueError(f"'{name}': the base corpus emitted a BOS at position 0 — "
                                 f"the chat prefix already carries one.")
            m_pre = torch.full((len(pre),), -100, dtype=torch.long)
            m_post = torch.full((len(post),), -100, dtype=torch.long)
            if n_ans:
                # pre + context + post + tail: the header lands exactly at
                # seqlen - n_ans, so prompt_len == that boundary.
                c, t = d[:-n_ans], d[-n_ans:]
                ids_l.append(torch.cat([pre_t, c, post_t, t]))
                am_l.append(torch.cat([ones_pre, doc_am[r][:-n_ans], ones_post,
                                       doc_am[r][-n_ans:]]))
                lab_l.append(torch.cat([m_pre, doc_lab[r][:-n_ans], m_post,
                                        doc_lab[r][-n_ans:]]))
            else:
                ids_l.append(torch.cat([pre_t, d, post_t]))
                am_l.append(torch.cat([ones_pre, doc_am[r], ones_post]))
                # affix positions are context, never scored
                lab_l.append(torch.cat([m_pre, doc_lab[r], m_post]))
    if not ids_l:
        raise ValueError(f"'{name}': base corpus produced no samples "
                         f"(n_sample={n_sample}, budget={inner})")
    return DataLoader(TensorDataset(torch.stack(ids_l), torch.stack(am_l),
                                    torch.stack(lab_l)), batch_size=batch_size)


def get_trainloaders(name, n_sample=128, seed=0, seqlen=2048, model='', batch_size=1, cache_dir=None):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)
    if 'c4' in name:
        return get_c4_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)
    if 'gsm8k' in name:
        return get_gsm8k_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)

def get_loader(name, n_sample=128, train=True, seed=0, seqlen=2048, min_seqlen=0, batch_size=1, tokenizer=None, model='', cache_dir=None, sub_dataset=None, require_answer=False, answer_tokens=None):
    if tokenizer is None:
        tokenizer = get_tokenizer(model, cache_dir=cache_dir)
    if name.startswith((CHAT_PREFIX, CHATDOC_PREFIX)):
        return get_chat_loader(name, seed=seed, n_sample=n_sample, tokenizer=tokenizer,
                               batch_size=batch_size, seqlen=seqlen, min_seqlen=min_seqlen,
                               train=train, cache_dir=cache_dir, model=model,
                               answer_tokens=answer_tokens)
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
        # LongBench long-document corpora ('longbench:<subset>', e.g.
        # 'longbench:narrativeqa'). Like gov_report there is only a test split,
        # so both sides read it (the loss side is then a calibration probe on
        # the same documents, not a held-out set).
        if name.startswith('longbench:'):
            return get_longbench_ppl(name.split(':', 1)[1], seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, tokenizer=tokenizer, min_seqlen=min_seqlen, cache_dir=cache_dir)
    else:
        if 'wikitext2' in name:
            return get_wikitext2(tokenizer=tokenizer, batch_size=batch_size, seqlen=seqlen, cache_dir=cache_dir)
        if 'c4' in name:
            return get_c4(tokenizer=tokenizer, batch_size=batch_size, seqlen=seqlen, cache_dir=cache_dir)
        if 'gsm8k' in name:
            return None
        if 'gov_report' in name:
            return get_gov_report(seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, tokenizer=tokenizer, split='test', min_seqlen=min_seqlen, cache_dir=cache_dir)
        if name.startswith('longbench:'):
            return get_longbench_ppl(name.split(':', 1)[1], seed=seed, n_sample=n_sample, batch_size=batch_size, seqlen=seqlen, tokenizer=tokenizer, min_seqlen=min_seqlen, cache_dir=cache_dir)
