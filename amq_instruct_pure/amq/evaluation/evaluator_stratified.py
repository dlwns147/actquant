"""Benchmark-proxy JSD evaluator (HF), stratified by difficulty group.

Same shape as ``evaluator.py``, but draws ``n_sample`` rows split across
five groups: the four monotonic buckets (``000``, ``100``, ``110``,
``111``) plus a single ``non_monotonic`` pool merged from the four
remaining buckets (``001``, ``010``, ``011``, ``101``). Matches the
``_BALANCED_GROUPS`` scheme used in ``amq/amq/evaluation/evaluator.py``.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from statistics import fmean

import warnings
warnings.simplefilter("ignore")

import torch
import numpy as np

from utils.func import (
    init_accelerator,
    get_hfmodel,
    get_quantization_proxy,
    get_bits_usage,
    clean_up,
    getsubattr,
    setsubattr,
    getblock,
)
from utils.data import get_tokenizer
from utils.loss import JSD


# ---------------------------------------------------------------------------
# constants

_DATASET_ROOTS = {
    "gsm8k":     Path("/NAS/SJ/actquant/poc/benchmark_datasets/gsm8k_train"),
    "livebench": Path("/NAS/SJ/actquant/poc/benchmark_datasets/livebench"),
}
_MONOTONIC       = ("000", "100", "110", "111")
_NON_MONOTONIC   = ("001", "010", "011", "101")
_BALANCED_GROUPS = ("000", "100", "110", "111", "non_monotonic")

_LIVEBENCH_HF = "livebench/instruction_following"


# ---------------------------------------------------------------------------
# bucket loading

def _load_bucket(dataset, pattern):
    path = _DATASET_ROOTS[dataset] / f"{pattern}.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if dataset == "livebench":
                row["doc_id"] = row["question_id"]
            rows.append(row)
    return rows


_LIVEBENCH_KWARGS_CACHE = None

def _livebench_kwargs_by_id():
    global _LIVEBENCH_KWARGS_CACHE
    if _LIVEBENCH_KWARGS_CACHE is None:
        from datasets import load_dataset
        ds = load_dataset(_LIVEBENCH_HF, split="test")
        _LIVEBENCH_KWARGS_CACHE = {q["question_id"]: q["kwargs"] for q in ds}
    return _LIVEBENCH_KWARGS_CACHE


def _attach_livebench_kwargs(rows):
    if not rows:
        return rows
    kwargs_by_id = _livebench_kwargs_by_id()
    for row in rows:
        row["kwargs"] = kwargs_by_id[row["question_id"]]
    return rows


def _load_buckets(dataset):
    """Return ``{pattern: [rows]}`` for every difficulty bucket."""
    buckets = {pattern: _load_bucket(dataset, pattern)
               for pattern in (_MONOTONIC + _NON_MONOTONIC)}
    if dataset == "livebench":
        for pattern, rows in buckets.items():
            _attach_livebench_kwargs(rows)
    return buckets


def _load_fp16_answers(dataset):
    path = _DATASET_ROOTS[dataset] / "gt.jsonl"
    answers = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if dataset == "gsm8k":
                answers[row["doc_id"]] = row["resps"][0][0]
            else:
                answers[row["question_id"]] = row["response"]
    return answers


# ---------------------------------------------------------------------------
# stratified sampling (balanced groups: 4 monotonic + 1 merged non_monotonic)

def _group_pool(buckets, group_key):
    if group_key == "non_monotonic":
        pool = []
        for p in _NON_MONOTONIC:
            pool.extend(buckets.get(p, ()))
        return pool
    return list(buckets.get(group_key, ()))


def _balanced_sample(buckets, n, rng, group_keys):
    k = len(group_keys)
    base, extra = divmod(n, k)
    counts = [base + (1 if i < extra else 0) for i in range(k)]
    picked = []
    for group_key, count in zip(group_keys, counts):
        if count == 0:
            continue
        pool = _group_pool(buckets, group_key)
        if not pool:
            continue
        picked.extend(rng.sample(pool, k=min(count, len(pool))))
    return picked


def _proportional_counts(weights, n):
    """Largest-remainder allocation of ``n`` slots across ``weights``.

    Returns a list of ints summing to exactly ``n`` (assuming ``n`` is
    non-negative and at least one weight is positive). Empty / zero
    weights get zero slots.
    """
    total = sum(weights)
    if total <= 0 or n <= 0:
        return [0] * len(weights)
    raw = [n * w / total for w in weights]
    floors = [int(r) for r in raw]
    remainder = n - sum(floors)
    if remainder > 0:
        # hand out leftover slots to the largest fractional remainders
        order = sorted(range(len(weights)), key=lambda i: raw[i] - floors[i], reverse=True)
        for i in order[:remainder]:
            floors[i] += 1
    return floors


def _proportional_sample(buckets, n, rng, group_keys, weights=None):
    """Stratified sample where slots-per-group is proportional to weight.

    If ``weights`` is None, weights default to each group's pool size
    (so groups get sampled in proportion to how many rows they have).
    Otherwise ``weights`` must be a sequence aligned to ``group_keys``.
    """
    pools = [_group_pool(buckets, gk) for gk in group_keys]
    if weights is None:
        weights = [len(pool) for pool in pools]
    counts = _proportional_counts(weights, n)
    picked = []
    for pool, count in zip(pools, counts):
        if count == 0 or not pool:
            continue
        picked.extend(rng.sample(pool, k=min(count, len(pool))))
    return picked


class _SampleLoader:
    """Owns per-bucket caches and draws balanced samples per call.

    Splits ``n_sample`` across ``_BALANCED_GROUPS`` (4 monotonic buckets
    + 1 merged non-monotonic pool). Workers seeded identically see the
    same rows on call N.
    """

    def __init__(self, datasets=("gsm8k", "livebench"), n_sample=16, seed=0,
                 load_fp16=False):
        self.datasets = tuple(datasets)
        self.n_sample = n_sample
        self.seed = seed
        self._call = 0
        self.buckets = {ds: _load_buckets(ds) for ds in self.datasets}
        self.fp16_answers = (
            {ds: _load_fp16_answers(ds) for ds in self.datasets}
            if load_fp16 else {}
        )

    def extract_sample(self, groups=_BALANCED_GROUPS):
        rng = random.Random(self.seed * 1_000_003 + self._call)
        self._call += 1
        return {
            # ds: _balanced_sample(self.buckets[ds], self.n_sample, rng, _BALANCED_GROUPS)
            ds: _proportional_sample(self.buckets[ds], self.n_sample, rng, groups)
            for ds in self.datasets
        }


# ---------------------------------------------------------------------------
# prompts

def _build_prompt(dataset, row, tokenizer=None):
    if dataset == "gsm8k":
        return row["arguments"]["gen_args_0"]["arg_0"]
    messages = [{"role": "user", "content": row["prompt"]}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )


# ---------------------------------------------------------------------------
# answer-token span

def _answer_span_ids(dataset, tokenizer, row, answer_text):
    if dataset == "gsm8k":
        prompt = row["arguments"]["gen_args_0"]["arg_0"]
    else:
        prompt = _build_prompt("livebench", row, tokenizer)

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids   = tokenizer(prompt + answer_text, return_tensors="pt",
                           add_special_tokens=False).input_ids[0]
    answer_ids = full_ids[prompt_ids.numel():]
    return prompt_ids, answer_ids


# ---------------------------------------------------------------------------
# evaluator

class BenchmarkProxyEvaluator:
    def __init__(
        self,
        config,
        gpu_id,
        model_id="",
        quantization_proxy_paths=(),
        quantization_proxy_fake_paths=(),
        bits_range=(),
        group_size=128,
        datasets=("gsm8k", "livebench"),
        n_sample=16,
        batch_size=4,
        seed=0,
        dev="cuda",
        dtype="auto",
        tqdm_position=0,
        **kwargs,
    ):
        self.config = config
        self.gpu_id = gpu_id
        self.model_id = model_id
        self.group_size = group_size
        self.dev = dev
        self.dtype = dtype
        self.datasets = tuple(datasets)
        self.batch_size = batch_size
        self.tqdm_position = tqdm_position

        self.accelerator, self.device_map = init_accelerator(self.gpu_id, config)
        self.eval = self.eval_jsd

        self.tokenizer = get_tokenizer(model_id)

        self.quantization_proxies = get_quantization_proxy(
            quantization_proxy_paths, self.device_map, False,
        )

        self.bits_range = list(bits_range)
        assert len(self.bits_range) == len(self.quantization_proxies), (
            "Number of bits range and quantization proxies must be the same"
        )

        self.sample_loader = _SampleLoader(
            datasets=self.datasets,
            n_sample=n_sample,
            seed=seed,
            load_fp16=True,
        )

        print("Loading FP16 model")
        self.fp16_model = get_hfmodel(self.model_id, dtype=self.dtype, device_map=self.device_map)
        self.fp16_answers = self.sample_loader.fp16_answers

        self.model = deepcopy(self.quantization_proxies[-1])
        self.model = self.model.to(dev)

        self.accelerator.wait_for_everyone()

    def sample(self, arch):
        for linear, linear_bits in arch["linear"].items():
            for blk_idx, bits in enumerate(linear_bits):
                flag = False
                for q_bits, q_model in zip(self.bits_range, self.quantization_proxies):
                    if math.isclose(int(bits), q_bits) and q_bits > 0:
                        setsubattr(
                            getblock(self.model, self.config)[blk_idx],
                            linear,
                            getsubattr(getblock(q_model, self.config)[blk_idx], linear),
                        )
                        flag = True
                if not flag:
                    raise NotImplementedError(f"{linear}: {bits} is not available")
        return self.model

    @torch.inference_mode()
    def _score_jsd(self, dataset, samples):
        jsd = JSD()
        answers_by_id = self.fp16_answers[dataset]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        spans = []
        for row in samples:
            answer_text = answers_by_id.get(row["doc_id"], "")
            if not answer_text:
                continue
            prompt_ids, answer_ids = _answer_span_ids(
                dataset, self.tokenizer, row, answer_text,
            )
            if answer_ids.numel() == 0:
                continue
            spans.append((prompt_ids, answer_ids))

        # sort by total length so each padded chunk has similar lengths
        spans.sort(key=lambda pa: pa[0].numel() + pa[1].numel())

        losses = []
        for start in range(0, len(spans), self.batch_size):
            chunk = spans[start:start + self.batch_size]
            seqs  = [torch.cat([p, a]) for p, a in chunk]
            max_len = max(s.numel() for s in seqs)

            ids  = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, :s.numel()] = s
                attn[i, :s.numel()] = 1
            ids  = ids.to(self.dev)
            attn = attn.to(self.dev)

            fp16_logits = self.fp16_model(ids, attention_mask=attn).logits
            cand_logits = self.model(ids, attention_mask=attn).logits

            for i, (prompt_ids, answer_ids) in enumerate(chunk):
                start_pos = prompt_ids.numel() - 1
                end_pos   = start_pos + answer_ids.numel()
                losses.append(
                    jsd(cand_logits[i, start_pos:end_pos], fp16_logits[i, start_pos:end_pos]).item()
                )
        return fmean(losses) if losses else 0.0

    def eval_jsd(self, architectures, max_value):
        samples = self.sample_loader.extract_sample()
        metric_list = {f"{ds}_jsd": [] for ds in self.datasets}
        bits_usage_list = []

        for architecture in tqdm(
            architectures,
            desc=f"Evaluating JSD [w{self.tqdm_position}]",
            position=self.tqdm_position,
            leave=True,
            dynamic_ncols=True,
        ):
            self.sample(architecture)

            for ds in self.datasets:
                metric_list[f"{ds}_jsd"].append(self._score_jsd(ds, samples[ds]))

            bits_usage = get_bits_usage(architecture, self.config, self.group_size)
            bits_usage_list.append(bits_usage)

            clean_up()

        return metric_list, bits_usage_list
