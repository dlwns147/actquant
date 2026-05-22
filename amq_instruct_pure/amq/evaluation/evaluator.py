"""Benchmark-proxy JSD evaluator (HF).

JSD-only variant for ``amq_instruct_pure``: scores answer-portion JSD against
a cached fp16 baseline using the HF candidate model. Samples are drawn
from the *whole* dataset (no monotonic / non-monotonic split).

Dataset I/O is delegated to ``evaluation/data_io.py`` so adding a new
lm_eval-derived task (gsm8k_cot_train / ifeval_train / mbpp_train / ...)
only touches that file.
"""

from __future__ import annotations

import math
import random
from tqdm import tqdm
from copy import deepcopy
from statistics import fmean

import warnings
warnings.simplefilter("ignore")

import torch

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

from evaluation.data_io import (
    _load_all_rows,
    _load_fp16_answers,
    _build_prompt,
    _answer_span_ids,
)


# ---------------------------------------------------------------------------
# whole-dataset sampling

class _SampleLoader:
    """Owns the row cache and draws ``n_sample`` rows per call.

    No bucket balancing; samples are drawn uniformly from the whole
    dataset (without replacement). Workers seeded identically see the
    same rows on call N.
    """

    def __init__(self, datasets=("gsm8k", "livebench"), n_sample=16, seed=0,
                 load_fp16=False, tokenizer=None):
        self.datasets = tuple(datasets)
        self.n_sample = n_sample
        self.seed = seed
        self._call = 0
        self.rows = {ds: _load_all_rows(ds, tokenizer=tokenizer,
                                         wikitext2_seed=seed)
                     for ds in self.datasets}
        self.fp16_answers = (
            {ds: _load_fp16_answers(ds, tokenizer=tokenizer,
                                     wikitext2_seed=seed)
             for ds in self.datasets}
            if load_fp16 else {}
        )

    def extract_sample(self):
        rng = random.Random(self.seed * 1_000_003 + self._call)
        self._call += 1
        out = {}
        for ds in self.datasets:
            pool = self.rows[ds]
            # wikitext2 anchor: use the entire (small, fixed) pool every iter
            # rather than sub-sampling — same chunks every search call.
            from evaluation.data_io import is_wikitext2
            if is_wikitext2(ds):
                out[ds] = list(pool)
                continue
            k = min(self.n_sample, len(pool))
            out[ds] = rng.sample(pool, k=k)
        return out


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
            tokenizer=self.tokenizer,
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
        """Return J_arch: per-doc JSD mean over this arch's sample."""
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
                start_pos = max(prompt_ids.numel() - 1, 0)
                end_pos   = start_pos + answer_ids.numel()
                losses.append(
                    jsd(cand_logits[i, start_pos:end_pos], fp16_logits[i, start_pos:end_pos]).item()
                )
        if not losses:
            return 0.0
        return fmean(losses)

    def eval_jsd(self, architectures, max_value):
        samples = self.sample_loader.extract_sample()
        metric_list = {}
        for ds in self.datasets:
            metric_list[f"{ds}_jsd"]  = []
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
                J_arch = self._score_jsd(ds, samples[ds])
                metric_list[f"{ds}_jsd"].append(J_arch)

            bits_usage = get_bits_usage(architecture, self.config, self.group_size)
            bits_usage_list.append(bits_usage)

            clean_up()

        return metric_list, bits_usage_list
