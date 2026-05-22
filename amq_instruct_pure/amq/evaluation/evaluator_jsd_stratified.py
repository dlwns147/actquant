"""Benchmark-proxy JSD evaluator (HF), stratified by JSD bin labels.

Rows are split into ``n_sample`` quantile bins defined by a per-row JSD
score (fp16-vs-w3 token-level JSD cached under
``evaluation/DISCO/<task>_jsd.json``). The loader operates in
1-row-per-bin mode: ``n_sample`` rows are drawn per round, one from each
of ``n_sample`` quantile bins. ``n_sample`` may be an int (same count for
every dataset) or a per-dataset mapping (e.g.
``{"gsm8k_cot_train": 16, "ifeval_train": 16}``).
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections.abc import Mapping
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

LOSS_FUNCTION = JSD()
print("Loss Function is JSD (returns J per arch)")

# ---------------------------------------------------------------------------
# Dataset I/O helpers + constants live in evaluation/data_io.py so adding
# a new task only touches one file. Re-export for back-compat with callers
# that still import from this module.
# ---------------------------------------------------------------------------
from evaluation.data_io import (
    _DEFAULT_N_SAMPLE,
    _load_all_rows,
    _load_fp16_answers,
    _build_prompt,
    _answer_span_ids,
    _normalized_ensemble_bin_labels,
)


# ---------------------------------------------------------------------------
# Helpers (_normalized_ensemble_bin_labels, _build_prompt, _answer_span_ids,
# _load_all_rows, _load_fp16_answers) are imported from data_io above.


# ---------------------------------------------------------------------------
# sample loader: jsd-bin stratified (always 1-row-per-bin)


class _SampleLoader:
    """Owns the row cache and draws ``n_sample`` rows per call.

    Each dataset is split into ``n_sample[ds]`` jsd-quantile bins, and one
    row is drawn from each bin per ``extract_sample()`` call (the canonical
    PSS 1-per-bin mode analyzed in ``evaluation/sampling_analysis_jsd.md``).

    ``n_sample`` may be an int (same count for every dataset) or a mapping
    ``{dataset: count}``.

    Workers seeded identically see the same rows on call N.
    """

    def __init__(self, datasets=("gsm8k", "livebench"), n_sample=_DEFAULT_N_SAMPLE,
                 seed=0, load_fp16=False, tokenizer=None):

        print("Using _SampleLoader class")
        self.datasets = tuple(datasets)
        self.seed = seed
        self._call = 0

        from evaluation.data_io import is_wikitext2 as _is_wikitext2
        if isinstance(n_sample, Mapping):
            # wikitext2 ignores n_sample entirely (entire fixed pool used per
            # iter), so missing entries are fine for it.
            missing = [ds for ds in self.datasets
                       if ds not in n_sample and not _is_wikitext2(ds)]
            if missing:
                raise ValueError(f"n_sample missing entries for {missing}")
            self.n_sample = {ds: int(n_sample[ds])
                             for ds in self.datasets if ds in n_sample}
        else:
            self.n_sample = {ds: int(n_sample) for ds in self.datasets
                             if not _is_wikitext2(ds)}

        self.rows = {ds: _load_all_rows(ds, tokenizer=tokenizer,
                                         wikitext2_seed=seed)
                     for ds in self.datasets}
        self.fp16_answers = (
            {ds: _load_fp16_answers(ds, tokenizer=tokenizer,
                                     wikitext2_seed=seed)
             for ds in self.datasets}
            if load_fp16 else {}
        )

        from evaluation.data_io import is_wikitext2 as _is_wikitext2
        self._bin_pools = {}
        self._fixed_full_pools = {}  # ds → full row list (sampling-skipped)
        for ds in self.datasets:
            if _is_wikitext2(ds):
                # Anchor: use the entire fixed pool every iteration. Skip
                # binning so the (small, ~4-chunk) pool stays intact.
                self._fixed_full_pools[ds] = list(self.rows[ds])
                print(f"[jsd_stratified] {ds}: fixed full pool of "
                      f"{len(self._fixed_full_pools[ds])} chunks (no sub-sampling)")
                continue
            n = self.n_sample[ds]
            labels = _normalized_ensemble_bin_labels(ds, n_bins=n)
            pools = [[] for _ in range(n)]
            n_unlabeled = 0
            for row in self.rows[ds]:
                did = row.get("doc_id")
                if did is None:
                    n_unlabeled += 1
                    continue
                b = labels.get(str(did))
                if b is None:
                    n_unlabeled += 1
                    continue
                pools[b].append(row)
            self._bin_pools[ds] = pools
            sizes = [len(p) for p in pools]
            print(f"[jsd_stratified] {ds}: n_sample={n} (1-per-bin), "
                  f"bin sizes min={min(sizes)} max={max(sizes)} median={int(np.median(sizes))}, "
                  f"unlabeled={n_unlabeled}")

    def extract_sample(self):
        rng = random.Random(self.seed * 1_000_003 + self._call)
        self._call += 1
        out = {}
        for ds in self.datasets:
            if ds in self._fixed_full_pools:
                out[ds] = list(self._fixed_full_pools[ds])
                continue
            picks = []
            for pool in self._bin_pools[ds]:
                if not pool:
                    continue
                picks.append(rng.choice(pool))
            out[ds] = picks
        return out


# ---------------------------------------------------------------------------
# DISCO sample loader lives in evaluation/DISCO/. Imported here so bare-name
# references inside this module (e.g. inside ``BenchmarkProxyEvaluator.__init__``)
# resolve.
from evaluation.DISCO.sample_loader_disco import (          # noqa: E402
    _SampleLoader_DISCO,
    _disco_allocate,
)


# ---------------------------------------------------------------------------
# evaluator (identical shape to evaluator.py)

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
        n_sample=_DEFAULT_N_SAMPLE,
        batch_size=4,
        seed=0,
        dev="cuda",
        dtype="auto",
        tqdm_position=0,
        loader="plain",
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
        self.loader = loader

        self.accelerator, self.device_map = init_accelerator(self.gpu_id, config)

        self.tokenizer = get_tokenizer(model_id)

        self.quantization_proxies = get_quantization_proxy(
            quantization_proxy_paths, self.device_map, False,
        )

        self.bits_range = list(bits_range)
        assert len(self.bits_range) == len(self.quantization_proxies), (
            "Number of bits range and quantization proxies must be the same"
        )

        if loader == "plain":
            self.sample_loader = _SampleLoader(
                datasets=self.datasets,
                n_sample=n_sample,
                seed=seed,
                load_fp16=True,
                tokenizer=self.tokenizer,
            )
            self.eval = self.eval_jsd
        elif loader == "disco":
            self.sample_loader = _SampleLoader_DISCO(
                datasets=self.datasets,
                n_sample=n_sample,
                seed=seed,
                load_fp16=True,
                tokenizer=self.tokenizer,
            )
            self.eval = self.eval_jsd
        else:
            raise ValueError(
                f"unknown loader={loader!r} (expected plain/disco)"
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

        spans.sort(key=lambda pa: pa[0].numel() + pa[1].numel())

        Js = []
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
                J_doc = LOSS_FUNCTION(
                    cand_logits[i, start_pos:end_pos],
                    fp16_logits[i, start_pos:end_pos])
                Js.append(float(J_doc.item()))
        if not Js:
            return 0.0
        return fmean(Js)

    def eval_jsd(self, architectures, max_value):
        """Shared-sample evaluator (one sample set per call → all archs scored
        on the same rows). Used by ``loader in {plain, disco}``."""
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
            bits_usage = get_bits_usage(architecture, self.config, self.group_size)

            for ds in self.datasets:
                J_arch = self._score_jsd(ds, samples[ds])
                metric_list[f"{ds}_jsd"].append(J_arch)

            bits_usage_list.append(bits_usage)

            clean_up()

        return metric_list, bits_usage_list
