"""Static DISCO sample loader.

For supported datasets (``gsm8k_cot_train``, ``ifeval_train``, ``mbpp_train``):
rows are split into 4 DISCO groups by fp16/quant correctness, then within
each group rows are sorted by the normalized w2/w3/w4 JSD ensemble score and
quantile-binned into ``count[g]`` bins where ``count[g]`` is allocated by
:func:`_disco_allocate`. Each round picks 1 row per bin → ``n_sample`` rows.

Group convention (matches lm_eval_vllm/grouping.py):
    A: fp16 o, w3 x   (quantization broke it)
    B: fp16 o, w3 o   (both got it)
    C: fp16 x, w3 x   (both missed)
    D: fp16 x, w3 o   (only quantized got it — rare)

Labels for the supported tasks live in ``*_train_labels.json``
(gsm8k_cot_train / ifeval_train / mbpp_train) and follow the
lm_eval_vllm grouping.py convention above. C/D label strings are only
used as stratum keys, so the labeling does not affect sampling behavior
beyond stratum identity.

Allocation: ``_disco_allocate`` does *not* force min-1. Empty / proportionally
zero groups (typically D with frac < 1/K) are skipped. This avoids the
"force tiny group into 25% of estimator" over-representation bias.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from evaluation.data_io import (
    _DEFAULT_N_SAMPLE,
    _load_all_rows,
    _load_fp16_answers,
    _normalized_ensemble_bin_labels,
)


import os as _os

# Which DISCO groups to include in allocation. Default is the full set
# (A,B,C,D); override at run time via env var ``DISCO_GROUPS_INCLUDE``
# (e.g. ``DISCO_GROUPS_INCLUDE=A,B`` keeps only the FP16-correct rows).
# Rows whose label is not in this set are treated as ``n_unlabeled`` and
# dropped from the per-iter sample pool.
_DISCO_GROUPS = tuple(
    g.strip() for g in _os.environ.get(
        "DISCO_GROUPS_INCLUDE", "A,B,C,D"
    ).split(",") if g.strip()
)

# Per-dataset labels + JSD-score files (lm_eval_vllm/grouping.py convention).
# Keys are the dataset names used by SampleLoader; values are filenames in
# this directory.
_DATASET_LABELS_FILES = {
    "gsm8k_cot_train": "gsm8k_cot_train_labels.json",
    "ifeval_train":    "ifeval_train_labels.json",
    "mbpp_train":      "mbpp_train_labels.json",
    "ifeval_pp_train": "ifeval_pp_train_labels.json",
    "apps_train":      "apps_train_labels.json",
}
# Per-doc JSD scores used for within-group quantile ordering (copied from
# lm_eval_vllm/train_set_jsd/<task>_jsd_w3.json).
_DATASET_JSD_FILES = {
    "gsm8k_cot_train": "gsm8k_cot_train_jsd.json",
    "ifeval_train":    "ifeval_train_jsd.json",
    "mbpp_train":      "mbpp_train_jsd.json",
    "ifeval_pp_train": "ifeval_pp_train_jsd.json",
    "apps_train":      "apps_train_jsd.json",
}


def _disco_allocate(sizes, n):
    """Allocate ``n`` total slots across groups by group size.

    Rule: pure largest-remainder. No min-1 forcing — groups with raw share
    below 1/K (e.g. D with frac < 1.3%) may receive 0 slots. This avoids
    over-representing tiny groups whose forced 1 slot would otherwise count
    for 1/K of the estimator (e.g. K=4 with a 1%-frequency D becomes a 25%
    weight on D).

    If raw[i] >= 0.5 the group naturally rounds up to 1 via largest-remainder;
    if raw[i] < 0.5 the group is dropped from this round.
    """
    sizes = list(sizes)
    total = sum(sizes)
    if total <= 0 or n <= 0:
        return tuple(0 for _ in sizes)
    raw = [n * s / total for s in sizes]
    counts = [int(r) for r in raw]  # floors
    cur = sum(counts)
    if cur < n:
        order = sorted(range(len(sizes)),
                       key=lambda i: raw[i] - counts[i], reverse=True)
        for i in order:
            if cur >= n:
                break
            counts[i] += 1
            cur += 1
    elif cur > n:
        # shouldn't happen with pure floors, but kept defensively
        while cur > n:
            order = sorted(range(len(sizes)),
                           key=lambda i: counts[i] - raw[i], reverse=True)
            reduced = False
            for i in order:
                if counts[i] > 0:
                    counts[i] -= 1
                    cur -= 1
                    reduced = True
                    break
            if not reduced:
                break
    return tuple(counts)


class _SampleLoader_DISCO:
    """Static DISCO loader. Workers seeded identically see the same rows on call N."""

    def __init__(self, datasets=("gsm8k",), n_sample=_DEFAULT_N_SAMPLE,
                 seed=0, load_fp16=False, tokenizer=None):
        print("Using _SampleLoader_DISCO class")
        self.datasets = tuple(datasets)
        self.seed = seed
        self._call = 0

        from evaluation.data_io import is_wikitext2 as _is_wikitext2
        if isinstance(n_sample, Mapping):
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

        self._bin_pools = {}        # ds -> list[list[row]]
        self._allocation = {}       # ds -> dict[group->count] or None
        self._fixed_full_pools = {}  # ds -> full row list (sampling-skipped)
        for ds in self.datasets:
            if _is_wikitext2(ds):
                self._fixed_full_pools[ds] = list(self.rows[ds])
                self._allocation[ds] = None
                print(f"[disco] {ds}: fixed full pool of "
                      f"{len(self._fixed_full_pools[ds])} chunks (no sub-sampling)")
                continue
            n = self.n_sample[ds]
            if ds in _DATASET_LABELS_FILES:
                pools, alloc = self._build_disco_pools(ds, n)
                self._allocation[ds] = alloc
                mode = f"DISCO  alloc=" + " ".join(f"{g}:{c}" for g, c in alloc.items())
            else:
                pools = self._build_jsd_pools(ds, n)
                self._allocation[ds] = None
                mode = "JSD-quantile (fallback)"
            self._bin_pools[ds] = pools
            sizes = [len(p) for p in pools]
            print(f"[disco] {ds}: n_sample={n}, total_bins={len(pools)}, "
                  f"bin sizes min={min(sizes)} max={max(sizes)} "
                  f"median={int(np.median(sizes))}  [{mode}]")

    # -- DISCO branch ------------------------------------------------------
    def _build_disco_pools(self, ds, n):
        labels_name = _DATASET_LABELS_FILES.get(ds)
        if labels_name is None:
            raise ValueError(
                f"DISCO mode requested for dataset={ds!r}, but no labels "
                f"file is configured. Supported: {list(_DATASET_LABELS_FILES)}"
            )
        labels_path = Path(__file__).parent / labels_name
        if not labels_path.exists():
            raise FileNotFoundError(
                f"DISCO labels not found at {labels_path}. "
                f"Generate it from lm_eval_vllm/split_results/{ds}_grouping.json."
            )
        with labels_path.open() as f:
            payload = json.load(f)
        disco_by_doc = payload["by_doc"] if "by_doc" in payload else payload

        # Score variable for within-group sorting. Currently only gsm8k has
        # the normalized w2/w3/w4 ensemble JSD cache; for other tasks fall
        # back to the doc_id itself (degenerates to "no within-group quantile
        # ordering" when K_g == 1).
        score_by_doc = self._score_by_doc(ds)

        groups = {g: [] for g in _DISCO_GROUPS}
        n_unlabeled = 0
        for row in self.rows[ds]:
            did = row.get("doc_id")
            if did is None:
                n_unlabeled += 1
                continue
            g = disco_by_doc.get(str(did))
            if g is None or g not in groups:
                n_unlabeled += 1
                continue
            groups[g].append((row, score_by_doc.get(str(did), 0.0)))

        group_sizes = [len(groups[g]) for g in _DISCO_GROUPS]
        counts = _disco_allocate(group_sizes, n)
        alloc = {g: c for g, c in zip(_DISCO_GROUPS, counts)}

        pools = []
        for g, k in zip(_DISCO_GROUPS, counts):
            if k == 0 or not groups[g]:
                continue
            sorted_rows = sorted(groups[g], key=lambda x: x[1])
            edges = np.linspace(0, len(sorted_rows), k + 1).astype(int)
            for i in range(k):
                pool = [r for r, _ in sorted_rows[edges[i]:edges[i + 1]]]
                pools.append(pool)
        if n_unlabeled:
            print(f"[disco]   gsm8k unlabeled rows skipped: {n_unlabeled}")
        return pools, alloc

    def _score_by_doc(self, ds):
        """Return {doc_id_str: score_float} used for within-group quantile
        ordering. We use the per-doc fp16-vs-w3 token-level JSD pre-computed
        and cached under DISCO/<ds>_jsd.json (sourced from
        lm_eval_vllm/train_set_jsd/<task>_jsd_w3.json).
        """
        jsd_name = _DATASET_JSD_FILES.get(ds)
        if jsd_name is None:
            raise ValueError(
                f"No JSD score file configured for dataset={ds!r}. "
                f"Supported: {list(_DATASET_JSD_FILES)}"
            )
        jsd_path = Path(__file__).parent / jsd_name
        if not jsd_path.exists():
            raise FileNotFoundError(
                f"JSD score file not found: {jsd_path}. "
                f"Regenerate from lm_eval_vllm/train_set_jsd/{ds}_jsd_w3.json."
            )
        with jsd_path.open() as f:
            payload = json.load(f)
        return {str(k): float(v) for k, v in payload["by_doc"].items()}

    # -- JSD-quantile fallback for non-gsm8k datasets ---------------------
    def _build_jsd_pools(self, ds, n):
        labels = _normalized_ensemble_bin_labels(ds, n_bins=n)
        pools = [[] for _ in range(n)]
        for row in self.rows[ds]:
            did = row.get("doc_id")
            if did is None:
                continue
            b = labels.get(str(did))
            if b is None:
                continue
            pools[b].append(row)
        return pools

    # -- sampling --------------------------------------------------------
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
