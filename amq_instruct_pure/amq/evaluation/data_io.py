"""Dataset I/O helpers + cache paths shared by evaluator_jsd_stratified and
the DISCO sample loaders.

Supported datasets
------------------
* ``gsm8k_cot_train``, ``ifeval_train``, ``mbpp_train`` — lm_eval-derived
  layout. Each is a single FP16 ``samples_<task>_*.jsonl`` produced by
  ``lm_eval_vllm/lm_eval_vllm.py``. The same file is used as both the row
  source AND the FP16 answer source (its own generations live in
  ``row["resps"][0][0]``). For ifeval_train / mbpp_train we filter rows to
  ``datasets/<task>/split_meta.json::train_ids``; gsm8k_cot_train has no
  split_meta and uses the whole file.
* ``wikitext2`` — long-form natural-text calibration anchor. The pool is a
  fixed list of chunks built from the wikitext-2 train split, each
  ``_WIKITEXT2_SEQLEN`` tokens long, wrapped into a task-row shape with an
  empty prompt and the chunk text as ``answer_text``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_DEFAULT_N_SAMPLE = 16


# ---------------------------------------------------------------------------
# Per-task population JSD mean (FP16 vs HQQ-w3 token-level JSD on the train
# split). Used to normalize predictor targets across tasks so the NSGA2
# objective gives each task equal weight regardless of its absolute JSD
# scale. Measured by lm_eval_vllm/train_set_jsd/compute_jsd_train.py.
#
# Source: lm_eval_vllm/train_set_jsd/<task>_jsd_w3.json -> summary.mean_jsd
# Re-measure and update here if the underlying FP16 / w3 model changes.
# ---------------------------------------------------------------------------
_TASK_JSD_MEAN = {
    # "gsm8k_cot_train": 0.04807,
    # "ifeval_train":    0.09838,
    # "mbpp_train":      0.08017,
    "gsm8k_cot_train": 1.0,
    "ifeval_train":    1.0,
    "mbpp_train":      1.0,
    # wikitext2 calibration: full-sequence token JSD, no per-task population
    # measurement. Defaults to 1.0 (no normalization) until measured.
    "wikitext2": 1.0,
}


def task_jsd_mean(dataset: str) -> float:
    """Population JSD mean used for cross-task normalization. Defaults to 1.0
    for unknown datasets (i.e. no normalization)."""
    return float(_TASK_JSD_MEAN.get(dataset, 1.0))


# ---------------------------------------------------------------------------
# lm_eval-derived layout
# ---------------------------------------------------------------------------

_LM_EVAL_RESULTS = Path("/NAS/SJ/actquant/poc/benchmark_proxy/lm_eval_vllm/results")
_LM_EVAL_DATASETS = Path("/NAS/SJ/actquant/poc/benchmark_proxy/lm_eval_vllm/datasets")
_FP16_MODEL_ID = "meta-llama__Llama-3.1-8B-Instruct"

# task → (results_subdir, filename_glob, split_meta_subdir or None)
# split_meta_subdir = None means "use whole file as train" (gsm8k_cot_train).
_LM_EVAL_TASKS = {
    "gsm8k_cot_train": ("gsm8k_cot_train", "samples_gsm8k_cot_train_*.jsonl", None),
    "ifeval_train":    ("ifeval",          "samples_ifeval_*.jsonl",          "ifeval"),
    "mbpp_train":      ("mbpp",            "samples_mbpp_*.jsonl",            "mbpp"),
    # ifeval_pp / apps have no split_meta.json -> whole file is "train".
    "ifeval_pp_train": ("ifeval_pp",       "samples_ifeval_pp_*.jsonl",       None),
    "apps_train":      ("apps",            "samples_apps_*.jsonl",            None),
}

# Per-task filter applied to the lm_eval rows (some tasks emit one row per
# filter; we keep only the canonical one).
_LM_EVAL_FILTER = {
    "gsm8k_cot_train": "strict-match",
    "ifeval_train":    None,
    "mbpp_train":      None,
    "ifeval_pp_train": None,
    "apps_train":      None,
}


def is_lm_eval_dataset(dataset: str) -> bool:
    return dataset in _LM_EVAL_TASKS


# ---------------------------------------------------------------------------
# wikitext2 calibration source
# ---------------------------------------------------------------------------
#
# wikitext2 isn't an instruction task; we use it as a "long-form natural text"
# anchor. The pool is a fixed list of chunks built from the wikitext-2 train
# split, each ``_WIKITEXT2_SEQLEN`` tokens long. Each chunk is wrapped into a
# task-row shape so the rest of the pipeline (SampleLoader / DISCO / evaluator)
# can consume it uniformly:
#
#     {"doc_id": <int>, "prompt_text": "", "answer_text": <chunk_text>}
#
# Because ``prompt_text`` is empty, ``_build_prompt`` returns "" and
# ``_answer_span_ids`` returns (empty prompt_ids, full chunk token ids) — the
# evaluator then scores token-level JSD over the *entire* chunk (i.e. no
# prompt/answer split, just NLL-style coverage).

_WIKITEXT2_DATASET_NAME = "wikitext2"
_WIKITEXT2_SEQLEN = 2048
_WIKITEXT2_N_ROWS = 128             # number of shuffled wikitext-2 train rows to concat
_WIKITEXT2_CACHE = {}               # {(id(tokenizer), seed): chunks}
# After tokenizing the joined first-128 rows we get ~8.6k tokens, i.e. only
# 4 chunks of seqlen=2048. This entire pool (≈4 chunks) is used every iter,
# never sub-sampled. Matches amq/utils/data.get_wikitext2_trainenc behavior.
# The shuffle seed is supplied by the caller (typically the SampleLoader's
# search seed) so wikitext2 pools differ across search seeds.


def is_wikitext2(dataset: str) -> bool:
    return dataset == _WIKITEXT2_DATASET_NAME


def _build_wikitext2_chunks(tokenizer, seed):
    """Mirror amq/utils/data.get_wikitext2_trainenc: shuffle wikitext-2 train
    with ``seed``, take the first ``_WIKITEXT2_N_ROWS`` rows, join with "\n\n",
    tokenize, and slice into ``_WIKITEXT2_SEQLEN``-long chunks. Each chunk is
    decoded back to text so the evaluator can re-tokenize via the standard
    ``prompt + answer_text`` path.

    Typical result with n_rows=128, seqlen=2048: ≈4 chunks.
    """
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    traindata = traindata.shuffle(seed=int(seed))

    joined = "\n\n".join(traindata[: _WIKITEXT2_N_ROWS]["text"])
    ids = tokenizer(joined, return_tensors="pt").input_ids[0]
    n_chunks = ids.numel() // _WIKITEXT2_SEQLEN

    chunks = []
    for i in range(n_chunks):
        chunk_ids = ids[i * _WIKITEXT2_SEQLEN: (i + 1) * _WIKITEXT2_SEQLEN]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
        chunks.append({
            "doc_id": i,
            "prompt_text": "",
            "answer_text": chunk_text,
        })
    return chunks


def _get_wikitext2_pool(tokenizer, seed):
    key = (id(tokenizer) if tokenizer is not None else None, int(seed))
    cached = _WIKITEXT2_CACHE.get(key)
    if cached is not None:
        return cached
    if tokenizer is None:
        raise ValueError(
            "wikitext2 pool not built yet for this seed; the first caller "
            "must pass a tokenizer (typically the SampleLoader)."
        )
    chunks = _build_wikitext2_chunks(tokenizer, seed)
    _WIKITEXT2_CACHE[key] = chunks
    return chunks


def build_wikitext2_pool(tokenizer, seed):
    """Public hook for the SampleLoader to eagerly materialize the pool with
    a known tokenizer + shuffle seed. Subsequent
    ``_load_all_rows('wikitext2', tokenizer=..., wikitext2_seed=seed)`` calls
    hit the cache."""
    return _get_wikitext2_pool(tokenizer, seed)


# ---------------------------------------------------------------------------
# lm_eval samples_*.jsonl loader
# ---------------------------------------------------------------------------

def _find_one_jsonl(results_subdir: str, pattern_glob: str) -> Path:
    sub = _LM_EVAL_RESULTS / results_subdir / _FP16_MODEL_ID
    if not sub.is_dir():
        raise FileNotFoundError(
            f"FP16 results directory not found: {sub}. "
            f"Expected lm_eval output under {_LM_EVAL_RESULTS}.")
    candidates = sorted(sub.glob(pattern_glob))
    if not candidates:
        raise FileNotFoundError(
            f"No samples jsonl matching {pattern_glob!r} under {sub}.")
    # If multiple timestamps exist, prefer the newest.
    return candidates[-1]


def _load_split_train_ids(split_meta_subdir: str) -> set[int] | None:
    meta_path = _LM_EVAL_DATASETS / split_meta_subdir / "split_meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open() as f:
        return set(int(x) for x in json.load(f)["train_ids"])


def _load_lm_eval_rows(dataset: str) -> list[dict]:
    subdir, glob, split_dir = _LM_EVAL_TASKS[dataset]
    jsonl = _find_one_jsonl(subdir, glob)
    flt = _LM_EVAL_FILTER.get(dataset)

    rows = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if flt is not None and r.get("filter") != flt:
                continue
            rows.append(r)

    if split_dir is not None:
        ids = _load_split_train_ids(split_dir)
        if ids is None:
            raise FileNotFoundError(
                f"split_meta.json not found for {dataset!r} "
                f"under {_LM_EVAL_DATASETS}/{split_dir}/. "
                f"Run lm_eval_vllm/split_dataset.py first.")
        rows = [r for r in rows if int(r["doc_id"]) in ids]
    return rows


# ---------------------------------------------------------------------------
# Public: load rows / fp16 answers
# ---------------------------------------------------------------------------

def _load_all_rows(dataset, tokenizer=None, wikitext2_seed=0):
    if is_wikitext2(dataset):
        return list(_get_wikitext2_pool(tokenizer, wikitext2_seed))
    if is_lm_eval_dataset(dataset):
        return _load_lm_eval_rows(dataset)
    raise KeyError(f"Unknown dataset: {dataset!r}")


def _load_fp16_answers(dataset, tokenizer=None, wikitext2_seed=0):
    """{doc_id: answer_text}. For lm_eval-derived datasets we use the FP16
    model's own ``resps[0][0]`` (i.e. its actual generation on that doc).

    For wikitext2 the "answer" is the chunk text itself (no prompt/answer
    split); the same chunk text is used as both the row body and the fp16
    target so the evaluator scores token JSD over the whole sequence."""
    if is_wikitext2(dataset):
        return {r["doc_id"]: r["answer_text"]
                for r in _get_wikitext2_pool(tokenizer, wikitext2_seed)}

    if is_lm_eval_dataset(dataset):
        rows = _load_lm_eval_rows(dataset)
        out = {}
        for r in rows:
            did = r.get("doc_id")
            if did is None:
                continue
            resps = r.get("resps")
            if not resps or not resps[0]:
                continue
            ans = resps[0][0]
            if isinstance(ans, str):
                out[did] = ans
        return out

    raise KeyError(f"Unknown dataset: {dataset!r}")


# ---------------------------------------------------------------------------
# jsd-bin labeling (one-time per dataset, at SampleLoader init).
# ---------------------------------------------------------------------------

# Per-doc fp16-vs-w3 JSD score files used as the binning score for the
# lm_eval-derived tasks. Located alongside the DISCO labels JSONs in
# evaluation/DISCO/.
_DISCO_DIR = Path(__file__).resolve().parent / "DISCO"
_LM_EVAL_BIN_SCORE_FILES = {
    "gsm8k_cot_train": _DISCO_DIR / "gsm8k_cot_train_jsd.json",
    "ifeval_train":    _DISCO_DIR / "ifeval_train_jsd.json",
    "mbpp_train":      _DISCO_DIR / "mbpp_train_jsd.json",
}


def _quantile_bin_labels_from_score(score_by_id: dict, n_bins: int) -> dict:
    """Given {doc_id_str: float_score}, return {doc_id_str: bin_idx} using
    quantile-uniform binning (np.array_split on sorted order)."""
    ids = list(score_by_id.keys())
    scores = np.array([score_by_id[i] for i in ids])
    order = np.argsort(scores)
    bin_of = np.empty(len(order), dtype=int)
    bins_per = max(1.0, len(order) / n_bins)
    for i, oi in enumerate(order):
        bin_of[oi] = min(int(i / bins_per), n_bins - 1)
    return {ids[i]: int(bin_of[i]) for i in range(len(ids))}


def _normalized_ensemble_bin_labels(dataset, n_bins):
    """Returns ``{doc_id_str: bin_idx in [0, n_bins)}`` for ``dataset``,
    binning by a per-doc JSD score.

    * lm_eval tasks (``gsm8k_cot_train`` / ``ifeval_train`` / ``mbpp_train``):
      uses the precomputed fp16-vs-w3 token-level JSD cached under
      ``evaluation/DISCO/<task>_jsd.json`` (produced by
      ``lm_eval_vllm/train_set_jsd/compute_jsd_train.py``). This is the same
      score the DISCO loader uses for within-group quantile ordering, so the
      plain loader and the DISCO loader bin by a consistent quantity.
    * ``wikitext2``: no per-chunk JSD score; assign chunks to bins uniformly
      by index so stratified samplers degrade to a deterministic, balanced
      partition.
    """
    if is_wikitext2(dataset):
        if not _WIKITEXT2_CACHE:
            raise RuntimeError(
                "wikitext2 pool not built; call build_wikitext2_pool("
                "tokenizer, seed) before requesting jsd-bin labels."
            )
        pool = next(iter(_WIKITEXT2_CACHE.values()))
        ids = [str(r["doc_id"]) for r in pool]
        labels = {}
        per_bin = max(1.0, len(ids) / n_bins)
        for i, did in enumerate(ids):
            labels[did] = min(int(i / per_bin), n_bins - 1)
        return labels

    if dataset in _LM_EVAL_BIN_SCORE_FILES:
        path = _LM_EVAL_BIN_SCORE_FILES[dataset]
        if not path.exists():
            raise FileNotFoundError(
                f"jsd-bin labels for {dataset!r} require {path}. "
                f"Regenerate from lm_eval_vllm/train_set_jsd/{dataset}_jsd_w3.json."
            )
        with path.open() as f:
            payload = json.load(f)
        score_by_id = {str(k): float(v) for k, v in payload["by_doc"].items()}
        return _quantile_bin_labels_from_score(score_by_id, n_bins)

    raise KeyError(f"Unknown dataset for jsd-bin labels: {dataset!r}")


# ---------------------------------------------------------------------------
# Prompt + answer-span
# ---------------------------------------------------------------------------

def _build_prompt(dataset, row, tokenizer=None):
    """Reconstruct the chat-templated prompt for `row`.

    lm_eval-derived tasks store the already-templated prompt in
    ``row.arguments.gen_args_0.arg_0``. wikitext2 has no prompt — the whole
    chunk text is the "answer".
    """
    if is_wikitext2(dataset):
        return row.get("prompt_text", "")
    if is_lm_eval_dataset(dataset):
        return row["arguments"]["gen_args_0"]["arg_0"]
    raise KeyError(f"Unknown dataset: {dataset!r}")


def _answer_span_ids(dataset, tokenizer, row, answer_text):
    """Tokenize prompt and (prompt + answer); return (prompt_ids, answer_ids).

    answer_ids are the token positions corresponding to ``answer_text`` after
    the prompt. Used to slice logits to the answer span only. For wikitext2
    prompt is empty, so prompt_ids has length 0 and answer_ids covers the
    entire chunk — the evaluator must clamp ``start_pos = max(0, ...)``.
    """
    if is_wikitext2(dataset):
        prompt = row.get("prompt_text", "")
    elif is_lm_eval_dataset(dataset):
        prompt = row["arguments"]["gen_args_0"]["arg_0"]
    else:
        raise KeyError(f"Unknown dataset: {dataset!r}")

    if prompt:
        prompt_ids = tokenizer(prompt, return_tensors="pt",
                               add_special_tokens=False).input_ids[0]
        full_ids = tokenizer(prompt + answer_text, return_tensors="pt",
                             add_special_tokens=False).input_ids[0]
        answer_ids = full_ids[prompt_ids.numel():]
    else:
        # No prompt (wikitext2): the whole sequence is the answer.
        import torch as _torch
        full_ids = tokenizer(answer_text, return_tensors="pt",
                             add_special_tokens=False).input_ids[0]
        prompt_ids = _torch.empty(0, dtype=full_ids.dtype)
        answer_ids = full_ids
    return prompt_ids, answer_ids
