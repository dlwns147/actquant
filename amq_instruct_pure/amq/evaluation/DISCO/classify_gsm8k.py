"""Classify gsm8k_train rows into DISCO groups by fp16/quant correctness.

Groups (strict-match):
    A: fp16=1, quant=0   — "needs fixing"
    B: fp16=1, quant=1   — preserve
    C: fp16=0, quant=1   — noise but kept to track distribution
    D: fp16=0, quant=0   — trivially wrong

Reads vllm sample dumps from
  ``/NAS/SJ/actquant/poc/benchmark_proxy/gsm8k/gsm8k_train_vllm_result/``
for the fp16 baseline and the 3-bit-uniform quantized model, writes
``gsm8k_labels.json`` next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path


FP16_DIR = Path(
    "/NAS/SJ/actquant/poc/benchmark_proxy/gsm8k/gsm8k_train_vllm_result/"
    "meta-llama__Llama-3.1-8B-Instruct"
)
QUANT_DIR = Path(
    "/NAS/SJ/actquant/poc/benchmark_proxy/gsm8k/gsm8k_train_vllm_result/"
    "__SSD__Woo__hqq__Llama-3.1-8B-Instruct_3bit_128gs_1axis_fake"
)
OUT_PATH = Path(__file__).parent / "gsm8k_labels.json"
FILTER = "strict-match"


def _load_correctness(d):
    """Returns {doc_id: bool(strict-match == 1.0)} from a samples_*.jsonl."""
    sample_files = list(d.glob("samples_gsm8k_cot_train_*.jsonl"))
    if not sample_files:
        raise FileNotFoundError(f"no samples_*.jsonl in {d}")
    if len(sample_files) > 1:
        sample_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"  multiple sample files in {d.name}, using newest: {sample_files[0].name}")
    out = {}
    with sample_files[0].open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("filter") != FILTER:
                continue
            out[int(row["doc_id"])] = float(row["exact_match"]) == 1.0
    return out


def main():
    fp = _load_correctness(FP16_DIR)
    qq = _load_correctness(QUANT_DIR)
    common = sorted(set(fp) & set(qq))

    labels = {}
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for did in common:
        f, q = fp[did], qq[did]
        if f and not q:
            g = "A"
        elif f and q:
            g = "B"
        elif (not f) and q:
            g = "C"
        else:
            g = "D"
        labels[str(did)] = g
        counts[g] += 1

    payload = {
        "filter": FILTER,
        "fp16_dir": str(FP16_DIR),
        "quant_dir": str(QUANT_DIR),
        "total": len(common),
        "counts": counts,
        "by_doc": labels,
    }
    OUT_PATH.write_text(json.dumps(payload))
    print(f"saved {OUT_PATH}")
    print(f"  total={len(common)}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
