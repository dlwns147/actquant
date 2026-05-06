"""Dump fp16 reference: per-sample answer-position logits AND CE.

For each (seqlen, sample):
  - logits at last `answer_len` positions saved to data/fp16_logits/*.pt
    (used as JSD reference distribution)
  - cross-entropy on those positions saved to data/fp16_baseline.json
    (used as ΔL = L_quant − L_fp16 reference)

One fp16 forward per sample.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]   # → actquant/search
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent       # → kv_proxy_study/


@torch.no_grad()
def forward_one(model, prompt_ids, answer_ids):
    full = torch.cat([prompt_ids, answer_ids], dim=1)
    out = model(full, use_cache=False)
    A = answer_ids.shape[1]
    shift_logits = out.logits[:, :-1, :]
    shift_labels = full[:, 1:]
    ans_logits = shift_logits[:, -A:, :].squeeze(0).contiguous()  # [A, V]
    ans_labels = shift_labels[:, -A:].squeeze(0)                   # [A]
    ce = F.cross_entropy(ans_logits, ans_labels).item()
    return ans_logits.cpu(), ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--seqlens", type=str, default="2048,4096,8192")
    ap.add_argument("--answer_len", type=int, default=128)
    ap.add_argument("--n_samples", type=int, default=4)
    args = ap.parse_args()

    seqlens = [int(s) for s in args.seqlens.split(",")]
    out_logits_dir = HERE / "data" / "fp16_logits"
    out_logits_dir.mkdir(parents=True, exist_ok=True)
    out_ce_path = HERE / "data" / "fp16_baseline.json"

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encs = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    rows = []
    for seq in seqlens:
        seg = seq + args.answer_len
        ces = []
        for i in range(args.n_samples):
            s = i * seg
            if s + seg > encs.shape[0]:
                break
            ids = encs[s:s+seg].unsqueeze(0).cuda()
            p, a = ids[:, :seq], ids[:, seq:]
            logits, ce = forward_one(model, p, a)
            torch.save(logits, out_logits_dir / f"seq{seq}_sample{i}.pt")
            ces.append(ce)
            print(f"  seq={seq} sample={i}  L_fp16={ce:.4f}  logits saved", flush=True)
        rows.append({"prompt_len": seq, "answer_len": args.answer_len, "fp16": ces})
        with open(out_ce_path, "w") as f:
            json.dump(rows, f, indent=2)
    print(f"\nSaved CE -> {out_ce_path}")
    print(f"Saved logits -> {out_logits_dir}")


if __name__ == "__main__":
    main()
