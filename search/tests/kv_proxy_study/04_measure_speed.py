"""Wall-clock benchmark for each proxy at multiple stride × seqlen values.

Quant-config-independent: fake_quant op count is the same regardless of bits.
Defaults to K2V2 for concreteness.

Outputs: data/speed_results.json
"""
import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from model.replace import replace_kv_cache  # noqa: E402

# Import loss_generate/loss_stride/loss_single from sibling 02_measure_ce.py
_spec = importlib.util.spec_from_file_location("_measure_ce", HERE / "02_measure_ce.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
loss_generate, loss_stride, loss_single = _m.loss_generate, _m.loss_stride, _m.loss_single


def build(model_path, kbits, vbits, kgs, vgs, R):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda")
    m = replace_kv_cache(model=m, tokenizer=tok, method=["kivi"],
                         n_block=m.config.num_hidden_layers,
                         k_quant_scheme="channel", v_quant_scheme="token",
                         residual_length=R, packing=False, quant_kv_output=False,
                         k_pruning_dim=0, v_pruning_dim=0)
    cfg = m.config.kivi_config
    n = m.config.num_hidden_layers
    cfg.k_bits = [kbits]*n; cfg.v_bits = [vbits]*n
    cfg.k_group_size = [kgs]*n; cfg.v_group_size = [vgs]*n
    m.eval()
    return m, tok


def timed(fn, n_runs=3):
    # Warmup
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompt_len", type=int, required=True)
    ap.add_argument("--answer_len", type=int, default=128)
    ap.add_argument("--residual_length", type=int, default=128)
    ap.add_argument("--strides", type=str, default="128,256,512,1024")
    ap.add_argument("--kbits", type=int, default=2)
    ap.add_argument("--vbits", type=int, default=2)
    ap.add_argument("--kgs", type=int, default=128)
    ap.add_argument("--vgs", type=int, default=128)
    args = ap.parse_args()

    strides = [int(s) for s in args.strides.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encs = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    seg = args.prompt_len + args.answer_len
    p = encs[:args.prompt_len].unsqueeze(0).cuda()
    a = encs[args.prompt_len:seg].unsqueeze(0).cuda()

    m, _ = build(args.model_path, args.kbits, args.vbits, args.kgs, args.vgs, args.residual_length)

    print(f"\n=== seq={args.prompt_len}+{args.answer_len}  K{args.kbits}V{args.vbits} ===")

    results = {}
    t = timed(lambda: loss_generate(m, tok, p, a, args.residual_length), n_runs=2)
    results["gen"] = t * 1000
    print(f"  L_gen        : {t*1000:8.1f} ms  (1 prefill + {args.answer_len} 1-token decodes)")

    t = timed(lambda: loss_single(m, tok, p, a, residual_length=0), n_runs=2)
    results["single_0"] = t * 1000
    print(f"  L_single_0   : {t*1000:8.1f} ms  (1 forward, R=0)")
    t = timed(lambda: loss_single(m, tok, p, a, residual_length=args.residual_length), n_runs=2)
    results["single_R"] = t * 1000
    print(f"  L_single_R   : {t*1000:8.1f} ms  (1 forward, R={args.residual_length})")

    for s in strides:
        n_chunks = (seg + s - 1) // s
        t = timed(lambda s=s: loss_stride(m, tok, p, a, args.residual_length, s), n_runs=2)
        results[f"stride_{s}"] = t * 1000
        print(f"  L_stride{s:>4}: {t*1000:8.1f} ms  ({n_chunks} chunks)")

    # Persist (resume-friendly: keyed by prompt_len)
    out = HERE / "data" / "speed_results.json"
    rows = json.load(open(out)) if out.exists() else []
    rows = [r for r in rows if r.get("prompt_len") != args.prompt_len]
    rows.append({"prompt_len": args.prompt_len, "answer_len": args.answer_len,
                 "kbits": args.kbits, "vbits": args.vbits, "ms": results})
    rows.sort(key=lambda r: r["prompt_len"])
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  → saved to {out}")

    del m
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
