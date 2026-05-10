"""Measure L_gen JSD across an expanded architecture pool:
  · 9 fixed-precision configs  (K∈{2,3,4} × V∈{2,3,4})
  · N_RANDOM mixed-precision random archs (per-layer K,V bits)

Loads the model once, swaps model.config.kivi_config.{k_bits, v_bits} per arch,
runs prefill+stride=A (1 chunk for answer = 2 forwards per sample). Uses
fp16 logits already dumped to data/fp16_logits.

Saves data/random_arch_jsd.json with arch list + per-(arch, seq) JSD.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from model.replace import replace_kv_cache  # noqa: E402
from utils.loss import JSD  # noqa: E402


def avg_bits(arch):
    return (float(np.mean(arch["k"])) + float(np.mean(arch["v"]))) / 2


def make_archs(n_layers, n_random, seed=42):
    rng = np.random.default_rng(seed)
    archs = []
    # 9 fixed configs
    for kb in [2, 3, 4]:
        for vb in [2, 3, 4]:
            archs.append({
                "name": f"K{kb}V{vb}", "type": "fixed",
                "k": [kb] * n_layers, "v": [vb] * n_layers,
            })
    # Random mixed: span 3 bit-budget regimes for spread
    n_per_regime = max(1, n_random // 3)
    for regime, choices in [("low", [2, 3]), ("mid", [2, 3, 4]), ("high", [3, 4])]:
        for i in range(n_per_regime):
            archs.append({
                "name": f"rand_{regime}_{i}", "type": "random",
                "k": rng.choice(choices, n_layers).tolist(),
                "v": rng.choice(choices, n_layers).tolist(),
            })
    return archs


@torch.no_grad()
def jsd_prefill_stride_one(model, arch, prompt_ids, answer_ids, R, fp16_logits):
    """JSD of prefill-prompt + 1-chunk answer forward against fp16 reference."""
    cfg = model.config.kivi_config
    cfg.k_bits = arch["k"]
    cfg.v_bits = arch["v"]
    cfg.residual_length = R
    model.config.quant_kv_output = False
    model.config.use_cache = True

    A = answer_ids.shape[1]
    out_p = model(prompt_ids, use_cache=True)
    past = out_p.past_key_values
    last_prompt_logit = out_p.logits[:, -1:, :]
    out_a = model(answer_ids, past_key_values=past, use_cache=True)
    pred_logits = torch.cat([last_prompt_logit, out_a.logits[:, :-1, :]], dim=1).squeeze(0)
    return JSD()(pred_logits, fp16_logits).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--seqlens", type=str, default="2048,4096,8192")
    ap.add_argument("--answer_len", type=int, default=128)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--n_random", type=int, default=30)
    ap.add_argument("--residual_length", type=int, default=128)
    ap.add_argument("--kgs", type=int, default=128)
    ap.add_argument("--vgs", type=int, default=128)
    args = ap.parse_args()

    seqlens = [int(s) for s in args.seqlens.split(",")]
    fp16_dir = HERE / "data" / "fp16_logits"

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cuda")
    model = replace_kv_cache(
        model=model, tokenizer=tok, method=["kivi"],
        n_block=model.config.num_hidden_layers,
        k_quant_scheme="channel", v_quant_scheme="token",
        residual_length=args.residual_length, packing=False, quant_kv_output=False,
        k_pruning_dim=0, v_pruning_dim=0)
    n_layers = model.config.num_hidden_layers
    cfg = model.config.kivi_config
    cfg.k_group_size = [args.kgs] * n_layers
    cfg.v_group_size = [args.vgs] * n_layers
    model.eval()

    # Pre-load WikiText samples + fp16 logits
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encs = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    samples = {}  # (seq, i) -> (p, a, fp16_logits)
    for seq in seqlens:
        seg = seq + args.answer_len
        for i in range(args.n_samples):
            s = i * seg
            if s + seg > encs.shape[0]:
                break
            ids = encs[s:s+seg].unsqueeze(0).cuda()
            p, a = ids[:, :seq], ids[:, seq:]
            fp = fp16_dir / f"seq{seq}_a{args.answer_len}_sample{i}.pt"
            assert fp.exists(), f"Missing {fp}"
            samples[(seq, i)] = (p, a, torch.load(fp).cuda())

    # Build arch pool
    archs = make_archs(n_layers, args.n_random)
    print(f"Total archs: {len(archs)}  ({sum(a['type']=='fixed' for a in archs)} fixed + "
          f"{sum(a['type']=='random' for a in archs)} random)")

    out_json = HERE / "data" / "random_arch_jsd.json"
    rows = []
    for ai, arch in enumerate(archs):
        per_seq = {}
        for seq in seqlens:
            jsds = []
            for i in range(args.n_samples):
                if (seq, i) not in samples:
                    continue
                p, a, fp = samples[(seq, i)]
                j = jsd_prefill_stride_one(model, arch, p, a, args.residual_length, fp)
                jsds.append(j)
            per_seq[seq] = jsds
        ab = avg_bits(arch)
        rows.append({
            "name": arch["name"], "type": arch["type"],
            "avg_bits": ab,
            "k_bits": arch["k"], "v_bits": arch["v"],
            "jsd": {str(s): per_seq[s] for s in seqlens},
        })
        means = {s: float(np.mean(per_seq[s])) for s in seqlens}
        print(f"  [{ai+1:>3}/{len(archs)}] {arch['name']:>20}  avg_bits={ab:.3f}  "
              + "  ".join(f"seq{s}={means[s]:.5f}" for s in seqlens), flush=True)
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)
    print(f"\nSaved → {out_json}")


if __name__ == "__main__":
    main()
