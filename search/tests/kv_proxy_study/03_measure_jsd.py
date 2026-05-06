"""JSD-based proxy comparison.

Same proxies as 02_measure_ce.py but the metric is
  JSD(quant_logits, fp16_logits) averaged over answer positions
instead of CE against ground-truth tokens. Requires fp16 logits dumped by
01_dump_fp16.py.

Outputs: data/jsd_results.json — same row layout as ce_results.json.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]   # → actquant/search
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent       # → kv_proxy_study/

from model.replace import replace_kv_cache  # noqa: E402
from utils.loss import JSD  # noqa: E402


def build_kivi_model(model_path, kbits, vbits, kgs, vgs, residual_length,
                     packing=False, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="cuda")
    model = replace_kv_cache(
        model=model, tokenizer=tok, method=["kivi"], n_block=model.config.num_hidden_layers,
        k_quant_scheme="channel", v_quant_scheme="token",
        residual_length=residual_length, packing=packing, quant_kv_output=False,
        k_pruning_dim=0, v_pruning_dim=0,
    )
    cfg = model.config.kivi_config
    n = model.config.num_hidden_layers
    cfg.k_bits = [kbits]*n; cfg.v_bits = [vbits]*n
    cfg.k_group_size = [kgs]*n; cfg.v_group_size = [vgs]*n
    model.eval()
    return model, tok

@torch.no_grad()
def logits_generate(model, prompt_ids, answer_ids, residual_length):
    """Logits at answer positions via teacher-forced single-step decode."""
    model.config.kivi_config.residual_length = residual_length
    model.config.quant_kv_output = False
    model.config.use_cache = True

    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    last_logit = out.logits[:, -1, :]
    answer_logits = []
    for i in range(answer_ids.shape[1]):
        answer_logits.append(last_logit)
        if i < answer_ids.shape[1] - 1:
            step = model(answer_ids[:, i:i+1], past_key_values=past, use_cache=True)
            past = step.past_key_values
            last_logit = step.logits[:, -1, :]
    return torch.stack(answer_logits, dim=1).squeeze(0)  # [A, V]


@torch.no_grad()
def logits_stride(model, prompt_ids, answer_ids, residual_length, stride):
    model.config.kivi_config.residual_length = residual_length
    model.config.quant_kv_output = False
    model.config.use_cache = True

    full = torch.cat([prompt_ids, answer_ids], dim=1)
    L = full.shape[1]; A = answer_ids.shape[1]
    past = None
    chunks = []
    for s in range(0, L, stride):
        e = min(s+stride, L)
        out = model(full[:, s:e], past_key_values=past, use_cache=True)
        past = out.past_key_values
        chunks.append(out.logits)
    logits = torch.cat(chunks, dim=1)
    shift_logits = logits[:, :-1, :]
    return shift_logits[:, -A:, :].squeeze(0)


@torch.no_grad()
def logits_single(model, prompt_ids, answer_ids, residual_length):
    model.config.kivi_config.residual_length = residual_length
    model.config.quant_kv_output = True
    model.config.use_cache = False

    full = torch.cat([prompt_ids, answer_ids], dim=1)
    A = answer_ids.shape[1]
    out = model(full, use_cache=False)
    shift_logits = out.logits[:, :-1, :]
    return shift_logits[:, -A:, :].squeeze(0)


def jsd_per_position(quant_logits, fp16_logits):
    """JSD averaged over answer positions. Inputs: [A, V] each."""
    jsd = JSD()
    return jsd(quant_logits, fp16_logits).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--residual_length", type=int, default=128)
    ap.add_argument("--strides", type=str, default="128,256,512,1024")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--prompt_len", type=int, required=True)
    ap.add_argument("--answer_len", type=int, default=128)
    ap.add_argument("--configs", type=str, required=True)
    ap.add_argument("--kgs", type=int, default=128)
    ap.add_argument("--vgs", type=int, default=128)
    args = ap.parse_args()

    strides = [int(s) for s in args.strides.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encs = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    seg = args.prompt_len + args.answer_len
    samples, fp16_logits = [], []
    fp16_dir = HERE / "data" / "fp16_logits"
    for i in range(args.n_samples):
        s = i * seg
        if s + seg > encs.shape[0]:
            break
        ids = encs[s:s+seg].unsqueeze(0).cuda()
        samples.append((ids[:, :args.prompt_len], ids[:, args.prompt_len:]))
        fp = fp16_dir / f"seq{args.prompt_len}_sample{i}.pt"
        assert fp.exists(), f"Missing fp16 logits: {fp}"
        fp16_logits.append(torch.load(fp).cuda())  # [A, V]

    out_json = HERE / "data" / "jsd_results.json"
    existing = []
    if out_json.exists():
        try:
            existing = json.load(open(out_json))
        except Exception:
            existing = []
    done = {(r["cfg"], r.get("prompt_len", -1)) for r in existing}
    rows = list(existing)

    for cfg in args.configs.split(","):
        if (cfg, args.prompt_len) in done:
            print(f"[skip] {cfg} @ {args.prompt_len}", flush=True); continue
        kb, vb = [int(x) for x in cfg.split(":")]
        print(f"\n=== Config K{kb}V{vb} seq={args.prompt_len} R={args.residual_length} strides={strides} ===")
        model, _ = build_kivi_model(args.model_path, kb, vb, args.kgs, args.vgs, args.residual_length)
        l_gen, l_sgl, l_sgl_r = [], [], []
        l_strides = {s: [] for s in strides}
        for i, (p, a) in enumerate(samples):
            fp = fp16_logits[i]
            j_gen = jsd_per_position(logits_generate(model, p, a, args.residual_length), fp)
            j_sgl0 = jsd_per_position(logits_single(model, p, a, residual_length=0), fp)
            j_sglR = jsd_per_position(logits_single(model, p, a, residual_length=args.residual_length), fp)
            l_gen.append(j_gen); l_sgl.append(j_sgl0); l_sgl_r.append(j_sglR)
            stride_strs = []
            for st in strides:
                jv = jsd_per_position(logits_stride(model, p, a, args.residual_length, st), fp)
                l_strides[st].append(jv)
                stride_strs.append(f"str{st}={jv:.5f}")
            print(f"  sample {i}: gen={j_gen:.5f}  sgl0={j_sgl0:.5f}  sglR={j_sglR:.5f}  "
                  + "  ".join(stride_strs), flush=True)
        rows.append({
            "cfg": cfg, "prompt_len": args.prompt_len, "answer_len": args.answer_len,
            "residual_length": args.residual_length,
            "gen": l_gen, "single": l_sgl, "single_r": l_sgl_r,
            "strides": {str(s): l_strides[s] for s in strides},
        })
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)
        del model
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
