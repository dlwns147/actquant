"""CE-based proxy comparison for KV-cache quantisation.

Compares answer-token cross-entropy across four KV-cache simulation paths
on the same quantised model:
  L_gen     : real-decode CE  (use_cache=True, R=128, token-by-token).
  L_stride  : chunked forward (use_cache=True, R=128, stride ∈ {128..1024}).
  L_single0 : 1-shot forward  (use_cache=False, quant_kv_output=True, R=0).
  L_singleR : 1-shot forward  (use_cache=False, quant_kv_output=True, R=128 — residual-aware).

Outputs: data/ce_results.json — {cfg, prompt_len, gen, single, single_r, strides:{s:[..]}}
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

from model.replace import replace_kv_cache  # noqa: E402


def build_kivi_model(model_path, kbits, vbits, kgs, vgs, residual_length, packing=False, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="cuda")
    model = replace_kv_cache(
        model=model, tokenizer=tok, method=["kivi"], n_block=model.config.num_hidden_layers,
        k_quant_scheme="channel", v_quant_scheme="token",
        residual_length=residual_length, packing=packing, quant_kv_output=False,
        k_pruning_dim=0, v_pruning_dim=0,
    )
    cfg = model.config.kivi_config
    cfg.k_bits = [kbits]*model.config.num_hidden_layers
    cfg.v_bits = [vbits]*model.config.num_hidden_layers
    cfg.k_group_size = [kgs]*model.config.num_hidden_layers
    cfg.v_group_size = [vgs]*model.config.num_hidden_layers
    model.eval()
    return model, tok


@torch.no_grad()
def loss_generate(model, tok, prompt_ids, answer_ids, residual_length):
    """Real-decode answer loss: prefill prompt, then teacher-force each answer
    token through generate-equivalent single-step forwards, compute CE."""
    model.config.kivi_config.residual_length = residual_length
    model.config.quant_kv_output = False
    model.config.use_cache = True

    full = torch.cat([prompt_ids, answer_ids], dim=1)
    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    losses = []
    # logits at position -1 of prefill predicts answer[0]
    last_logit = out.logits[:, -1, :]
    for i in range(answer_ids.shape[1]):
        target = answer_ids[:, i]
        losses.append(F.cross_entropy(last_logit, target, reduction="none"))
        if i < answer_ids.shape[1] - 1:
            step = model(answer_ids[:, i:i+1], past_key_values=past, use_cache=True)
            past = step.past_key_values
            last_logit = step.logits[:, -1, :]
    return torch.stack(losses).mean().item()


@torch.no_grad()
def loss_stride(model, tok, prompt_ids, answer_ids, residual_length, stride):
    """Stride mode: chunked use_cache=True forward, last_tokens=len(answer)."""
    model.config.kivi_config.residual_length = residual_length
    model.config.quant_kv_output = False
    model.config.use_cache = True

    full = torch.cat([prompt_ids, answer_ids], dim=1)
    L = full.shape[1]
    past = None
    chunk_logits = []
    for s in range(0, L, stride):
        e = min(s+stride, L)
        out = model(full[:, s:e], past_key_values=past, use_cache=True)
        past = out.past_key_values
        chunk_logits.append(out.logits)
    logits = torch.cat(chunk_logits, dim=1)
    # answer positions in shifted space: last len(answer) of logits[:-1]
    A = answer_ids.shape[1]
    shift_logits = logits[:, :-1, :]
    shift_labels = full[:, 1:]
    ans_logits = shift_logits[:, -A:, :].reshape(-1, shift_logits.size(-1))
    ans_labels = shift_labels[:, -A:].reshape(-1)
    return F.cross_entropy(ans_logits, ans_labels).item()


@torch.no_grad()
def loss_single(model, tok, prompt_ids, answer_ids, residual_length=0):
    """Single-pass mode (quant_kv_output=True, use_cache=False).

    With the residual-aware quant_kv_output(), residual_length=R leaves the
    last R tokens of K/V at full precision (matches real-decode FP residual).
    residual_length=0 is the legacy "quantise everything" path.
    """
    model.config.kivi_config.residual_length = residual_length
    model.config.quant_kv_output = True
    model.config.use_cache = False

    full = torch.cat([prompt_ids, answer_ids], dim=1)
    out = model(full, use_cache=False)
    A = answer_ids.shape[1]
    shift_logits = out.logits[:, :-1, :]
    shift_labels = full[:, 1:]
    ans_logits = shift_logits[:, -A:, :].reshape(-1, shift_logits.size(-1))
    ans_labels = shift_labels[:, -A:].reshape(-1)
    return F.cross_entropy(ans_logits, ans_labels).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--residual_length", type=int, default=128)
    ap.add_argument("--stride", type=int, default=128,
                    help="Single stride value (used when --strides is empty)")
    ap.add_argument("--strides", type=str, default="",
                    help="Comma-separated stride values to sweep, e.g. 128,256,512,1024")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--prompt_len", type=int, default=2048)
    ap.add_argument("--answer_len", type=int, default=128)
    # Sweep configs (multiple bit settings to populate a correlation scatter)
    ap.add_argument("--configs", type=str, default="2:2,2:4,3:3,4:4,8:8",
                    help="comma-sep list of kbits:vbits to sweep")
    ap.add_argument("--kgs", type=int, default=128)
    ap.add_argument("--vgs", type=int, default=128)
    args = ap.parse_args()

    # Build a small set of long prompts from any text — answer is the last `answer_len` tokens.
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encs = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    samples = []
    seg = args.prompt_len + args.answer_len
    for i in range(args.n_samples):
        s = i * seg
        if s + seg > encs.shape[0]:
            break
        ids = encs[s:s+seg].unsqueeze(0).cuda()
        samples.append((ids[:, :args.prompt_len], ids[:, args.prompt_len:]))

    strides = [int(s) for s in args.strides.split(",")] if args.strides else [args.stride]

    out_json = HERE / "data" / "ce_results.json"
    # Resume support: load existing results and skip configs already done.
    # Entries keyed by (cfg, prompt_len) so multiple seqlens accumulate.
    existing = []
    if out_json.exists():
        try:
            existing = json.load(open(out_json))
        except Exception:
            existing = []
    done_keys = {(r["cfg"], r.get("prompt_len", -1)) for r in existing}
    rows = list(existing)

    for cfg in args.configs.split(","):
        if (cfg, args.prompt_len) in done_keys:
            print(f"[skip] {cfg} @ seq={args.prompt_len} already in results", flush=True)
            continue
        kb, vb = [int(x) for x in cfg.split(":")]
        print(f"\n=== Config K{kb}V{vb} (kgs={args.kgs}, vgs={args.vgs}, R={args.residual_length}, strides={strides}) ===")
        model, _ = build_kivi_model(args.model_path, kb, vb, args.kgs, args.vgs, args.residual_length)
        l_gen, l_sgl, l_sgl_r = [], [], []
        l_strides = {s: [] for s in strides}
        for i, (p, a) in enumerate(samples):
            g = loss_generate(model, tok, p, a, args.residual_length)
            n = loss_single(model, tok, p, a, residual_length=0)              # legacy single (R=0)
            nr = loss_single(model, tok, p, a, residual_length=args.residual_length)  # residual-aware single
            l_gen.append(g); l_sgl.append(n); l_sgl_r.append(nr)
            stride_strs = []
            for s in strides:
                v = loss_stride(model, tok, p, a, args.residual_length, s)
                l_strides[s].append(v)
                stride_strs.append(f"str{s}={v:.4f}(Δ{v-g:+.4f})")
            print(f"  sample {i}: gen={g:.4f}  single0={n:.4f}(Δ{n-g:+.4f})  singleR={nr:.4f}(Δ{nr-g:+.4f})  "
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
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"\nSaved raw results -> {out_json}")


if __name__ == "__main__":
    main()
