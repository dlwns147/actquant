"""Prefill-then-answer-stride proxy.

Idea: prefill prompt once (use_cache=True, R=128, KIVI on), then push the answer
in chunks of `stride` answer-tokens. With prompt_len % R == 0 the residual is
empty after prefill, and the answer alone fills the residual:
  - stride=1   →   _update_generation per token   →  identical to L_gen.
  - stride=2..R →  _update_stride per chunk.       Bias appears only for the
                   last chunk that crosses the R boundary (combined % R == 0).
                   Number of biased queries ≈ stride (so bias ≈ stride/A).
  - stride>R   →   first chunk already packs prefix; ≥R queries biased.

Computes both CE and JSD against fp16 reference. Reuses dump from 01_dump_fp16.py.

Outputs:
  data/prefill_stride_ce.json
  data/prefill_stride_jsd.json
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from model.replace import replace_kv_cache  # noqa: E402
from utils.loss import JSD  # noqa: E402


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


@torch.no_grad()
def logits_prefill_stride(model, prompt_ids, answer_ids, residual_length, stride):
    """Prefill prompt, then push answer in chunks of `stride` answer-tokens.

    The first answer chunk also needs to predict answer[0]. We use the prompt's
    last logit as that prediction (i.e. at the prefill-end position) and shift
    the rest from each chunk's outputs. This matches what L_gen / L_stride do
    for the answer span.
    """
    model.config.kivi_config.residual_length = residual_length
    model.config.quant_kv_output = False
    model.config.use_cache = True

    A = answer_ids.shape[1]
    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    last_prompt_logit = out.logits[:, -1:, :]   # predicts answer[0]

    chunk_outs = []
    for s in range(0, A, stride):
        e = min(s + stride, A)
        ck = model(answer_ids[:, s:e], past_key_values=past, use_cache=True)
        past = ck.past_key_values
        chunk_outs.append(ck.logits)
    answer_logits = torch.cat(chunk_outs, dim=1)  # [1, A, V]
    # We want logits at positions predicting answer[0..A-1]:
    # answer_logits[:, t, :] predicts answer[t+1]; last_prompt_logit predicts answer[0]
    pred_logits = torch.cat([last_prompt_logit, answer_logits[:, :-1, :]], dim=1)  # [1, A, V]
    return pred_logits.squeeze(0)  # [A, V]


def ce_loss(logits, labels):
    return F.cross_entropy(logits, labels).item()


def jsd_loss(quant_logits, fp16_logits):
    return JSD()(quant_logits, fp16_logits).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--residual_length", type=int, default=128)
    ap.add_argument("--strides", type=str, default="2,4,8,16,32,64,128",
                    help="answer-only stride values to test")
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
        fp = fp16_dir / f"seq{args.prompt_len}_a{args.answer_len}_sample{i}.pt"
        assert fp.exists(), f"Missing fp16 logits: {fp}"
        fp16_logits.append(torch.load(fp).cuda())

    out_ce = HERE / "data" / "prefill_stride_ce.json"
    out_jsd = HERE / "data" / "prefill_stride_jsd.json"

    def load(p):
        if p.exists():
            try: return json.load(open(p))
            except Exception: pass
        return []

    rows_ce, rows_jsd = load(out_ce), load(out_jsd)
    done = {(r["cfg"], r["prompt_len"], r.get("answer_len", 128)) for r in rows_ce}

    for cfg in args.configs.split(","):
        if (cfg, args.prompt_len, args.answer_len) in done:
            print(f"[skip] {cfg} @ p={args.prompt_len} a={args.answer_len}", flush=True); continue
        kb, vb = [int(x) for x in cfg.split(":")]
        print(f"\n=== K{kb}V{vb} seq={args.prompt_len} strides={strides} ===")
        model, _ = build(args.model_path, kb, vb, args.kgs, args.vgs, args.residual_length)

        ce_strides = {s: [] for s in strides}
        jsd_strides = {s: [] for s in strides}
        for i, (p, a) in enumerate(samples):
            fp = fp16_logits[i]
            labels = a.squeeze(0)
            ss = []
            for st in strides:
                lg = logits_prefill_stride(model, p, a, args.residual_length, st)
                ce = ce_loss(lg, labels); js = jsd_loss(lg, fp)
                ce_strides[st].append(ce); jsd_strides[st].append(js)
                ss.append(f"s{st}: ce={ce:.4f} jsd={js:.5f}")
            print(f"  sample {i}: " + "  ".join(ss), flush=True)

        rows_ce.append({"cfg": cfg, "prompt_len": args.prompt_len,
                        "answer_len": args.answer_len,
                        "strides": {str(s): ce_strides[s] for s in strides}})
        rows_jsd.append({"cfg": cfg, "prompt_len": args.prompt_len,
                         "answer_len": args.answer_len,
                         "strides": {str(s): jsd_strides[s] for s in strides}})
        with open(out_ce, "w") as f: json.dump(rows_ce, f, indent=2)
        with open(out_jsd, "w") as f: json.dump(rows_jsd, f, indent=2)
        del model
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    print(f"\nSaved -> {out_ce}\n         {out_jsd}")


if __name__ == "__main__":
    main()
