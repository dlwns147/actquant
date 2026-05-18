"""Smoke test: KIVI + ThinK KV-cache on google/gemma-3-12b-it (text decoder).

Verifies model/gemma3_kivi.py wiring end-to-end at seqlen 2048:
  * BF16 baseline ppl (single pass) + greedy generation
  * KIVI / KIVI+ThinK loss path   : single forward, quant_kv_output=True
  * KIVI / KIVI+ThinK cache path  : strided (stride=512) forward through the
                                    KIVIFakeCache (use_cache=True) — exercises
                                    prefill + _update_stride + lazy_update
  * KIVI / KIVI+ThinK gen path    : model.generate via KIVIFakeCache

ppl uses the same strided loop as utils/eval.py:eval_ppl.

Run:
  CUDA_VISIBLE_DEVICES=0 python tests/smoke_kivi_think_gemma3.py
"""
import sys, gc, torch
import torch.nn as nn
sys.path.insert(0, "/NAS/SJ/actquant/search")

from transformers import AutoConfig, AutoTokenizer, Gemma3ForCausalLM
from model.replace import replace_kv_cache

MODEL = "/SSD/huggingface/google/gemma-3-12b-it"
N_BLOCK = 48
DTYPE = torch.bfloat16
SEQLEN = 2048
N_SEG = 4

TOK = AutoTokenizer.from_pretrained(MODEL)


def load_fp16():
    cfg = AutoConfig.from_pretrained(MODEL)
    m = Gemma3ForCausalLM.from_pretrained(
        MODEL, config=cfg.text_config, torch_dtype=DTYPE,
        device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    m.eval()
    return m


def _enc():
    from datasets import load_dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return TOK("\n\n".join(data["text"]), return_tensors="pt").input_ids


@torch.no_grad()
def ppl(model, enc, seqlen=SEQLEN, n_seg=N_SEG, stride=0):
    """Mirror utils/eval.py:eval_ppl — single pass (stride=0) or strided
    forward carrying past_key_values (stride>0) to exercise the KIVI cache."""
    dev = next(model.parameters()).device
    nlls, n = [], min(n_seg, enc.shape[1] // seqlen)
    for i in range(n):
        inputs = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
        if stride and stride > 0:
            chunked, pkv = [], None
            for s in range(0, seqlen, stride):
                e = min(s + stride, seqlen)
                out = model(inputs[:, s:e], past_key_values=pkv, use_cache=True)
                chunked.append(out.logits)
                pkv = out.past_key_values
            logits = torch.cat(chunked, dim=1)
        else:
            logits = model(inputs).logits
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = inputs[:, 1:].reshape(-1)
        loss = nn.CrossEntropyLoss()(shift_logits.float(), shift_labels)
        nlls.append(loss.float() * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n * seqlen)).item()


@torch.no_grad()
def sample_gen(model):
    msgs = [{"role": "user", "content": "In one sentence, what is the capital of France?"}]
    ids = TOK.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt").to(next(model.parameters()).device)
    attn = torch.ones_like(ids)
    out = model.generate(ids, attention_mask=attn, max_new_tokens=32,
                         do_sample=False)
    return TOK.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def set_kivi_arch(model, kbits, vbits, kgs, vgs, residual_length=128,
                   k_prune=0, v_prune=0):
    c = model.config.kivi_config
    c.k_bits = [kbits] * N_BLOCK
    c.v_bits = [vbits] * N_BLOCK
    c.k_group_size = [kgs] * N_BLOCK
    c.v_group_size = [vgs] * N_BLOCK
    c.residual_length = residual_length
    c.k_pruning_dim = [k_prune] * N_BLOCK
    c.v_pruning_dim = [v_prune] * N_BLOCK


def run_case(tag, enc, methods, kbits=2, vbits=2, k_prune=0, v_prune=0, base=None):
    print(f"\n=== {tag} ===", flush=True)
    m = load_fp16()
    m = replace_kv_cache(model=m, tokenizer=TOK, method=methods,
                         n_block=N_BLOCK, k_quant_scheme="channel",
                         v_quant_scheme="token", residual_length=128,
                         packing=False, quant_kv_output=False,
                         k_pruning_dim=k_prune, v_pruning_dim=v_prune)
    set_kivi_arch(m, kbits=kbits, vbits=vbits, kgs=128, vgs=128,
                  residual_length=128, k_prune=k_prune, v_prune=v_prune)

    def _r(p):
        return f" (ratio {p / base:.3f})" if base else ""

    # loss path: single forward, no cache, fake-quant KV output
    m.config.use_cache = False
    m.config.quant_kv_output = True
    p_loss = ppl(m, enc, stride=0)
    print(f"[{tag}] loss-path  (1-pass)        ppl = {p_loss:.4f}{_r(p_loss)}", flush=True)

    # cache path: strided forward through KIVIFakeCache
    m.config.use_cache = True
    m.config.quant_kv_output = False
    p_str = ppl(m, enc, stride=512)
    print(f"[{tag}] cache-path (stride=512)    ppl = {p_str:.4f}{_r(p_str)}", flush=True)

    g = sample_gen(m)
    print(f"[{tag}] gen-path   greedy = {g!r}", flush=True)

    del m
    gc.collect()
    torch.cuda.empty_cache()
    return p_loss


def main():
    enc = _enc()
    print(f"wikitext2 test tokens: {enc.shape[1]} | seqlen={SEQLEN} n_seg={N_SEG}")
    m = load_fp16()
    print(f"model class: {m.model.__class__.__name__}")
    base = ppl(m, enc, stride=0)
    print(f"[baseline BF16] wikitext2 ppl = {base:.4f}")
    print(f"[baseline BF16] gen: {sample_gen(m)!r}")
    del m
    gc.collect()
    torch.cuda.empty_cache()

    run_case("KIVI k4v4 gs128", enc, ["kivi"], kbits=4, vbits=4, base=base)
    run_case("KIVI k2v2 gs128", enc, ["kivi"], kbits=2, vbits=2, base=base)
    run_case("KIVI+ThinK k4v4 kp64", enc, ["kivi", "think"],
             kbits=4, vbits=4, k_prune=64, v_prune=0, base=base)
    run_case("KIVI+ThinK k4v4 kp64 vp128", enc, ["kivi", "think"],
             kbits=4, vbits=4, k_prune=64, v_prune=128, base=base)
    print("\nDONE")


if __name__ == "__main__":
    main()
