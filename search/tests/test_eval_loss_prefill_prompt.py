"""Smoke test: eval_loss(prefill_prompt=True, stride=S, last_tokens=A)
should match a standalone prefill+answer-stride implementation.

Builds a single quant-config model, constructs a tiny Dataset+DataLoader from
WikiText, runs eval_loss in 3 modes and compares to the kv_proxy_study
reference numbers.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.replace import replace_kv_cache
from utils.eval import eval_loss


def make_loader(tok, prompt_len, answer_len, n_samples=1):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encs = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    seg = prompt_len + answer_len
    inputs, masks, labels = [], [], []
    for i in range(n_samples):
        s = i * seg
        ids = encs[s:s+seg]
        inputs.append(ids)
        masks.append(torch.ones_like(ids))
        labels.append(ids.clone())
    inputs = torch.stack(inputs).cuda()
    masks = torch.stack(masks).cuda()
    labels = torch.stack(labels).cuda()
    ds = TensorDataset(inputs, masks, labels)
    return DataLoader(ds, batch_size=1)


def build(model_path, kbits, vbits, R=128, kgs=128, vgs=128):
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


def main():
    model_path = "/SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct"
    R, A, P = 128, 128, 2048
    accelerator = Accelerator()

    m, tok = build(model_path, kbits=2, vbits=2, R=R)
    loader = make_loader(tok, prompt_len=P, answer_len=A, n_samples=1)

    # Mode A: full-seq stride=128 (legacy)
    m.config.kivi_config.residual_length = R
    m.config.quant_kv_output = False
    m.config.use_cache = True
    L_full_str = eval_loss(m, accelerator, loader, seqlen=P+A,
                           loss_func='cross_entropy',
                           stride=128, last_tokens=A, prefill_prompt=False)
    print(f"L_full_seq_stride128       = {L_full_str:.4f}")

    # Mode B: prefill_prompt + stride=2 (≈ L_gen)
    L_pp2 = eval_loss(m, accelerator, loader, seqlen=P+A,
                      loss_func='cross_entropy',
                      stride=2, last_tokens=A, prefill_prompt=True)
    print(f"L_prefill_prompt_stride2   = {L_pp2:.4f}")

    # Mode C: prefill_prompt + stride=128 (1 chunk for answer)
    L_pp128 = eval_loss(m, accelerator, loader, seqlen=P+A,
                        loss_func='cross_entropy',
                        stride=128, last_tokens=A, prefill_prompt=True)
    print(f"L_prefill_prompt_stride128 = {L_pp128:.4f}")

    # Mode D: prefill_prompt + stride=0 (defaults to one chunk of A)
    L_pp0 = eval_loss(m, accelerator, loader, seqlen=P+A,
                      loss_func='cross_entropy',
                      stride=0, last_tokens=A, prefill_prompt=True)
    print(f"L_prefill_prompt_stride0   = {L_pp0:.4f}  (should match stride128)")

    # Reference: kv_proxy_study reported sample 0 K2V2 seq=2048 → L_gen=2.2734
    print("\nReference (kv_proxy_study, sample 0 K2V2 seq=2k):")
    print("  L_gen (real decode)        = 2.2734")
    print("  full-seq stride=128        = 2.4922")
    print("  prefill+stride=2           = 2.2734")
    print("  prefill+stride=128         = 2.3594")


if __name__ == "__main__":
    main()
