"""Compare wikitext2 perplexity between FP and AWQ-quantized model.

Run from the amq_instruct_pure repo root, e.g.

    cd /NAS/SJ/actquant/amq_instruct_pure
    CUDA_VISIBLE_DEVICES=2 python amq/eval_ppl_compare.py \
        --fp_path /SSD/huggingface/google/gemma-3-12b-it \
        --awq_path /tmp/amq_gen_XXXXXX/awq_model \
        --seqlen 2048
"""

from __future__ import annotations

import argparse
import gc
import math

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# repo-local imports (assumes CWD == amq/, like amq_quantization_gen.py)
from utils.func import get_hfmodel
from utils.data import get_wikitext2


@torch.no_grad()
def eval_wikitext2_ppl(model, tokenizer, seqlen=2048, device='cuda'):
    """Standard chunked-NLL wikitext2 test PPL (GPTQ-style)."""
    model.eval()
    loader = get_wikitext2(tokenizer, seqlen=seqlen, batch_size=1)
    loss_fn = nn.CrossEntropyLoss()

    nlls = []
    n_tokens = 0
    for batch in loader:
        batch = batch.to(device)  # [1, seqlen]
        out = model(batch)
        logits = out.logits if hasattr(out, 'logits') else out[0]
        # shift one
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * shift_labels.numel())
        n_tokens += shift_labels.numel()

    ppl = torch.exp(torch.stack(nlls).sum() / n_tokens)
    return ppl.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp_path', required=True)
    parser.add_argument('--awq_path', required=True)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--dtype', default='auto',
                        help="HF dtype string ('auto', 'bfloat16', 'float16'). "
                             "Gemma-3 needs bf16 (auto) — fp16 overflows.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.fp_path, use_fast=True)

    # --- FP baseline ---
    print(f"[fp] loading {args.fp_path} dtype={args.dtype}")
    fp = get_hfmodel(args.fp_path, dtype=args.dtype, device_map='cuda')
    print("[fp] eval wikitext2 PPL ...")
    ppl_fp = eval_wikitext2_ppl(fp, tokenizer, seqlen=args.seqlen)
    print(f"[fp] PPL = {ppl_fp:.4f}")
    del fp
    gc.collect()
    torch.cuda.empty_cache()

    # --- AWQ quantized ---
    print(f"[awq] loading {args.awq_path} dtype={args.dtype}")
    awq = get_hfmodel(args.awq_path, dtype=args.dtype, device_map='cuda')
    print("[awq] eval wikitext2 PPL ...")
    ppl_awq = eval_wikitext2_ppl(awq, tokenizer, seqlen=args.seqlen)
    print(f"[awq] PPL = {ppl_awq:.4f}")

    delta = ppl_awq - ppl_fp
    rel = delta / ppl_fp * 100.0
    print(f"\n=== PPL summary (wikitext2 test, seqlen={args.seqlen}) ===")
    print(f"FP   PPL : {ppl_fp:.4f}")
    print(f"AWQ  PPL : {ppl_awq:.4f}")
    print(f"Delta    : {delta:+.4f}  ({rel:+.2f}%)")


if __name__ == '__main__':
    main()
