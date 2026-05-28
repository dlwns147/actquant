"""End-to-end FP vs. AWQ wikitext2 PPL comparison in one process.

Avoids the vLLM model load (which OOM'd with other tenants on the GPU) and
the EXIT-trap deletion of the temp save dir — everything happens in memory.

Usage (run from amq/ working dir, like amq_quantization_gen.py):

    cd /NAS/SJ/actquant/amq_instruct_pure
    CUDA_VISIBLE_DEVICES=3 python amq/amq_awq_ppl.py \
        --model_path /SSD/huggingface/google \
        --model_name gemma-3-12b-it \
        --config amq/configs/gemma.json \
        --load .../iter_200.stats \
        --target_bits 4.01 --target_bits_offset 0.005 \
        --group_size 128 --seed 0 --seqlen 2048
"""

from __future__ import annotations

import argparse
import gc
import json

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from utils.func import get_hfmodel, get_bits_usage, clean_up, build_device_map, set_seed
from utils.data import get_wikitext2
from quantization.awq import AWQ
from quantization.gptq import GPTQ
from post_search import load_archive, select_near_target_by_asf


@torch.no_grad()
def eval_wikitext2_ppl(model, tokenizer, seqlen=2048, device='cuda', ce_chunk=256):
    """CE-based wikitext2 PPL.

    Chunks the seq dim so we never materialize a full [S, vocab] fp32 logits
    tensor — Gemma-3 has vocab=262144 so that would be ~2 GiB per fp32 cast
    and OOMs on a shared GPU.
    """
    model.eval()
    loader = get_wikitext2(tokenizer, seqlen=seqlen, batch_size=1)
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        logits = out.logits if hasattr(out, 'logits') else out[0]  # [1, S, V] bf16/fp16
        labels = batch[:, 1:].contiguous().view(-1)
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        # chunked CE in fp32
        n = labels.numel()
        for s in range(0, n, ce_chunk):
            lc = logits[s:s + ce_chunk].float()
            lab = labels[s:s + ce_chunk]
            loss = torch.nn.functional.cross_entropy(lc, lab, reduction='sum')
            total_nll += loss.item()
            total_tokens += lab.numel()
        del logits, labels, out
    import math
    return math.exp(total_nll / total_tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--model_name', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--load', default=None,
                   help='search archive .stats file; required unless --uniform_bits is set')
    p.add_argument('--target_bits', type=float, default=4.0,
                   help='ignored when --uniform_bits is set')
    p.add_argument('--target_bits_offset', type=float, default=0.005)
    p.add_argument('--group_size', type=int, default=128)
    p.add_argument('--num_of_candidates', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--uniform_bits', type=int, default=None,
                   help='if set, skip arch search and use this bit-width on every '
                        'linear of every block (vanilla AWQ baseline).')
    p.add_argument('--calib_nsamples', type=int, default=128,
                   help='calibration samples (default 128).')
    p.add_argument('--calib_seqlen', type=int, default=512,
                   help='calibration sequence length (default 512).')
    p.add_argument('--method', choices=['awq', 'gptq'], default='awq')
    p.add_argument('--gptq_act_order', action='store_true',
                   help='GPTQ activation reordering')
    p.add_argument('--gptq_percdamp', type=float, default=0.01,
                   help='GPTQ damping fraction (RedHatAI uses 0.07 for Gemma 3)')
    p.add_argument('--gptq_true_sequential', action='store_true',
                   help='GPTQ sequential quantization within block')
    p.add_argument('--gpu_id', default='0',
                   help='used to build the AWQ device_map; CUDA_VISIBLE_DEVICES'
                        ' already remaps the physical GPU.')
    args = p.parse_args()

    set_seed(args.seed)

    with open(args.config) as f:
        cfg = json.load(f)[args.model_name]

    # arch selection (or uniform-bit override)
    if args.uniform_bits is not None:
        n_block = cfg['n_block']
        linear_names = cfg['linear']
        arch = {'linear': {ln: [args.uniform_bits] * n_block for ln in linear_names}}
        sel_bits_val = float(args.uniform_bits)
        print(f"[select] uniform {args.uniform_bits}-bit across {n_block} blocks, "
              f"{len(linear_names)} linears")
    else:
        assert args.load is not None, "--load is required unless --uniform_bits is set"
        archive = load_archive(args.load)
        sel_idx, sel_archs, sel_jsd, sel_bits = select_near_target_by_asf(
            archive,
            target_bits=args.target_bits,
            target_bits_offset=args.target_bits_offset,
            num_of_candidates=args.num_of_candidates,
            owq_offset=0.0,
        )
        arch = sel_archs[0]
        sel_bits_val = float(sel_bits[0])
        print(f"[select] idx={int(sel_idx[0])} bits={sel_bits_val:.4f}")

    model_id = f"{args.model_path}/{args.model_name}"
    bits_usage = get_bits_usage(arch, cfg, args.group_size)

    # device map (single GPU here, but builds the layer→device dict awq expects)
    gpu_ids = [x.strip() for x in args.gpu_id.split(',')]
    device_map = build_device_map(gpu_ids, cfg)

    # --- FP baseline ---
    load_dtype = 'auto' if 'gemma-3' in args.model_name.lower() else 'float16'
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    print(f"[fp] load {model_id} dtype={load_dtype}")
    fp_model = get_hfmodel(model_id, dtype=load_dtype, device_map='cuda')
    fp_model.eval()
    print("[fp] eval wikitext2 ppl ...")
    ppl_fp = eval_wikitext2_ppl(fp_model, tokenizer, seqlen=args.seqlen)
    print(f"[fp] PPL = {ppl_fp:.4f}")

    # AWQ wants to start from a CPU-loaded fresh model (matching awq_save's
    # flow); the FP we just evaluated lives on cuda, so swap it out cleanly.
    del fp_model
    clean_up()
    base = get_hfmodel(model_id, dtype=load_dtype, device_map='cpu')

    method_tag = args.method
    print(f"[{method_tag}] run on arch bits_usage={bits_usage:.4f} "
          f"calib(nsamples={args.calib_nsamples}, seqlen={args.calib_seqlen})")
    if args.method == 'awq':
        quantizer = AWQ(
            model=base, tokenizer=tokenizer, method='awq',
            arch=arch, avg_bits=bits_usage, group_size=args.group_size,
            config=cfg, dev='cuda', device_map=device_map,
        )
        quantizer.run(nsamples=args.calib_nsamples, seqlen=args.calib_seqlen)
    else:  # gptq
        quantizer = GPTQ(
            model=base, tokenizer=tokenizer, method='gptq',
            arch=arch, avg_bits=bits_usage, group_size=args.group_size,
            config=cfg, dev='cuda',
        )
        quantizer.run(
            nsamples=args.calib_nsamples, seqlen=args.calib_seqlen,
            act_order=args.gptq_act_order, percdamp=args.gptq_percdamp,
            true_sequential=args.gptq_true_sequential,
        )
    clean_up()
    quantized = quantizer.model
    quantized = quantized.to('cuda').eval()
    print(f"[{method_tag}] eval wikitext2 ppl ...")
    ppl_awq = eval_wikitext2_ppl(quantized, tokenizer, seqlen=args.seqlen)
    print(f"[{method_tag}] PPL = {ppl_awq:.4f}")

    delta = ppl_awq - ppl_fp
    rel = (delta / ppl_fp) * 100.0
    print()
    print(f"=== PPL summary [{method_tag.upper()}] (wikitext2 test, seqlen={args.seqlen}) ===")
    print(f"model    : {args.model_name}")
    if args.uniform_bits is not None:
        print(f"arch_bits: {sel_bits_val:.4f}  (uniform {args.uniform_bits}-bit)")
    else:
        print(f"arch_bits: {sel_bits_val:.4f}  (target {args.target_bits} ±{args.target_bits_offset})")
    print(f"FP   PPL : {ppl_fp:.4f}")
    print(f"{method_tag.upper():4s} PPL : {ppl_awq:.4f}")
    print(f"Delta    : {delta:+.4f}  ({rel:+.2f}%)")


if __name__ == '__main__':
    main()
