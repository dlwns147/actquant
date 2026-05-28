"""
Quick fp16-vs-bf16 KV-cache comparison driver.

Loads a *single uniform* arch (per-layer 3-bit HQQ weight + 2-bit gs=32
KV via KIVI packed) and runs both:
  • PPL on wikitext2 (seqlen=2048, n_sample=20)
  • RULER niah_multikey_3 accuracy (50 samples, length=16384)

Call twice — once with `--dtype float16` (and the float16 HQQ dir) and once
with `--dtype bfloat16` (and the bfloat16 HQQ dir) — to compare. Both runs
use the exact same architecture; only the activation/KV dtype changes.

Designed as a thin shim on top of LlamaEvaluator + utils.ruler.eval_ruler.
"""
import argparse
import json
import os
import time

import torch

from evaluator import LlamaEvaluator
from utils.data import get_tokenizer
from utils.func import init_accelerator
from utils.ruler import eval_ruler


def build_uniform_arch(config, n_block, w_bits, k_bits, k_gs, v_bits, v_gs):
    """Uniform per-layer arch: same bits/gs across all blocks/projections."""
    q_arch = {
        'w': {linear: [w_bits] * n_block for linear in config['linear']},
        'k': [[k_bits, k_gs]] * n_block,
        'v': [[v_bits, v_gs]] * n_block,
    }
    return {'q': q_arch, 'p': {'k': [0] * n_block, 'v': [0] * n_block}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='/SSD/huggingface/meta-llama')
    p.add_argument('--model_name', default='Llama-3.1-8B-Instruct')
    p.add_argument('--config', default='config/llama.json')
    p.add_argument('--dtype', choices=['float16', 'bfloat16'], required=True)
    p.add_argument('--quant_model_path', default='',
                   help='Pre-quantised HQQ dir; leave empty with --no_quant '
                        'to evaluate the unquantised HF model.')
    p.add_argument('--no_quant', action='store_true',
                   help='Skip weight + KV quantisation; evaluate the dense HF '
                        'model at --dtype as a reference baseline.')
    p.add_argument('--w_bits', type=int, default=3)
    p.add_argument('--k_bits', type=int, default=2)
    p.add_argument('--k_gs', type=int, default=32)
    p.add_argument('--v_bits', type=int, default=2)
    p.add_argument('--v_gs', type=int, default=32)
    p.add_argument('--residual_length', type=int, default=128)
    p.add_argument('--n_sample_ppl', type=int, default=20)
    p.add_argument('--seqlen_ppl', type=int, default=2048)
    p.add_argument('--stride_ppl', type=int, default=512,
                   help='Sliding-window stride for PPL (use_cache=True path).')
    p.add_argument('--ruler_length', type=int, default=16384)
    p.add_argument('--ruler_sample', type=int, default=50)
    p.add_argument('--ruler_task', default='niah_multikey_3')
    p.add_argument('--result_path', required=True, help='output JSON path')
    p.add_argument('--skip_ppl', action='store_true')
    p.add_argument('--skip_ruler', action='store_true')
    p.add_argument('--gpu_id', default='0',
                   help='comma-separated gpu ids forwarded to init_accelerator '
                        '(used to build a real device_map dict).')
    args = p.parse_args()

    with open(args.config) as f:
        config = json.load(f)[args.model_name]
    n_block = int(config['n_block'])

    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]

    model_id = os.path.join(args.model_path, args.model_name)
    accelerator, device_map = init_accelerator(args.gpu_id, config)

    if args.no_quant:
        # Dense reference: fp16/bf16 HF model, no weight quant, no KV quant.
        # bits['k']/['v'] = 16 → evaluator skips replace_kv_cache().
        method = {'w': 'fp16', 'kv': []}
        bits = {'w': [], 'k': [16], 'v': [16]}
        group_size = {'w': 128, 'k': [args.k_gs], 'v': [args.v_gs]}
        quant_paths = []
        label = f"DENSE dtype={args.dtype}"
    else:
        assert args.quant_model_path, "--quant_model_path is required unless --no_quant"
        method = {'w': 'hqq', 'kv': ['kivi']}
        bits = {'w': [args.w_bits], 'k': [args.k_bits], 'v': [args.v_bits]}
        group_size = {'w': 128, 'k': [args.k_gs], 'v': [args.v_gs]}
        quant_paths = [args.quant_model_path]
        label = (f"dtype={args.dtype} | w={args.w_bits}b "
                 f"kvs={args.k_bits}b/gs={args.k_gs}")

    print(f"\n=== compare_dtype_eval: {args.model_name} | {label} ===")
    print(f"  quant_model_path = {args.quant_model_path or '(none / dense)'}")

    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        method=method,
        model_id=model_id,
        quant_model_paths=quant_paths,
        datasets=['wikitext2'],
        data_batch_size=1,
        seed=0,
        seqlen=args.seqlen_ppl,
        n_sample=args.n_sample_ppl,
        device_map=device_map,
        dtype=dtype,
        loss_func='cross_entropy',
        bits=bits,
        group_size=group_size,
        residual_length=args.residual_length,
        k_quant_scheme='channel',
        v_quant_scheme='token',
        packing=True,
    )

    arch = build_uniform_arch(config, n_block,
                              args.w_bits, args.k_bits, args.k_gs,
                              args.v_bits, args.v_gs)

    out = {
        'dtype': args.dtype,
        'model_name': args.model_name,
        'no_quant': args.no_quant,
        'quant_model_path': args.quant_model_path,
        'w_bits': None if args.no_quant else args.w_bits,
        'k_bits': None if args.no_quant else args.k_bits,
        'k_gs': None if args.no_quant else args.k_gs,
        'v_bits': None if args.no_quant else args.v_bits,
        'v_gs': None if args.no_quant else args.v_gs,
        'residual_length': args.residual_length,
    }

    def _enable_cache(model):
        # Dense fp16/bf16 model has no kivi_config; only flip use_cache.
        if hasattr(model.config, 'kivi_config'):
            model.config.kivi_config.residual_length = args.residual_length
        model.config.quant_kv_output = False
        model.config.use_cache = True

    # ---- PPL ----
    if not args.skip_ppl:
        # stride=512 path: needs use_cache=True (sliding-window with past_kv).
        _enable_cache(evaluator.model)
        print(f"\n[PPL] wikitext2 (seqlen={args.seqlen_ppl}, "
              f"stride={args.stride_ppl}, n_sample={args.n_sample_ppl}) ...")
        t0 = time.time()
        metric_dict, _complexity = evaluator.eval(
            accelerator=accelerator, arch=arch, metric='ppl',
            loss_func='cross_entropy', stride=args.stride_ppl,
        )
        out['ppl_wikitext2'] = float(metric_dict['wikitext2'])
        out['ppl_time_sec'] = time.time() - t0
        print(f"  PPL wikitext2 = {out['ppl_wikitext2']:.4f}  "
              f"(took {out['ppl_time_sec']:.1f}s)")

    # ---- RULER ----
    if not args.skip_ruler:
        # If PPL was just run, evaluator.model is already the sampled mixed
        # model — just re-enable the KV cache. Else sample now.
        if evaluator.model is None or not args.skip_ppl:
            # PPL path leaves the model already sampled; only need to flip
            # cache flags.
            model = evaluator.model
        else:
            model = evaluator.sample(arch)
        _enable_cache(model)
        ruler_file = os.path.splitext(args.result_path)[0] + '_ruler.json'
        print(f"\n[RULER] {args.ruler_task}  length={args.ruler_length}  "
              f"nsample={args.ruler_sample} ...")
        t0 = time.time()
        eval_ruler(
            model=model,
            tokenizer=get_tokenizer(model_id),
            model_id=model_id,
            tasks=[args.ruler_task],
            yaml_path='utils/ruler_utils',
            batch_size=1,
            length=[args.ruler_length],
            nsample=args.ruler_sample,
            gen_toks=128,
            result_path=ruler_file,
            seed=0,
        )
        out['ruler_time_sec'] = time.time() - t0
        if os.path.exists(ruler_file):
            with open(ruler_file) as f:
                ruler_data = json.load(f)
            out['ruler_score'] = ruler_data.get(args.ruler_task, None)
            out['ruler_raw'] = ruler_data
        print(f"  RULER {args.ruler_task} = {out.get('ruler_score', 'N/A')}  "
              f"(took {out['ruler_time_sec']:.1f}s)")

    # Write JSON
    os.makedirs(os.path.dirname(args.result_path) or '.', exist_ok=True)
    with open(args.result_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.result_path}")


if __name__ == '__main__':
    main()
