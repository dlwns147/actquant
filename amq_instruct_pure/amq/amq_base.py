"""Pipeline — uniform-bit AWQ baselines (+ fp16) + lm_eval.

For each (w_bit, group_size) pair, build a uniform arch (every block,
every linear projection = same bit), AWQ-quantize, save, and run
lm_eval. ``w_bit=16`` is a special case: skip AWQ entirely and run
lm_eval directly against the original HF model (fp16/bf16 reference).
Reuses ``post_search.awq_quantize_and_save`` and
``post_search.run_lm_eval`` so the search archive is not needed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from utils.func import build_device_map, clean_up, set_seed

from post_search import awq_quantize_and_save, run_lm_eval


def make_uniform_arch(config, bit):
    n_block = int(config['n_block'])
    return {
        'linear': {ln: [int(bit)] * n_block for ln in config['linear']},
    }


def parse_args():
    p = argparse.ArgumentParser(
        description='Uniform-bit AWQ baselines (e.g. w3gs128, w3gs-1, w4gs-1) + lm_eval',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--gpu_id', type=str, default='0')
    p.add_argument('--save_path', type=str, required=True,
                   help='Where to write the combined JSON + per-baseline lm_eval outputs')
    p.add_argument('--model_save_path', type=str, required=True,
                   help='Parent dir for ephemeral AWQ checkpoints (one subdir per baseline)')
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--model_name', type=str, required=True)
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--method', type=str, default='awq', choices=['awq'])
    p.add_argument('--w_bit', type=int, nargs='+', required=True,
                   help='Per-baseline weight bits, e.g. "16 3 3 4" (16 = fp16/bf16, no quant)')
    p.add_argument('--group_size', type=int, nargs='+', required=True,
                   help='Per-baseline group sizes (-1 = per-channel; ignored when w_bit=16)')
    p.add_argument('--lm_eval_task', type=str, nargs='+',
                   default=['gsm8k_cot', 'ifeval', 'mbpp'])
    p.add_argument('--lm_eval_batch_size', type=str, default='auto')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--keep_quantized', action='store_true',
                   help='Keep each baseline AWQ checkpoint on disk after lm_eval (default: delete)')
    return p.parse_args()


def run(args, config):
    set_seed(args.seed)

    if len(args.w_bit) != len(args.group_size):
        raise ValueError(
            f'--w_bit ({args.w_bit}) and --group_size ({args.group_size}) must be same length'
        )

    print('=== Uniform-bit AWQ baselines + lm_eval ===')
    print(args)

    gpu_ids = args.gpu_id if isinstance(args.gpu_id, list) else \
        [x.strip() for x in args.gpu_id.split(',')]
    device_map = build_device_map(gpu_ids, config)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.model_save_path, exist_ok=True)

    # awq_quantize_and_save reads args.group_size + args.method off the
    # Namespace — stash the originals so the per-baseline mutation below
    # doesn't corrupt the final JSON dump.
    orig_w_bits = list(args.w_bit)
    orig_group_sizes = list(args.group_size)

    results = []
    for i, (w, gs) in enumerate(zip(orig_w_bits, orig_group_sizes)):
        is_fp16 = (w >= 16)
        if is_fp16:
            tag = 'fp16'
            eval_model_path = os.path.join(args.model_path, args.model_name)
            arch = None
            ckpt_dir = None
        else:
            tag = f'awq_w{w}_gs{gs}'
            arch = make_uniform_arch(config, w)

            # awq_quantize_and_save reads args.group_size + args.method off
            # the Namespace — set them per baseline.
            args.group_size = gs
            args.method = 'awq'

            ckpt_dir = os.path.join(args.model_save_path, tag)
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir)
            eval_model_path = ckpt_dir

        print(f'\n[{i + 1}/{len(orig_w_bits)}] === {tag} ===')

        if not is_fp16:
            awq_quantize_and_save(args, config, arch, ckpt_dir, device_map)
            clean_up()

        out_dir = os.path.join(args.save_path, tag)
        os.makedirs(out_dir, exist_ok=True)
        print(f'[{i + 1}/{len(orig_w_bits)}] lm_eval ({args.lm_eval_task}) -> {out_dir}')
        run_lm_eval(
            model_path=eval_model_path,
            task=args.lm_eval_task,
            batch_size=args.lm_eval_batch_size,
            device='cuda',
            output_path=out_dir,
        )

        results.append({
            'tag': tag,
            'w_bit': w,
            'group_size': gs,
            'arch': arch,
            'eval_model_path': eval_model_path,
            'lm_eval_task': args.lm_eval_task,
            'lm_eval_output_path': out_dir,
        })

        if not is_fp16 and not args.keep_quantized:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        clean_up()

    # Restore the original lists so the JSON dump reflects what was requested.
    args.w_bit = orig_w_bits
    args.group_size = orig_group_sizes
    args.method = 'awq'

    out_json = os.path.join(args.save_path, 'amq_base.json')
    with open(out_json, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f,
                  ensure_ascii=False, indent=4, default=str)
    print(f'\nsaved -> {out_json}')

    clean_up()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    run(args, config)


if __name__ == '__main__':
    main()
