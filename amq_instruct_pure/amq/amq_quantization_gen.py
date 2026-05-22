"""Pipeline #3 — ASF candidate selection + AWQ + lm_eval (no gen-eval).

Steps
-----
1. Load search archive.
2. ASF-select ``num_of_candidates`` archs with bits near ``target_bits``.
3. AWQ-quantize each selected arch, save to disk, run lm_eval.
4. Dump JSON.
"""

from __future__ import annotations

import os
import json

from utils.args import parse_args
from utils.func import build_device_map, set_seed, clean_up

from post_search import (
    load_archive,
    select_near_target_by_asf,
    awq_quantize_and_save,
    run_lm_eval,
)


def run(args, config):
    set_seed(args.seed)

    print("=== ASF candidate selection + AWQ + lm_eval ===")
    print(args)

    # 1-2. archive + ASF selection
    archive = load_archive(args.load)
    owq_offset = 0.1 if args.method == 'owq' else 0.0
    sel_idx, sel_archs, sel_jsd, sel_bits = select_near_target_by_asf(
        archive,
        target_bits=args.target_bits,
        target_bits_offset=args.target_bits_offset,
        num_of_candidates=args.num_of_candidates,
        owq_offset=owq_offset,
    )
    print(f"archive: {len(archive)}, selected: {len(sel_archs)}")

    # AWQ runs in this parent process; build the device_map for it
    gpu_ids = args.gpu_id if isinstance(args.gpu_id, list) else \
        [x.strip() for x in args.gpu_id.split(",")]
    device_map = build_device_map(gpu_ids, config)

    # 3. AWQ + lm_eval per selected arch
    os.makedirs(args.save_path, exist_ok=True)
    awq_root = os.path.join(args.save_path, 'awq')
    selected_records = []
    for i, arch in enumerate(sel_archs):
        tag = f"cand{i}_idx{int(sel_idx[i])}_bits{float(sel_bits[i]):.4f}"
        save_dir = os.path.join(awq_root, tag)
        print(f"[{i + 1}/{len(sel_archs)}] AWQ -> {save_dir}")
        awq_quantize_and_save(args, config, arch, args.model_save_path, device_map)
        clean_up()

        print(f"[{i + 1}/{len(sel_archs)}] lm_eval ({args.lm_eval_task})")
        lm_output_dir = os.path.join(save_dir, 'lm_eval')
        os.makedirs(lm_output_dir, exist_ok=True)
        run_lm_eval(
            model_path=args.model_save_path,
            task=args.lm_eval_task,
            batch_size=args.lm_eval_batch_size,
            device='cuda',
            output_path=lm_output_dir,
        )

        selected_records.append({
            'rank': i,
            'archive_idx': int(sel_idx[i]),
            'arch': arch,
            'bits_usage': float(sel_bits[i]),
            'mean_jsd': float(sel_jsd[i]),
            'awq_save_dir': save_dir,
            'lm_eval_task': args.lm_eval_task,
            'lm_eval_output_path': lm_output_dir,
        })

    # 4. persist
    out_payload = {
        'args': {k: v for k, v in vars(args).items()},
        'selected': selected_records,
    }
    out_json = os.path.join(args.save_path, 'quantization_gen.json')
    with open(out_json, 'w') as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=4, default=str)
    print(f"saved -> {out_json}")

    clean_up()


def main():
    args = parse_args(mode='quantization')

    if not os.path.exists(args.load):
        raise ValueError("Search results not found. Please run search first.")

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    run(args, config)


if __name__ == '__main__':
    main()
