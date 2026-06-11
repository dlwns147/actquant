"""sample_surrogate.py — stage 1 of the two-stage post-search.

Samples architectures from the per-axis search archives (--w_expr / --kv_expr /
--kvdim_expr / --eff_kv_expr) using quantile / random / coverage sampling,
evaluates each on the calibration set, and writes results.csv — the training
data for the surrogate fitted in post_search.py.

Sampling-only by design: prefer / high_tradeoff / top-n selection and all
downstream benchmarks live in post_search.py.

results.csv row layout (see post_search.load_sample_csv / analysis/v5):
    rows 0..n_comp-1   complexity (get_net_info keys)
    rows n_comp..       measured metric, one row per --datasets entry
    next row            combined predicted metric (pf column 0, args scales)
    next n_axes rows    per-axis search metric, order == expr_keys
"""
import os
import csv
import argparse
import warnings

import torch
from tqdm import tqdm

from evaluator import LlamaEvaluator
from utils.func import (init_run, build_expr_map, build_nd, evaluate_metric,
                        comp_key_order, get_net_info)
from utils.select import (
    LazyPs, build_arch, draw_random, assemble_F, select_valid_nd_idx,
    quantile_select, axis_of_map, coverage_subset_nsga2_extras, maximin_extras)

warnings.simplefilter("ignore")


def main(args):
    print(args)
    ctx = init_run(args)

    n_comp_obj = len(args.comp_obj)
    assert n_comp_obj == len(args.comp_obj_min) == len(args.comp_obj_max)

    expr_map = build_expr_map(args, ctx)
    nd = build_nd(args, ctx, expr_map)
    print(f"[sample_surrogate] expr_front={args.expr_front} → "
          f"{'lazy comp_obj-pruned (no NDS, full archives)' if getattr(nd, 'lazy', False) else 'dense'} "
          f"path  (n_total={nd.n_total:.3e})")
    expr_keys, _esm, _efm = nd.expr_keys, nd.esm, nd.efm

    # Parse --quantile_sample. Abort if a quantile metric's axis isn't searched:
    # it would collapse to a constant and silently undersample.
    quantile_specs = {}
    for spec in args.quantile_sample:
        k, v = spec.split('#')
        quantile_specs[k] = [float(q) for q in v.split(',')]
    if quantile_specs:
        _axis_map = axis_of_map(expr_keys)
        _expr_flag = {'w': '--w_expr', 'kv': '--kv_expr',
                      'kvdim': '--kvdim_expr', 'eff_kv': '--eff_kv_expr'}
        _missing = [(k, _axis_map.get(k)) for k in quantile_specs
                    if _axis_map.get(k) is not None
                    and _axis_map.get(k) not in expr_keys]
        if _missing:
            for k, ax in _missing:
                print(f"[quantile_sample] ERROR: metric '{k}' depends on axis "
                      f"'{ax}' but {_expr_flag.get(ax, ax)} was not provided; "
                      f"quantile would collapse to a constant.")
            raise SystemExit(1)

    valid_nd_idx = select_valid_nd_idx(
        nd.nd_shape, nd.new_metric_nd, nd.comp_nd_list,
        comp_obj_min=args.comp_obj_min, comp_obj_max=args.comp_obj_max,
        random_sample=args.random_sample,
        has_quantile=bool(quantile_specs), has_prefer=False)
    F = assemble_F(valid_nd_idx, expr_keys, _efm, nd.comp_nd_list,
                   nd.new_metric_nd)
    pf = F
    ps = LazyPs(lambda row: build_arch(ctx.default_arch, expr_keys, _esm, row),
                valid_nd_idx)

    # ─────────────────────── sampling (no prefer / top-n) ───────────────────────
    if quantile_specs:
        axis_cache = {}
        I_quant, metric_vals = quantile_select(
            quantile_specs, valid_nd_idx, expr_keys, _esm, ctx.default_arch,
            ctx.config, ctx.group_size, args.n_token, axis_cache=axis_cache,
            efm=_efm)
        print(f'[quantile_sample] selected {len(I_quant)} unique architectures '
              f'out of {len(valid_nd_idx)} candidates')
        for i in I_quant:
            print(f'  arch[{i}]: ' + str(
                {k: f'{metric_vals[k][i]:.4f}' for k in quantile_specs}))

        I_extra = []
        if args.random_sample is not None and args.random_sample > 0:
            if args.sampling_method == 'random':
                I_extra = draw_random(args.random_sample, len(valid_nd_idx),
                                      exclude=I_quant)
                print(f'[random_sample] adding {len(I_extra)} additional random '
                      f'samples (excluding {len(I_quant)} quantile-selected; '
                      f'pool={len(valid_nd_idx) - len(I_quant)})')
            elif args.sampling_method == 'maximin':
                # model-free farthest-point coverage on the per-axis metric
                # space (validated best global-representation sampler; works
                # with any surrogate incl. rbf-tps). M = F per-axis metric cols.
                _M = pf[:, [1 + 2 * i for i in range(len(expr_keys))]]
                I_extra = maximin_extras(_M, anchor_idx=list(I_quant),
                                         K=args.random_sample, seed=args.seed)
                print(f'[maximin] adding {len(I_extra)} farthest-point coverage '
                      f'samples (excluding {len(I_quant)} quantile-selected; '
                      f'pool={len(valid_nd_idx) - len(I_quant)})')
            else:
                fit_mode = args.sampling_method.replace('coverage_nsga2_', '')
                I_extra = coverage_subset_nsga2_extras(
                    valid_nd_idx, _efm, expr_keys, anchor_idx=I_quant,
                    K=args.random_sample, fitness=fit_mode,
                    coord=args.coverage_coord,
                    per_axis_agg=args.coverage_per_axis_agg,
                    pareto_select=args.coverage_pareto_select,
                    seed=args.seed, verbose=False)
                print(f'[coverage_nsga2 fitness={fit_mode} '
                      f'coord={args.coverage_coord} '
                      f'per_axis_agg={args.coverage_per_axis_agg} '
                      f'pareto_select={args.coverage_pareto_select}] adding '
                      f'{len(I_extra)} coverage-optimised samples (excluding '
                      f'{len(I_quant)} quantile-selected; '
                      f'pool={len(valid_nd_idx) - len(I_quant)})')
        I = sorted(set(I_quant) | set(I_extra))
        assert len(I) == len(I_quant) + len(I_extra), \
            'quantile and random samples must be disjoint'
        print(f'[total] {len(I)} architectures to evaluate '
              f'({len(I_quant)} quantile + {len(I_extra)} random)')
        sel_mode = 'quantile+random' if args.random_sample else 'quantile'
        # Describe the extras-drawing technique (the actual "sampling method").
        if not (args.random_sample and args.random_sample > 0):
            samp_desc = 'quantile-only (no extras)'
        elif args.sampling_method == 'random':
            samp_desc = 'random'
        elif args.sampling_method == 'maximin':
            samp_desc = 'maximin (model-free farthest-point coverage)'
        else:
            samp_desc = (f"{args.sampling_method} ("
                         f"coord={args.coverage_coord}, "
                         f"per_axis_agg={args.coverage_per_axis_agg}, "
                         f"pareto_select={args.coverage_pareto_select})")

    elif args.random_sample is not None:
        # select_valid_nd_idx already pre-sampled (or narrowed to the filter).
        I = list(range(len(valid_nd_idx)))
        sel_mode = 'random'
        samp_desc = 'random (no quantile anchors)'

    else:
        raise SystemExit(
            "sample_surrogate.py is sampling-only: provide --quantile_sample "
            "and/or --random_sample. prefer / high_tradeoff / top-n selection "
            "lives in post_search.py.")

    print(f'[sampling_method] {samp_desc}')
    print(f'[selection] mode={sel_mode}  method={samp_desc}  |I|={len(I)}  '
          f'|candidates|={len(valid_nd_idx)}  (n_total={nd.n_total})')

    # ─────────────────────── evaluate + write results.csv ───────────────────────
    model_id = f'{args.model_path}/{args.model_name}'
    if 'hqq' not in args.w_method:
        args.quant_model_paths = []
    evaluator = LlamaEvaluator(
        ctx.config, accelerator=ctx.accelerator, model_id=model_id,
        method={'w': args.w_method, 'kv': args.kv_method},
        quant_model_paths=args.quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=args.seqlen, min_seqlen=args.min_seqlen, n_sample=args.n_sample,
        datasets=args.datasets, device_map=ctx.device_map, dtype=ctx.dtype,
        bits={'w': args.w_bits, 'k': args.k_bits, 'v': args.v_bits},
        group_size=ctx.group_size, residual_length=args.residual_length,
        attn_sink=args.attn_sink,
        k_quant_scheme=args.k_quant_scheme, v_quant_scheme=args.v_quant_scheme,
        loss_func=args.loss_func, last_tokens=args.last_tokens,
        use_key_token=args.use_key_token, trunc_len=args.trunc_len,
        sliding_window=args.sliding_window, alpha=args.alpha, beta=args.beta,
        key_token_path=args.key_token_path)

    comp_save_list = [list() for _ in comp_key_order(ctx.config, ctx.group_size)]
    pf_metric_idx = [0] + [1 + 2 * i for i in range(len(expr_keys))]
    metric_save_list = [list() for _ in
                        range(len(args.datasets) + len(pf_metric_idx))]

    for idx in tqdm(I):
        arch = ps[idx]
        complexity = get_net_info(arch, ctx.config, ctx.group_size,
                                  n_token=args.n_token)
        print(f'complexity: {list(complexity.keys())}')
        print(f'complexity: {list(complexity.values())}')
        ctx.accelerator.print(f'arch: {arch}')
        model = evaluator.sample(arch)

        if not args.datasets:
            continue
        metric = evaluate_metric(args, arch, model, evaluator, ctx.accelerator)
        print(f'[{idx}] {args.metric}: {[p for p in metric.values()]}, '
              f'metric: {[pf[idx, 0]]}, '
              f'prev_metric: {pf[idx, pf_metric_idx[1:]].tolist()}')

        if args.save and args.results_csv_file:
            for c_i, c in enumerate(complexity.values()):
                comp_save_list[c_i].append(c)
            for m_i, m in enumerate(metric.values()):
                metric_save_list[m_i].append(m)
            for m_i, col in enumerate(pf_metric_idx):
                metric_save_list[m_i + len(args.datasets)].append(pf[idx, col])
            os.makedirs(args.save, exist_ok=True)
            with open(os.path.join(args.save, args.results_csv_file), 'w') as f:
                writer = csv.writer(f)
                for c in comp_save_list:
                    writer.writerow(c)
                for m in metric_save_list:
                    writer.writerow(m)

    print(args)


def build_parser():
    p = argparse.ArgumentParser(
        description='Stage 1: sample architectures + write surrogate '
                    'training results.csv (quantile / random / coverage only).')
    # model / config
    p.add_argument('--model_path', type=str, default='')
    p.add_argument('--model_name', type=str, default='')
    p.add_argument('--config', type=str, default='config/llama.json')
    p.add_argument('--dtype', type=str, default='auto',
                   choices=['float16', 'float', 'fp16', 'bfloat16', 'bfloat',
                            'bf16', 'auto'])
    p.add_argument('--gpu_id', type=str, default='0')
    p.add_argument('--seed', type=int, default=0)
    # quant methods / bits
    p.add_argument('--w_method', type=str, nargs='+', default=[],
                   choices=['fp16', 'awq', 'gptq', 'qeft', 'hqq'])
    p.add_argument('--kv_method', type=str, nargs='+', default=['kivi'],
                   choices=['fp16', 'hqq', 'kivi', 'think'],
                   help="space-separated list (e.g. 'kivi think'). Matches "
                        "search.py.")
    p.add_argument('--quant_model_paths', type=str, nargs='+', default=[])
    p.add_argument('--w_bits', type=int, nargs='+', default=[])
    p.add_argument('--k_bits', type=int, nargs='+', default=[2, 4])
    p.add_argument('--v_bits', type=int, nargs='+', default=[2, 4])
    p.add_argument('--w_group_size', type=int, default=128)
    p.add_argument('--k_group_size', type=int, default=128)
    p.add_argument('--v_group_size', type=int, default=128)
    p.add_argument('--residual_length', type=int, default=128)
    p.add_argument('--attn_sink', type=int, default=0,
                   help='Keep first S KV tokens in FP (KVSink). 0=off. Match the search-time value.')
    p.add_argument('--k_quant_scheme', type=str, choices=['channel', 'token'])
    p.add_argument('--v_quant_scheme', type=str, choices=['channel', 'token'])
    p.add_argument('--outlier_path', type=str, default='')
    # calibration data / metric
    p.add_argument('--datasets', type=str, nargs='+', default=[])
    p.add_argument('--metric', type=str, default='ppl')
    p.add_argument('--loss_func', type=str, default='cross_entropy')
    p.add_argument('--stride', type=int, default=None)
    p.add_argument('--last_tokens', type=int, default=None)
    p.add_argument('--prefill_prompt', action='store_true')
    p.add_argument('--n_sample', type=int, default=128)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--min_seqlen', type=int, default=0)
    p.add_argument('--data_batch_size', type=int, default=1)
    p.add_argument('--n_token', type=int, default=0)
    # expr archives + combined-metric scales
    p.add_argument('--w_expr', type=str, default='')
    p.add_argument('--kv_expr', type=str, default='')
    p.add_argument('--kvdim_expr', type=str, default='')
    p.add_argument('--eff_kv_expr', type=str, default='')
    p.add_argument('--expr_front', action='store_true')
    p.add_argument('--sqrt', action='store_true')
    p.add_argument('--w_scale', type=float, default=1.0)
    p.add_argument('--kv_scale', type=float, default=1.0)
    p.add_argument('--kvdim_scale', type=float, default=1.0)
    p.add_argument('--eff_kv_scale', type=float, default=1.0)
    # comp_obj filter (optional pre-filter before sampling)
    p.add_argument('--comp_obj', type=str, nargs='+', default=[])
    p.add_argument('--comp_obj_min', type=float, nargs='+', default=[])
    p.add_argument('--comp_obj_max', type=float, nargs='+', default=[])
    # sampling
    p.add_argument('--random_sample', type=int, default=None)
    p.add_argument('--quantile_sample', type=str, nargs='+', default=[],
                   help='metric#q1,q2,...  e.g. metric_w#0.01,0.5,0.99')
    p.add_argument('--sampling_method', type=str,
                   default='coverage_nsga2_marginal',
                   choices=['random', 'maximin', 'coverage_nsga2_joint',
                            'coverage_nsga2_marginal',
                            'coverage_nsga2_combined'],
                   help='maximin = model-free farthest-point coverage '
                        '(validated best global-representation sampler, works '
                        'with any surrogate incl. rbf-tps). default = '
                        'sampling_design FINAL DEFAULT (N4: '
                        'marginal + --coverage_coord rank + '
                        '--coverage_per_axis_agg max). coverage_nsga2_combined '
                        '= 2-obj (cov_rad, std_max); use with '
                        '--coverage_pareto_select knee for both-low.')
    p.add_argument('--coverage_coord', type=str, default='rank',
                   choices=['z', 'rank'])
    p.add_argument('--coverage_per_axis_agg', type=str, default='max',
                   choices=['max', 'sum', 'pareto'])
    p.add_argument('--coverage_pareto_select', type=str, default='auto',
                   choices=['auto', 'strategy3', 'knee'],
                   help="multi-obj front -> final K. 'auto' = knee for "
                        "combined, strategy3 otherwise. 'knee' picks the "
                        "balanced (both-low) Pareto solution.")
    # long-ppl key-token options (consumed by the evaluator)
    p.add_argument('--use_key_token', action='store_true')
    p.add_argument('--trunc_len', type=int, default=512)
    p.add_argument('--sliding_window', type=int, default=128)
    p.add_argument('--alpha', type=int, default=2)
    p.add_argument('--beta', type=int, default=-2)
    p.add_argument('--key_token_path', type=str, default='')
    # output
    p.add_argument('--save', type=str, default='')
    p.add_argument('--results_csv_file', type=str, default='results.csv')
    return p


if __name__ == '__main__':
    main(build_parser().parse_args())
