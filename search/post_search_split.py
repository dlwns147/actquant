import os
import json
import argparse
import itertools
import torch
import numpy as np
from pymoo.decomposition.asf import ASF
from pymoo.visualization.scatter import Scatter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# from pymoo.model.decision_making import DecisionMaking, normalize, find_outliers_upper_tail, NeighborFinder
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder
from pymoo.util.normalization import normalize

from evaluator import LlamaEvaluator
from tqdm import tqdm
from time import time
import csv
import scipy.stats as stats
from matplotlib import pyplot as plt
from utils.func import (init_accelerator, get_net_info, clean_up, process_dtype, set_seed,
                        compute_weight_memory, compute_cache_memory_single,
                        compute_cache_memory_batch, compute_eff_kvbits_batch)
from utils.eval import measure_latency, eval_zeroshot
from utils.longbench import pred_longbench, eval_longbench
from utils.data import get_tokenizer
from utils.ruler import eval_ruler
from utils.longeval import eval_longeval_lines, generate_lines_testcases
from utils.minilongbench import pred_minilongbench, eval_minilongbench
import warnings
warnings.simplefilter("ignore")

class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, normalize=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected
        self.normalize = normalize

    def _do(self, F, **kwargs):
        n, m = F.shape

        if self.normalize:
            F = normalize(F, estimate_bounds_if_none=True)
            # F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

        neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            # np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            return find_outliers_upper_tail(mu)  # return points with trade-off > 2*sigma


def main(args):
    print(args)

    # Generate testcases if requested
    if args.generate_testcases:
        print("Generating LongEval testcases...")
        generate_lines_testcases(
            num_lines_list=args.generate_testcases_num_lines,
            num_test_samples=args.generate_testcases_num_samples,
            line_idx_opt=args.generate_testcases_line_idx_opt,
            output_dir=args.generate_testcases_output_dir
        )
        print("Testcase generation completed.")
        if args.generate_testcases_only:
            return

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    dtype = process_dtype(args.dtype)

    # assert len(args.expr) == len(args.expr_comp_obj)
    n_comp_obj, n_comp_obj_min, n_comp_obj_max = len(args.comp_obj), len(args.comp_obj_min), len(args.comp_obj_max)
    assert n_comp_obj == n_comp_obj_min and n_comp_obj_min == n_comp_obj_max

    group_size = {'w': args.w_group_size, 'k': args.k_group_size, 'v': args.v_group_size}
    # Build default arch from args for components not covered by any expr
    n_block = config['n_block']
    w_linears = config['linear']
    default_w_bits = max(args.w_bits) if args.w_bits else 16
    default_k_bits = max(args.k_bits) if args.k_bits else 4
    default_v_bits = max(args.v_bits) if args.v_bits else 4
    k_gs = args.k_group_size if isinstance(args.k_group_size, int) else min(args.k_group_size)
    v_gs = args.v_group_size if isinstance(args.v_group_size, int) else min(args.v_group_size)
    default_arch = {
        'q': {
            'w': {linear: [default_w_bits] * n_block for linear in w_linears},
            'k': [[default_k_bits, k_gs]] * n_block,
            'v': [[default_v_bits, v_gs]] * n_block,
        },
        'p': {
            'k': [0] * n_block,
            'v': [0] * n_block,
        }
    }

    # Load and optionally Pareto-filter each provided expr archive
    def load_expr(expr_path, comp_obj_key):
        with open(expr_path, 'r') as f:
            result_json = json.load(f)
        archive = result_json['archive'] + result_json['candidates']
        subnets_arr = np.array([v[0] for v in archive])
        metric_vals = [v[1] for v in archive]
        comp_vals = [get_net_info(n, config, group_size)[comp_obj_key] for n in subnets_arr]
        sort_idx = np.argsort(metric_vals)
        F = np.column_stack((metric_vals, comp_vals))[sort_idx]
        subnets_arr = subnets_arr[sort_idx]
        if args.expr_front:
            front = NonDominatedSorting().do(F, only_non_dominated_front=True)
            F = F[front]
            subnets_arr = subnets_arr[front]
        return subnets_arr, F

    expr_map = {}
    if args.w_expr:
        expr_map['w'] = load_expr(args.w_expr, 'wbits')
    if args.kv_expr:
        expr_map['kv'] = load_expr(args.kv_expr, 'kvbits')
    if args.kvdim_expr:
        expr_map['kvdim'] = load_expr(args.kvdim_expr, 'kvdim')
    if args.eff_kv_expr:
        expr_map['eff_kv'] = load_expr(args.eff_kv_expr, 'eff_kvbits')
    assert len(expr_map) >= 1, "At least 1 of --w_expr, --kv_expr, --kvdim_expr, --eff_kv_expr must be provided"

    expr_keys = list(expr_map.keys())
    scales = {'w': args.w_scale, 'kv': args.kv_scale, 'kvdim': args.kvdim_scale, 'eff_kv': args.eff_kv_scale}

    # ──────────────────────────────────────────────────────────────────────────
    # Vectorized combo generation — filter-first approach
    # Builds comp_obj as ND broadcast arrays, applies range mask before sorting,
    # then assembles F only for the valid subset (typically << n_total combos).
    # ──────────────────────────────────────────────────────────────────────────
    _esm = {k: sv for k, (sv, fv) in expr_map.items()}  # subnet arrays per component
    _efm = {k: fv for k, (sv, fv) in expr_map.items()}  # F arrays per component
    nd_shape = tuple(len(_esm[k]) for k in expr_keys)
    n_dims   = len(nd_shape)

    # Helper: broadcast 1-D array onto axis `ax` of nd_shape (lazy, no copy)
    def _bcast_1d(arr, ax):
        shape = [1] * n_dims
        shape[ax] = len(arr)
        return np.broadcast_to(arr.reshape(shape), nd_shape)

    # Helper: broadcast 2-D array onto axes (ax0, ax1) of nd_shape (lazy, no copy)
    def _bcast_2d(arr, ax0, ax1):
        shape = [1] * n_dims
        shape[ax0] = arr.shape[0]
        shape[ax1] = arr.shape[1]
        return np.broadcast_to(arr.reshape(shape), nd_shape)

    # Step 1: new_metric ND array via outer sum (lazy np.ix_ broadcasting)
    metric_1d = [_efm[k][:, 0] for k in expr_keys]
    metric_1d_used = [np.sqrt(m) if args.sqrt else m for m in metric_1d]
    scale_vals = [scales[k] for k in expr_keys]
    new_metric_nd = sum(s * a for s, a in zip(scale_vals, np.ix_(*metric_1d_used)))

    # Step 2: Build comp_obj as ND broadcast arrays (shape nd_shape)
    comp_nd_list = []
    for obj in args.comp_obj:
        if obj == 'wbits':
            ax = expr_keys.index('w')
            v = np.array([get_net_info(sv, config, group_size)['wbits'] for sv in _esm['w']])
            comp_nd_list.append(_bcast_1d(v, ax))

        elif obj in ('kvbits', 'kbits', 'vbits'):
            kv_key = 'eff_kv' if 'eff_kv' in expr_keys else 'kv'
            ax = expr_keys.index(kv_key)
            v = np.array([get_net_info(sv, config, group_size)[obj] for sv in _esm[kv_key]])
            comp_nd_list.append(_bcast_1d(v, ax))

        elif obj in ('kvdim', 'kdim', 'vdim'):
            kv_key = 'eff_kv' if 'eff_kv' in expr_keys else 'kvdim'
            ax = expr_keys.index(kv_key)
            v = np.array([get_net_info(sv, config, group_size)[obj] for sv in _esm[kv_key]])
            comp_nd_list.append(_bcast_1d(v, ax))

        elif obj in ('eff_kvbits', 'eff_kbits', 'eff_vbits'):
            t_map = {'eff_kvbits': 'kv', 'eff_kbits': 'k', 'eff_vbits': 'v'}
            t = t_map[obj]
            if 'eff_kv' in expr_keys:
                ax = expr_keys.index('eff_kv')
                v = np.array([get_net_info(sv, config, group_size)[obj] for sv in _esm['eff_kv']])
                comp_nd_list.append(_bcast_1d(v, ax))
            else:
                assert 'kv' in expr_keys and 'kvdim' in expr_keys, \
                    f"'{obj}' requires eff_kv_expr or both kv_expr and kvdim_expr"
                vals_2d = compute_eff_kvbits_batch(_esm['kv'], _esm['kvdim'], config, target=t)
                comp_nd_list.append(_bcast_2d(vals_2d,
                                              expr_keys.index('kv'), expr_keys.index('kvdim')))

        elif obj == 'memory':
            # Weight memory is separable on the w-axis
            if 'w' in expr_keys:
                w_mem = np.array([compute_weight_memory(sv, config, group_size) for sv in _esm['w']])
                mem_nd = _bcast_1d(w_mem, expr_keys.index('w')).astype(np.float64)
            else:
                mem_nd = np.full(nd_shape,
                                 compute_weight_memory(default_arch, config, group_size),
                                 dtype=np.float64)
            # KV cache memory (kv + kvdim not fully separable → materialise once)
            if 'eff_kv' in expr_keys:
                kv_cache = np.array([compute_cache_memory_single(sv, config, args.n_token)
                                     for sv in _esm['eff_kv']])
                mem_nd = mem_nd + _bcast_1d(kv_cache, expr_keys.index('eff_kv'))
            elif 'kv' in expr_keys and 'kvdim' in expr_keys:
                kv_2d = compute_cache_memory_batch(_esm['kv'], _esm['kvdim'], config, args.n_token)
                mem_nd = mem_nd + _bcast_2d(kv_2d,
                                            expr_keys.index('kv'), expr_keys.index('kvdim'))
            elif 'kv' in expr_keys:
                kv_1d = compute_cache_memory_batch(_esm['kv'], None, config, args.n_token)
                mem_nd = mem_nd + _bcast_1d(kv_1d, expr_keys.index('kv'))
            elif 'kvdim' in expr_keys:
                kv_1d = compute_cache_memory_batch([default_arch], _esm['kvdim'],
                                                   config, args.n_token)[0]
                mem_nd = mem_nd + _bcast_1d(kv_1d, expr_keys.index('kvdim'))
            comp_nd_list.append(mem_nd)

        else:
            raise ValueError(
                f"comp_obj='{obj}' not supported for vectorized computation. "
                f"Supported: wbits, kvbits, kbits, vbits, kvdim, kdim, vdim, "
                f"eff_kvbits, eff_kbits, eff_vbits, memory")

    # Step 3: Get valid ND indices — three mutually exclusive paths
    n_total = int(np.prod(nd_shape))

    if args.random_sample is not None:
        # Sample from the full combo space (all Pareto combinations), ignoring range filter
        n_draw = min(args.random_sample, n_total)
        rng_flat = np.random.choice(n_total, size=n_draw, replace=False)
        valid_nd_idx = np.stack(np.unravel_index(rng_flat, nd_shape), axis=1)
        valid_metrics = new_metric_nd[tuple(valid_nd_idx.T)]
        valid_nd_idx  = valid_nd_idx[np.argsort(valid_metrics)]

    elif n_comp_obj_min > 0:
        # Filter-first: apply range mask on ND arrays → only N_valid << n_total indices
        mask = np.ones(nd_shape, dtype=bool)
        for comp_nd, lo, hi in zip(comp_nd_list, args.comp_obj_min, args.comp_obj_max):
            mask &= (comp_nd >= lo) & (comp_nd <= hi)
        valid_nd_idx = np.argwhere(mask)   # (N_valid, n_dims)
        print(f'range_idx : {len(valid_nd_idx)}')
        if len(valid_nd_idx) == 0 and comp_nd_list:
            first_nd = np.asarray(comp_nd_list[0])
            print(f'[debug] comp_obj[0] range in results: min={first_nd.min():.3f}, max={first_nd.max():.3f}')
            print(f'[debug] comp_obj_min={args.comp_obj_min}, comp_obj_max={args.comp_obj_max}')
        valid_metrics = new_metric_nd[tuple(valid_nd_idx.T)]
        valid_nd_idx  = valid_nd_idx[np.argsort(valid_metrics)]

    else:
        # No filter, no random: unavoidably sort all combos O(n_total)
        sort_order   = np.argsort(new_metric_nd.ravel())
        valid_nd_idx = np.stack(np.unravel_index(sort_order, nd_shape), axis=1)

    # Step 5: Assemble small F for N_valid rows only
    vt = tuple(valid_nd_idx.T)   # tuple of (N_valid,) index arrays, one per dim
    new_metric_vals = new_metric_nd[vt]
    comp_metrics    = np.column_stack([_efm[k][valid_nd_idx[:, i]]
                                       for i, k in enumerate(expr_keys)])
    F_parts = [new_metric_vals.reshape(-1, 1), comp_metrics]
    if comp_nd_list:
        comp_obj_vals = np.column_stack([np.asarray(nd)[vt] for nd in comp_nd_list])
        F_parts.append(comp_obj_vals)
    F = np.column_stack(F_parts)

    # Step 6: Build merged arch dicts only for final selected rows
    def _build_merged_arch_nd(nd_idx_row):
        """nd_idx_row: int array of length n_dims (one entry per expr component)."""
        arch = {
            'q': {
                'w': default_arch['q']['w'],
                'k': default_arch['q']['k'],
                'v': default_arch['q']['v'],
            },
            'p': {
                'k': default_arch['p']['k'],
                'v': default_arch['p']['v'],
            }
        }
        for dim_i, key in enumerate(expr_keys):
            sv = _esm[key][nd_idx_row[dim_i]]
            if key == 'w':
                arch['q']['w'] = sv['q']['w']
            elif key == 'kv':
                arch['q']['k'] = sv['q']['k']
                arch['q']['v'] = sv['q']['v']
            elif key == 'kvdim':
                arch['p']['k'] = sv['p']['k']
                arch['p']['v'] = sv['p']['v']
            elif key == 'eff_kv':
                arch['q']['k'] = sv['q']['k']
                arch['q']['v'] = sv['q']['v']
                arch['p']['k'] = sv['p']['k']
                arch['p']['v'] = sv['p']['v']
        return arch

    def _build_subnets(row_indices):
        """row_indices: indices into valid_nd_idx (the filtered+sorted set)."""
        return np.array([_build_merged_arch_nd(valid_nd_idx[r]) for r in row_indices],
                        dtype=object)

    pf = F
    ps = _build_subnets(np.arange(len(F)))
        
    if args.high_tradeoff:
        I = NonDominatedSorting().do(pf[:, [0, *[i for i in range(-n_comp_obj, 0)]]], only_non_dominated_front=True)

    if args.prefer:
        # preferences
        preferences = {}
        # for p in args.prefer.split("+"):
        for p in args.prefer:
            k, v = p.split("#")
            preferences[k] = float(v)
        weights = np.fromiter(preferences.values(), dtype=float)        
        # I = ASF().do(pf[:, [0, *[i for i in range(-n_comp_obj, 0)]]], weights).argsort()[0]
        I = ASF().do(pf[:, [0, *[i for i in range(-n_comp_obj, 0)]]], weights).argsort()[:args.n].reshape(args.n)
        
    else:
        I = list(range(len(pf)))
        if args.quantile_sample:
            # Parse: "wbits#0.1,0.5,0.9" → {'wbits': [0.1, 0.5, 0.9]}
            quantile_specs = {}
            for spec in args.quantile_sample:
                k, v = spec.split('#')
                quantile_specs[k] = [float(q) for q in v.split(',')]

            # Compute metric values for all architectures in filtered set (ps)
            metric_vals = {}
            for key in quantile_specs:
                metric_vals[key] = np.array([
                    get_net_info(a, config, group_size, n_token=args.n_token)[key] for a in ps
                ])

            # Compute target values at each quantile position
            target_vals = {}
            for key, quantiles in quantile_specs.items():
                target_vals[key] = [np.quantile(metric_vals[key], q) for q in quantiles]
                print(f'[quantile_sample] {key}: range=[{metric_vals[key].min():.4f}, {metric_vals[key].max():.4f}]')
                print(f'[quantile_sample] {key}: targets={[f"{t:.4f}" for t in target_vals[key]]}')

            # For each combination of quantile targets, find the nearest architecture (normalized L2)
            I_set = set()
            keys = list(quantile_specs.keys())
            for combo in itertools.product(*[range(len(quantile_specs[k])) for k in keys]):
                targets = {k: target_vals[k][qi] for k, qi in zip(keys, combo)}
                dists = np.zeros(len(ps))
                for k, t in targets.items():
                    vals = metric_vals[k]
                    val_range = vals.max() - vals.min()
                    dists += ((vals - t) / val_range) ** 2 if val_range > 0 else (vals - t) ** 2
                I_set.add(int(np.argmin(dists)))

            I = sorted(I_set)
            print(f'[quantile_sample] selected {len(I)} unique architectures out of {len(ps)} candidates')
            for i in I:
                info = {k: f'{metric_vals[k][i]:.4f}' for k in quantile_specs}
                print(f'  arch[{i}]: {info}')
        elif args.random_sample is not None and args.random_sample < len(pf):
            I = np.random.choice(I, size=args.random_sample, replace=False)
            I.sort()
        else:
            I = I[:args.n]


    # always add most accurate architectures
    # I = np.append(I, 0)

    # for idx_list in I_list:
    #     # print(f'Selected arch[{idx}] {args.sec_obj}: {pf[idx, 1]:.4f}, metric: {pf[idx, 0]:.4f}, arch: {ps[idx]}')
    #     # print(f'arch : {ps[idx]}')
    #     for i, comp_obj in enumerate(args.comp_obj):
    #         accelerator.print(f'Selected arch[{idx_list[i]}] {comp_obj}: {pf_list[i][idx_list[i], 1:].tolist()}, metric: {pf_list[i][idx_list[i], 0].tolist()}')

    model_id = f'{args.model_path}/{args.model_name}'
    # use_awq_gptq_owq = 'awq' in args.w_method or 'gptq' in args.w_method or 'owq' in args.w_method
    
    if 'hqq' not in args.w_method:
        args.quant_model_paths = []

    evaluator = LlamaEvaluator(
        config,
        accelerator=accelerator,
        model_id=model_id,
        method={'w': args.w_method, 'kv': args.kv_method},
        quant_model_paths=args.quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=args.seqlen,
        min_seqlen=args.min_seqlen,
        n_sample=args.n_sample,        
        datasets=args.datasets,
        device_map=device_map,
        dtype=dtype,
        bits={'w': args.w_bits, 'k': args.k_bits, 'v': args.v_bits},
        group_size=group_size,
        residual_length=args.residual_length,
        # use_flash=args.use_flash,
        k_quant_scheme=args.k_quant_scheme,
        v_quant_scheme=args.v_quant_scheme,
        loss_func=args.loss_func,
        last_tokens=args.last_tokens,
        use_key_token=args.use_key_token,
        trunc_len=args.trunc_len,
        sliding_window=args.sliding_window,
        alpha=args.alpha,
        beta=args.beta,
        key_token_path=args.key_token_path
    )
    
    comp_save_list = [list() for _ in get_net_info({}, config, group_size=-1, n_token=0).keys()]
    n_pf_metric_cols = F.shape[1] - n_comp_obj  # new_metric + one component metric per expr key
    metric_save_list = [list() for _ in range(len(args.datasets) + n_pf_metric_cols)]
    for idx in tqdm(I):
        arch = ps[idx]
            
        # complexity = get_net_info(arch, config, group_size, n_token=args.n_token)
        # latency = measure_latency(model, generation=True, device=model.device) if args.latency else 0
        # print(f'complexity: {complexity}')
        # print(f'arch: {arch}')
        
        complexity = get_net_info(arch, config, group_size, n_token=args.n_token)
        print(f'complexity: {list(complexity.keys())}')
        print(f'complexity: {list(complexity.values())}')
        accelerator.print(f'arch: {arch}')
        model = evaluator.sample(arch)
        
        # for i, comp_obj in enumerate(args.comp_obj):
        #     # for idx in 
        #     # accelerator.print(f'Selected arch[{idx}] {comp_obj}: {pf_list[i][idx_list[i], 1:].tolist()}, metric: {pf_list[i][idx_list[i], 0].tolist()}')   
        #     accelerator.print(f'Selected arch[{idx}] {comp_obj}: {pf_list[i][idx, 1:].tolist()}, metric: {pf_list[i][idx_list[i], 0].tolist()}')            

        if args.datasets:
            if args.stride is not None:
                if 'kivi' in args.kv_method:
                    model.config.kivi_config.residual_length = args.residual_length
                elif 'hqq' in args.kv_method:
                    model.generation_config.cache_config = args.residual_length
                model.config.quant_kv_output = False
                model.config.use_cache = True
                
            else:
                if 'kivi' in args.kv_method:
                    model.config.kivi_config.residual_length = 0
                elif 'hqq' in args.kv_method:
                    model.generation_config.cache_config = 0
                model.config.quant_kv_output = True
                model.config.use_cache = False
            # model.config.quant_kv_output = True
            # model.config.use_cache = False
            # model.config.quant_kv_output = True if args.stride is None else False
            # model.config.use_cache = True if args.stride is not None else False

            metric = evaluator.eval(arch=arch, metric=args.metric, model=model, accelerator=accelerator, loss_func=args.loss_func, stride=args.stride)[0] if args.datasets else 0
            # latency = measure_latency(model, generation=True, device=model.device) if args.latency else 0
            # print(f'[{idx}] complexity: {complexity}, {args.metric}: {[p for p in metric.values()]}, metric: {[pf[idx, 0]]}, prev_metric: {pf[idx, 1: -n_comp_obj]}')
            print(f'[{idx}] {args.metric}: {[p for p in metric.values()]}, metric: {[pf[idx, 0]]}, prev_metric: {pf[idx, 1: -n_comp_obj]}')
            if (args.random_sample is not None or args.quantile_sample) and args.save and args.results_csv_file:
                for c_i, c in enumerate(complexity.values()):
                    comp_save_list[c_i].append(c)
                for m_i, m in enumerate(metric.values()):
                    metric_save_list[m_i].append(m)
                for m_i, m in enumerate(pf[idx, 0: -n_comp_obj]):
                    metric_save_list[m_i + len(args.datasets)].append(m)
                    
                os.makedirs(args.save, exist_ok=True)
                with open(os.path.join(args.save, args.results_csv_file), 'w') as f:
                    writer = csv.writer(f)
                    for c in comp_save_list:
                        writer.writerow(c)                        
                    for m in metric_save_list:
                        writer.writerow(m)
                        

        if args.pass_key_file:
            clean_up()
            # model.config.residual_length = args.residual_length
            if 'kivi' in args.kv_method:
                model.config.kivi_config.residual_length = args.residual_length
            elif 'hqq' in args.kv_method:
                model.generation_config.cache_config = args.residual_length
            model.config.quant_kv_output = False
            model.config.use_cache = True
            
            # method_name = f"K{config.k_bits}V{config.v_bits} KiVi"
            print( "-----------------------------------" )
            enc = get_tokenizer(model_id)
            for line in open(args.pass_key_file, "r"):
                clean_up()
                torch.cuda.reset_max_memory_allocated()
                example = json.loads(line)
                prompt_postfix = "What is the pass key? The pass key is "
                prompt = example["input"] + prompt_postfix
                input_ids = enc(prompt, return_tensors="pt").input_ids.cuda()
                print( "-----------------------------------" )
                print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
                print( "Passkey target:", example["target"] )

                tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
                answer = prompt_postfix + enc.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
                answer = answer.replace("\n", "\\n")
                # answer= f"{method_name}:\n     [ {answer} ]"
                answer= f"[ {answer} ]"
                
                peak_memory = torch.cuda.max_memory_allocated()
                print( answer )
                print(f"Mem: {peak_memory / 1024 / 1024 / 1024:.3f} GB")
                # print(f"Mem: {peak_memory / 1024 / 1024:.3f} MB")
                print( "-----------------------------------\n" )
        
        if args.zeroshot:
            clean_up()
            # model.config.residual_length = args.residual_length
            if 'kivi' in args.kv_method:
                model.config.kivi_config.residual_length = args.residual_length
            elif 'hqq' in args.kv_method:
                model.generation_config.cache_config = args.residual_length
            model.config.quant_kv_output = False
            model.config.use_cache = True
            
            results = eval_zeroshot(model, tokenizer=get_tokenizer(model_id), task_list=args.tasks, batch_size=args.lm_eval_batch_size)
            
            task = list(results.keys())
            total_result = []
            print(f'task : {task}')
            for task, result in results.items():
                # print(f'task: {task}, result: {result}')
                new_result = {}
                for k, v in result.items():
                    if k in ['em,none', 'exact_match,strict-match', 'exact_match,flexible-extract', 'bleu_max,none', 'bleu_acc,none', 'acc,none']:
                        new_result[k] = float(v)
                print(f'task: {task}, result: {list(new_result.keys())}, {list(new_result.values())}')
                total_result += list(new_result.values())
            print(f'total_result: {total_result}')
        
        if args.longbench:
            clean_up()
            # model.config.residual_length = args.residual_length
            if 'kivi' in args.kv_method:
                model.config.kivi_config.residual_length = args.residual_length
            elif 'hqq' in args.kv_method:
                model.generation_config.cache_config = args.residual_length
            model.config.quant_kv_output = False
            model.config.use_cache = True
            
            # if len(args.longbench_task) == 0 and not args.longbench_task_e:
            #     args.longbench_task = []
            longbench_start = time()
            pred_longbench(model, tokenizer=get_tokenizer(model_id), save_path=args.longbench_result_path, longbench_config=args.longbench_config, e=args.longbench_e)
            eval_longbench(args.longbench_result_path, args.longbench_e)
            longbench_time = time() - longbench_start
            
            sentences = []
            for k, v in vars(args).items():
                sentences.append(f"{k}: {v}\n")
            sentences.append(f'Longbench Time: {longbench_time:.2f}s')
            sentences.append("\n")

            with open(os.path.join(args.longbench_result_path, "pred_e" if args.longbench_e else "pred", 'result.txt'), 'w') as f:
                for sentence in sentences:
                    f.write(sentence)

        if args.minilongbench:
            clean_up()
            if 'kivi' in args.kv_method:
                model.config.kivi_config.residual_length = args.residual_length
            elif 'hqq' in args.kv_method:
                model.generation_config.cache_config = args.residual_length
            model.config.quant_kv_output = False
            model.config.use_cache = True

            mlb_start = time()
            pred_minilongbench(
                model,
                tokenizer=get_tokenizer(model_id),
                save_path=args.minilongbench_result_path,
                longbench_config=args.longbench_config,
                data_dir=args.minilongbench_data_dir if args.minilongbench_data_dir else None,
                model_name=args.model_name,
            )
            eval_minilongbench(args.minilongbench_result_path)
            mlb_time = time() - mlb_start
            print(f'MiniLongBench Time: {mlb_time:.2f}s')

            sentences = []
            for k, v in vars(args).items():
                sentences.append(f"{k}: {v}\n")
            sentences.append(f'MiniLongBench Time: {mlb_time:.2f}s\n')
            with open(os.path.join(args.minilongbench_result_path, "pred", "result.txt"), 'w') as f:
                for sentence in sentences:
                    f.write(sentence)

        if args.ruler:
            clean_up()
            if 'kivi' in args.kv_method:
                model.config.kivi_config.residual_length = args.residual_length
            elif 'hqq' in args.kv_method:
                model.generation_config.cache_config = args.residual_length
            model.config.quant_kv_output = False
            model.config.use_cache = True
            # tokenizer=get_tokenizer(model_id)
            # tokenizer.pad_token = tokenizer.eos_token
            
            ruler_start = time()
            eval_ruler(model, tokenizer=get_tokenizer(model_id), model_id=model_id, tasks=args.ruler_task, yaml_path=args.ruler_yaml_path, batch_size=args.ruler_batch_size, length=args.ruler_length, nsample=args.ruler_sample, gen_toks=args.ruler_gen_toks, result_path=args.ruler_result_path, seed=args.seed)
            ruler_time = time() - ruler_start
            print(f'RULER Time: {ruler_time:.2f}s')
            

        if args.longeval:
            clean_up()
            if 'kivi' in args.kv_method:
                model.config.kivi_config.residual_length = args.residual_length
            elif 'hqq' in args.kv_method:
                model.generation_config.cache_config = args.residual_length
            model.config.quant_kv_output = False
            model.config.use_cache = True
            
            tokenizer = get_tokenizer(model_id)
            longeval_start = time()
            
            # Prepare result path
            if args.longeval_result_path:
                os.makedirs(os.path.dirname(args.longeval_result_path) if os.path.dirname(args.longeval_result_path) else '.', exist_ok=True)
                result_path = args.longeval_result_path
            else:
                result_path = ''
            
            # Evaluate longeval lines task
            results = eval_longeval_lines(
                model=model,
                tokenizer=tokenizer,
                test_dir=args.longeval_test_dir,
                model_name_or_path=model_id,
                num_lines_list=args.longeval_num_lines,
                eval_shortest_only=args.longeval_shortest_only,
                result_path=result_path,
                use_cache=True
            )
            
            longeval_time = time() - longeval_start
            print(f'LongEval Lines Task Time: {longeval_time:.2f}s')
            
            # Save results summary
            if args.longeval_result_path:
                sentences = []
                for k, v in vars(args).items():
                    sentences.append(f"{k}: {v}\n")
                sentences.append(f'LongEval Time: {longeval_time:.2f}s')
                sentences.append("\n")
                
                summary_path = args.longeval_result_path.replace('.json', '_summary.txt')
                with open(summary_path, 'w') as f:
                    for sentence in sentences:
                        f.write(sentence)
            
            
        if 'awq' in args.w_method or 'gptq' in args.w_method or 'qeft' in args.w_method:
            del model, evaluator.model
            clean_up()

    print(args)
    return

    # if args.debug:
    #     # print(ps[I])
    #     # plot = Scatter()
    #     # plot.add(pf, alpha=0.2)
    #     # plot.add(pf[I, :], color="blue", s=10)
    #     # plot.add(gs_data, color="red", s=10)
    #     # plot.show()
    #     # plot.save(os.path.join(args.save, "best_trade_off_line.png"))
    #     os.makedirs(args.save, exist_ok=True)
        
    #     plt.scatter(complexity_list, [p[args.datasets[0]] for p in ppl_list], color='b', s=5, label='NSGA2')
    #     if args.greedy_search_result_path:
    #         with open(args.greedy_search_result_path, 'r') as f:
    #             gs_data = list(csv.reader(f))
    #             gs_bits = list(map(float, gs_data[1]))[:-3]
    #             gs_metric = list(map(float, gs_data[2]))[:-3]
    #             plt.scatter(gs_bits, gs_metric, color='r', s=5, label='Greedy Search')
        
    #     plt.xlabel(f'{args.sec_obj}')
    #     plt.ylabel('PPL')
    #     plt.legend()
    #     plt.show()
    #     plt.savefig(os.path.join(args.save, "best_trade_off_line.png"), dpi=300)

    sentences = []
    for k, v in vars(args).items():
        sentences.append(f"{k}: {v}\n")
    sentences.append("\n")
    for a, c, p in zip(arch_list, complexity_list, ppl_list):
        sentences.append(f"arch: {a}, bits: {c:.4f}, ppl: {p}\n")

    with open(os.path.join(args.save, args.results_file), 'w') as f:
        for sentence in sentences:
            f.write(sentence)

    with open(os.path.join(args.save, args.results_csv_file), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['arch', 'bits', 'params', 'sparsity', 'metric', 'latency'] + args.datasets)
        for a, b, p, s, m, l, ppl in zip(arch_list, bits_list, param_list, sparsity_list, metric_list, latency_list, ppl_list):
            writer.writerow([a, b, p, s, m, l] + list(ppl.values()))

    with open(os.path.join(args.save, args.results_arch_file), 'w') as f:
        json.dump({'archive': [[a, c, p] for a, c, p in zip(arch_list, complexity_list, ppl_list)]}, f, ensure_ascii=False, indent=4)

    return


def lsq(jsd_w, jsd_kv, jsd_actual, add_intercept=False):
    """
    Least-squares fit for: jsd_pred = alpha * jsd_w + beta * jsd_kv (+ gamma if add_intercept).
    """
    jsd_w = np.asarray(jsd_w, dtype=float).reshape(-1)
    jsd_kv = np.asarray(jsd_kv, dtype=float).reshape(-1)
    y = np.asarray(jsd_actual, dtype=float).reshape(-1)
    assert jsd_w.shape == jsd_kv.shape == y.shape, "All inputs must have same length"
    
    # Design matrix X
    if add_intercept:
        X = np.column_stack([jsd_w, jsd_kv, np.ones_like(jsd_w)])
    else:
        X = np.column_stack([jsd_w, jsd_kv])

    theta, *_ = np.linalg.lstsq(X, y, rcond=None)

    gamma = 0
    if add_intercept:
        alpha, beta, gamma = theta
    else:
        alpha, beta = theta    
    return {"alpha": float(alpha), "beta": float(beta), "gamma": float(gamma)}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--dtype', type=str, default='auto', choices=['float16', 'float', 'fp16', 'bfloat16', 'bfloat', 'bf16', 'auto'],
                        help='')
    parser.add_argument('--comp_obj', type=str, nargs='+', default=[], 
                        help='second objective to optimize simultaneously')
    parser.add_argument('--comp_obj_min', type=float, nargs='+', default=[],
                        help='')
    parser.add_argument('--comp_obj_max', type=float, nargs='+', default=[],
                        help='')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    
    parser.add_argument('--w_method', type=str, nargs='+', default=[], choices=['fp16', 'awq', 'gptq', 'qeft', 'hqq'],
                        help='')
    parser.add_argument('--kv_method', type=str, default='kivi', choices=['fp16', 'hqq', 'kivi'],
                        help='')
    
    parser.add_argument('--w_bits', type=int, nargs='+', default=[], 
                        help='')
    parser.add_argument('--k_bits', type=int, nargs='+', default=[2, 4], 
                        help='')
    parser.add_argument('--v_bits', type=int, nargs='+', default=[2, 4], 
                        help='')
    
    parser.add_argument('--w_group_size', type=int, default=128, 
                        help='')
    parser.add_argument('--k_group_size', type=int, default=128, 
                        help='')
    parser.add_argument('--v_group_size', type=int, default=128, 
                        help='')
    
    parser.add_argument('--residual_length', type=int, default=128, 
                        help='')
    parser.add_argument('--use_flash', action='store_true', help='')

    parser.add_argument('--k_quant_scheme', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_scheme', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--score', type=str, default='kivi', choices=['hqq', 'kivi'],
                        help='')

    parser.add_argument('--metric', type=str, default='ppl',
                        help='which metric predictor model to fit (ppl/loss)')
    parser.add_argument('--loss_func', type=str, default='cross_entropy',
                        help='')
    parser.add_argument('--stride', type=int, default=None, 
                        help='')
    parser.add_argument('--last_tokens', type=int, default=None, 
                        help='')
    parser.add_argument('--seed', type=int, default=0,
                        help='')
                        
    parser.add_argument('--save', type=str, default='',
                        help='location of dir to save')
    # parser.add_argument('--expr', type=str, nargs='+', default=[''],
    #                     help='')
    parser.add_argument('--w_expr', type=str, default='',
                        help='')
    parser.add_argument('--kv_expr', type=str, default='',
                        help='')
    parser.add_argument('--kvdim_expr', type=str, default='',
                        help='')
    parser.add_argument('--eff_kv_expr', type=str, default='',
                        help='path to eff_kvbits (joint KV bits+dim) search stats file')
    parser.add_argument('--expr_front', action='store_true', help='')
    # parser.add_argument('--expr_comp_obj', type=str, nargs='+', default=[''],
    #                     help='')
    parser.add_argument('--prefer', type=str, nargs='+', default=[], 
                        help='preferences in choosing architectures (metric#10 bits#150)')
    # parser.add_argument('--high_tradeoff', action='store_true', help='')
    parser.add_argument('--high_tradeoff', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('-n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--n_token', type=int, default=0, 
                        help='target sequence length for memory calculation')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='sample number of the calibration set')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='sequential length of the calibaration (train) set')
    parser.add_argument('--min_seqlen', type=int, default=0,
                        help='minimum sequential length of the calibaration gsm8k set')
    parser.add_argument('--data_batch_size', type=int, default=1,
                        help='')
    
    parser.add_argument('--debug', action='store_true', help='')
    parser.add_argument('--datasets', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--only_front', action='store_true', help='')
    parser.add_argument('--results_file', type=str, default='results.txt',
                        help='')
    parser.add_argument('--results_csv_file', type=str, default='results.csv',
                        help='')
    parser.add_argument('--results_arch_file', type=str, default='results_arch.json',
                        help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
    parser.add_argument('--latency', action='store_true', help='')
    parser.add_argument('--zeroshot', action='store_true', help='')
    parser.add_argument('--tasks', type=str, nargs='+', default=['coqa', 'gsm8k', 'truthfulqa'])
    parser.add_argument('--lm_eval_batch_size', type=int, default=None,
                        help='')
    parser.add_argument('--longbench', action='store_true', help='')
    parser.add_argument('--longbench_e', action='store_true',
                        help='number of architectures desired')
    parser.add_argument('--longbench_result_path', type=str, default='',
                        help='')
    parser.add_argument('--longbench_config', type=str, default='',
                        help='')
    parser.add_argument('--longbench_task', type=str, nargs='+', default=[])
    parser.add_argument('--pass_key_file', type=str, default='',
                        help='')
    
    parser.add_argument('--random_sample', type=int, default=None,
                        help='')
    parser.add_argument('--quantile_sample', type=str, nargs='+', default=[],
                        help='sample architectures at specific quantile positions of complexity metrics. '
                             'Format: metric#q1,q2,q3  e.g. --quantile_sample wbits#0.1,0.5,0.9 kvbits#0.1,0.5,0.9')
    parser.add_argument('--random_sample_path', type=str, default='', 
                        help='')
    parser.add_argument('--grid_search', type=float, nargs='+', default=[])
    
    parser.add_argument('--sqrt', action='store_true', help='')
    parser.add_argument('--w_scale', type=float, default=1.0,
                        help='scale weight for w metric in combined metric')
    parser.add_argument('--kv_scale', type=float, default=1.0,
                        help='scale weight for kv metric in combined metric')
    parser.add_argument('--kvdim_scale', type=float, default=1.0,
                        help='scale weight for kvdim metric in combined metric')
    parser.add_argument('--eff_kv_scale', type=float, default=1.0,
                        help='scale weight for eff_kv metric in combined metric')

    parser.add_argument('--use_key_token', action='store_true',
                        help='Only use key tokens for loss calculation (Long PPL/JSD)')
    parser.add_argument('--trunc_len', type=int, default=512,
                        help='truncation length for long PPL/JSD calculation')
    parser.add_argument('--sliding_window', type=int, default=128,
                        help='sliding_window length for long PPL/JSD calculation')
    parser.add_argument('--alpha', type=int, default=2,
                        help='Long-short distance (LSD) threshold for long PPL/JSD calculation')
    parser.add_argument('--beta', type=int, default=-2,
                        help='Long context likelihood (LCL) threshold for long PPL/JSD calculation')
    parser.add_argument('--key_token_path', type=str, default='',
                        help='')

    # MiniLongBench arguments
    parser.add_argument('--minilongbench', action='store_true', help='Run MiniLongBench evaluation')
    parser.add_argument('--minilongbench_result_path', type=str, default='',
                        help='Directory to save MiniLongBench results')
    parser.add_argument('--minilongbench_data_dir', type=str, default='',
                        help='Path to MiniLongBench data directory (default: utils/minilongbench_data/data)')

    parser.add_argument('--ruler', action='store_true', help='')
    parser.add_argument("--ruler_task", type=str, default=None, help="Task name", nargs="+",
                        choices=["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery", "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot"])
    # parser.add_argument("--max_seq_length", type=int, default=4096, 
                        # choices=[4096,8192,16384,32768,65536,131072,262144,524288,1048576], help="Maximum sequence length")
    parser.add_argument("--ruler_length", type=int, nargs='+', default=[4096])
    parser.add_argument('--ruler_yaml_path', type=str, default='',
                        help='')
    parser.add_argument("--ruler_sample", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--ruler_gen_toks", type=int, default=None, help="Number of tokens to generate")
    parser.add_argument("--ruler_batch_size", type=int, default=1, help="Batch size")
    parser.add_argument('--ruler_result_path', type=str, default='',
                        help='')
    
    
    # LongEval arguments
    parser.add_argument('--longeval', action='store_true', help='Enable LongEval lines task evaluation')
    parser.add_argument('--longeval_test_dir', type=str, default='',
                        help='Directory containing longeval test cases (should have lines/testcases/ subdirectory)')
    parser.add_argument('--longeval_num_lines', type=int, nargs='+', default=[200, 300, 400, 500, 600, 680, 700, 800, 900, 1000, 1100, 1200, 1350],
                        help='List of number of lines to test')
    parser.add_argument('--longeval_shortest_only', action='store_true', help='Only evaluate the shortest case')
    parser.add_argument('--longeval_result_path', type=str, default='',
                        help='Path to save LongEval results JSON file')
    
    # LongEval testcase generation arguments
    parser.add_argument('--generate_testcases', action='store_true', help='Generate LongEval testcases')
    parser.add_argument('--generate_testcases_only', action='store_true', help='Only generate testcases and exit')
    parser.add_argument('--generate_testcases_num_lines', type=int, nargs='+', default=[200, 300, 400, 500, 600, 680, 700, 800, 900, 1000, 1100, 1200, 1350],
                        help='List of number of lines to generate testcases for')
    parser.add_argument('--generate_testcases_num_samples', type=int, default=50,
                        help='Number of test samples per number of lines')
    parser.add_argument('--generate_testcases_line_idx_opt', type=str, default='LRT',
                        choices=['LRT', 'LRT-ABCindex', 'LRT-UUID', 'LRT-NL'],
                        help='Type of line index option')
    parser.add_argument('--generate_testcases_output_dir', type=str, default='evaluation',
                        help='Directory to save generated testcases (will create lines/testcases/ subdirectory)')
    


    cfgs = parser.parse_args()
    main(cfgs)
