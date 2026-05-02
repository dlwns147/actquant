"""Dry-run for the QUANTILE_SAMPLE + RANDOM_SAMPLE flow with NO comp_obj filter.

Exercises the optimized post_search_split.py path on the full 32M combo space.
Checks the optimisation eliminates the O(N_valid) get_net_info loop while still
producing correct quantile selections.
"""

import json, sys, time, itertools
import numpy as np

sys.path.insert(0, '/NAS/SJ/actquant/search')
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from utils.func import get_net_info

CONFIG_PATH = '/NAS/SJ/actquant/search/config/llama.json'
W_EXPR = '/NAS/SJ/actquant/search/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_EXPR = '/NAS/SJ/actquant/search/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_EXPR = '/NAS/SJ/actquant/search/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'

GROUP_SIZE = {'w': 128, 'k': 128, 'v': 128}
N_TOKEN = 16384


def load_expr(p, key, cfg, gs):
    with open(p) as f:
        j = json.load(f)
    archive = j['archive'] + j['candidates']
    sv = np.array([v[0] for v in archive])
    mv = [v[1] for v in archive]
    cv = [get_net_info(n, cfg, gs)[key] for n in sv]
    sort = np.argsort(mv)
    F = np.column_stack((mv, cv))[sort]
    sv = sv[sort]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return sv[front], F[front]


def main():
    np.random.seed(0)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)['Llama-3.1-8B-Instruct']

    t0 = time.time()
    sv_w, F_w = load_expr(W_EXPR, 'wbits', cfg, GROUP_SIZE)
    sv_kv, F_kv = load_expr(KV_EXPR, 'kvbits', cfg, GROUP_SIZE)
    sv_kvd, F_kvd = load_expr(KVDIM_EXPR, 'kvdim', cfg, GROUP_SIZE)
    print(f'[load] sizes: w={len(sv_w)} kv={len(sv_kv)} kvdim={len(sv_kvd)}  ({time.time()-t0:.1f}s)')

    expr_keys = ['w', 'kv', 'kvdim']
    _esm = {'w': sv_w, 'kv': sv_kv, 'kvdim': sv_kvd}
    _efm = {'w': F_w, 'kv': F_kv, 'kvdim': F_kvd}
    nd_shape = (len(sv_w), len(sv_kv), len(sv_kvd))
    n_total = int(np.prod(nd_shape))

    # Combined metric ND
    metric_1d = [_efm[k][:, 0] for k in expr_keys]
    new_metric_nd = sum(s * a for s, a in zip([1.0, 1.0, 1.0], np.ix_(*metric_1d)))

    # No filter: full sort path
    t0 = time.time()
    sort_order = np.argsort(new_metric_nd.ravel())
    valid_nd_idx = np.stack(np.unravel_index(sort_order, nd_shape), axis=1)
    print(f'[full-sort] valid_nd_idx shape={valid_nd_idx.shape}  ({time.time()-t0:.1f}s)')

    # ---- Optimised quantile_sample (per-axis lookup) ----
    quantile_specs = {'wbits': [0.1, 0.5, 0.9],
                      'kvbits': [0.1, 0.5, 0.9],
                      'kvdim': [0.1, 0.5, 0.9]}

    # Per-axis precompute (cheap: max 754 calls)
    t0 = time.time()
    per_axis = {
        'wbits':  (expr_keys.index('w'),    np.array([get_net_info(s, cfg, GROUP_SIZE)['wbits']  for s in _esm['w']])),
        'kvbits': (expr_keys.index('kv'),   np.array([get_net_info(s, cfg, GROUP_SIZE)['kvbits'] for s in _esm['kv']])),
        'kvdim':  (expr_keys.index('kvdim'),np.array([get_net_info(s, cfg, GROUP_SIZE)['kvdim']  for s in _esm['kvdim']])),
    }
    print(f'[per-axis precompute] ({time.time()-t0:.2f}s)')

    t0 = time.time()
    metric_vals = {k: arr[valid_nd_idx[:, ax]] for k, (ax, arr) in per_axis.items()}
    print(f'[per-axis lookup over 32M] ({time.time()-t0:.2f}s)')
    for k, v in metric_vals.items():
        print(f'  {k}: shape={v.shape}, range=[{v.min():.3f}, {v.max():.3f}]')

    # Quantile selection
    target_vals = {k: [np.quantile(v, q) for q in quantile_specs[k]] for k, v in metric_vals.items()}
    for k in quantile_specs:
        print(f'  {k} targets: {[f"{t:.3f}" for t in target_vals[k]]}')

    t0 = time.time()
    I_set = set()
    keys = list(quantile_specs.keys())
    for combo in itertools.product(*[range(len(quantile_specs[k])) for k in keys]):
        targets = {k: target_vals[k][qi] for k, qi in zip(keys, combo)}
        dists = np.zeros(len(valid_nd_idx))
        for k, t in targets.items():
            v = metric_vals[k]
            rng = v.max() - v.min()
            dists += ((v - t) / rng) ** 2 if rng > 0 else (v - t) ** 2
        I_set.add(int(np.argmin(dists)))
    I_quant = sorted(I_set)
    print(f'[quantile selection 27 combos × 32M argmin] ({time.time()-t0:.2f}s)')
    print(f'I_quant: {len(I_quant)} unique')
    for i in I_quant[:9]:
        info = {k: f'{metric_vals[k][i]:.3f}' for k in metric_vals}
        print(f'  arch[{i}]: {info}')
    if len(I_quant) > 9:
        print(f'  ... ({len(I_quant)-9} more)')

    # Random complement (vectorised)
    t0 = time.time()
    n_extra = 23
    quant_set = set(I_quant)
    mask = np.ones(len(valid_nd_idx), dtype=bool)
    mask[np.fromiter(quant_set, dtype=np.int64)] = False
    available = np.flatnonzero(mask)
    extra = np.random.choice(available, size=n_extra, replace=False)
    print(f'[random complement] pool={len(available)}, extra={len(extra)}  ({time.time()-t0:.2f}s)')
    I_total = sorted(list(quant_set) + sorted(int(e) for e in extra))
    print(f'I_total: {len(I_total)} (should be {len(I_quant)} + {n_extra})')
    assert len(I_total) == len(I_quant) + n_extra
    assert len(set(I_total)) == len(I_total)


if __name__ == '__main__':
    main()
