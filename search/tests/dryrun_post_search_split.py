"""Dry-run reproducing the index-selection block of post_search_split.py.

Loads the real .stats files referenced in scripts/post_search_split.sh and
exercises three sampling modes WITHOUT touching the model / GPU:
  1) RANDOM only           — current default in the .sh (working baseline)
  2) QUANTILE only          — needs comp_obj range filter to be tractable
  3) QUANTILE + RANDOM      — combined mode

It prints the chosen indices, the n_total combo space, and how big `ps` would
get along each path, so we can confirm whether the script's planned config
(quantile_sample + comp_obj filter + random_sample) is actually feasible.
"""

import json
import sys
import itertools
import time
import numpy as np

sys.path.insert(0, '/NAS/SJ/actquant/search')

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from utils.func import get_net_info, compute_weight_memory, compute_cache_memory_batch


CONFIG_PATH = '/NAS/SJ/actquant/search/config/llama.json'
MODEL_NAME = 'Llama-3.1-8B-Instruct'
W_EXPR = '/NAS/SJ/actquant/search/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_EXPR = '/NAS/SJ/actquant/search/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_EXPR = '/NAS/SJ/actquant/search/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'

GROUP_SIZE = {'w': 128, 'k': 128, 'v': 128}
N_TOKEN = 16384
COMP_OBJ_VAL = 6271016960  # memory target
COMP_OBJ_TOL = COMP_OBJ_VAL * 0.001


def load_expr(expr_path, comp_obj_key, config, group_size, expr_front=True):
    with open(expr_path) as f:
        result_json = json.load(f)
    archive = result_json['archive'] + result_json['candidates']
    subnets = np.array([v[0] for v in archive])
    metric_vals = [v[1] for v in archive]
    comp_vals = [get_net_info(n, config, group_size)[comp_obj_key] for n in subnets]
    sort_idx = np.argsort(metric_vals)
    F = np.column_stack((metric_vals, comp_vals))[sort_idx]
    subnets = subnets[sort_idx]
    if expr_front:
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F = F[front]
        subnets = subnets[front]
    return subnets, F


def build_default_arch(config):
    n_block = config['n_block']
    w_linears = config['linear']
    return {
        'q': {
            'w': {ln: [4] * n_block for ln in w_linears},
            'k': [[4, 128]] * n_block,
            'v': [[4, 128]] * n_block,
        },
        'p': {'k': [0] * n_block, 'v': [0] * n_block},
    }


def build_merged_arch(default_arch, expr_keys, _esm, nd_idx_row):
    arch = {
        'q': {'w': default_arch['q']['w'], 'k': default_arch['q']['k'], 'v': default_arch['q']['v']},
        'p': {'k': default_arch['p']['k'], 'v': default_arch['p']['v']},
    }
    for dim_i, key in enumerate(expr_keys):
        sv = _esm[key][nd_idx_row[dim_i]]
        if key == 'w':
            arch['q']['w'] = sv['q']['w']
        elif key == 'kv':
            arch['q']['k'] = sv['q']['k']; arch['q']['v'] = sv['q']['v']
        elif key == 'kvdim':
            arch['p']['k'] = sv['p']['k']; arch['p']['v'] = sv['p']['v']
    return arch


def main():
    np.random.seed(0)
    with open(CONFIG_PATH) as f:
        config = json.load(f)[MODEL_NAME]

    # ---------- Load expr archives ----------
    t0 = time.time()
    sv_w, F_w = load_expr(W_EXPR, 'wbits', config, GROUP_SIZE)
    sv_kv, F_kv = load_expr(KV_EXPR, 'kvbits', config, GROUP_SIZE)
    sv_kvd, F_kvd = load_expr(KVDIM_EXPR, 'kvdim', config, GROUP_SIZE)
    print(f'[load] sizes: w={len(sv_w)} kv={len(sv_kv)} kvdim={len(sv_kvd)}'
          f'  product={len(sv_w)*len(sv_kv)*len(sv_kvd)}  ({time.time()-t0:.1f}s)')

    expr_keys = ['w', 'kv', 'kvdim']
    _esm = {'w': sv_w, 'kv': sv_kv, 'kvdim': sv_kvd}
    _efm = {'w': F_w, 'kv': F_kv, 'kvdim': F_kvd}
    nd_shape = (len(sv_w), len(sv_kv), len(sv_kvd))
    n_dims = 3
    n_total = int(np.prod(nd_shape))

    # ---------- Step 1: combined metric ND ----------
    t0 = time.time()
    metric_1d = [_efm[k][:, 0] for k in expr_keys]
    new_metric_nd = sum(s * a for s, a in zip([1.0, 1.0, 1.0], np.ix_(*metric_1d)))
    print(f'[metric] nd shape={new_metric_nd.shape}  ({time.time()-t0:.1f}s)')

    # ---------- Step 2: memory comp_obj ND ----------
    t0 = time.time()
    default_arch = build_default_arch(config)
    w_mem = np.array([compute_weight_memory(sv, config, GROUP_SIZE) for sv in sv_w])
    mem_nd = np.broadcast_to(w_mem.reshape(-1, 1, 1), nd_shape).astype(np.float64)
    # kv + kvdim → 2D batch
    kv_2d = compute_cache_memory_batch(sv_kv, sv_kvd, config, N_TOKEN)
    mem_nd = mem_nd + np.broadcast_to(kv_2d.reshape(1, kv_2d.shape[0], kv_2d.shape[1]), nd_shape)
    print(f'[memory] mem_nd built  range=[{mem_nd.min():.3e}, {mem_nd.max():.3e}]  ({time.time()-t0:.1f}s)')

    # ---------- Mode A: RANDOM only (random_only path) ----------
    print('\n=== Mode A: RANDOM only (random_only) ===')
    np.random.seed(0)
    n_draw = min(200, n_total)
    rng_flat = np.random.choice(n_total, size=n_draw, replace=False)
    valid_nd_idx_A = np.stack(np.unravel_index(rng_flat, nd_shape), axis=1)
    valid_metrics_A = new_metric_nd[tuple(valid_nd_idx_A.T)]
    valid_nd_idx_A = valid_nd_idx_A[np.argsort(valid_metrics_A)]
    print(f'  valid_nd_idx_A: {valid_nd_idx_A.shape},  ps size that would be built: {len(valid_nd_idx_A)}')

    # ---------- Mode B: QUANTILE only — REQUIRES comp_obj filter for tractability ----------
    print('\n=== Mode B: QUANTILE + comp_obj filter (memory range) ===')
    t0 = time.time()
    lo, hi = COMP_OBJ_VAL - COMP_OBJ_TOL, COMP_OBJ_VAL + COMP_OBJ_TOL
    print(f'  memory range: [{lo:.3e}, {hi:.3e}]')
    mask = (mem_nd >= lo) & (mem_nd <= hi)
    valid_nd_idx_B = np.argwhere(mask)
    print(f'  valid after filter: {len(valid_nd_idx_B)}  ({time.time()-t0:.1f}s)')
    if len(valid_nd_idx_B) == 0:
        print('  !!! filter produced empty set — broaden the threshold')
        return
    valid_metrics_B = new_metric_nd[tuple(valid_nd_idx_B.T)]
    valid_nd_idx_B = valid_nd_idx_B[np.argsort(valid_metrics_B)]

    # Build ps for filtered set
    t0 = time.time()
    ps_B = [build_merged_arch(default_arch, expr_keys, _esm, valid_nd_idx_B[r]) for r in range(len(valid_nd_idx_B))]
    print(f'  ps_B size: {len(ps_B)}  ({time.time()-t0:.1f}s to build)')

    # Quantile selection
    t0 = time.time()
    quantile_specs = {'wbits': [0.1, 0.5, 0.9], 'kvbits': [0.1, 0.5, 0.9]}
    metric_vals = {}
    for key in quantile_specs:
        metric_vals[key] = np.array([
            get_net_info(a, config, GROUP_SIZE, n_token=N_TOKEN)[key] for a in ps_B
        ])
    target_vals = {k: [np.quantile(metric_vals[k], q) for q in qs] for k, qs in quantile_specs.items()}
    for k in quantile_specs:
        print(f'  {k}: range=[{metric_vals[k].min():.4f}, {metric_vals[k].max():.4f}]  targets={[f"{t:.4f}" for t in target_vals[k]]}')

    I_set = set()
    keys = list(quantile_specs.keys())
    for combo in itertools.product(*[range(len(quantile_specs[k])) for k in keys]):
        targets = {k: target_vals[k][qi] for k, qi in zip(keys, combo)}
        dists = np.zeros(len(ps_B))
        for k, t in targets.items():
            v = metric_vals[k]; rng = v.max() - v.min()
            dists += ((v - t) / rng) ** 2 if rng > 0 else (v - t) ** 2
        I_set.add(int(np.argmin(dists)))
    I_quant = sorted(I_set)
    print(f'  quantile picks: {len(I_quant)} unique  indices={I_quant}  ({time.time()-t0:.2f}s)')

    # ---------- Mode C: QUANTILE + RANDOM ----------
    print('\n=== Mode C: QUANTILE + RANDOM extra ===')
    np.random.seed(0)
    n_extra_request = 200
    quant_set = set(I_quant)
    available = np.array([j for j in range(len(ps_B)) if j not in quant_set], dtype=np.int64)
    n_extra = int(min(n_extra_request, len(available)))
    extra = np.random.choice(available, size=n_extra, replace=False) if n_extra > 0 else np.array([], dtype=np.int64)
    extra_list = sorted(int(e) for e in extra)
    assert quant_set.isdisjoint(extra_list)
    I_total = sorted(list(quant_set) + extra_list)
    print(f'  quant={len(quant_set)}  extra={len(extra_list)}  total={len(I_total)}')
    print(f'  pool size (complement): {len(available)}')
    assert len(set(I_total)) == len(I_total), 'duplicates in I_total'
    print('  --> indices unique, disjoint with quantile set: OK')

    # ---------- Mode D: QUANTILE without filter (the BROKEN path) ----------
    print('\n=== Mode D (warning): QUANTILE without comp_obj filter ===')
    print(f'  full sort would produce ps of size n_total={n_total}.')
    print(f'  Building {n_total} arch dicts and computing get_net_info for each is INFEASIBLE')
    print(f'  → script must enable --comp_obj/--comp_obj_min/--comp_obj_max when --quantile_sample is set.')


if __name__ == '__main__':
    main()
