"""Why does wbits/kvbits collapse to a narrow band even though the search
space allows [2, 4]?  Hypothesis: the --comp_obj memory filter (target
6.271e9 ± 0.1%) is so tight that only one specific bit combination satisfies
it (weights ~4-bit, KV ~3-4-bit).  This script proves it by sweeping the
filter from very tight → very loose and reporting the resulting wbits/kvbits
ranges in the filtered set.
"""

import json, sys, time
import numpy as np

sys.path.insert(0, '/NAS/SJ/actquant/search')
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from utils.func import get_net_info, compute_weight_memory, compute_cache_memory_batch

CONFIG_PATH = '/NAS/SJ/actquant/search/config/llama.json'
W_EXPR = '/NAS/SJ/actquant/search/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_EXPR = '/NAS/SJ/actquant/search/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_EXPR = '/NAS/SJ/actquant/search/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'

GROUP_SIZE = {'w': 128, 'k': 128, 'v': 128}
N_TOKEN = 16384
COMP_OBJ_VAL = 6271016960


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
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)['Llama-3.1-8B-Instruct']

    sv_w, _ = load_expr(W_EXPR, 'wbits', cfg, GROUP_SIZE)
    sv_kv, _ = load_expr(KV_EXPR, 'kvbits', cfg, GROUP_SIZE)
    sv_kvd, _ = load_expr(KVDIM_EXPR, 'kvdim', cfg, GROUP_SIZE)
    print(f'pareto sizes: w={len(sv_w)} kv={len(sv_kv)} kvdim={len(sv_kvd)}')

    # 1) Distribution of per-component bits in the FULL pareto fronts (no filter)
    w_wbits = np.array([get_net_info(s, cfg, GROUP_SIZE)['wbits'] for s in sv_w])
    kv_kvbits = np.array([get_net_info(s, cfg, GROUP_SIZE)['kvbits'] for s in sv_kv])
    print(f'\n[no filter] sv_w wbits range = [{w_wbits.min():.3f}, {w_wbits.max():.3f}]   (n={len(w_wbits)})')
    print(f'[no filter] sv_kv kvbits range = [{kv_kvbits.min():.3f}, {kv_kvbits.max():.3f}]   (n={len(kv_kvbits)})')

    # 2) Build memory ND grid (754 x 331 x 129 ≈ 32M)
    nd_shape = (len(sv_w), len(sv_kv), len(sv_kvd))
    w_mem = np.array([compute_weight_memory(s, cfg, GROUP_SIZE) for s in sv_w])
    mem_nd = np.broadcast_to(w_mem.reshape(-1, 1, 1), nd_shape).astype(np.float64)
    kv2d = compute_cache_memory_batch(sv_kv, sv_kvd, cfg, N_TOKEN)
    mem_nd = mem_nd + np.broadcast_to(kv2d.reshape(1, *kv2d.shape), nd_shape)
    print(f'\nmemory grid: range=[{mem_nd.min():.3e}, {mem_nd.max():.3e}]   target={COMP_OBJ_VAL:.3e}')

    # 3) For each axis, compute the per-arch bits — then for each tolerance,
    # report which wbits/kvbits values survive the memory filter.
    wbits_axis = w_wbits                          # (754,)
    kvbits_axis = kv_kvbits                       # (331,)
    # Broadcast wbits/kvbits onto the same shape as mem_nd so we can mask jointly.
    wbits_nd = np.broadcast_to(wbits_axis.reshape(-1, 1, 1), nd_shape)
    kvbits_nd = np.broadcast_to(kvbits_axis.reshape(1, -1, 1), nd_shape)

    print('\n  filter_pct       n_valid     wbits_range          kvbits_range')
    print('  ' + '-' * 75)
    for pct in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.00]:
        tol = COMP_OBJ_VAL * pct
        lo, hi = COMP_OBJ_VAL - tol, COMP_OBJ_VAL + tol
        mask = (mem_nd >= lo) & (mem_nd <= hi)
        n = int(mask.sum())
        if n == 0:
            print(f'  ±{pct*100:5.2f}%       {n:>8}    (empty)')
            continue
        wb = wbits_nd[mask]
        kb = kvbits_nd[mask]
        print(f'  ±{pct*100:5.2f}%       {n:>8}    [{wb.min():.3f}, {wb.max():.3f}]    [{kb.min():.3f}, {kb.max():.3f}]')

    # 4) Confirm: at the 6.27e9 target, what wbits would let the lightest 2-bit
    # weights (memory ≈ ?) and heaviest 4-bit weights (memory ≈ ?) co-exist?
    print('\n[reference] weight_memory range over sv_w pareto front:')
    print(f'  weights @{w_wbits.min():.2f}-bit (lightest in pareto): {w_mem.min():.3e}')
    print(f'  weights @{w_wbits.max():.2f}-bit (heaviest in pareto): {w_mem.max():.3e}')
    print(f'  diff = {w_mem.max() - w_mem.min():.3e}  vs filter tol@0.1% = {COMP_OBJ_VAL*0.001:.3e}')


if __name__ == '__main__':
    main()
