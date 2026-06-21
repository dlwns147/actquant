"""Standalone (numpy-only) sweep of the post_search COMP_OBJ threshold.

Replicates utils/select.py:_memory_block pass-1 counting (searchsorted) for the
lazy (no --expr_front) path, over the actual w/kv/kvdim .stats archives, to find
the largest threshold fraction that keeps the feasible (w,kv) count under
_LAZY_MAX_FEASIBLE for every COMP_OBJ value in sbatch/actquant/iter/post_search.sh.
"""
import json
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = json.load(open(os.path.join(BASE, "config/llama.json")))["Llama-3.1-8B-Instruct"]
GROUP_SIZE = {"w": 128, "k": 128, "v": 128}
N_BLOCK = CONFIG["n_block"]
LAZY_MAX_FEASIBLE = 5e7

W_EXPR = "save/search/think/2605112032_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_200.stats"
KV_EXPR = "save/search/think/2605112033_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x2_128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_100.stats"
KVDIM_EXPR = "save/search/think/2605112036_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats"

# COMP_OBJ lists per N_TOKEN from sbatch/actquant/iter/post_search.sh
COMP_OBJ_BY_NTOKEN = {
    16384:  [4957511680, 5074591744, 5103165440, 5307637760, 5170110464, 5287190528, 5315764224, 5520236544, 5829926912, 5947006976, 5975580672, 6180052992],
    32768:  [5146255360, 5376581632, 5438709760, 5844508672, 5358854144, 5589180416, 5651308544, 6057107456, 6018670592, 6248996864, 6311124992, 6716923904],
    65536:  [5523742720, 5980561408, 6109798400, 6918250496, 5736341504, 6193160192, 6322397184, 7130849280, 6396157952, 6852976640, 6982213632, 7790665728],
    131072: [6278717440, 7188520960, 7451975680, 9065734144, 6491316224, 7401119744, 7664574464, 9278332928, 7151132672, 8060936192, 8324390912, 9938149376],
}


def _adapt(value, n):
    if isinstance(value, dict):
        return {k: _adapt(v, n) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) == 0 or not isinstance(value[0], (int, float, list)):
            return value
        if len(value) == n:
            return value
        if len(value) > n:
            return value[:n]
        return list(value) + [value[-1]] * (n - len(value))
    return value


def load_subnets(path):
    rj = json.load(open(os.path.join(BASE, path)))
    archive = rj["archive"] + rj["candidates"]
    return [_adapt(v[0], N_BLOCK) for v in archive]


def compute_weight_memory(arch):
    wgs = GROUP_SIZE["w"]
    m = 0
    for linear in CONFIG["linear"]:
        out_dim, in_dim = map(int, CONFIG["linear_shape"][linear])
        lgs = in_dim if wgs == -1 else wgs
        for bits in arch["q"]["w"][linear]:
            m += out_dim * in_dim * bits // 8
            if bits < 16:
                m += (in_dim // lgs) * out_dim * 4
    m += int(CONFIG["vocab_size"]) * int(CONFIG["hidden_size"]) * 4
    m += int(CONFIG["n_norm"]) * int(CONFIG["hidden_size"]) * 2
    m += int(CONFIG["max_position_embeddings"]) * int(CONFIG["head_dim"]) * 2
    return m


def cache_memory_base(kv_subnets, kvdim_subnets):
    """KV cache memory at n_token=1, exact equivalent of compute_cache_memory_batch
    but via matmul (no (N_kv,N_kvdim,L) 3D allocation). Returns (N_kv, N_kvdim).
    Memory at any n_token is this * n_token (linear)."""
    head_dim = int(CONFIG["head_dim"])
    n_kv_h = {t: int(CONFIG["linear_shape"][CONFIG[f"{t}_linear"]][0]) // head_dim for t in ("k", "v")}
    out = None
    for t in ("k", "v"):
        bits = np.array([[e[0] for e in sv["q"][t]] for sv in kv_subnets], float)   # (N_kv, L)
        gs = np.array([[e[1] for e in sv["q"][t]] for sv in kv_subnets], float)
        prune = np.array([sv["p"][t] for sv in kvdim_subnets], float)               # (N_kvdim, L)
        eff = n_kv_h[t] * (head_dim - prune)                                        # (N_kvdim, L)
        mem = (bits / 8.0) @ eff.T                                                  # sum_l bits*eff/8
        coef = np.where(bits < 16, 1.0 / gs, 0.0)
        scale = (coef * 4.0) @ eff.T                                                # sum_l mask*eff/gs*4
        out = (mem + scale) if out is None else out + (mem + scale)
    return out  # (N_kv, N_kvdim), n_token=1


def count_feasible(w_mem, kv_sorted, lo, hi):
    """Pass-1 feasible count, exactly utils/select.py:_memory_block."""
    L = np.searchsorted(kv_sorted, lo - w_mem, side="left")
    R = np.searchsorted(kv_sorted, hi - w_mem, side="right")
    return int(np.maximum(R - L, 0).sum())


def main():
    print("Loading archives ...")
    w_subnets = load_subnets(W_EXPR)
    kv_subnets = load_subnets(KV_EXPR)
    kvdim_subnets = load_subnets(KVDIM_EXPR)
    w_mem = np.array([compute_weight_memory(a) for a in w_subnets], float)
    print(f"  |w|={len(w_subnets)}  |kv|={len(kv_subnets)}  |kvdim|={len(kvdim_subnets)}"
          f"  full product={len(w_subnets)*len(kv_subnets)*len(kvdim_subnets):.3e}")
    print(f"  w_mem range [{w_mem.min():.4e}, {w_mem.max():.4e}]")

    thresholds = [1e-3, 1e-4, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5]
    print("Computing KV cache base (n_token=1) ...")
    kv_base = cache_memory_base(kv_subnets, kvdim_subnets).ravel()  # (N_kv*N_kvdim,)

    # global tallies across ALL 48 (n_token, comp_obj) jobs
    g_max = {t: 0 for t in thresholds}
    g_min = {t: np.inf for t in thresholds}
    g_empty = {t: 0 for t in thresholds}

    for n_token, comp_list in COMP_OBJ_BY_NTOKEN.items():
        kv2d = kv_base * n_token
        kv_sorted = np.sort(kv2d)
        print(f"\n===== N_TOKEN={n_token} =====")
        print("thr".ljust(9) + "".join(f"{t:>12.0e}" for t in thresholds))
        gmax = {t: 0 for t in thresholds}
        gmin = {t: np.inf for t in thresholds}
        nemp = {t: 0 for t in thresholds}
        for co in comp_list:
            for t in thresholds:
                thr = co * t
                c = count_feasible(w_mem, kv_sorted, co - thr, co + thr)
                gmax[t] = max(gmax[t], c)
                gmin[t] = min(gmin[t], c)
                nemp[t] += (c == 0)
                g_max[t] = max(g_max[t], c)
                g_min[t] = min(g_min[t], c)
                g_empty[t] += (c == 0)
        print("max".ljust(9) + "".join(f"{gmax[t]:>12.2e}" for t in thresholds))
        print("min".ljust(9) + "".join(f"{gmin[t]:>12.2e}" for t in thresholds))
        print("#empty".ljust(9) + "".join(f"{nemp[t]:>12d}" for t in thresholds))

    print("\n========== GLOBAL (all 48 jobs) ==========")
    print("thr".ljust(12) + "".join(f"{t:>12.0e}" for t in thresholds))
    print("max".ljust(12) + "".join(f"{g_max[t]:>12.2e}" for t in thresholds))
    print("min".ljust(12) + "".join(f"{g_min[t]:>12.2e}" for t in thresholds))
    print("#empty/48".ljust(12) + "".join(f"{g_empty[t]:>12d}" for t in thresholds))
    print("max<=5e7?".ljust(12) + "".join(f"{('OK' if g_max[t] <= LAZY_MAX_FEASIBLE else 'FAIL'):>12}" for t in thresholds))
    print("\n(threshold = fraction of COMP_OBJ_VAL; current script uses 1e-3 = +-0.1%)")
    print("Need simultaneously: max<=5e7 AND #empty==0.")


if __name__ == "__main__":
    main()
