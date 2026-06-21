"""Verify/decompose the expr_front achievable memory ceiling at n_token=131072.

Replicates load_expr's per-axis Pareto filter (NonDominatedSorting on
(metric, comp_key)) + build_nd's dense memory (compute_weight_memory +
compute_cache_memory_batch), to explain why post_search reports
achievable memory max = 9876021248 (508671.out) and why --expr_front
LOWERS the ceiling vs the full archive.
"""
import json, os
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(BASE, "config/llama.json")))["Llama-3.1-8B-Instruct"]
GS = {"w": 128, "k": 128, "v": 128}
N_BLOCK = CFG["n_block"]
HEAD_DIM = int(CFG["head_dim"])
N_TOKEN = 131072

W="save/search/think/2605112032_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_200.stats"
KV="save/search/think/2605112033_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x2_128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_100.stats"
KVD="save/search/think/2605112036_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2560seq_0token_rbf_128stride_pp512/iter_150.stats"


def load(path):
    rj = json.load(open(os.path.join(BASE, path)))
    arch = rj["archive"] + rj["candidates"]
    subnets = [v[0] for v in arch]
    metric = np.array([v[1] for v in arch], float)
    return subnets, metric


def weight_mem(subnets):
    wgs = GS["w"]
    out = np.zeros(len(subnets))
    for li in CFG["linear"]:
        o, ic = map(int, CFG["linear_shape"][li])
        lgs = ic if wgs == -1 else wgs
        bits = np.array([s["q"]["w"][li] for s in subnets], float)  # (N,L)
        m = o * ic * bits // 8
        m += np.where(bits < 16, (ic // lgs) * o * 4, 0.0)
        out += m.sum(1)
    out += int(CFG["vocab_size"]) * int(CFG["hidden_size"]) * 4
    out += int(CFG["n_norm"]) * int(CFG["hidden_size"]) * 2
    out += int(CFG["max_position_embeddings"]) * HEAD_DIM * 2
    return out


def wbits(subnets):
    wgs = GS["w"]
    num = np.zeros(len(subnets)); npar = 0
    for li in CFG["linear"]:
        o, ic = map(int, CFG["linear_shape"][li])
        lgs = ic if wgs == -1 else wgs
        bits = np.array([s["q"]["w"][li] for s in subnets], float)
        npar += o * ic * bits.shape[1]
        mu = o * ic * bits + np.where(bits < 16, (ic // lgs) * o * 32, 0.0)
        num += mu.sum(1)
    return num / npar


def _kv_arrays(subnets, t):
    bits = np.array([[e[0] for e in s["q"][t]] for s in subnets], float)
    gs = np.array([[e[1] for e in s["q"][t]] for s in subnets], float)
    return bits, gs


def kvbits(subnets):  # target='kv', include_pruning=False
    bk, gk = _kv_arrays(subnets, "k"); bv, gv = _kv_arrays(subnets, "v")
    bk2 = bk + np.where(gk != 0, 32.0 / gk, 0.0)
    bv2 = bv + np.where(gv != 0, 32.0 / gv, 0.0)
    return np.concatenate([bk2, bv2], 1).mean(1)


def kvdim(subnets):
    pk = np.array([s["p"]["k"] for s in subnets], float)
    pv = np.array([s["p"]["v"] for s in subnets], float)
    return np.concatenate([HEAD_DIM - pk, HEAD_DIM - pv], 1).mean(1)


def cache_mem_2d(kv_subnets, kvdim_subnets):  # n_token=131072, build_nd formula
    n_kv_h = {t: int(CFG["linear_shape"][CFG[f"{t}_linear"]][0]) // HEAD_DIM for t in ("k", "v")}
    out = None
    for t in ("k", "v"):
        b, g = _kv_arrays(kv_subnets, t)
        prune = np.array([s["p"][t] for s in kvdim_subnets], float)
        eff = n_kv_h[t] * (HEAD_DIM - prune)
        mem = (b / 8.0) @ eff.T
        scale = (np.where(b < 16, 1.0 / g, 0.0) * 4.0) @ eff.T
        out = (mem + scale) if out is None else out + (mem + scale)
    return out * N_TOKEN  # (N_kv, N_kvdim)


def front(metric, comp):
    F = np.column_stack([metric, comp])
    idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return idx


def main():
    ws, wm = load(W); kvs, kvm = load(KV); kds, kdm = load(KVD)
    wmem = weight_mem(ws)
    wb, kvb, kd = wbits(ws), kvbits(kvs), kvdim(kds)
    cache = cache_mem_2d(kvs, kds)  # (N_kv, N_kvdim)

    print(f"|w|={len(ws)} |kv|={len(kvs)} |kvdim|={len(kds)}  n_token={N_TOKEN}")
    print(f"weight_mem: min {wmem.min():.4e}  max {wmem.max():.4e}")
    print(f"cache_mem : min {cache.min():.4e}  max {cache.max():.4e}")

    # ---- FULL archive (no expr_front) ----
    full_max = wmem.max() + cache.max()
    full_min = wmem.min() + cache.min()
    print(f"\n[FULL]       memory min {full_min:.6e}  max {full_max:.0f}")

    # ---- expr_front: per-axis Pareto front on (metric, comp_key) ----
    fw = front(wm, wb); fkv = front(kvm, kvb); fkd = front(kdm, kd)
    wmem_f = wmem[fw]
    cache_f = cache[np.ix_(fkv, fkd)]
    ef_max = wmem_f.max() + cache_f.max()
    ef_min = wmem_f.min() + cache_f.min()
    print(f"[EXPR_FRONT] front sizes  w {len(fw)}  kv {len(fkv)}  kvdim {len(fkd)}")
    print(f"[EXPR_FRONT] weight_mem max {wmem_f.max():.4e} (full {wmem.max():.4e})")
    print(f"[EXPR_FRONT] cache_mem  max {cache_f.max():.4e} (full {cache.max():.4e})")
    print(f"[EXPR_FRONT] memory min {ef_min:.6e}  max {ef_max:.0f}")
    print(f"\nProgram reported (508671.out): min 5910306816  max 9876021248")
    print(f"Requested window top of 131072 list = 9938149376 (+-0.1%)")
    print(f"  -> 9938149376 > expr_front max {ef_max:.0f}? {9938149376 > ef_max}")
    print(f"  -> 9938149376 > full       max {full_max:.0f}? {9938149376 > full_max}")

    # which corner is lost: max-memory subnet on each axis on the front?
    print(f"\nmax-weight subnet on w-front? {wmem.argmax() in set(fw.tolist())}")
    iflat = cache.argmax(); ikv, ikd = np.unravel_index(iflat, cache.shape)
    print(f"max-cache (kv={ikv},kvdim={ikd}) kept? kv {ikv in set(fkv.tolist())}  kvdim {ikd in set(fkd.tolist())}")


if __name__ == "__main__":
    main()
