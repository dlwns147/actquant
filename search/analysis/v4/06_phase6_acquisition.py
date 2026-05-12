"""06_phase6_acquisition.py — severe falsification candidate acquisition.

Replaces the old Phase 6.  Implements all design decisions from the discussion:

  Pool definition
    R_i^(≤3) = Layer 1 ∪ Layer 2 ∪ Layer 3 per axis archive (NDS-peeled),
    each layer complexity-stratified capped at CAP_PER_LAYER.

  Baseline (3D Cartesian-PF envelope)
    structural Cartesian-PF subset: PF_W^(1) × PF_KV^(1) × PF_KVD^(1)
    f_C*(c) = pymoo NDS on 4D objective (RBF μ, wbits, kvbits, kvdim)
    restricted to the structural subset.  Predicted loss column uses
    RBF cubic+linear μ (Phase-1 winner — R² 0.9951 Llama / 0.9823 Qwen),
    NOT naive_add.

  Predictor
    RBF cubic+linear trained on 260510 *_rs50 (50 random 3-way samples).
    σ_conf = q_{0.95}( |LOOCV residual| ) — predictor-agnostic conformal noise.

  Acquisition score
    EVI_ε(a) = (B_ε − μ)·Φ(t) + σ_conf·ϕ(t),   t = (B_ε − μ)/σ_conf
    B_ε(a)   = f_C*(c(a)) − ε,   primary ε = 0.005.

  Bucket allocation (per model, total 90)
    B1. EVI top adversarial (50)
        Pool = off-structural-PF (≥ 1 axis in Layer 2-3).
        Score = EVI_{0.005}.  2-level stratification:
          (1) rank-tuple group fractions
              one-axis off    : 40 % (20)
              two-axis off    : 35 % (17 → trimmed to fit)
              three-axis off  : 25 % (13)
          (2) inside each group: top-EVI per (wbits × kvbits × kvdim) 3³=27 bucket
    B3. Paired projection (30 = 15 pair)
        15 EVI-top off-PF a's (allowed to overlap with B1)
        For each a, π(a) = nearest dominating Layer-1 PF point per axis
        (= structural Cartesian-PF projection).  Both a and π(a) are measured.
    B5. Low-P controls (10)
        Off-structural-PF candidates with P_{0.005}(a) < 0.10, complexity-
        stratified random.  Anchors the left bin of the calibration curve.

  Archive-coverage slack
    δ_i^(3) = sup over a_i ∈ archive_i of
              min over b_i ∈ R_i^(≤3) of [z_i(b) − z_i(a)]_+
              subject to c_i(b) ≤ c_i(a) + ρ_i
    where ρ_i is a small complexity slack (~ local bin width).

Outputs per model:
    acquired_falsifiers_{tag}.json
    figures/v4_fig6_acquire_{tag}.png

Llama paths use `2604101639_*` archives, Qwen `2605031537/2605031614_*`.
"""
import os, sys, json, time, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from utils.func import get_net_info
from utils.select import build_arch
from predictor.rbf import RBF as PySOTRBF
from _common import load_csv, extract_xy, PATHS

BASE = '/NAS/SJ/actquant/search'
OUT  = f'{BASE}/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
MODELS = {
    'llama': dict(
        tag='llama', pretty='Llama-3.1-8B-Instruct',
        model_name='Llama-3.1-8B-Instruct',
        config_path=f'{BASE}/config/llama.json',
        w_stats=f'{BASE}/save/search/think/2604101639_Llama-3.1-8B-Instruct_wbits_w234kv4_hqq_kivi_i200n50_w128kv128_k0v0_r128_chto_2-5_jsd_wik_128s2048l0t_rbf_s512/iter_200.stats',
        kv_stats=f'{BASE}/save/search/think/2604101639_Llama-3.1-8B-Instruct_kvbits_w4kv234_hqq_kivi_i150n30_w128kv3264128x3_k0v0_r128_chto_1-5_jsd_wik_128s2048l0t_rbf_s512/iter_150.stats',
        kvdim_stats=f'{BASE}/save/search/think/2604101639_Llama-3.1-8B-Instruct_kvdim_w4kv4_hqq_think_i150n30_w128kv128_k0_16_32_48_64v0_r128_chto_0-128_jsd_wik_128s2048l0t_rbf_s512/iter_150.stats',
        rs50_csv=PATHS['llama_rs50'],
    ),
    'qwen': dict(
        tag='qwen', pretty='Qwen2.5-7B-Instruct',
        model_name='Qwen2.5-7B-Instruct',
        config_path=f'{BASE}/config/qwen2.json',
        w_stats=f'{BASE}/save/search/think/2605031537_Qwen2.5-7B-Instruct_wbits_w234kv4_hqq_kivi_i200n50_w128kv128_k0v0_r128_chto_2-5_jsd_wik_128s2048l0t_rbf_s512/iter_200.stats',
        kv_stats=f'{BASE}/save/search/think/2605031614_Qwen2.5-7B-Instruct_kvbits_w4kv234_hqq_kivi_i150n30_w128kv3264128x3_k0v0_r128_chto_1-5_jsd_wik_128s2048l0t_rbf_s512/iter_150.stats',
        kvdim_stats=f'{BASE}/save/search/think/2605031614_Qwen2.5-7B-Instruct_kvdim_w4kv4_hqq_think_i150n30_w128kv128_k0_16_32_48_64v0_r128_chto_0-128_jsd_wik_128s2048l0t_rbf_s512/iter_150.stats',
        rs50_csv=PATHS['qwen_rs50'],
    ),
}

EPS_PRIMARY    = 0.005
CAP_PER_LAYER  = 100
N_LAYERS       = 3
N_B1, N_B3_PAIR, N_B5 = 50, 15, 10
RANK_TUPLE_FRACS = dict(one_off=0.40, two_off=0.35, three_off=0.25)
HEAD_DIM       = 128
RBF_KERNEL     = 'cubic'
RNG = np.random.RandomState(0)

# ── helpers ─────────────────────────────────────────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]; m = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < m: nd.append(i); m = F_s[i, 1]
    return order[nd]


def pareto_layers_2d(losses, comps, n_layers=N_LAYERS):
    """Layered 2D PF via O(n log n) sweep — equivalent to pymoo NDS in 2D."""
    F = np.column_stack([losses, comps])
    layers, remaining = [], np.arange(len(F))
    for _ in range(n_layers):
        if len(remaining) == 0: break
        front_local = pareto_front_2d(F[remaining])
        layers.append(remaining[front_local])
        remaining = np.setdiff1d(remaining, remaining[front_local])
    return layers, remaining


def cap_layer(layer, comps, cap):
    """Complexity-stratified subsample of a layer down to `cap` entries."""
    if len(layer) <= cap:
        return layer
    s = np.argsort(comps[layer])
    pick = np.linspace(0, len(layer) - 1, cap, dtype=int)
    return layer[s[pick]]


def build_layered_pool(losses, comps, n_layers=N_LAYERS, cap=CAP_PER_LAYER):
    """Return (pool_idx, layer_mask) where layer_mask[i] ∈ {1,2,3,...} indicates
    which layer each pool entry came from."""
    layers, _ = pareto_layers_2d(losses, comps, n_layers=n_layers)
    pool_parts = []
    layer_marks = []
    for L_i, layer in enumerate(layers, start=1):
        sel = cap_layer(layer, comps, cap)
        pool_parts.append(sel)
        layer_marks.append(np.full(len(sel), L_i, dtype=np.int8))
    return np.concatenate(pool_parts), np.concatenate(layer_marks)


def load_archive(stats_path, comp_key, config, group_size):
    t0 = time.time()
    with open(stats_path) as f: data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs = [v[0] for v in archive]
    losses = np.array([v[1] for v in archive])
    comps  = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
    print(f"    {os.path.basename(stats_path)[:50]}: "
          f"n={len(archs)} in {time.time()-t0:.1f}s", flush=True)
    return archs, losses, comps


def loocv_rbf_residuals(X, y, kernel=RBF_KERNEL):
    """LOO residuals on the train pool — used to derive conformal σ."""
    n = len(y); res = np.zeros(n)
    lb, ub = X.min(0), X.max(0)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        m = PySOTRBF(kernel=kernel, tail='linear', lb=lb, ub=ub)
        m.fit(X[mask], y[mask])
        res[i] = y[i] - float(m.predict(X[i:i+1]).ravel()[0])
    return res


def archive_slack_delta(archive_z, archive_c, pool_z, pool_c, rho):
    """δ_i^(K) = sup over archive_i of min over R_i^(≤K) of [z(b)-z(a)]_+
    subject to c(b) ≤ c(a) + ρ.   Returns scalar slack."""
    worst = 0.0
    for j in range(len(archive_z)):
        a_z, a_c = archive_z[j], archive_c[j]
        valid = pool_c <= a_c + rho
        if not valid.any():
            continue
        gap = np.maximum(0.0, pool_z[valid] - a_z).min()
        if gap > worst:
            worst = gap
    return float(worst)


def fC_star_3D(c_query, pf_pts):
    """3D Cartesian-PF envelope: for query complexity (wb, kb, kd), return the
    minimum μ among PF points that dominate it in all 3 complexity dims.
    Falls back to max μ in the PF if no dominating point exists (= corner)."""
    wb, kb, kd = c_query[..., 0], c_query[..., 1], c_query[..., 2]
    out = np.full(wb.shape, pf_pts[:, 0].max(), dtype=float)
    for i in range(len(out)):
        cover = ((pf_pts[:, 1] <= wb[i]) &
                 (pf_pts[:, 2] <= kb[i]) &
                 (pf_pts[:, 3] <= kd[i]))
        if cover.any():
            out[i] = pf_pts[cover, 0].min()
    return out


def compute_local_pf_projection(z_a, c_a, pf_z, pf_c):
    """π_i(a_i) = nearest Layer-1 PF point with z_i ≤ z_i(a_i) AND c_i ≤ c_i(a_i).
    If none exists, returns the PF point with smallest (z + c) gap.
    Returns archive_index_within_PF1."""
    valid = (pf_z <= z_a + 1e-9) & (pf_c <= c_a + 1e-9)
    if valid.any():
        cand_idx = np.where(valid)[0]
        # pick the one with smallest distance in (z, c) — typically the one closest
        d = np.hypot(pf_z[cand_idx] - z_a, pf_c[cand_idx] - c_a)
        return int(cand_idx[np.argmin(d)])
    # fallback: nearest in (z, c)
    d = np.hypot(pf_z - z_a, pf_c - c_a)
    return int(np.argmin(d))


def bcast_1d(arr, axis, shape):
    s = [1] * len(shape); s[axis] = len(arr)
    return np.broadcast_to(arr.reshape(s), shape)


def stratified_top_k_evi(scores, comp_cols, group_mask, n_target, nb=(3, 3, 3)):
    """Inside `group_mask`, bucket by 3D complexity quantile grid and pick top-EVI
    per bucket, fill to n_target by round-robin top-EVI overall."""
    if group_mask.sum() == 0 or n_target == 0:
        return []
    sel_pool = np.where(group_mask)[0]
    # 3D bucket edges from group's complexity distribution
    edges = []
    for k in range(3):
        eg = np.quantile(comp_cols[k][sel_pool], np.linspace(0, 1, nb[k] + 1))
        eg[0] -= 1e-9; eg[-1] += 1e-9
        edges.append(eg)
    def bid(i):
        bw = max(0, min(nb[0]-1, int(np.searchsorted(edges[0], comp_cols[0][i], side='right') - 1)))
        bk = max(0, min(nb[1]-1, int(np.searchsorted(edges[1], comp_cols[1][i], side='right') - 1)))
        bd = max(0, min(nb[2]-1, int(np.searchsorted(edges[2], comp_cols[2][i], side='right') - 1)))
        return (bw * nb[1] + bk) * nb[2] + bd
    n_buckets = int(np.prod(nb))
    buckets = [[] for _ in range(n_buckets)]
    for i in sel_pool:
        buckets[bid(i)].append(int(i))
    for b in range(n_buckets):
        buckets[b].sort(key=lambda j: -scores[j])
    # round-robin pick
    sel = []
    round_idx = 0
    while len(sel) < n_target and any(len(b) > round_idx for b in buckets):
        for b in buckets:
            if round_idx < len(b) and len(sel) < n_target:
                sel.append(b[round_idx])
        round_idx += 1
    return sel[:n_target]


# ── per-model run ──────────────────────────────────────────────────────────
def run_model(cfg):
    tt = time.time()
    tag = cfg['tag']
    print("=" * 100, flush=True)
    print(f"Phase 6 acquisition — {cfg['pretty']}", flush=True)
    print("=" * 100, flush=True)

    with open(cfg['config_path']) as f:
        config = json.load(f)[cfg['model_name']]
    group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

    # 1) Archives ----------------------------------------------------------
    W_archs,  W_losses,  W_comps   = load_archive(cfg['w_stats'],     'wbits',  config, group_size)
    KV_archs, KV_losses, KV_comps  = load_archive(cfg['kv_stats'],    'kvbits', config, group_size)
    KVD_archs,KVD_losses,KVD_comps = load_archive(cfg['kvdim_stats'], 'kvdim',  config, group_size)
    print(f"  archives: |W|={len(W_archs)} |KV|={len(KV_archs)} |KVDIM|={len(KVD_archs)}  "
          f"[t={time.time()-tt:.1f}s]", flush=True)

    # 2) Layered pools (R^(≤3)) --------------------------------------------
    W_pool,   W_layer   = build_layered_pool(W_losses,   W_comps)
    KV_pool,  KV_layer  = build_layered_pool(KV_losses,  KV_comps)
    KVD_pool, KVD_layer = build_layered_pool(KVD_losses, KVD_comps)
    NW, NKV, NKVD = len(W_pool), len(KV_pool), len(KVD_pool)
    N_pool = NW * NKV * NKVD
    print(f"  pools  R^(≤3): W={NW} (L1={int((W_layer==1).sum())}/L2={int((W_layer==2).sum())}/L3={int((W_layer==3).sum())})  "
          f"KV={NKV} (L1={int((KV_layer==1).sum())}/...)  KVD={NKVD}  "
          f"Cart={N_pool:,}", flush=True)

    # 3) RBF cubic + LOO conformal σ ---------------------------------------
    X_rs, y_rs, _, _ = extract_xy(load_csv(cfg['rs50_csv']))
    lb_rs, ub_rs = X_rs.min(0), X_rs.max(0)
    loo_res = loocv_rbf_residuals(X_rs, y_rs, kernel=RBF_KERNEL)
    σ_conf = float(np.quantile(np.abs(loo_res), 0.95))
    print(f"  RS50 N={len(y_rs)}  LOO σ_conf (q_0.95 of |resid|) = {σ_conf:.5f}  "
          f"[t={time.time()-tt:.1f}s]", flush=True)

    # 4) ND-broadcast candidate space --------------------------------------
    nd_shape = (NW, NKV, NKVD)
    W_loss_p = W_losses[W_pool]; W_wbits_p = W_comps[W_pool]
    KV_loss_p = KV_losses[KV_pool]; KV_kvbits_p = KV_comps[KV_pool]
    KVD_loss_p = KVD_losses[KVD_pool]; KVD_kvdim_p = KVD_comps[KVD_pool]

    loss_W_nd   = bcast_1d(W_loss_p,   0, nd_shape)
    loss_KV_nd  = bcast_1d(KV_loss_p,  1, nd_shape)
    loss_KVD_nd = bcast_1d(KVD_loss_p, 2, nd_shape)
    wbits_nd  = bcast_1d(W_wbits_p,    0, nd_shape).astype(float)
    kvbits_nd = bcast_1d(KV_kvbits_p,  1, nd_shape).astype(float)
    kvdim_nd  = bcast_1d(KVD_kvdim_p,  2, nd_shape).astype(float)

    X_cand = np.column_stack([np.asarray(loss_W_nd).ravel(),
                              np.asarray(loss_KV_nd).ravel(),
                              np.asarray(loss_KVD_nd).ravel()])
    wbits_flat  = np.asarray(wbits_nd).ravel()
    kvbits_flat = np.asarray(kvbits_nd).ravel()
    kvdim_flat  = np.asarray(kvdim_nd).ravel()

    # 5) RBF μ over pool ----------------------------------------------------
    lb_ext = np.minimum(lb_rs, X_cand.min(0))
    ub_ext = np.maximum(ub_rs, X_cand.max(0))
    m_rbf = PySOTRBF(kernel=RBF_KERNEL, tail='linear', lb=lb_ext, ub=ub_ext)
    m_rbf.fit(X_rs, y_rs)
    μ_cand = m_rbf.predict(X_cand).ravel()
    print(f"  RBF μ range = [{μ_cand.min():.4f}, {μ_cand.max():.4f}]  "
          f"[t={time.time()-tt:.1f}s]", flush=True)

    # 6) Structural Cartesian-PF baseline (Layer-1 only, NDS on μ, wb, kb, kd) -
    W_l1_mask   = bcast_1d((W_layer  == 1), 0, nd_shape)
    KV_l1_mask  = bcast_1d((KV_layer == 1), 1, nd_shape)
    KVD_l1_mask = bcast_1d((KVD_layer== 1), 2, nd_shape)
    struct_mask_flat = (np.asarray(W_l1_mask) & np.asarray(KV_l1_mask) &
                        np.asarray(KVD_l1_mask)).ravel()
    print(f"  structural Cartesian-PF size = {struct_mask_flat.sum():,}", flush=True)

    F_struct = np.column_stack([μ_cand[struct_mask_flat],
                                wbits_flat[struct_mask_flat],
                                kvbits_flat[struct_mask_flat],
                                kvdim_flat[struct_mask_flat]])
    pf_idx_local = NonDominatedSorting().do(F_struct, only_non_dominated_front=True)
    pf_pts = F_struct[pf_idx_local]
    print(f"  baseline PF (NDS on 4D, structural only): n={len(pf_pts)}  "
          f"[t={time.time()-tt:.1f}s]", flush=True)

    # 7) EVI + P_ε for all OFF-structural-PF candidates -------------------
    off_struct = ~struct_mask_flat
    fC_star = fC_star_3D(np.column_stack([wbits_flat, kvbits_flat, kvdim_flat]), pf_pts)
    B_eps = fC_star - EPS_PRIMARY
    diff = B_eps - μ_cand
    t_norm = diff / max(σ_conf, 1e-9)
    EVI = diff * norm.cdf(t_norm) + σ_conf * norm.pdf(t_norm)
    P_eps = norm.cdf(t_norm)
    EVI[~off_struct] = -np.inf      # never pick on-structural-PF as falsifier
    P_eps[~off_struct] = 0.0
    print(f"  off-struct count = {int(off_struct.sum()):,};  "
          f"EVI max = {EVI[off_struct].max():.5f},  P_ε max = {P_eps[off_struct].max():.4f}  "
          f"[t={time.time()-tt:.1f}s]", flush=True)

    # rank-tuple counts per candidate
    W_layer_nd   = np.asarray(bcast_1d(W_layer.astype(np.int8),   0, nd_shape)).ravel()
    KV_layer_nd  = np.asarray(bcast_1d(KV_layer.astype(np.int8),  1, nd_shape)).ravel()
    KVD_layer_nd = np.asarray(bcast_1d(KVD_layer.astype(np.int8), 2, nd_shape)).ravel()
    off_count = ((W_layer_nd > 1).astype(int) + (KV_layer_nd > 1).astype(int) +
                 (KVD_layer_nd > 1).astype(int))   # 0/1/2/3

    # 8) B1 — EVI-top with rank-tuple × complexity stratification -----------
    comp_cols = [wbits_flat, kvbits_flat, kvdim_flat]
    n_per_group = {1: int(N_B1 * RANK_TUPLE_FRACS['one_off']),     # 20
                   2: int(N_B1 * RANK_TUPLE_FRACS['two_off']),     # 17
                   3: N_B1 - 0}                                     # placeholder
    n_per_group[3] = N_B1 - n_per_group[1] - n_per_group[2]        # 13
    B1_sel = []
    for k_off in (1, 2, 3):
        mask_grp = off_struct & (off_count == k_off)
        picked = stratified_top_k_evi(EVI, comp_cols, mask_grp, n_per_group[k_off])
        B1_sel.extend(picked)
    B1_sel = list(dict.fromkeys(B1_sel))[:N_B1]
    print(f"  B1 EVI-top: {len(B1_sel)} (target {N_B1})  "
          f"groups [1,2,3]-axis-off = "
          f"[{sum(off_count[i]==1 for i in B1_sel)}, "
          f"{sum(off_count[i]==2 for i in B1_sel)}, "
          f"{sum(off_count[i]==3 for i in B1_sel)}]", flush=True)

    # 9) B3 — paired projection (15 pairs) ---------------------------------
    # Pick 15 distinct off-PF a's with highest EVI (may overlap with B1).
    off_struct_idx = np.where(off_struct)[0]
    sorted_off = off_struct_idx[np.argsort(-EVI[off_struct_idx])]
    B3_a_sel = list(sorted_off[:N_B3_PAIR])
    print(f"  B3 paired (a side): {len(B3_a_sel)} off-PF candidates", flush=True)

    # Compute π(a) per axis: project each axis sub-arch to nearest Layer-1 PF
    # entry dominating in (z, c).
    W_l1_idx_in_W_pool   = np.where(W_layer   == 1)[0]
    KV_l1_idx_in_KV_pool = np.where(KV_layer  == 1)[0]
    KVD_l1_idx_in_KVD_pool = np.where(KVD_layer == 1)[0]
    # archive index of each layer-1 point
    W_l1_arch = W_pool[W_l1_idx_in_W_pool]
    KV_l1_arch = KV_pool[KV_l1_idx_in_KV_pool]
    KVD_l1_arch = KVD_pool[KVD_l1_idx_in_KVD_pool]

    B3_pi_sel = []   # (a, pi(a)) pairs as pool-flat indices
    for a_flat in B3_a_sel:
        i_W = a_flat // (NKV * NKVD)
        i_KV = (a_flat // NKVD) % NKV
        i_KVD = a_flat % NKVD
        # Per-axis projection
        pj_W = compute_local_pf_projection(
            W_loss_p[i_W], W_wbits_p[i_W],
            W_losses[W_l1_arch], W_comps[W_l1_arch])
        pj_KV = compute_local_pf_projection(
            KV_loss_p[i_KV], KV_kvbits_p[i_KV],
            KV_losses[KV_l1_arch], KV_comps[KV_l1_arch])
        pj_KVD = compute_local_pf_projection(
            KVD_loss_p[i_KVD], KVD_kvdim_p[i_KVD],
            KVD_losses[KVD_l1_arch], KVD_comps[KVD_l1_arch])
        # convert local PF index → pool flat index of projected combo
        pj_W_pool   = int(W_l1_idx_in_W_pool[pj_W])
        pj_KV_pool  = int(KV_l1_idx_in_KV_pool[pj_KV])
        pj_KVD_pool = int(KVD_l1_idx_in_KVD_pool[pj_KVD])
        pi_flat = (pj_W_pool * NKV + pj_KV_pool) * NKVD + pj_KVD_pool
        B3_pi_sel.append((int(a_flat), int(pi_flat)))
    print(f"  B3 paired (a, π(a)): {len(B3_pi_sel)} pairs", flush=True)

    # 10) B5 — Low-P controls (10) -----------------------------------------
    low_p_mask = off_struct & (P_eps < 0.10)
    print(f"  B5 low-P candidates available: {int(low_p_mask.sum())}", flush=True)
    low_p_idx = np.where(low_p_mask)[0]
    if len(low_p_idx) == 0:
        # fallback: lowest-P off-PF candidates
        low_p_idx = off_struct_idx[np.argsort(P_eps[off_struct_idx])[:max(10, N_B5)]]
    if len(low_p_idx) > N_B5:
        # complexity-stratified random in 3 wbits bins
        wb_lp = wbits_flat[low_p_idx]
        bins = np.digitize(wb_lp, np.quantile(wb_lp, [1/3, 2/3]))
        sel = []
        for b in range(3):
            cand = low_p_idx[bins == b]
            if len(cand) > 0:
                sel.extend(RNG.choice(cand, min(N_B5 // 3 + 1, len(cand)), replace=False).tolist())
        B5_sel = list(dict.fromkeys(sel))[:N_B5]
    else:
        B5_sel = list(low_p_idx)
    print(f"  B5 selected: {len(B5_sel)}", flush=True)

    # 11) Archive-coverage slack δ_i^(3) -----------------------------------
    delta = {}
    for axis_name, archive_z, archive_c, pool_idx in [
        ('W',     W_losses,   W_comps,   W_pool),
        ('KV',    KV_losses,  KV_comps,  KV_pool),
        ('KVDIM', KVD_losses, KVD_comps, KVD_pool),
    ]:
        rho = 0.02 * (archive_c.max() - archive_c.min())
        delta[axis_name] = archive_slack_delta(
            archive_z, archive_c, archive_z[pool_idx], archive_c[pool_idx], rho)
    print(f"  archive-coverage slack δ^(3) = "
          f"{ {k: f'{v:.5f}' for k, v in delta.items()} }", flush=True)

    # 12) Build arch dicts + serialise --------------------------------------
    n_block = config['n_block']; w_linears = config['linear']
    default_arch = {'q': {'w': {l: [4] * n_block for l in w_linears},
                          'k': [[4, 128]] * n_block, 'v': [[4, 128]] * n_block},
                    'p': {'k': [0] * n_block, 'v': [0] * n_block}}
    esm = {'w':     np.array([W_archs[i]   for i in W_pool],   dtype=object),
           'kv':    np.array([KV_archs[i]  for i in KV_pool],  dtype=object),
           'kvdim': np.array([KVD_archs[i] for i in KVD_pool], dtype=object)}
    expr_keys = ['w', 'kv', 'kvdim']

    def make_record(s_idx, bucket):
        i = s_idx // (NKV * NKVD); j = (s_idx // NKVD) % NKV; k = s_idx % NKVD
        arch = build_arch(default_arch, expr_keys, esm, np.array([i, j, k]))
        comp = get_net_info(arch, config, group_size)
        return dict(
            arch=arch, bucket=bucket,
            pool_idx=int(s_idx),
            w_pool_idx=int(i), kv_pool_idx=int(j), kvd_pool_idx=int(k),
            w_layer=int(W_layer[i]), kv_layer=int(KV_layer[j]), kvd_layer=int(KVD_layer[k]),
            w_archive_idx=int(W_pool[i]), kv_archive_idx=int(KV_pool[j]),
            kvd_archive_idx=int(KVD_pool[k]),
            sub_loss_W=float(W_loss_p[i]),
            sub_loss_KV=float(KV_loss_p[j]),
            sub_loss_KVD=float(KVD_loss_p[k]),
            rbf_mu=float(μ_cand[s_idx]),
            fC_star_3D=float(fC_star[s_idx]),
            EVI_eps0p005=float(EVI[s_idx]) if np.isfinite(EVI[s_idx]) else 0.0,
            P_eps0p005=float(P_eps[s_idx]),
            wbits=float(comp['wbits']),
            kvbits=float(comp['kvbits']),
            kvdim=float(comp['kvdim']),
            eff_kvbits=float(comp['eff_kvbits']),
        )

    records = []
    for s_idx in B1_sel:
        records.append(make_record(s_idx, 'B1_EVI_top'))
    for (a, pi) in B3_pi_sel:
        records.append(make_record(a, 'B3_paired_a'))
        records.append(make_record(pi, 'B3_paired_pi'))
    for s_idx in B5_sel:
        records.append(make_record(s_idx, 'B5_low_P_control'))

    # Dedupe by pool_idx across buckets (keep first appearance, preserve bucket label)
    seen = set(); deduped = []
    for r in records:
        if r['pool_idx'] in seen: continue
        seen.add(r['pool_idx']); deduped.append(r)
    print(f"  total candidates pre-dedupe: {len(records)};  unique: {len(deduped)}", flush=True)

    out_json = dict(
        model=cfg['model_name'],
        eps_primary=EPS_PRIMARY,
        n_pool=int(N_pool),
        n_structural_pf=int(struct_mask_flat.sum()),
        n_baseline_pf=int(len(pf_pts)),
        sigma_conf=σ_conf,
        rbf_kernel=RBF_KERNEL,
        cap_per_layer=CAP_PER_LAYER,
        n_layers=N_LAYERS,
        bucket_counts=dict(
            B1=int(sum(1 for r in deduped if r['bucket'] == 'B1_EVI_top')),
            B3_paired_a=int(sum(1 for r in deduped if r['bucket'] == 'B3_paired_a')),
            B3_paired_pi=int(sum(1 for r in deduped if r['bucket'] == 'B3_paired_pi')),
            B5_low_P_control=int(sum(1 for r in deduped if r['bucket'] == 'B5_low_P_control')),
        ),
        baseline_pf_obj_columns=['rbf_mu', 'wbits', 'kvbits', 'kvdim'],
        baseline_pf_points=pf_pts.tolist(),
        archive_slack_delta_3=delta,
        rs50_input_box=dict(min=lb_rs.tolist(), max=ub_rs.tolist()),
        b3_pair_map=[dict(a_pool_idx=a, pi_pool_idx=pi) for (a, pi) in B3_pi_sel],
        candidates=deduped,
    )
    out_path = f'{OUT}/acquired_falsifiers_{tag}.json'
    with open(out_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"  saved {out_path}  (n_candidates={len(deduped)})", flush=True)

    # 13) Quick figure -----------------------------------------------------
    sel_idx_arr = np.array([r['pool_idx'] for r in deduped])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [('wbits',  wbits_flat,  pf_pts[:, 1]),
              ('kvbits', kvbits_flat, pf_pts[:, 2]),
              ('kvdim',  kvdim_flat,  pf_pts[:, 3])]
    bucket_color = {'B1_EVI_top': 'C3', 'B3_paired_a': 'C1',
                    'B3_paired_pi': 'C0', 'B5_low_P_control': 'C2'}
    for ax, (name, x_all, x_pf) in zip(axes, panels):
        ax.scatter(x_all[off_struct], μ_cand[off_struct],
                   s=2, alpha=0.04, color='gray')
        ax.scatter(x_pf, pf_pts[:, 0],
                   s=8, color='C0', alpha=0.5,
                   label=f'baseline PF projection (n={len(pf_pts)})')
        for b, c in bucket_color.items():
            idx_b = [r['pool_idx'] for r in deduped if r['bucket'] == b]
            if idx_b:
                ax.scatter(x_all[idx_b], μ_cand[idx_b], s=22, color=c,
                           edgecolor='k', linewidth=0.4,
                           label=f'{b} (n={len(idx_b)})')
        ax.set_xlabel(name); ax.set_ylabel('predicted loss')
        ax.set_title(f'loss vs {name}')
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    plt.suptitle(f"Phase 6 acquisition — {cfg['pretty']}  "
                 f"(RBF cubic, σ_conf={σ_conf:.4f}, ε=0.005)",
                 fontsize=10)
    plt.tight_layout()
    fig_path = f"{FIGDIR}/v4_fig6_acquire_{tag}.png"
    plt.savefig(fig_path, dpi=140, bbox_inches='tight'); plt.close()
    print(f"  saved figure {fig_path}  [total t={time.time()-tt:.1f}s]", flush=True)


if __name__ == '__main__':
    for tag, cfg in MODELS.items():
        run_model(cfg)
    print("\nPhase 6 done.", flush=True)
