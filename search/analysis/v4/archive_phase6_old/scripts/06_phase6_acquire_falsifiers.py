"""06_phase6_acquire_falsifiers.py — acquire Cartesian-combined-PF falsification
candidates (CPU side).

Runs in **two complexity dimensionalities** per model:
  - 4D: F = (loss, wbits, kvbits, kvdim)           — 3D complexity preserved
  - 3D: F = (loss, wbits, eff_kvbits)              — eff_kvbits = kvbits·kvdim/128

Per model and per mode, saves:
  acquired_falsifiers_{tag}_{mode}.json
  figures/v4_fig6_acquire_{tag}_{mode}.png
where mode ∈ {'4d', '3d'}.

Pipeline (shared between modes):
  1. Load per-axis search archives (W, KV, KVDIM).
  2. Build per-axis pool: PF (complexity-stratified) + Pareto layers 2-3 +
     dominated random.
  3. Train RBF tps+linear on 260510 *_rs50 (X = z_W, z_KV, z_KVD; y = row 12).
     Estimate residual σ via leave-one-out.
  4. Build Cartesian candidate space via ND-broadcasting (post_search_split.py
     pattern — no materialised meshgrid).
  5. RBF-predict μ for every pool point (single materialisation).

Per-mode:
  6. Build the baseline PF (NonDominatedSorting on F_baseline).
  7. For each off-PF candidate, vectorised dominance check with RBF μ:
     a candidate is a *predicted falsifier* iff (μ, complexity*) is not
     dominated by any baseline-PF point.
     Score = max improvement over PF points with worse-or-equal complexity
     in all complexity dims. P(falsifier) ≈ Φ(score / σ_resid).
  8. Bucket-stratify in complexity space, top-K per bucket; fill with top-P
     remaining; trim to N_BUDGET total.
  9. Build merged arch dicts, save JSON + figure (2D projections — never
     collapse complexity to 1D).
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

from utils.func import get_net_info, compute_eff_kvbits_batch
from utils.select import build_arch
from predictor.rbf import RBF as PySOTRBF
from _common import load_csv, extract_xy, PATHS

# ─── Config ─────────────────────────────────────────────────────────────────
BASE = '/NAS/SJ/actquant/search'
OUT  = f'{BASE}/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

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

N_BUDGET     = 50
TOP_K_PER_AXIS = 30
HEAD_DIM = 128
RNG = np.random.RandomState(0)

# Bucket grids per mode (3D mode: 2 complexity axes; 4D mode: 3 complexity axes)
BUCKETS_4D = dict(nb_w=3, nb_kv=3, nb_kvdim=3, n_per_bucket=2)   # 27 buckets × 2
BUCKETS_3D = dict(nb_w=4, nb_eff=4,           n_per_bucket=4)   # 16 buckets × 4


# ─── 2D Pareto helpers (for per-axis archive PF) ────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]; m = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < m: nd.append(i); m = F_s[i, 1]
    return order[nd]


def pareto_layers_2d(losses, comps, n_layers=3):
    F = np.column_stack([losses, comps])
    layers, remaining = [], np.arange(len(F))
    for _ in range(n_layers):
        if len(remaining) == 0: break
        front_local = pareto_front_2d(F[remaining])
        layers.append(remaining[front_local])
        remaining = np.setdiff1d(remaining, remaining[front_local])
    return layers, remaining


def select_axis_pool(losses, comps, K_total=30, K_pf=15, K_subpf=10, K_dom=5):
    layers, dominated = pareto_layers_2d(losses, comps, n_layers=3)
    pf = layers[0]
    sub = np.concatenate(layers[1:]) if len(layers) > 1 else np.array([], dtype=int)
    if len(pf) <= K_pf:
        pf_sel = pf
    else:
        s = np.argsort(comps[pf])
        pick = np.linspace(0, len(pf) - 1, K_pf, dtype=int)
        pf_sel = pf[s[pick]]
    sub_sel = sub if len(sub) <= K_subpf else RNG.choice(sub, K_subpf, replace=False)
    dom_sel = dominated if len(dominated) <= K_dom else RNG.choice(dominated, K_dom, replace=False)
    return np.unique(np.concatenate([pf_sel, sub_sel, dom_sel]))[:K_total]


def load_archive(stats_path, comp_key, config, group_size):
    t0 = time.time()
    with open(stats_path) as f:
        data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs = [v[0] for v in archive]
    losses = np.array([v[1] for v in archive])
    comps  = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
    print(f"    load_archive({os.path.basename(stats_path)[:50]}): "
          f"n={len(archs)} in {time.time()-t0:.1f}s", flush=True)
    return archs, losses, comps


def loocv_rbf_sigma(X, y, kernel='cubic'):
    n = len(y); res = np.zeros(n)
    lb, ub = X.min(0), X.max(0)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        m = PySOTRBF(kernel=kernel, tail='linear', lb=lb, ub=ub)
        m.fit(X[mask], y[mask])
        res[i] = y[i] - float(m.predict(X[i:i+1]).ravel()[0])
    return float(np.std(res)), res


def bcast_1d(arr, axis, nd_shape):
    shape = [1] * len(nd_shape); shape[axis] = len(arr)
    return np.broadcast_to(arr.reshape(shape), nd_shape)


# ─── Per-mode PF + acquisition (shared logic) ───────────────────────────────
def acquire_falsifiers(*, mode, naive_add_flat, mu_cand, comp_cols, comp_names,
                       sigma_resid, N_pool, n_budget, bucket_cfg, tt):
    """
    Build PF, vectorised dominance check, bucket-stratified acquisition.

    Parameters
    ----------
    mode : '4d' or '3d'
    naive_add_flat, mu_cand : (N_pool,) arrays  — predicted loss baselines
    comp_cols : list of (N_pool,) arrays for each complexity column
        4d: [wbits, kvbits, kvdim]
        3d: [wbits, eff_kvbits]
    comp_names : matching list of names
    bucket_cfg : dict with per-axis bucket counts + n_per_bucket
    """
    n_comp = len(comp_cols)
    # 1) Baseline PF
    F_baseline = np.column_stack([naive_add_flat, *comp_cols])
    t_nds = time.time()
    pf_idx = NonDominatedSorting().do(F_baseline, only_non_dominated_front=True)
    print(f"  [{mode}] PF (pymoo NDS): n={len(pf_idx)}  in {time.time()-t_nds:.1f}s  "
          f"[t={time.time()-tt:.1f}s]", flush=True)
    on_pf = np.zeros(N_pool, dtype=bool); on_pf[pf_idx] = True
    pf_pts = F_baseline[pf_idx]                           # (P, 1+n_comp)

    # 2) Vectorised dominance: (μ, comp_cols) for each candidate vs all PF pts
    cand_pts = np.column_stack([mu_cand, *comp_cols])
    off_idx = np.where(~on_pf)[0]
    score_improve = np.zeros(N_pool)
    not_dominated = np.zeros(N_pool, dtype=bool)
    chunk = 4000
    for s in range(0, len(off_idx), chunk):
        e = min(s + chunk, len(off_idx))
        idx_c = off_idx[s:e]
        cand = cand_pts[idx_c]
        le = np.all(pf_pts[None, :, :] <= cand[:, None, :], axis=2)
        lt = np.any(pf_pts[None, :, :] <  cand[:, None, :], axis=2)
        not_dominated[idx_c] = ~np.any(le & lt, axis=1)
        # Worse-or-equal complexity (all complexity dims ≥ candidate's)
        worse_or_eq_c = np.ones((cand.shape[0], pf_pts.shape[0]), dtype=bool)
        for k in range(n_comp):
            worse_or_eq_c &= (pf_pts[None, :, 1 + k] >= cand[:, None, 1 + k])
        loss_diff = pf_pts[None, :, 0] - cand[:, None, 0]
        loss_diff = np.where(worse_or_eq_c, loss_diff, -np.inf)
        max_imp = np.max(loss_diff, axis=1)
        max_imp = np.where(np.isfinite(max_imp), max_imp, 0.0)
        score_improve[idx_c] = np.maximum(0.0, max_imp)
    print(f"  [{mode}] off-PF count={len(off_idx)}; predicted-not-dominated={int(not_dominated.sum())}; "
          f"max improvement={score_improve[off_idx].max():.4f}; "
          f"mean={score_improve[off_idx].mean():.5f}", flush=True)

    # 3) P(falsifier) under RBF residual σ
    prob_falsifier = norm.cdf(score_improve / max(sigma_resid, 1e-6))
    prob_falsifier[on_pf] = 0.0
    print(f"  [{mode}] P(falsifier): max={prob_falsifier[off_idx].max():.4f}, "
          f"mean={prob_falsifier[off_idx].mean():.4f}, "
          f"#(P>0.5)={int(np.sum(prob_falsifier[off_idx] > 0.5))}, "
          f"#(P>0.9)={int(np.sum(prob_falsifier[off_idx] > 0.9))}", flush=True)

    # 4) Bucket stratification across complexity axes
    nbs = []
    edges = []
    for arr, nb in zip(comp_cols, [bucket_cfg[k] for k in
                                    (['nb_w','nb_kv','nb_kvdim'] if n_comp == 3
                                     else ['nb_w','nb_eff'])]):
        eg = np.quantile(arr[off_idx], np.linspace(0, 1, nb + 1))
        eg[0] -= 1e-9; eg[-1] += 1e-9
        edges.append(eg); nbs.append(nb)
    def bid(values):
        b = 0
        for k, (v, eg, nb) in enumerate(zip(values, edges, nbs)):
            bk = max(0, min(nb - 1, int(np.searchsorted(eg, v, side='right') - 1)))
            b = b * nb + bk
        return b
    n_buckets = int(np.prod(nbs))
    n_per_bucket = bucket_cfg['n_per_bucket']
    buckets = [[] for _ in range(n_buckets)]
    for i in off_idx:
        if prob_falsifier[i] <= 0.0: continue
        buckets[bid([c[i] for c in comp_cols])].append(int(i))
    for b in range(n_buckets):
        buckets[b].sort(key=lambda j: -prob_falsifier[j])
    sel_main = []
    for b in range(n_buckets):
        sel_main.extend(buckets[b][:n_per_bucket])
    sel_main = list(dict.fromkeys(sel_main))
    remaining = np.setdiff1d(off_idx[prob_falsifier[off_idx] > 0.0],
                             np.array(sel_main, dtype=int))
    n_extra = n_budget - len(sel_main)
    extras = remaining[np.argsort(-prob_falsifier[remaining])[:n_extra]].tolist() \
        if n_extra > 0 else []
    selected = list(dict.fromkeys(sel_main + extras))[:n_budget]
    print(f"  [{mode}] selected {len(selected)} candidates "
          f"({len(sel_main)} stratified + {len(extras)} extras)  "
          f"[t={time.time()-tt:.1f}s]", flush=True)

    return dict(on_pf=on_pf, pf_idx=pf_idx, pf_pts=pf_pts,
                score_improve=score_improve, not_dominated=not_dominated,
                prob_falsifier=prob_falsifier, selected=selected,
                n_buckets=n_buckets, n_per_bucket=n_per_bucket,
                obj_columns=['naive_add'] + comp_names)


def save_outputs(*, model_name, mode, cfg, result, mu_cand, naive_add_flat,
                 comp_cols_flat, comp_names,
                 NW, NKV, NKVD, W_pool, KV_pool, KVD_pool,
                 W_archs, KV_archs, KVD_archs,
                 W_loss_p, KV_loss_p, KVD_loss_p,
                 config, group_size, sigma_resid, N_pool, lb_rs, ub_rs):
    on_pf       = result['on_pf']
    pf_pts      = result['pf_pts']
    selected    = result['selected']
    score_improve   = result['score_improve']
    not_dominated   = result['not_dominated']
    prob_falsifier  = result['prob_falsifier']

    # Build merged arch dicts
    n_block = config['n_block']; w_linears = config['linear']
    default_arch = {
        'q': {'w': {l: [4] * n_block for l in w_linears},
              'k': [[4, 128]] * n_block, 'v': [[4, 128]] * n_block},
        'p': {'k': [0] * n_block, 'v': [0] * n_block},
    }
    esm = {
        'w':     np.array([W_archs[i]   for i in W_pool],   dtype=object),
        'kv':    np.array([KV_archs[i]  for i in KV_pool],  dtype=object),
        'kvdim': np.array([KVD_archs[i] for i in KVD_pool], dtype=object),
    }
    expr_keys = ['w', 'kv', 'kvdim']
    records = []
    for s_idx in selected:
        i = s_idx // (NKV * NKVD)
        j = (s_idx // NKVD) % NKV
        k = s_idx % NKVD
        arch = build_arch(default_arch, expr_keys, esm, np.array([i, j, k]))
        comp = get_net_info(arch, config, group_size)
        records.append(dict(
            arch=arch,
            pool_idx=int(s_idx),
            w_pool_idx=int(i), kv_pool_idx=int(j), kvd_pool_idx=int(k),
            w_archive_idx=int(W_pool[i]), kv_archive_idx=int(KV_pool[j]),
            kvd_archive_idx=int(KVD_pool[k]),
            sub_loss_W=float(W_loss_p[i]),
            sub_loss_KV=float(KV_loss_p[j]),
            sub_loss_KVD=float(KVD_loss_p[k]),
            naive_add=float(naive_add_flat[s_idx]),
            rbf_mu=float(mu_cand[s_idx]),
            wbits=float(comp['wbits']),
            kvbits=float(comp['kvbits']),
            kvdim=float(comp['kvdim']),
            eff_kvbits=float(comp['eff_kvbits']),
            score_improve=float(score_improve[s_idx]),
            prob_falsifier=float(prob_falsifier[s_idx]),
            predicted_not_dominated=bool(not_dominated[s_idx]),
            on_pf=bool(on_pf[s_idx]),
        ))
    out_json = dict(
        model=model_name,
        mode=mode,
        n_pool=int(N_pool),
        n_pf_baseline=int(on_pf.sum()),
        sigma_resid_rbf=float(sigma_resid),
        n_budget=N_BUDGET,
        n_buckets=result['n_buckets'],
        n_per_bucket=result['n_per_bucket'],
        pf_obj_columns=result['obj_columns'],
        baseline_pf_points=pf_pts.tolist(),
        rs50_input_box=dict(min=lb_rs.tolist(), max=ub_rs.tolist()),
        candidates=records,
    )
    out_path = f'{OUT}/acquired_falsifiers_{cfg["tag"]}_{mode}.json'
    with open(out_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"  [{mode}] saved: {out_path}", flush=True)

    # Figure: 2D projections (loss vs each complexity axis), never collapse to 1D
    sel_arr = np.array(selected, dtype=int)
    n_panel = len(comp_cols_flat)
    fig, axes = plt.subplots(1, n_panel, figsize=(5.3 * n_panel, 5.0))
    if n_panel == 1: axes = [axes]
    for ax, x_all, x_pf, label in zip(axes, comp_cols_flat,
                                       [pf_pts[:, 1 + k] for k in range(n_panel)],
                                       comp_names):
        ax.scatter(x_all[~on_pf], naive_add_flat[~on_pf],
                   s=3, alpha=0.05, color='gray')
        ax.scatter(x_pf, pf_pts[:, 0], s=8, color='C0', alpha=0.6,
                   label=f'baseline PF projection (n={on_pf.sum()})')
        ax.scatter(x_all[sel_arr], mu_cand[sel_arr], s=24, color='C3',
                   edgecolor='k', linewidth=0.4, zorder=3,
                   label=f'candidates (RBF μ, n={len(sel_arr)})')
        ax.set_xlabel(label); ax.set_ylabel('predicted loss')
        ax.set_title(f"projection: loss vs {label}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    pf_obj_str = ', '.join(result['obj_columns'])
    plt.suptitle(f"Phase 6 [{mode.upper()}] — baseline PF on ({pf_obj_str});  "
                 f"RBF μ falsifier candidates (σ_resid_LOO={sigma_resid:.4f})  "
                 f"— {cfg['pretty']}", fontsize=10)
    plt.tight_layout()
    fig_path = f"{FIGDIR}/v4_fig6_acquire_{cfg['tag']}_{mode}.png"
    plt.savefig(fig_path, dpi=140, bbox_inches='tight'); plt.close()
    print(f"  [{mode}] saved figure: {fig_path}", flush=True)


def run_model(cfg):
    tt = time.time()
    print('=' * 100, flush=True)
    print(f"Phase 6 acquisition — {cfg['pretty']}", flush=True)
    print('=' * 100, flush=True)

    with open(cfg['config_path']) as f:
        config = json.load(f)[cfg['model_name']]
    group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

    # Per-axis archives + pools
    W_archs,   W_losses,   W_comps   = load_archive(cfg['w_stats'],     'wbits',  config, group_size)
    KV_archs,  KV_losses,  KV_comps  = load_archive(cfg['kv_stats'],    'kvbits', config, group_size)
    KVD_archs, KVD_losses, KVD_comps = load_archive(cfg['kvdim_stats'], 'kvdim',  config, group_size)
    print(f"  archives: |W|={len(W_archs)} |KV|={len(KV_archs)} |KVDIM|={len(KVD_archs)}  "
          f"[t={time.time()-tt:.1f}s]", flush=True)

    W_pool   = select_axis_pool(W_losses,   W_comps,   K_total=TOP_K_PER_AXIS)
    KV_pool  = select_axis_pool(KV_losses,  KV_comps,  K_total=TOP_K_PER_AXIS)
    KVD_pool = select_axis_pool(KVD_losses, KVD_comps, K_total=TOP_K_PER_AXIS)
    NW, NKV, NKVD = len(W_pool), len(KV_pool), len(KVD_pool)
    print(f"  pools: |W|={NW} |KV|={NKV} |KVDIM|={NKVD}  cartesian={NW*NKV*NKVD:,}  "
          f"[t={time.time()-tt:.1f}s]", flush=True)

    # RBF + σ
    X_rs, y_rs, _, _ = extract_xy(load_csv(cfg['rs50_csv']))
    print(f"  RS50: N={len(y_rs)}", flush=True)
    lb_rs, ub_rs = X_rs.min(0), X_rs.max(0)
    sigma_resid, _ = loocv_rbf_sigma(X_rs, y_rs, kernel='cubic')
    print(f"  RBF LOO residual σ = {sigma_resid:.5f}  [t={time.time()-tt:.1f}s]", flush=True)

    # ND-broadcast vectorised candidate space
    nd_shape = (NW, NKV, NKVD)
    W_loss_p   = W_losses[W_pool]
    KV_loss_p  = KV_losses[KV_pool]
    KVD_loss_p = KVD_losses[KVD_pool]
    W_wbits_p  = W_comps[W_pool]
    KV_kvbits_p = KV_comps[KV_pool]
    KVD_kvdim_p = KVD_comps[KVD_pool]

    loss_W_nd   = bcast_1d(W_loss_p,   0, nd_shape)
    loss_KV_nd  = bcast_1d(KV_loss_p,  1, nd_shape)
    loss_KVD_nd = bcast_1d(KVD_loss_p, 2, nd_shape)
    naive_add_nd = loss_W_nd + loss_KV_nd + loss_KVD_nd
    wbits_nd     = bcast_1d(W_wbits_p,    0, nd_shape).astype(float)
    kvbits_nd    = bcast_1d(KV_kvbits_p,  1, nd_shape).astype(float)
    kvdim_nd     = bcast_1d(KVD_kvdim_p,  2, nd_shape).astype(float)

    # eff_kvbits per (kv, kvdim) pair via utils.func.compute_eff_kvbits_batch:
    # mean_layers((bits + 32/gs) * (1 - prune/head_dim)) — NOT just kvbits*kvdim/128.
    # Returns shape (NKV, NKVD); broadcast over the W axis.
    kv_subnets  = [KV_archs[i]  for i in KV_pool]
    kvd_subnets = [KVD_archs[i] for i in KVD_pool]
    eff_kvbits_2d = compute_eff_kvbits_batch(kv_subnets, kvd_subnets, config, target='kv')  # (NKV, NKVD)
    eff_kvbits_nd = np.broadcast_to(
        eff_kvbits_2d[None, :, :], nd_shape).astype(float)

    N_pool = NW * NKV * NKVD
    X_cand = np.column_stack([np.asarray(loss_W_nd).ravel(),
                              np.asarray(loss_KV_nd).ravel(),
                              np.asarray(loss_KVD_nd).ravel()])
    naive_add_flat   = np.asarray(naive_add_nd).ravel()
    wbits_flat       = np.asarray(wbits_nd).ravel()
    kvbits_flat      = np.asarray(kvbits_nd).ravel()
    kvdim_flat       = np.asarray(kvdim_nd).ravel()
    eff_kvbits_flat  = np.asarray(eff_kvbits_nd).ravel()
    print(f"  ND broadcast built: N_pool={N_pool:,}  "
          f"eff_kvbits range=[{eff_kvbits_flat.min():.3f},{eff_kvbits_flat.max():.3f}]  "
          f"[t={time.time()-tt:.1f}s]", flush=True)

    # RBF predict (single materialisation, reused for both modes)
    lb_ext = np.minimum(lb_rs, X_cand.min(0)); ub_ext = np.maximum(ub_rs, X_cand.max(0))
    m_rbf = PySOTRBF(kernel='cubic', tail='linear', lb=lb_ext, ub=ub_ext)
    m_rbf.fit(X_rs, y_rs)
    mu_cand = m_rbf.predict(X_cand).ravel()
    print(f"  RBF μ: range=[{mu_cand.min():.4f}, {mu_cand.max():.4f}]  "
          f"[t={time.time()-tt:.1f}s]", flush=True)

    # ── 4D mode: (loss, wbits, kvbits, kvdim) ──────────────────────────────
    r4 = acquire_falsifiers(
        mode='4d',
        naive_add_flat=naive_add_flat, mu_cand=mu_cand,
        comp_cols=[wbits_flat, kvbits_flat, kvdim_flat],
        comp_names=['wbits', 'kvbits', 'kvdim'],
        sigma_resid=sigma_resid, N_pool=N_pool, n_budget=N_BUDGET,
        bucket_cfg=BUCKETS_4D, tt=tt,
    )
    save_outputs(model_name=cfg['model_name'], mode='4d', cfg=cfg, result=r4,
                 mu_cand=mu_cand, naive_add_flat=naive_add_flat,
                 comp_cols_flat=[wbits_flat, kvbits_flat, kvdim_flat],
                 comp_names=['wbits', 'kvbits', 'kvdim'],
                 NW=NW, NKV=NKV, NKVD=NKVD,
                 W_pool=W_pool, KV_pool=KV_pool, KVD_pool=KVD_pool,
                 W_archs=W_archs, KV_archs=KV_archs, KVD_archs=KVD_archs,
                 W_loss_p=W_loss_p, KV_loss_p=KV_loss_p, KVD_loss_p=KVD_loss_p,
                 config=config, group_size=group_size, sigma_resid=sigma_resid,
                 N_pool=N_pool, lb_rs=lb_rs, ub_rs=ub_rs)

    # ── 3D mode: (loss, wbits, eff_kvbits) ─────────────────────────────────
    r3 = acquire_falsifiers(
        mode='3d',
        naive_add_flat=naive_add_flat, mu_cand=mu_cand,
        comp_cols=[wbits_flat, eff_kvbits_flat],
        comp_names=['wbits', 'eff_kvbits'],
        sigma_resid=sigma_resid, N_pool=N_pool, n_budget=N_BUDGET,
        bucket_cfg=BUCKETS_3D, tt=tt,
    )
    save_outputs(model_name=cfg['model_name'], mode='3d', cfg=cfg, result=r3,
                 mu_cand=mu_cand, naive_add_flat=naive_add_flat,
                 comp_cols_flat=[wbits_flat, eff_kvbits_flat],
                 comp_names=['wbits', 'eff_kvbits'],
                 NW=NW, NKV=NKV, NKVD=NKVD,
                 W_pool=W_pool, KV_pool=KV_pool, KVD_pool=KVD_pool,
                 W_archs=W_archs, KV_archs=KV_archs, KVD_archs=KVD_archs,
                 W_loss_p=W_loss_p, KV_loss_p=KV_loss_p, KVD_loss_p=KVD_loss_p,
                 config=config, group_size=group_size, sigma_resid=sigma_resid,
                 N_pool=N_pool, lb_rs=lb_rs, ub_rs=ub_rs)

    # Overlap of 4D vs 3D selected candidate sets (sanity diagnostic)
    s4 = set(int(i) for i in r4['selected'])
    s3 = set(int(i) for i in r3['selected'])
    print(f"  candidate overlap |4D ∩ 3D|={len(s4 & s3)}  |4D \\ 3D|={len(s4 - s3)}  "
          f"|3D \\ 4D|={len(s3 - s4)}  [total t={time.time()-tt:.1f}s]", flush=True)


for tag, cfg in MODELS.items():
    run_model(cfg)

print("\nPhase 6 done (4D + 3D each for both models).", flush=True)
