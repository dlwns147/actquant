"""analysis_v3_ardrbf_cokriging.py — Try ARD-RBF and Co-Kriging extensions.

Compares 4 surrogates on the fixed 50-train / 150-test split:
  • Direct-RBF (pySOT cubic+linear, sorted-index input)         — current baseline
  • Direct-ARD-GP (sklearn anisotropic Matern/RBF, JSD input)   — new
  • Hier-WD-TPS-linear (current best hier with constrained pair)
  • Co-Kriging-WD (NARGP-style: low-fi pair-GP → augmented 3D-GP)

For total budgets B ∈ {27, 30, 35, 40, 45, 50}:
  • Direct: B 3-way samples (B pulled from train pool)
  • Hier / Co-Kriging: same (n3, n_pair) split as optimal hier from earlier sweep
"""
import sys, os, json, csv
sys.path.insert(0, '/NAS/SJ/actquant/search')

import warnings
warnings.simplefilter("ignore")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SKRBF, ConstantKernel as C, Matern, WhiteKernel
from utils.func import get_net_info
from predictor.rbf import RBF as PySOTRBF

# ─── Data loading ────────────────────────────────────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]; min2 = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < min2: nd.append(i); min2 = F_s[i, 1]
    return order[nd]
def load_archive_pareto(stats_path, comp_key, config, group_size):
    with open(stats_path) as f: data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs = [v[0] for v in archive]
    metrics = np.array([v[1] for v in archive])
    comps = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
    F = np.column_stack((metrics, comps))
    return F[pareto_front_2d(F)]
def load_csv(path):
    with open(path) as f: rows = [r for r in csv.reader(f) if r]
    max_cols = max(len(r) for r in rows)
    mat = np.full((len(rows), max_cols), np.nan)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            try: mat[i, j] = float(v)
            except: pass
    return mat
def match_metric(comp_vals, pf): return np.array([pf[np.argmin(np.abs(pf[:, 1] - c)), 0] for c in comp_vals])
def match_index(comp_vals, pf_metric_sorted): return np.array([np.argmin(np.abs(pf_metric_sorted[:, 1] - c)) for c in comp_vals])

BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
AWQ_W_KVD= f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim/results.csv'

with open(f'{BASE}/config/llama.json') as f: config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

print("Loading PFs and 3-way + WD pair data...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
pf_W_m, pf_KV_m, pf_KVDIM_m = (p[np.argsort(p[:, 0])] for p in (pf_W, pf_KV, pf_KVDIM))

mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y3   = mat3[12, :N0]; v3 = ~np.isnan(y3)
xW3   = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3  = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD3 = match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y3    = y3[v3]
iW3   = match_index(mat3[0, :N0][v3], pf_W_m   ).astype(float)
iKV3  = match_index(mat3[1, :N0][v3], pf_KV_m  ).astype(float)
iKVD3 = match_index(mat3[4, :N0][v3], pf_KVDIM_m).astype(float)
N = len(y3)
X3_jsd = np.column_stack([xW3, xKV3, xKVD3])
X3_idx = np.column_stack([iW3, iKV3, iKVD3])
lb_idx = np.array([0., 0., 0.])
ub_idx = np.array([len(pf_W_m)-1, len(pf_KV_m)-1, len(pf_KVDIM_m)-1], dtype=float)
JSD_KV_def = pf_KV_m[0, 0]

mat_wd = load_csv(AWQ_W_KVD); N_wd = mat_wd.shape[1]
y_WD = mat_wd[12, :N_wd]; v_wd = ~np.isnan(y_WD)
xW_WD   = match_metric(mat_wd[0, :N_wd], pf_W   )[v_wd]
xKVD_WD = match_metric(mat_wd[4, :N_wd], pf_KVDIM)[v_wd]
y_WD    = y_WD[v_wd]
print(f"  3-way N={N}; WD pair N={len(y_WD)}")

# ─── Build 27-grid + 23 maximin = 50 train pool / 150 test ─────────────────
qs = [0.1, 0.5, 0.9]
qW   = np.quantile(xW3,   qs); qKV  = np.quantile(xKV3,  qs); qKVD = np.quantile(xKVD3, qs)
grid27 = np.array([[w,kv,kvd] for w in qW for kv in qKV for kvd in qKVD])
scale  = X3_jsd.std(0) + 1e-10
X3n    = X3_jsd / scale
grid_n = grid27 / scale
cost   = np.zeros((27, N))
for j in range(27): cost[j] = np.sum((X3n - grid_n[j])**2, axis=1)
_, col_ind = linear_sum_assignment(cost)
grid_samples = col_ind.astype(int)
remaining = np.setdiff1d(np.arange(N), grid_samples)
extras = []; sel = list(grid_samples)
for _ in range(23):
    sel_arr = np.array(sel, dtype=int)
    d = np.array([np.min(np.sum((X3n[r] - X3n[sel_arr])**2, axis=1)) for r in remaining])
    far = int(np.argmax(d))
    extras.append(int(remaining[far])); sel.append(int(remaining[far]))
    remaining = np.delete(remaining, far)
extras = np.array(extras, dtype=int)
train_pool = np.concatenate([grid_samples, extras])
test_set   = np.setdiff1d(np.arange(N), train_pool)
print(f"  Fixed split: 50 train pool / 150 test")

# ─── Helpers ─────────────────────────────────────────────────────────────────
def r2(y, yp):
    ss_t = max(np.sum((y - y.mean())**2), 1e-30)
    return 1 - np.sum((y-yp)**2)/ss_t
def rmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)))

X_te_jsd = X3_jsd[test_set]; X_te_idx = X3_idx[test_set]; y_te = y3[test_set]

# ─── Surrogate models ────────────────────────────────────────────────────────
def fit_pred_pysot_rbf(X_tr_idx, y_tr, X_te_idx_, kernel='cubic'):
    m = PySOTRBF(kernel=kernel, tail='linear', lb=lb_idx, ub=ub_idx)
    m.fit(X_tr_idx, y_tr)
    return m.predict(X_te_idx_).ravel()

def fit_pred_ard_gp(X_tr, y_tr, X_te, n_restarts=5):
    """ARD-GP (anisotropic kernel, per-dim length scale)."""
    n_dim = X_tr.shape[1]
    kernel = (C(1.0, (1e-4, 1e2)) *
              SKRBF(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-3, 1e3)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X_tr, y_tr)
    return gp.predict(X_te), gp.kernel_

def fit_pred_cokriging(X_tr_3, y_tr_3, X_te_3, X_pair, y_pair, pair_dims=(0,2), n_restarts=3):
    """NARGP-style Co-Kriging: low-fi pair GP → augment 3D-GP input with pair_pred.

    y_3way ≈ GP_h([W, KV, KVD, pair_pred(W, KVD)])

    This generalizes both:
      • linear residual cascade  (pair_pred enters linearly)
      • nonlinear scaling        (pair_pred can be modulated by other dims)
    """
    # Step 1: low-fidelity pair GP
    kernel_p = (C(1.0, (1e-4, 1e2)) *
                SKRBF(length_scale=[1.0]*2, length_scale_bounds=(1e-3, 1e3)) +
                WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp_p = GaussianProcessRegressor(kernel=kernel_p, normalize_y=True,
                                    n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp_p.fit(X_pair, y_pair)

    # Step 2: pair predictions on 3-way data
    pp_tr = gp_p.predict(X_tr_3[:, list(pair_dims)])
    pp_te = gp_p.predict(X_te_3[:, list(pair_dims)])

    # Step 3: high-fidelity augmented 3D-GP
    Xa_tr = np.column_stack([X_tr_3, pp_tr])
    Xa_te = np.column_stack([X_te_3, pp_te])
    kernel_h = (C(1.0, (1e-4, 1e2)) *
                SKRBF(length_scale=[1.0]*4, length_scale_bounds=(1e-3, 1e3)) +
                WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp_h = GaussianProcessRegressor(kernel=kernel_h, normalize_y=True,
                                    n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp_h.fit(Xa_tr, y_tr_3)
    return gp_h.predict(Xa_te), gp_h.kernel_

def fit_pred_hier_tps_linear(X_tr_3, y_tr_3, X_te_3, X_pair, y_pair, pair_dims=(0,2)):
    """Reference: existing best Hier (TPS pair + linear residual on third var)."""
    a_idx, b_idx = pair_dims
    third_idx = [i for i in range(3) if i not in pair_dims][0]

    X2 = X_pair
    m_pair = PySOTRBF(kernel='tps', tail='linear', lb=X2.min(0), ub=X2.max(0))
    m_pair.fit(X2, y_pair)
    pp_tr = m_pair.predict(np.column_stack([X_tr_3[:, a_idx], X_tr_3[:, b_idx]])).ravel()
    pp_te = m_pair.predict(np.column_stack([X_te_3[:, a_idx], X_te_3[:, b_idx]])).ravel()

    r_tr = y_tr_3 - pp_tr
    t_tr = X_tr_3[:, third_idx]; t_te = X_te_3[:, third_idx]
    c = np.polyfit(t_tr, r_tr, 1)
    return pp_te + np.polyval(c, t_te)

def _fit_ard_gp(X, y, n_dim, n_restarts=3):
    kernel = (C(1.0, (1e-4, 1e2)) *
              SKRBF(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-3, 1e3)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y); return gp

def fit_pred_hier_ard_lin(X_tr_3, y_tr_3, X_te_3, X_pair, y_pair, pair_dims=(0,2), n_restarts=3):
    """Hier with ARD-GP pair (2D) + linear residual on third var."""
    a_idx, b_idx = pair_dims
    third_idx = [i for i in range(3) if i not in pair_dims][0]
    gp_p = _fit_ard_gp(X_pair, y_pair, 2, n_restarts)
    pp_tr = gp_p.predict(np.column_stack([X_tr_3[:, a_idx], X_tr_3[:, b_idx]]))
    pp_te = gp_p.predict(np.column_stack([X_te_3[:, a_idx], X_te_3[:, b_idx]]))
    r_tr = y_tr_3 - pp_tr
    t_tr = X_tr_3[:, third_idx]; t_te = X_te_3[:, third_idx]
    c = np.polyfit(t_tr, r_tr, 1)
    return pp_te + np.polyval(c, t_te)

def fit_pred_hier_ard_ard(X_tr_3, y_tr_3, X_te_3, X_pair, y_pair, pair_dims=(0,2), n_restarts=3):
    """Hier with ARD-GP pair (2D) + ARD-GP residual on full 3D input."""
    a_idx, b_idx = pair_dims
    gp_p = _fit_ard_gp(X_pair, y_pair, 2, n_restarts)
    pp_tr = gp_p.predict(np.column_stack([X_tr_3[:, a_idx], X_tr_3[:, b_idx]]))
    pp_te = gp_p.predict(np.column_stack([X_te_3[:, a_idx], X_te_3[:, b_idx]]))
    r_tr = y_tr_3 - pp_tr
    gp_r = _fit_ard_gp(X_tr_3, r_tr, 3, n_restarts)
    return pp_te + gp_r.predict(X_te_3)

def fit_pred_hier_ard_ard1D(X_tr_3, y_tr_3, X_te_3, X_pair, y_pair, pair_dims=(0,2), n_restarts=3):
    """Hier with ARD-GP pair (2D) + 1D ARD-GP residual on third var only."""
    a_idx, b_idx = pair_dims
    third_idx = [i for i in range(3) if i not in pair_dims][0]
    gp_p = _fit_ard_gp(X_pair, y_pair, 2, n_restarts)
    pp_tr = gp_p.predict(np.column_stack([X_tr_3[:, a_idx], X_tr_3[:, b_idx]]))
    pp_te = gp_p.predict(np.column_stack([X_te_3[:, a_idx], X_te_3[:, b_idx]]))
    r_tr = y_tr_3 - pp_tr
    gp_r = _fit_ard_gp(X_tr_3[:, [third_idx]], r_tr, 1, n_restarts)
    return pp_te + gp_r.predict(X_te_3[:, [third_idx]])

# ─── Evaluation: total budget B ───────────────────────────────────────────────
# Optimal hier splits from prior analysis (n3, n_pair) summing to B
optimal_split = {27: (9, 18), 30: (9, 21), 35: (12, 23), 40: (15, 25), 45: (18, 27), 50: (21, 29)}
N_TRAINS = list(optimal_split.keys())
N_SEED = 30

def get_3way_train(n_3way, seed):
    rng = np.random.RandomState(seed)
    n_extra = n_3way - 27
    if n_extra == 0: tr3 = grid_samples
    elif n_extra == 23: tr3 = train_pool
    elif 0 < n_extra < 23:
        tr3 = np.concatenate([grid_samples, extras[rng.choice(23, n_extra, replace=False)]])
    elif n_extra < 0:
        tr3 = grid_samples[rng.choice(27, n_3way, replace=False)]
    else: tr3 = train_pool
    return tr3

def get_pair_train(n_pair, seed):
    rng = np.random.RandomState(seed + 100000)
    return rng.choice(len(y_WD), min(n_pair, len(y_WD)), replace=False)

print(f"\nEvaluating 4 surrogates × {len(N_TRAINS)} budgets × {N_SEED} seeds...")
results = {}  # (method, B) -> list of (r2, rmse)

for B in N_TRAINS:
    n_3way, n_pair = optimal_split[B]
    print(f"\n  B={B} (n3={n_3way}, n_pair={n_pair}):")
    for seed in range(N_SEED):
        tr_3way = get_3way_train(B, seed) if B <= 50 else None
        x3_tr_jsd = X3_jsd[tr_3way]; x3_tr_idx = X3_idx[tr_3way]; y3_tr = y3[tr_3way]

        # Direct-RBF (pySOT cubic+linear, sorted-index)
        try:
            yp = fit_pred_pysot_rbf(x3_tr_idx, y3_tr, X_te_idx)
            results.setdefault(('Direct-RBF', B), []).append((r2(y_te, yp), rmse(y_te, yp)))
        except Exception as e:
            pass

        # Direct-ARD-GP (sklearn ARD on JSD input)
        try:
            yp, _ = fit_pred_ard_gp(x3_tr_jsd, y3_tr, X_te_jsd)
            results.setdefault(('Direct-ARD-GP', B), []).append((r2(y_te, yp), rmse(y_te, yp)))
        except Exception:
            pass

        # Hier variants (constrained budget: n_3way + n_pair = B)
        tr_3way_hier = get_3way_train(n_3way, seed)
        x3_h_jsd = X3_jsd[tr_3way_hier]; y_h = y3[tr_3way_hier]
        pair_idx = get_pair_train(n_pair, seed)
        X_pair_2d = np.column_stack([xW_WD[pair_idx], xKVD_WD[pair_idx]])
        y_pair_arr = y_WD[pair_idx]

        try:
            yp = fit_pred_hier_tps_linear(x3_h_jsd, y_h, X_te_jsd, X_pair_2d, y_pair_arr, pair_dims=(0,2))
            results.setdefault(('Hier-WD-TPS-lin', B), []).append((r2(y_te, yp), rmse(y_te, yp)))
        except Exception: pass

        try:
            yp = fit_pred_hier_ard_lin(x3_h_jsd, y_h, X_te_jsd, X_pair_2d, y_pair_arr, pair_dims=(0,2))
            results.setdefault(('Hier-WD-ARD-lin', B), []).append((r2(y_te, yp), rmse(y_te, yp)))
        except Exception: pass

        try:
            yp = fit_pred_hier_ard_ard1D(x3_h_jsd, y_h, X_te_jsd, X_pair_2d, y_pair_arr, pair_dims=(0,2))
            results.setdefault(('Hier-WD-ARD-ARD1D', B), []).append((r2(y_te, yp), rmse(y_te, yp)))
        except Exception: pass

        try:
            yp = fit_pred_hier_ard_ard(x3_h_jsd, y_h, X_te_jsd, X_pair_2d, y_pair_arr, pair_dims=(0,2))
            results.setdefault(('Hier-WD-ARD-ARD3D', B), []).append((r2(y_te, yp), rmse(y_te, yp)))
        except Exception: pass

        try:
            yp, _ = fit_pred_cokriging(x3_h_jsd, y_h, X_te_jsd, X_pair_2d, y_pair_arr, pair_dims=(0,2))
            results.setdefault(('CoKriging-WD', B), []).append((r2(y_te, yp), rmse(y_te, yp)))
        except Exception: pass

    for m in ['Direct-RBF', 'Direct-ARD-GP', 'Hier-WD-TPS-lin',
              'Hier-WD-ARD-lin', 'Hier-WD-ARD-ARD1D', 'Hier-WD-ARD-ARD3D', 'CoKriging-WD']:
        rs = results.get((m, B), [])
        if rs:
            r2s = [r[0] for r in rs]
            print(f"    {m:20s}: R²={np.median(r2s):.4f} [{np.percentile(r2s,10):.3f},{np.percentile(r2s,90):.3f}]")
        else:
            print(f"    {m:20s}: failed")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("Summary: median test R² per (method, B)")
print("="*100)
methods = ['Direct-RBF', 'Direct-ARD-GP', 'Hier-WD-TPS-lin',
           'Hier-WD-ARD-lin', 'Hier-WD-ARD-ARD1D', 'Hier-WD-ARD-ARD3D', 'CoKriging-WD']
hdr = f"  {'Method':22s}"
for B in N_TRAINS: hdr += f"  B={B:<3d}    "
print(hdr)
for m in methods:
    line = f"  {m:22s}"
    for B in N_TRAINS:
        rs = results.get((m, B), [])
        med = float(np.median([r[0] for r in rs])) if rs else np.nan
        line += f"  {med:.4f}    "
    print(line)

# Best per B
print("\n" + "="*100)
print("Best method at each B (with Δ vs Direct-RBF)")
print("="*100)
for B in N_TRAINS:
    cands = []
    for m in methods:
        rs = results.get((m, B), [])
        if rs: cands.append((m, float(np.median([r[0] for r in rs]))))
    cands.sort(key=lambda x: -x[1])
    direct_r2 = next(c[1] for c in cands if c[0] == 'Direct-RBF')
    best = cands[0]
    delta = best[1] - direct_r2
    print(f"  B={B:2d}  best={best[0]:18s} R²={best[1]:.4f}    Δ vs Direct-RBF = {delta:+.4f}")

# ─── Figures ─────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')
COLOR = {'Direct-RBF':'#3C5488', 'Direct-ARD-GP':'#E64B35',
         'Hier-WD-TPS-lin':'#4DBBD5', 'Hier-WD-ARD-lin':'#00A087',
         'Hier-WD-ARD-ARD1D':'#F39B7F', 'Hier-WD-ARD-ARD3D':'#9B59B6',
         'CoKriging-WD':'#7F7F7F'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for m in methods:
    xs, med, lo, hi = [], [], [], []
    for B in N_TRAINS:
        rs = results.get((m, B), [])
        if not rs: continue
        v = [r[0] for r in rs]
        xs.append(B); med.append(np.median(v))
        lo.append(np.percentile(v, 10)); hi.append(np.percentile(v, 90))
    ax.plot(xs, med, '-o', color=COLOR[m], label=m, lw=2.4, ms=8,
            markeredgecolor='white', markeredgewidth=0.7)
    ax.fill_between(xs, lo, hi, color=COLOR[m], alpha=0.13)
ax.set_xlabel('Total measurements  B'); ax.set_ylabel('Test R² (median, [10%, 90%])')
ax.set_title('Test R² vs total budget', fontweight='bold')
ax.set_xticks(N_TRAINS); ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.25, lw=0.5)

# RMSE
ax = axes[1]
for m in methods:
    xs, med = [], []
    for B in N_TRAINS:
        rs = results.get((m, B), [])
        if not rs: continue
        xs.append(B); med.append(np.median([r[1] for r in rs]))
    ax.plot(xs, med, '-o', color=COLOR[m], label=m, lw=2.4, ms=8,
            markeredgecolor='white', markeredgewidth=0.7)
ax.set_xlabel('Total measurements  B'); ax.set_ylabel('Test RMSE (median)')
ax.set_title('Test RMSE vs total budget', fontweight='bold')
ax.set_xticks(N_TRAINS); ax.legend(fontsize=9); ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle('ARD-GP and Co-Kriging vs current Direct/Hier (150 hold-out, 30 seeds)',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/07_ardrbf_cokriging.png', **PLT_KW); plt.close()

print(f"\nFigure saved: 07_ardrbf_cokriging.png")
print("Done.\n")
