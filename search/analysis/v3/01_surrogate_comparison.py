"""01_surrogate_comparison.py — Section 4 of the narrative.

Compares surrogate models for response-surface regression of
y_full = f_0 + Σ f_i + Σ f_ij + ... on per-method JSD inputs.

Models compared (fixed 50-train / 150-test split, 27-grid + 23 maximin):
  • M0  (linear, no intercept)        — first-order additive only
  • M1  (linear, w/ intercept)
  • M8  (M1 + all single squared)     — quadratic main effects
  • M9  (M1 + all pairwise products)  — interaction main effects
  • M10 (full quadratic)              — main + quadratic + interactions
  • RBF cubic+linear (JSD input)      — nonparametric
  • RBF tps+linear   (JSD input)
  • ARD-GP (sklearn, JSD input)       — anisotropic kernel, automatic sensitivity

Key result: First-order additive (M0/M1) already gives R² ≈ 0.95;
adding quadratic+interaction (M10) → 0.97; ARD-GP achieves R² = 0.996.

This justifies that **interaction is small** and supports the 2ε-Pareto
theorem (see 02_ard_gp_analysis.py and 03_pareto_combination.py).
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
from sklearn.gaussian_process.kernels import RBF as SKRBF, ConstantKernel as C, WhiteKernel
from utils.func import get_net_info
from predictor.rbf import RBF

# ─── Data loading (mirrors analysis_v3.py) ────────────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]
    min2 = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < min2:
            nd.append(i); min2 = F_s[i, 1]
    return order[nd]

def load_archive_pareto(stats_path, comp_key, config, group_size):
    with open(stats_path) as f: data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs   = [v[0] for v in archive]
    metrics = np.array([v[1] for v in archive])
    comps   = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
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

with open(f'{BASE}/config/llama.json') as f:
    config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

print("Loading PFs and 3-way data...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
pf_W_m     = pf_W    [np.argsort(pf_W    [:, 0])]
pf_KV_m    = pf_KV   [np.argsort(pf_KV   [:, 0])]
pf_KVDIM_m = pf_KVDIM[np.argsort(pf_KVDIM[:, 0])]

mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y3   = mat3[12, :N0]
v3   = ~np.isnan(y3)
xW3   = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3  = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD3 = match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y3    = y3[v3]
iW3   = match_index(mat3[0, :N0][v3], pf_W_m   ).astype(float)
iKV3  = match_index(mat3[1, :N0][v3], pf_KV_m  ).astype(float)
iKVD3 = match_index(mat3[4, :N0][v3], pf_KVDIM_m).astype(float)
N = len(y3)
print(f"  Total samples: N={N}")

X_jsd = np.column_stack([xW3, xKV3, xKVD3])
X_idx = np.column_stack([iW3, iKV3, iKVD3])
lb_idx = np.array([0., 0., 0.])
ub_idx = np.array([len(pf_W_m)-1, len(pf_KV_m)-1, len(pf_KVDIM_m)-1], dtype=float)

# ─── 27-grid via Hungarian assignment to get 27 distinct samples ──────────────
print("\nBuilding fixed train pool...")
qs = [0.1, 0.5, 0.9]
qW   = np.quantile(xW3,   qs)
qKV  = np.quantile(xKV3,  qs)
qKVD = np.quantile(xKVD3, qs)
print(f"  Quantiles: W={qW.round(4)}  KV={qKV.round(4)}  KVD={qKVD.round(4)}")

grid27 = np.array([[w, kv, kvd] for w in qW for kv in qKV for kvd in qKVD])  # (27, 3)
scale = X_jsd.std(0) + 1e-10
X_n = X_jsd / scale
grid27_n = grid27 / scale

# Cost matrix: rows=27 grid points, cols=200 samples (squared L2 in normalized space)
cost = np.zeros((27, N))
for j in range(27):
    cost[j] = np.sum((X_n - grid27_n[j])**2, axis=1)
row_ind, col_ind = linear_sum_assignment(cost)  # 27 unique sample indices
grid_samples = col_ind.astype(int)
assert len(np.unique(grid_samples)) == 27, "Hungarian gave duplicates"
total_grid_dist = cost[row_ind, col_ind].sum()
print(f"  27 distinct grid-base samples assigned (total normalized SSD={total_grid_dist:.3f})")

# ─── 23 extras via farthest-point (maximin) sampling ──────────────────────────
remaining = np.setdiff1d(np.arange(N), grid_samples)
extras = []
selected_set = list(grid_samples)
for _ in range(23):
    # Distance from each remaining to nearest already-selected
    sel_arr = np.array(selected_set, dtype=int)
    d = np.full(len(remaining), np.inf)
    for i, r in enumerate(remaining):
        d[i] = np.min(np.sum((X_n[r] - X_n[sel_arr])**2, axis=1))
    far = int(np.argmax(d))
    chosen = int(remaining[far])
    extras.append(chosen)
    selected_set.append(chosen)
    remaining = np.delete(remaining, far)
extras = np.array(extras, dtype=int)
print(f"  23 maximin extras selected")

# Train pool = 27 + 23 = 50, Test = 150 (fixed)
train_pool = np.concatenate([grid_samples, extras])
assert len(np.unique(train_pool)) == 50
test_set = np.setdiff1d(np.arange(N), train_pool)
assert len(test_set) == 150
print(f"  Train pool: 50 (27 grid + 23 extras), Test set: 150 (fixed)")
print(f"  JSD ranges in test set:  W [{xW3[test_set].min():.3f},{xW3[test_set].max():.3f}]  "
      f"KV [{xKV3[test_set].min():.3f},{xKV3[test_set].max():.3f}]  "
      f"KVD [{xKVD3[test_set].min():.3f},{xKVD3[test_set].max():.3f}]")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def features(xw, xkv, xkvd, mode):
    n = len(xw); o = np.ones(n)
    if mode == 'M0' : return np.column_stack([xw, xkv, xkvd])
    if mode == 'M1' : return np.column_stack([o, xw, xkv, xkvd])
    if mode == 'M8' : return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2])
    if mode == 'M9' : return np.column_stack([o, xw, xkv, xkvd, xw*xkv, xw*xkvd, xkv*xkvd])
    if mode == 'M10': return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2,
                                              xw*xkv, xw*xkvd, xkv*xkvd])

def fit_pred_ols(mode, x_tr, y_tr, x_te):
    Phi_tr = features(x_tr[:,0], x_tr[:,1], x_tr[:,2], mode)
    Phi_te = features(x_te[:,0], x_te[:,1], x_te[:,2], mode)
    coef, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
    return Phi_tr @ coef, Phi_te @ coef

def r2(y_t, y_p):
    ss_r = np.sum((y_t - y_p)**2); ss_t = np.sum((y_t - y_t.mean())**2)
    return 1 - ss_r / max(ss_t, 1e-30)
def rmse(y_t, y_p): return float(np.sqrt(np.mean((y_t - y_p)**2)))

# ─── Evaluation loop ──────────────────────────────────────────────────────────
N_TRS = [27, 30, 35, 40, 45, 50]
N_SEED = 50
math_models = ['M0', 'M1', 'M8', 'M9', 'M10']
rbf_variants = [('cubic','linear'), ('tps','linear')]
rbf_names = [f'RBF-{k}+{t}' for k,t in rbf_variants]
ard_names  = ['ARD-GP']

def fit_ard_gp(X_tr, y_tr, X_te, n_dim=3, n_restarts=5):
    """ARD-GP (sklearn) with per-dim length scale on JSD input."""
    kernel = (C(1.0, (1e-4, 1e2)) *
              SKRBF(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-3, 1e3)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X_tr, y_tr)
    yp_tr = gp.predict(X_tr); yp_te = gp.predict(X_te)
    return yp_tr, yp_te

# storage[(model, n_tr)] = list of (r2_train, r2_test, rmse_test)
storage = {}

X_te = X_jsd[test_set]; X_te_idx = X_idx[test_set]; y_te = y3[test_set]

# RBF (pySOT) bounds for JSD input
lb_jsd = X_jsd.min(0); ub_jsd = X_jsd.max(0)

print(f"\nRunning low-N evaluation (50 seeds for N_tr ∈ {{30,35,40,45}}, deterministic for 27 & 50)...")
for n_tr in N_TRS:
    n_extra = n_tr - 27
    if n_extra == 0 or n_extra == 23:
        # Deterministic: use exactly one config
        seeds = [None]
    else:
        seeds = list(range(N_SEED))
    for seed in seeds:
        if n_extra == 0:
            tr_idx = grid_samples
        elif n_extra == 23:
            tr_idx = train_pool
        else:
            rng = np.random.RandomState(seed)
            extra_subset = rng.choice(23, n_extra, replace=False)
            tr_idx = np.concatenate([grid_samples, extras[extra_subset]])

        x_tr = X_jsd[tr_idx]; y_tr = y3[tr_idx]; x_tr_idx = X_idx[tr_idx]

        for m in math_models:
            try:
                yp_tr, yp_te = fit_pred_ols(m, x_tr, y_tr, X_te)
                storage.setdefault((m, n_tr), []).append(
                    (r2(y_tr, yp_tr), r2(y_te, yp_te), rmse(y_te, yp_te)))
            except Exception:
                pass

        for kernel, tail in rbf_variants:
            name = f'RBF-{kernel}+{tail}'
            try:
                # JSD input (continuous-valued) instead of sorted-index
                m_rbf = RBF(kernel=kernel, tail=tail, lb=lb_jsd, ub=ub_jsd)
                m_rbf.fit(x_tr, y_tr)
                yp_tr_r = m_rbf.predict(x_tr).ravel()
                yp_te_r = m_rbf.predict(X_te).ravel()
                storage.setdefault((name, n_tr), []).append(
                    (r2(y_tr, yp_tr_r), r2(y_te, yp_te_r), rmse(y_te, yp_te_r)))
            except Exception: pass

        # ARD-GP on JSD input
        try:
            yp_tr_a, yp_te_a = fit_ard_gp(x_tr, y_tr, X_te)
            storage.setdefault(('ARD-GP', n_tr), []).append(
                (r2(y_tr, yp_tr_a), r2(y_te, yp_te_a), rmse(y_te, yp_te_a)))
        except Exception:
            pass

# ─── Reporting ────────────────────────────────────────────────────────────────
def fmt_cell(rs, idx):
    vals = [r[idx] for r in rs]
    if not vals: return f"{'N/A':14s}"
    if len(vals) == 1: return f"{vals[0]:.4f}        "
    return f"{np.median(vals):.3f}[{np.percentile(vals,10):.2f},{np.percentile(vals,90):.2f}]"

def stat_block(model_list, title, idx):
    print("\n" + "="*108)
    print(title)
    print("="*108)
    hdr = f"  {'Model':22s}"
    for n_tr in N_TRS:
        marker = "*" if n_tr in [27, 50] else " "
        hdr += f"  {marker}N={n_tr:<3d}        "
    print(hdr + "    (* = deterministic single value)")
    for m in model_list:
        line = f"  {m:22s}"
        for n_tr in N_TRS:
            rs = storage.get((m, n_tr), [])
            line += f"  {fmt_cell(rs, idx):16s}"
        print(line)

stat_block(math_models, "SECTION 1 — Math models, test R² on 150 held-out samples", idx=1)
stat_block(rbf_names,   "SECTION 2 — RBF (JSD input), test R² on 150 held-out samples", idx=1)
stat_block(ard_names,   "SECTION 3 — ARD-GP (JSD input), test R² on 150 held-out samples", idx=1)
stat_block(math_models, "SECTION 1 — Math models, test RMSE on 150 held-out samples", idx=2)
stat_block(rbf_names,   "SECTION 2 — RBF, test RMSE on 150 held-out samples", idx=2)
stat_block(ard_names,   "SECTION 3 — ARD-GP, test RMSE on 150 held-out samples", idx=2)

# ─── Best-of summary at each N_tr ─────────────────────────────────────────────
print("\n" + "="*108)
print("Best test R² (median) at each N_tr:")
print("="*108)
for n_tr in N_TRS:
    cands = []
    for m in math_models + rbf_names + ard_names:
        rs = storage.get((m, n_tr), [])
        if rs:
            r2s = [r[1] for r in rs]
            cands.append((m, np.median(r2s), np.percentile(r2s, 10), np.percentile(r2s, 90)))
    cands.sort(key=lambda x: -x[1])
    top = cands[:3]
    print(f"  N={n_tr}: "
          + " | ".join([f"{c[0]:20s} {c[1]:.4f} [{c[2]:.3f},{c[3]:.3f}]" for c in top]))

# ─── Figures ──────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')

math_colors = {'M0':'#E64B35', 'M1':'#3C5488', 'M8':'#4DBBD5', 'M9':'#F39B7F', 'M10':'#9B59B6'}
rbf_colors  = {'RBF-cubic+linear':'#E64B35', 'RBF-tps+linear':'#4DBBD5'}
ard_colors  = {'ARD-GP':'#00A087'}

def collect(name, idx=1):
    xs, med, lo, hi = [], [], [], []
    for n_tr in N_TRS:
        rs = storage.get((name, n_tr), [])
        if not rs: continue
        v = [r[idx] for r in rs]
        xs.append(n_tr); med.append(np.median(v))
        lo.append(np.percentile(v, 10) if len(v) > 1 else v[0])
        hi.append(np.percentile(v, 90) if len(v) > 1 else v[0])
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)

# Fig 1: math + RBF + ARD-GP side-by-side test R²
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axes[0]
for m in math_models:
    xs, med, lo, hi = collect(m, idx=1)
    ax.plot(xs, med, '-o', color=math_colors[m], label=m, lw=2, ms=6)
    ax.fill_between(xs, lo, hi, color=math_colors[m], alpha=0.13)
ax.set_xlabel('N_train')
ax.set_ylabel('Test R² on fixed 150 hold-out')
ax.set_title('Section 1 — Math models (M0–M10)', fontweight='bold')
ax.set_xticks(N_TRS); ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.25, lw=0.5)

ax = axes[1]
for r_name in rbf_names:
    xs, med, lo, hi = collect(r_name, idx=1)
    ax.plot(xs, med, '-o', color=rbf_colors[r_name], label=r_name.replace('RBF-',''), lw=2, ms=6)
    ax.fill_between(xs, lo, hi, color=rbf_colors[r_name], alpha=0.13)
ax.set_xlabel('N_train')
ax.set_ylabel('Test R²')
ax.set_title('Section 2 — RBF (JSD input)', fontweight='bold')
ax.set_xticks(N_TRS); ax.legend(fontsize=9); ax.grid(True, alpha=0.25, lw=0.5)

ax = axes[2]
for a_name in ard_names:
    xs, med, lo, hi = collect(a_name, idx=1)
    ax.plot(xs, med, '-o', color=ard_colors[a_name], label=a_name, lw=2, ms=7,
            markeredgecolor='white', markeredgewidth=0.6)
    ax.fill_between(xs, lo, hi, color=ard_colors[a_name], alpha=0.13)
ax.set_xlabel('N_train')
ax.set_ylabel('Test R²')
ax.set_title('Section 3 — ARD-GP (sklearn, JSD input)', fontweight='bold')
ax.set_xticks(N_TRS); ax.legend(fontsize=9); ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle('Fixed split: 50 train pool / 150 test (median + [10%,90%] over 50 seeds)',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/01_surrogate_sections12.png', **PLT_KW); plt.close()

# Fig 2: all-models comparison
fig, ax = plt.subplots(figsize=(10, 5))
for m in math_models:
    xs, med, _, _ = collect(m, idx=1)
    ax.plot(xs, med, '-o', color=math_colors[m], label=m, lw=1.8, ms=5)
for r_name in rbf_names:
    xs, med, _, _ = collect(r_name, idx=1)
    ax.plot(xs, med, '--s', color=rbf_colors[r_name], label=r_name.replace('RBF-',''),
            lw=1.8, ms=5, alpha=0.85)
for a_name in ard_names:
    xs, med, _, _ = collect(a_name, idx=1)
    ax.plot(xs, med, '-D', color=ard_colors[a_name], label=a_name, lw=2.6, ms=8,
            markeredgecolor='white', markeredgewidth=0.7, zorder=10)
ax.set_xlabel('N_train'); ax.set_ylabel('Median test R² on 150 hold-out')
ax.set_title('All surrogates — math (solid), RBF (dashed), ARD-GP (bold diamonds)', fontweight='bold')
ax.set_xticks(N_TRS); ax.legend(fontsize=8, ncol=3, loc='lower right')
ax.grid(True, alpha=0.25, lw=0.5)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/01_surrogate_all_models.png', **PLT_KW); plt.close()

# Fig 3: train pool visualization in JSD space
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (i, j, lbl_x, lbl_y) in zip(axes, [(0,1,'JSD_W','JSD_KV'),(0,2,'JSD_W','JSD_KVD'),(1,2,'JSD_KV','JSD_KVD')]):
    ax.scatter(X_jsd[test_set, i], X_jsd[test_set, j], c='#CCCCCC', s=14, alpha=0.5, label='test (150)', zorder=1)
    ax.scatter(X_jsd[extras, i],   X_jsd[extras, j],   c=math_colors['M8'], s=42, marker='^',
               edgecolor='black', linewidth=0.5, label='maximin extras (23)', zorder=3)
    ax.scatter(X_jsd[grid_samples, i], X_jsd[grid_samples, j], c=math_colors['M0'], s=70, marker='*',
               edgecolor='black', linewidth=0.5, label='27-grid base', zorder=4)
    # quantile lines
    qx = [qW, qKV, qKVD][i]; qy = [qW, qKV, qKVD][j]
    for q in qx: ax.axvline(q, color='gray', ls=':', alpha=0.5, lw=0.7)
    for q in qy: ax.axhline(q, color='gray', ls=':', alpha=0.5, lw=0.7)
    ax.set_xlabel(lbl_x); ax.set_ylabel(lbl_y)
    ax.grid(True, alpha=0.2, lw=0.4)
    if ax is axes[0]: ax.legend(fontsize=8.5, loc='upper right')
plt.suptitle('Fixed split layout in JSD space (dotted = quantile lines)', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/01_surrogate_split_layout.png', **PLT_KW); plt.close()

print(f"\nFigures saved: 01_surrogate_sections12.png, 01_surrogate_all_models.png, 01_surrogate_split_layout.png")
print("Done.\n")
