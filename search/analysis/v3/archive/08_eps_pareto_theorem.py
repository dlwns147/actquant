"""08_eps_pareto_theorem.py — 2ε-Pareto Theorem: ε measurement on 150 test samples.

THEOREM:
  Let y_actual = y_add(x) + r(x), with ε := ‖r‖_∞ (or ε := ‖r‖_2).
  Then on ANY candidate set X:
        Pareto(y_add)  ⊆  2ε-Pareto(y_actual)
  i.e. every Combined-PF point (selected by additive y_add) is within 2ε in y-direction
  of the True-PF (scored by y_actual).

Setup:
  • Fixed 50-train pool (27-grid + 23 maximin) / 150-test split (same as analyses 1-4)
  • ARD-GP trained on 50-train
  • Hoeffding additive components g_W, g_KV, g_KVD extracted via 1D conditional means
  • For 150 test samples: y_actual (measured), y_full (ARD-GP), y_add (additive)
  • ε bounds = ‖y_actual − y_add‖_{∞,2}  on the 150 test residuals

NOTE: Pareto frontier comparison on 150 random samples is too sparse to be statistically
       meaningful (PF size very sample-dependent). We only report:
   (1) ε bounds — directly measure additive-approximation error on real data
   (2) Pred-vs-actual scatter — visualize where y_add deviates from y_actual,
       broken down per complexity axis (wbits / eff_kvbits / both)

The theorem then guarantees, for ANY candidate set, that Combined-PF stays within
2ε of the True-PF in y-direction.
"""
import sys, os, json, csv, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SKRBF, ConstantKernel as C, WhiteKernel
from scipy.optimize import linear_sum_assignment
from utils.func import get_net_info

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

BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
with open(f'{BASE}/config/llama.json') as f: config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w':128, 'k':[128,128], 'v':[128,128]}

print("Loading data...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y_all = mat3[12, :N0]; v3 = ~np.isnan(y_all)
xW3   = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3  = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD  = match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y_all = y_all[v3]; N = len(y_all)
X3 = np.column_stack([xW3, xKV3, xKVD])
HEAD_DIM = 128

wbits_all  = mat3[0, :N0][v3]
kvbits_all = mat3[1, :N0][v3]
kvdim_all  = mat3[4, :N0][v3]
eff_kv_all = kvbits_all * kvdim_all / HEAD_DIM
print(f"  N={N} 3-way samples")

# ─── Fixed 50/150 split ───────────────────────────────────────────────────────
qs = [0.1, 0.5, 0.9]
qW_, qKV_, qKVD_ = np.quantile(xW3, qs), np.quantile(xKV3, qs), np.quantile(xKVD, qs)
grid27 = np.array([[w,kv,kvd] for w in qW_ for kv in qKV_ for kvd in qKVD_])
scale  = X3.std(0) + 1e-10
X3n    = X3 / scale
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
print(f"  Fixed split: {len(train_pool)} train pool / {len(test_set)} test")

# ─── Fit ARD-GP on 50-train pool ─────────────────────────────────────────────
def fit_ard_gp(X, y, n_restarts=20):
    kernel = (C(1.0,(1e-4,1e2)) *
              SKRBF(length_scale=[1.0]*X.shape[1], length_scale_bounds=(1e-3,1e3)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9,1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y); return gp

print("\nFitting ARD-GP on 50-train pool...")
gp = fit_ard_gp(X3[train_pool], y_all[train_pool])
ls = np.array(gp.kernel_.k1.k2.length_scale)
print(f"  Length scales: l_W={ls[0]:.4f}, l_KV={ls[1]:.4f}, l_KVD={ls[2]:.4f}")

# ─── Hoeffding additive decomposition ────────────────────────────────────────
print("\nComputing Hoeffding additive components g_i ...")
bounds_jsd = [[xW3.min(), xW3.max()], [xKV3.min(), xKV3.max()], [xKVD.min(), xKVD.max()]]
rng = np.random.RandomState(0); n_mc = 5000
X_mc = np.array([rng.uniform(b[0], b[1], n_mc) for b in bounds_jsd]).T
y_mc = gp.predict(X_mc)
f0 = float(y_mc.mean())

def cmean(x_query, dim, h_factor=0.15):
    h = X_mc[:, dim].std() * h_factor
    out = np.zeros(len(x_query))
    for k, xq in enumerate(x_query):
        w = np.exp(-0.5 * ((X_mc[:, dim] - xq) / h)**2)
        out[k] = (w * y_mc).sum() / w.sum()
    return out

X_te = X3[test_set]
y_actual = y_all[test_set]
y_full   = gp.predict(X_te)
g_W_te   = cmean(X_te[:,0], 0) - f0
g_KV_te  = cmean(X_te[:,1], 1) - f0
g_KVD_te = cmean(X_te[:,2], 2) - f0
y_add    = f0 + g_W_te + g_KV_te + g_KVD_te

wbits_te  = wbits_all[test_set]
kvbits_te = kvbits_all[test_set]
kvdim_te  = kvdim_all[test_set]
eff_kv_te = eff_kv_all[test_set]

# ─── ε bounds on 150 test ────────────────────────────────────────────────────
def r2(y, yp):
    ss = max(np.sum((y - y.mean())**2), 1e-30)
    return 1 - np.sum((y-yp)**2)/ss
def rmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)))

residual_full = y_actual - y_full
residual_add  = y_actual - y_add
eps_inf  = float(np.max(np.abs(residual_add)))
eps_2    = float(np.sqrt(np.mean(residual_add**2)))
eps_95   = float(np.percentile(np.abs(residual_add), 95))
eps_inf_full = float(np.max(np.abs(residual_full)))
eps_2_full   = float(np.sqrt(np.mean(residual_full**2)))

print("\n" + "="*100)
print("ε BOUNDS on 150 test samples (residual = y_actual − y_pred)")
print("="*100)
print(f"\n  Additive (Hoeffding) prediction y_add:")
print(f"    ε_∞  = {eps_inf:.5f}  ({eps_inf/np.ptp(y_actual)*100:.1f}% of y span)")
print(f"    ε_95 = {eps_95:.5f}")
print(f"    ε_2  = {eps_2:.5f}")
print(f"    R²   = {r2(y_actual, y_add):.4f},  RMSE = {rmse(y_actual, y_add):.4f}")
print(f"    Var(r)/Var(y_actual) = {np.var(residual_add)/np.var(y_actual)*100:.1f}%")
print(f"\n  ARD-GP full prediction y_full (reference):")
print(f"    ε_∞  = {eps_inf_full:.5f}")
print(f"    ε_2  = {eps_2_full:.5f}")
print(f"    R²   = {r2(y_actual, y_full):.4f},  RMSE = {rmse(y_actual, y_full):.4f}")

print(f"\n  Theorem implication (for ANY candidate set X):")
print(f"    Pareto(y_add) ⊆ 2ε_∞-Pareto(y_actual)  with 2ε_∞ = {2*eps_inf:.4f}")
print(f"    Probabilistic (L²): typical gap ≤ 2ε_2 = {2*eps_2:.4f}")

# ─── Figures ─────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')

# Fig 1: pred vs actual scatter — overall + per-axis (wbits, eff_kvbits)
fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

def scatter_pred_vs_actual(ax, ypred, label, color_axis, color_axis_label, eps_inf, eps_2):
    sc = ax.scatter(y_actual, ypred, c=color_axis, cmap='viridis', s=28, alpha=0.85,
                    edgecolor='black', linewidth=0.3)
    lo = min(y_actual.min(), ypred.min()); hi = max(y_actual.max(), ypred.max())
    ax.plot([lo,hi], [lo,hi], 'k--', lw=1.2, label='y_pred = y_actual')
    # ε corridors (around the diagonal in pred-vs-actual; shows ±ε on the predicted axis)
    xx = np.array([lo, hi])
    ax.fill_between(xx, xx-eps_inf, xx+eps_inf, color='gray', alpha=0.10, label=f'±ε_∞={eps_inf:.4f}')
    ax.fill_between(xx, xx-eps_2,   xx+eps_2,   color='gray', alpha=0.22, label=f'±ε_2={eps_2:.4f}')
    rmse_v = rmse(y_actual, ypred); r2_v = r2(y_actual, ypred)
    ax.set_xlabel('y_actual'); ax.set_ylabel(f'y_pred ({label})')
    ax.set_title(f'{label}  (R²={r2_v:.4f}, RMSE={rmse_v:.4f})\ncolored by {color_axis_label}',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_xlim([lo*0.95, hi*1.05]); ax.set_ylim([lo*0.95, hi*1.05])
    plt.colorbar(sc, ax=ax, fraction=0.04, label=color_axis_label)

# Top row: ARD-GP full
scatter_pred_vs_actual(axes[0,0], y_full, 'ARD-GP full',
                       wbits_te,  'wbits',     eps_inf_full, eps_2_full)
scatter_pred_vs_actual(axes[0,1], y_full, 'ARD-GP full',
                       eff_kv_te, 'eff_kvbits', eps_inf_full, eps_2_full)
scatter_pred_vs_actual(axes[0,2], y_full, 'ARD-GP full',
                       wbits_te + eff_kv_te, 'wbits + eff_kvbits', eps_inf_full, eps_2_full)
# Bottom row: Additive (Hoeffding)
scatter_pred_vs_actual(axes[1,0], y_add, 'Additive y_add',
                       wbits_te,  'wbits',     eps_inf, eps_2)
scatter_pred_vs_actual(axes[1,1], y_add, 'Additive y_add',
                       eff_kv_te, 'eff_kvbits', eps_inf, eps_2)
scatter_pred_vs_actual(axes[1,2], y_add, 'Additive y_add',
                       wbits_te + eff_kv_te, 'wbits + eff_kvbits', eps_inf, eps_2)

plt.suptitle('Prediction vs Actual on 150 test samples — colored by complexity axis',
             fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/08_eps_pred_vs_actual.png', **PLT_KW); plt.close()

# Fig 2: residual distribution + residual vs each complexity axis
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

ax = axes[0]
ax.hist(residual_add, bins=25, color='#9B59B6', alpha=0.85, edgecolor='#333', label='r = y_actual − y_add')
ax.axvline(eps_inf,  color='#E64B35', ls='-',  lw=2, label=f'+ε_∞={eps_inf:.4f}')
ax.axvline(-eps_inf, color='#E64B35', ls='-',  lw=2)
ax.axvline(eps_2,    color='#3C5488', ls='--', lw=2, label=f'+ε_2={eps_2:.4f}')
ax.axvline(-eps_2,   color='#3C5488', ls='--', lw=2)
ax.set_xlabel('Residual r = y_actual − y_add'); ax.set_ylabel('# test samples')
ax.set_title('Residual distribution\n(150 test)', fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25, lw=0.5)

for ax, comp, lbl in [(axes[1], wbits_te,  'wbits'),
                      (axes[2], eff_kv_te, 'eff_kvbits'),
                      (axes[3], wbits_te + eff_kv_te, 'wbits + eff_kvbits')]:
    ax.scatter(comp, residual_add, c='#9B59B6', s=22, alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(eps_inf,  color='#E64B35', ls='-',  lw=1.5, label=f'±ε_∞')
    ax.axhline(-eps_inf, color='#E64B35', ls='-',  lw=1.5)
    ax.axhline(eps_2,    color='#3C5488', ls='--', lw=1.5, label=f'±ε_2')
    ax.axhline(-eps_2,   color='#3C5488', ls='--', lw=1.5)
    ax.set_xlabel(lbl); ax.set_ylabel('r = y_actual − y_add')
    ax.set_title(f'Residual vs {lbl}', fontweight='bold')
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle(f'Additive prediction residual on 150 test samples '
             f'(ε_∞={eps_inf:.4f}, ε_2={eps_2:.4f})',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/08_eps_residual_vs_complexity.png', **PLT_KW); plt.close()

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n" + "="*100)
print("SUMMARY (150 test samples; only ε measurement, no per-150 Pareto comparison)")
print("="*100)
print(f"\n  Additive (Hoeffding) prediction:")
print(f"    ε_∞ = {eps_inf:.5f},  ε_2 = {eps_2:.5f},  R² = {r2(y_actual, y_add):.4f}")
print(f"  ARD-GP full prediction:")
print(f"    ε_∞ = {eps_inf_full:.5f},  ε_2 = {eps_2_full:.5f},  R² = {r2(y_actual, y_full):.4f}")
print(f"\n  Theorem (for ANY candidate set X):")
print(f"    Pareto(y_add) ⊆ 2ε_∞-Pareto(y_actual)  with 2ε_∞ = {2*eps_inf:.4f}")
print(f"    Tighter L² (typical): 2ε_2 = {2*eps_2:.4f}")
print(f"\n  150 sample은 Pareto 직접 비교에는 너무 적음 → 본 분석은 ε 측정에만 집중.")
print(f"  실제 NAS workflow 상의 candidate set (e.g. per-method PF Cartesian)에 ")
print(f"  theorem 적용 가능: combined-PF가 true-PF의 2ε corridor 내부에 위치 보장.")

print(f"\nFigures saved:")
print(f"  08_eps_pred_vs_actual.png         — pred vs actual scatter (full / additive × 3 color axes)")
print(f"  08_eps_residual_vs_complexity.png — residual hist + residual vs each complexity axis")
print("Done.\n")
