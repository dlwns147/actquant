"""verify_main_effect_monotonicity.py

Theorem 2 (PF-product coverage) requires the additive main-effect functions
h_i(z_i) to be monotone increasing in z_i so that per-method Pareto fronts
defined on raw z_i coincide with PFs defined on h_i(z_i).

This script verifies the monotonicity numerically:
  1. Refit ARD-GP on the same 50-train pool as 02_ard_gp_analysis.py
  2. Compute the Hoeffding 1D main effect g_i(z_i) = E_{z_{-i}}[F(z)] - f_0
     on a fine grid of z_i values via Monte-Carlo marginalization
  3. Numerically differentiate g_i to obtain gradient signs
  4. Report range and sign of dg_i/dz_i for i ∈ {W, KV, KVD}
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

# ─── Data loading (same as 02) ──────────────────────────────────────────────
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
def match_metric(comp_vals, pf):
    return np.array([pf[np.argmin(np.abs(pf[:, 1] - c)), 0] for c in comp_vals])

BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY  = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'

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
y_all = y_all[v3]
N = len(y_all)
X3 = np.column_stack([xW3, xKV3, xKVD])
bounds_jsd = [[xW3.min(), xW3.max()], [xKV3.min(), xKV3.max()], [xKVD.min(), xKVD.max()]]
print(f"  N={N}  bounds: W=[{bounds_jsd[0][0]:.3f},{bounds_jsd[0][1]:.3f}], "
      f"KV=[{bounds_jsd[1][0]:.3f},{bounds_jsd[1][1]:.3f}], "
      f"KVD=[{bounds_jsd[2][0]:.3f},{bounds_jsd[2][1]:.3f}]")

# ─── Same 50/150 split as script 02 ─────────────────────────────────────────
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
print(f"  Fixed split: {len(train_pool)} train pool")

# ─── Fit ARD-GP ─────────────────────────────────────────────────────────────
def fit_ard_gp(X, y, n_dim, n_restarts=20):
    kernel = (C(1.0,(1e-4,1e2)) *
              SKRBF(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-4,1e4)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9,1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y); return gp

print("\nFitting ARD-GP on 50-train pool (JSD input)...")
gp = fit_ard_gp(X3[train_pool], y_all[train_pool], 3, n_restarts=20)
ls = np.array(gp.kernel_.k1.k2.length_scale)
print(f"  Length scales: l_W={ls[0]:.4f}, l_KV={ls[1]:.4f}, l_KVD={ls[2]:.4f}")

# ─── 1D main effect via MC marginalization ──────────────────────────────────
print("\n" + "="*78)
print("Verifying monotonicity of 1D main effects g_i(z_i)")
print("="*78)

rng = np.random.RandomState(0); n_mc = 5000
X_mc = np.array([rng.uniform(b[0], b[1], n_mc) for b in bounds_jsd]).T
f0   = float(gp.predict(X_mc).mean())
print(f"  f_0 (global mean) = {f0:.5f}")

def main_effect_curve(dim, n_grid=40, n_mc_inner=8000, seed=42):
    """g_i(z_i) = E_{z_{-i}}[F(z_i, z_{-i})] - f_0 on a fine grid.

    Use a SHARED set of MC marginalization points across all grid values
    (correlated MC) to reduce variance of d g_i / d z_i differences.
    """
    grid = np.linspace(bounds_jsd[dim][0], bounds_jsd[dim][1], n_grid)
    rng_in = np.random.RandomState(seed)
    # Shared MC samples for the OTHER dims (correlated MC ⇒ much smaller grad noise)
    other = [j for j in range(3) if j != dim]
    Xmc_other = np.zeros((n_mc_inner, 3))
    for j in other:
        Xmc_other[:, j] = rng_in.uniform(bounds_jsd[j][0], bounds_jsd[j][1], n_mc_inner)
    means = np.zeros(n_grid)
    for k, xk in enumerate(grid):
        Xmc = Xmc_other.copy()
        Xmc[:, dim] = xk
        means[k] = gp.predict(Xmc).mean()
    return grid, means - f0  # g_i(z_i)

print("\nComputing 1D main effect curves (40 grid × 8000 shared-MC each, correlated MC)...")
grids, gs = {}, {}
for d, name in enumerate(['W','KV','KVD']):
    grids[name], gs[name] = main_effect_curve(d)

# ─── Gradient computation (central difference) ──────────────────────────────
def gradient(grid, g):
    """Central difference; returns gradient at interior points."""
    return (g[2:] - g[:-2]) / (grid[2:] - grid[:-2])

print("\n" + "─"*78)
print(f"  {'axis':5s}  {'min grad':>11s}  {'max grad':>11s}  "
      f"{'min sign':>10s}  {'max sign':>10s}  {'all positive?':>14s}")
print("─"*78)

results = {}
for name in ['W', 'KV', 'KVD']:
    grad = gradient(grids[name], gs[name])
    gmin, gmax = float(grad.min()), float(grad.max())
    sign_min = '+' if gmin > 0 else ('−' if gmin < 0 else '0')
    sign_max = '+' if gmax > 0 else ('−' if gmax < 0 else '0')
    all_pos = bool(np.all(grad > 0))
    n_neg = int(np.sum(grad < 0))
    n_total = len(grad)
    results[name] = dict(grad=grad, gmin=gmin, gmax=gmax, all_pos=all_pos,
                         n_neg=n_neg, n_total=n_total)
    print(f"  {name:5s}  {gmin:>+11.5f}  {gmax:>+11.5f}  "
          f"{sign_min:>10s}  {sign_max:>10s}  "
          f"{('YES' if all_pos else f'NO ({n_neg}/{n_total} neg)'):>14s}")
print("─"*78)

print(f"\n  Theorem 2 condition (M2): h_i(z_i) monotone in z_i")
all_axes_pos = all(results[n]['all_pos'] for n in ['W','KV','KVD'])
print(f"  → Monotonicity verdict: {'CONFIRMED ✓' if all_axes_pos else 'VIOLATED ✗'}")
if all_axes_pos:
    print(f"  → raw z_i 기준 PF = h_i(z_i) 기준 PF (Theorem 2 valid)")
else:
    print(f"  → 일부 region 에서 monotonicity 깨짐. local archive 를 h_i 기준 re-rank 필요")

# ─── Plot ───────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'W':'#E64B35', 'KV':'#4DBBD5', 'KVD':'#00A087'}

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for c, name in enumerate(['W','KV','KVD']):
    g = gs[name]; grid = grids[name]; grad = results[name]['grad']
    ax = axes[0, c]
    ax.plot(grid, g, '-', color=COL[name], lw=2.6)
    ax.axhline(0, color='black', lw=0.7, ls=':')
    ax.set_xlabel(f'JSD_{name}'); ax.set_ylabel(f'g_{name}(z_{name}) = E[F | z_{name}] − f_0')
    ax.set_title(f'(a) Main effect g_{name}(z_{name})', fontweight='bold')
    ax.grid(True, alpha=0.25, lw=0.5)

    ax = axes[1, c]
    grid_int = grid[1:-1]
    ax.plot(grid_int, grad, '-o', color=COL[name], lw=2, markersize=4)
    ax.axhline(0, color='black', lw=0.8)
    ymin, ymax = grad.min(), grad.max()
    ax.fill_between(grid_int, 0, grad, where=(grad < 0), color='#E64B35', alpha=0.25, label='gradient < 0')
    ax.fill_between(grid_int, 0, grad, where=(grad >= 0), color='#3C5488', alpha=0.20, label='gradient ≥ 0')
    sign_str = 'all +' if results[name]['all_pos'] else f"{results[name]['n_neg']}/{results[name]['n_total']} neg"
    ax.set_xlabel(f'JSD_{name}'); ax.set_ylabel(f'dg_{name}/dz_{name}')
    ax.set_title(f'(b) Gradient (range [{ymin:+.3f}, {ymax:+.3f}], {sign_str})',
                 fontweight='bold', color=COL[name])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle('Verification of Theorem 2 condition (M2): h_i(z_i) monotone increasing in z_i',
             fontweight='bold', y=1.005)
plt.tight_layout()
out = f'{FIG_DIR}/verify_monotonicity.png'
plt.savefig(out, dpi=170, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {out}")
print("Done.")
