"""verify_ardgp_coordinate_monotonicity.py

Theorem 3 — Case A (ARD-GP as final scorer with corridor 2ε_GP = 0.051) requires
the surrogate G(z_W, z_KV, z_KVD) itself to be coordinate-wise nondecreasing
on the candidate-relevant region — NOT just its 1D main effects.

This script computes ∂F̂_GP/∂z_i on the candidate-relevant region (covered by
the 200 measured joint samples + a Latin-hypercube grid) via finite differences
and reports the fraction of points where each partial derivative is positive.
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
bounds_jsd = np.array([[xW3.min(), xW3.max()],
                       [xKV3.min(), xKV3.max()],
                       [xKVD.min(), xKVD.max()]])

# Same fixed split as 02
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

# Fit ARD-GP on 50-train pool
def fit_ard_gp(X, y, n_dim, n_restarts=20):
    kernel = (C(1.0,(1e-4,1e2)) *
              SKRBF(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-4,1e4)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9,1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y); return gp

print("Fitting ARD-GP on 50-train pool...")
gp = fit_ard_gp(X3[train_pool], y_all[train_pool], 3, n_restarts=20)
ls = np.array(gp.kernel_.k1.k2.length_scale)
print(f"  Length scales: l_W={ls[0]:.4f}, l_KV={ls[1]:.4f}, l_KVD={ls[2]:.4f}")

# ─── Coordinate-wise monotonicity audit ───────────────────────────────────
def partial_derivative(gp, X, dim, h=None):
    """∂F̂_GP/∂z_dim via central differences."""
    if h is None:
        h = (bounds_jsd[dim, 1] - bounds_jsd[dim, 0]) * 1e-3
    Xp = X.copy(); Xp[:, dim] += h
    Xm = X.copy(); Xm[:, dim] -= h
    return (gp.predict(Xp) - gp.predict(Xm)) / (2 * h)

# Audit point sets:
#   (1) 200 measured joint samples (in-distribution / candidate-relevant region)
#   (2) Latin-hypercube within bounds (uniform fill)
print("\n" + "="*78)
print("ARD-GP coordinate-wise monotonicity audit")
print("="*78)

# Latin-hypercube
rng = np.random.RandomState(0)
n_lh = 5000
u = (rng.permutation(np.arange(n_lh).reshape(-1,1).repeat(3, axis=1).T).T + rng.uniform(0,1,(n_lh,3))) / n_lh
X_lh = bounds_jsd[:,0] + u * (bounds_jsd[:,1] - bounds_jsd[:,0])

audit_sets = [('200 measured joint samples', X3),
              (f'{n_lh} Latin-hypercube fill', X_lh)]

print(f"\n{'audit set':32s}  {'axis':5s}  {'mean ∂':>10s}  {'min ∂':>10s}  "
      f"{'frac ≥0':>10s}  {'frac ≥0 (95%-CI)':>22s}")
print("─"*116)
results = {}
for set_name, X_audit in audit_sets:
    results[set_name] = {}
    for d, name in enumerate(['W','KV','KVD']):
        dF = partial_derivative(gp, X_audit, d)
        frac_pos = float((dF >= 0).mean())
        n = len(dF)
        ci_lo = max(0.0, frac_pos - 1.96*np.sqrt(frac_pos*(1-frac_pos)/n))
        ci_hi = min(1.0, frac_pos + 1.96*np.sqrt(frac_pos*(1-frac_pos)/n))
        results[set_name][name] = dict(dF=dF, mean=dF.mean(), min=dF.min(),
                                       frac_pos=frac_pos, ci=(ci_lo, ci_hi))
        print(f"{set_name:32s}  {name:5s}  {dF.mean():>+10.4f}  {dF.min():>+10.4f}  "
              f"{frac_pos:>10.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
    print()

# Identify violation regions on candidate-relevant set
print("="*78)
print("Coordinate-wise monotonicity verdict (candidate-relevant region)")
print("="*78)
for d, name in enumerate(['W','KV','KVD']):
    r = results['200 measured joint samples'][name]
    if r['frac_pos'] >= 0.99:
        verdict = 'CONFIRMED ✓ (≥99% pos)'
    elif r['frac_pos'] >= 0.90:
        verdict = f"MOSTLY ({r['frac_pos']*100:.1f}% pos) — Case A claim with caveat"
    else:
        verdict = f"VIOLATED ({r['frac_pos']*100:.1f}% pos) — use Case B / C"
    print(f"  ∂F̂_GP/∂z_{name}: {verdict}")

# ─── Plot ───────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'W':'#E64B35', 'KV':'#4DBBD5', 'KVD':'#00A087'}

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for c, name in enumerate(['W','KV','KVD']):
    for r_, (set_label, X_audit) in enumerate(audit_sets):
        ax = axes[r_, c]
        dF = results[set_label][name]['dF']
        ax.hist(dF, bins=60, color=COL[name], alpha=0.85, edgecolor='#333')
        ax.axvline(0, color='black', lw=1.2)
        ax.set_xlabel(f'∂F̂_GP / ∂z_{name}')
        ax.set_ylabel('# points')
        fp = results[set_label][name]['frac_pos']
        ax.set_title(f'{set_label}\n∂/∂z_{name}: {fp*100:.1f}% ≥ 0',
                     fontweight='bold', color=COL[name])
        ax.grid(True, alpha=0.25, lw=0.5)
plt.suptitle('ARD-GP coordinate-wise monotonicity audit (Theorem 3 Case A)',
             fontweight='bold', y=1.005)
plt.tight_layout()
out = f'{FIG_DIR}/verify_ardgp_monotonicity.png'
plt.savefig(out, dpi=170, bbox_inches='tight')
plt.close()
print(f"\nFigure: {out}")
print("Done.")
