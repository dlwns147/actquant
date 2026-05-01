"""analysis_v3_hier_budget.py — Direct vs Hierarchical at equal TOTAL measurement budget.

X-axis = total measurements B = N_3way + N_pair.
  • Direct          : N_3way = B,         N_pair = 0      (only feasible for B ≤ 50, 3-way pool size)
  • Hier-fix3way27  : N_3way = 27 (min),  N_pair = B − 27 (varies pair from 6 to 200)
  • Hier-fix3way50  : N_3way = 50 (max),  N_pair = B − 50
  • Hier-50/50      : N_3way = N_pair = B/2  (only when B/2 ∈ [27, 50])

Best hierarchical config from constrained analysis: pair=WD (W+KVD) with TPS kernel
+ linear residual on KV.  Direct surrogate: cubic+linear RBF (sorted-index input).
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

# ─── Data loading ─────────────────────────────────────────────────────────────
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
AWQ_3WAY  = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
AWQ_W_KVD = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim/results.csv'

with open(f'{BASE}/config/llama.json') as f: config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

print("Loading PFs and 3-way + WD pair data...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
pf_W_m, pf_KV_m, pf_KVDIM_m = (p[np.argsort(p[:, 0])] for p in (pf_W, pf_KV, pf_KVDIM))
JSD_KV_def = pf_KV_m[0, 0]

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

# WD pair data
mat_wd = load_csv(AWQ_W_KVD); N_wd = mat_wd.shape[1]
y_WD = mat_wd[12, :N_wd]; v_wd = ~np.isnan(y_WD)
xW_WD   = match_metric(mat_wd[0, :N_wd], pf_W   )[v_wd]
xKVD_WD = match_metric(mat_wd[4, :N_wd], pf_KVDIM)[v_wd]
y_WD    = y_WD[v_wd]
print(f"  3-way N={N}; WD pair N={len(y_WD)}")

# ─── 27-grid + maximin = 50 train pool / 150 test ─────────────────────────────
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

# ─── Helpers ──────────────────────────────────────────────────────────────────
def r2(y, yp):
    ss_t = max(np.sum((y - y.mean())**2), 1e-30)
    return 1 - np.sum((y-yp)**2)/ss_t
def rmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)))

X_te_jsd = X3_jsd[test_set]; X_te_idx = X3_idx[test_set]; y_te = y3[test_set]

# ─── Direct evaluation (RBF on sorted-index input, OR ARD-GP on JSD input) ────
def _3way_subsample(n_3way, seed):
    rng = np.random.RandomState(seed)
    n_extra = n_3way - 27
    if n_extra == 0: return grid_samples
    elif n_extra == 23: return train_pool
    elif 0 < n_extra < 23:
        return np.concatenate([grid_samples, extras[rng.choice(23, n_extra, replace=False)]])
    elif n_extra < 0:
        return grid_samples[rng.choice(27, n_3way, replace=False)]
    return train_pool

def direct_eval(n_3way, n_seeds=50, surrogate='RBF'):
    """surrogate ∈ {'RBF', 'ARD-GP'}"""
    r2s, rmses = [], []
    for seed in range(n_seeds):
        tr3 = _3way_subsample(n_3way, seed)
        try:
            if surrogate == 'RBF':
                m = RBF(kernel='cubic', tail='linear', lb=lb_idx, ub=ub_idx)
                m.fit(X3_idx[tr3], y3[tr3])
                yp = m.predict(X_te_idx).ravel()
            elif surrogate == 'ARD-GP':
                kernel = (C(1.0, (1e-4, 1e2)) *
                          SKRBF(length_scale=[1.0]*3, length_scale_bounds=(1e-3, 1e3)) +
                          WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                              n_restarts_optimizer=5, alpha=1e-8)
                gp.fit(X3_jsd[tr3], y3[tr3])
                yp = gp.predict(X_te_jsd)
            r2s.append(r2(y_te, yp)); rmses.append(rmse(y_te, yp))
        except Exception:
            r2s.append(np.nan); rmses.append(np.nan)
    return np.array(r2s), np.array(rmses)

# ─── Hierarchical evaluation (WD-TPS-linear) ─────────────────────────────────
def hier_eval(pair_N, threeway_N, n_seeds=50):
    r2s, rmses = [], []
    if pair_N < 6 or threeway_N < 6: return np.array([np.nan]), np.array([np.nan])
    for seed in range(n_seeds):
        rng3 = np.random.RandomState(seed)
        rng_p = np.random.RandomState(seed + 100000)
        # 3-way subsample
        n_extra = threeway_N - 27
        if n_extra == 0: tr3 = grid_samples
        elif n_extra == 23: tr3 = train_pool
        elif 0 < n_extra < 23:
            tr3 = np.concatenate([grid_samples, extras[rng3.choice(23, n_extra, replace=False)]])
        elif n_extra < 0:  # threeway_N < 27, sub-grid
            tr3 = grid_samples[rng3.choice(27, threeway_N, replace=False)]
        else:
            tr3 = train_pool  # capped at 50
        # Pair subsample
        n_p = min(pair_N, len(y_WD))
        pair_idx = rng_p.choice(len(y_WD), n_p, replace=False) if n_p < len(y_WD) else np.arange(len(y_WD))
        # Train TPS pair on WD (W, KVD)
        xa = xW_WD[pair_idx]; xb = xKVD_WD[pair_idx]; yp_pair = y_WD[pair_idx]
        X2 = np.column_stack([xa, xb])
        try:
            m_pair = RBF(kernel='tps', tail='linear', lb=X2.min(0), ub=X2.max(0))
            m_pair.fit(X2, yp_pair)
            # Apply pair to 3-way train (W=col0, KVD=col2)
            x3_tr = X3_jsd[tr3]
            yp_pair_tr = m_pair.predict(np.column_stack([x3_tr[:,0], x3_tr[:,2]])).ravel()
            yp_pair_te = m_pair.predict(np.column_stack([X_te_jsd[:,0], X_te_jsd[:,2]])).ravel()
            # Linear residual on KV (col1)
            r_tr = y3[tr3] - yp_pair_tr
            c = np.polyfit(x3_tr[:,1], r_tr, 1)
            yp_t = yp_pair_te + np.polyval(c, X_te_jsd[:,1])
            r2s.append(r2(y_te, yp_t)); rmses.append(rmse(y_te, yp_t))
        except Exception:
            r2s.append(np.nan); rmses.append(np.nan)
    return np.array(r2s), np.array(rmses)

# ─── Sweeps ────────────────────────────────────────────────────────────────────
print("\nSweeping budgets...")

# Limit total budget to B ≤ 50 (Direct's data feasibility ceiling)
# Direct: B = N_3way ∈ [27, 50]
print("\n[Direct-RBF (cubic+linear, sorted-idx)]:")
direct_results = []
for B in [27, 30, 33, 35, 40, 45, 50]:
    r2s, rms = direct_eval(B, surrogate='RBF')
    direct_results.append((B, np.nanmedian(r2s), np.nanpercentile(r2s,10), np.nanpercentile(r2s,90),
                           np.nanmedian(rms)))
    print(f"  Direct-RBF (B={B}): R²={direct_results[-1][1]:.4f}  RMSE={direct_results[-1][4]:.4f}")

print("\n[Direct-ARD-GP (sklearn ARD, JSD input)]:")
direct_ard_results = []
for B in [27, 30, 33, 35, 40, 45, 50]:
    r2s, rms = direct_eval(B, surrogate='ARD-GP', n_seeds=30)
    direct_ard_results.append((B, np.nanmedian(r2s), np.nanpercentile(r2s,10), np.nanpercentile(r2s,90),
                               np.nanmedian(rms)))
    print(f"  Direct-ARD-GP (B={B}): R²={direct_ard_results[-1][1]:.4f}  RMSE={direct_ard_results[-1][4]:.4f}")

# ── Hier sweep: for each B, vary N_3way ∈ {6, 9, ..., 50} with N_pair = B − N_3way
print()
budgets_hier = [27, 30, 33, 35, 40, 45, 50]
n3_candidates = [6, 9, 12, 15, 18, 21, 24, 27, 30, 35, 40, 45, 50]
hier_grid = []   # (B, n_3way, n_pair, r2_med, r2_lo, r2_hi, rmse_med)
hier_best = []   # best (B, ...) per budget

for B in budgets_hier:
    feasible = [(n3, B - n3) for n3 in n3_candidates if 6 <= n3 <= 50 and 6 <= (B - n3) <= 200]
    print(f"  B={B:3d}: ", end="")
    best = None
    for n3, n_p in feasible:
        r2s, rms = hier_eval(n_p, n3)
        med = float(np.nanmedian(r2s))
        rec = (B, n3, n_p, med, float(np.nanpercentile(r2s,10)),
               float(np.nanpercentile(r2s,90)), float(np.nanmedian(rms)))
        hier_grid.append(rec)
        if best is None or med > best[3]: best = rec
        print(f"({n3},{n_p}):{med:.3f} ", end="")
    if best is not None:
        hier_best.append(best)
        print(f"  → best (n3={best[1]}, n_p={best[2]}): R²={best[3]:.4f}")
    else: print(" no feasible split")

# ─── Compare at equal budget ──────────────────────────────────────────────────
print("\n" + "="*120)
print("Direct (RBF) vs Direct-ARD-GP vs Hier-best at EQUAL TOTAL BUDGET")
print("="*120)
print(f"  {'B':4s}  {'Direct-RBF':12s}  {'Direct-ARD-GP':14s}  {'Hier-best (n3*, n_p*)':30s}  {'best of 3':10s}")
def lookup(results, B):
    for r in results:
        if r[0] == B: return r[1]
    return None
for B in budgets_hier:
    d   = lookup(direct_results,     B)
    da  = lookup(direct_ard_results, B)
    h   = next((rec for rec in hier_best if rec[0] == B), None)
    if d is None or h is None: continue
    h_r2 = h[3]; vals = {'Direct-RBF': d, 'Direct-ARD-GP': da, 'Hier': h_r2}
    best = max(vals, key=vals.get)
    print(f"  B={B:3d}  {d:.4f}      {da:.4f}        Hier {h_r2:.4f} (n3={h[1]:2d}, n_p={h[2]:2d})       BEST = {best} ({vals[best]:.4f})")

# ─── Figures ──────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))

# (a) Line plot: Direct vs Hier-best (best over all N_3way)
ax = axes[0]
def line(ax, results, label, color, marker='o', ls='-', y_idx=1, lo_idx=2, hi_idx=3):
    if not results: return
    xs  = [r[0] for r in results]; ys = [r[y_idx] for r in results]
    if lo_idx is not None:
        los = [r[lo_idx] for r in results]; his = [r[hi_idx] for r in results]
        ax.fill_between(xs, los, his, color=color, alpha=0.13)
    ax.plot(xs, ys, marker=marker, color=color, label=label, lw=2.4, ms=8, ls=ls,
            markeredgecolor='white', markeredgewidth=0.7)

line(ax, direct_results,     'Direct-RBF  (cubic+linear, sorted-idx)', '#3C5488', 'o', '-')
line(ax, direct_ard_results, 'Direct-ARD-GP  (sklearn ARD, JSD)',     '#00A087', 'D', '-')
xs_h = [r[0] for r in hier_best]; ys_h = [r[3] for r in hier_best]
los_h = [r[4] for r in hier_best]; his_h = [r[5] for r in hier_best]
ax.fill_between(xs_h, los_h, his_h, color='#E64B35', alpha=0.13)
ax.plot(xs_h, ys_h, '-s', color='#E64B35', lw=2.4, ms=8,
        markeredgecolor='white', markeredgewidth=0.7,
        label='Hier-WD-TPS  (N_3way* + N_pair* optimised)')

# Annotate Hier optimal split per B
for rec in hier_best:
    B, n3, n_p, med, *_ = rec
    ax.annotate(f'({n3},{n_p})', (B, med - 0.003), fontsize=7.5,
                ha='center', color='#E64B35', fontweight='bold')
# Annotate Δ (Hier − Direct) above each B
for r_d in direct_results:
    h = next((r for r in hier_best if r[0] == r_d[0]), None)
    if h is None: continue
    delta = h[3] - r_d[1]
    color = '#E64B35' if delta > 0 else '#3C5488'
    ax.annotate(f'Δ={delta:+.4f}',
                (r_d[0], max(r_d[1], h[3]) + 0.0018),
                fontsize=7.5, ha='center', color=color, fontweight='bold')

ax.set_xlabel('Total measurements  B = N_3way + N_pair', fontsize=11)
ax.set_ylabel('Test R² on 150 hold-out\n(median, [10%, 90%])', fontsize=11)
ax.set_title('(a) Direct vs Hier-best  (B ≤ 50)', fontweight='bold')
ax.set_xticks([27, 30, 33, 35, 40, 45, 50])
ax.legend(fontsize=9.5, loc='lower right'); ax.grid(True, alpha=0.25, lw=0.5)

# (b) Heatmap: Hier R² over (B, N_3way) grid
ax = axes[1]
budgets_h = sorted(set(r[0] for r in hier_grid))
n3s_h     = sorted(set(r[1] for r in hier_grid))
M = np.full((len(n3s_h), len(budgets_h)), np.nan)
for B, n3, n_p, med, *_ in hier_grid:
    i = n3s_h.index(n3); j = budgets_h.index(B)
    M[i, j] = med
finite = M[np.isfinite(M)]
vmin, vmax = (finite.min(), finite.max()) if finite.size else (0.9, 1.0)
im = ax.imshow(M, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
for i in range(len(n3s_h)):
    for j in range(len(budgets_h)):
        v = M[i, j]
        if not np.isnan(v):
            color = 'white' if v < vmin + 0.55*(vmax-vmin) else 'black'
            ax.text(j, i, f"{v:.3f}", ha='center', va='center', fontsize=8, color=color)
# Mark Hier-best per B with red box
for rec in hier_best:
    B, n3, *_ = rec
    j = budgets_h.index(B); i = n3s_h.index(n3)
    ax.add_patch(plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                fill=False, edgecolor='red', linewidth=2.5))
# Mark Direct points (N_3way = B, N_pair = 0) — only when B is in n3s_h
for B in budgets_h:
    if B in n3s_h:
        i = n3s_h.index(B); j = budgets_h.index(B)
        ax.text(j, i, '★', ha='center', va='center', fontsize=14, color='gold')

ax.set_xticks(range(len(budgets_h))); ax.set_xticklabels(budgets_h)
ax.set_yticks(range(len(n3s_h))); ax.set_yticklabels(n3s_h)
ax.set_xlabel('Total measurements  B', fontsize=11)
ax.set_ylabel('N_3way (per Hier evaluation)', fontsize=11)
ax.set_title('(b) Hier R² heatmap  (red box = best per B; ★ = "all 3-way" = Direct)',
             fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.04, label='Test R²')

plt.suptitle('Direct vs Hierarchical at equal TOTAL budget — N_3way swept along with B '
             '(150 hold-out, 50 seeds; pair=WD/TPS, residual=linear)',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_hier_vs_direct.png', **PLT_KW); plt.close()

print(f"\nFigure saved: 02_hier_vs_direct.png")
print("Done.\n")
