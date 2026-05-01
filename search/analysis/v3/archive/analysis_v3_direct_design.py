"""analysis_v3_direct_design.py — Sensitivity-weighted train design for Direct surrogate.

Hypothesis: allocating more grid points to high-marginal-range dimensions
(W, KVD) and fewer to low-range (KV) yields a better Direct RBF at the same N_train.

Marginal JSD ranges (from per-method PFs):
  W   : [0.019, 0.657]  span = 0.638  ← largest
  KVD : [0.019, 0.328]  span = 0.309
  KV  : [0.018, 0.136]  span = 0.118  ← smallest
  ratio  W : KVD : KV  ≈  5.4 : 2.6 : 1.0

Designs compared (all build a 50-train pool: K grid + (50-K) maximin extras):
  A 3×3×3 uniform                    K=27   (baseline)
  B 5×2×3 W↑, KV↓                    K=30
  C 4×2×4 W↑, KVD↑, KV↓              K=32
  D 6×2×3 W↑↑, KV↓                   K=36
  E 5×2×4 W↑, KVD↑, KV↓              K=40
  F 4×2×6 KVD↑↑, KV↓                 K=48
  G random (no grid)                 (averaged over 30 seeds)

For each design and N_train ∈ {27, 30, 35, 40, 45, 50}: RBF (sorted-index input)
trained on first N_train of that design's pool, eval on remaining 200 − N_train.
"""
import sys, os, json, csv
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from utils.func import get_net_info
from predictor.rbf import RBF

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

with open(f'{BASE}/config/llama.json') as f: config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

print("Loading PFs and 3-way data...")
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

scale = X3_jsd.std(0) + 1e-10
X3n = X3_jsd / scale

range_W   = xW3.max()  - xW3.min()
range_KV  = xKV3.max() - xKV3.min()
range_KVD = xKVD3.max()- xKVD3.min()
print(f"  Marginal ranges:  W={range_W:.3f}  KVD={range_KVD:.3f}  KV={range_KV:.3f}")
print(f"  Ratio  W:KVD:KV = {range_W/range_KV:.1f} : {range_KVD/range_KV:.1f} : 1.0")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def r2(y, yp):
    ss_t = max(np.sum((y - y.mean())**2), 1e-30)
    return 1 - np.sum((y-yp)**2)/ss_t
def rmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)))

def build_pool(L_W, L_KV, L_KVD, N_total=50, maximin_space='normalized'):
    """Build train pool of size N_total using L_W × L_KV × L_KVD quantile grid + maximin extras.

    maximin_space: 'normalized' (per-dim std-scaled, isotropic exploration) or
                   'raw' (unscaled JSD distance, naturally over-samples high-range dims).
    """
    def lvls(L):
        if L == 1: return np.array([0.5])
        return np.linspace(0.1, 0.9, L)
    qs_W, qs_KV, qs_KVD = lvls(L_W), lvls(L_KV), lvls(L_KVD)
    qW   = np.quantile(xW3,   qs_W)
    qKV  = np.quantile(xKV3,  qs_KV)
    qKVD = np.quantile(xKVD3, qs_KVD)
    grid = np.array([[w, kv, kvd] for w in qW for kv in qKV for kvd in qKVD])
    K = len(grid)

    # Hungarian in normalized space (consistent grid match irrespective of maximin choice)
    grid_n_for_hungarian = grid / scale
    cost = np.zeros((K, N))
    for j in range(K):
        cost[j] = np.sum((X3n - grid_n_for_hungarian[j])**2, axis=1)
    if K <= N:
        row_ind, col_ind = linear_sum_assignment(cost)
        grid_samples = col_ind.astype(int)
    else:
        grid_samples = np.unique(np.argmin(cost, axis=1))[:K]

    grid_samples = np.unique(grid_samples)
    K_actual = len(grid_samples)

    # Maximin extras — choose space
    X_for_maximin = X3n if maximin_space == 'normalized' else X3_jsd

    N_extras = N_total - K_actual
    if N_extras > 0:
        remaining = np.setdiff1d(np.arange(N), grid_samples)
        extras = []; sel = list(grid_samples)
        for _ in range(N_extras):
            sel_arr = np.array(sel, dtype=int)
            d = np.array([np.min(np.sum((X_for_maximin[r] - X_for_maximin[sel_arr])**2, axis=1)) for r in remaining])
            far = int(np.argmax(d))
            extras.append(int(remaining[far])); sel.append(int(remaining[far]))
            remaining = np.delete(remaining, far)
        pool = np.concatenate([grid_samples, np.array(extras, dtype=int)])
    else:
        pool = grid_samples[:N_total]
    return pool[:N_total], K_actual

def random_pool(N_total, seed):
    rng = np.random.RandomState(seed)
    return rng.choice(N, N_total, replace=False)

def eval_rbf_on_test(train_idx):
    test_idx = np.setdiff1d(np.arange(N), train_idx)
    m = RBF(kernel='cubic', tail='linear', lb=lb_idx, ub=ub_idx)
    m.fit(X3_idx[train_idx], y3[train_idx])
    yp = m.predict(X3_idx[test_idx]).ravel()
    return r2(y3[test_idx], yp), rmse(y3[test_idx], yp), len(test_idx)

# ─── Designs ──────────────────────────────────────────────────────────────────
designs = [
    ('A 3×3×3 uniform (norm-mxmn)',          (3, 3, 3, 'normalized'), '#3C5488'),
    ('B 5×2×3 W↑, KV↓ (norm-mxmn)',          (5, 2, 3, 'normalized'), '#E64B35'),
    ('C 4×2×4 W↑, KVD↑, KV↓ (norm-mxmn)',    (4, 2, 4, 'normalized'), '#4DBBD5'),
    ('D 5×2×4 W+KVD↑, KV↓ (norm-mxmn)',      (5, 2, 4, 'normalized'), '#00A087'),
    ('E 3×3×3 uniform (raw-mxmn)',           (3, 3, 3, 'raw'),        '#9B59B6'),
    ('F 5×2×3 W↑, KV↓ (raw-mxmn)',           (5, 2, 3, 'raw'),        '#F39B7F'),
    ('G 5×2×4 W+KVD↑, KV↓ (raw-mxmn)',       (5, 2, 4, 'raw'),        '#F1C40F'),
    ('H random',                              None,                    '#7F7F7F'),
]
N_TRAINS = [27, 30, 35, 40, 45, 50]

# Build pools
print("\nBuilding train pools per design:")
pools = {}
for name, dims, _ in designs:
    if dims is None:
        pools[name] = None
        print(f"  {name:38s}: random (per-seed)")
    else:
        L_W, L_KV, L_KVD, mxmn = dims
        pool, K = build_pool(L_W, L_KV, L_KVD, N_total=50, maximin_space=mxmn)
        pools[name] = pool
        print(f"  {name:38s}: K_grid={K:2d}, pool size = {len(pool)}, maximin={mxmn}")

# ─── Evaluate ─────────────────────────────────────────────────────────────────
print("\nEvaluating Direct-RBF for each (design, N_train)...")
results = {}
N_SEEDS_RANDOM = 30
for name, dims, _ in designs:
    print(f"\n  {name}:")
    if dims is None:
        for N_tr in N_TRAINS:
            r2s, rmses = [], []
            for seed in range(N_SEEDS_RANDOM):
                tr = random_pool(N_tr, seed)
                R2, RM, _ = eval_rbf_on_test(tr)
                r2s.append(R2); rmses.append(RM)
            results[(name, N_tr)] = (np.array(r2s), np.array(rmses))
            print(f"    N={N_tr:2d}: R²={np.median(r2s):.4f} [{np.percentile(r2s,10):.3f},{np.percentile(r2s,90):.3f}]"
                  f"  RMSE={np.median(rmses):.4f}")
    else:
        pool = pools[name]
        for N_tr in N_TRAINS:
            tr = pool[:N_tr]
            R2, RM, n_te = eval_rbf_on_test(tr)
            results[(name, N_tr)] = (np.array([R2]), np.array([RM]))
            print(f"    N={N_tr:2d}: R²={R2:.4f}  RMSE={RM:.4f}  (test N={n_te})")

# ─── Compare ──────────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("Test R² per (design, N_train)")
print("="*100)
hdr = f"  {'Design':30s}"
for N_tr in N_TRAINS: hdr += f"  N={N_tr:<3d}    "
print(hdr)
for name, _, _ in designs:
    line = f"  {name:30s}"
    for N_tr in N_TRAINS:
        r2s, _ = results[(name, N_tr)]
        med = float(np.median(r2s))
        line += f"  {med:.4f}    "
    print(line)

# Identify best design at each N_train
print("\n" + "="*100)
print("Best design at each N_train")
print("="*100)
print(f"  {'N_tr':5s}  {'Best design':35s}  {'R²':10s}  {'vs uniform Δ':12s}")
for N_tr in N_TRAINS:
    cands = []
    for name, _, _ in designs:
        r2s, _ = results[(name, N_tr)]
        cands.append((name, float(np.median(r2s))))
    cands.sort(key=lambda x: -x[1])
    best = cands[0]
    uniform_r2 = next(c[1] for c in cands if 'uniform' in c[0])
    delta = best[1] - uniform_r2
    print(f"  N={N_tr:2d}  {best[0]:35s}  R²={best[1]:.4f}  Δ={delta:+.4f}")

# ─── Figures ──────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')

# Fig 1: line plot — R² vs N_train per design
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, dims, color in designs:
    xs, med, lo, hi = [], [], [], []
    for N_tr in N_TRAINS:
        r2s, _ = results[(name, N_tr)]
        xs.append(N_tr); med.append(float(np.median(r2s)))
        if len(r2s) > 1:
            lo.append(float(np.percentile(r2s, 10)))
            hi.append(float(np.percentile(r2s, 90)))
        else:
            lo.append(med[-1]); hi.append(med[-1])
    ax.plot(xs, med, '-o', color=color, label=name, lw=2.2, ms=7,
            markeredgecolor='white', markeredgewidth=0.6)
    if any(lo[i] != hi[i] for i in range(len(lo))):
        ax.fill_between(xs, lo, hi, color=color, alpha=0.10)
ax.set_xlabel('N_train (Direct-RBF on sorted-index input)')
ax.set_ylabel('Test R² on hold-out')
ax.set_title('(a) Sensitivity-weighted grid designs vs uniform (Direct surrogate)',
             fontweight='bold')
ax.set_xticks(N_TRAINS); ax.legend(fontsize=8.5, loc='lower right')
ax.grid(True, alpha=0.25, lw=0.5)

# Fig 1b: bar at N=50
ax = axes[1]
N_show = 50
names = [d[0] for d in designs]
r2_vals = [float(np.median(results[(d[0], N_show)][0])) for d in designs]
colors = [d[2] for d in designs]
bars = ax.barh(range(len(names)), r2_vals, color=colors, alpha=0.85, edgecolor='#333')
for i, v in enumerate(r2_vals):
    ax.text(v + 0.0002, i, f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel('Test R² at N_train = 50')
ax.set_title('(b) Design ranking at N_train = 50', fontweight='bold')
ax.set_xlim([min(r2_vals) - 0.002, max(r2_vals) + 0.001])
ax.grid(True, alpha=0.25, lw=0.5, axis='x')
ax.invert_yaxis()
plt.suptitle('Direct surrogate: design effect of allocating more grid to high-range dimensions',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/05_direct_design.png', **PLT_KW); plt.close()

# Fig 2: train-pool 3D scatter for a few designs
fig, axes = plt.subplots(2, 4, figsize=(18, 9), subplot_kw={'projection': '3d'})
ax_list = axes.flatten()
for ax_i, (name, dims, color) in enumerate(designs[:8]):
    ax = ax_list[ax_i]
    pool = pools[name] if dims is not None else random_pool(50, 0)
    test_idx = np.setdiff1d(np.arange(N), pool)
    ax.scatter(xW3[test_idx], xKV3[test_idx], xKVD3[test_idx], c='lightgray', s=8, alpha=0.4, label='test')
    ax.scatter(xW3[pool], xKV3[pool], xKVD3[pool], c=color, s=42, edgecolor='black', linewidth=0.4, label='train (50)')
    ax.set_xlabel('JSD_W'); ax.set_ylabel('JSD_KV'); ax.set_zlabel('JSD_KVD')
    ax.set_title(name, fontweight='bold', fontsize=10)
    ax.view_init(elev=20, azim=45)
plt.suptitle('Train pool layout in JSD space (50 train + 150 test)', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/05_direct_design_layouts.png', **PLT_KW); plt.close()

print(f"\nFigures saved:")
print(f"  05_direct_design.png            — R² vs N_train per design + bar at N=50")
print(f"  05_direct_design_layouts.png    — 3D scatter of train pools per design")
print("Done.\n")
