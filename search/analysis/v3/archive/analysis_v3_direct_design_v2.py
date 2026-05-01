"""analysis_v3_direct_design_v2.py — Sensitivity-weighted with proper extreme anchoring.

KEY DESIGN CHANGE vs v1:
  All dimensions keep ≥ 3 levels with extreme anchoring [0.1, 0.5, 0.9].
  Sensitivity-weighting now means ADDING more interior levels in the wide-range
  dimensions (W, KVD), not removing levels from low-range (KV).

Designs compared (all with [0.1..0.9]-anchored quantiles, normalized maximin extras):
  A 3×3×3  baseline (K=27)
  B 4×3×3  W slightly more (K=36)
  C 5×3×3  W heavy (K=45)
  D 5×3×4  W heavy + KVD more (K=60, top-50 subsample)
  E 6×3×3  W very heavy (K=54, top-50)
  F 4×3×4  W + KVD moderately more (K=48)
  G 3×3×3 + W-axis aug at KV=mid × KVD=anchored (K=36)
  H 3×3×3 + 2D aug along W,KVD (K=33)
  I random (no grid)

For each design and N_train ∈ {27, 30, 35, 40, 45, 50}, fit RBF on first
N_train of pool, evaluate test R² on remaining samples.
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
print("Loading...")
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

# ─── Helpers ──────────────────────────────────────────────────────────────────
def r2(y, yp):
    ss_t = max(np.sum((y - y.mean())**2), 1e-30)
    return 1 - np.sum((y-yp)**2)/ss_t
def rmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)))

def lvls(L):
    if L == 1: return np.array([0.5])
    return np.linspace(0.1, 0.9, L)  # always anchored at extremes

def grid_to_pool(grid, N_total=50, maximin_space='normalized'):
    """Match grid (Q, 3) JSD-quantile points to N samples via Hungarian + maximin extras."""
    K = len(grid)
    grid_n = grid / scale
    cost = np.zeros((K, N))
    for j in range(K):
        cost[j] = np.sum((X3n - grid_n[j])**2, axis=1)

    # Hungarian — handle K > N edge case
    if K <= N:
        row_ind, col_ind = linear_sum_assignment(cost)
        # If K > N_total, take top N_total cheapest matches
        if K > N_total:
            match_costs = cost[row_ind, col_ind]
            order = np.argsort(match_costs)
            grid_samples = col_ind[order[:N_total]]
        else:
            grid_samples = col_ind
    else:
        cost_trunc = cost[:N]
        row_ind, col_ind = linear_sum_assignment(cost_trunc)
        grid_samples = col_ind
    grid_samples = np.unique(grid_samples)
    K_actual = len(grid_samples)

    if K_actual < N_total:
        X_for_maximin = X3n if maximin_space == 'normalized' else X3_jsd
        remaining = np.setdiff1d(np.arange(N), grid_samples)
        extras = []; sel = list(grid_samples)
        for _ in range(N_total - K_actual):
            sel_arr = np.array(sel, dtype=int)
            d = np.array([np.min(np.sum((X_for_maximin[r] - X_for_maximin[sel_arr])**2, axis=1)) for r in remaining])
            far = int(np.argmax(d))
            extras.append(int(remaining[far])); sel.append(int(remaining[far]))
            remaining = np.delete(remaining, far)
        pool = np.concatenate([grid_samples, np.array(extras, dtype=int)])
    else:
        pool = grid_samples
    return pool[:N_total], K_actual

def make_grid(L_W, L_KV, L_KVD):
    qs_W, qs_KV, qs_KVD = lvls(L_W), lvls(L_KV), lvls(L_KVD)
    qW   = np.quantile(xW3,   qs_W)
    qKV  = np.quantile(xKV3,  qs_KV)
    qKVD = np.quantile(xKVD3, qs_KVD)
    return np.array([[w, kv, kvd] for w in qW for kv in qKV for kvd in qKVD])

def make_grid_aug_W(intermediate_W=[0.25, 0.5, 0.75], aug_KVD=[0.1, 0.5, 0.9], aug_KV=[0.5]):
    """3×3×3 base + augment along W at specified KV/KVD positions."""
    base = make_grid(3, 3, 3)
    qW_aug = np.quantile(xW3, intermediate_W)
    qKV_aug = np.quantile(xKV3, aug_KV)
    qKVD_aug = np.quantile(xKVD3, aug_KVD)
    aug = np.array([[w, kv, kvd] for w in qW_aug for kv in qKV_aug for kvd in qKVD_aug])
    return np.concatenate([base, aug])

def make_grid_aug_2D(intermediate_W=[0.3, 0.7], intermediate_KVD=[0.3, 0.7]):
    """3×3×3 base + 2D augmentation in W and KVD plane (KV at center)."""
    base = make_grid(3, 3, 3)
    qW_aug = np.quantile(xW3, intermediate_W)
    qKV_mid = np.quantile(xKV3, [0.5])
    qKVD_aug = np.quantile(xKVD3, intermediate_KVD)
    aug_W = np.array([[w, kv, kvd] for w in qW_aug for kv in qKV_mid for kvd in np.quantile(xKVD3, [0.1, 0.5, 0.9])])
    aug_KVD = np.array([[w, kv, kvd] for w in np.quantile(xW3, [0.1, 0.5, 0.9]) for kv in qKV_mid for kvd in qKVD_aug])
    return np.unique(np.concatenate([base, aug_W, aug_KVD]), axis=0)

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
    ('A 3×3×3 baseline (anchored)',         lambda: make_grid(3, 3, 3),                  '#3C5488'),
    ('B 4×3×3 W↑moderate',                  lambda: make_grid(4, 3, 3),                  '#E64B35'),
    ('C 5×3×3 W↑',                          lambda: make_grid(5, 3, 3),                  '#4DBBD5'),
    ('D 6×3×3 W↑↑ (K=54, top-50)',          lambda: make_grid(6, 3, 3),                  '#9B59B6'),
    ('E 4×3×4 W+KVD↑',                      lambda: make_grid(4, 3, 4),                  '#00A087'),
    ('F 5×3×4 W+KVD↑↑ (K=60, top-50)',      lambda: make_grid(5, 3, 4),                  '#F39B7F'),
    ('G 3×3×3+W-aug (W=[.25,.5,.75]×KVD)',  lambda: make_grid_aug_W(),                   '#F1C40F'),
    ('H 3×3×3+2D-aug (W,KVD interior)',     lambda: make_grid_aug_2D(),                  '#16A085'),
    ('I random (no grid)',                  None,                                         '#7F7F7F'),
]
N_TRAINS = [27, 30, 35, 40, 45, 50]

# Build pools
print("\nBuilding train pools per design (50 train):")
pools = {}
for name, fn, _ in designs:
    if fn is None:
        pools[name] = None
        print(f"  {name:50s}: random (per-seed)")
    else:
        grid = fn()
        pool, K = grid_to_pool(grid, N_total=50, maximin_space='normalized')
        pools[name] = pool
        print(f"  {name:50s}: K_grid={K:2d}, pool size = {len(pool)}")

# ─── Evaluate ─────────────────────────────────────────────────────────────────
print("\nEvaluating Direct-RBF for each (design, N_train)...")
results = {}
N_SEEDS_RANDOM = 30
for name, fn, _ in designs:
    print(f"\n  {name}:")
    if fn is None:
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
print("\n" + "="*120)
print("Test R² per (design, N_train)")
print("="*120)
hdr = f"  {'Design':52s}"
for N_tr in N_TRAINS: hdr += f"  N={N_tr:<3d}    "
print(hdr)
for name, _, _ in designs:
    line = f"  {name:52s}"
    for N_tr in N_TRAINS:
        r2s, _ = results[(name, N_tr)]
        med = float(np.median(r2s))
        line += f"  {med:.4f}    "
    print(line)

print("\n" + "="*120)
print("Best design at each N_train")
print("="*120)
print(f"  {'N_tr':5s}  {'Best design':55s}  {'R²':10s}  {'vs A baseline Δ':16s}")
for N_tr in N_TRAINS:
    cands = []
    for name, _, _ in designs:
        r2s, _ = results[(name, N_tr)]
        cands.append((name, float(np.median(r2s))))
    cands.sort(key=lambda x: -x[1])
    best = cands[0]
    A_r2 = next(c[1] for c in cands if c[0].startswith('A '))
    delta = best[1] - A_r2
    print(f"  N={N_tr:2d}  {best[0]:55s}  R²={best[1]:.4f}  Δ={delta:+.4f}")

# ─── Figures ──────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
for name, _, color in designs:
    xs, med = [], []
    for N_tr in N_TRAINS:
        r2s, _ = results[(name, N_tr)]
        xs.append(N_tr); med.append(float(np.median(r2s)))
    ax.plot(xs, med, '-o', color=color, label=name, lw=2.0, ms=6,
            markeredgecolor='white', markeredgewidth=0.6)
ax.set_xlabel('N_train (Direct-RBF, sorted-index)')
ax.set_ylabel('Test R² on hold-out')
ax.set_title('(a) Sensitivity-weighted with proper anchoring\n(all dims ≥ 3 levels)',
             fontweight='bold')
ax.set_xticks(N_TRAINS); ax.legend(fontsize=8, loc='lower right', ncol=1)
ax.grid(True, alpha=0.25, lw=0.5)

# Bar at N=50
ax = axes[1]
N_show = 50
names = [d[0] for d in designs]
r2_vals = [float(np.median(results[(d[0], N_show)][0])) for d in designs]
colors = [d[2] for d in designs]
bars = ax.barh(range(len(names)), r2_vals, color=colors, alpha=0.85, edgecolor='#333')
A_r2 = next(r for n, r in zip(names, r2_vals) if n.startswith('A '))
ax.axvline(A_r2, color='#3C5488', ls='--', lw=1.5, alpha=0.7, label=f'baseline A = {A_r2:.4f}')
for i, v in enumerate(r2_vals):
    delta = v - A_r2
    sym = '+' if delta >= 0 else ''
    ax.text(v + 0.0003, i, f'{v:.4f} ({sym}{delta:+.4f})',
            va='center', fontsize=8.5, fontweight='bold')
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Test R² at N_train = 50')
ax.set_title('(b) Design ranking at N=50 (Δ vs baseline)', fontweight='bold')
ax.set_xlim([min(r2_vals) - 0.002, max(r2_vals) + 0.005])
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.25, lw=0.5, axis='x')
ax.invert_yaxis()
plt.suptitle('Direct surrogate: sensitivity-weighted designs with proper extreme anchoring',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_direct_design_v2.png', **PLT_KW); plt.close()

# Pool layout 3D scatter (top 9 designs)
fig, axes = plt.subplots(3, 3, figsize=(15, 13), subplot_kw={'projection': '3d'})
ax_list = axes.flatten()
for ax_i, (name, fn, color) in enumerate(designs[:9]):
    ax = ax_list[ax_i]
    pool = pools[name] if fn is not None else random_pool(50, 0)
    test_idx = np.setdiff1d(np.arange(N), pool)
    ax.scatter(xW3[test_idx], xKV3[test_idx], xKVD3[test_idx], c='lightgray', s=6, alpha=0.4)
    ax.scatter(xW3[pool], xKV3[pool], xKVD3[pool], c=color, s=42, edgecolor='black', linewidth=0.4)
    ax.set_xlabel('W'); ax.set_ylabel('KV'); ax.set_zlabel('KVD')
    ax.set_title(name.split(' (')[0], fontweight='bold', fontsize=9)
    ax.view_init(elev=20, azim=45)
plt.suptitle('Train pool layouts (50 train + 150 test) per design', fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_direct_design_v2_layouts.png', **PLT_KW); plt.close()

print(f"\nFigures saved:")
print(f"  06_direct_design_v2.png            — R² curves + ranking bar")
print(f"  06_direct_design_v2_layouts.png    — 3D pool layouts")
print("Done.\n")
