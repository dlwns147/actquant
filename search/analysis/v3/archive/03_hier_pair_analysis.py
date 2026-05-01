"""analysis_v3_hier_pair_split.py — Pair × split sensitivity analysis.

Sweeps the full design space of hierarchical 2-then-1 surrogates under fixed
total measurement budget:
  • 3 pair choices: WK (W+KV), WD (W+KVD), KD (KV+KVD)
  • Total budgets B ∈ {27, 30, 33, 35, 40, 45, 50}
  • N_3way ∈ {6, 9, 12, 15, 18, 21, 24, 27, 30, 35, 40, 45, 50}
    with N_pair = B − N_3way ∈ [6, 200]

Pair model = TPS + linear tail (best from prior analysis).
Residual    = 1D linear regression on the third variable.
30 seeds per cell, evaluation on the same fixed 150 hold-out.

Reports & figures:
  (a) Best Hier R² per pair across B (3 curves vs Direct)
  (b) Per-pair (B × N_3way) heatmap of R²
  (c) Best (pair, N_3way) per B — recommendation table
  (d) Sensitivity: R² gap between pair choices and between splits
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

# ─── Data loading (mirrors prior scripts) ─────────────────────────────────────
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
AWQ_W_KV  = f'{BASE}/save/result/awq/2604162010_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr/results.csv'
AWQ_W_KVD = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim/results.csv'
AWQ_KV_KVD= f'{BASE}/save/result/awq/2604162013_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_kv_expr_kv_dim/results.csv'

with open(f'{BASE}/config/llama.json') as f: config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

print("Loading PFs and all datasets...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
pf_W_m, pf_KV_m, pf_KVDIM_m = (p[np.argsort(p[:, 0])] for p in (pf_W, pf_KV, pf_KVDIM))
JSD_W_def, JSD_KV_def, JSD_KVD_def = pf_W_m[0,0], pf_KV_m[0,0], pf_KVDIM_m[0,0]

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

def load_2way(path, has_W, has_KV, has_KVD):
    mat = load_csv(path); N0 = mat.shape[1]
    y = mat[12, :N0]; valid = ~np.isnan(y)
    xW   = match_metric(mat[0, :N0], pf_W   ) if has_W   else np.full(N0, JSD_W_def)
    xKV  = match_metric(mat[1, :N0], pf_KV  ) if has_KV  else np.full(N0, JSD_KV_def)
    xKVD = match_metric(mat[4, :N0], pf_KVDIM) if has_KVD else np.full(N0, JSD_KVD_def)
    return xW[valid], xKV[valid], xKVD[valid], y[valid]

xW_WK, xKV_WK, xKVD_WK, y_WK = load_2way(AWQ_W_KV,  True,  True,  False)
xW_WD, xKV_WD, xKVD_WD, y_WD = load_2way(AWQ_W_KVD, True,  False, True )
xW_KD, xKV_KD, xKVD_KD, y_KD = load_2way(AWQ_KV_KVD,False, True,  True )
print(f"  3-way N={N}; pairs: WK={len(y_WK)}, WD={len(y_WD)}, KD={len(y_KD)}")

# Pair containers
pair_data = {
    'WK': dict(xa=xW_WK, xb=xKV_WK, y=y_WK, vars=('W','KV')),
    'WD': dict(xa=xW_WD, xb=xKVD_WD, y=y_WD, vars=('W','KVD')),
    'KD': dict(xa=xKV_KD, xb=xKVD_KD, y=y_KD, vars=('KV','KVD')),
}
fmap = {'W': 0, 'KV': 1, 'KVD': 2}

# ─── 27-grid + 23 maximin = 50 train pool / 150 test ─────────────────────────
qs = [0.1, 0.5, 0.9]
qW_, qKV_, qKVD_ = np.quantile(xW3, qs), np.quantile(xKV3, qs), np.quantile(xKVD3, qs)
grid27 = np.array([[w,kv,kvd] for w in qW_ for kv in qKV_ for kvd in qKVD_])
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
print(f"  Fixed split: 50 train pool / {len(test_set)} test")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def r2(y, yp):
    ss_t = max(np.sum((y - y.mean())**2), 1e-30)
    return 1 - np.sum((y-yp)**2)/ss_t
def rmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)))

X_te_jsd = X3_jsd[test_set]; y_te = y3[test_set]; X_te_idx = X3_idx[test_set]

# ─── Direct evaluation ────────────────────────────────────────────────────────
def direct_eval(n_3way, n_seeds=30):
    r2s, rmses = [], []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        n_extra = n_3way - 27
        if n_extra == 0: tr3 = grid_samples
        elif n_extra == 23: tr3 = train_pool
        elif 0 < n_extra < 23:
            tr3 = np.concatenate([grid_samples, extras[rng.choice(23, n_extra, replace=False)]])
        elif n_extra < 0:
            tr3 = grid_samples[rng.choice(27, n_3way, replace=False)]
        else: tr3 = train_pool
        try:
            m = RBF(kernel='cubic', tail='linear', lb=lb_idx, ub=ub_idx)
            m.fit(X3_idx[tr3], y3[tr3])
            yp = m.predict(X_te_idx).ravel()
            r2s.append(r2(y_te, yp)); rmses.append(rmse(y_te, yp))
        except Exception:
            r2s.append(np.nan); rmses.append(np.nan)
    return np.array(r2s), np.array(rmses)

# ─── Hier evaluation parameterized by pair ────────────────────────────────────
def hier_eval(pair_key, pair_N, threeway_N, n_seeds=30):
    if pair_N < 6 or threeway_N < 6: return np.array([np.nan]), np.array([np.nan])
    d = pair_data[pair_key]
    a_idx, b_idx = fmap[d['vars'][0]], fmap[d['vars'][1]]
    third_idx = [i for i in range(3) if i not in (a_idx, b_idx)][0]

    r2s, rmses = [], []
    for seed in range(n_seeds):
        rng3 = np.random.RandomState(seed)
        rng_p = np.random.RandomState(seed + 100000)
        n_extra = threeway_N - 27
        if n_extra == 0: tr3 = grid_samples
        elif n_extra == 23: tr3 = train_pool
        elif 0 < n_extra < 23:
            tr3 = np.concatenate([grid_samples, extras[rng3.choice(23, n_extra, replace=False)]])
        elif n_extra < 0:
            tr3 = grid_samples[rng3.choice(27, threeway_N, replace=False)]
        else: tr3 = train_pool

        n_p = min(pair_N, len(d['y']))
        pair_idx = rng_p.choice(len(d['y']), n_p, replace=False) if n_p < len(d['y']) else np.arange(len(d['y']))
        xa = d['xa'][pair_idx]; xb = d['xb'][pair_idx]; ypair = d['y'][pair_idx]
        X2 = np.column_stack([xa, xb])
        try:
            m_pair = RBF(kernel='tps', tail='linear', lb=X2.min(0), ub=X2.max(0))
            m_pair.fit(X2, ypair)
            x3_tr = X3_jsd[tr3]
            yp_pair_tr = m_pair.predict(np.column_stack([x3_tr[:, a_idx], x3_tr[:, b_idx]])).ravel()
            yp_pair_te = m_pair.predict(np.column_stack([X_te_jsd[:, a_idx], X_te_jsd[:, b_idx]])).ravel()
            r_tr = y3[tr3] - yp_pair_tr
            t_tr = x3_tr[:, third_idx]; t_te = X_te_jsd[:, third_idx]
            c = np.polyfit(t_tr, r_tr, 1)
            yp_t = yp_pair_te + np.polyval(c, t_te)
            r2s.append(r2(y_te, yp_t)); rmses.append(rmse(y_te, yp_t))
        except Exception:
            r2s.append(np.nan); rmses.append(np.nan)
    return np.array(r2s), np.array(rmses)

# ─── Sweep ────────────────────────────────────────────────────────────────────
budgets = [27, 30, 33, 35, 40, 45, 50]
n3_candidates = [6, 9, 12, 15, 18, 21, 24, 27, 30, 35, 40, 45, 50]
N_SEED = 30

print(f"\nDirect baseline:")
direct = {}
for B in budgets:
    r2s, _ = direct_eval(B, N_SEED)
    direct[B] = float(np.nanmedian(r2s))
    print(f"  B={B}: R²={direct[B]:.4f}")

print(f"\nHier sweep over pair × budget × split (30 seeds each)...")
all_records = {}  # (pair, B, n3, n_p) -> r2_med
for pair in ['WK', 'WD', 'KD']:
    print(f"\n  pair={pair}:")
    for B in budgets:
        feasible = [n3 for n3 in n3_candidates if 6 <= n3 <= 50 and 6 <= (B - n3) <= 200]
        line = f"    B={B:2d}: "
        for n3 in feasible:
            n_p = B - n3
            r2s, _ = hier_eval(pair, n_p, n3, N_SEED)
            med = float(np.nanmedian(r2s))
            all_records[(pair, B, n3, n_p)] = med
            line += f"({n3},{n_p})={med:.3f} "
        print(line)

# ─── Best per pair per B ──────────────────────────────────────────────────────
print("\n" + "="*100)
print("Best Hier per (pair, B)")
print("="*100)
best_pair_B = {}  # (pair, B) -> (n3, n_p, r2)
for pair in ['WK','WD','KD']:
    for B in budgets:
        cands = [(n3, n_p, r2) for (p, BB, n3, n_p), r2 in all_records.items() if p == pair and BB == B]
        if not cands: continue
        best = max(cands, key=lambda x: x[2])
        best_pair_B[(pair, B)] = best
        print(f"  pair={pair}  B={B:2d}: best (n3={best[0]:2d}, n_p={best[1]:2d})  R²={best[2]:.4f}")

# ─── Best across all pairs per B ─────────────────────────────────────────────
print("\n" + "="*100)
print("Best Hier across all pairs at each B (with Direct comparison)")
print("="*100)
print(f"  {'B':3s}  {'Direct R²':10s}  {'Best (pair, n3, n_p) R²':35s}  {'Δ Hier−Direct':14s}")
overall_best = {}
for B in budgets:
    cands = [(p, n3, n_p, r2) for (p, BB, n3, n_p), r2 in all_records.items() if BB == B]
    if not cands: continue
    best = max(cands, key=lambda x: x[3])
    overall_best[B] = best
    delta = best[3] - direct[B]
    print(f"  B={B:2d}  Direct {direct[B]:.4f}  Hier-{best[0]:2s} (n3={best[1]:2d}, n_p={best[2]:2d}) {best[3]:.4f}  Δ={delta:+.4f}")

# ─── Sensitivity analysis ────────────────────────────────────────────────────
print("\n" + "="*100)
print("Sensitivity: at each B, R² range across pair choices (best split per pair)")
print("="*100)
print(f"  {'B':3s}  {'WK R²':9s}  {'WD R²':9s}  {'KD R²':9s}  {'pair-spread':12s}  {'best pair':10s}")
for B in budgets:
    rs = {p: best_pair_B.get((p, B), (None, None, np.nan))[2] for p in ['WK','WD','KD']}
    valid = {k: v for k, v in rs.items() if not np.isnan(v)}
    if not valid: continue
    spread = max(valid.values()) - min(valid.values())
    bp = max(valid, key=valid.get)
    print(f"  B={B:2d}  {rs['WK']:.4f}     {rs['WD']:.4f}     {rs['KD']:.4f}     {spread:.4f}        {bp}")

print("\n" + "="*100)
print("Sensitivity: at each (pair, B), R² range across N_3way splits")
print("="*100)
for pair in ['WK','WD','KD']:
    print(f"\n  pair={pair}:")
    for B in budgets:
        cands = [(n3, n_p, r2) for (p, BB, n3, n_p), r2 in all_records.items() if p == pair and BB == B]
        if not cands: continue
        r2s_only = [c[2] for c in cands]
        spread = max(r2s_only) - min(r2s_only)
        best = max(cands, key=lambda x: x[2])
        worst = min(cands, key=lambda x: x[2])
        print(f"    B={B:2d}: R² range [{min(r2s_only):.3f}, {max(r2s_only):.3f}] spread={spread:.3f}  "
              f"best (n3={best[0]:2d}) → R²={best[2]:.4f}  worst (n3={worst[0]:2d}) → R²={worst[2]:.4f}")

# ─── Figures ──────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')
PAIR_COLOR = {'WK': '#E64B35', 'WD': '#4DBBD5', 'KD': '#9B59B6'}

# Fig 1: Best per pair vs Direct
fig, ax = plt.subplots(figsize=(9, 5))
for pair in ['WK','WD','KD']:
    xs, ys = [], []
    for B in budgets:
        if (pair, B) in best_pair_B:
            xs.append(B); ys.append(best_pair_B[(pair, B)][2])
    ax.plot(xs, ys, '-o', color=PAIR_COLOR[pair], lw=2.4, ms=8,
            markeredgecolor='white', markeredgewidth=0.7,
            label=f'Hier-{pair} ({"+".join(pair_data[pair]["vars"])})')
xs_d = list(direct.keys()); ys_d = list(direct.values())
ax.plot(xs_d, ys_d, '-s', color='#3C5488', lw=2.4, ms=8,
        markeredgecolor='white', markeredgewidth=0.7, label='Direct-RBF')
ax.set_xlabel('Total measurements  B'); ax.set_ylabel('Best test R² (median, 30 seeds)')
ax.set_title('Pair × Budget — best Hier R² per pair vs Direct', fontweight='bold')
ax.set_xticks(budgets)
ax.legend(); ax.grid(True, alpha=0.25, lw=0.5)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/03_hier_pair_best.png', **PLT_KW); plt.close()

# Fig 2: 3 heatmaps (one per pair) of R²(B, N_3way)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
all_vals = [v for v in all_records.values() if not np.isnan(v)]
vmin, vmax = min(all_vals), max(all_vals)
for ax, pair in zip(axes, ['WK','WD','KD']):
    M = np.full((len(n3_candidates), len(budgets)), np.nan)
    for (p, B, n3, n_p), v in all_records.items():
        if p != pair: continue
        i = n3_candidates.index(n3); j = budgets.index(B)
        M[i, j] = v
    im = ax.imshow(M, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    for i in range(len(n3_candidates)):
        for j in range(len(budgets)):
            if not np.isnan(M[i, j]):
                color = 'white' if M[i, j] < vmin + 0.55*(vmax - vmin) else 'black'
                ax.text(j, i, f"{M[i,j]:.3f}", ha='center', va='center', fontsize=7.5, color=color)
    # Mark best per B with red box
    for j, B in enumerate(budgets):
        if (pair, B) in best_pair_B:
            n3 = best_pair_B[(pair, B)][0]
            i = n3_candidates.index(n3)
            ax.add_patch(plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                        fill=False, edgecolor='red', linewidth=2.0))
    ax.set_xticks(range(len(budgets))); ax.set_xticklabels(budgets)
    ax.set_yticks(range(len(n3_candidates))); ax.set_yticklabels(n3_candidates)
    ax.set_xlabel('Total measurements  B')
    ax.set_ylabel('N_3way')
    ax.set_title(f'pair={pair} ({"+".join(pair_data[pair]["vars"])})', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04)
plt.suptitle('Hier R² heatmap (B × N_3way) per pair  (red box = best per B)',
             fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/03_hier_pair_heatmaps.png', **PLT_KW); plt.close()

# Fig 3: Bar chart of best (pair, n3) per B
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
xs = budgets
direct_vals = [direct[B] for B in xs]
hier_vals = [overall_best[B][3] for B in xs]
hier_pair = [overall_best[B][0] for B in xs]
hier_n3 = [overall_best[B][1] for B in xs]
hier_np = [overall_best[B][2] for B in xs]
w = 0.4
ax.bar([x - w/2 for x in xs], direct_vals, w, color='#3C5488', label='Direct', edgecolor='#333', alpha=0.85)
ax.bar([x + w/2 for x in xs], hier_vals,   w,
       color=[PAIR_COLOR[p] for p in hier_pair],
       label='Best Hier', edgecolor='#333', alpha=0.85)
for i, x in enumerate(xs):
    ax.text(x + w/2, hier_vals[i] + 0.0008,
            f"{hier_pair[i]}\n({hier_n3[i]},{hier_np[i]})",
            ha='center', va='bottom', fontsize=7.5, fontweight='bold',
            color=PAIR_COLOR[hier_pair[i]])
ax.set_xticks(xs); ax.set_xlabel('Total measurements B'); ax.set_ylabel('Test R²')
ax.set_ylim([0.96, 0.995])
ax.set_title('Direct vs best (pair, split) Hier per B', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.25, lw=0.5, axis='y')

# Sensitivity bar — pair-spread & split-spread per B
ax = axes[1]
pair_spreads, split_spreads = [], []
for B in budgets:
    rs = [best_pair_B.get((p, B), (None, None, np.nan))[2] for p in ['WK','WD','KD']]
    valid = [r for r in rs if not np.isnan(r)]
    pair_spreads.append(max(valid) - min(valid) if valid else 0)
    splits_for_best = []
    for p in ['WK','WD','KD']:
        cands = [v for (pp, BB, _, _), v in all_records.items() if pp == p and BB == B]
        splits_for_best.extend(cands)
    split_spreads.append(max(splits_for_best) - min(splits_for_best) if splits_for_best else 0)
w = 0.4
x = np.arange(len(budgets))
ax.bar(x - w/2, pair_spreads,  w, color='#E64B35', label='pair-choice spread', edgecolor='#333', alpha=0.85)
ax.bar(x + w/2, split_spreads, w, color='#4DBBD5', label='split spread (within-pair)', edgecolor='#333', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(budgets)
ax.set_xlabel('Total measurements B'); ax.set_ylabel('R² spread (max − min)')
ax.set_title('Sensitivity: which choice matters more?', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.25, lw=0.5, axis='y')
plt.suptitle('Pair × split summary  (left: best Hier per B; right: choice-sensitivity)',
             fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/03_hier_pair_summary.png', **PLT_KW); plt.close()

print(f"\nFigures saved:")
print(f"  03_hier_pair_best.png   — best Hier per pair vs Direct (line)")
print(f"  03_hier_pair_heatmaps.png        — R² (B × N_3way) per pair")
print(f"  03_hier_pair_summary.png         — best per B + sensitivity")
print("Done.\n")
