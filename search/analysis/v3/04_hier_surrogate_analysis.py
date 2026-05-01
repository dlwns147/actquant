"""04_hier_surrogate_analysis.py — §7 of the narrative.

Hier surrogate analysis under the 50-train fixed-pool constraint, comparing
two pair-surface kernels and three residual models.

Decomposition (Hoeffding-inspired):
    y(x_W, x_KV, x_KVD)  ≈  f_pair(x_a, x_b)  +  g(x_c)

  Stage 1 — Pair surface  f_pair(x_a, x_b)   (2-D regression)
      Kernel options:
          • TPS   + linear tail   (thin-plate spline, default)
          • Cubic + linear tail   (φ(r) = r^3)
  Stage 2 — Residual  g(x_c)                  (1-D regression)
      Model options:
          • lin   : 1-D linear polynomial (deg=1)
          • quad  : 1-D quadratic polynomial (deg=2)
          • rbfc  : 1-D RBF cubic + linear tail (nonparametric)

Three pair orderings (Sobol pairwise S_ab, ref §5.2):
    Hier-WK : pair=(W,KV),   residual=KVD,   S_ab = 0.0066
    Hier-WD : pair=(W,KVD),  residual=KV,    S_ab = 0.0366  ← largest
    Hier-KD : pair=(KV,KVD), residual=W,     S_ab = 0.0077

Two allocation strategies under 50-train constraint:
    disjoint : n_pair + n_3way ≤ 50, disjoint subsets of train_pool
    overlap  : both stages use all 50 samples (no split, may overfit)

Reports R², ε_∞, ε_2 on the same fixed 150-test split (matches §5).
"""
import sys, os, json, csv, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SKRBF, ConstantKernel as C, WhiteKernel
from utils.func import get_net_info
from predictor.rbf import RBF as PySOTRBF


# ─── Data loading ─────────────────────────────────────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]; m = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i,1] < m: nd.append(i); m = F_s[i,1]
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

with open(f'{BASE}/config/llama.json') as f:
    config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w':128, 'k':[128,128], 'v':[128,128]}

print("Loading 3-way data...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y3 = mat3[12, :N0]; v3 = ~np.isnan(y3)
xW3   = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3  = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD3 = match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y3 = y3[v3]
N = len(y3)
X3 = np.column_stack([xW3, xKV3, xKVD3])
print(f"  3-way N={N}")

# Fixed 50/150 split (matches §5)
qs = [0.1, 0.5, 0.9]
qW_, qKV_, qKVD_ = np.quantile(xW3, qs), np.quantile(xKV3, qs), np.quantile(xKVD3, qs)
grid27 = np.array([[w,kv,kvd] for w in qW_ for kv in qKV_ for kvd in qKVD_])
scale  = X3.std(0) + 1e-10
X3n    = X3 / scale; grid_n = grid27 / scale
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
X_te = X3[test_set]; y_te = y3[test_set]

pair_cfg = {
    'WK': dict(vars=('W','KV'),  resid='KVD', a_idx=0, b_idx=1, c_idx=2, S_pair=0.0066),
    'WD': dict(vars=('W','KVD'), resid='KV',  a_idx=0, b_idx=2, c_idx=1, S_pair=0.0366),
    'KD': dict(vars=('KV','KVD'),resid='W',   a_idx=1, b_idx=2, c_idx=0, S_pair=0.0077),
}

def metrics_eval(y_true, y_pred):
    r = y_true - y_pred
    ss_t = max(np.sum((y_true - y_true.mean())**2), 1e-30)
    return dict(
        r2 = 1 - np.sum(r**2)/ss_t,
        eps_inf = float(np.max(np.abs(r))),
        eps_2 = float(np.sqrt(np.mean(r**2))),
    )

def fit_pair(X2, y, kernel):
    """kernel ∈ {'tps','cubic'}."""
    m = PySOTRBF(kernel=kernel, tail='linear', lb=X2.min(0), ub=X2.max(0))
    m.fit(X2, y)
    return m

def fit_resid(xc, r, model):
    """model ∈ {'lin','quad','rbfc'}."""
    if model == 'lin':
        return ('poly', np.polyfit(xc, r, 1))
    if model == 'quad':
        return ('poly', np.polyfit(xc, r, 2))
    if model == 'rbfc':
        # 1-D RBF cubic + linear tail (need ≥ 4 samples for cubic+linear)
        lb = np.array([xc.min()]); ub = np.array([xc.max()])
        m = PySOTRBF(kernel='cubic', tail='linear', lb=lb, ub=ub)
        m.fit(xc.reshape(-1, 1), r)
        return ('rbf', m)
    raise ValueError(model)

def predict_resid(model_obj, xc):
    kind, mdl = model_obj
    if kind == 'poly': return np.polyval(mdl, xc)
    if kind == 'rbf':  return mdl.predict(xc.reshape(-1, 1)).ravel()

def hier_predict(pair_key, n_pair, allocation, pair_kernel, resid_model, seed):
    d = pair_cfg[pair_key]
    a, b, c = d['a_idx'], d['b_idx'], d['c_idx']
    rng = np.random.RandomState(seed)
    if allocation == 'overlap':
        tr_pair = train_pool; tr_res = train_pool
    else:
        perm = rng.permutation(len(train_pool))
        tr_pair = train_pool[perm[:n_pair]]
        tr_res  = train_pool[perm[n_pair:]] if allocation == 'disjoint' else train_pool
    if len(tr_pair) < 6: return None
    min_res = 4 if resid_model == 'rbfc' else 3
    if len(tr_res) < min_res: return None

    X2_p = np.column_stack([X3[tr_pair, a], X3[tr_pair, b]])
    m_pair = fit_pair(X2_p, y3[tr_pair], pair_kernel)
    X2_r = np.column_stack([X3[tr_res, a], X3[tr_res, b]])
    yp_pair_r = m_pair.predict(X2_r).ravel()
    r_r = y3[tr_res] - yp_pair_r
    m_res = fit_resid(X3[tr_res, c], r_r, resid_model)

    X2_te = np.column_stack([X_te[:, a], X_te[:, b]])
    yp_pair_te = m_pair.predict(X2_te).ravel()
    yp_te = yp_pair_te + predict_resid(m_res, X_te[:, c])
    return yp_te

def direct_ardgp(n_3way, seed):
    rng = np.random.RandomState(seed)
    if n_3way >= 50: tr = train_pool
    elif n_3way == 27: tr = grid_samples
    else: tr = train_pool[rng.choice(50, n_3way, replace=False)]
    kernel = (C(1.0, (1e-4, 1e2)) *
              SKRBF(length_scale=[1.0]*3, length_scale_bounds=(1e-3, 1e3)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=5, alpha=1e-8, random_state=0)
    gp.fit(X3[tr], y3[tr])
    return gp.predict(X_te)

# =============================================================================
# Direct ARD-GP baseline (B=50)
# =============================================================================
print("\n" + "="*94)
print("Direct ARD-GP baseline @ B=50 (50 3-way, full train_pool)")
print("="*94)
direct_pred = direct_ardgp(50, 0)
direct_m = metrics_eval(y_te, direct_pred)
print(f"  R²={direct_m['r2']:.4f}  ε_∞={direct_m['eps_inf']:.4f}  ε_2={direct_m['eps_2']:.4f}")

# =============================================================================
# PART A — Sweep over (pair, alloc, pair_kernel, resid_model, n_pair)
# =============================================================================
n_pair_grid = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
N_SEED = 30

print("\n" + "="*94)
print("PART A — Hier sweep [median 30 seeds]")
print("="*94)

results = {}  # (pair, alloc, pair_kernel, resid, n_pair) -> metrics dict
for pair_key in ['WK', 'WD', 'KD']:
    for alloc in ['disjoint', 'overlap']:
        for pk in ['tps', 'cubic']:
            for rm in ['lin', 'quad', 'rbfc']:
                if alloc == 'overlap':
                    yp = hier_predict(pair_key, 50, alloc, pk, rm, 0)
                    if yp is not None:
                        results[(pair_key, alloc, pk, rm, 50)] = metrics_eval(y_te, yp)
                else:
                    for np_ in n_pair_grid:
                        if alloc == 'disjoint' and np_ > 44 and rm != 'rbfc':
                            continue
                        if alloc == 'disjoint' and np_ > 42 and rm == 'rbfc':
                            continue
                        preds = []
                        for s in range(N_SEED):
                            yp = hier_predict(pair_key, np_, alloc, pk, rm, s)
                            if yp is not None: preds.append(yp)
                        if not preds: continue
                        ms = [metrics_eval(y_te, p) for p in preds]
                        med = {k: float(np.median([mm[k] for mm in ms])) for k in ms[0]}
                        results[(pair_key, alloc, pk, rm, np_)] = med

# =============================================================================
# Summary
# =============================================================================
def best_in(filt):
    cands = [(k, v) for k, v in results.items() if filt(k, v)]
    if not cands: return None, None
    return max(cands, key=lambda kv: kv[1]['r2'])

print("\n" + "="*94)
print("PART B — Best per (pair, allocation, pair_kernel, resid_model)")
print("="*94)
print(f"  {'pair':4s}  {'alloc':9s}  {'pkernel':8s}  {'resid':6s}  {'n_p':4s}  "
      f"{'R²':8s}  {'ε_∞':8s}  {'ε_2':8s}")
print(f"  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*6}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
for pair_key in ['WK','WD','KD']:
    for alloc in ['disjoint','overlap']:
        for pk in ['tps','cubic']:
            for rm in ['lin','quad','rbfc']:
                bk, bv = best_in(lambda k,v: k[0]==pair_key and k[1]==alloc and k[2]==pk and k[3]==rm)
                if bv is None: continue
                np_ = bk[4]
                print(f"  {pair_key:4s}  {alloc:9s}  {pk:8s}  {rm:6s}  {np_:4d}  "
                      f"{bv['r2']:.4f}  {bv['eps_inf']:.4f}  {bv['eps_2']:.4f}")

print("\n" + "="*94)
print("PART C — Best per (pair, pair_kernel) — across all alloc × resid")
print("="*94)
print(f"  {'pair':4s}  {'pkernel':8s}  {'best (alloc, resid, n_p)':30s}  "
      f"{'R²':8s}  {'ε_∞':8s}  {'ε_2':8s}")
for pair_key in ['WK','WD','KD']:
    for pk in ['tps', 'cubic']:
        bk, bv = best_in(lambda k,v: k[0]==pair_key and k[2]==pk)
        cfg_str = f"({bk[1]}, {bk[3]}, n_p={bk[4]})"
        print(f"  {pair_key:4s}  {pk:8s}  {cfg_str:30s}  "
              f"{bv['r2']:.4f}  {bv['eps_inf']:.4f}  {bv['eps_2']:.4f}")

print("\n" + "="*94)
print("PART D — Pair kernel head-to-head: TPS vs Cubic (best alloc · resid · n_p per pair)")
print("="*94)
for pair_key in ['WK','WD','KD']:
    bk_tps, bv_tps     = best_in(lambda k,v: k[0]==pair_key and k[2]=='tps')
    bk_cube, bv_cube   = best_in(lambda k,v: k[0]==pair_key and k[2]=='cubic')
    delta = bv_cube['r2'] - bv_tps['r2']
    print(f"  Hier-{pair_key}: TPS R²={bv_tps['r2']:.4f}  vs  Cubic R²={bv_cube['r2']:.4f}   "
          f"Δ={delta:+.4f}")

print("\n" + "="*94)
print("PART E — Residual model head-to-head per (pair, pair_kernel) at disjoint")
print("="*94)
for pair_key in ['WK','WD','KD']:
    print(f"  Hier-{pair_key}:")
    for pk in ['tps', 'cubic']:
        for rm in ['lin','quad','rbfc']:
            bk, bv = best_in(lambda k,v: k[0]==pair_key and k[1]=='disjoint'
                             and k[2]==pk and k[3]==rm)
            if bv is None: continue
            print(f"    pkernel={pk:6s}  resid={rm:5s}  best n_p={bk[4]:2d}  "
                  f"R²={bv['r2']:.4f}  ε_∞={bv['eps_inf']:.4f}")

# =============================================================================
# Figures
# =============================================================================
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

# Fig 1: R² vs n_pair, panels = (pair_kernel × resid_model), color = pair_config (disjoint only)
fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
PAIR_COL = {'WK':'#E64B35','WD':'#4DBBD5','KD':'#9B59B6'}
panel_grid = [(pk, rm) for pk in ['tps','cubic'] for rm in ['lin','quad','rbfc']]
for ax, (pk, rm) in zip(axes.flat, panel_grid):
    for pair_key in ['WK','WD','KD']:
        xs = [n for n in n_pair_grid if (pair_key, 'disjoint', pk, rm, n) in results]
        ys = [results[(pair_key, 'disjoint', pk, rm, n)]['r2'] for n in xs]
        d = pair_cfg[pair_key]
        ax.plot(xs, ys, '-o', color=PAIR_COL[pair_key], lw=2.4, ms=8,
                markeredgecolor='white', markeredgewidth=0.7,
                label=f'Hier-{pair_key} ({"+".join(d["vars"])} | res {d["resid"]})')
    ax.axhline(direct_m['r2'], ls='--', color='#3C5488', lw=1.6,
               label=f'Direct ARD-GP (R²={direct_m["r2"]:.3f})')
    ax.set_xlabel('n_pair'); ax.set_ylabel('Test R² (median, 30 seeds)')
    ax.set_title(f'pair_kernel={pk},  resid={rm}', fontweight='bold')
    ax.set_ylim(bottom=-0.5)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc='lower right')
plt.suptitle('Hier R² vs n_pair (disjoint allocation)  —  pair_kernel × resid_model panels',
             fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/04_hier_sweep_curves.png', dpi=170, bbox_inches='tight')
plt.close()

# Fig 2: best per (pair × pair_kernel) bar
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.35
xs = np.arange(3)
ys_tps   = [best_in(lambda k,v: k[0]==pk and k[2]=='tps')[1]['r2']   for pk in ['WK','WD','KD']]
ys_cubic = [best_in(lambda k,v: k[0]==pk and k[2]=='cubic')[1]['r2'] for pk in ['WK','WD','KD']]
ax.bar(xs - width/2, ys_tps,   width, color='#E67E22', label='Pair: TPS+linear', edgecolor='#333', alpha=0.85)
ax.bar(xs + width/2, ys_cubic, width, color='#16A085', label='Pair: RBF cubic+linear', edgecolor='#333', alpha=0.85)
ax.axhline(direct_m['r2'], ls='--', color='#3C5488', lw=2, label=f'Direct ARD-GP (R²={direct_m["r2"]:.3f})')
for i, (yt, yc) in enumerate(zip(ys_tps, ys_cubic)):
    ax.text(i - width/2, yt, f'{yt:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.text(i + width/2, yc, f'{yc:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_xticks(xs); ax.set_xticklabels(['Hier-WK','Hier-WD','Hier-KD'])
ax.set_ylabel('Best test R²  (across alloc × resid × n_pair)')
ax.set_title('Pair kernel head-to-head: TPS vs RBF Cubic', fontweight='bold')
ax.legend(loc='lower left'); ax.grid(True, alpha=0.25, axis='y', lw=0.5)
ax.set_ylim([-0.3, 1.05])
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/04_hier_kernel_compare.png', dpi=170, bbox_inches='tight')
plt.close()

# Fig 3: residual model head-to-head per pair (best across kernels)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, pair_key in zip(axes, ['WK','WD','KD']):
    for j, pk in enumerate(['tps','cubic']):
        ys = [best_in(lambda k,v: k[0]==pair_key and k[2]==pk and k[3]==rm and k[1]=='disjoint')[1]
              for rm in ['lin','quad','rbfc']]
        ys = [m['r2'] if m else 0 for m in ys]
        x = np.arange(3) + (j-0.5)*0.4
        ax.bar(x, ys, 0.36,
               color=['#E67E22','#16A085'][j],
               label=f'Pair: {pk}', edgecolor='#333', alpha=0.85)
    ax.axhline(direct_m['r2'], ls='--', color='#3C5488', lw=1.5, label='Direct ARD-GP')
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(['linear','quad','RBF cubic'])
    ax.set_xlabel('Residual model'); ax.set_ylabel('Best R²')
    ax.set_title(f'Hier-{pair_key} ({"+".join(pair_cfg[pair_key]["vars"])})', fontweight='bold')
    ax.set_ylim([-0.2, 1.05])
    ax.grid(True, alpha=0.25, axis='y', lw=0.5)
    ax.legend(fontsize=8, loc='lower left')
plt.suptitle('Residual model effect (disjoint, best n_pair) — per pair × pair_kernel',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/04_hier_resid_compare.png', dpi=170, bbox_inches='tight')
plt.close()

print(f"\nFigures saved:")
print(f"  04_hier_sweep_curves.png   — R² vs n_pair, panels = pair_kernel × resid_model (6 panels)")
print(f"  04_hier_kernel_compare.png — TPS vs Cubic head-to-head (3 pairs)")
print(f"  04_hier_resid_compare.png  — residual model comparison per pair × kernel")
print("Done.")
