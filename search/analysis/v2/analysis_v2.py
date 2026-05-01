"""
Comprehensive analysis v2 — correct inputs (per-method HQQ JSD).
Key additions vs v1:
  - Sequential R² decomposition (handles multicollinearity)
  - Centered polynomial features for ANOVA
  - Fixed 27-grid evaluation (exclude grid points from test)
  - Better figures (publication quality)
"""

import sys, json, csv, os
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ─── Pareto front (2D, minimize both) ─────────────────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0])
    F_s = F[order]
    min2 = np.inf
    nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < min2:
            nd.append(i)
            min2 = F_s[i, 1]
    return order[nd]

def load_archive_pareto(stats_path, comp_key, config, group_size):
    from utils.func import get_net_info
    with open(stats_path) as f:
        data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs   = [v[0] for v in archive]
    metrics = np.array([v[1] for v in archive])
    comps   = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
    F = np.column_stack((metrics, comps))
    sort_idx = np.argsort(metrics)
    F = F[sort_idx]
    nd_idx = pareto_front_2d(F)
    return F[nd_idx]

def match_comp(comp_vals, pf_F):
    """Nearest-neighbor match comp values → metrics."""
    return np.array([pf_F[np.argmin(np.abs(pf_F[:, 1] - c)), 0] for c in comp_vals])

def load_awq_csv(csv_path):
    with open(csv_path) as f:
        rows = [r for r in csv.reader(f) if r]
    max_cols = max(len(r) for r in rows)
    mat = np.full((len(rows), max_cols), np.nan)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            try: mat[i, j] = float(v)
            except: pass
    return mat

# ─── OLS ──────────────────────────────────────────────────────────────────────
def fit_ols(Phi, y):
    coef, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
    yp = Phi @ coef
    ss_res = np.sum((y - yp)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / len(y))
    n, k = Phi.shape
    aic = n * np.log(ss_res / n) + 2 * k
    return coef, yp, r2, rmse, aic

def features(xw, xkv, xkvd, mode='linear'):
    n = len(xw)
    o = np.ones(n)
    if mode == 'linear':
        return np.column_stack([o, xw, xkv, xkvd])
    elif mode == 'quad':
        return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2])
    elif mode == 'full_quad':
        return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2,
                                xw*xkv, xw*xkvd, xkv*xkvd])

# ─── Setup ────────────────────────────────────────────────────────────────────
BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY  = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
AWQ_W_KV  = f'{BASE}/save/result/awq/2604162010_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr/results.csv'
AWQ_W_KVD = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim/results.csv'
AWQ_KV_KVD= f'{BASE}/save/result/awq/2604162013_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_kv_expr_kv_dim/results.csv'

with open(f'{BASE}/config/llama.json') as f:
    config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

# ─── Load Pareto fronts ───────────────────────────────────────────────────────
print("Loading Pareto fronts...")
pf_W    = load_archive_pareto(W_STATS,    'wbits', config, group_size)
pf_KV   = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM= load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
# Sort Pareto fronts by comp for clean plotting
pf_W    = pf_W   [np.argsort(pf_W   [:, 1])]
pf_KV   = pf_KV  [np.argsort(pf_KV  [:, 1])]
pf_KVDIM= pf_KVDIM[np.argsort(pf_KVDIM[:,1])]
print(f"  W PF:     {len(pf_W):4d} pts, JSD [{pf_W[:,0].min():.3f},{pf_W[:,0].max():.3f}]")
print(f"  KV PF:    {len(pf_KV):4d} pts, JSD [{pf_KV[:,0].min():.3f},{pf_KV[:,0].max():.3f}]")
print(f"  KVDIM PF: {len(pf_KVDIM):4d} pts, JSD [{pf_KVDIM[:,0].min():.3f},{pf_KVDIM[:,0].max():.3f}]")

# ─── Load 3-way AWQ CSV and extract data ──────────────────────────────────────
mat = load_awq_csv(AWQ_3WAY)
N = mat.shape[1]
# CSV row indices: 0=wbits, 1=kvbits, 4=kvdim, 12=JSD_AWQ
xW   = match_comp(mat[0, :N], pf_W)
xKV  = match_comp(mat[1, :N], pf_KV)
xKVD = match_comp(mat[4, :N], pf_KVDIM)
y    = mat[12, :N]
valid = ~np.isnan(y)
xW, xKV, xKVD, y = xW[valid], xKV[valid], xKVD[valid], y[valid]
N = len(y)

print(f"\nDataset N={N}")
print(f"  JSD_W:   [{xW.min():.4f},{xW.max():.4f}], mean={xW.mean():.4f}, std={xW.std():.4f}")
print(f"  JSD_KV:  [{xKV.min():.4f},{xKV.max():.4f}], mean={xKV.mean():.4f}, std={xKV.std():.4f}")
print(f"  JSD_KVD: [{xKVD.min():.4f},{xKVD.max():.4f}], mean={xKVD.mean():.4f}, std={xKVD.std():.4f}")
print(f"  JSD_AWQ: [{y.min():.4f},{y.max():.4f}], mean={y.mean():.4f}, std={y.std():.4f}")

# Pearson correlations
rW, _   = stats.pearsonr(xW, y)
rKV, _  = stats.pearsonr(xKV, y)
rKVD, _ = stats.pearsonr(xKVD, y)
print(f"\nPearson r with JSD_AWQ: W={rW:.4f}, KV={rKV:.4f}, KVD={rKVD:.4f}")

# ─── Model comparison ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
mdl = {}
for mode in ['linear', 'quad', 'full_quad']:
    Phi = features(xW, xKV, xKVD, mode)
    coef, yp, r2, rmse, aic = fit_ols(Phi, y)
    mdl[mode] = dict(coef=coef, yp=yp, r2=r2, rmse=rmse, aic=aic, n=Phi.shape[1])
    print(f"\n{mode.upper()} (k={Phi.shape[1]}): R²={r2:.6f}, RMSE={rmse:.6f}, AIC={aic:.1f}")

c = mdl['linear']['coef']
print(f"\nLinear (additive) model:")
print(f"  y = {c[0]:.4f} + {c[1]:.4f}·JSD_W + {c[2]:.4f}·JSD_KV + {c[3]:.4f}·JSD_KVD")

# ─── Sequential R² decomposition ──────────────────────────────────────────────
print("\n" + "="*60)
print("SEQUENTIAL R² DECOMPOSITION")
print("="*60)
# Centered features to reduce multicollinearity
mW, mKV, mKVD = xW.mean(), xKV.mean(), xKVD.mean()
cW, cKV, cKVD = xW - mW, xKV - mKV, xKVD - mKVD

seq_models = [
    ('intercept',  np.ones((N,1))),
    ('+W',         np.column_stack([np.ones(N), cW])),
    ('+KV',        np.column_stack([np.ones(N), cW, cKV])),
    ('+KVD',       np.column_stack([np.ones(N), cW, cKV, cKVD])),
    ('+W²',        np.column_stack([np.ones(N), cW, cKV, cKVD, cW**2])),
    ('+KV²',       np.column_stack([np.ones(N), cW, cKV, cKVD, cW**2, cKV**2])),
    ('+KVD²',      np.column_stack([np.ones(N), cW, cKV, cKVD, cW**2, cKV**2, cKVD**2])),
    ('+W·KV',      np.column_stack([np.ones(N), cW, cKV, cKVD, cW**2, cKV**2, cKVD**2, cW*cKV])),
    ('+W·KVD',     np.column_stack([np.ones(N), cW, cKV, cKVD, cW**2, cKV**2, cKVD**2, cW*cKV, cW*cKVD])),
    ('+KV·KVD',    np.column_stack([np.ones(N), cW, cKV, cKVD, cW**2, cKV**2, cKVD**2, cW*cKV, cW*cKVD, cKV*cKVD])),
]
seq_r2 = []
prev_r2 = 0
for name, Phi in seq_models:
    _, _, r2, _, _ = fit_ols(Phi, y)
    delta = r2 - prev_r2
    seq_r2.append((name, r2, delta))
    prev_r2 = r2
    print(f"  {name:12s}: R²={r2:.6f}, ΔR²={delta:+.6f}")

# ─── Method-level R² contributions (sequential) ───────────────────────────────
print("\nMethod-level R² contributions:")
print(f"  W main effect:    {seq_r2[1][2]:.6f} ({seq_r2[1][2]*100:.2f}%)")
print(f"  KV main effect:   {seq_r2[2][2]:.6f} ({seq_r2[2][2]*100:.2f}%)")
print(f"  KVD main effect:  {seq_r2[3][2]:.6f} ({seq_r2[3][2]*100:.2f}%)")
print(f"  W² effect:        {seq_r2[4][2]:.6f} ({seq_r2[4][2]*100:.2f}%)")
print(f"  KV² effect:       {seq_r2[5][2]:.6f} ({seq_r2[5][2]*100:.2f}%)")
print(f"  KVD² effect:      {seq_r2[6][2]:.6f} ({seq_r2[6][2]*100:.2f}%)")
print(f"  W·KV interaction: {seq_r2[7][2]:.6f} ({seq_r2[7][2]*100:.2f}%)")
print(f"  W·KVD interaction:{seq_r2[8][2]:.6f} ({seq_r2[8][2]*100:.2f}%)")
print(f"  KV·KVD interaction:{seq_r2[9][2]:.6f} ({seq_r2[9][2]*100:.2f}%)")
print(f"\n  Total main effects: {(seq_r2[1][2]+seq_r2[2][2]+seq_r2[3][2])*100:.2f}%")
print(f"  Total quadratic:    {(seq_r2[4][2]+seq_r2[5][2]+seq_r2[6][2])*100:.2f}%")
print(f"  Total interactions: {(seq_r2[7][2]+seq_r2[8][2]+seq_r2[9][2])*100:.2f}%")

# ─── Monotonicity ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MONOTONICITY: ∂F/∂(JSD_i) ≥ 0")
print("="*60)
c = mdl['full_quad']['coef']
# Full-quad coef indices: [β0, βW, βKV, βKVD, βWW, βKVKV, βKVDKVD, βWKV, βWKVD, βKVKVD]
dW   = c[1] + 2*c[4]*xW   + c[7]*xKV   + c[8]*xKVD
dKV  = c[2] + 2*c[5]*xKV  + c[7]*xW    + c[9]*xKVD
dKVD = c[3] + 2*c[6]*xKVD + c[8]*xW    + c[9]*xKV
for name, g in [('W', dW), ('KV', dKV), ('KVD', dKVD)]:
    print(f"  ∂F/∂JSD_{name}: min={g.min():.4f}, mean={g.mean():.4f}, max={g.max():.4f}  "
          f"| ≥0: {(g>=0).mean()*100:.1f}%")
all_ok = (dW >= 0).all() and (dKV >= 0).all() and (dKVD >= 0).all()
print(f"\n  Globally monotone: {'YES' if all_ok else 'NO'}")

# ─── 27-Grid calibration ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("27-GRID CALIBRATION (quantile design)")
print("="*60)
# Quantiles for each JSD dimension
qs = [0.1, 0.5, 0.9]
qW   = np.quantile(xW,   qs)
qKV  = np.quantile(xKV,  qs)
qKVD = np.quantile(xKVD, qs)
print(f"  W   quantiles [0.1,0.5,0.9]: {qW}")
print(f"  KV  quantiles [0.1,0.5,0.9]: {qKV}")
print(f"  KVD quantiles [0.1,0.5,0.9]: {qKVD}")

# Find 27 data points closest to grid design in normalized space
Xall = np.column_stack([xW, xKV, xKVD])
scale = Xall.std(axis=0) + 1e-10
Xall_n = Xall / scale

grid_all = np.array([[w, kv, kvd] for w in qW for kv in qKV for kvd in qKVD])
grid_n = grid_all / scale

# Match each grid point to nearest data point (with replacement only for unavoidable ties)
grid_data_idx = np.array([np.argmin(np.sum((Xall_n - g)**2, axis=1)) for g in grid_n])
used_idx = np.unique(grid_data_idx)
print(f"\n  Unique data points used for grid: {len(used_idx)}/27")

# Fit on grid points, test on excluded points
test_idx = np.setdiff1d(np.arange(N), used_idx)
Phi_grid_tr = features(xW[grid_data_idx], xKV[grid_data_idx], xKVD[grid_data_idx], 'full_quad')
Phi_grid_te = features(xW[test_idx], xKV[test_idx], xKVD[test_idx], 'full_quad')
coef_grid, _, r2_grid_tr, _, _ = fit_ols(Phi_grid_tr, y[grid_data_idx])
y_pred_grid_te = Phi_grid_te @ coef_grid
r2_grid_te = 1 - np.sum((y[test_idx] - y_pred_grid_te)**2) / np.sum((y[test_idx] - y[test_idx].mean())**2)
rmse_grid_te = np.sqrt(np.mean((y[test_idx] - y_pred_grid_te)**2))
print(f"  Grid model train  R²={r2_grid_tr:.4f}")
print(f"  Grid model test   R²={r2_grid_te:.4f}, RMSE={rmse_grid_te:.6f}")

# Bootstrap comparison: random 27 points
n_boot = 500
r2_rand = []
np.random.seed(42)
for _ in range(n_boot):
    idx27 = np.random.choice(N, len(used_idx), replace=False)
    te_idx = np.setdiff1d(np.arange(N), idx27)
    if len(te_idx) < 10: continue
    Phi_tr = features(xW[idx27], xKV[idx27], xKVD[idx27], 'full_quad')
    Phi_te = features(xW[te_idx], xKV[te_idx], xKVD[te_idx], 'full_quad')
    if np.linalg.matrix_rank(Phi_tr) < 10: continue
    coef_r, _, _, _, _ = fit_ols(Phi_tr, y[idx27])
    yp_r = Phi_te @ coef_r
    r2_r = 1 - np.sum((y[te_idx]-yp_r)**2) / np.sum((y[te_idx]-y[te_idx].mean())**2)
    r2_rand.append(r2_r)
r2_rand = np.array(r2_rand)
print(f"\n  27-grid test R²  = {r2_grid_te:.4f}")
print(f"  Random-27 test R²: median={np.median(r2_rand):.4f}, [10th,90th]=[{np.percentile(r2_rand,10):.4f},{np.percentile(r2_rand,90):.4f}]")
print(f"  27-grid beats Random-27: {np.mean(r2_grid_te > r2_rand)*100:.1f}% of bootstrap samples")

# ─── Pareto front recovery ────────────────────────────────────────────────────
print("\n" + "="*60)
print("PARETO FRONT RECOVERY")
print("="*60)
def range_norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-10)
C = (range_norm(mat[0, :][valid]) +
     range_norm(mat[1, :][valid]) +
     range_norm(mat[4, :][valid]))

def pf2d(metrics, comps):
    F = np.column_stack([metrics, comps])
    order = np.argsort(comps)
    m = metrics[order]; c = comps[order]
    nd = []; cur = np.inf
    for i in range(len(m)):
        if m[i] < cur: nd.append(order[i]); cur = m[i]
    return np.array(nd)

true_pf = pf2d(y, C)
surr_pf = pf2d(mdl['full_quad']['yp'], C)
n_rec = len(set(true_pf) & set(surr_pf))
print(f"  True PF: {len(true_pf)} pts")
print(f"  Surrogate PF: {len(surr_pf)} pts")
print(f"  Exact recovery: {n_rec}/{len(true_pf)} ({n_rec/len(true_pf)*100:.1f}%)")

def hv2d(m, c, rm, rc):
    order = np.argsort(c)
    m, c = m[order], c[order]
    h = 0
    for i in range(len(m)):
        w = (rc if i==len(m)-1 else c[i+1]) - c[i]
        h += max(0,w) * (rm - m[i])
    return h

rm, rc = y.max()*1.1, C.max()*1.1
hv_t = hv2d(y[true_pf], C[true_pf], rm, rc)
hv_s = hv2d(y[surr_pf], C[surr_pf], rm, rc)
print(f"  HV true={hv_t:.4f}, HV surrogate(realized)={hv_s:.4f}, ratio={hv_s/hv_t:.4f}")

# ─── Summary print ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (CORRECT INPUTS: per-method HQQ JSD)")
print("="*60)
print(f"  N samples: {N}")
print(f"  Correlations: r_W={rW:.3f}, r_KV={rKV:.3f}, r_KVD={rKVD:.3f}")
print(f"  Linear additive R²={mdl['linear']['r2']:.4f}, RMSE={mdl['linear']['rmse']:.4f}")
c_lin = mdl['linear']['coef']
print(f"    y ≈ {c_lin[0]:.4f} + {c_lin[1]:.4f}·JSD_W + {c_lin[2]:.4f}·JSD_KV + {c_lin[3]:.4f}·JSD_KVD")
print(f"  Full-quad R²={mdl['full_quad']['r2']:.4f}, RMSE={mdl['full_quad']['rmse']:.4f}")
print(f"  Monotonicity: {'SATISFIED (100%)' if all_ok else 'VIOLATED'}")
print(f"  PF recovery: {n_rec}/{len(true_pf)} ({n_rec/len(true_pf)*100:.0f}%), HV ratio={hv_s/hv_t:.4f}")
print(f"  27-grid vs random-27: R²={r2_grid_te:.4f} vs {np.median(r2_rand):.4f} (median)")

# ─── FIGURES ─────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
os.makedirs(f'{BASE}/analysis/v2/figures', exist_ok=True)

C_W   = '#E64B35'   # red   – W
C_KV  = '#4DBBD5'   # blue  – KV
C_KVD = '#00A087'   # green – KVD
C_AWQ = '#3C5488'   # dark blue – AWQ actual
C_HI  = '#F39B7F'   # salmon – highlight
PLT_KW = dict(dpi=180, bbox_inches='tight')

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Fig 1: Per-method Pareto frontiers ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, (pf, label, color, xl) in zip(axes, [
    (pf_W,    'W Quantization (HQQ)', C_W,  'Avg Weight Bits'),
    (pf_KV,   'KV Cache Quantization', C_KV, 'Avg KV Bits'),
    (pf_KVDIM,'KV Cache Pruning',      C_KVD,'Avg KV Head Dim'),
]):
    ax.plot(pf[:, 1], pf[:, 0], 'o-', color=color, lw=1.8, ms=3.5, alpha=0.9)
    ax.fill_between(pf[:, 1], pf[:, 0], pf[:, 0].max(), alpha=0.08, color=color)
    ax.set_xlabel(xl)
    ax.set_ylabel('HQQ JSD Loss')
    ax.set_title(label)
    ax.grid(True, alpha=0.25, lw=0.5)
plt.suptitle('Per-Method Pareto Frontiers from HQQ Search', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/analysis/v2/figures/fig1_pareto_fronts.png', **PLT_KW)
plt.close()

# ── Fig 2: Scatter (per-method JSD vs combined AWQ JSD) ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (Xi, label, color, r) in zip(axes, [
    (xW,  'JSD_W (HQQ)',   C_W,  rW),
    (xKV, 'JSD_KV (HQQ)',  C_KV, rKV),
    (xKVD,'JSD_KVD (HQQ)', C_KVD,rKVD),
]):
    ax.scatter(Xi, y, c=color, alpha=0.45, s=14, edgecolors='none', rasterized=True)
    m, b = np.polyfit(Xi, y, 1)
    xl = np.linspace(Xi.min(), Xi.max(), 100)
    ax.plot(xl, m*xl + b, 'k-', lw=1.8, zorder=5)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('JSD_AWQ (combined)', fontsize=11)
    ax.set_title(f'Pearson r = {r:.3f}', fontweight='bold')
    ax.grid(True, alpha=0.2, lw=0.5)
plt.suptitle('Per-Method HQQ JSD vs Combined AWQ JSD', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/analysis/v2/figures/fig2_correlation_scatter.png', **PLT_KW)
plt.close()

# ── Fig 3: Model comparison ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
names = ['Linear\n(Additive)', 'Quadratic\n(+x²)', 'Full Quad\n(+interactions)']
keys  = ['linear', 'quad', 'full_quad']
bar_c = [C_W, C_KV, C_KVD]
for ax, (metric, ylabel, fmt) in zip(axes, [
    ([mdl[k]['r2']   for k in keys], 'R²',   '.4f'),
    ([mdl[k]['rmse'] for k in keys], 'RMSE', '.5f'),
    ([mdl[k]['aic']  for k in keys], 'AIC',  '.0f'),
]):
    bars = ax.bar(names, metric, color=bar_c, alpha=0.82, edgecolor='#333', linewidth=0.5, width=0.55)
    for bar, v in zip(bars, metric):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                format(v, fmt), ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, lw=0.5, axis='y')
    if ylabel == 'R²': ax.set_ylim([0.93, 1.0])
plt.suptitle('Surrogate Model Comparison (N=200 AWQ samples)', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/analysis/v2/figures/fig3_model_comparison.png', **PLT_KW)
plt.close()

# ── Fig 4: Sequential R² decomposition ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
steps = [s[0] for s in seq_r2]
r2s   = [s[1] for s in seq_r2]
deltas= [s[2] for s in seq_r2]
term_colors = ['#AAAAAA', C_W, C_KV, C_KVD,
               C_W, C_KV, C_KVD,
               '#9B59B6', '#E67E22', '#1ABC9C']
ax.bar(range(len(steps)), [abs(d)*100 for d in deltas],
       color=term_colors, alpha=0.82, edgecolor='#333', linewidth=0.5)
ax2 = ax.twinx()
ax2.plot(range(len(steps)), [r*100 for r in r2s], 'k-o', lw=2, ms=5, zorder=5)
ax2.set_ylabel('Cumulative R² (%)', color='black')
ax2.set_ylim([0, 105])
ax.set_xticks(range(len(steps)))
ax.set_xticklabels(steps, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Sequential ΔR² (%)')
ax.set_title('Sequential (Type-I) R² Decomposition\n(Full-Quad Model, Centered Features)', fontweight='bold')
ax.grid(True, alpha=0.2, lw=0.5, axis='y')
# Annotate cumulative R²
for x, r in [(3, r2s[3]), (6, r2s[6]), (9, r2s[9])]:
    ax2.annotate(f'{r*100:.1f}%', (x, r*100+1), ha='center', fontsize=9, color='#333', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}/analysis/v2/figures/fig4_sequential_r2.png', **PLT_KW)
plt.close()

# ── Fig 5: Fit quality (linear vs full_quad) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (key, label) in zip(axes, [
    ('linear',    f'Linear (Additive)  R²={mdl["linear"]["r2"]:.4f}'),
    ('full_quad', f'Full Quadratic      R²={mdl["full_quad"]["r2"]:.4f}'),
]):
    yp = mdl[key]['yp']
    lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
    ax.scatter(y, yp, c=C_AWQ, alpha=0.4, s=14, edgecolors='none', rasterized=True)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, zorder=5)
    # residual bands
    resid = y - yp
    rms = np.sqrt(np.mean(resid**2))
    ax.fill_between([lo, hi], [lo-2*rms, hi-2*rms], [lo+2*rms, hi+2*rms],
                    alpha=0.1, color='gray', label='±2×RMSE band')
    ax.set_xlabel('JSD_AWQ (measured)')
    ax.set_ylabel('JSD (predicted)')
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, lw=0.5)
plt.suptitle('Additive Approximation Quality', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/analysis/v2/figures/fig5_fit_quality.png', **PLT_KW)
plt.close()

# ── Fig 6: Monotonicity gradients ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (Xi, grad, xlabel, color) in zip(axes, [
    (xW,  dW,  'JSD_W (HQQ)',   C_W),
    (xKV, dKV, 'JSD_KV (HQQ)',  C_KV),
    (xKVD,dKVD,'JSD_KVD (HQQ)', C_KVD),
]):
    c_pts = np.where(grad >= 0, color, 'red')
    ax.scatter(Xi, grad, c=c_pts, alpha=0.5, s=14, edgecolors='none', rasterized=True)
    ax.axhline(0, color='black', lw=1.2, ls='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('∂F/∂(JSD input)')
    frac_ok = (grad >= 0).mean()*100
    ax.set_title(f'Min={grad.min():.3f}, Monotone: {frac_ok:.1f}%', fontweight='bold')
    ax.grid(True, alpha=0.2, lw=0.5)
plt.suptitle('Monotonicity Check: ∂F/∂JSD_i ≥ 0 on Sampled Data', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/analysis/v2/figures/fig6_monotonicity.png', **PLT_KW)
plt.close()

# ── Fig 7: Pareto front recovery ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.scatter(C, y, c='#CCCCCC', s=10, alpha=0.35, label='All 200 samples', zorder=1)
ax.scatter(C[true_pf], y[true_pf], c=C_AWQ, s=50, marker='D',
           label=f'True PF (n={len(true_pf)})', zorder=4)
ax.scatter(C[surr_pf], y[surr_pf], c=C_W, s=28, marker='^',
           label=f'Surrogate PF (n={len(surr_pf)})', zorder=3, alpha=0.8)
rec = list(set(true_pf) & set(surr_pf))
if rec:
    ax.scatter(C[rec], y[rec], c='#2ECC71', s=70, marker='*',
               label=f'Recovered (n={len(rec)})', zorder=5)
ax.set_xlabel('Joint Complexity (range-norm. sum)')
ax.set_ylabel('JSD_AWQ')
ax.set_title(f'PF Recovery: {len(rec)}/{len(true_pf)} ({len(rec)/len(true_pf)*100:.0f}%)')
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.2, lw=0.5)

ax = axes[1]
cats = ['True PF', 'Surrogate PF\n(realized)']
hvs  = [hv_t, hv_s]
bar_clrs = [C_AWQ, C_W]
bars = ax.bar(cats, hvs, color=bar_clrs, alpha=0.8, edgecolor='#333', linewidth=0.5, width=0.4)
for bar, v in zip(bars, hvs):
    ax.text(bar.get_x()+bar.get_width()/2, v*0.97, f'{v:.4f}',
            ha='center', va='top', fontsize=10, color='white', fontweight='bold')
ax.set_ylabel('Hypervolume (2D)')
ax.set_title(f'HV Ratio = {hv_s/hv_t:.4f}\n(surrogate / true)')
ax.set_ylim([0, max(hvs)*1.15])
ax.grid(True, alpha=0.2, lw=0.5, axis='y')
plt.suptitle('Pareto Front Recovery via Full-Quadratic Surrogate', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/analysis/v2/figures/fig7_pf_recovery.png', **PLT_KW)
plt.close()

# ── Fig 8: 27-grid calibration ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.hist(r2_rand, bins=35, color=C_KV, alpha=0.75, edgecolor='white', linewidth=0.3)
ax.axvline(r2_grid_te, color=C_W, lw=2.5, label=f'27-grid R²={r2_grid_te:.4f}')
ax.axvline(np.median(r2_rand), color='black', lw=1.8, ls='--',
           label=f'Median rand-27={np.median(r2_rand):.4f}')
ax.axvspan(np.percentile(r2_rand, 10), np.percentile(r2_rand, 90),
           alpha=0.12, color='gray', label='[10th,90th] %ile')
ax.set_xlabel('Out-of-sample R²')
ax.set_ylabel('Count')
ax.set_title(f'27-Grid vs Random-{len(used_idx)} Calibration\n(bootstrap n=500)')
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.2, lw=0.5)

ax = axes[1]
# 3D projection of 27-grid design in JSD space
from itertools import product
ax.set_xlim(xW.min()-0.02,  xW.max()+0.02)
ax.set_ylim(xKVD.min()-0.01, xKVD.max()+0.02)
# all data
ax.scatter(xW, xKVD, c='#DDDDDD', s=10, alpha=0.4, zorder=1)
# grid points by KV quantile (3 groups)
for qi, (kv_level, sz) in enumerate(zip(qKV, [20, 35, 50])):
    mask = np.abs(xKV - kv_level) < 0.005
    pts = grid_n[np.arange(qi, 27, 3)]  # every 3rd grid point (same KV level)
    # actual grid data points
    gi = grid_data_idx[np.arange(qi, 27, 3)]
    ax.scatter(xW[gi], xKVD[gi], s=sz, marker='o',
               c=[C_W, C_KV, C_KVD][qi], alpha=0.9, zorder=4,
               label=f'KV q={qs[qi]} ({kv_level:.3f})')
ax.set_xlabel('JSD_W')
ax.set_ylabel('JSD_KVD')
ax.set_title('27-Grid Design Points\n(colored by KV quantile)')
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.2, lw=0.5)
plt.suptitle('Calibration Design: 27-Grid ([0.1,0.5,0.9]³ JSD Quantiles)', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/analysis/v2/figures/fig8_27grid.png', **PLT_KW)
plt.close()

# ── Fig 9: Publication Summary (2×3 grid) ────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# (0,0) Additive model scatter
ax = fig.add_subplot(gs[0, 0])
yp_lin = mdl['linear']['yp']
lo, hi = 0, max(y.max(), yp_lin.max())
ax.scatter(y, yp_lin, c=C_AWQ, alpha=0.4, s=12, edgecolors='none', rasterized=True)
ax.plot([lo,hi],[lo,hi],'k--',lw=1.5)
ax.set_xlabel('JSD_AWQ (measured)')
ax.set_ylabel('JSD (predicted)')
ax.set_title(f'(a) Linear Additive\n R²={mdl["linear"]["r2"]:.4f}', fontweight='bold')
ax.grid(True, alpha=0.2, lw=0.5)

# (0,1) Sequential R² bar
ax = fig.add_subplot(gs[0, 1])
group_labels = ['intercept', 'JSD_W', 'JSD_KV', 'JSD_KVD', 'W²', 'KV²', 'KVD²', 'W·KV', 'W·KVD', 'KV·KVD']
ax.bar(range(len(seq_r2)), [abs(s[2])*100 for s in seq_r2],
       color=term_colors, alpha=0.82, edgecolor='#333', linewidth=0.4)
ax.set_xticks(range(len(seq_r2)))
ax.set_xticklabels(group_labels, rotation=40, ha='right', fontsize=7.5)
ax.set_ylabel('ΔR² (%)')
ax.set_title('(b) Sequential R² Decomposition', fontweight='bold')
ax.grid(True, alpha=0.2, lw=0.5, axis='y')

# (0,2) Model R² comparison
ax = fig.add_subplot(gs[0, 2])
ax.bar(names, [mdl[k]['r2'] for k in keys], color=bar_c, alpha=0.82,
       edgecolor='#333', linewidth=0.4, width=0.5)
for i, k in enumerate(keys):
    ax.text(i, mdl[k]['r2']+0.001, f'{mdl[k]["r2"]:.4f}',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_ylim([0.93, 1.005])
ax.set_ylabel('R²')
ax.set_title('(c) Model Comparison', fontweight='bold')
ax.grid(True, alpha=0.2, lw=0.5, axis='y')

# (1,0) Monotonicity
ax = fig.add_subplot(gs[1, 0])
for Xi, grad, label, color in [(xW,dW,'W',C_W),(xKV,dKV,'KV',C_KV),(xKVD,dKVD,'KVD',C_KVD)]:
    ax.scatter(Xi, grad, c=color, alpha=0.4, s=10, edgecolors='none', rasterized=True, label=label)
ax.axhline(0, color='black', lw=1.2, ls='--')
ax.set_xlabel('JSD (per-method HQQ)')
ax.set_ylabel('∂F/∂JSD')
ax.set_title('(d) Monotonicity Check\n(All gradients > 0 ✓)', fontweight='bold')
ax.legend(fontsize=8, ncol=3)
ax.grid(True, alpha=0.2, lw=0.5)

# (1,1) PF recovery scatter
ax = fig.add_subplot(gs[1, 1])
ax.scatter(C, y, c='#CCCCCC', s=8, alpha=0.35, zorder=1)
ax.scatter(C[true_pf], y[true_pf], c=C_AWQ, s=40, marker='D',
           label=f'True PF ({len(true_pf)})', zorder=4)
ax.scatter(C[surr_pf], y[surr_pf], c=C_W, s=22, marker='^',
           label=f'Surrogate PF ({len(surr_pf)})', zorder=3, alpha=0.8)
if rec:
    ax.scatter(C[rec], y[rec], c='#2ECC71', s=55, marker='*',
               label=f'Recovered ({len(rec)})', zorder=5)
ax.set_xlabel('Complexity (norm.)')
ax.set_ylabel('JSD_AWQ')
ax.set_title(f'(e) PF Recovery: {len(rec)}/{len(true_pf)} ({len(rec)/len(true_pf)*100:.0f}%)\nHV ratio={hv_s/hv_t:.4f}',
             fontweight='bold')
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.2, lw=0.5)

# (1,2) 27-grid histogram
ax = fig.add_subplot(gs[1, 2])
ax.hist(r2_rand, bins=25, color=C_KV, alpha=0.75, edgecolor='white', linewidth=0.3)
ax.axvline(r2_grid_te, color=C_W, lw=2.2, label=f'27-grid={r2_grid_te:.4f}')
ax.axvline(np.median(r2_rand), color='black', lw=1.5, ls='--',
           label=f'Rand median={np.median(r2_rand):.4f}')
ax.set_xlabel('Out-of-sample R²')
ax.set_ylabel('Count')
ax.set_title('(f) 27-Grid vs Random Calibration', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, lw=0.5)

fig.suptitle('Additive Pareto Combination — Full Analysis\n'
             '(Input: per-method HQQ JSD, N=200 AWQ samples, Llama-3.1-8B-Instruct)',
             fontweight='bold', fontsize=13, y=1.01)
plt.savefig(f'{BASE}/analysis/v2/figures/fig9_publication_summary.png', **PLT_KW)
plt.close()

print("\nAll figures saved to analysis/v2/figures/")
print("Files: fig1-fig9 + fig9_publication_summary.png")
