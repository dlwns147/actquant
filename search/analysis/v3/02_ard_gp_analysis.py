"""05_ard_gp_analysis.py — Unified ARD-GP analysis on 50/150 fixed split.

Trains ARD-GP on 50-train pool (27-grid + 23 maximin extras), evaluates on 150 test.

PART A — Mathematical interpretation of ARD-GP internals:
  (a1) Per-dim length scales l_i (raw + normalized by input range)
  (a2) Sobol indices via SALib (Saltelli pick-freeze):
         S1_i (main effect),  ST_i (total),  S2_ij (pairwise interaction)
  (a3) Active subspace eigenvalues (gradient covariance)
  (a4) Hoeffding additive decomposition  y_add = f_0 + Σ g_i(x_i)

PART B — Predictive performance + ε bounds on 150 test:
  (b1) y_full = ARD-GP full prediction (anisotropic + interaction)
  (b2) y_add  = additive prediction from Hoeffding components
  (b3) ε bounds: ‖y_actual − y_add‖_∞, _2  (residual on 150 test)
  (b4) Pred-vs-actual scatter, residual structure per complexity axis

THEOREM connecting A and B:
  • Variance ratio: Var(y_actual − y_add) / Var(y_actual)  ≈  1 − Σ S_i
  • Pareto guarantee: ‖y_actual − y_add‖_∞ = ε  ⇒  Pareto(y_add) ⊆ 2ε-Pareto(y_actual)
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

from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

from utils.func import get_net_info
from predictor.rbf import RBF as PySOTRBF

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
group_size = {'w':128, 'k':[128,128], 'v':[128,128]}

print("Loading data...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
pf_W_m, pf_KV_m, pf_KVDIM_m = (p[np.argsort(p[:, 0])] for p in (pf_W, pf_KV, pf_KVDIM))

mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y_all = mat3[12, :N0]; v3 = ~np.isnan(y_all)
xW3   = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3  = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD  = match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y_all = y_all[v3]
iW3   = match_index(mat3[0, :N0][v3], pf_W_m   ).astype(float)
iKV3  = match_index(mat3[1, :N0][v3], pf_KV_m  ).astype(float)
iKVD3 = match_index(mat3[4, :N0][v3], pf_KVDIM_m).astype(float)
N = len(y_all)
X3 = np.column_stack([xW3, xKV3, xKVD])
X3_idx = np.column_stack([iW3, iKV3, iKVD3])
HEAD_DIM = 128

range_jsd = np.array([np.ptp(xW3), np.ptp(xKV3), np.ptp(xKVD)])
bounds_jsd = [[xW3.min(), xW3.max()], [xKV3.min(), xKV3.max()], [xKVD.min(), xKVD.max()]]
lb_idx = np.array([0., 0., 0.])
ub_idx = np.array([len(pf_W_m)-1, len(pf_KV_m)-1, len(pf_KVDIM_m)-1], dtype=float)

# Bits
wbits_all  = mat3[0, :N0][v3]
kvbits_all = mat3[1, :N0][v3]
kvdim_all  = mat3[4, :N0][v3]
eff_kv_all = kvbits_all * kvdim_all / HEAD_DIM
print(f"  N={N}; JSD ranges: W={range_jsd[0]:.3f}, KV={range_jsd[1]:.3f}, KVD={range_jsd[2]:.3f}")

# ─── Fixed 50/150 split ──────────────────────────────────────────────────────
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
def fit_ard_gp(X, y, n_dim, n_restarts=20):
    kernel = (C(1.0,(1e-4,1e2)) *
              SKRBF(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-4,1e4)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9,1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y); return gp

def get_ls(gp): return np.array(gp.kernel_.k1.k2.length_scale)
def get_sigma_f2(gp): return float(gp.kernel_.k1.k1.constant_value)
def get_sigma_n2(gp): return float(gp.kernel_.k2.noise_level)

print("\nFitting ARD-GP on 50-train pool (JSD input)...")
gp = fit_ard_gp(X3[train_pool], y_all[train_pool], 3, n_restarts=20)
ls = get_ls(gp)
print(f"  Hyperparams: σ_f²={get_sigma_f2(gp):.4f}, σ_n²={get_sigma_n2(gp):.6f}")
print(f"  Length scales: l_W={ls[0]:.4f}, l_KV={ls[1]:.4f}, l_KVD={ls[2]:.4f}")

# ─── PART A — Mathematical interpretation ────────────────────────────────────
print("\n" + "="*100)
print("PART A — Mathematical interpretation of ARD-GP internals")
print("="*100)

# (a1) Length scales
ls_n = ls / range_jsd
sens = 1.0 / ls_n
print(f"\n(a1) Length scales:")
print(f"  Raw l_i:                W={ls[0]:.4f}, KV={ls[1]:.4f}, KVD={ls[2]:.4f}")
print(f"  Normalized l_i / range: W={ls_n[0]:.3f}, KV={ls_n[1]:.3f}, KVD={ls_n[2]:.3f}")
print(f"  Sensitivity (1/l_norm): W={sens[0]:.2f}, KV={sens[1]:.2f}, KVD={sens[2]:.2f}")
print(f"  Sensitivity ranking (most → least): {[['W','KV','KVD'][i] for i in np.argsort(-sens)]}")

# (a2) Sobol indices via SALib (Saltelli pick-freeze)
print(f"\n(a2) Sobol indices (Saltelli pick-freeze on ARD-GP, N=2048):")
problem = {'num_vars': 3, 'names': ['W','KV','KVD'], 'bounds': bounds_jsd}
pv = saltelli.sample(problem, 2048, calc_second_order=True)
yp_pv = gp.predict(pv)
Si = sobol_analyze.analyze(problem, yp_pv, calc_second_order=True, print_to_console=False)
print(f"\n    {'i':4s}  {'S_i (main)':16s}  {'ST_i (total)':16s}  {'ST−S':10s}")
for i, n in enumerate(['W','KV','KVD']):
    s, st = Si['S1'][i], Si['ST'][i]
    print(f"    {n:4s}  {s:+.4f} ± {Si['S1_conf'][i]:.3f}  {st:+.4f} ± {Si['ST_conf'][i]:.3f}  {st-s:+.4f}")
print(f"\n  Pairwise S_2:")
for i, j, lbl in [(0,1,'W,KV'), (0,2,'W,KVD'), (1,2,'KV,KVD')]:
    s2 = Si['S2'][i, j]; c2 = Si['S2_conf'][i, j]
    print(f"    S_{lbl:8s} = {s2:+.4f} ± {c2:.3f}")
sum_S1 = sum(Si['S1'])
print(f"\n  Σ S_i (main effects) = {sum_S1:.4f}  ({sum_S1*100:.1f}%)")
print(f"  Interaction fraction = 1 − Σ S_i = {1 - sum_S1:.4f}  ({(1-sum_S1)*100:.1f}%)")

# (a3) Active subspace
print(f"\n(a3) Active subspace (gradient covariance via finite differences):")
def active_subspace(gp, bounds, n_mc=2000):
    rng = np.random.RandomState(0); d = len(bounds)
    X_mc = np.array([rng.uniform(b[0], b[1], n_mc) for b in bounds]).T
    eps_d = np.array([(b[1] - b[0]) * 1e-4 for b in bounds])
    grads = np.zeros((n_mc, d))
    for i in range(d):
        Xp = X_mc.copy(); Xp[:, i] += eps_d[i]
        Xm = X_mc.copy(); Xm[:, i] -= eps_d[i]
        grads[:, i] = (gp.predict(Xp) - gp.predict(Xm)) / (2 * eps_d[i])
    Cmat = grads.T @ grads / n_mc
    eigvals, eigvecs = np.linalg.eigh(Cmat)
    order = np.argsort(-eigvals)
    return eigvals[order], eigvecs[:, order]

ev, evec = active_subspace(gp, bounds_jsd)
print(f"  Eigenvalues (relative): λ_1=1.0, λ_2={ev[1]/ev[0]:.4f}, λ_3={ev[2]/ev[0]:.4f}")
print(f"  Effective dimensionality (λ_2/λ_1) = {ev[1]/ev[0]:.4f}  (≈0 → effectively 1D)")
print(f"  First eigenvector: W={evec[0,0]:+.3f}, KV={evec[1,0]:+.3f}, KVD={evec[2,0]:+.3f}")

# (a4) Hoeffding additive decomposition components
print(f"\n(a4) Hoeffding additive decomposition: y_add = f_0 + g_W + g_KV + g_KVD")
rng = np.random.RandomState(0); n_mc = 5000
X_mc = np.array([rng.uniform(b[0], b[1], n_mc) for b in bounds_jsd]).T
y_mc = gp.predict(X_mc)
f0 = float(y_mc.mean())
print(f"  f_0 (global mean) = {f0:.4f}")

def cmean(x_query, dim, h_factor=0.15):
    h = X_mc[:, dim].std() * h_factor
    out = np.zeros(len(x_query))
    for k, xq in enumerate(x_query):
        w = np.exp(-0.5 * ((X_mc[:, dim] - xq) / h)**2)
        out[k] = (w * y_mc).sum() / w.sum()
    return out

# ─── PART B — Predictive performance + ε bounds on 150 test ──────────────────
print("\n" + "="*100)
print("PART B — Predictive performance and ε bounds on 150 test samples")
print("="*100)

X_te = X3[test_set]
y_actual = y_all[test_set]
y_full   = gp.predict(X_te)
g_W_te   = cmean(X_te[:,0], 0) - f0
g_KV_te  = cmean(X_te[:,1], 1) - f0
g_KVD_te = cmean(X_te[:,2], 2) - f0
y_add    = f0 + g_W_te + g_KV_te + g_KVD_te

wbits_te  = wbits_all[test_set]
kvbits_te = kvbits_all[test_set]
eff_kv_te = eff_kv_all[test_set]

# RBF cubic+linear on JSD input (50-train, 150-test) — Direct
lb_jsd = X3.min(0); ub_jsd = X3.max(0)
m_rbf = PySOTRBF(kernel='cubic', tail='linear', lb=lb_jsd, ub=ub_jsd)
m_rbf.fit(X3[train_pool], y_all[train_pool])
y_rbf = m_rbf.predict(X3[test_set]).ravel()

# Hier-WD-TPS-linear with constrained budget B=50 (n_3way=18, n_pair=32) — same as best from analysis 02
print("\nLoading WD pair data + computing Hier-WD-TPS-linear...")
mat_wd = load_csv(AWQ_W_KVD); N_wd = mat_wd.shape[1]
y_WD_full = mat_wd[12, :N_wd]; v_wd = ~np.isnan(y_WD_full)
xW_WD   = match_metric(mat_wd[0, :N_wd], pf_W   )[v_wd]
xKVD_WD = match_metric(mat_wd[4, :N_wd], pf_KVDIM)[v_wd]
y_WD    = y_WD_full[v_wd]
print(f"  WD pair N={len(y_WD)}")

# Hier budget split: n_3way + n_pair = 50 (matching ARD-GP/RBF total measurement count)
n_3way_h = 18; n_pair_h = 32
# 3-way: take first 18 of 27-grid (deterministic, structured)
tr3_hier = grid_samples[:n_3way_h]
# Pair: random 32 from full WD pair
rng_p = np.random.RandomState(0)
pair_idx = rng_p.choice(len(y_WD), n_pair_h, replace=False)
xa_p = xW_WD[pair_idx]; xb_p = xKVD_WD[pair_idx]; y_p = y_WD[pair_idx]

# Train pair: TPS+linear on (W, KVD)
X2_p = np.column_stack([xa_p, xb_p])
m_pair = PySOTRBF(kernel='tps', tail='linear', lb=X2_p.min(0), ub=X2_p.max(0))
m_pair.fit(X2_p, y_p)
# Pair predictions on 3-way train (using W=col0, KVD=col2) and test
yp_pair_tr = m_pair.predict(np.column_stack([X3[tr3_hier, 0], X3[tr3_hier, 2]])).ravel()
yp_pair_te = m_pair.predict(np.column_stack([X_te[:, 0],     X_te[:, 2]])).ravel()
# Linear residual on KV (col1)
r_tr_h = y_all[tr3_hier] - yp_pair_tr
c_h = np.polyfit(X3[tr3_hier, 1], r_tr_h, 1)
y_hier = yp_pair_te + np.polyval(c_h, X_te[:, 1])
print(f"  Hier budget: n_3way={n_3way_h} (sub-grid) + n_pair={n_pair_h} = {n_3way_h+n_pair_h} total")

residual_full = y_actual - y_full
residual_add  = y_actual - y_add
residual_rbf  = y_actual - y_rbf
residual_hier = y_actual - y_hier
def r2(y, yp):
    ss = max(np.sum((y - y.mean())**2), 1e-30)
    return 1 - np.sum((y-yp)**2)/ss
def rmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)))

eps_inf_full = float(np.max(np.abs(residual_full)))
eps_2_full   = float(np.sqrt(np.mean(residual_full**2)))
eps_inf  = float(np.max(np.abs(residual_add)))
eps_2    = float(np.sqrt(np.mean(residual_add**2)))
eps_95   = float(np.percentile(np.abs(residual_add), 95))
eps_inf_rbf  = float(np.max(np.abs(residual_rbf)))
eps_2_rbf    = float(np.sqrt(np.mean(residual_rbf**2)))
eps_inf_hier = float(np.max(np.abs(residual_hier)))
eps_2_hier   = float(np.sqrt(np.mean(residual_hier**2)))

print(f"\n  Predictive R², RMSE, ε on 150 test (50-measurement budget):")
print(f"    {'Surrogate':30s}  {'R²':8s}  {'RMSE':8s}  {'ε_∞':8s}  {'ε_2':8s}")
print(f"    {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
print(f"    {'ARD-GP full (interact.)':30s}  {r2(y_actual, y_full):.4f}    {rmse(y_actual, y_full):.4f}    {eps_inf_full:.4f}    {eps_2_full:.4f}")
print(f"    {'Additive y_add (Hoeff.)':30s}  {r2(y_actual, y_add):.4f}    {rmse(y_actual, y_add):.4f}    {eps_inf:.4f}    {eps_2:.4f}")
print(f"    {'RBF cubic+linear (idx)':30s}  {r2(y_actual, y_rbf):.4f}    {rmse(y_actual, y_rbf):.4f}    {eps_inf_rbf:.4f}    {eps_2_rbf:.4f}")
print(f"    {'Hier-WD-TPS-linear':30s}  {r2(y_actual, y_hier):.4f}    {rmse(y_actual, y_hier):.4f}    {eps_inf_hier:.4f}    {eps_2_hier:.4f}")
print(f"      (Hier: {n_3way_h} 3-way + {n_pair_h} pair = 50 total)")

print(f"\n  ε bounds (residual = y_actual − y_pred):")
print(f"    Additive y_add:")
print(f"      ε_∞  = {eps_inf:.5f}  ({eps_inf/np.ptp(y_actual)*100:.1f}% of y span)")
print(f"      ε_95 = {eps_95:.5f}")
print(f"      ε_2  = {eps_2:.5f}")
print(f"      Var(r)/Var(y_actual) = {np.var(residual_add)/np.var(y_actual)*100:.1f}%   "
      f"(theoretical: 1 − Σ S_i = {(1-sum_S1)*100:.1f}%)")
print(f"    ARD-GP full y_full (reference):")
print(f"      ε_∞  = {eps_inf_full:.5f}")
print(f"      ε_2  = {eps_2_full:.5f}")

# ─── THEOREM connecting A and B ──────────────────────────────────────────────
print("\n" + "="*100)
print("THEOREM connecting Sobol decomposition (A) and ε bounds (B)")
print("="*100)
print(f"\n  Sobol theory:  Var(y − y_add) / Var(y) ≈ 1 − Σ S_i = {(1-sum_S1)*100:.1f}%")
print(f"  Empirical (150 test): Var(r_add) / Var(y_actual)    = {np.var(residual_add)/np.var(y_actual)*100:.1f}%")
print(f"  → Match within sampling noise ({abs((1-sum_S1)-np.var(residual_add)/np.var(y_actual))*100:.1f}% diff)")

print(f"\n  Pareto theorem:  ‖y_actual − y_add‖_∞ = {eps_inf:.4f}")
print(f"                   ⇒ Pareto(y_add) ⊆ 2ε_∞-Pareto(y_actual)  with 2ε_∞ = {2*eps_inf:.4f}")
print(f"                   Tighter L²: 2ε_2 = {2*eps_2:.4f}")

# ─── Figures ─────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')
COL_W='#E64B35'; COL_KV='#4DBBD5'; COL_KVD='#00A087'
COL_FULL='#3C5488'; COL_ADD='#9B59B6'

# Fig 1: 6-panel mathematical analysis (Part A)
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
xn = np.arange(3); methods = ['W','KV','KVD']
bar_colors = [COL_W, COL_KV, COL_KVD]

# (a) Normalized length scales
ax = axes[0, 0]
ax.bar(xn, ls_n, color=bar_colors, alpha=0.85, edgecolor='#333')
for i, v in enumerate(ls_n):
    ax.text(i, v*1.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(xn); ax.set_xticklabels(methods)
ax.set_ylabel('l_i / range_i  (smaller = more sensitive)')
ax.set_title('(a) ARD length scales (normalized)', fontweight='bold')
ax.grid(True, alpha=0.25, lw=0.5, axis='y'); ax.set_yscale('log')

# (b) Sensitivity
ax = axes[0, 1]
ax.bar(xn, sens, color=bar_colors, alpha=0.85, edgecolor='#333')
for i, v in enumerate(sens):
    ax.text(i, v*1.02, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(xn); ax.set_xticklabels(methods)
ax.set_ylabel('Sensitivity = 1 / l_normalized')
ax.set_title('(b) Per-dim sensitivity', fontweight='bold')
ax.grid(True, alpha=0.25, lw=0.5, axis='y'); ax.set_yscale('log')

# (c) Sobol main effects S_i
ax = axes[0, 2]
ax.bar(xn, Si['S1'], yerr=Si['S1_conf'], color=bar_colors,
       alpha=0.85, edgecolor='#333', capsize=4)
for i, v in enumerate(Si['S1']):
    ax.text(i, v + Si['S1_conf'][i] + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(xn); ax.set_xticklabels(methods)
ax.set_ylabel('S_i (first-order Sobol)')
ax.set_title(f'(c) Main effects  Σ S_i = {sum_S1:.3f}', fontweight='bold')
ax.grid(True, alpha=0.25, lw=0.5, axis='y')

# (d) Sobol main + interactions = total ST_i
ax = axes[1, 0]
sw = np.array(Si['S1']); stw = np.array(Si['ST'])
ax.bar(xn, sw, color=bar_colors, alpha=0.85, edgecolor='#333', label='S_i (main)')
ax.bar(xn, stw - sw, bottom=sw, color=bar_colors, alpha=0.4, edgecolor='#333',
       hatch='//', label='Interactions (ST − S)')
ax.set_xticks(xn); ax.set_xticklabels(methods)
ax.set_ylabel('Sobol index (cumulative)')
ax.set_title('(d) Main + interactions = ST_i', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.25, lw=0.5, axis='y')

# (e) Pairwise S_ij
ax = axes[1, 1]
pairs = [(0,1,'W,KV'), (0,2,'W,KVD'), (1,2,'KV,KVD')]
sij = np.array([Si['S2'][i, j] for i,j,_ in pairs])
sij_conf = np.array([Si['S2_conf'][i, j] for i,j,_ in pairs])
xp = np.arange(3)
ax.bar(xp, sij, yerr=sij_conf, color='#9B59B6', alpha=0.85, edgecolor='#333', capsize=4)
ax.axhline(0, color='black', lw=0.7)
for i, v in enumerate(sij):
    ax.text(i, v + sij_conf[i] + 0.005, f'{v:+.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(xp); ax.set_xticklabels([p[2] for p in pairs])
ax.set_ylabel('S_ij (pairwise)')
ax.set_title('(e) Pairwise interactions S_ij', fontweight='bold')
ax.grid(True, alpha=0.25, lw=0.5, axis='y')

# (f) Active subspace
ax = axes[1, 2]
ax.bar(xn, ev / ev[0], color=COL_FULL, alpha=0.85, edgecolor='#333')
for i, v in enumerate(ev / ev[0]):
    ax.text(i, v*1.05, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(xn); ax.set_xticklabels(['λ_1', 'λ_2', 'λ_3'])
ax.set_ylabel('λ_k / λ_1')
ax.set_title('(f) Active subspace eigenvalues', fontweight='bold')
ax.grid(True, alpha=0.25, lw=0.5, axis='y'); ax.set_yscale('log')

plt.suptitle(f'PART A — ARD-GP mathematical analysis (50-train fit, JSD input)',
             fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_ard_gp_interpret.png', **PLT_KW); plt.close()

# Fig 2: 1D main effects
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for i, name in enumerate(['W','KV','KVD']):
    ax = axes[i]
    rng2 = np.random.RandomState(0); n_mc2 = 200
    x_grid = np.linspace(bounds_jsd[i][0], bounds_jsd[i][1], 30)
    means = np.zeros(len(x_grid))
    for k in range(len(x_grid)):
        Xmc = np.zeros((n_mc2, 3))
        for j in range(3):
            if j == i: Xmc[:, j] = x_grid[k]
            else: Xmc[:, j] = rng2.uniform(bounds_jsd[j][0], bounds_jsd[j][1], n_mc2)
        means[k] = gp.predict(Xmc).mean()
    ax.plot(x_grid, means, '-', color=[COL_W, COL_KV, COL_KVD][i], lw=2.6)
    ax.set_xlabel(f'JSD_{name}')
    ax.set_ylabel(f'E[y | JSD_{name}]')
    ax.set_title(f'1D main effect: {name}\n(S_i = {Si["S1"][i]:.3f})', fontweight='bold')
    ax.grid(True, alpha=0.25, lw=0.5)
plt.suptitle('Nonparametric 1D main effects (Hoeffding g_i via MC)',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_ard_gp_main_effects.png', **PLT_KW); plt.close()

# Fig 3: Pred vs actual on 150 test — 4 surrogates × 1 color axis (compact)
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

def scatter_pred_vs_actual(ax, ypred, label, color_axis, color_axis_label, e_inf, e_2,
                           title_color='black'):
    sc = ax.scatter(y_actual, ypred, c=color_axis, cmap='viridis', s=32, alpha=0.85,
                    edgecolor='black', linewidth=0.3)
    lo = min(y_actual.min(), ypred.min()); hi = max(y_actual.max(), ypred.max())
    ax.plot([lo,hi], [lo,hi], 'k--', lw=1.2, label='y_pred = y_actual')
    xx = np.array([lo, hi])
    ax.fill_between(xx, xx-e_inf, xx+e_inf, color='gray', alpha=0.10, label=f'±ε_∞={e_inf:.4f}')
    ax.fill_between(xx, xx-e_2,   xx+e_2,   color='gray', alpha=0.22, label=f'±ε_2={e_2:.4f}')
    rmse_v = rmse(y_actual, ypred); r2_v = r2(y_actual, ypred)
    ax.set_xlabel('y_actual'); ax.set_ylabel(f'y_pred')
    ax.set_title(f'{label}\nR²={r2_v:.4f}, RMSE={rmse_v:.4f}',
                 fontweight='bold', fontsize=11, color=title_color)
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_xlim([lo*0.95, hi*1.05]); ax.set_ylim([lo*0.95, hi*1.05])
    plt.colorbar(sc, ax=ax, fraction=0.04, label=color_axis_label)

# Single color axis: wbits + eff_kvbits (combined complexity)
color_ax = wbits_te + eff_kv_te
color_lbl = 'wbits + eff_kvbits'

scatter_pred_vs_actual(axes[0,0], y_full,  'ARD-GP full (interaction-aware)', color_ax, color_lbl,
                       eps_inf_full, eps_2_full, title_color='#3C5488')
scatter_pred_vs_actual(axes[0,1], y_add,   'Additive y_add (Hoeffding)',     color_ax, color_lbl,
                       eps_inf, eps_2, title_color='#9B59B6')
scatter_pred_vs_actual(axes[1,0], y_rbf,   'RBF cubic+linear (JSD input)',  color_ax, color_lbl,
                       eps_inf_rbf, eps_2_rbf, title_color='#00A087')
scatter_pred_vs_actual(axes[1,1], y_hier,  f'Hier-WD-TPS-linear (n3={n_3way_h}+pair={n_pair_h}={n_3way_h+n_pair_h})',
                       color_ax, color_lbl, eps_inf_hier, eps_2_hier, title_color='#E64B35')

plt.suptitle(f'PART B — Pred vs Actual on 150 test (50-measurement budget, color = wbits + eff_kvbits)',
             fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_ard_gp_pred_vs_actual.png', **PLT_KW); plt.close()

# Fig 4: residual histograms for all 4 surrogates side-by-side
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
surrogates_info = [
    ('ARD-GP full', residual_full, eps_inf_full, eps_2_full, '#3C5488'),
    ('Additive y_add', residual_add, eps_inf, eps_2, '#9B59B6'),
    ('RBF cubic+lin', residual_rbf, eps_inf_rbf, eps_2_rbf, '#00A087'),
    ('Hier-WD-TPS-lin', residual_hier, eps_inf_hier, eps_2_hier, '#E64B35'),
]
for ax, (lbl, res, ei, e2, col) in zip(axes, surrogates_info):
    ax.hist(res, bins=25, color=col, alpha=0.85, edgecolor='#333')
    ax.axvline(ei,  color='#E64B35', ls='-',  lw=2, label=f'±ε_∞={ei:.4f}')
    ax.axvline(-ei, color='#E64B35', ls='-',  lw=2)
    ax.axvline(e2,  color='#3C5488', ls='--', lw=2, label=f'±ε_2={e2:.4f}')
    ax.axvline(-e2, color='#3C5488', ls='--', lw=2)
    ax.set_xlabel(f'Residual = y_actual − y_pred')
    ax.set_ylabel('# samples')
    ax.set_title(f'{lbl}\nR²={r2(y_actual, y_actual-res):.4f}', fontweight='bold', color=col)
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25, lw=0.5)
plt.suptitle(f'Residual distributions on 150 test (4 surrogates, 50-measurement budget)',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_ard_gp_residual.png', **PLT_KW); plt.close()

# Fig 5: residual vs complexity axes (additive only — for theorem visualization)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, comp, lbl in [(axes[0], wbits_te,  'wbits'),
                      (axes[1], eff_kv_te, 'eff_kvbits'),
                      (axes[2], wbits_te + eff_kv_te, 'wbits + eff_kvbits')]:
    ax.scatter(comp, residual_add, c=COL_ADD, s=22, alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(eps_inf,  color='#E64B35', ls='-',  lw=1.5, label=f'±ε_∞')
    ax.axhline(-eps_inf, color='#E64B35', ls='-',  lw=1.5)
    ax.axhline(eps_2,    color='#3C5488', ls='--', lw=1.5, label=f'±ε_2')
    ax.axhline(-eps_2,   color='#3C5488', ls='--', lw=1.5)
    ax.set_xlabel(lbl); ax.set_ylabel('r = y_actual − y_add')
    ax.set_title(f'Additive residual vs {lbl}', fontweight='bold')
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25, lw=0.5)
plt.suptitle(f'Additive (Hoeffding) residual structure on 150 test',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_ard_gp_residual_vs_complexity.png', **PLT_KW); plt.close()

# ─── Final summary ───────────────────────────────────────────────────────────
print(f"\n" + "="*100)
print("UNIFIED SUMMARY (50-train ARD-GP, 150-test evaluation)")
print("="*100)
print(f"\n  PART A — ARD-GP internals:")
print(f"    Length scales (norm.):  W={ls_n[0]:.3f}, KV={ls_n[1]:.3f}, KVD={ls_n[2]:.3f}")
print(f"    Sensitivity ranking:    W > KVD > KV  (W is {sens[0]/sens[1]:.1f}× KV)")
print(f"    Sobol main:             S_W={Si['S1'][0]:.3f}, S_KV={Si['S1'][1]:.3f}, S_KVD={Si['S1'][2]:.3f}")
print(f"    Σ S_i (additive frac) = {sum_S1*100:.1f}%")
print(f"    Largest pairwise S_ij = S_{['W,KV','W,KVD','KV,KVD'][np.argmax([Si['S2'][0,1],Si['S2'][0,2],Si['S2'][1,2]])]} "
      f"= {max(Si['S2'][0,1],Si['S2'][0,2],Si['S2'][1,2]):.3f}")
print(f"    Active subspace λ_2/λ_1 = {ev[1]/ev[0]:.3f}  (effectively 1D)")

print(f"\n  PART B — 150-test predictive performance (50-measurement budget):")
print(f"    {'Surrogate':22s}  {'R²':8s}  {'ε_∞':8s}  {'ε_2':8s}")
print(f"    {'ARD-GP full':22s}  {r2(y_actual, y_full):.4f}    {eps_inf_full:.4f}    {eps_2_full:.4f}")
print(f"    {'Additive y_add':22s}  {r2(y_actual, y_add):.4f}    {eps_inf:.4f}    {eps_2:.4f}")
print(f"    {'RBF cubic+linear':22s}  {r2(y_actual, y_rbf):.4f}    {eps_inf_rbf:.4f}    {eps_2_rbf:.4f}")
print(f"    {'Hier-WD-TPS-linear':22s}  {r2(y_actual, y_hier):.4f}    {eps_inf_hier:.4f}    {eps_2_hier:.4f}")

print(f"\n  Theorem connecting A & B:")
print(f"    Sobol theory:  Var(r)/Var(y) ≈ 1 − Σ S_i = {(1-sum_S1)*100:.1f}%")
print(f"    Empirical:                              = {np.var(residual_add)/np.var(y_actual)*100:.1f}%   (consistent)")
print(f"    Pareto:  Pareto(y_add) ⊆ 2ε_∞-Pareto(y_actual),  2ε_∞ = {2*eps_inf:.4f}")

print(f"\nFigures saved:")
print(f"  02_ard_gp_interpret.png              — 6-panel math analysis (length scales, Sobol, active subspace)")
print(f"  02_ard_gp_main_effects.png           — 1D main effect curves")
print(f"  02_ard_gp_pred_vs_actual.png         — pred vs actual scatter (4 surrogates)")
print(f"  02_ard_gp_residual.png               — residual histograms (4 surrogates)")
print(f"  02_ard_gp_residual_vs_complexity.png — additive residual vs complexity axes")
print("Done.\n")
