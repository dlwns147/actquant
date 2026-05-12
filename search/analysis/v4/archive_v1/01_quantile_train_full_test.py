"""01_quantile_train_full_test.py — Phase 1.

Train surrogates on the 50-sample 3-way quantile-sample set, evaluate on the 200-sample AWQ
3-way set (full hold-out — no overlap by (wbits, kvbits, kvdim)).

Surrogates compared (all use per-method JSD inputs from per-method PFs):
  • M1   linear additive (intercept + 3 main)
  • M10  full quadratic (3 main + 3 squared + 3 pairwise)
  • RBF cubic+linear      (pySOT)
  • RBF tps  +linear      (pySOT)
  • ARD-GP  (sklearn anisotropic — sweep across kernels: RBF/Matérn-5/2/Matérn-3/2/RQ
            × {noise on, noise off}, n_restarts=50; best chosen by train log-marginal-likelihood)

Outputs:
  • figures/v4_fig1_phase1_scatter.png         — true-vs-pred scatter for each surrogate
  • figures/v4_fig2_phase1_residual.png        — residual vs each input dim (best ARD-GP)
  • figures/v4_fig3_phase1_ardgp_kernel_sweep.png — kernel comparison: train LMML vs test R²/RMSE
  • figures/v4_fig4_phase1_sobol.png           — Sobol S1/ST/S2 from best ARD-GP
  • phase1_results.json                        — all metrics (R², RMSE, ε_inf, ε_2, Sobol,
                                                  length scales, kernel sweep)

Note: PF recovery / HV ratio metrics were intentionally removed — they rely on prediction
values to *select* PF members, so they conflate selection bias with surrogate quality.
"""
import sys, os, json, csv, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF as SKRBF, ConstantKernel as C,
                                                WhiteKernel, Matern, RationalQuadratic)
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze
from utils.func import get_net_info
from predictor.rbf import RBF as PySOTRBF

# ─── Helpers ──────────────────────────────────────────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]; min2 = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < min2: nd.append(i); min2 = F_s[i, 1]
    return order[nd]

def load_archive_pareto(stats_path, comp_key, config, group_size):
    with open(stats_path) as f: data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs   = [v[0] for v in archive]
    metrics = np.array([v[1] for v in archive])
    comps   = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
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

def features_M1(X):
    n = len(X); o = np.ones(n)
    return np.column_stack([o, X[:,0], X[:,1], X[:,2]])

def features_M10(X):
    n = len(X); o = np.ones(n)
    w, kv, kvd = X[:,0], X[:,1], X[:,2]
    return np.column_stack([o, w, kv, kvd, w**2, kv**2, kvd**2, w*kv, w*kvd, kv*kvd])

def fit_ols(Phi_tr, y_tr, Phi_te):
    coef, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
    return Phi_tr @ coef, Phi_te @ coef, coef

def make_kernel(name, n_dim):
    """Returns a fresh sklearn-GP kernel instance for a given name."""
    base_C = C(1.0, (1e-4, 1e2))
    ls = [1.0] * n_dim
    ls_bnd = (1e-4, 1e4)
    if name == 'rbf':                core = SKRBF(ls, ls_bnd)
    elif name == 'matern52':         core = Matern(ls, ls_bnd, nu=2.5)
    elif name == 'matern32':         core = Matern(ls, ls_bnd, nu=1.5)
    elif name == 'rq':               core = RationalQuadratic(length_scale=1.0, alpha=1.0,
                                                              length_scale_bounds=ls_bnd,
                                                              alpha_bounds=(1e-2, 1e2))
    else: raise ValueError(name)
    return base_C * core

def fit_ard_gp_variant(X_tr, y_tr, X_te, kernel_name, with_noise, n_restarts=50):
    n_dim = X_tr.shape[1]
    k = make_kernel(kernel_name, n_dim)
    if with_noise:
        k = k + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2))
        alpha = 1e-8
    else:
        alpha = 1e-10
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=alpha)
    gp.fit(X_tr, y_tr)
    return gp.predict(X_tr), gp.predict(X_te), gp

def get_ls_anisotropic(gp, kernel_name, with_noise, n_dim):
    """Extract length scales from the fitted GP kernel."""
    k = gp.kernel_
    # Strip WhiteKernel if present
    if with_noise: k = k.k1
    # k is now C * core
    core = k.k2
    ls = getattr(core, 'length_scale', None)
    if ls is None: return None
    arr = np.atleast_1d(np.asarray(ls, dtype=float))
    if arr.size == 1: arr = np.full(n_dim, arr[0])
    return arr

def r2(y_t, y_p):
    ss_r = np.sum((y_t - y_p)**2); ss_t = np.sum((y_t - y_t.mean())**2)
    return 1 - ss_r / max(ss_t, 1e-30)

def rmse(y_t, y_p): return float(np.sqrt(np.mean((y_t - y_p)**2)))

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = '/NAS/SJ/actquant/search'
OUT  = f'{BASE}/analysis/v4'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
QS_3WAY  = f'{BASE}/save/result/260506/llama_3.1_8b_inst_quantile_sample/2605060818_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_qs_w159_kv159_kvdim159_rs23/results.csv'

with open(f'{BASE}/config/llama.json') as f:
    config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

print("="*100)
print("PHASE 1 — Train on 50 quantile samples, test on 200 AWQ samples")
print("="*100)

print("\nLoading per-method Pareto fronts...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
print(f"  |PF_W|={len(pf_W)}, |PF_KV|={len(pf_KV)}, |PF_KVD|={len(pf_KVDIM)}")

print("\nLoading data...")
mat_qs = load_csv(QS_3WAY); n_qs = mat_qs.shape[1]
mat_aw = load_csv(AWQ_3WAY); n_aw = mat_aw.shape[1]
print(f"  QS train CSV shape: {mat_qs.shape}  (50 architectures)")
print(f"  AWQ test CSV shape: {mat_aw.shape}  (200 architectures)")

# Train: per-method JSD inputs from PF + measured y
y_tr_raw = mat_qs[12, :n_qs]; v_tr = ~np.isnan(y_tr_raw)
xW_tr  = match_metric(mat_qs[0, :n_qs], pf_W   )[v_tr]
xKV_tr = match_metric(mat_qs[1, :n_qs], pf_KV  )[v_tr]
xKVD_tr= match_metric(mat_qs[4, :n_qs], pf_KVDIM)[v_tr]
y_tr   = y_tr_raw[v_tr]
X_tr   = np.column_stack([xW_tr, xKV_tr, xKVD_tr])
N_TR   = len(y_tr)

# Test: same mapping
y_te_raw = mat_aw[12, :n_aw]; v_te = ~np.isnan(y_te_raw)
xW_te  = match_metric(mat_aw[0, :n_aw], pf_W   )[v_te]
xKV_te = match_metric(mat_aw[1, :n_aw], pf_KV  )[v_te]
xKVD_te= match_metric(mat_aw[4, :n_aw], pf_KVDIM)[v_te]
y_te   = y_te_raw[v_te]
X_te   = np.column_stack([xW_te, xKV_te, xKVD_te])
N_TE   = len(y_te)

# Bits / kvdim arrays for PF computation in (y, comp) plane
wbits_te  = mat_aw[0, :n_aw][v_te]
kvbits_te = mat_aw[1, :n_aw][v_te]
kvdim_te  = mat_aw[4, :n_aw][v_te]
HEAD_DIM  = 128
eff_kv_te = kvbits_te * kvdim_te / HEAD_DIM
total_bits_te = wbits_te + eff_kv_te  # complexity proxy

print(f"  Train N={N_TR}, Test N={N_TE}")
print(f"  Train JSD ranges:  W [{xW_tr.min():.3f}, {xW_tr.max():.3f}]  "
      f"KV [{xKV_tr.min():.3f}, {xKV_tr.max():.3f}]  "
      f"KVD [{xKVD_tr.min():.3f}, {xKVD_tr.max():.3f}]")
print(f"  Test  JSD ranges:  W [{xW_te.min():.3f}, {xW_te.max():.3f}]  "
      f"KV [{xKV_te.min():.3f}, {xKV_te.max():.3f}]  "
      f"KVD [{xKVD_te.min():.3f}, {xKVD_te.max():.3f}]")

# Overlap check
keys_tr = set(zip(np.round(mat_qs[0,:n_qs],6), np.round(mat_qs[1,:n_qs],6), np.round(mat_qs[4,:n_qs],6)))
keys_te = set(zip(np.round(mat_aw[0,:n_aw],6), np.round(mat_aw[1,:n_aw],6), np.round(mat_aw[4,:n_aw],6)))
n_overlap = len(keys_tr & keys_te)
print(f"  Architecture overlap (train ∩ test by (wbits,kvbits,kvdim)): {n_overlap}")

# ─── Fit surrogates ───────────────────────────────────────────────────────────
print("\nFitting surrogates...")
results = {}

# Linear additive
Phi_tr_M1, Phi_te_M1 = features_M1(X_tr), features_M1(X_te)
yp_tr_M1, yp_te_M1, coef_M1 = fit_ols(Phi_tr_M1, y_tr, Phi_te_M1)

# Full quadratic
Phi_tr_M10, Phi_te_M10 = features_M10(X_tr), features_M10(X_te)
yp_tr_M10, yp_te_M10, coef_M10 = fit_ols(Phi_tr_M10, y_tr, Phi_te_M10)

# RBF cubic / tps
lb_jsd = np.minimum(X_tr.min(0), X_te.min(0))
ub_jsd = np.maximum(X_tr.max(0), X_te.max(0))
m_rbf_c = PySOTRBF(kernel='cubic', tail='linear', lb=lb_jsd, ub=ub_jsd)
m_rbf_c.fit(X_tr, y_tr); yp_tr_rc = m_rbf_c.predict(X_tr).ravel(); yp_te_rc = m_rbf_c.predict(X_te).ravel()
m_rbf_t = PySOTRBF(kernel='tps', tail='linear', lb=lb_jsd, ub=ub_jsd)
m_rbf_t.fit(X_tr, y_tr); yp_tr_rt = m_rbf_t.predict(X_tr).ravel(); yp_te_rt = m_rbf_t.predict(X_te).ravel()

# ARD-GP — kernel sweep
print("\nARD-GP kernel sweep (n_restarts=50)...")
kernel_variants = [
    ('rbf',       True ),  # original
    ('rbf',       False),  # no noise (interpolation mode)
    ('matern52',  True ),
    ('matern52',  False),
    ('matern32',  True ),
    ('matern32',  False),
    ('rq',        True ),
    ('rq',        False),
]
gp_sweep = {}  # name -> dict
print(f"  {'kernel':22s}  {'LMML(train)':>13s}  {'R²_train':>10s}  {'R²_test':>10s}  {'RMSE':>10s}  {'eps_∞':>10s}")
print("  " + "-"*86)
for kn, wn in kernel_variants:
    name = f"ARD-{kn}{' +noise' if wn else ' nonoise'}"
    yp_tr_v, yp_te_v, gp_v = fit_ard_gp_variant(X_tr, y_tr, X_te, kn, wn, n_restarts=50)
    lmml = float(gp_v.log_marginal_likelihood_value_)
    r2_tr = r2(y_tr, yp_tr_v); r2_te = r2(y_te, yp_te_v); rm = rmse(y_te, yp_te_v)
    eps_inf = float(np.max(np.abs(y_te - yp_te_v)))
    ls_v = get_ls_anisotropic(gp_v, kn, wn, X_tr.shape[1])
    sigma_n = float(gp_v.kernel_.k2.noise_level) if wn else 0.0
    gp_sweep[name] = dict(kernel=kn, with_noise=wn, lmml=lmml, gp=gp_v,
                          yp_tr=yp_tr_v, yp_te=yp_te_v,
                          r2_train=r2_tr, r2_test=r2_te, rmse=rm, eps_inf=eps_inf,
                          length_scales=ls_v.tolist() if ls_v is not None else None,
                          sigma_n=sigma_n)
    print(f"  {name:22s}  {lmml:13.3f}  {r2_tr:10.4f}  {r2_te:10.4f}  {rm:10.5f}  {eps_inf:10.5f}")

# Pick best kernel by training log-marginal-likelihood (NOT by test R² → avoids leakage)
best_name = max(gp_sweep.keys(), key=lambda n: gp_sweep[n]['lmml'])
best = gp_sweep[best_name]
gp = best['gp']
ls = np.array(best['length_scales']) if best['length_scales'] is not None else None
sigma_n2 = best['sigma_n']
yp_tr_gp = best['yp_tr']
yp_te_gp = best['yp_te']
print(f"\n  → Best ARD-GP by train LMML: {best_name}  "
      f"(LMML={best['lmml']:.3f}, R²_test={best['r2_test']:.4f})")
if ls is not None:
    print(f"  Length scales: l_W={ls[0]:.4f}, l_KV={ls[1]:.4f}, l_KVD={ls[2]:.4f}  σ_n={sigma_n2:.2e}")

# Predictions table
preds_te = {
    'M1 linear additive': yp_te_M1,
    'M10 full quadratic': yp_te_M10,
    'RBF cubic+linear':   yp_te_rc,
    'RBF tps+linear':     yp_te_rt,
    'ARD-GP (RBF +noise)': gp_sweep['ARD-rbf +noise']['yp_te'],
    f'ARD-GP (best={best_name})': yp_te_gp,
}
preds_tr = {
    'M1 linear additive': yp_tr_M1,
    'M10 full quadratic': yp_tr_M10,
    'RBF cubic+linear':   yp_tr_rc,
    'RBF tps+linear':     yp_tr_rt,
    'ARD-GP (RBF +noise)': gp_sweep['ARD-rbf +noise']['yp_tr'],
    f'ARD-GP (best={best_name})': yp_tr_gp,
}

print("\n" + "="*100)
print(f"{'Surrogate':30s}  {'R²_train':>10s}  {'R²_test':>10s}  {'RMSE_te':>10s}  {'eps_∞':>10s}  {'eps_2':>10s}")
print("-"*100)
metrics_table = {}
for name, yp_te in preds_te.items():
    r2_tr = r2(y_tr, preds_tr[name])
    r2_te = r2(y_te, yp_te)
    rm    = rmse(y_te, yp_te)
    res   = y_te - yp_te
    eps_inf = float(np.max(np.abs(res)))
    eps_2   = float(np.sqrt(np.mean(res**2)))
    metrics_table[name] = dict(r2_train=float(r2_tr), r2_test=float(r2_te),
                               rmse=rm, eps_inf=eps_inf, eps_2=eps_2)
    print(f"{name:30s}  {r2_tr:10.4f}  {r2_te:10.4f}  {rm:10.5f}  {eps_inf:10.5f}  {eps_2:10.5f}")

# ─── Sobol indices (ARD-GP) ───────────────────────────────────────────────────
print("\nSobol indices (Saltelli pick-freeze on ARD-GP, N=2048)...")
bounds_sobol = [[X_tr[:,i].min(), X_tr[:,i].max()] for i in range(3)]
problem = {'num_vars': 3, 'names': ['W','KV','KVD'], 'bounds': bounds_sobol}
pv = saltelli.sample(problem, 2048, calc_second_order=True)
yp_pv = gp.predict(pv)
Si = sobol_analyze.analyze(problem, yp_pv, calc_second_order=True, print_to_console=False)
sobol_data = {
    'S1':  {n: float(Si['S1'][i])      for i, n in enumerate(['W','KV','KVD'])},
    'ST':  {n: float(Si['ST'][i])      for i, n in enumerate(['W','KV','KVD'])},
    'S1_conf': {n: float(Si['S1_conf'][i]) for i, n in enumerate(['W','KV','KVD'])},
    'ST_conf': {n: float(Si['ST_conf'][i]) for i, n in enumerate(['W','KV','KVD'])},
    'S2':  {f'{a},{b}': float(Si['S2'][i, j]) for (i, j, a, b) in
            [(0,1,'W','KV'), (0,2,'W','KVD'), (1,2,'KV','KVD')]},
    'S2_conf': {f'{a},{b}': float(Si['S2_conf'][i, j]) for (i, j, a, b) in
            [(0,1,'W','KV'), (0,2,'W','KVD'), (1,2,'KV','KVD')]},
    'sum_S1': float(sum(Si['S1'])),
    'interaction_fraction': float(1 - sum(Si['S1'])),
}
print(f"  S1: W={Si['S1'][0]:+.4f}  KV={Si['S1'][1]:+.4f}  KVD={Si['S1'][2]:+.4f}  "
      f"(sum = {sum(Si['S1']):.4f})")
print(f"  ST: W={Si['ST'][0]:+.4f}  KV={Si['ST'][1]:+.4f}  KVD={Si['ST'][2]:+.4f}")
print(f"  S2 W,KV  = {Si['S2'][0,1]:+.4f}")
print(f"  S2 W,KVD = {Si['S2'][0,2]:+.4f}")
print(f"  S2 KV,KVD= {Si['S2'][1,2]:+.4f}")

# ─── Plots ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

# Fig 1: scatter true vs pred
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharex=True, sharey=True)
for ax, (name, yp) in zip(axes, preds_te.items()):
    ax.scatter(y_te, yp, s=14, alpha=0.6, color='C0', label=f'test (N={N_TE})')
    lo, hi = min(y_te.min(), yp.min()), max(y_te.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
    r2v = r2(y_te, yp); rmv = rmse(y_te, yp)
    ax.set_title(f'{name}\nR²={r2v:.4f}  RMSE={rmv:.4f}', fontsize=10)
    ax.set_xlabel('y_actual'); ax.grid(alpha=0.3)
axes[0].set_ylabel('y_pred')
plt.suptitle(f'Phase 1 — Trained on 50 quantile samples → tested on 200 AWQ samples', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/v4_fig1_phase1_scatter.png', dpi=140, bbox_inches='tight')
plt.close()

# Fig 2: residual (ARD-GP) vs each input dim
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
res_gp = y_te - yp_te_gp
for ax, (i, name) in zip(axes, enumerate(['JSD_W', 'JSD_KV', 'JSD_KVD'])):
    ax.scatter(X_te[:, i], res_gp, s=14, alpha=0.6)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel(name); ax.set_ylabel('y_actual − y_pred (ARD-GP)')
    ax.grid(alpha=0.3); ax.set_title(f'{name} (mean={res_gp.mean():.4f})')
plt.suptitle('Phase 1 — ARD-GP residual structure on 200 AWQ test', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/v4_fig2_phase1_residual.png', dpi=140, bbox_inches='tight')
plt.close()

# Fig 3: ARD-GP kernel sweep — train LMML vs test R² / RMSE
sweep_names = list(gp_sweep.keys())
lmml_vals  = [gp_sweep[n]['lmml']    for n in sweep_names]
r2_vals    = [gp_sweep[n]['r2_test'] for n in sweep_names]
rmse_vals  = [gp_sweep[n]['rmse']    for n in sweep_names]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
xpos = np.arange(len(sweep_names))
axes[0].bar(xpos, lmml_vals, color='C0')
axes[0].set_xticks(xpos); axes[0].set_xticklabels(sweep_names, rotation=25, ha='right')
axes[0].set_ylabel('train log-marginal-likelihood (higher = better)')
axes[0].grid(alpha=0.3)
best_idx = sweep_names.index(best_name)
axes[0].bar([best_idx], [lmml_vals[best_idx]], color='C3', label='best by LMML')
axes[0].legend()
axes[0].set_title('Phase 1 — train LMML across ARD-GP kernels')

ax2 = axes[1]
ax2_b = ax2.twinx()
ax2.plot(xpos, r2_vals, 'o-', color='C0', label='R² test')
ax2.set_ylabel('R² test', color='C0')
ax2_b.plot(xpos, rmse_vals, 's--', color='C3', label='RMSE test')
ax2_b.set_ylabel('RMSE test', color='C3')
ax2.set_xticks(xpos); ax2.set_xticklabels(sweep_names, rotation=25, ha='right')
ax2.grid(alpha=0.3)
ax2.set_title('Phase 1 — test R²/RMSE across ARD-GP kernels (selection by train LMML, not test)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/v4_fig3_phase1_ardgp_kernel_sweep.png', dpi=140, bbox_inches='tight')
plt.close()

# Fig 4: Sobol bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
names = ['W', 'KV', 'KVD']
S1_vals = [Si['S1'][i] for i in range(3)]
ST_vals = [Si['ST'][i] for i in range(3)]
S1_err  = [Si['S1_conf'][i] for i in range(3)]
ST_err  = [Si['ST_conf'][i] for i in range(3)]
axes[0].bar(np.arange(3) - 0.18, S1_vals, 0.34, yerr=S1_err, label='S1 (main)', color='C0')
axes[0].bar(np.arange(3) + 0.18, ST_vals, 0.34, yerr=ST_err, label='ST (total)', color='C3')
axes[0].set_xticks(np.arange(3)); axes[0].set_xticklabels(names)
axes[0].set_ylabel('Sobol index'); axes[0].grid(alpha=0.3); axes[0].legend()
axes[0].set_title(f'Σ S1 = {sum(S1_vals):.3f}  (interaction = {1-sum(S1_vals):.3f})')

S2_lbls = ['W,KV', 'W,KVD', 'KV,KVD']
S2_vals = [Si['S2'][0,1], Si['S2'][0,2], Si['S2'][1,2]]
S2_err  = [Si['S2_conf'][0,1], Si['S2_conf'][0,2], Si['S2_conf'][1,2]]
axes[1].bar(S2_lbls, S2_vals, yerr=S2_err, color='C2')
axes[1].set_title('Pairwise interaction S2'); axes[1].grid(alpha=0.3)
axes[1].axhline(0, color='k', lw=0.6)
plt.suptitle('Phase 1 — Sobol decomposition (ARD-GP)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/v4_fig4_phase1_sobol.png', dpi=140, bbox_inches='tight')
plt.close()

# ─── Save JSON results ────────────────────────────────────────────────────────
ardgp_sweep_json = {n: {k: v for k, v in d.items()
                        if k not in ('gp', 'yp_tr', 'yp_te')}
                    for n, d in gp_sweep.items()}
out_json = {
    'phase': 1,
    'description': 'Train on 50 quantile_sample, test on 200 AWQ',
    'n_train': N_TR, 'n_test': N_TE,
    'train_test_arch_overlap': n_overlap,
    'surrogates': metrics_table,
    'ard_gp_best': {
        'name': best_name,
        'length_scales': ({'W': float(ls[0]), 'KV': float(ls[1]), 'KVD': float(ls[2])}
                          if ls is not None else None),
        'sigma_n': sigma_n2,
        'lmml': best['lmml'],
    },
    'ard_gp_kernel_sweep': ardgp_sweep_json,
    'sobol_best_ardgp': sobol_data,
}
with open(f'{OUT}/phase1_results.json', 'w') as f:
    json.dump(out_json, f, indent=2)
print(f"\nSaved: {OUT}/phase1_results.json")
print(f"Saved figures: v4_fig1..v4_fig4 in {OUT}/figures/")
print("\nDone (Phase 1).")
