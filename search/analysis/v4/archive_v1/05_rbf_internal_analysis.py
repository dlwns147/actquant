"""05_rbf_internal_analysis.py — internal predictor analysis on RBF (the v4 winner).

Trained on the full 50 QS samples; analysed on 200 AWQ test samples.

Primary surrogate: RBF tps+linear (best Phase-1 R²/RMSE).
Comparators: RBF cubic+linear, ARD-GP (Matérn-3/2 + WhiteKernel).

Analyses (originally done with ARD-GP in v3 / v4 Phase 1):
  PART A — predictor internals (no test labels used):
    (a1) Sobol indices via Saltelli pick-freeze on the surrogate's predictions
         (S1, ST, S2). The surrogate is queried as a callable, so the analysis
         transfers from ARD-GP to RBF without modification.
    (a2) Active-subspace eigenvalues (gradient covariance via finite differences).
    (a3) Hoeffding additive decomposition  y_add(z) = f_0 + g_W(z_W) + g_KV(z_KV) + g_KVD(z_KVD)
         via Monte-Carlo on the surrogate (input bounds = QS-train range).
  PART B — empirical ε bounds on the 200 AWQ test set:
    (b1) ε_∞ = ‖y_actual − y_add‖_∞, ε_2 = ‖y_actual − y_add‖_2
    (b2) Variance ratio: Var(residual_add) / Var(y_actual)  ≈  1 − ΣS_i

Outputs:
  • figures/v4_fig11_internal_sobol.png       — S1/ST/S2 across the 3 surrogates
  • figures/v4_fig12_internal_active_sub.png  — active-subspace eigenvalue spectra
  • figures/v4_fig13_internal_hoeffding.png   — y_actual vs y_add scatter per surrogate
  • internal_analysis_results.json
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
                                                WhiteKernel, Matern)
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
def r2(y_t, y_p):
    ss_r = np.sum((y_t - y_p)**2); ss_t = np.sum((y_t - y_t.mean())**2)
    return 1 - ss_r / max(ss_t, 1e-30)
def rmse(y_t, y_p): return float(np.sqrt(np.mean((y_t - y_p)**2)))

# ─── Paths & data ────────────────────────────────────────────────────────────
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
print("PHASE 1c — RBF-based internal analysis (Sobol / active subspace / Hoeffding additive)")
print("="*100)

pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)

mat_qs = load_csv(QS_3WAY); n_qs = mat_qs.shape[1]
y_tr_raw = mat_qs[12, :n_qs]; v_tr = ~np.isnan(y_tr_raw)
xW_tr  = match_metric(mat_qs[0,:n_qs], pf_W   )[v_tr]
xKV_tr = match_metric(mat_qs[1,:n_qs], pf_KV  )[v_tr]
xKVD_tr= match_metric(mat_qs[4,:n_qs], pf_KVDIM)[v_tr]
y_tr   = y_tr_raw[v_tr]
X_tr   = np.column_stack([xW_tr, xKV_tr, xKVD_tr])

mat_aw = load_csv(AWQ_3WAY); n_aw = mat_aw.shape[1]
y_te_raw = mat_aw[12, :n_aw]; v_te = ~np.isnan(y_te_raw)
xW_te  = match_metric(mat_aw[0,:n_aw], pf_W   )[v_te]
xKV_te = match_metric(mat_aw[1,:n_aw], pf_KV  )[v_te]
xKVD_te= match_metric(mat_aw[4,:n_aw], pf_KVDIM)[v_te]
y_te   = y_te_raw[v_te]
X_te   = np.column_stack([xW_te, xKV_te, xKVD_te])

print(f"  Train (QS, full): N={len(y_tr)}, Test (AWQ): N={len(y_te)}")

# Bounds for the surrogate domain (Saltelli, MC) — use train-set range as input distribution
bounds_jsd = [[X_tr[:,i].min(), X_tr[:,i].max()] for i in range(3)]
range_jsd  = np.array([X_tr[:,i].max() - X_tr[:,i].min() for i in range(3)])

# ─── Fit surrogates ───────────────────────────────────────────────────────────
print("\nFitting surrogates on full 50 QS samples...")
lb = np.minimum(X_tr.min(0), X_te.min(0))
ub = np.maximum(X_tr.max(0), X_te.max(0))
m_tps = PySOTRBF(kernel='tps', tail='linear', lb=lb, ub=ub); m_tps.fit(X_tr, y_tr)
m_cub = PySOTRBF(kernel='cubic', tail='linear', lb=lb, ub=ub); m_cub.fit(X_tr, y_tr)
kern_gp = (C(1.0,(1e-4,1e2)) *
           Matern(length_scale=[1.0]*3, length_scale_bounds=(1e-4,1e4), nu=1.5) +
           WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9,1e-2)))
gp = GaussianProcessRegressor(kernel=kern_gp, normalize_y=True,
                              n_restarts_optimizer=20, alpha=1e-8)
gp.fit(X_tr, y_tr)

# Surrogate callables (vectorised)
surrogates = {
    'RBF tps+linear (primary)': lambda Xq: m_tps.predict(Xq).ravel(),
    'RBF cubic+linear':         lambda Xq: m_cub.predict(Xq).ravel(),
    'ARD-GP (Matern-3/2)':      lambda Xq: gp.predict(Xq),
}
print(f"  Test R²: tps={r2(y_te, surrogates['RBF tps+linear (primary)'](X_te)):.4f}  "
      f"cubic={r2(y_te, surrogates['RBF cubic+linear'](X_te)):.4f}  "
      f"ARD-Matern32={r2(y_te, surrogates['ARD-GP (Matern-3/2)'](X_te)):.4f}")

# ─── PART A1 — Sobol via Saltelli pick-freeze ────────────────────────────────
print("\n=== PART A — predictor internals ===")
print("\n(a1) Sobol indices (Saltelli pick-freeze, N_base=2048)...")
problem = {'num_vars': 3, 'names': ['W','KV','KVD'], 'bounds': bounds_jsd}
pv = saltelli.sample(problem, 2048, calc_second_order=True)
sobol_results = {}
for name, fn in surrogates.items():
    yp = fn(pv)
    Si = sobol_analyze.analyze(problem, yp, calc_second_order=True, print_to_console=False)
    sobol_results[name] = {
        'S1':  {n: float(Si['S1'][i])  for i,n in enumerate(['W','KV','KVD'])},
        'ST':  {n: float(Si['ST'][i])  for i,n in enumerate(['W','KV','KVD'])},
        'S1_conf': {n: float(Si['S1_conf'][i]) for i,n in enumerate(['W','KV','KVD'])},
        'ST_conf': {n: float(Si['ST_conf'][i]) for i,n in enumerate(['W','KV','KVD'])},
        'S2': {f'{a},{b}': float(Si['S2'][i,j])
               for (i,j,a,b) in [(0,1,'W','KV'),(0,2,'W','KVD'),(1,2,'KV','KVD')]},
        'S2_conf': {f'{a},{b}': float(Si['S2_conf'][i,j])
                    for (i,j,a,b) in [(0,1,'W','KV'),(0,2,'W','KVD'),(1,2,'KV','KVD')]},
        'sum_S1': float(sum(Si['S1'])),
        'interaction_fraction': float(1 - sum(Si['S1'])),
    }
    print(f"\n  {name}:")
    print(f"    S1: W={Si['S1'][0]:+.4f}  KV={Si['S1'][1]:+.4f}  KVD={Si['S1'][2]:+.4f}  "
          f"(Σ={sum(Si['S1']):.4f}, interaction={1-sum(Si['S1']):.4f})")
    print(f"    ST: W={Si['ST'][0]:+.4f}  KV={Si['ST'][1]:+.4f}  KVD={Si['ST'][2]:+.4f}")
    print(f"    S2 W,KV={Si['S2'][0,1]:+.4f}  W,KVD={Si['S2'][0,2]:+.4f}  KV,KVD={Si['S2'][1,2]:+.4f}")

# ─── PART A2 — Active subspace ───────────────────────────────────────────────
print("\n(a2) Active subspace (gradient covariance via finite differences, n_mc=2000)...")
def active_subspace(fn, bounds, n_mc=2000, seed=0):
    rng = np.random.RandomState(seed); d = len(bounds)
    X_mc = np.array([rng.uniform(b[0], b[1], n_mc) for b in bounds]).T
    eps = np.array([(b[1]-b[0]) * 1e-4 for b in bounds])
    grads = np.zeros((n_mc, d))
    for i in range(d):
        Xp = X_mc.copy(); Xp[:,i] += eps[i]
        Xm = X_mc.copy(); Xm[:,i] -= eps[i]
        grads[:,i] = (fn(Xp) - fn(Xm)) / (2 * eps[i])
    Cmat = grads.T @ grads / n_mc
    eigvals, eigvecs = np.linalg.eigh(Cmat)
    order = np.argsort(-eigvals)
    return eigvals[order], eigvecs[:, order]

active_results = {}
for name, fn in surrogates.items():
    ev, evec = active_subspace(fn, bounds_jsd)
    active_results[name] = {
        'eigvals': ev.tolist(),
        'eigvals_relative': (ev / max(ev[0], 1e-30)).tolist(),
        'first_eigvec': evec[:, 0].tolist(),
    }
    print(f"  {name}: λ rel = "
          f"[1.000, {ev[1]/ev[0]:.4f}, {ev[2]/ev[0]:.4f}]   "
          f"first eigvec [W,KV,KVD] = [{evec[0,0]:+.3f}, {evec[1,0]:+.3f}, {evec[2,0]:+.3f}]")

# ─── PART A3 — Hoeffding additive decomposition ──────────────────────────────
print("\n(a3) Hoeffding additive: y_add = f_0 + g_W + g_KV + g_KVD via MC+KDE smoothing...")
def hoeffding_components(fn, bounds, n_mc=5000, seed=0, h_factor=0.15):
    rng = np.random.RandomState(seed)
    X_mc = np.array([rng.uniform(b[0], b[1], n_mc) for b in bounds]).T
    y_mc = fn(X_mc)
    f0 = float(y_mc.mean())
    sigma = X_mc.std(0); h = sigma * h_factor
    return X_mc, y_mc, f0, h

def conditional_mean(X_mc, y_mc, h, x_query, dim):
    out = np.zeros(len(x_query))
    for k, xq in enumerate(x_query):
        w = np.exp(-0.5 * ((X_mc[:, dim] - xq) / h[dim])**2)
        out[k] = (w * y_mc).sum() / max(w.sum(), 1e-30)
    return out

hoeffding_results = {}
y_add_te_by_model = {}
for name, fn in surrogates.items():
    X_mc, y_mc, f0, h = hoeffding_components(fn, bounds_jsd)
    g_W   = conditional_mean(X_mc, y_mc, h, X_te[:, 0], 0) - f0
    g_KV  = conditional_mean(X_mc, y_mc, h, X_te[:, 1], 1) - f0
    g_KVD = conditional_mean(X_mc, y_mc, h, X_te[:, 2], 2) - f0
    y_add = f0 + g_W + g_KV + g_KVD
    y_full_pred = fn(X_te)
    y_add_te_by_model[name] = (y_full_pred, y_add)

    # ε bounds (against actual measured y)
    res_add  = y_te - y_add
    res_full = y_te - y_full_pred
    eps_inf_add = float(np.max(np.abs(res_add))); eps_2_add = float(np.sqrt(np.mean(res_add**2)))
    eps_inf_full= float(np.max(np.abs(res_full))); eps_2_full= float(np.sqrt(np.mean(res_full**2)))

    # Variance check
    var_y    = float(np.var(y_te))
    var_res  = float(np.var(res_add))
    sumS1    = sobol_results[name]['sum_S1']
    var_ratio= var_res / max(var_y, 1e-30)
    expected = 1 - sumS1

    hoeffding_results[name] = {
        'f0': f0,
        'eps_inf_full': eps_inf_full, 'eps_2_full': eps_2_full,
        'eps_inf_add':  eps_inf_add,  'eps_2_add':  eps_2_add,
        'var_residual_over_var_y': var_ratio,
        'one_minus_sumS1': expected,
    }
    print(f"\n  {name}:")
    print(f"    full pred: ε_∞={eps_inf_full:.5f}  ε_2={eps_2_full:.5f}  R²={r2(y_te,y_full_pred):.4f}")
    print(f"    additive : ε_∞={eps_inf_add:.5f}  ε_2={eps_2_add:.5f}  R²={r2(y_te,y_add):.4f}")
    print(f"    Var(res_add)/Var(y_te) = {var_ratio:.4f}   (1 − ΣS1 = {expected:.4f}, match={abs(var_ratio-expected)<0.05})")

# ─── Plots ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
SURR = list(surrogates.keys())

# Fig 11 — Sobol comparison across surrogates
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(3); w = 0.26
for i, name in enumerate(SURR):
    sr = sobol_results[name]
    s1 = [sr['S1']['W'], sr['S1']['KV'], sr['S1']['KVD']]
    e1 = [sr['S1_conf']['W'], sr['S1_conf']['KV'], sr['S1_conf']['KVD']]
    axes[0].bar(x + (i-1)*w, s1, w, yerr=e1, label=name)
axes[0].set_xticks(x); axes[0].set_xticklabels(['W','KV','KVD'])
axes[0].set_ylabel('S1 (main effect)'); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
axes[0].set_title('S1 across surrogates')

s2_lbl = ['W,KV','W,KVD','KV,KVD']
for i, name in enumerate(SURR):
    sr = sobol_results[name]
    s2 = [sr['S2']['W,KV'], sr['S2']['W,KVD'], sr['S2']['KV,KVD']]
    e2 = [sr['S2_conf']['W,KV'], sr['S2_conf']['W,KVD'], sr['S2_conf']['KV,KVD']]
    axes[1].bar(np.arange(3) + (i-1)*w, s2, w, yerr=e2, label=name)
axes[1].set_xticks(np.arange(3)); axes[1].set_xticklabels(s2_lbl)
axes[1].set_ylabel('S2 (pairwise interaction)'); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
axes[1].axhline(0, color='k', lw=0.6)
axes[1].set_title('S2 across surrogates')
plt.suptitle('Phase 1c — Sobol decomposition (RBF primary)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/v4_fig11_internal_sobol.png', dpi=140, bbox_inches='tight')
plt.close()

# Fig 12 — active subspace eigenvalues
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, name in enumerate(SURR):
    rel = active_results[name]['eigvals_relative']
    ax.plot([1, 2, 3], rel, 'o-', label=name)
ax.set_yscale('log')
ax.set_xticks([1, 2, 3])
ax.set_xlabel('eigenvalue index'); ax.set_ylabel('λ / λ_1 (relative)')
ax.set_title('Phase 1c — active-subspace eigenvalue spectra')
ax.grid(alpha=0.3, which='both'); ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/figures/v4_fig12_internal_active_sub.png', dpi=140, bbox_inches='tight')
plt.close()

# Fig 13 — Hoeffding additive scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
for ax, name in zip(axes, SURR):
    y_full_pred, y_add = y_add_te_by_model[name]
    ax.scatter(y_te, y_full_pred, s=14, alpha=0.5, color='C0', label='full surrogate')
    ax.scatter(y_te, y_add,       s=14, alpha=0.5, color='C3', label='additive (Hoeffding)')
    lo, hi = min(y_te.min(), y_add.min(), y_full_pred.min()), max(y_te.max(), y_add.max(), y_full_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('y_actual'); ax.set_ylabel('y_pred')
    h = hoeffding_results[name]
    ax.set_title(f"{name}\nfull ε_∞={h['eps_inf_full']:.4f}   add ε_∞={h['eps_inf_add']:.4f}",
                 fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.suptitle('Phase 1c — full vs Hoeffding-additive surrogate on the 200 AWQ test', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/v4_fig13_internal_hoeffding.png', dpi=140, bbox_inches='tight')
plt.close()

# ─── JSON ─────────────────────────────────────────────────────────────────────
out_json = {'phase': '1c',
            'description': 'RBF-primary internal predictor analysis (Sobol / active subspace / Hoeffding)',
            'sobol': sobol_results,
            'active_subspace': active_results,
            'hoeffding': hoeffding_results}
with open(f'{OUT}/internal_analysis_results.json', 'w') as f:
    json.dump(out_json, f, indent=2)
print(f"\nSaved: {OUT}/internal_analysis_results.json")
print("Saved figures: v4_fig11..v4_fig13")
print("Done (Phase 1c).")
