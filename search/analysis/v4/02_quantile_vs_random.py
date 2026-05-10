"""02_quantile_vs_random.py — Phase 2.

Quantile-sampling efficacy: split 200 AWQ samples into a 50-train / 150-test partition for
each of N_SEED=10 random seeds. Compare two training pools, holding the 150 test fixed
within each seed:

  (a) train_random : 50 randomly sampled AWQ points (changes per seed)
  (b) train_qs     : the 50 fixed quantile_sample points (constant across seeds)

For each (surrogate, seed): paired R² / RMSE / eps_inf on the same 150 holdout.

Outputs:
  • figures/v4_fig5_phase2_R2_bars.png      — mean ± std R² per surrogate, random vs QS
  • figures/v4_fig6_phase2_RMSE_bars.png    — mean ± std RMSE per surrogate
  • figures/v4_fig7_phase2_paired_delta.png — paired Δ = (QS) − (random) per surrogate
  • phase2_results.json                     — full per-seed table + summaries
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
    return Phi_te @ coef

def fit_ard_gp_rbf(X_tr, y_tr, n_dim=3, n_restarts=10):
    kernel = (C(1.0, (1e-4, 1e2)) *
              SKRBF(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-4, 1e4)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X_tr, y_tr)
    return gp

def fit_ard_gp_matern32(X_tr, y_tr, n_dim=3, n_restarts=10):
    kernel = (C(1.0, (1e-4, 1e2)) *
              Matern(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-4, 1e4), nu=1.5) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X_tr, y_tr)
    return gp

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
print("PHASE 2 — Quantile-sampling efficacy via 10-seed paired comparison")
print("="*100)

# Load PFs
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)

# Load AWQ 200
mat_aw = load_csv(AWQ_3WAY); n_aw = mat_aw.shape[1]
y_aw_raw = mat_aw[12, :n_aw]; v_aw = ~np.isnan(y_aw_raw)
xW_aw  = match_metric(mat_aw[0, :n_aw], pf_W   )[v_aw]
xKV_aw = match_metric(mat_aw[1, :n_aw], pf_KV  )[v_aw]
xKVD_aw= match_metric(mat_aw[4, :n_aw], pf_KVDIM)[v_aw]
y_aw   = y_aw_raw[v_aw]
X_aw   = np.column_stack([xW_aw, xKV_aw, xKVD_aw])
N_AWQ  = len(y_aw)
keys_aw = list(zip(np.round(mat_aw[0,:n_aw][v_aw],6),
                   np.round(mat_aw[1,:n_aw][v_aw],6),
                   np.round(mat_aw[4,:n_aw][v_aw],6)))

# Load QS 50
mat_qs = load_csv(QS_3WAY); n_qs = mat_qs.shape[1]
y_qs_raw = mat_qs[12, :n_qs]; v_qs = ~np.isnan(y_qs_raw)
xW_qs  = match_metric(mat_qs[0, :n_qs], pf_W   )[v_qs]
xKV_qs = match_metric(mat_qs[1, :n_qs], pf_KV  )[v_qs]
xKVD_qs= match_metric(mat_qs[4, :n_qs], pf_KVDIM)[v_qs]
y_qs   = y_qs_raw[v_qs]
X_qs   = np.column_stack([xW_qs, xKV_qs, xKVD_qs])
N_QS   = len(y_qs)
keys_qs = list(zip(np.round(mat_qs[0,:n_qs][v_qs],6),
                   np.round(mat_qs[1,:n_qs][v_qs],6),
                   np.round(mat_qs[4,:n_qs][v_qs],6)))

print(f"  AWQ pool: N={N_AWQ}  |  QS train pool: N={N_QS}")
keys_qs_set = set(keys_qs)
n_overlap = sum(1 for k in keys_aw if k in keys_qs_set)
print(f"  Architecture overlap (AWQ ∩ QS by (wbits, kvbits, kvdim)): {n_overlap}")
if n_overlap == 0:
    print("  → No dedup needed. Full 150-holdout used per seed.")
else:
    print(f"  → Will exclude {n_overlap} overlapping AWQ points from each holdout.")

# Pre-compute valid AWQ indices (excluding any architecture seen in QS)
overlap_mask = np.array([k in keys_qs_set for k in keys_aw])
clean_aw_idx = np.where(~overlap_mask)[0]  # AWQ indices safe for QS-evaluation

# ─── Surrogate runner ─────────────────────────────────────────────────────────
def predict_all(X_tr, y_tr, X_te, lb, ub):
    """Returns dict surrogate_name -> y_pred on X_te."""
    out = {}
    out['M1 linear additive'] = fit_ols(features_M1(X_tr), y_tr, features_M1(X_te))
    out['M10 full quadratic'] = fit_ols(features_M10(X_tr), y_tr, features_M10(X_te))
    m_c = PySOTRBF(kernel='cubic', tail='linear', lb=lb, ub=ub); m_c.fit(X_tr, y_tr)
    out['RBF cubic+linear'] = m_c.predict(X_te).ravel()
    m_t = PySOTRBF(kernel='tps', tail='linear', lb=lb, ub=ub); m_t.fit(X_tr, y_tr)
    out['RBF tps+linear'] = m_t.predict(X_te).ravel()
    gp = fit_ard_gp(X_tr, y_tr)
    out['ARD-GP'] = gp.predict(X_te)
    return out

SURROGATES = ['M1 linear additive', 'M10 full quadratic', 'RBF cubic+linear',
              'RBF tps+linear', 'ARD-GP (RBF)', 'ARD-GP (Matern-3/2)']

# Pre-fit QS surrogates ONCE (training data is fixed across seeds)
print("\nFitting QS-trained surrogates once (training data is constant across seeds)...")
lb_combined = np.minimum(X_aw.min(0), X_qs.min(0))
ub_combined = np.maximum(X_aw.max(0), X_qs.max(0))
# We will predict on different X_te per seed. Pre-fit by keeping the fitted models in dict.
def fit_models_only(X_tr, y_tr, lb, ub):
    out = {}
    Phi1_tr = features_M1(X_tr); coef1, *_ = np.linalg.lstsq(Phi1_tr, y_tr, rcond=None)
    out['M1 linear additive'] = ('lin', features_M1, coef1)
    Phi10_tr = features_M10(X_tr); coef10, *_ = np.linalg.lstsq(Phi10_tr, y_tr, rcond=None)
    out['M10 full quadratic'] = ('lin', features_M10, coef10)
    m_c = PySOTRBF(kernel='cubic', tail='linear', lb=lb, ub=ub); m_c.fit(X_tr, y_tr)
    out['RBF cubic+linear'] = ('rbf', m_c)
    m_t = PySOTRBF(kernel='tps', tail='linear', lb=lb, ub=ub); m_t.fit(X_tr, y_tr)
    out['RBF tps+linear'] = ('rbf', m_t)
    gp_rbf = fit_ard_gp_rbf(X_tr, y_tr)
    out['ARD-GP (RBF)'] = ('gp', gp_rbf)
    gp_m32 = fit_ard_gp_matern32(X_tr, y_tr)
    out['ARD-GP (Matern-3/2)'] = ('gp', gp_m32)
    return out

def predict_with(models, X_te):
    out = {}
    for name, item in models.items():
        kind = item[0]
        if kind == 'lin':
            _, feat_fn, coef = item
            out[name] = feat_fn(X_te) @ coef
        elif kind == 'rbf':
            out[name] = item[1].predict(X_te).ravel()
        elif kind == 'gp':
            out[name] = item[1].predict(X_te)
    return out

models_qs = fit_models_only(X_qs, y_qs, lb_combined, ub_combined)
print("  QS-trained surrogates fitted.")

# ─── 10-seed loop ─────────────────────────────────────────────────────────────
N_SEED = 10
N_TR   = 50
results = {s: {'random': {}, 'quantile': {}} for s in SURROGATES}
all_seeds_data = []

print(f"\nRunning {N_SEED} seeds (50 train / 150 test, paired holdout)...")
for seed in range(N_SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N_AWQ)
    train_idx = perm[:N_TR]
    test_idx_full = perm[N_TR:]

    # For QS-trained: drop overlapping points from holdout. (Safe for both: random's holdout
    # is also the same dropped set, so paired comparison still consistent.)
    if n_overlap > 0:
        keep_mask = np.array([not overlap_mask[i] for i in test_idx_full])
        test_idx = test_idx_full[keep_mask]
    else:
        test_idx = test_idx_full
    # Random's training set may itself overlap with QS — that's NOT a leak for random
    # since it's training on AWQ; but for paired comparison we keep train_idx unchanged
    # and only restrict the test set so the holdout is identical for both. (Currently 0
    # overlap, so no-op.)

    X_tr_r = X_aw[train_idx]; y_tr_r = y_aw[train_idx]
    X_te   = X_aw[test_idx];  y_te   = y_aw[test_idx]
    n_te   = len(y_te)

    lb_seed = np.minimum(X_tr_r.min(0), X_te.min(0))
    ub_seed = np.maximum(X_tr_r.max(0), X_te.max(0))
    lb_seed = np.minimum(lb_seed, X_qs.min(0))  # ensure RBF fit-bounds cover QS-train too
    ub_seed = np.maximum(ub_seed, X_qs.max(0))

    # (a) random-trained
    models_r = fit_models_only(X_tr_r, y_tr_r, lb_seed, ub_seed)
    preds_r  = predict_with(models_r, X_te)

    # (b) QS-trained: just predict on X_te (fitted once already)
    preds_q  = predict_with(models_qs, X_te)

    seed_row = {'seed': seed, 'n_test': int(n_te)}
    for name in SURROGATES:
        r2_r  = r2(y_te, preds_r[name]); rm_r = rmse(y_te, preds_r[name])
        r2_q  = r2(y_te, preds_q[name]); rm_q = rmse(y_te, preds_q[name])
        eps_r = float(np.max(np.abs(y_te - preds_r[name])))
        eps_q = float(np.max(np.abs(y_te - preds_q[name])))
        results[name]['random'].setdefault('r2', []).append(r2_r)
        results[name]['random'].setdefault('rmse', []).append(rm_r)
        results[name]['random'].setdefault('eps_inf', []).append(eps_r)
        results[name]['quantile'].setdefault('r2', []).append(r2_q)
        results[name]['quantile'].setdefault('rmse', []).append(rm_q)
        results[name]['quantile'].setdefault('eps_inf', []).append(eps_q)
        seed_row[name] = {'random':   {'r2': float(r2_r), 'rmse': rm_r, 'eps_inf': eps_r},
                          'quantile': {'r2': float(r2_q), 'rmse': rm_q, 'eps_inf': eps_q}}
    all_seeds_data.append(seed_row)
    keyref = 'ARD-GP (Matern-3/2)'
    print(f"  seed={seed}  n_test={n_te}  "
          f"{keyref} R² random={results[keyref]['random']['r2'][-1]:.4f}  "
          f"QS={results[keyref]['quantile']['r2'][-1]:.4f}  "
          f"Δ={results[keyref]['quantile']['r2'][-1] - results[keyref]['random']['r2'][-1]:+.4f}")

# ─── Summarize ────────────────────────────────────────────────────────────────
print("\n" + "="*112)
print(f"{'Surrogate':24s}  {'random R² (μ±σ)':22s}  {'QS R² (μ±σ)':22s}  "
      f"{'Δ R² (QS−rand)':18s}  {'sign(Δ>0)':>10s}")
print("-"*112)

summary = {}
for name in SURROGATES:
    r_a = np.array(results[name]['random']['r2'])
    r_b = np.array(results[name]['quantile']['r2'])
    rm_a = np.array(results[name]['random']['rmse'])
    rm_b = np.array(results[name]['quantile']['rmse'])
    eps_a = np.array(results[name]['random']['eps_inf'])
    eps_b = np.array(results[name]['quantile']['eps_inf'])
    delta_r2 = r_b - r_a
    n_pos = int(np.sum(delta_r2 > 0))
    summary[name] = {
        'random_r2_mean':   float(r_a.mean()),    'random_r2_std':   float(r_a.std()),
        'quantile_r2_mean': float(r_b.mean()),    'quantile_r2_std': float(r_b.std()),
        'random_rmse_mean':  float(rm_a.mean()),  'random_rmse_std':  float(rm_a.std()),
        'quantile_rmse_mean':float(rm_b.mean()),  'quantile_rmse_std':float(rm_b.std()),
        'random_eps_inf_mean':  float(eps_a.mean()),
        'quantile_eps_inf_mean':float(eps_b.mean()),
        'delta_r2_mean': float(delta_r2.mean()), 'delta_r2_std': float(delta_r2.std()),
        'sign_qs_better_count': n_pos, 'sign_qs_better_frac': n_pos / N_SEED,
    }
    print(f"{name:24s}  "
          f"{r_a.mean():.4f} ± {r_a.std():.4f}     "
          f"{r_b.mean():.4f} ± {r_b.std():.4f}     "
          f"{delta_r2.mean():+.4f} ± {delta_r2.std():.4f}   "
          f"{n_pos:>2d}/{N_SEED}")

print("\n" + "="*112)
print(f"{'Surrogate':24s}  {'random RMSE (μ±σ)':24s}  {'QS RMSE (μ±σ)':24s}")
print("-"*112)
for name in SURROGATES:
    s = summary[name]
    print(f"{name:24s}  "
          f"{s['random_rmse_mean']:.5f} ± {s['random_rmse_std']:.5f}      "
          f"{s['quantile_rmse_mean']:.5f} ± {s['quantile_rmse_std']:.5f}")

# ─── Plots ────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")
x = np.arange(len(SURROGATES))
w = 0.36

# Fig 5: R² bars
fig, ax = plt.subplots(figsize=(11, 5))
mu_a = [summary[n]['random_r2_mean']   for n in SURROGATES]
mu_b = [summary[n]['quantile_r2_mean'] for n in SURROGATES]
sd_a = [summary[n]['random_r2_std']    for n in SURROGATES]
sd_b = [summary[n]['quantile_r2_std']  for n in SURROGATES]
ax.bar(x - w/2, mu_a, w, yerr=sd_a, label=f'random 50 (mean ± std, {N_SEED} seeds)', color='C0')
ax.bar(x + w/2, mu_b, w, yerr=sd_b, label=f'quantile 50 (paired holdout)',         color='C3')
ax.set_xticks(x); ax.set_xticklabels(SURROGATES, rotation=18, ha='right')
ax.set_ylabel('R² on 150 holdout'); ax.grid(alpha=0.3)
ax.set_title(f'Phase 2 — R² (50 train → 150 test, {N_SEED} seeds)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/figures/v4_fig5_phase2_R2_bars.png', dpi=140, bbox_inches='tight')
plt.close()

# Fig 6: RMSE bars
fig, ax = plt.subplots(figsize=(11, 5))
mu_a = [summary[n]['random_rmse_mean']   for n in SURROGATES]
mu_b = [summary[n]['quantile_rmse_mean'] for n in SURROGATES]
sd_a = [summary[n]['random_rmse_std']    for n in SURROGATES]
sd_b = [summary[n]['quantile_rmse_std']  for n in SURROGATES]
ax.bar(x - w/2, mu_a, w, yerr=sd_a, label=f'random 50',   color='C0')
ax.bar(x + w/2, mu_b, w, yerr=sd_b, label=f'quantile 50', color='C3')
ax.set_xticks(x); ax.set_xticklabels(SURROGATES, rotation=18, ha='right')
ax.set_ylabel('RMSE on 150 holdout'); ax.grid(alpha=0.3)
ax.set_title(f'Phase 2 — RMSE (50 train → 150 test, {N_SEED} seeds)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/figures/v4_fig6_phase2_RMSE_bars.png', dpi=140, bbox_inches='tight')
plt.close()

# Fig 7: Paired Δ R² scatter
fig, ax = plt.subplots(figsize=(11, 5))
for i, name in enumerate(SURROGATES):
    rs_a = np.array(results[name]['random']['r2'])
    rs_b = np.array(results[name]['quantile']['r2'])
    deltas = rs_b - rs_a
    ax.scatter([i] * len(deltas), deltas, s=30, alpha=0.6, color='C2')
    ax.scatter([i], [deltas.mean()], s=120, marker='_', color='k', label='mean' if i == 0 else None)
ax.axhline(0, color='r', lw=0.8, linestyle='--')
ax.set_xticks(np.arange(len(SURROGATES)))
ax.set_xticklabels(SURROGATES, rotation=18, ha='right')
ax.set_ylabel('Δ R² = (quantile) − (random)')
ax.set_title(f'Phase 2 — paired per-seed Δ R² ({N_SEED} seeds, 150-holdout per seed)')
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/figures/v4_fig7_phase2_paired_delta.png', dpi=140, bbox_inches='tight')
plt.close()

# ─── Save JSON ────────────────────────────────────────────────────────────────
out_json = {
    'phase': 2,
    'description': 'Quantile vs random training comparison (10 seeds, 50 train / 150 test)',
    'n_seeds': N_SEED, 'n_train': N_TR, 'n_pool_awq': N_AWQ, 'n_pool_qs': N_QS,
    'arch_overlap_qs_awq': int(n_overlap),
    'summary': summary,
    'per_seed': all_seeds_data,
}
with open(f'{OUT}/phase2_results.json', 'w') as f:
    json.dump(out_json, f, indent=2)
print(f"\nSaved: {OUT}/phase2_results.json")
print(f"Saved figures: v4_fig5..v4_fig7 in {OUT}/figures/")
print("\nDone (Phase 2).")
