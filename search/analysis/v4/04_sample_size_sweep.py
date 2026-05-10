"""04_sample_size_sweep.py — training-set-size sensitivity (Phase 1 protocol).

Train pool: 50 quantile-sample CSV columns. Sweep N_train ∈ {27, 30, 35, 40, 45, 50}:
  • N=27 — deterministic, the first 27 columns of the QS CSV (the 27-quantile-grid base).
  • N=50 — deterministic, all 50 columns (full QS pool, same as Phase 1).
  • N ∈ {30, 35, 40, 45} — first 27 columns + (N-27) random extras drawn from columns 27..49,
    repeated for N_SEED=50 seeds, summarised as median [10/90 percentile].

Test set: 200 AWQ samples (same as Phase 1, full hold-out, no overlap with QS).

Surrogates:
  • M1 (linear additive)
  • M10 (full quadratic)
  • RBF cubic+linear
  • RBF tps+linear
  • ARD-GP (Matérn-3/2 + WhiteKernel) — best by Phase-1 train LMML

Outputs:
  • figures/v4_fig8_lcurve_R2.png      — learning curve (R²_test vs N_train)
  • figures/v4_fig9_lcurve_RMSE.png    — learning curve (RMSE_test vs N_train)
  • figures/v4_fig10_lcurve_eps.png    — learning curve (ε_∞_test vs N_train)
  • sample_size_sweep_results.json     — per-(model, N_train) summary
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

def features_M1(X):
    n = len(X); o = np.ones(n)
    return np.column_stack([o, X[:,0], X[:,1], X[:,2]])
def features_M10(X):
    n = len(X); o = np.ones(n)
    w, kv, kvd = X[:,0], X[:,1], X[:,2]
    return np.column_stack([o, w, kv, kvd, w**2, kv**2, kvd**2, w*kv, w*kvd, kv*kvd])
def fit_ols_predict(Phi_tr, y_tr, Phi_te):
    coef, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
    return Phi_te @ coef

def fit_ard_matern32(X_tr, y_tr, X_te, n_restarts=20):
    n_dim = X_tr.shape[1]
    kernel = (C(1.0, (1e-4, 1e2)) *
              Matern(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-4, 1e4), nu=1.5) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X_tr, y_tr)
    return gp.predict(X_te)

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
print("PHASE 1b — Sample-size sweep on the 50-row QS train pool, tested on 200 AWQ")
print("="*100)

pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)

# Train pool (50 QS rows) — preserves CSV column order so columns 0..26 are the quantile-grid base
mat_qs = load_csv(QS_3WAY); n_qs = mat_qs.shape[1]
y_full = mat_qs[12, :n_qs]
xW_full  = match_metric(mat_qs[0, :n_qs], pf_W   )
xKV_full = match_metric(mat_qs[1, :n_qs], pf_KV  )
xKVD_full= match_metric(mat_qs[4, :n_qs], pf_KVDIM)
X_full = np.column_stack([xW_full, xKV_full, xKVD_full])
assert n_qs == 50, f"Expected 50 QS cols, got {n_qs}"

# Test (200 AWQ)
mat_aw = load_csv(AWQ_3WAY); n_aw = mat_aw.shape[1]
y_aw = mat_aw[12, :n_aw]; v_aw = ~np.isnan(y_aw)
xW_aw = match_metric(mat_aw[0, :n_aw], pf_W   )[v_aw]
xKV_aw= match_metric(mat_aw[1, :n_aw], pf_KV  )[v_aw]
xKVD_aw=match_metric(mat_aw[4, :n_aw], pf_KVDIM)[v_aw]
y_aw  = y_aw[v_aw]
X_aw  = np.column_stack([xW_aw, xKV_aw, xKVD_aw])

print(f"  QS train pool: {n_qs} columns (first 27 = quantile-grid base, 27..49 = random extras)")
print(f"  AWQ test set: {len(y_aw)} samples")

# Bounds for RBF (cover both sets)
lb = np.minimum(X_full.min(0), X_aw.min(0))
ub = np.maximum(X_full.max(0), X_aw.max(0))

# ─── Sweep ────────────────────────────────────────────────────────────────────
N_TRS  = [27, 30, 35, 40, 45, 50]
N_SEED = 50
SURROGATES = ['M1 linear additive', 'M10 full quadratic',
              'RBF cubic+linear', 'RBF tps+linear',
              'ARD-GP (Matern-3/2)']

base27 = np.arange(27)
extras_pool = np.arange(27, 50)  # 23 extras

storage = {}  # (model, N) -> list of (r2_test, rmse_test, eps_inf_test)

def run_predictors(X_tr, y_tr):
    out = {}
    out['M1 linear additive'] = lambda Xt: fit_ols_predict(features_M1(X_tr), y_tr, features_M1(Xt))
    out['M10 full quadratic'] = lambda Xt: fit_ols_predict(features_M10(X_tr), y_tr, features_M10(Xt))
    m_c = PySOTRBF(kernel='cubic', tail='linear', lb=lb, ub=ub); m_c.fit(X_tr, y_tr)
    out['RBF cubic+linear'] = lambda Xt: m_c.predict(Xt).ravel()
    m_t = PySOTRBF(kernel='tps', tail='linear', lb=lb, ub=ub); m_t.fit(X_tr, y_tr)
    out['RBF tps+linear'] = lambda Xt: m_t.predict(Xt).ravel()
    out['ARD-GP (Matern-3/2)'] = lambda Xt: fit_ard_matern32(X_tr, y_tr, Xt)
    return out

print(f"\nRunning sweep N ∈ {N_TRS} ...")
for n_tr in N_TRS:
    n_extra = n_tr - 27
    if n_extra in (0, 23):
        seeds = [None]
        seed_label = "deterministic"
    else:
        seeds = list(range(N_SEED))
        seed_label = f"N_SEED={N_SEED}"
    print(f"  N_train={n_tr}  ({seed_label})")
    for seed in seeds:
        if n_extra == 0:
            tr_idx = base27
        elif n_extra == 23:
            tr_idx = np.arange(50)
        else:
            rng = np.random.RandomState(seed)
            chosen = rng.choice(extras_pool, n_extra, replace=False)
            tr_idx = np.concatenate([base27, chosen])

        X_tr = X_full[tr_idx]; y_tr = y_full[tr_idx]
        preds = run_predictors(X_tr, y_tr)
        for name in SURROGATES:
            try:
                yp = preds[name](X_aw)
                r2v = r2(y_aw, yp); rmv = rmse(y_aw, yp)
                eps = float(np.max(np.abs(y_aw - yp)))
                storage.setdefault((name, n_tr), []).append((float(r2v), float(rmv), float(eps)))
            except Exception as e:
                print(f"    [skip] {name} N={n_tr} seed={seed}: {e}")

# ─── Summarise ────────────────────────────────────────────────────────────────
def stats(rs, idx):
    vals = np.array([r[idx] for r in rs])
    if vals.size == 1:
        return dict(mean=float(vals[0]), median=float(vals[0]), p10=float(vals[0]),
                    p90=float(vals[0]), std=0.0, n=1)
    return dict(mean=float(vals.mean()), median=float(np.median(vals)),
                p10=float(np.percentile(vals, 10)), p90=float(np.percentile(vals, 90)),
                std=float(vals.std()), n=int(vals.size))

print("\n" + "="*132)
print("R²_test  (median [10/90 pct] for stochastic N, exact value for N=27 and N=50)")
print("-"*132)
hdr = f"  {'Model':22s}"
for n_tr in N_TRS:
    mark = "*" if n_tr in (27, 50) else " "
    hdr += f"  {mark}N={n_tr:<3d}              "
print(hdr + "    (* = deterministic)")
for m in SURROGATES:
    line = f"  {m:22s}"
    for n_tr in N_TRS:
        rs = storage.get((m, n_tr), [])
        if not rs: line += "  N/A                  "; continue
        s = stats(rs, 0)
        if s['n'] == 1: line += f"  {s['mean']:.4f}              "
        else:           line += f"  {s['median']:.3f}[{s['p10']:.3f},{s['p90']:.3f}]"
    print(line)

print("\n" + "="*132)
print("RMSE_test")
print("-"*132)
print(hdr)
for m in SURROGATES:
    line = f"  {m:22s}"
    for n_tr in N_TRS:
        rs = storage.get((m, n_tr), [])
        if not rs: line += "  N/A                  "; continue
        s = stats(rs, 1)
        if s['n'] == 1: line += f"  {s['mean']:.5f}             "
        else:           line += f"  {s['median']:.4f}[{s['p10']:.4f},{s['p90']:.4f}]"
    print(line)

# ─── Plots ────────────────────────────────────────────────────────────────────
print("\nGenerating learning-curve figures...")
colors = {'M1 linear additive': 'C0', 'M10 full quadratic': 'C1',
          'RBF cubic+linear': 'C2', 'RBF tps+linear': 'C3',
          'ARD-GP (Matern-3/2)': 'C4'}

def plot_curve(metric_idx, ylabel, fname, log=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in SURROGATES:
        med, p10, p90 = [], [], []
        for n_tr in N_TRS:
            rs = storage.get((m, n_tr), [])
            s = stats(rs, metric_idx)
            med.append(s['median']); p10.append(s['p10']); p90.append(s['p90'])
        med, p10, p90 = np.array(med), np.array(p10), np.array(p90)
        ax.plot(N_TRS, med, 'o-', color=colors[m], label=m, lw=1.5)
        ax.fill_between(N_TRS, p10, p90, alpha=0.15, color=colors[m])
    ax.set_xlabel('N_train')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if log: ax.set_yscale('log')
    ax.legend()
    ax.set_title(f'Phase 1b — learning curve (50 seeds for N∈{{30,35,40,45}}, deterministic at 27 & 50)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/figures/{fname}', dpi=140, bbox_inches='tight')
    plt.close()

plot_curve(0, 'R² on 200 AWQ test', 'v4_fig8_lcurve_R2.png',  log=False)
plot_curve(1, 'RMSE on 200 AWQ test', 'v4_fig9_lcurve_RMSE.png', log=True)
plot_curve(2, 'eps_∞ on 200 AWQ test', 'v4_fig10_lcurve_eps.png', log=True)

# ─── JSON ─────────────────────────────────────────────────────────────────────
out_json = {'n_train_grid': N_TRS, 'n_seed': N_SEED, 'surrogates': SURROGATES,
            'metrics': {f"{m}|{n}": {
                'r2': stats(storage.get((m,n),[]),0),
                'rmse': stats(storage.get((m,n),[]),1),
                'eps_inf': stats(storage.get((m,n),[]),2),
            } for m in SURROGATES for n in N_TRS}}
with open(f'{OUT}/sample_size_sweep_results.json', 'w') as f:
    json.dump(out_json, f, indent=2)
print(f"\nSaved: {OUT}/sample_size_sweep_results.json")
print("Done (Phase 1b).")
