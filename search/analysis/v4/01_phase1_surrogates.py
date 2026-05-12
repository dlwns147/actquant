"""01_phase1_surrogates.py — Phase 1 + Phase 1b (260510 data).

Inputs come straight from the CSVs (no .stats files):
    X       = rows 14..16 (per-method z_W, z_KV, z_KVD)
    y       = row 12       (measured wikitext2 metric, JSD)
    y_add0  = row 13       (naive additive baseline = z_W + z_KV + z_KVD)

Phase 1 — Train on 50 QS (27 quantile + 23 random extras), test on 200 RS.
    Surrogates: M1, M_full_quad, RBF cubic, RBF tps, ARD-GP (Matérn-3/2 +noise).
    Also reports the no-learning baseline (y_add0 = row 13) for reference.
    Ran for Llama-3.1-8B-Instruct AND Qwen2.5-7B-Instruct.

Phase 1b — Sample-size sweep:
    N ∈ {27, 30, 35, 40, 45, 50}; first 27 = deterministic quantile grid,
    extras drawn from cols 27..49 with 50 seeds for the stochastic N values.

Outputs:
    figures/v4_fig1_scatter.png         — true-vs-pred (Llama)
    figures/v4_fig1q_scatter.png        — true-vs-pred (Qwen)
    figures/v4_fig1b_lcurve_R2.png      — learning curve, R² (Llama)
    figures/v4_fig1b_lcurve_RMSE.png    — learning curve, RMSE (Llama)
    phase1_results.json
    phase1b_sweep_results.json
"""
import os, sys, json, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import (load_csv, extract_xy, all_surrogate_fits,
                     get_ard_lengthscales, r2, rmse, eps_inf, PATHS)

OUT = '/NAS/SJ/actquant/search/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

# ─── Helper: full surrogate evaluation given (X_tr,y_tr) → (X_te,y_te) ────────
def eval_all(X_tr, y_tr, X_te, y_te, n_restarts=30):
    fits = all_surrogate_fits(X_tr, y_tr, X_te, ard_kernel='matern32', n_restarts=n_restarts)
    metrics = {}
    for n, d in fits.items():
        r2_tr = r2(y_tr, d['yp_tr']); r2_te = r2(y_te, d['yp_te'])
        rm = rmse(y_te, d['yp_te']);  ei = eps_inf(y_te, d['yp_te'])
        metrics[n] = dict(r2_train=r2_tr, r2_test=r2_te, rmse=rm, eps_inf=ei,
                          yp_tr=d['yp_tr'], yp_te=d['yp_te'])
        if 'gp' in d:
            ls = get_ard_lengthscales(d['gp'], with_noise=True, d=X_tr.shape[1])
            sigma_n = float(d['gp'].kernel_.k2.noise_level)
            metrics[n]['length_scales'] = ls.tolist() if ls is not None else None
            metrics[n]['sigma_n'] = sigma_n
            metrics[n]['lmml'] = float(d['gp'].log_marginal_likelihood_value_)
    return metrics

# ─── Phase 1 ──────────────────────────────────────────────────────────────────
def run_phase1(model_tag, qs_path, te_path):
    print(f"\n=== Phase 1 [{model_tag}] ===")
    mat_tr = load_csv(qs_path); mat_te = load_csv(te_path)
    X_tr, y_tr, _, _ = extract_xy(mat_tr)
    X_te, y_te, _, _ = extract_xy(mat_te)
    print(f"  train N={len(y_tr)}  test N={len(y_te)}")
    m = eval_all(X_tr, y_tr, X_te, y_te, n_restarts=30)
    print(f"  {'surrogate':28s} {'R²_tr':>8s} {'R²_te':>8s} {'RMSE':>8s} {'ε_∞':>8s}")
    for n, d in m.items():
        print(f"  {n:28s} {d['r2_train']:8.4f} {d['r2_test']:8.4f} {d['rmse']:8.5f} {d['eps_inf']:8.5f}")
    return X_tr, y_tr, X_te, y_te, m


def scatter_fig(X_tr, y_tr, X_te, y_te, m, title, outfile):
    # surrogate order — M1 is the baseline
    order = ['M1 linear additive', 'M_quad full quadratic',
             'RBF cubic+linear', 'RBF tps+linear', 'ARD-GP (matern32+noise)']
    pretty = {'ARD-GP (matern32+noise)': 'ARD-Matérn-3/2'}
    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.7), sharex=True, sharey=True)
    for ax, key in zip(axes, order):
        d = m[key]; yp = d['yp_te']
        ax.scatter(y_te, yp, s=12, alpha=0.55, color='C0')
        lo = min(y_te.min(), yp.min()); hi = max(y_te.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7, alpha=0.5)
        ax.set_title(f"{pretty.get(key, key)}\nR²={d['r2_test']:.4f}  RMSE={d['rmse']:.4f}", fontsize=8.5)
        ax.set_xlabel('y_actual'); ax.grid(alpha=0.3)
    axes[0].set_ylabel('y_pred')
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(outfile, dpi=140, bbox_inches='tight'); plt.close()


phase1 = {}
for tag, qs, te in [('llama', PATHS['llama_qs50'], PATHS['llama_rs200']),
                    ('qwen',  PATHS['qwen_qs50'],  PATHS['qwen_rs200'])]:
    X_tr, y_tr, X_te, y_te, m = run_phase1(tag, qs, te)
    title = f"Phase 1 — {('Llama-3.1-8B' if tag == 'llama' else 'Qwen2.5-7B')} — 50 QS train → 200 RS test"
    outfile = f'{FIGDIR}/v4_fig1{"" if tag == "llama" else "q"}_scatter.png'
    scatter_fig(X_tr, y_tr, X_te, y_te, m, title, outfile)
    # Save lite (no arrays)
    phase1[tag] = {n: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in d.items() if k not in ('yp_tr', 'yp_te')}
                   for n, d in m.items()}

with open(f'{OUT}/phase1_results.json', 'w') as f:
    json.dump(phase1, f, indent=2)

print(f"\nSaved phase1_results.json, figures/v4_fig1_scatter.png, v4_fig1q_scatter.png")
print("Done.")
import sys; sys.exit(0)

# ─── (Phase 1b moved to 01b_phase1b_sweep.py) ────────────────────────────────
print("\n=== Phase 1b — sample-size sweep (Llama) ===")
mat_qs = load_csv(PATHS['llama_qs50']); mat_te = load_csv(PATHS['llama_rs200'])
X_all, y_all, _, _ = extract_xy(mat_qs)  # 50 in QS order (first 27 = quantile grid)
X_te,  y_te,  _, _ = extract_xy(mat_te)
N_grid = 27
det_idx = np.arange(N_grid)
rand_pool = np.arange(N_grid, len(y_all))  # indices 27..49

N_LIST = [27, 30, 35, 40, 45, 50]
SEEDS  = 20    # reduced from 50 — captures variance, keeps wall-time manageable

sweep = {n: {} for n in N_LIST}
for n in N_LIST:
    if n in (27, 50):
        if n == 27:
            idx = det_idx
        else:
            idx = np.arange(len(y_all))
        Xtr, ytr = X_all[idx], y_all[idx]
        m = eval_all(Xtr, ytr, X_te, y_te, n_restarts=10)
        for k, d in m.items():
            sweep[n][k] = dict(r2_test=[d['r2_test']], rmse=[d['rmse']],
                               eps_inf=[d['eps_inf']])
    else:
        rs = np.random.RandomState(0)
        for seed in range(SEEDS):
            rs.seed(seed)
            extras = rs.choice(rand_pool, n - N_grid, replace=False)
            idx = np.concatenate([det_idx, extras])
            Xtr, ytr = X_all[idx], y_all[idx]
            m = eval_all(Xtr, ytr, X_te, y_te, n_restarts=3)  # heavy: 5 surrogates × seeds
            for k, d in m.items():
                sweep[n].setdefault(k, dict(r2_test=[], rmse=[], eps_inf=[]))
                sweep[n][k]['r2_test'].append(d['r2_test'])
                sweep[n][k]['rmse'].append(d['rmse'])
                sweep[n][k]['eps_inf'].append(d['eps_inf'])
    print(f"  N={n:>2d}  done.", flush=True)

# Summarise: median + p10/p90
def summary(arr):
    a = np.asarray(arr, dtype=float)
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                p90=float(np.percentile(a, 90)), mean=float(a.mean()),
                n_seeds=int(len(a)))

sweep_sum = {}
for n, d in sweep.items():
    sweep_sum[n] = {}
    for k, dd in d.items():
        sweep_sum[n][k] = {m: summary(dd[m]) for m in ('r2_test', 'rmse', 'eps_inf')}

print("\nR²_test (median [p10, p90]):")
header = '  N    ' + '  '.join(f'{k[:18]:18s}' for k in sweep_sum[27].keys())
print(header)
for n in N_LIST:
    vals = []
    for k in sweep_sum[n].keys():
        s = sweep_sum[n][k]['r2_test']
        if s['n_seeds'] == 1:
            vals.append(f"{s['median']:.3f}             ")
        else:
            vals.append(f"{s['median']:.3f}[{s['p10']:.2f},{s['p90']:.2f}]")
    print(f"  N={n:>2d}  " + '  '.join(f'{v:18s}' for v in vals))

with open(f'{OUT}/phase1b_sweep_results.json', 'w') as f:
    json.dump({str(n): sweep_sum[n] for n in N_LIST}, f, indent=2)

# Plot learning curves
surrogates = [k for k in sweep_sum[27].keys()]
fig, ax = plt.subplots(figsize=(8, 5))
for k in surrogates:
    med = [sweep_sum[n][k]['r2_test']['median'] for n in N_LIST]
    p10 = [sweep_sum[n][k]['r2_test']['p10']    for n in N_LIST]
    p90 = [sweep_sum[n][k]['r2_test']['p90']    for n in N_LIST]
    line, = ax.plot(N_LIST, med, 'o-', label=k, lw=1.3)
    ax.fill_between(N_LIST, p10, p90, color=line.get_color(), alpha=0.12)
ax.axvline(27, color='gray', ls=':', alpha=0.7); ax.text(27.1, 0.05, 'N=27\n(QS grid)', fontsize=8)
ax.set_xlabel('train N'); ax.set_ylabel('R²_test')
ax.set_title('Phase 1b — R²_test vs train N (first 27 = QS grid, +random extras)\n[Llama-3.1-8B, 50 seeds for stochastic N]', fontsize=10)
ax.set_ylim(-0.3, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/v4_fig1b_lcurve_R2.png', dpi=140, bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
for k in surrogates:
    med = [sweep_sum[n][k]['rmse']['median'] for n in N_LIST]
    p10 = [sweep_sum[n][k]['rmse']['p10']    for n in N_LIST]
    p90 = [sweep_sum[n][k]['rmse']['p90']    for n in N_LIST]
    line, = ax.plot(N_LIST, med, 'o-', label=k, lw=1.3)
    ax.fill_between(N_LIST, p10, p90, color=line.get_color(), alpha=0.12)
ax.axvline(27, color='gray', ls=':', alpha=0.7)
ax.set_xlabel('train N'); ax.set_ylabel('RMSE_test'); ax.set_yscale('log')
ax.set_title('Phase 1b — RMSE_test vs train N (log scale)\n[Llama-3.1-8B, 50 seeds for stochastic N]', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/v4_fig1b_lcurve_RMSE.png', dpi=140, bbox_inches='tight'); plt.close()

print(f"\nSaved: phase1_results.json, phase1b_sweep_results.json")
print(f"Figures: v4_fig1_scatter.png, v4_fig1q_scatter.png, v4_fig1b_lcurve_R2.png, v4_fig1b_lcurve_RMSE.png")
print("Done.")
