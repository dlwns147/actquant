"""01b_phase1b_sweep.py — Sample-size sweep N=27..50 (Llama + Qwen).

First 27 = deterministic quantile-grid base; extras chosen uniformly from cols 27..49.
N ∈ {27, 50} are single deterministic points; N ∈ {30, 35, 40, 45} use 20 seeds each.

Surrogates: M1, M_full_quad, RBF cubic, RBF tps, ARD-Matérn-3/2 +noise.
ARD-GP is included only at deterministic N to keep wall-time bounded.

Outputs:
    figures/v4_fig1b_lcurve_R2.png    (Llama)
    figures/v4_fig1b_lcurve_RMSE.png  (Llama)
    figures/v4_fig1bq_lcurve_R2.png   (Qwen)
    figures/v4_fig1bq_lcurve_RMSE.png (Qwen)
    phase1b_sweep_results.json        — {llama, qwen} → {N → surrogate → metric → summary}
"""
import os, sys, json, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import (load_csv, extract_xy, r2, rmse, eps_inf, PATHS)


def eval_all(X_tr, y_tr, X_te, y_te, n_restarts, include_gp=True):
    """Lighter sweep: optionally skip ARD-GP (slow)."""
    import _common as cm
    out = {}
    Phi_tr1, Phi_te1 = cm.features_M1(X_tr), cm.features_M1(X_te)
    ytr1, yte1, _ = cm.fit_ols(Phi_tr1, y_tr, Phi_te1)
    out['M1 linear additive'] = dict(yp_tr=ytr1, yp_te=yte1)

    Phi_trQ, Phi_teQ = cm.features_M_full_quad(X_tr), cm.features_M_full_quad(X_te)
    ytrQ, yteQ, _ = cm.fit_ols(Phi_trQ, y_tr, Phi_teQ)
    out['M_quad full quadratic'] = dict(yp_tr=ytrQ, yp_te=yteQ)

    yptr, ypte, _ = cm.fit_rbf(X_tr, y_tr, X_te, kernel='cubic')
    out['RBF cubic+linear'] = dict(yp_tr=yptr, yp_te=ypte)
    yptr, ypte, _ = cm.fit_rbf(X_tr, y_tr, X_te, kernel='tps')
    out['RBF tps+linear']   = dict(yp_tr=yptr, yp_te=ypte)
    if include_gp:
        yptr, ypte, _ = cm.fit_ard_gp(X_tr, y_tr, X_te, kernel='matern32',
                                       with_noise=True, n_restarts=n_restarts)
        out['ARD-GP (matern32+noise)'] = dict(yp_tr=yptr, yp_te=ypte)
    return {n: dict(r2_test=r2(y_te, d['yp_te']), rmse=rmse(y_te, d['yp_te']),
                    eps_inf=eps_inf(y_te, d['yp_te'])) for n, d in out.items()}


def run_sweep(tag, qs_key, te_key, N_LIST, SEEDS):
    print(f"\n=== Phase 1b — sample-size sweep [{tag}] ===", flush=True)
    X_all, y_all, _, _ = extract_xy(load_csv(PATHS[qs_key]))
    X_te,  y_te,  _, _ = extract_xy(load_csv(PATHS[te_key]))
    N_grid = 27
    det_idx = np.arange(N_grid)
    rand_pool = np.arange(N_grid, len(y_all))

    sweep = {n: {} for n in N_LIST}
    for n in N_LIST:
        if n in (27, 50):
            idx = det_idx if n == 27 else np.arange(len(y_all))
            m = eval_all(X_all[idx], y_all[idx], X_te, y_te, n_restarts=5, include_gp=True)
            for k, d in m.items():
                sweep[n][k] = dict(r2_test=[d['r2_test']], rmse=[d['rmse']],
                                   eps_inf=[d['eps_inf']])
            print(f"  [{tag}] N={n:>2d}  det  done.", flush=True)
        else:
            rs = np.random.RandomState(0)
            for seed in range(SEEDS):
                rs.seed(seed)
                extras = rs.choice(rand_pool, n - N_grid, replace=False)
                idx = np.concatenate([det_idx, extras])
                # In the sweep loop, drop ARD-GP (slow); we report it for det N only.
                m = eval_all(X_all[idx], y_all[idx], X_te, y_te, n_restarts=2, include_gp=False)
                for k, d in m.items():
                    sweep[n].setdefault(k, dict(r2_test=[], rmse=[], eps_inf=[]))
                    sweep[n][k]['r2_test'].append(d['r2_test'])
                    sweep[n][k]['rmse'].append(d['rmse'])
                    sweep[n][k]['eps_inf'].append(d['eps_inf'])
                if seed % 5 == 0:
                    print(f"  [{tag}] N={n}  seed {seed}/{SEEDS}", flush=True)
            print(f"  [{tag}] N={n:>2d}  done.", flush=True)
    return sweep


def summary(arr):
    a = np.asarray(arr, dtype=float)
    return dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                p90=float(np.percentile(a, 90)), mean=float(a.mean()),
                n_seeds=int(len(a)))


def summarise(sweep, N_LIST):
    return {n: {k: {m: summary(d[k][m]) for m in ('r2_test', 'rmse', 'eps_inf')}
                for k in d} for n, d in sweep.items()}


def print_table(sweep_sum, N_LIST):
    keys = list(sweep_sum[27].keys())
    keys = [k for k in keys if all(k in sweep_sum[n] for n in N_LIST)]
    print('  N    ' + '  '.join(f'{k[:18]:>20s}' for k in keys))
    for n in N_LIST:
        vals = []
        for k in keys:
            s = sweep_sum[n][k]['r2_test']
            if s['n_seeds'] == 1:
                vals.append(f"{s['median']:.3f}")
            else:
                vals.append(f"{s['median']:.3f}[{s['p10']:.2f},{s['p90']:.2f}]")
        print(f"  N={n:>2d}  " + '  '.join(f'{v:>20s}' for v in vals))


def plot_curves(sweep_sum, N_LIST, fig_r2, fig_rmse, pretty_tag):
    keys = list(sweep_sum[27].keys())
    keys = [k for k in keys if all(k in sweep_sum[n] for n in N_LIST)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in keys:
        med = [sweep_sum[n][k]['r2_test']['median'] for n in N_LIST]
        p10 = [sweep_sum[n][k]['r2_test']['p10']    for n in N_LIST]
        p90 = [sweep_sum[n][k]['r2_test']['p90']    for n in N_LIST]
        line, = ax.plot(N_LIST, med, 'o-', label=k, lw=1.3)
        ax.fill_between(N_LIST, p10, p90, color=line.get_color(), alpha=0.12)
    ax.axvline(27, color='gray', ls=':', alpha=0.7)
    ax.text(27.1, 0.05, 'N=27\n(QS grid)', fontsize=8)
    ax.set_xlabel('train N'); ax.set_ylabel('R²_test')
    ax.set_title(f'Phase 1b — R²_test vs train N (first 27 = QS grid, +random extras)\n'
                 f'[{pretty_tag}, 20 seeds for stochastic N]', fontsize=10)
    ax.set_ylim(-0.6, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_r2, dpi=140, bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in keys:
        med = [sweep_sum[n][k]['rmse']['median'] for n in N_LIST]
        p10 = [sweep_sum[n][k]['rmse']['p10']    for n in N_LIST]
        p90 = [sweep_sum[n][k]['rmse']['p90']    for n in N_LIST]
        line, = ax.plot(N_LIST, med, 'o-', label=k, lw=1.3)
        ax.fill_between(N_LIST, p10, p90, color=line.get_color(), alpha=0.12)
    ax.axvline(27, color='gray', ls=':', alpha=0.7)
    ax.set_xlabel('train N'); ax.set_ylabel('RMSE_test'); ax.set_yscale('log')
    ax.set_title(f'Phase 1b — RMSE_test vs train N (log scale)\n'
                 f'[{pretty_tag}, 20 seeds]', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(fig_rmse, dpi=140, bbox_inches='tight'); plt.close()


OUT = '/NAS/SJ/actquant/search/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

N_LIST = [27, 30, 35, 40, 45, 50]
SEEDS  = 20

all_results = {}
for tag, qs_key, te_key, pretty, fig_suffix in [
    ('llama', 'llama_qs50', 'llama_rs200', 'Llama-3.1-8B', ''),
    ('qwen',  'qwen_qs50',  'qwen_rs200',  'Qwen2.5-7B',   'q'),
]:
    sweep = run_sweep(tag, qs_key, te_key, N_LIST, SEEDS)
    sweep_sum = summarise(sweep, N_LIST)
    all_results[tag] = sweep_sum
    print(f"\n[{tag}] R²_test (median [p10, p90]):")
    print_table(sweep_sum, N_LIST)
    plot_curves(sweep_sum, N_LIST,
                f'{FIGDIR}/v4_fig1b{fig_suffix}_lcurve_R2.png',
                f'{FIGDIR}/v4_fig1b{fig_suffix}_lcurve_RMSE.png',
                pretty)

with open(f'{OUT}/phase1b_sweep_results.json', 'w') as f:
    json.dump({tag: {str(n): all_results[tag][n] for n in N_LIST}
               for tag in all_results}, f, indent=2)

print(f"\nSaved phase1b_sweep_results.json, figures/v4_fig1b*.png")
print("Done.")
