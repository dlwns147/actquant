"""01c_phase1c_qs_vs_rs_sweep.py — Why does N=27 quantile-grid give poor R²,
and how does pure random at the same N compare?

For each N ∈ {27, 30, 35, 40, 45, 50} we draw 20 seeded training pools from
**two different sources**, evaluated on the same 200 RS test:

  • QS-first : first 27 = deterministic quantile-grid base from *_qs50;
               extras (N − 27) chosen uniformly from cols 27..49 of *_qs50.
               (For N=27 there is a single deterministic pool.)
  • RS-pure  : N samples drawn uniformly from *_rs50 (the dedicated 50-sample
               pure-random training file).  20 seeds; for N=50 the pool IS the
               full RS50 file (single deterministic point).

Surrogates: M1 (baseline), M_quad, RBF cubic+linear, RBF tps+linear.  ARD-GP
omitted (slow) — Phase 1b already characterised it at deterministic N.

We also report two **design-coverage** diagnostics on each train pool, useful
for explaining the QS-N=27 cliff:

  • z-range coverage ratio (per axis, relative to the test range):
        (z_max_train - z_min_train) / (z_max_test - z_min_test)
  • z-range coverage volume = product of the per-axis ratios (Π_k)

A small coverage volume means the surrogate is forced to **extrapolate** for
much of the test set, which is what breaks the spline-style RBFs at N=27.

Outputs:
    figures/v4_fig1c_lcurve_R2_overlay.png    Llama overlay (QS-first vs RS-pure)
    figures/v4_fig1cq_lcurve_R2_overlay.png   Qwen overlay
    figures/v4_fig1c_coverage_volume.png      train-pool coverage volume vs N
    phase1c_qs_vs_rs_sweep_results.json
"""
import os, sys, json, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import (load_csv, extract_xy, fit_rbf, features_M1, features_M_full_quad,
                     fit_ols, r2, rmse, eps_inf, PATHS)


def eval_surrogates(X_tr, y_tr, X_te, y_te):
    out = {}
    P1tr, P1te = features_M1(X_tr), features_M1(X_te)
    _, yp1, _ = fit_ols(P1tr, y_tr, P1te)
    out['M1 linear additive'] = yp1
    PQtr, PQte = features_M_full_quad(X_tr), features_M_full_quad(X_te)
    _, ypq, _ = fit_ols(PQtr, y_tr, PQte)
    out['M_quad full quadratic'] = ypq
    _, yprc, _ = fit_rbf(X_tr, y_tr, X_te, kernel='cubic')
    out['RBF cubic+linear'] = yprc
    _, yprt, _ = fit_rbf(X_tr, y_tr, X_te, kernel='tps')
    out['RBF tps+linear'] = yprt
    return {n: dict(r2_test=r2(y_te, yp), rmse=rmse(y_te, yp), eps_inf=eps_inf(y_te, yp))
            for n, yp in out.items()}


def coverage_volume(X_tr, X_te):
    """Per-axis range ratio (train / test) and product over axes."""
    tr_range = X_tr.max(0) - X_tr.min(0)
    te_range = X_te.max(0) - X_te.min(0)
    ratios = tr_range / np.where(te_range > 0, te_range, 1.0)
    return ratios.tolist(), float(np.prod(ratios))


def run(tag, qs_key, rs_key, te_key, N_LIST, SEEDS):
    print(f"\n=== Phase 1c [{tag}] ===", flush=True)
    Xqs, yqs, _, _ = extract_xy(load_csv(PATHS[qs_key]))     # 50; first 27 = quantile grid
    Xrs, yrs, _, _ = extract_xy(load_csv(PATHS[rs_key]))     # 50 pure random
    Xte, yte, _, _ = extract_xy(load_csv(PATHS[te_key]))
    N_grid = 27
    det_idx_qs   = np.arange(N_grid)
    extras_pool  = np.arange(N_grid, len(yqs))               # 23 QS random extras

    qs_first = {n: {} for n in N_LIST}
    rs_pure  = {n: {} for n in N_LIST}
    cov_qs   = {n: [] for n in N_LIST}                       # list of per-seed coverage
    cov_rs   = {n: [] for n in N_LIST}

    for n in N_LIST:
        # --- QS-first ---
        if n == 27:
            seeds_qs = [None]
        elif n == 50:
            seeds_qs = [None]
        else:
            seeds_qs = list(range(SEEDS))
        for seed in seeds_qs:
            if n == 27:
                idx = det_idx_qs
            elif n == 50:
                idx = np.arange(len(yqs))
            else:
                rs_local = np.random.RandomState(seed)
                extras = rs_local.choice(extras_pool, n - N_grid, replace=False)
                idx = np.concatenate([det_idx_qs, extras])
            Xtr, ytr = Xqs[idx], yqs[idx]
            m = eval_surrogates(Xtr, ytr, Xte, yte)
            for k, d in m.items():
                qs_first[n].setdefault(k, dict(r2_test=[], rmse=[], eps_inf=[]))
                qs_first[n][k]['r2_test'].append(d['r2_test'])
                qs_first[n][k]['rmse'].append(d['rmse'])
                qs_first[n][k]['eps_inf'].append(d['eps_inf'])
            _, v = coverage_volume(Xtr, Xte)
            cov_qs[n].append(v)
        # --- RS-pure ---
        if n == 50:
            seeds_rs = [None]
        else:
            seeds_rs = list(range(SEEDS))
        for seed in seeds_rs:
            if n == 50:
                idx = np.arange(len(yrs))
            else:
                rs_local = np.random.RandomState(seed + 1000)  # different seed namespace
                idx = rs_local.choice(np.arange(len(yrs)), n, replace=False)
            Xtr, ytr = Xrs[idx], yrs[idx]
            m = eval_surrogates(Xtr, ytr, Xte, yte)
            for k, d in m.items():
                rs_pure[n].setdefault(k, dict(r2_test=[], rmse=[], eps_inf=[]))
                rs_pure[n][k]['r2_test'].append(d['r2_test'])
                rs_pure[n][k]['rmse'].append(d['rmse'])
                rs_pure[n][k]['eps_inf'].append(d['eps_inf'])
            _, v = coverage_volume(Xtr, Xte)
            cov_rs[n].append(v)
        print(f"  [{tag}] N={n:>2d}  QS-first done ({len(seeds_qs)} pools), "
              f"RS-pure done ({len(seeds_rs)} pools)", flush=True)

    def summarise(d):
        out = {}
        for n in N_LIST:
            out[n] = {}
            for k, dd in d[n].items():
                out[n][k] = {m: dict(median=float(np.median(dd[m])),
                                      p10=float(np.percentile(dd[m], 10)),
                                      p90=float(np.percentile(dd[m], 90)),
                                      mean=float(np.mean(dd[m])),
                                      n_seeds=int(len(dd[m])))
                              for m in ('r2_test', 'rmse', 'eps_inf')}
        return out

    cov_qs_sum = {n: dict(median=float(np.median(cov_qs[n])),
                          p10=float(np.percentile(cov_qs[n], 10)),
                          p90=float(np.percentile(cov_qs[n], 90)),
                          n_seeds=int(len(cov_qs[n]))) for n in N_LIST}
    cov_rs_sum = {n: dict(median=float(np.median(cov_rs[n])),
                          p10=float(np.percentile(cov_rs[n], 10)),
                          p90=float(np.percentile(cov_rs[n], 90)),
                          n_seeds=int(len(cov_rs[n]))) for n in N_LIST}

    return dict(qs_first=summarise(qs_first), rs_pure=summarise(rs_pure),
                coverage_qs=cov_qs_sum, coverage_rs=cov_rs_sum)


OUT = '/NAS/SJ/actquant/search/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

N_LIST = [27, 30, 35, 40, 45, 50]
SEEDS  = 20

all_results = {}
for tag, qs_key, rs_key, te_key in [
    ('llama', 'llama_qs50', 'llama_rs50', 'llama_rs200'),
    ('qwen',  'qwen_qs50',  'qwen_rs50',  'qwen_rs200'),
]:
    all_results[tag] = run(tag, qs_key, rs_key, te_key, N_LIST, SEEDS)

with open(f'{OUT}/phase1c_qs_vs_rs_sweep_results.json', 'w') as f:
    json.dump({tag: {k: {str(n): all_results[tag][k][n] for n in N_LIST}
                     for k in all_results[tag]}
               for tag in all_results}, f, indent=2)

# ─── Plots ────────────────────────────────────────────────────────────────────
SURROGATES = ['M1 linear additive', 'M_quad full quadratic',
              'RBF cubic+linear', 'RBF tps+linear']
COLORS = dict(zip(SURROGATES, ['C0', 'C1', 'C2', 'C3']))

for tag, fig_suffix, pretty in [('llama', '', 'Llama-3.1-8B'),
                                ('qwen',  'q', 'Qwen2.5-7B')]:
    qsf = all_results[tag]['qs_first']
    rsp = all_results[tag]['rs_pure']

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for k in SURROGATES:
        med_qs = [qsf[n][k]['r2_test']['median'] for n in N_LIST]
        p10_qs = [qsf[n][k]['r2_test']['p10']    for n in N_LIST]
        p90_qs = [qsf[n][k]['r2_test']['p90']    for n in N_LIST]
        med_rs = [rsp[n][k]['r2_test']['median'] for n in N_LIST]
        p10_rs = [rsp[n][k]['r2_test']['p10']    for n in N_LIST]
        p90_rs = [rsp[n][k]['r2_test']['p90']    for n in N_LIST]
        c = COLORS[k]
        ax.plot(N_LIST, med_qs, 'o-',  color=c, lw=1.6, label=f'{k} — QS-first', alpha=0.95)
        ax.fill_between(N_LIST, p10_qs, p90_qs, color=c, alpha=0.10)
        ax.plot(N_LIST, med_rs, 's--', color=c, lw=1.4, label=f'{k} — RS-pure', alpha=0.65)
        ax.fill_between(N_LIST, p10_rs, p90_rs, color=c, alpha=0.05, hatch='//')
    ax.axvline(27, color='gray', ls=':', alpha=0.6)
    ax.text(27.1, -0.05, 'N=27\nQS grid\n(single det. pool)', fontsize=8)
    ax.set_xlabel('train N'); ax.set_ylabel('R²_test')
    ax.set_title(f'Phase 1c — QS-first (solid) vs RS-pure (dashed)\n'
                 f'[{pretty}, 20 seeds for stochastic N]', fontsize=10)
    ax.set_ylim(-0.6, 1.02); ax.legend(fontsize=7, ncol=2, loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/v4_fig1c{fig_suffix}_lcurve_R2_overlay.png',
                dpi=140, bbox_inches='tight'); plt.close()

# Coverage volume figure (both models on one plot)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (tag, pretty) in zip(axes, [('llama', 'Llama-3.1-8B'), ('qwen', 'Qwen2.5-7B')]):
    cov_qs = all_results[tag]['coverage_qs']
    cov_rs = all_results[tag]['coverage_rs']
    med_qs = [cov_qs[n]['median'] for n in N_LIST]
    med_rs = [cov_rs[n]['median'] for n in N_LIST]
    p10_qs = [cov_qs[n]['p10'] for n in N_LIST]
    p90_qs = [cov_qs[n]['p90'] for n in N_LIST]
    p10_rs = [cov_rs[n]['p10'] for n in N_LIST]
    p90_rs = [cov_rs[n]['p90'] for n in N_LIST]
    ax.plot(N_LIST, med_qs, 'o-',  color='C0', lw=1.6, label='QS-first')
    ax.fill_between(N_LIST, p10_qs, p90_qs, color='C0', alpha=0.15)
    ax.plot(N_LIST, med_rs, 's--', color='C3', lw=1.6, label='RS-pure')
    ax.fill_between(N_LIST, p10_rs, p90_rs, color='C3', alpha=0.15)
    ax.axhline(1.0, color='gray', ls=':', alpha=0.6)
    ax.set_xlabel('train N'); ax.set_ylabel('train z-range / test z-range  (product over 3 axes)')
    ax.set_title(f'{pretty}: train-pool z-range coverage volume')
    ax.legend(); ax.grid(alpha=0.3)
plt.suptitle('Phase 1c — train-pool coverage of the test z-range', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/v4_fig1c_coverage_volume.png', dpi=140, bbox_inches='tight')
plt.close()

# Console summary
print("\n" + "=" * 100)
print("Headline R² comparison (median over 20 seeds for stochastic N):")
print("=" * 100)
for tag in ('llama', 'qwen'):
    print(f"\n--- {tag} ---")
    qsf = all_results[tag]['qs_first']; rsp = all_results[tag]['rs_pure']
    print(f"  {'N':>3s}  " + '  '.join(f'{k[:12]:>26s}' for k in SURROGATES))
    for n in N_LIST:
        cells = []
        for k in SURROGATES:
            mq = qsf[n][k]['r2_test']['median']
            mr = rsp[n][k]['r2_test']['median']
            cells.append(f'QS:{mq:+.3f}  RS:{mr:+.3f}')
        print(f"  {n:>3d}  " + '  '.join(f'{c:>26s}' for c in cells))
    print(f"\n  coverage volume (train z-range / test z-range, product over 3 axes):")
    cq, cr = all_results[tag]['coverage_qs'], all_results[tag]['coverage_rs']
    for n in N_LIST:
        print(f"   N={n:>2d}: QS={cq[n]['median']:.3f}  RS={cr[n]['median']:.3f}")

print(f"\nSaved phase1c_qs_vs_rs_sweep_results.json, figures/v4_fig1c*.png")
print("Done.")
