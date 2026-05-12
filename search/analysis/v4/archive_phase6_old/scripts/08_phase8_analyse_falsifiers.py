"""08_phase8_analyse_falsifiers.py — analyse Phase 7 GPU measurements.

Inputs:
    acquired_falsifiers_{tag}.json   (Phase 6 output: predicted candidates + 4D PF)
    eval_falsifiers_{tag}.json       (Phase 7 output: per-arch measured y)

Per model, computes:
  • measured-falsification rate: fraction of candidates whose ACTUAL (y, wbits,
    kvbits, kvdim) is not dominated by the 4D baseline PF
  • RBF prediction quality on falsifier candidates: R², RMSE, ε_∞ of (μ vs y)
  • per-bucket falsification rate (3D buckets over complexity)
  • Wilson 95% CI on falsification rate
  • 3 figures (loss vs each of wbits / kvbits / kvdim), each showing:
      - baseline PF projection (naive_add)
      - candidates: RBF μ (predicted) vs y_actual (measured)
      - falsification status colour

Outputs:
    figures/v4_fig8_falsification_{tag}.png
    phase8_falsification_results.json    (keyed by llama / qwen)
"""
import os, sys, json, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as _beta

OUT = '/NAS/SJ/actquant/search/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)


def wilson_ci(k, n, alpha=0.05):
    """Wilson 95% CI for k/n; for k=0 use rule-of-three upper bound."""
    if n == 0: return (0.0, 1.0)
    if k == 0: return (0.0, -np.log(alpha) / n)
    if k == n: return (1 - (-np.log(alpha) / n), 1.0)
    lo = float(_beta.ppf(alpha/2,     k,     n - k + 1))
    hi = float(_beta.ppf(1 - alpha/2, k + 1, n - k))
    return (lo, hi)


def is_dominated_by_any_4d(point, pf_pts):
    le = np.all(pf_pts <= point, axis=1)
    lt = np.any(pf_pts <  point, axis=1)
    return bool(np.any(le & lt))


def analyse_model(tag, pretty, fig_suffix):
    acq_path = f'{OUT}/acquired_falsifiers_{tag}.json'
    eval_path = f'{OUT}/eval_falsifiers_{tag}.json'
    if not os.path.exists(eval_path):
        print(f"[{tag}] no eval file yet: {eval_path}")
        return None
    with open(acq_path) as f: acq = json.load(f)
    with open(eval_path) as f: ev = json.load(f)
    results = ev['results']
    pf_pts = np.array(acq['baseline_pf_points'])    # (P, 4) = (naive_add, wbits, kvbits, kvdim)
    print(f"\n=== Phase 8 [{tag}] — {pretty} ===")
    print(f"  baseline PF size: {len(pf_pts)}")
    print(f"  candidates evaluated: {len(results)}")
    if not results: return None

    # Build candidate arrays
    y_actual = np.array([r['y_actual']  for r in results])
    mu_pred  = np.array([r['rbf_mu']    for r in results])
    naive_add = np.array([r['naive_add'] for r in results])
    wbits    = np.array([r['wbits']     for r in results])
    kvbits   = np.array([r['kvbits']    for r in results])
    kvdim    = np.array([r['kvdim']     for r in results])

    # Falsification: actual (y, wb, kb, kd) not dominated by any PF point
    falsifies = np.array([
        not is_dominated_by_any_4d(np.array([y_actual[i], wbits[i], kvbits[i], kvdim[i]]),
                                    pf_pts)
        for i in range(len(results))
    ])
    n_fals = int(falsifies.sum()); n = len(results)
    ci = wilson_ci(n_fals, n)
    print(f"  measured-falsification: {n_fals}/{n} = {100*n_fals/n:.1f}%  "
          f"(95% CI: [{100*ci[0]:.1f}%, {100*ci[1]:.1f}%])")

    # RBF prediction quality
    res = y_actual - mu_pred
    rmse = float(np.sqrt(np.mean(res ** 2)))
    eps_inf = float(np.max(np.abs(res)))
    ss_t = float(np.sum((y_actual - y_actual.mean()) ** 2))
    r2 = float(1 - np.sum(res ** 2) / max(ss_t, 1e-30))
    print(f"  RBF μ vs y_actual: R²={r2:.4f}  RMSE={rmse:.4f}  ε_∞={eps_inf:.4f}")
    # Bias
    print(f"  bias (mean residual y - μ) = {res.mean():+.4f}")

    # Naive-add quality on candidates
    res_naive = y_actual - naive_add
    rmse_naive = float(np.sqrt(np.mean(res_naive ** 2)))
    print(f"  naive_add vs y_actual: RMSE={rmse_naive:.4f}  (bias={res_naive.mean():+.4f})")

    # Per-bucket breakdown (3D complexity buckets, 2 levels each)
    def b3(arr, q=(0.5,)):
        e = np.quantile(arr, [0.0, *q, 1.0]); e[0] -= 1e-9; e[-1] += 1e-9
        return np.digitize(arr, e[1:-1])
    bw = b3(wbits); bk = b3(kvbits); bd = b3(kvdim)
    print("  bucket falsification (wbits×kvbits×kvdim, low/high split):")
    for w_b in (0, 1):
        for k_b in (0, 1):
            for d_b in (0, 1):
                mask = (bw == w_b) & (bk == k_b) & (bd == d_b)
                if mask.sum() == 0: continue
                print(f"    [wb={'lo' if w_b==0 else 'hi'} kvb={'lo' if k_b==0 else 'hi'} "
                      f"kvd={'lo' if d_b==0 else 'hi'}]: "
                      f"{int(falsifies[mask].sum())}/{int(mask.sum())} falsify")

    # ── Figures: 3 panels, one per complexity axis ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axis_data = [
        ('wbits',  wbits,  pf_pts[:, 1]),
        ('kvbits', kvbits, pf_pts[:, 2]),
        ('kvdim',  kvdim,  pf_pts[:, 3]),
    ]
    for ax, (name, x_cand, x_pf) in zip(axes, axis_data):
        # baseline PF projection
        ax.scatter(x_pf, pf_pts[:, 0], s=4, color='C0', alpha=0.25,
                   label=f'baseline PF projection (n={len(pf_pts)})')
        # candidates RBF μ
        ax.scatter(x_cand, mu_pred, s=20, color='C2', alpha=0.5,
                   marker='^', edgecolor='k', linewidth=0.3,
                   label=f'cand RBF μ (n={n})')
        # candidates y_actual
        ax.scatter(x_cand[~falsifies], y_actual[~falsifies], s=28, color='C0',
                   marker='o', edgecolor='k', linewidth=0.3,
                   label=f'y_actual — dominated by PF ({n - n_fals})')
        ax.scatter(x_cand[ falsifies], y_actual[ falsifies], s=42, color='C3',
                   marker='*', edgecolor='k', linewidth=0.4, zorder=4,
                   label=f'y_actual — FALSIFIES PF ({n_fals})')
        # connect μ to y for each candidate
        for i in range(n):
            ax.plot([x_cand[i], x_cand[i]], [mu_pred[i], y_actual[i]],
                    color='gray', lw=0.4, alpha=0.5)
        ax.set_xlabel(name); ax.set_ylabel('loss (JSD on wikitext2)')
        ax.set_title(f"loss vs {name}")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper left')
    plt.suptitle(f"Phase 8 — Falsification on {pretty} "
                 f"({n_fals}/{n} = {100*n_fals/n:.1f}%, 95% CI [{100*ci[0]:.1f}, {100*ci[1]:.1f}]%)  "
                 f"|  RBF μ R²={r2:.3f}, RMSE={rmse:.4f}", fontsize=10)
    plt.tight_layout()
    fig_path = f"{FIGDIR}/v4_fig8_falsification_{fig_suffix}.png" if fig_suffix \
        else f"{FIGDIR}/v4_fig8_falsification.png"
    plt.savefig(fig_path, dpi=140, bbox_inches='tight'); plt.close()
    print(f"  saved figure: {fig_path}")

    return dict(
        n_total=int(n),
        n_falsifies=int(n_fals),
        falsification_rate=float(n_fals / n),
        wilson_95ci=[float(ci[0]), float(ci[1])],
        rbf_R2=r2, rbf_RMSE=rmse, rbf_eps_inf=eps_inf,
        rbf_bias=float(res.mean()),
        naive_add_RMSE=rmse_naive, naive_add_bias=float(res_naive.mean()),
        sigma_resid_rbf_train=float(acq['sigma_resid_rbf']),
        baseline_PF_size=int(len(pf_pts)),
    )


phase8 = {}
for tag, pretty, fig_suffix in [('llama', 'Llama-3.1-8B', ''),
                                ('qwen', 'Qwen2.5-7B', 'q')]:
    r = analyse_model(tag, pretty, fig_suffix)
    if r is not None: phase8[tag] = r

with open(f'{OUT}/phase8_falsification_results.json', 'w') as f:
    json.dump(phase8, f, indent=2)

print(f"\nSaved phase8_falsification_results.json")
print("Done.")
