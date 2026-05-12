"""02_phase2_qs_vs_random.py — Quantile+random vs pure-random training.

We compare two 50-sample training pools, evaluated on the SAME 200 RS held-out test:

    QS+RS50  : 27 deterministic quantile-grid + 23 random extras (file llama_qs50)
    RS50     : 50 pure random samples                            (file llama_rs50)

Test:  llama_rs200 (200 random samples, no overlap by construction; we also
       verify there is no architecture overlap with either training pool).

Surrogates: M1, M_full_quad, RBF cubic, RBF tps, ARD-Matérn-3/2 +noise.

Reports:
  • Single fixed-pool comparison: QS surrogate vs RS surrogate evaluated on 200 RS test.
  • Bootstrap variability of the RS pool: 30 different 50-sample subsamples of the
    RS50 file (with replacement; the file itself has only 50 rows so this captures
    intra-pool variance, not new draws). Comparison anchors the QS-pool as a fixed point.
"""
import os, sys, json, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import (load_csv, extract_xy, all_surrogate_fits, r2, rmse, eps_inf, PATHS)

OUT = '/NAS/SJ/actquant/search/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

def fit_and_score(X_tr, y_tr, X_te, y_te, n_restarts=20):
    fits = all_surrogate_fits(X_tr, y_tr, X_te, ard_kernel='matern32', n_restarts=n_restarts)
    return {n: dict(r2=r2(y_te, d['yp_te']), rmse=rmse(y_te, d['yp_te']),
                    eps_inf=eps_inf(y_te, d['yp_te']))
            for n, d in fits.items()}

# ─── Load ─────────────────────────────────────────────────────────────────────
def run(model_tag, qs_key, rs_key, te_key):
    print(f"\n=== Phase 2 [{model_tag}] ===")
    Xqs, yqs, _, _ = extract_xy(load_csv(PATHS[qs_key]))
    Xrs, yrs, _, _ = extract_xy(load_csv(PATHS[rs_key]))
    Xte, yte, _, _ = extract_xy(load_csv(PATHS[te_key]))
    print(f"  QS={len(yqs)}  RS={len(yrs)}  test={len(yte)}")

    # Fixed (single-trial) comparison
    print("  --- single-trial scores (full QS-50 vs full RS-50) ---", flush=True)
    m_qs = fit_and_score(Xqs, yqs, Xte, yte, n_restarts=8)
    m_rs = fit_and_score(Xrs, yrs, Xte, yte, n_restarts=8)
    print(f"  {'surrogate':28s} {'R²_QS':>8s} {'R²_RS':>8s} {'Δ R²':>9s}  "
          f"{'RMSE_QS':>9s} {'RMSE_RS':>9s}")
    single = {}
    for n in m_qs:
        d_qs, d_rs = m_qs[n], m_rs[n]
        delta = d_qs['r2'] - d_rs['r2']
        single[n] = dict(R2_QS=d_qs['r2'], R2_RS=d_rs['r2'],
                         RMSE_QS=d_qs['rmse'], RMSE_RS=d_rs['rmse'],
                         eps_QS=d_qs['eps_inf'], eps_RS=d_rs['eps_inf'])
        print(f"  {n:28s} {d_qs['r2']:8.4f} {d_rs['r2']:8.4f} {delta:+9.4f}  "
              f"{d_qs['rmse']:9.5f} {d_rs['rmse']:9.5f}")

    # ── Bootstrap: 20 50-of-50 resamples of RS to estimate intra-pool variance ──
    rng = np.random.RandomState(0)
    SEEDS = 20
    boots = {n: dict(r2=[], rmse=[]) for n in m_qs}
    for s in range(SEEDS):
        rng.seed(s)
        idx = rng.choice(len(yrs), 50, replace=True)
        Xb, yb = Xrs[idx], yrs[idx]
        m = fit_and_score(Xb, yb, Xte, yte, n_restarts=2)
        for n, d in m.items():
            boots[n]['r2'].append(d['r2'])
            boots[n]['rmse'].append(d['rmse'])
        if s % 5 == 0: print(f"  bootstrap seed {s}/{SEEDS}", flush=True)
    boot_sum = {n: dict(R2_mean=float(np.mean(v['r2'])),
                        R2_std=float(np.std(v['r2'])),
                        RMSE_mean=float(np.mean(v['rmse'])),
                        RMSE_std=float(np.std(v['rmse'])))
                for n, v in boots.items()}
    print(f"\n  --- bootstrap RS-50 (30 resamples, with replacement) ---")
    print(f"  {'surrogate':28s} {'R²_QS':>8s} {'R²_RS μ±σ':>14s} {'R²_RS - R²_QS':>16s}")
    for n in m_qs:
        print(f"  {n:28s} {single[n]['R2_QS']:8.4f} "
              f"{boot_sum[n]['R2_mean']:.4f}±{boot_sum[n]['R2_std']:.4f}    "
              f"{boot_sum[n]['R2_mean'] - single[n]['R2_QS']:+8.4f}")
    return single, boot_sum, list(m_qs.keys())


phase2 = {}
for tag, qs_key, rs_key, te_key in [
    ('llama', 'llama_qs50', 'llama_rs50', 'llama_rs200'),
    ('qwen',  'qwen_qs50',  'qwen_rs50',  'qwen_rs200'),
]:
    single, boots, names = run(tag, qs_key, rs_key, te_key)
    phase2[tag] = dict(single=single, bootstrap=boots, order=names)

with open(f'{OUT}/phase2_results.json', 'w') as f:
    json.dump(phase2, f, indent=2)

# ─── Figure: R² bars (Llama + Qwen, side-by-side) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, tag in zip(axes, ['llama', 'qwen']):
    names = phase2[tag]['order']
    qs_r2 = [phase2[tag]['single'][n]['R2_QS'] for n in names]
    rs_r2 = [phase2[tag]['single'][n]['R2_RS'] for n in names]
    rs_b  = [phase2[tag]['bootstrap'][n]['R2_mean'] for n in names]
    rs_be = [phase2[tag]['bootstrap'][n]['R2_std']  for n in names]
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.22, qs_r2, 0.21, color='C0', label='QS+RS-50 (fixed)')
    ax.bar(xpos + 0.00, rs_r2, 0.21, color='C1', label='RS-50 (file)')
    ax.bar(xpos + 0.22, rs_b,  0.21, yerr=rs_be, color='C3', label='RS-50 bootstrap (mean±σ)')
    ax.set_xticks(xpos); ax.set_xticklabels([n.replace(' ', '\n', 1) for n in names], fontsize=7.5)
    ax.set_ylabel('R²_test'); ax.set_ylim(0.5, 1.0); ax.grid(alpha=0.3, axis='y')
    ax.set_title(f"{('Llama-3.1-8B' if tag == 'llama' else 'Qwen2.5-7B')}: 50 train → 200 RS test")
    ax.legend(fontsize=8)
plt.suptitle('Phase 2 — QS+RS-50 vs pure RS-50 training', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/v4_fig2_qs_vs_random.png', dpi=140, bbox_inches='tight'); plt.close()

print(f"\nSaved phase2_results.json, figures/v4_fig2_qs_vs_random.png")
print("Done.")
