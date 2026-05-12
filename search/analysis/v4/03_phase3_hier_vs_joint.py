"""03_phase3_hier_vs_joint.py — Hierarchical vs joint surrogate.

Two flavours of "hierarchical":

(A) Naive additive (no learning):
        y_hat_A(x) = z_W(x) + z_KV(x) + z_KVD(x)        (row 13 in the CSV)
    This is what the per-method search itself outputs as a sum-of-locals score.

(B) Hoeffding-style additive collapse of a fitted joint surrogate:
        y_hat_B(x) = f_0 + g_W(z_W) + g_KV(z_KV) + g_KVD(z_KVD)
    The g_k are KDE-smoothed conditional means from MC samples on the input box
    of the joint surrogate; this is the "best additive 1-D fit to the joint".

Joint surrogate: the winner from Phase 1 (RBF tps+linear) trained on the same 50 QS pool.

Comparing (A), (B), and the full joint surrogate on the 200 RS test gives the
ε-cost of going hierarchical.  The 2ε-Pareto theorem then bounds the loss to the
Pareto front induced by either additive predictor.

Run on Llama-3.1-8B and Qwen2.5-7B for cross-model agreement.

Outputs:
    figures/v4_fig3_hier_vs_joint.png   — bar chart of R², RMSE, ε_∞ per method
    figures/v4_fig3b_additive_collapse_scatter.png — pred vs actual scatter
    phase3_results.json
"""
import os, sys, json, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import (load_csv, extract_xy, fit_rbf, r2, rmse, eps_inf, PATHS, AXIS_NAMES_3)

OUT = '/NAS/SJ/actquant/search/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)


def additive_collapse(model, X_te, bounds, n_mc=8000, h_factor=0.15, seed=0):
    """f_0 + Σ_k (E[f | x_k] − f_0), Monte-Carlo with KDE-smoothed conditional means."""
    rng = np.random.RandomState(seed)
    X_mc = np.column_stack([rng.uniform(b[0], b[1], n_mc) for b in bounds])
    y_mc = model.predict(X_mc).ravel()
    f0 = float(y_mc.mean())
    d = X_te.shape[1]
    out = np.full(len(X_te), f0)
    for k in range(d):
        h = X_mc[:, k].std() * h_factor
        diff = X_te[:, k:k+1] - X_mc[:, k][None, :]
        w = np.exp(-0.5 * (diff / h) ** 2)
        g = (w * y_mc).sum(1) / w.sum(1)
        out = out + (g - f0)
    return out


def run(tag, qs_key, te_key):
    print(f"\n=== Phase 3 [{tag}] ===")
    X_tr, y_tr, yadd_tr, _ = extract_xy(load_csv(PATHS[qs_key]))
    X_te, y_te, yadd_te, _ = extract_xy(load_csv(PATHS[te_key]))

    # Joint surrogate (Phase 1 winner)
    _, yp_te_joint, model_joint = fit_rbf(X_tr, y_tr, X_te, kernel='tps')

    # (A) Naive additive baseline (no learning)
    yp_te_A = yadd_te

    # (B) Hoeffding additive collapse of joint surrogate
    bounds = np.column_stack([X_tr.min(0), X_tr.max(0)])
    yp_te_B = additive_collapse(model_joint, X_te, bounds, n_mc=8000, h_factor=0.15)

    rows = [
        ('Joint surrogate (RBF tps+linear)', yp_te_joint),
        ('Hier-A naive additive (row 13)',   yp_te_A),
        ('Hier-B Hoeffding collapse of joint', yp_te_B),
    ]
    out = {}
    print(f"  {'predictor':40s} {'R²':>8s} {'RMSE':>9s} {'ε_∞':>9s} {'ε_2':>9s}")
    for n, yp in rows:
        r2v = r2(y_te, yp); rm = rmse(y_te, yp); ei = eps_inf(y_te, yp); e2 = rm
        out[n] = dict(r2=r2v, rmse=rm, eps_inf=ei, eps_2=e2, yp=yp.tolist())
        print(f"  {n:40s} {r2v:8.4f} {rm:9.5f} {ei:9.5f} {e2:9.5f}")
    # Var-fraction of additive vs full
    var_full = float(np.var(yp_te_joint))
    var_resid_collapse = float(np.var(yp_te_joint - yp_te_B))
    out['_var_fraction_resid_over_full'] = var_resid_collapse / max(var_full, 1e-30)
    out['_2eps_Pareto_corridor'] = dict(
        from_A=2 * out['Hier-A naive additive (row 13)']['eps_inf'],
        from_B=2 * out['Hier-B Hoeffding collapse of joint']['eps_inf'],
        from_joint=2 * out['Joint surrogate (RBF tps+linear)']['eps_inf'],
    )
    print(f"  Var(joint − collapse) / Var(joint) = {out['_var_fraction_resid_over_full']:.4f}")
    print(f"  2ε corridor:  joint={out['_2eps_Pareto_corridor']['from_joint']:.4f}  "
          f"A={out['_2eps_Pareto_corridor']['from_A']:.4f}  "
          f"B={out['_2eps_Pareto_corridor']['from_B']:.4f}")
    return out, y_te, yp_te_joint, yp_te_A, yp_te_B


results = {}
plot_data = {}
for tag, qs_key, te_key in [('llama', 'llama_qs50', 'llama_rs200'),
                            ('qwen',  'qwen_qs50',  'qwen_rs200')]:
    out, y_te, yj, ya, yb = run(tag, qs_key, te_key)
    results[tag] = out
    plot_data[tag] = dict(y_te=y_te, yj=yj, ya=ya, yb=yb)

# Save (strip yp arrays for JSON)
def strip(d):
    return {k: ({kk: vv for kk, vv in v.items() if kk != 'yp'} if isinstance(v, dict) else v)
            for k, v in d.items()}

with open(f'{OUT}/phase3_results.json', 'w') as f:
    json.dump({k: strip(v) for k, v in results.items()}, f, indent=2)

# Figure 1: bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
labels = ['joint', 'Hier-A', 'Hier-B']
keys   = ['Joint surrogate (RBF tps+linear)',
          'Hier-A naive additive (row 13)',
          'Hier-B Hoeffding collapse of joint']
for ax, tag in zip(axes, ['llama', 'qwen']):
    r2v   = [results[tag][k]['r2']    for k in keys]
    rmsev = [results[tag][k]['rmse']  for k in keys]
    eiv   = [results[tag][k]['eps_inf'] for k in keys]
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.22, r2v,    0.21, color='C0', label='R²')
    ax.bar(xpos + 0.00, rmsev,  0.21, color='C1', label='RMSE')
    ax.bar(xpos + 0.22, eiv,    0.21, color='C3', label='ε_∞')
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_title(f"{('Llama-3.1-8B' if tag == 'llama' else 'Qwen2.5-7B')}: 200 RS test")
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)
plt.suptitle('Phase 3 — Hierarchical vs joint predictor on 200 RS test', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/v4_fig3_hier_vs_joint.png', dpi=140, bbox_inches='tight'); plt.close()

# Figure 2: scatter
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
for row, tag in enumerate(['llama', 'qwen']):
    d = plot_data[tag]
    y_te = d['y_te']
    rows = [('Joint (RBF tps)', d['yj'], 'C0'),
            ('Hier-A naive add (row13)', d['ya'], 'C1'),
            ('Hier-B collapse', d['yb'], 'C3')]
    for col, (name, yp, c) in enumerate(rows):
        ax = axes[row, col]
        ax.scatter(y_te, yp, s=12, alpha=0.55, color=c)
        lo, hi = min(y_te.min(), yp.min()), max(y_te.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7, alpha=0.5)
        e = eps_inf(y_te, yp)
        ax.fill_between([lo, hi], [lo - e, hi - e], [lo + e, hi + e], color='gray', alpha=0.12,
                        label=f'±ε_∞={e:.3f}')
        ax.set_title(f"{('Llama' if tag == 'llama' else 'Qwen')} — {name}\n"
                     f"R²={r2(y_te, yp):.4f} RMSE={rmse(y_te, yp):.4f}",
                     fontsize=9.5)
        ax.set_xlabel('y_actual'); ax.set_ylabel('y_pred'); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
plt.suptitle('Phase 3 — Predicted vs actual on 200 RS test', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/v4_fig3b_additive_collapse_scatter.png', dpi=140, bbox_inches='tight'); plt.close()

print(f"\nSaved phase3_results.json, figures/v4_fig3_*.png")
print("Done.")
