"""03c_phase3c_pair_orderings.py — does the AXIS ORDER inside the hierarchical
decomposition matter?

Two decomposition styles, all evaluated on the **same 50-sample 3-way QS train
pool** and the **same 200 RS test set** (Llama-3.1-8B):

A) **Pair-then-residual** (Phase 3b style, 3 pair directions):
       y ≈ f_pair(x_a, x_b) + g(x_c)
   f_pair = RBF tps+linear (2D),  g = degree-2 polynomial (1D).
   We use the *pair-CSV* (260510 *_wk / *_kvkd / *_wkd) for f_pair, and the
   3-way QS pool for g.  For a fair single-point comparison, n_pair is fixed at
   each pair's full pool (29 / 30 / 31) and n_3way = 50 (independent pools),
   so the 'total hier samples used' is 79/80/81. We also rerun with
   (n_pair=10, n_3way=40) — the Phase-3b sweet spot for KD — for an apples-to-
   apples 50-budget point.

B) **Sequential 1-D additive** (6 orderings):  for every permutation π of
   the 3 axes,
       g_a fit to (x_π(0), y)
       g_b fit to (x_π(1), y − g_a(x_π(0)))
       g_c fit to (x_π(2), y − g_a − g_b)
   y_pred(x) = g_a(x_π(0)) + g_b(x_π(1)) + g_c(x_π(2))
   Each g_k = degree-2 polynomial.  Order MATTERS because each stage greedily
   fits the residual of the previous stage (stagewise / forward-additive ≠ OLS
   additive). All 50 3-way QS samples used at every stage (total budget = 50).

Reference: joint RBF tps+linear on the full 50 3-way QS samples.

Outputs:
    figures/v4_fig3c_pair_orderings.png    bar chart: R² across all decompositions
    phase3c_orderings_results.json
"""
import os, sys, json, time, warnings, itertools
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import (load_csv, extract_xy, r2, rmse, eps_inf, PATHS)
from predictor.rbf import RBF as PySOTRBF

OUT = '/NAS/SJ/actquant/search/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

AXIS_NAMES = ('W', 'KV', 'KVD')
AXIS_IDX   = dict(W=0, KV=1, KVD=2)

PAIRS = {
    'WK':  dict(file_key='llama_wk',   pair_axes=(0, 1), res_axis=2),
    'KD':  dict(file_key='llama_kvkd', pair_axes=(1, 2), res_axis=0),
    'WD':  dict(file_key='llama_wkd',  pair_axes=(0, 2), res_axis=1),
}

POLY_DEG = 2
SEEDS    = 10


def fit_pair_rbf(X2, y, lb, ub):
    m = PySOTRBF(kernel='tps', tail='linear', lb=lb, ub=ub)
    m.fit(X2, y)
    return m


def fit_1d_poly(x, y, deg=POLY_DEG):
    if len(x) < deg + 1:
        return np.poly1d([float(np.mean(y))])
    return np.poly1d(np.polyfit(x, y, deg))


# ─── Load Llama 3-way + per-pair CSVs ───────────────────────────────────────
print("="*100); print("Phase 3c — pair / axis-ordering analysis (Llama)"); print("="*100, flush=True)
X3,  y3,  _, _ = extract_xy(load_csv(PATHS['llama_qs50']),  n_axes=3)
X_te, y_te, _, _ = extract_xy(load_csv(PATHS['llama_rs200']), n_axes=3)
print(f"  3-way QS train: N={len(y3)}   test: N={len(y_te)}", flush=True)

pair_data = {}
for tag, cfg in PAIRS.items():
    X_p, y_p, _, _ = extract_xy(load_csv(PATHS[cfg['file_key']]), n_axes=2)
    pa = cfg['pair_axes']
    lb = np.minimum.reduce([X_p.min(0), X3[:, list(pa)].min(0), X_te[:, list(pa)].min(0)])
    ub = np.maximum.reduce([X_p.max(0), X3[:, list(pa)].max(0), X_te[:, list(pa)].max(0)])
    pair_data[tag] = dict(X=X_p, y=y_p, lb=lb, ub=ub, cfg=cfg)
    print(f"  pair {tag}: N_pair={len(y_p)}", flush=True)

# ─── Joint baseline (RBF on N=50 3-way) ─────────────────────────────────────
def fit_joint_rbf(X_tr, y_tr, X_te):
    lb = np.minimum(X_tr.min(0), X_te.min(0)); ub = np.maximum(X_tr.max(0), X_te.max(0))
    m = PySOTRBF(kernel='tps', tail='linear', lb=lb, ub=ub)
    m.fit(X_tr, y_tr)
    return m.predict(X_te).ravel()

yp_joint = fit_joint_rbf(X3, y3, X_te)
joint_metrics = dict(r2=r2(y_te, yp_joint), rmse=rmse(y_te, yp_joint), eps_inf=eps_inf(y_te, yp_te if False else y_te,))
joint_metrics = dict(r2=r2(y_te, yp_joint), rmse=rmse(y_te, yp_joint), eps_inf=eps_inf(y_te, yp_joint))
print(f"\n[joint baseline] RBF on N=50 3-way QS: R²={joint_metrics['r2']:.4f}  "
      f"RMSE={joint_metrics['rmse']:.5f}  ε_∞={joint_metrics['eps_inf']:.4f}", flush=True)

# ─── (A) Pair-then-residual: 3 pair directions, two sample-budgets ──────────
print("\n[A] Pair-then-residual (pair-CSV + 3-way residual)", flush=True)
A_results = {}
def pair_then_residual(pair_tag, n_pair, n_3way, seed=0):
    rng = np.random.RandomState(seed)
    pd = pair_data[pair_tag]; cfg = pd['cfg']
    pa, ra = list(cfg['pair_axes']), cfg['res_axis']
    if n_pair < len(pd['y']):
        idx_p = rng.choice(len(pd['y']), n_pair, replace=False)
    else:
        idx_p = np.arange(len(pd['y']))
    Xp_tr, yp_tr = pd['X'][idx_p], pd['y'][idx_p]
    m_pair = fit_pair_rbf(Xp_tr, yp_tr, pd['lb'], pd['ub'])
    idx_3 = rng.choice(len(y3), n_3way, replace=False) if n_3way < len(y3) else np.arange(len(y3))
    Xtr3, ytr3 = X3[idx_3], y3[idx_3]
    r_tr = ytr3 - m_pair.predict(Xtr3[:, pa]).ravel()
    g_res = fit_1d_poly(Xtr3[:, ra], r_tr)
    yp_te = m_pair.predict(X_te[:, pa]).ravel() + g_res(X_te[:, ra])
    return yp_te

# A.1 — full pair pool + full 3-way pool (= 79-81 total samples; hier uses MORE than joint here)
for tag in PAIRS:
    yp = pair_then_residual(tag, len(pair_data[tag]['y']), len(y3))
    A_results[f'pair_full_{tag}'] = dict(
        r2=r2(y_te, yp), rmse=rmse(y_te, yp), eps_inf=eps_inf(y_te, yp),
        label=f"Hier-{tag} (n_pair={len(pair_data[tag]['y'])} + n_3way=50)")
    print(f"  pair_full_{tag}: R²={A_results[f'pair_full_{tag}']['r2']:+.4f}", flush=True)

# A.2 — 50-budget split (n_pair=10, n_3way=40), 10-seed median
for tag in PAIRS:
    r2s, rms, eis = [], [], []
    for s in range(SEEDS):
        yp = pair_then_residual(tag, 10, 40, seed=s)
        r2s.append(r2(y_te, yp)); rms.append(rmse(y_te, yp)); eis.append(eps_inf(y_te, yp))
    A_results[f'pair50_{tag}'] = dict(
        r2=float(np.median(r2s)), rmse=float(np.median(rms)), eps_inf=float(np.median(eis)),
        r2_p10=float(np.percentile(r2s, 10)), r2_p90=float(np.percentile(r2s, 90)),
        label=f"Hier-{tag} (n_pair=10 + n_3way=40, budget=50, {SEEDS} seeds)")
    print(f"  pair50_{tag}: R²={A_results[f'pair50_{tag}']['r2']:+.4f}  "
          f"[{A_results[f'pair50_{tag}']['r2_p10']:+.3f}, {A_results[f'pair50_{tag}']['r2_p90']:+.3f}]", flush=True)

# ─── (B) Sequential 1D-1D-1D additive: 6 orderings on 3-way QS only ─────────
print("\n[B] Sequential 1D additive (6 orderings of axes — all 50 3-way samples)", flush=True)
B_results = {}
for perm in itertools.permutations((0, 1, 2)):
    # Stage 1
    g1 = fit_1d_poly(X3[:, perm[0]], y3)
    r1 = y3 - g1(X3[:, perm[0]])
    # Stage 2
    g2 = fit_1d_poly(X3[:, perm[1]], r1)
    r2_arr = r1 - g2(X3[:, perm[1]])
    # Stage 3
    g3 = fit_1d_poly(X3[:, perm[2]], r2_arr)
    yp_te = g1(X_te[:, perm[0]]) + g2(X_te[:, perm[1]]) + g3(X_te[:, perm[2]])
    name = '→'.join(AXIS_NAMES[a] for a in perm)
    B_results[name] = dict(
        r2=r2(y_te, yp_te), rmse=rmse(y_te, yp_te), eps_inf=eps_inf(y_te, yp_te),
        permutation=list(perm), label=f"Seq-1D {name}")
    print(f"  {name}: R²={B_results[name]['r2']:+.4f}  "
          f"RMSE={B_results[name]['rmse']:.5f}  ε_∞={B_results[name]['eps_inf']:.4f}", flush=True)

# Order-sensitivity diagnostic
r2_vals = [B_results[n]['r2'] for n in B_results]
print(f"\n  Seq-1D R² range across 6 orderings: [{min(r2_vals):+.4f}, {max(r2_vals):+.4f}]  "
      f"(spread = {max(r2_vals)-min(r2_vals):+.4f})", flush=True)

# ─── Save + figure ──────────────────────────────────────────────────────────
out = dict(joint=joint_metrics, hier_A=A_results, hier_B=B_results,
           note="Llama-only; pair CSVs from 260510 (29/30/31 samples).")
with open(f'{OUT}/phase3c_orderings_results.json', 'w') as f:
    json.dump(out, f, indent=2)

# Bar chart: all decompositions on one axis, sorted by R²
all_items = [('Joint RBF (N=50)', joint_metrics['r2'], 'C0')]
for k, v in A_results.items():
    color = 'C2' if 'full' in k else 'C3'
    all_items.append((v['label'], v['r2'], color))
for k, v in B_results.items():
    all_items.append((v['label'], v['r2'], 'C4'))
labels = [x[0] for x in all_items]; vals = [x[1] for x in all_items]; cols = [x[2] for x in all_items]
order = np.argsort(vals)[::-1]
fig, ax = plt.subplots(figsize=(14, 6))
ax.barh(range(len(labels)), [vals[i] for i in order], color=[cols[i] for i in order],
        edgecolor='k', linewidth=0.4)
ax.set_yticks(range(len(labels))); ax.set_yticklabels([labels[i] for i in order], fontsize=8)
ax.axvline(joint_metrics['r2'], color='C0', ls='--', lw=1, label=f'joint R²={joint_metrics["r2"]:.4f}')
ax.set_xlabel('R² on 200 RS test'); ax.set_xlim(min(vals) - 0.05, 1.0)
ax.set_title('Phase 3c — pair/axis-order analysis  (Llama, 50-sample 3-way QS train)\n'
             'C0=joint, C2=hier pair-full, C3=hier pair-50-budget, C4=sequential 1D 6-orderings',
             fontsize=10)
ax.grid(alpha=0.3, axis='x'); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/v4_fig3c_pair_orderings.png', dpi=140, bbox_inches='tight'); plt.close()
print(f"\nsaved figure: {FIGDIR}/v4_fig3c_pair_orderings.png")
print("Done (Phase 3c).")
