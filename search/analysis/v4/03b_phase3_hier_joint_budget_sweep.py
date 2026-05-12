"""03b_phase3_hier_joint_budget_sweep.py — Hierarchical vs joint surrogate
with *budget-matched* sample-count sweep.

Constraint: hierarchical total = joint total = N. Sweep the (n_pair, n_3way)
split, with n_pair + n_3way = N.

Hierarchical decomposition (per pair direction):
    y(x_W, x_KV, x_KVD) ≈ f_pair(x_a, x_b)  +  g(x_c)
where (a, b) is one of {WK, KD, WD} and c is the residual axis.

Stage 1 — Pair surrogate f_pair:   RBF tps+linear on n_pair samples drawn from
the corresponding 2-axis 260510 CSV (z_a, z_b, y_pair_measured).
Stage 2 — Residual g(x_c):          fit on the 3-way QS pool:
    r_3way = y_3way − f_pair(z_a_3way, z_b_3way)        (n_3way rows)
    g = poly1d / RBF-1d on (z_c_3way, r_3way)
Test prediction:    y_pred(x) = f_pair(x_a, x_b) + g(x_c)

Joint baseline:  RBF tps+linear on n_3way' = N 3-way samples (same total budget).

Inputs:
    260510 2-axis CSVs (llama_wk 29 cols, llama_kvkd 30, llama_wkd 31)
    260510 3-way QS CSV (llama_qs50, 50 cols)
    260510 3-way RS200 test (llama_rs200, 200 cols)

We sweep the split for total budget N=50 (matches the joint Phase 1 setup).
For each (pair_direction, n_pair, n_3way) we run 10 seeded resamples.

NOTE: 2-axis CSVs only exist for Llama in 260510, so this analysis is Llama-only.
For Qwen the same workflow needs newly-collected 2-axis evaluations.

Outputs:
    figures/v4_fig3b_hier_joint_sweep.png   — R² heatmap and curves
    phase3b_hier_joint_results.json
"""
import os, sys, json, time, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import (load_csv, extract_xy, fit_rbf, r2, rmse, eps_inf, PATHS)
from predictor.rbf import RBF as PySOTRBF

OUT = '/NAS/SJ/actquant/search/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

# 2-axis pair files (Llama only — 260510 doesn't have Qwen 2-axis runs)
# col0 = first z, col1 = second z; pair axis tuple naming maps to 3-way axes:
#   wk   : (W, KV)        residual = KVD (axis index 2 in 3-way)
#   kvkd : (KV, KVD)      residual = W   (axis index 0)
#   wkd  : (W, KVD)       residual = KV  (axis index 1)
PAIRS = {
    'WK':  dict(file_key='llama_wk',   pair_axes=(0, 1), res_axis=2,
                pair_pretty='(z_W, z_KV)',  res_pretty='z_KVD'),
    'KD':  dict(file_key='llama_kvkd', pair_axes=(1, 2), res_axis=0,
                pair_pretty='(z_KV, z_KVD)', res_pretty='z_W'),
    'WD':  dict(file_key='llama_wkd',  pair_axes=(0, 2), res_axis=1,
                pair_pretty='(z_W, z_KVD)', res_pretty='z_KV'),
}

TOTAL_BUDGET = 50      # match joint Phase-1 budget
SEEDS = 10             # paired bootstrap seeds for the budget sweep
RESIDUAL_DEG = 2       # polynomial degree for residual stage (use poly to keep stage-2 simple)


def fit_pair_rbf(X2, y, lb_ext, ub_ext):
    m = PySOTRBF(kernel='tps', tail='linear', lb=lb_ext, ub=ub_ext)
    m.fit(X2, y)
    return m


def fit_residual_poly(x_c_train, r_train, deg=RESIDUAL_DEG):
    """Polynomial fit on residual; falls back to mean if n_3way < deg+1."""
    if len(x_c_train) < deg + 1:
        # not enough points — constant residual
        return np.poly1d([float(np.mean(r_train))])
    return np.poly1d(np.polyfit(x_c_train, r_train, deg))


def hier_predict(X_te, pair_axes, res_axis, pair_model, residual_poly):
    Xp_te = X_te[:, list(pair_axes)]
    xc_te = X_te[:, res_axis]
    yp_pair = pair_model.predict(Xp_te).ravel()
    yp_res  = residual_poly(xc_te)
    return yp_pair + yp_res


def joint_rbf_predict(X_tr, y_tr, X_te, lb_ext, ub_ext):
    m = PySOTRBF(kernel='tps', tail='linear', lb=lb_ext, ub=ub_ext)
    m.fit(X_tr, y_tr)
    return m.predict(X_te).ravel()


def run_one_split(pair_cfg, X_pair, y_pair, X3, y3, X_te, y_te,
                  n_pair, n_3way, seed, lb_pair_ext, ub_pair_ext,
                  lb_3_ext, ub_3_ext, lb_joint_ext, ub_joint_ext):
    """Returns dict(hier=metrics, joint=metrics) for one (n_pair, n_3way, seed)."""
    rng = np.random.RandomState(seed)
    pair_axes = pair_cfg['pair_axes']
    res_axis  = pair_cfg['res_axis']

    # Subsample pair pool
    if n_pair == 0:
        # no pair data — hier degenerates to "constant 0 pair + fit y on z_c"
        pair_model = None
        # residual = y_3way - 0, fit poly on z_c_3way
        idx_3 = rng.choice(len(y3), n_3way, replace=False) if n_3way < len(y3) \
                else np.arange(len(y3))
        Xtr3 = X3[idx_3]; ytr3 = y3[idx_3]
        residual_poly = fit_residual_poly(Xtr3[:, res_axis], ytr3)
        yp_te = residual_poly(X_te[:, res_axis])
    else:
        idx_pair = rng.choice(len(y_pair), min(n_pair, len(y_pair)), replace=False)
        Xp_tr = X_pair[idx_pair]; yp_tr = y_pair[idx_pair]
        pair_model = fit_pair_rbf(Xp_tr, yp_tr, lb_pair_ext, ub_pair_ext)

        if n_3way == 0:
            # no residual fit possible; assume residual = 0
            yp_te = pair_model.predict(X_te[:, list(pair_axes)]).ravel()
        else:
            idx_3 = rng.choice(len(y3), n_3way, replace=False) if n_3way < len(y3) \
                    else np.arange(len(y3))
            Xtr3 = X3[idx_3]; ytr3 = y3[idx_3]
            r_tr = ytr3 - pair_model.predict(Xtr3[:, list(pair_axes)]).ravel()
            residual_poly = fit_residual_poly(Xtr3[:, res_axis], r_tr)
            yp_te = hier_predict(X_te, pair_axes, res_axis, pair_model, residual_poly)

    hier = dict(r2=r2(y_te, yp_te), rmse=rmse(y_te, yp_te), eps_inf=eps_inf(y_te, yp_te))

    # Joint baseline at the same TOTAL budget = n_pair + n_3way
    N_joint = n_pair + n_3way
    if N_joint == 0:
        joint = dict(r2=float('nan'), rmse=float('nan'), eps_inf=float('nan'))
    else:
        idx_j = rng.choice(len(y3), min(N_joint, len(y3)), replace=False) \
                if N_joint < len(y3) else np.arange(len(y3))
        Xtr_j = X3[idx_j]; ytr_j = y3[idx_j]
        yp_te_j = joint_rbf_predict(Xtr_j, ytr_j, X_te, lb_joint_ext, ub_joint_ext)
        joint = dict(r2=r2(y_te, yp_te_j), rmse=rmse(y_te, yp_te_j),
                     eps_inf=eps_inf(y_te, yp_te_j))
    return dict(hier=hier, joint=joint)


# ─── Load Llama data ────────────────────────────────────────────────────────
print("=" * 100); print("Phase 3b — hierarchical vs joint with budget-matched sweep (Llama)")
print("=" * 100, flush=True)

# 3-way data: QS50 train pool (50 measurements available)
X3,  y3,  _, _ = extract_xy(load_csv(PATHS['llama_qs50']),  n_axes=3)
X_te, y_te, _, _ = extract_xy(load_csv(PATHS['llama_rs200']), n_axes=3)
print(f"  3-way QS pool: N={len(y3)}   test: N={len(y_te)}", flush=True)

# Test-set bounds for joint RBF
lb_joint_ext = np.minimum(X3.min(0), X_te.min(0))
ub_joint_ext = np.maximum(X3.max(0), X_te.max(0))

# Per-pair: load 2-axis data + extended bounds for the pair RBF
pair_data = {}
for tag, cfg in PAIRS.items():
    X_p, y_p, _, _ = extract_xy(load_csv(PATHS[cfg['file_key']]), n_axes=2)
    pair_axes = cfg['pair_axes']
    # Pair RBF input box = union of pair train + 3-way pair-axes + test pair-axes
    X_p_3way_proj = X3[:, list(pair_axes)]
    X_p_te_proj   = X_te[:, list(pair_axes)]
    lb_pair_ext = np.minimum.reduce([X_p.min(0), X_p_3way_proj.min(0), X_p_te_proj.min(0)])
    ub_pair_ext = np.maximum.reduce([X_p.max(0), X_p_3way_proj.max(0), X_p_te_proj.max(0)])
    pair_data[tag] = dict(X=X_p, y=y_p, lb=lb_pair_ext, ub=ub_pair_ext, cfg=cfg)
    print(f"  pair {tag} {cfg['pair_pretty']}: N_pair={len(y_p)}  pair_lb={lb_pair_ext}  pair_ub={ub_pair_ext}", flush=True)

# Splits: total = TOTAL_BUDGET=50; max n_pair capped by pair pool size (29 for WK, 30 for KD, 31 for WD)
# Choose 6 splits common across all pairs (use min pair pool = 29 to cap)
MAX_NPAIR = min(len(pd['y']) for pd in pair_data.values())     # = 29 for WK
print(f"\n  total budget N = {TOTAL_BUDGET}; max n_pair across pairs = {MAX_NPAIR}", flush=True)
SPLITS = [(0, 50), (5, 45), (10, 40), (15, 35), (20, 30), (25, 25), (MAX_NPAIR, 50 - MAX_NPAIR)]
print(f"  splits (n_pair, n_3way): {SPLITS}", flush=True)

# ─── Sweep ──────────────────────────────────────────────────────────────────
results = {tag: {f"{n_p}_{n_3}": {'hier': [], 'joint': []} for (n_p, n_3) in SPLITS}
           for tag in PAIRS}
t0 = time.time()
for tag, pd in pair_data.items():
    print(f"\n--- pair {tag} {pd['cfg']['pair_pretty']} (residual on {pd['cfg']['res_pretty']}) ---", flush=True)
    for (n_p, n_3) in SPLITS:
        # n_pair deterministic at MAX (only 1 seed makes sense)
        # for stochastic n_pair < pool: average over SEEDS
        n_seeds = 1 if n_p == 0 or n_p >= len(pd['y']) else SEEDS
        # Same for n_3way determinism
        for seed in range(n_seeds):
            r = run_one_split(pd['cfg'], pd['X'], pd['y'], X3, y3, X_te, y_te,
                              n_p, n_3, seed,
                              pd['lb'], pd['ub'],
                              X3.min(0), X3.max(0),
                              lb_joint_ext, ub_joint_ext)
            results[tag][f"{n_p}_{n_3}"]['hier'].append(r['hier'])
            results[tag][f"{n_p}_{n_3}"]['joint'].append(r['joint'])
        h = results[tag][f"{n_p}_{n_3}"]['hier']
        j = results[tag][f"{n_p}_{n_3}"]['joint']
        h_r2 = [x['r2'] for x in h]; j_r2 = [x['r2'] for x in j]
        print(f"  n_pair={n_p:>2d}  n_3way={n_3:>2d}  seeds={n_seeds:>2d}  "
              f"hier R²={np.median(h_r2):+.4f} [{np.percentile(h_r2,10):+.3f}, {np.percentile(h_r2,90):+.3f}]   "
              f"joint R²={np.median(j_r2):+.4f}", flush=True)
print(f"\nsweep wall: {time.time()-t0:.1f}s", flush=True)

# ─── Summarise ──────────────────────────────────────────────────────────────
def summary(arr):
    arr = np.asarray(arr, dtype=float)
    return dict(median=float(np.median(arr)), p10=float(np.percentile(arr, 10)),
                p90=float(np.percentile(arr, 90)), mean=float(np.mean(arr)),
                std=float(np.std(arr)), n=int(len(arr)))

summed = {}
for tag in PAIRS:
    summed[tag] = {}
    for (n_p, n_3) in SPLITS:
        key = f"{n_p}_{n_3}"
        h_runs = results[tag][key]['hier']
        j_runs = results[tag][key]['joint']
        summed[tag][key] = {
            'n_pair': n_p, 'n_3way': n_3, 'total': n_p + n_3,
            'hier':  {m: summary([r[m] for r in h_runs]) for m in ('r2', 'rmse', 'eps_inf')},
            'joint': {m: summary([r[m] for r in j_runs]) for m in ('r2', 'rmse', 'eps_inf')},
        }

with open(f'{OUT}/phase3b_hier_joint_results.json', 'w') as f:
    json.dump(dict(total_budget=TOTAL_BUDGET, splits=SPLITS, results=summed,
                   note="Llama-only — 260510 has no Qwen 2-axis CSVs."), f, indent=2)
print(f"\nsaved phase3b_hier_joint_results.json", flush=True)

# ─── Figure: R² vs n_3way fraction for each pair direction, plus joint reference
fig, axes = plt.subplots(1, 3, figsize=(17, 5.0), sharey=True)
for ax, tag in zip(axes, PAIRS):
    xs = [s[1] / TOTAL_BUDGET for s in SPLITS]    # n_3way / N
    h_med = [summed[tag][f"{n_p}_{n_3}"]['hier']['r2']['median']  for (n_p, n_3) in SPLITS]
    h_p10 = [summed[tag][f"{n_p}_{n_3}"]['hier']['r2']['p10']     for (n_p, n_3) in SPLITS]
    h_p90 = [summed[tag][f"{n_p}_{n_3}"]['hier']['r2']['p90']     for (n_p, n_3) in SPLITS]
    j_med = [summed[tag][f"{n_p}_{n_3}"]['joint']['r2']['median'] for (n_p, n_3) in SPLITS]
    ax.plot(xs, h_med, 'C0o-', lw=1.6, label=f'Hier-{tag} (pair {PAIRS[tag]["pair_pretty"]})')
    ax.fill_between(xs, h_p10, h_p90, color='C0', alpha=0.15)
    ax.plot(xs, j_med, 'C3s--', lw=1.3, label='Joint RBF on N 3-way samples')
    ax.axhline(0.99, color='gray', lw=0.6, ls=':')
    ax.set_xlabel('n_3way / total budget'); ax.set_ylabel('R² on 200 RS test')
    ax.set_title(f'Hier-{tag} vs joint at total = {TOTAL_BUDGET}\n'
                 f'(seeds={SEEDS} for stochastic splits)')
    ax.set_ylim(-0.5, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
plt.suptitle('Phase 3b — Hierarchical (pair+residual) vs joint surrogate, budget-matched (Llama)',
             fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/v4_fig3b_hier_joint_sweep.png', dpi=140, bbox_inches='tight'); plt.close()
print(f"saved figure: {FIGDIR}/v4_fig3b_hier_joint_sweep.png", flush=True)
print("Done (Phase 3b).")
