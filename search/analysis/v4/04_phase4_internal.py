"""04_phase4_internal.py — Internal analysis of the best surrogate (Llama + Qwen).

Best surrogate is taken from Phase 1 (RBF cubic+linear for Llama, RBF cubic+linear for
Qwen; we also run ARD-Matérn-3/2 since the GP exposes length scales directly).

Analyses:
  1) Sobol decomposition (Saltelli pick-freeze, N_base=2048):
        S1_W, S1_KV, S1_KVD, ST, S2 pair, sum(S1), interaction = 1 - sum(S1)
  2) Active subspace via gradient outer product (MC, N=4000):
        eigenvalues + first eigenvector (direction of dominant variation)
  3) ARD-GP length scales l_W, l_KV, l_KVD + noise sigma_n  (only for ARD-GP)
  4) Hoeffding additive ε bounds on 200 RS test set:
        y_add = f_0 + Σ g_k(x_k);  reports add R², ε_∞, ε_2 and
        Var(res_add)/Var(y_te) — small means surrogate is nearly additive.

Outputs (per model: '' for Llama, 'q' suffix for Qwen):
    figures/v4_fig4_sobol.png            v4_fig4q_sobol.png
    figures/v4_fig4_active_subspace.png  v4_fig4q_active_subspace.png
    figures/v4_fig4_hoeffding_add.png    v4_fig4q_hoeffding_add.png
    phase4_internal_results.json         — keyed by model tag
"""
import os, sys, json, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import (load_csv, extract_xy, fit_rbf, fit_ard_gp, get_ard_lengthscales,
                     r2, rmse, eps_inf, PATHS, AXIS_NAMES_3)

OUT = '/NAS/SJ/actquant/search/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze


def hoeffding_add(predict_fn, X_te, bounds, n_mc=8000, h_factor=0.15, seed=0):
    rng = np.random.RandomState(seed)
    X_mc = np.column_stack([rng.uniform(b[0], b[1], n_mc) for b in bounds])
    y_mc = predict_fn(X_mc)
    f0 = float(y_mc.mean())
    out = np.full(len(X_te), f0)
    for k in range(X_te.shape[1]):
        h = X_mc[:, k].std() * h_factor
        diff = X_te[:, k:k+1] - X_mc[:, k][None, :]
        w = np.exp(-0.5 * (diff / h) ** 2)
        g = (w * y_mc).sum(1) / w.sum(1)
        out = out + (g - f0)
    return out, f0


def run(tag, qs_key, te_key, pretty, fig_suffix):
    print(f"\n=== Phase 4 [{tag}] — internal analysis (50 QS train → 200 RS test) ===")
    X_tr, y_tr, _, _ = extract_xy(load_csv(PATHS[qs_key]))
    X_te, y_te, _, _ = extract_xy(load_csv(PATHS[te_key]))

    # Fit both surrogates
    print("\nFitting surrogates (RBF cubic+linear, ARD-Matérn-3/2 +noise)...")
    _, yp_te_rbf, m_rbf = fit_rbf(X_tr, y_tr, X_te, kernel='cubic')
    _, yp_te_gp,  m_gp  = fit_ard_gp(X_tr, y_tr, X_te, kernel='matern32',
                                      with_noise=True, n_restarts=20)

    print(f"  RBF cubic+linear:   R²={r2(y_te, yp_te_rbf):.4f}  "
          f"RMSE={rmse(y_te, yp_te_rbf):.5f}  ε_∞={eps_inf(y_te, yp_te_rbf):.5f}")
    print(f"  ARD-Matérn-3/2:   R²={r2(y_te, yp_te_gp):.4f}  "
          f"RMSE={rmse(y_te, yp_te_gp):.5f}  ε_∞={eps_inf(y_te, yp_te_gp):.5f}")

    ls = get_ard_lengthscales(m_gp, with_noise=True, d=3)
    sigma_n = float(m_gp.kernel_.k2.noise_level)
    print(f"  ARD length scales: l_W={ls[0]:.4f} l_KV={ls[1]:.4f} l_KVD={ls[2]:.4f}   σ_n={sigma_n:.2e}")

    def rbf_predict(X): return m_rbf.predict(X).ravel()
    def gp_predict(X):  return m_gp.predict(X)

    bounds = [[X_tr[:, i].min(), X_tr[:, i].max()] for i in range(3)]

    # 1) Sobol
    print("\nSobol decomposition (Saltelli pick-freeze, N_base=2048)...")
    problem = {'num_vars': 3, 'names': list(AXIS_NAMES_3), 'bounds': bounds}
    samples = saltelli.sample(problem, 2048, calc_second_order=True)
    sobol_per_model = {}
    for name, predict_fn in [('RBF cubic+linear', rbf_predict),
                             ('ARD-Matérn-3/2', gp_predict)]:
        yvals = predict_fn(samples)
        Si = sobol_analyze.analyze(problem, yvals, calc_second_order=True,
                                    print_to_console=False)
        sobol_per_model[name] = dict(
            S1={n: float(Si['S1'][i])      for i, n in enumerate(AXIS_NAMES_3)},
            ST={n: float(Si['ST'][i])      for i, n in enumerate(AXIS_NAMES_3)},
            S1_conf={n: float(Si['S1_conf'][i]) for i, n in enumerate(AXIS_NAMES_3)},
            ST_conf={n: float(Si['ST_conf'][i]) for i, n in enumerate(AXIS_NAMES_3)},
            S2={f'{a},{b}': float(Si['S2'][i, j]) for (i, j, a, b) in
                [(0, 1, 'W', 'KV'), (0, 2, 'W', 'KVD'), (1, 2, 'KV', 'KVD')]},
            sum_S1=float(sum(Si['S1'])),
            interaction=float(1 - sum(Si['S1'])),
        )
        s1_str = ' '.join(f'{Si["S1"][i]:.3f}' for i in range(3))
        sumS1 = float(sum(Si['S1']))
        print(f"  {name:18s}: S1=[{s1_str}]  ΣS1={sumS1:.3f}  interaction={1-sumS1:.3f}")

    # 2) Active subspace
    print("\nActive subspace (MC gradient outer product, N=4000)...")
    rng = np.random.RandomState(0)
    N_MC = 4000
    X_mc = np.column_stack([rng.uniform(b[0], b[1], N_MC) for b in bounds])
    h = (X_mc.max(0) - X_mc.min(0)) * 1e-3
    asubs_per_model = {}
    for name, predict_fn in [('RBF cubic+linear', rbf_predict),
                             ('ARD-Matérn-3/2', gp_predict)]:
        G = np.zeros((N_MC, 3))
        for k in range(3):
            dx = np.zeros_like(X_mc); dx[:, k] = h[k]
            G[:, k] = (predict_fn(X_mc + dx) - predict_fn(X_mc - dx)) / (2 * h[k])
        C = G.T @ G / N_MC
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
        asubs_per_model[name] = dict(
            eigvals=eigvals.tolist(),
            ratio_2_over_1=float(eigvals[1] / eigvals[0]),
            ratio_3_over_1=float(eigvals[2] / eigvals[0]),
            first_eigvec={n: float(eigvecs[i, 0]) for i, n in enumerate(AXIS_NAMES_3)},
        )
        print(f"  {name:18s}: eig=[{eigvals[0]:.3e}, {eigvals[1]:.3e}, {eigvals[2]:.3e}]  "
              f"λ_2/λ_1={eigvals[1]/eigvals[0]:.3f}  λ_3/λ_1={eigvals[2]/eigvals[0]:.3f}")
        print(f"        first_eigvec ≈ {eigvecs[:, 0]}")

    # 3) Hoeffding additive
    print("\nHoeffding additive collapse on 200 RS test...")
    hoeff_per_model = {}
    for name, predict_fn, yp_full in [('RBF cubic+linear', rbf_predict, yp_te_rbf),
                                      ('ARD-Matérn-3/2', gp_predict, yp_te_gp)]:
        y_add, f0 = hoeffding_add(predict_fn, X_te, bounds)
        full_r2  = r2(y_te, yp_full);     full_rm = rmse(y_te, yp_full);  full_ei = eps_inf(y_te, yp_full)
        add_r2   = r2(y_te, y_add);       add_rm  = rmse(y_te, y_add);    add_ei  = eps_inf(y_te, y_add)
        var_resid_add = float(np.var(y_te - y_add))
        var_y         = float(np.var(y_te))
        hoeff_per_model[name] = dict(
            full=dict(R2=full_r2, RMSE=full_rm, eps_inf=full_ei),
            add=dict(R2=add_r2,  RMSE=add_rm,  eps_inf=add_ei),
            var_resid_over_var_y=var_resid_add / max(var_y, 1e-30),
            f0=f0,
        )
        print(f"  {name:18s}: full ε_∞={full_ei:.4f}  add ε_∞={add_ei:.4f}  "
              f"(add ε_∞ / full ε_∞ = {add_ei/full_ei:.2f}×)")
        print(f"  {' '*18}  Var(y_te - y_add)/Var(y_te) = {var_resid_add/var_y:.4f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    # Sobol bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    xpos = np.arange(3)
    for ax, name in zip(axes, sobol_per_model):
        d = sobol_per_model[name]
        S1 = [d['S1'][a] for a in AXIS_NAMES_3]
        ST = [d['ST'][a] for a in AXIS_NAMES_3]
        ax.bar(xpos - 0.2, S1, 0.36, label='S1 main')
        ax.bar(xpos + 0.2, ST, 0.36, label='ST total')
        ax.set_xticks(xpos); ax.set_xticklabels(AXIS_NAMES_3)
        ax.set_title(f"{name}\nΣS1={d['sum_S1']:.3f}  "
                     f"interaction={d['interaction']:.3f}", fontsize=10)
        ax.set_ylabel('Sobol index'); ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.suptitle(f'Phase 4 — Sobol decomposition ({pretty}, 200 RS test bounds)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/v4_fig4{fig_suffix}_sobol.png', dpi=140, bbox_inches='tight')
    plt.close()

    # Active subspace
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, asubs_per_model):
        d = asubs_per_model[name]
        eig = d['eigvals']; rel = [e / eig[0] for e in eig]
        ax.bar([0, 1, 2], rel, color=['C0', 'C0', 'C0'])
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['λ₁', 'λ₂', 'λ₃'])
        ax.set_yscale('log')
        ax.set_title(f"{name}\nλ₂/λ₁={d['ratio_2_over_1']:.3f}  "
                     f"first_eigvec≈[{d['first_eigvec']['W']:+.2f},"
                     f"{d['first_eigvec']['KV']:+.2f},"
                     f"{d['first_eigvec']['KVD']:+.2f}]", fontsize=9)
        ax.set_ylabel('eigenvalue (normalized to λ₁)')
        ax.grid(alpha=0.3, axis='y', which='both')
    plt.suptitle(f'Phase 4 — Active subspace ({pretty})', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/v4_fig4{fig_suffix}_active_subspace.png',
                dpi=140, bbox_inches='tight')
    plt.close()

    # Hoeffding
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, hoeff_per_model):
        d = hoeff_per_model[name]
        ei_full = d['full']['eps_inf']; ei_add = d['add']['eps_inf']
        R2_full = d['full']['R2']; R2_add = d['add']['R2']
        ax.bar(['ε_∞ full', 'ε_∞ add'], [ei_full, ei_add], color=['C0', 'C3'])
        ax.set_title(f"{name}\nfull R²={R2_full:.4f} → add R²={R2_add:.4f}  "
                     f"(add ε_∞/full = {ei_add/ei_full:.2f}×)\n"
                     f"Var(res_add)/Var(y_te)={d['var_resid_over_var_y']:.3f}",
                     fontsize=9)
        ax.set_ylabel('ε_∞'); ax.grid(alpha=0.3, axis='y')
    plt.suptitle(f'Phase 4 — Hoeffding additive ε bounds ({pretty}, 200 RS test)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/v4_fig4{fig_suffix}_hoeffding_add.png',
                dpi=140, bbox_inches='tight')
    plt.close()

    return dict(
        sobol=sobol_per_model,
        active_subspace=asubs_per_model,
        hoeffding=hoeff_per_model,
        ard_gp=dict(length_scales={n: float(ls[i]) for i, n in enumerate(AXIS_NAMES_3)},
                    sigma_n=sigma_n),
        train_size=int(len(y_tr)), test_size=int(len(y_te)),
        bounds={n: bounds[i] for i, n in enumerate(AXIS_NAMES_3)},
    )


all_results = {}
for tag, qs_key, te_key, pretty, fig_suffix in [
    ('llama', 'llama_qs50', 'llama_rs200', 'Llama-3.1-8B', ''),
    ('qwen',  'qwen_qs50',  'qwen_rs200',  'Qwen2.5-7B',   'q'),
]:
    all_results[tag] = run(tag, qs_key, te_key, pretty, fig_suffix)

with open(f'{OUT}/phase4_internal_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved phase4_internal_results.json + figures/v4_fig4*.png")
print("Done.")
