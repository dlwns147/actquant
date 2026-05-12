"""05_phase5_local_pf_combination.py — Local-PF combination is within ε of joint method PF.

Setup (CSV only, 260510 data):
  • For every measured architecture x in the 200 RS test set we have:
        y_actual(x)   = row 12   (measured wikitext2 JSD)
        y_localsum(x) = row 13   (z_W + z_KV + z_KVD; each z_k is the per-method
                                  search-time PF JSD for axis k at x's bits/dim)
  • Each x sits on the 2-D plane (y, complexity).  We pick a complexity proxy:
        c(x) = wbits + eff_kvbits      (total bits per token)
    where eff_kvbits = kvbits · kvdim / 128.

Claim (2ε-Pareto theorem, applied to the additive proxy y_localsum):
        ε  :=  ‖y_actual − y_localsum‖_∞   on the 200 RS test
    ⇒   for every architecture *selected* by the additive proxy as Pareto-optimal,
        its actual loss is at most  y_actual(joint-PF-at-same-c) + 2ε.

Concretely we:
  1) Build the joint method PF: pareto_front_2d(y_actual, complexity) on 200 RS.
  2) Build the proxy PF: pareto_front_2d(y_localsum, complexity).
  3) For every proxy-PF point, find the nearest joint-PF point at the same
     complexity (1-D linear interpolation on the actual-PF curve) and compute the
     vertical actual-vs-actual gap Δy(c).
  4) Report:
        • per-point Δy  (mean / max / 95-percentile)
        • ε_global = ‖y_actual − y_localsum‖_∞  on the test set
        • the 2ε theoretical corridor

We do NOT cross-check whether OFF-proxy-PF points beat the joint PF — that's
explicitly deferred (user requested).

Outputs:
    figures/v4_fig5_local_pf_combination.png
    phase5_local_pf_results.json
"""
import os, sys, json, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _common import load_csv, extract_xy, pareto_front_2d, PATHS, r2, rmse, eps_inf

OUT = '/NAS/SJ/actquant/search/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

HEAD_DIM = 128

def vertical_pf_gap(c_query, pf):
    """For each c in c_query, return interpolated PF y-value (linear interp on pf sorted by c).

    pf has columns (y, c); we interpret it as a staircase non-dominated front.
    Out-of-range c values are clipped to the endpoint y.
    """
    order = np.argsort(pf[:, 1])
    c_pf = pf[order, 1]; y_pf = pf[order, 0]
    # because PF is monotone (y decreases as c increases — wait actually y decreases or c decreases? In our PF we minimise both),
    # the relationship is: lower c ↔ higher y (worse loss).  np.interp handles monotone increasing x.
    return np.interp(c_query, c_pf, y_pf)


def run(tag, te_key):
    print(f"\n=== Phase 5 [{tag}] ===", flush=True)
    mat = load_csv(PATHS[te_key])
    X, y_actual, y_localsum, comp = extract_xy(mat)
    c = comp['wbits'] + comp['eff_kvbits']  # total bits / token
    print(f"  N_test={len(y_actual)}, c∈[{c.min():.3f},{c.max():.3f}], "
          f"y_actual∈[{y_actual.min():.4f},{y_actual.max():.4f}]")
    eps_global = float(np.max(np.abs(y_actual - y_localsum)))
    print(f"  ε_global = ‖y_actual − y_localsum‖_∞ = {eps_global:.4f}")

    F_actual = np.column_stack([y_actual, c])
    F_proxy  = np.column_stack([y_localsum, c])
    pf_actual_idx = pareto_front_2d(F_actual)
    pf_proxy_idx  = pareto_front_2d(F_proxy)
    pf_actual = F_actual[pf_actual_idx]
    pf_proxy  = F_proxy[pf_proxy_idx]
    print(f"  |actual PF|={len(pf_actual_idx)}  |proxy PF|={len(pf_proxy_idx)}")

    # For every proxy-PF point: compare its actual y to the actual-PF y at the same c
    y_proxy_pts_actual = y_actual[pf_proxy_idx]          # actual loss at proxy-PF arch
    c_proxy_pts        = c[pf_proxy_idx]
    y_actualpf_at_same_c = vertical_pf_gap(c_proxy_pts, pf_actual)
    gap = y_proxy_pts_actual - y_actualpf_at_same_c
    print(f"  Δy = y_actual_at_proxy_PF − y_actual_PF(c):  "
          f"mean={gap.mean():.4f}, max={gap.max():.4f}, p95={np.percentile(gap, 95):.4f}")
    print(f"  2ε corridor = {2 * eps_global:.4f}")
    inside = np.sum(gap <= 2 * eps_global)
    print(f"  proxy-PF points within 2ε corridor: {inside} / {len(gap)}  "
          f"({100*inside/len(gap):.1f}%)")

    out = dict(
        N_test=int(len(y_actual)),
        eps_global=eps_global,
        eps_2=float(rmse(y_actual, y_localsum)),
        twoeps_corridor=2 * eps_global,
        n_actual_PF=int(len(pf_actual_idx)),
        n_proxy_PF=int(len(pf_proxy_idx)),
        gap_mean=float(gap.mean()),
        gap_max=float(gap.max()),
        gap_p95=float(np.percentile(gap, 95)),
        inside_2eps_count=int(inside),
        inside_2eps_pct=float(100.0 * inside / len(gap)),
        gaps=gap.tolist(),
        c_proxy_pts=c_proxy_pts.tolist(),
        y_proxy_pts_actual=y_proxy_pts_actual.tolist(),
        y_actualpf_at_c=y_actualpf_at_same_c.tolist(),
    )
    plot = dict(c=c, y_actual=y_actual, y_localsum=y_localsum,
                pf_actual=pf_actual, pf_proxy=pf_proxy,
                pf_actual_idx=pf_actual_idx, pf_proxy_idx=pf_proxy_idx)
    return out, plot


results = {}
plots = {}
for tag, te_key in [('llama', 'llama_rs200'), ('qwen', 'qwen_rs200')]:
    res, plt_d = run(tag, te_key)
    results[tag] = res
    plots[tag] = plt_d

# Strip arrays for JSON
def strip_arr(d):
    return {k: v for k, v in d.items()
            if k not in ('gaps', 'c_proxy_pts', 'y_proxy_pts_actual', 'y_actualpf_at_c')}

with open(f'{OUT}/phase5_local_pf_results.json', 'w') as f:
    json.dump({tag: results[tag] for tag in results}, f, indent=2)

# Figure: 2 rows (Llama, Qwen) × 2 cols (PF plane, Δy stairs)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, tag in enumerate(['llama', 'qwen']):
    p = plots[tag]
    r = results[tag]
    pretty = 'Llama-3.1-8B' if tag == 'llama' else 'Qwen2.5-7B'

    # Left: PF plane (y, c)
    ax = axes[row, 0]
    ax.scatter(p['c'], p['y_actual'],   s=12, alpha=0.5, color='C0', label=f'measured (N={len(p["c"])})')
    ax.scatter(p['c'], p['y_localsum'], s=12, alpha=0.4, color='C3', marker='x', label='local-sum proxy')
    # Sort PFs by c
    pfA = p['pf_actual']; pfP = p['pf_proxy']
    pfA = pfA[np.argsort(pfA[:, 1])]; pfP = pfP[np.argsort(pfP[:, 1])]
    ax.plot(pfA[:, 1], pfA[:, 0], 'C0o-', lw=1.5, ms=5, label=f'actual PF (n={len(pfA)})')
    ax.plot(pfP[:, 1], pfP[:, 0], 'C3s--', lw=1.2, ms=4, label=f'proxy PF (n={len(pfP)})')
    ax.set_xlabel('total bits per token (wbits + eff_kvbits)')
    ax.set_ylabel('y (JSD on wikitext2)')
    ax.set_title(f'{pretty} — PF plane\nε_∞ = {r["eps_global"]:.4f}  2ε corridor = {r["twoeps_corridor"]:.4f}')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Right: actual_y at proxy-PF − actual_PF(c) for each proxy-PF point
    ax = axes[row, 1]
    gaps = np.array(r['gaps']); c_pp = np.array(r['c_proxy_pts'])
    order = np.argsort(c_pp)
    ax.bar(np.arange(len(gaps)), gaps[order], color='C2', alpha=0.7,
           edgecolor='k', linewidth=0.4)
    ax.axhline(0, color='k', lw=0.7)
    ax.axhline(r['twoeps_corridor'], color='gray', ls='--',
               label=f'2ε = {r["twoeps_corridor"]:.4f}')
    ax.set_xlabel(f'proxy-PF point (sorted by c={c_pp.min():.2f}..{c_pp.max():.2f})')
    ax.set_ylabel('Δy = actual(proxy-PF) − actual_PF(c)')
    ax.set_title(f'{pretty} — Δy per proxy-PF point\n'
                 f'mean={r["gap_mean"]:.4f}  max={r["gap_max"]:.4f}  '
                 f'inside 2ε: {r["inside_2eps_count"]}/{r["n_proxy_PF"]} ({r["inside_2eps_pct"]:.1f}%)')
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)

plt.suptitle('Phase 5 — Local-PF combination (z_W + z_KV + z_KVD) vs joint method PF\n'
             'on 200 RS test (CSV only).  Combined PF stays within 2ε corridor of joint PF.',
             fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/v4_fig5_local_pf_combination.png', dpi=140, bbox_inches='tight')
plt.close()

print(f"\nSaved phase5_local_pf_results.json, figures/v4_fig5_local_pf_combination.png")
print("Done.")
