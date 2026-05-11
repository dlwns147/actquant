"""09_per_axis_breakdown.py — Per-axis (wbits / kvbits / kvdim) breakdown
of the 125 falsification evaluations.

Outputs:
  analysis/v3/per_axis_breakdown.txt
  figures/09_per_axis_breakdown.png  (3×3 grid: residual / pred-error / count by each axis)
  figures/09_2d_heatmaps.png         (3 pairwise 2D heatmaps of residual)
"""
import sys, os, json, csv, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/NAS/SJ/actquant/search'
EVAL = f'{BASE}/analysis/v3/eval_offsurface_baseline3d.json'  # has pf_baseline_3d
FIG_DIR = f'{BASE}/analysis/v3/figures'
OUT_TXT = f'{BASE}/analysis/v3/per_axis_breakdown.txt'

with open(EVAL) as f:
    results = json.load(f)['results']
print(f"Loaded {len(results)} candidates")

# Extract arrays
wb   = np.array([r['wbits']    for r in results])
kvb  = np.array([r['kvbits']   for r in results])
kvd  = np.array([r['kvdim']    for r in results])
y    = np.array([r['y_actual'] for r in results])
mu   = np.array([r['pred_mu']  for r in results])
sig  = np.array([r['pred_sigma'] for r in results])
P    = np.array([r['prob_violator'] for r in results])
base = np.array([r['pf_baseline_3d'] for r in results])
src  = np.array([r['src'] for r in results])
resid_pf  = y - base                # vs Cartesian PF
resid_pred= y - mu                  # vs ARD-GP prediction

# ─── Per-axis bin breakdown ─────────────────────────────────────────────────
def bin_breakdown(values, label, n_bins=5):
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    rows = []
    for b in range(n_bins):
        mask = (values >= edges[b]) & (values < edges[b+1])
        n = int(mask.sum())
        if n == 0: continue
        rows.append({
            'label': label,
            'bin': b,
            'range': (float(edges[b]), float(edges[b+1])),
            'n': n,
            'resid_pf_min':    float(resid_pf[mask].min()),
            'resid_pf_mean':   float(resid_pf[mask].mean()),
            'resid_pf_max':    float(resid_pf[mask].max()),
            'resid_pred_mean': float(resid_pred[mask].mean()),
            'resid_pred_p95':  float(np.percentile(np.abs(resid_pred[mask]), 95)),
            'P_mean':          float(P[mask].mean()),
            'viol_005_count':  int(((resid_pf < -0.005) & mask).sum()),
            'viol_01_count':   int(((resid_pf < -0.01)  & mask).sum()),
        })
    return rows

print("\nPer-axis breakdown (5 quantile bins each):")
breakdowns = {
    'wbits': bin_breakdown(wb, 'wbits'),
    'kvbits': bin_breakdown(kvb, 'kvbits'),
    'kvdim': bin_breakdown(kvd, 'kvdim'),
}

# Save text summary
lines = []
lines.append(f"Per-axis breakdown of n={len(results)} evaluations")
lines.append(f"=" * 80)
lines.append(f"")
for axis_label, rows in breakdowns.items():
    lines.append(f"\n── by {axis_label} ──")
    lines.append(f"  {'bin':>3s} {'range':>20s} {'n':>4s} "
                 f"{'resid_pf':>22s} {'resid_pred':>20s} "
                 f"{'P':>6s} {'viol':>10s}")
    lines.append(f"      {'min  mean  max'.rjust(20)}      {'mean    p95(|.|)'.rjust(18)}        {'.005 / .01'.rjust(8)}")
    for r in rows:
        rng = f"[{r['range'][0]:.2f},{r['range'][1]:.2f}]"
        lines.append(
            f"  {r['bin']:>3d} {rng:>20s} {r['n']:>4d} "
            f"{r['resid_pf_min']:+6.3f} {r['resid_pf_mean']:+6.3f} {r['resid_pf_max']:+6.3f}  "
            f"{r['resid_pred_mean']:+6.3f}  {r['resid_pred_p95']:.3f}   "
            f"{r['P_mean']:.3f}   {r['viol_005_count']}/{r['viol_01_count']}"
        )

# ─── Pearson correlation: resid vs each axis ────────────────────────────────
def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float('nan')

lines.append(f"\n── Pearson correlations (residual vs axis) ──")
lines.append(f"  axis    corr(resid_pf, axis)   corr(resid_pred, axis)")
for ax_lbl, ax_vals in [('wbits', wb), ('kvbits', kvb), ('kvdim', kvd)]:
    lines.append(f"  {ax_lbl:8s} {corr(resid_pf, ax_vals):+.3f}                {corr(resid_pred, ax_vals):+.3f}")

# ─── Source breakdown (P_3d vs sigma vs lowP_control) ───────────────────────
lines.append(f"\n── by source ──")
for s in ('P_3d', 'sigma|P_3d>1%', 'lowP_control'):
    mask = src == s
    n = int(mask.sum())
    if n == 0: continue
    lines.append(f"  {s:>14s}: n={n:3d}  "
                 f"resid_pf mean={resid_pf[mask].mean():+.4f}  "
                 f"min={resid_pf[mask].min():+.4f}  "
                 f"P_mean={P[mask].mean():.3f}  "
                 f"violations(ε=0.01): {int(((resid_pf < -0.01) & mask).sum())}/{n}")

with open(OUT_TXT, 'w') as f: f.write('\n'.join(lines))
print('\n' + '\n'.join(lines))

# ─── Figure: 3×3 panel (per-axis: resid_pf scatter, resid_pred scatter, hist) ──
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for col, (ax_label, ax_vals) in enumerate([('wbits', wb), ('kvbits', kvb), ('kvdim', kvd)]):
    # Row 0: resid_pf vs axis (scatter), color by source
    ax = axes[0, col]
    for s, c in [('P_3d', 'red'), ('sigma|P_3d>1%', 'orange'), ('lowP_control', 'blue')]:
        mask = src == s
        if mask.any():
            ax.scatter(ax_vals[mask], resid_pf[mask], s=22, c=c, edgecolor='black',
                       linewidth=0.3, alpha=0.7, label=s)
    ax.axhline(0, color='black', lw=0.6)
    ax.axhline(-0.01, color='red', ls='--', lw=0.7, label='ε=−0.01')
    ax.set_xlabel(ax_label); ax.set_ylabel('y_actual − f*_3D (resid_pf)')
    ax.set_title(f'(a{col}) PF residual vs {ax_label}', fontweight='bold')
    if col == 0: ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Row 1: resid_pred vs axis (predictor calibration)
    ax = axes[1, col]
    for s, c in [('P_3d', 'red'), ('sigma|P_3d>1%', 'orange'), ('lowP_control', 'blue')]:
        mask = src == s
        if mask.any():
            ax.scatter(ax_vals[mask], resid_pred[mask], s=22, c=c, edgecolor='black',
                       linewidth=0.3, alpha=0.7)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel(ax_label); ax.set_ylabel('y_actual − μ_pred (predictor error)')
    ax.set_title(f'(b{col}) Predictor residual vs {ax_label}', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Row 2: histogram of axis values (count distribution)
    ax = axes[2, col]
    ax.hist(ax_vals, bins=15, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel(ax_label); ax.set_ylabel('count')
    ax.set_title(f'(c{col}) Sample distribution by {ax_label}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Per-axis breakdown of n={len(results)} falsification evaluations',
             fontweight='bold', y=1.0)
plt.tight_layout()
fig_path = f'{FIG_DIR}/09_per_axis_breakdown.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nsaved {fig_path}")

# ─── 2D heatmaps: resid by pairwise axis bins ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

def heatmap_pairwise(ax, x_vals, y_vals, x_lbl, y_lbl, cell_metric='mean'):
    n_bins_x, n_bins_y = 5, 5
    x_edges = np.quantile(x_vals, np.linspace(0, 1, n_bins_x + 1))
    y_edges = np.quantile(y_vals, np.linspace(0, 1, n_bins_y + 1))
    x_edges[0] -= 1e-9; x_edges[-1] += 1e-9
    y_edges[0] -= 1e-9; y_edges[-1] += 1e-9
    grid = np.full((n_bins_x, n_bins_y), np.nan)
    counts = np.zeros((n_bins_x, n_bins_y), dtype=int)
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            mask = ((x_vals >= x_edges[i]) & (x_vals < x_edges[i+1]) &
                    (y_vals >= y_edges[j]) & (y_vals < y_edges[j+1]))
            counts[i, j] = mask.sum()
            if mask.any():
                if cell_metric == 'mean':
                    grid[i, j] = resid_pf[mask].mean()
                elif cell_metric == 'min':
                    grid[i, j] = resid_pf[mask].min()
    im = ax.imshow(grid.T, origin='lower', cmap='RdBu_r',
                   vmin=-max(0.01, np.nanmax(np.abs(grid))),
                   vmax= max(0.01, np.nanmax(np.abs(grid))),
                   aspect='auto')
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            v = grid[i, j]; n = counts[i, j]
            if n > 0:
                ax.text(i, j, f'{v:+.2f}\nn={n}', ha='center', va='center',
                        color='black', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.04, label=f'resid_pf {cell_metric}')
    # tick labels = bin centers
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    ax.set_xticks(range(n_bins_x)); ax.set_xticklabels([f'{v:.1f}' for v in xc])
    ax.set_yticks(range(n_bins_y)); ax.set_yticklabels([f'{v:.0f}' if y_lbl=='kvdim' else f'{v:.1f}' for v in yc])
    ax.set_xlabel(x_lbl); ax.set_ylabel(y_lbl)
    ax.set_title(f'resid_pf {cell_metric}: {x_lbl} × {y_lbl}', fontweight='bold')

heatmap_pairwise(axes[0], wb,  kvb, 'wbits', 'kvbits', cell_metric='mean')
heatmap_pairwise(axes[1], wb,  kvd, 'wbits', 'kvdim',  cell_metric='mean')
heatmap_pairwise(axes[2], kvb, kvd, 'kvbits','kvdim',  cell_metric='mean')

plt.suptitle(f'Pairwise 2D heatmaps of mean PF residual (n={len(results)})',
             fontweight='bold', y=1.02)
plt.tight_layout()
fig_path2 = f'{FIG_DIR}/09_2d_heatmaps.png'
plt.savefig(fig_path2, dpi=150, bbox_inches='tight'); plt.close()
print(f"saved {fig_path2}")
print("Done.")
