"""08_check_selection_bias.py — Recompute P(violator) on the existing 100
candidates using 3D-correct Pareto-envelope baseline, and compare ranking
to the 1D-based selection.

Purpose: decide whether to re-run 05_acquire with 3D baseline.

Logic:
  1. Reload acquired 100 records (have ARD-GP predictions μ, σ already).
  2. For each, compute pf_baseline_{1d, 2d, 3d} from the 200 AWQ training samples
     (same as 07_recompute_baseline_3d).
  3. For each baseline, compute gap = f* - μ, P_violator = Φ(gap/σ).
  4. Compare top-K rankings: how many of the 1D-top-100 are also in 3D-top-100,
     after equalising for "in-cloud" availability?

Output:
  Console summary + figure analysis/v3/figures/08_selection_bias.png
"""
import sys, os, json, csv, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

BASE = '/NAS/SJ/actquant/search'
ACQ_JSON  = f'{BASE}/analysis/v3/acquired_offsurface_100.json'
AWQ_3WAY  = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'

def load_csv(p):
    with open(p) as f: rows = [r for r in csv.reader(f) if r]
    M = max(len(r) for r in rows); m = np.full((len(rows), M), np.nan)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            try: m[i, j] = float(v)
            except: pass
    return m

# Load PF (200 AWQ training samples)
mat = load_csv(AWQ_3WAY); v = ~np.isnan(mat[12, :])
HEAD_DIM = 128
y_pf = mat[12, :][v]
wb_pf, kvb_pf, kvd_pf = mat[0,:][v], mat[1,:][v], mat[4,:][v]
eff_pf = kvb_pf * kvd_pf / HEAD_DIM
PF_3D = np.column_stack([y_pf, wb_pf, kvb_pf, kvd_pf])
PF_2D = np.column_stack([y_pf, wb_pf, eff_pf])

def pf_envelope_max(c, pf, cols):
    mask = np.all(pf[:, cols] >= c[None, :], axis=1)
    return float(np.max(pf[mask, 0])) if mask.any() else float('nan')

# 1D PF interpolant on (total_c, y_pf)
sort_idx = np.argsort(wb_pf + eff_pf)
pf_x = (wb_pf + eff_pf)[sort_idx]
pf_y = y_pf[sort_idx]
# Lower envelope (running min from right)
pf_y_mono = np.minimum.accumulate(pf_y[::-1])[::-1]
def pf_interp_1d(c):
    return np.interp(c, pf_x, pf_y_mono, left=pf_y_mono[0], right=pf_y_mono[-1])

# Load acquired 100 (have ARD-GP μ, σ for each)
with open(ACQ_JSON) as f:
    acq = json.load(f)
records = acq['records']
print(f"Loaded {len(records)} candidates")

# Compute baselines + P(violator) under each baseline
out = []
for r in records:
    cw, ckv, ckd, eff = r['wbits'], r['kvbits'], r['kvdim'], r['eff_kvbits']
    base_1d = float(pf_interp_1d(cw + eff))
    base_2d = pf_envelope_max(np.array([cw, eff]), PF_2D, [1, 2])
    base_3d = pf_envelope_max(np.array([cw, ckv, ckd]), PF_3D, [1, 2, 3])
    mu, sig = r['pred_mu'], r['pred_sigma']
    sig = max(sig, 1e-6)
    gap_1d = base_1d - mu
    gap_2d = base_2d - mu
    gap_3d = base_3d - mu
    P_1d = norm.cdf(gap_1d / sig)
    P_2d = norm.cdf(gap_2d / sig) if not np.isnan(gap_2d) else float('nan')
    P_3d = norm.cdf(gap_3d / sig) if not np.isnan(gap_3d) else float('nan')
    out.append({
        'sel_idx': r['sel_idx'], 'src': r['src'],
        'wbits': cw, 'kvbits': ckv, 'kvdim': ckd, 'eff_kvbits': eff,
        'mu': mu, 'sigma': r['pred_sigma'],
        'base_1d': base_1d, 'base_2d': base_2d, 'base_3d': base_3d,
        'gap_1d': gap_1d, 'gap_2d': gap_2d, 'gap_3d': gap_3d,
        'P_1d': float(P_1d), 'P_2d': float(P_2d), 'P_3d': float(P_3d),
        'P_orig': r['prob_violator'],
        'in_cloud_3d': not np.isnan(base_3d),
        'in_cloud_2d': not np.isnan(base_2d),
    })

# Coverage stats
n_in_3d = sum(1 for x in out if x['in_cloud_3d'])
n_in_2d = sum(1 for x in out if x['in_cloud_2d'])
print(f"\nIn-cloud (3D): {n_in_3d}/100")
print(f"In-cloud (2D): {n_in_2d}/100")

# Top-K overlap analysis
K_list = [10, 20, 30, 50]
print("\nTop-K overlap (1D vs 3D ranking):")
print(f"{'K':>4s}  {'1D-top-K':>10s}  {'3D-top-K':>10s}  overlap")
sorted_1d = sorted(out, key=lambda x: -x['P_1d'])
# For 3D, only rank in-cloud (others have nan)
cands_3d = [x for x in out if x['in_cloud_3d']]
sorted_3d = sorted(cands_3d, key=lambda x: -x['P_3d'])

for K in K_list:
    top_1d = set(x['sel_idx'] for x in sorted_1d[:K])
    top_3d = set(x['sel_idx'] for x in sorted_3d[:K])
    overlap = top_1d & top_3d
    print(f"{K:>4d}  {len(top_1d):>10d}  {len(top_3d):>10d}  "
          f"{len(overlap)}/{K}  ({100*len(overlap)/K:.0f}%)")

# Which 1D-top are out-of-cloud (selection bias indicator)
print(f"\nOf 1D-top-25 (=σ-extras + P(violator)-top), out-of-cloud count:")
top25_1d = sorted_1d[:25]
n_ooc = sum(1 for x in top25_1d if not x['in_cloud_3d'])
print(f"  {n_ooc}/25  ({100*n_ooc/25:.0f}%)")
print(f"  (these are wasted budget — cannot be ε-violators in strict Pareto sense)")

# Best 3D candidates (highest 3D-P, in-cloud only)
print(f"\nTop 10 candidates by 3D P(violator) (in-cloud only):")
print(f"  {'sel':>4s} {'P_3d':>6s} {'P_1d':>6s} {'P_orig':>6s} "
      f"{'wbits':>7s} {'kvbits':>7s} {'kvdim':>5s} {'gap_3d':>8s}")
for x in sorted_3d[:10]:
    print(f"  {x['sel_idx']:>4d} {x['P_3d']:>6.3f} {x['P_1d']:>6.3f} "
          f"{x['P_orig']:>6.3f} {x['wbits']:>7.3f} {x['kvbits']:>7.3f} "
          f"{x['kvdim']:>5.0f} {x['gap_3d']:>+8.5f}")

# Figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) P_1d vs P_3d
ax = axes[0]
in_cloud_mask = np.array([x['in_cloud_3d'] for x in out])
P1 = np.array([x['P_1d'] for x in out])
P3 = np.array([x['P_3d'] for x in out])
ax.scatter(P1[in_cloud_mask], P3[in_cloud_mask], s=30, c='steelblue',
           edgecolor='black', linewidth=0.3, label='in-cloud')
ax.scatter(P1[~in_cloud_mask], np.full((~in_cloud_mask).sum(), -0.05),
           s=30, c='red', marker='x', label='out-of-cloud (P_3d undef)')
ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='P_3d = P_1d')
ax.set_xlabel('P_1d (used by acquisition)')
ax.set_ylabel('P_3d (correct Pareto-envelope)')
ax.set_title('(a) P_1d vs P_3d  (red = out-of-cloud waste)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (b) Ranking comparison
ax = axes[1]
# 1D rank vs 3D rank for in-cloud
rank_1d = {x['sel_idx']: i for i, x in enumerate(sorted_1d)}
rank_3d = {x['sel_idx']: i for i, x in enumerate(sorted_3d)}
in_cloud_idx = [x['sel_idx'] for x in out if x['in_cloud_3d']]
r1 = [rank_1d[s] for s in in_cloud_idx]
r3 = [rank_3d[s] for s in in_cloud_idx]
ax.scatter(r1, r3, s=20, c='purple', edgecolor='black', linewidth=0.3, alpha=0.7)
ax.plot([0, 100], [0, 100], 'k--', lw=0.6)
ax.set_xlabel('1D ranking (used by acquisition)')
ax.set_ylabel('3D ranking (correct)')
ax.set_title('(b) ranking shift', fontweight='bold')
ax.grid(True, alpha=0.3)

# (c) Selection in (wbits, eff_kvbits)
ax = axes[2]
wb = np.array([x['wbits'] for x in out])
ek = np.array([x['eff_kvbits'] for x in out])
ax.scatter(wb_pf, eff_pf, s=8, c='lightgray', alpha=0.5, label='200 PF samples')
ax.scatter(wb[in_cloud_mask], ek[in_cloud_mask], s=30, c='steelblue',
           edgecolor='black', linewidth=0.3, label=f'in-cloud (n={in_cloud_mask.sum()})')
ax.scatter(wb[~in_cloud_mask], ek[~in_cloud_mask], s=40, c='red',
           marker='x', linewidth=2, label=f'out-of-cloud (n={(~in_cloud_mask).sum()})')
ax.set_xlabel('wbits'); ax.set_ylabel('eff_kvbits')
ax.set_title('(c) selection in 2D complexity', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle(f'Selection bias check: 1D-based selection → 3D analysis',
             fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = f'{BASE}/analysis/v3/figures/08_selection_bias.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nsaved {fig_path}")
