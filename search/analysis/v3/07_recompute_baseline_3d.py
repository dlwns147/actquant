"""07_recompute_baseline_3d.py — Re-analyse falsification with proper 3D
Pareto-envelope baseline in (wbits, kvbits, kvdim) instead of 1D total_c.

Reads:
  analysis/v3/eval_offsurface_100.json   (AWQ-measured y_actual per candidate)
  AWQ_3WAY existing 200 training samples (ground-truth PF in 3D)

Recomputes:
  pf_baseline_3d(c_w, c_kv, c_kd) = max{y_PF : PF point with c_w'≥c_w
                                                       AND c_kv'≥c_kv
                                                       AND c_kd'≥c_kd}
  (highest loss PF point at no-better complexity in any axis ⇒ standard
   Pareto-domination threshold; candidate ε-violates iff y_actual < this − ε)

Writes:
  analysis/v3/eval_offsurface_baseline3d.{json,csv,txt}
  figures/07_baseline3d_overview.png

Run:
  python analysis/v3/07_recompute_baseline_3d.py
"""
import sys, os, json, csv, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as scipy_beta

from utils.func import get_net_info

BASE = '/NAS/SJ/actquant/search'
EVAL_JSON = f'{BASE}/analysis/v3/eval_offsurface_100.json'
AWQ_3WAY  = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'

OUT_JSON = f'{BASE}/analysis/v3/eval_offsurface_baseline3d.json'
OUT_CSV  = f'{BASE}/analysis/v3/eval_offsurface_baseline3d.csv'
OUT_TXT  = f'{BASE}/analysis/v3/eval_offsurface_baseline3d.txt'
FIG      = f'{BASE}/analysis/v3/figures/07_baseline3d_overview.png'

# ─── Load existing AWQ training set (200 Cartesian PF tuples) ───────────────
def load_csv(path):
    with open(path) as f: rows = [r for r in csv.reader(f) if r]
    max_cols = max(len(r) for r in rows)
    mat = np.full((len(rows), max_cols), np.nan)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            try: mat[i, j] = float(v)
            except: pass
    return mat

print("Loading 200 AWQ training samples (ground-truth PF) ...")
mat = load_csv(AWQ_3WAY); N0 = mat.shape[1]
y_train = mat[12, :N0]
v = ~np.isnan(y_train)
# Columns of mat (from get_net_info insertion order in post_search_split.py:421):
# 0:wbits 1:kvbits 2:kbits 3:vbits 4:kvdim 5:kdim 6:vdim
# 7:eff_kvbits 8:eff_kbits 9:eff_vbits 10:memory 11:(zero) 12:loss
HEAD_DIM = 128
wbits  = mat[0, :N0][v]
kvbits = mat[1, :N0][v]
kvdim  = mat[4, :N0][v]
eff_kvbits = kvbits * kvdim / HEAD_DIM
y_pf  = y_train[v]
print(f"  N={len(y_pf)}; wbits=[{wbits.min():.2f},{wbits.max():.2f}], "
      f"kvbits=[{kvbits.min():.2f},{kvbits.max():.2f}], "
      f"kvdim=[{kvdim.min():.0f},{kvdim.max():.0f}]")

# Stack as (y, c_w, c_kv, c_kd) — using kvbits + kvdim as 3D complexity
PF_3D = np.column_stack([y_pf, wbits, kvbits, kvdim])
# 2D variant uses (wbits, eff_kvbits)
PF_2D = np.column_stack([y_pf, wbits, eff_kvbits])

# ─── Pareto-envelope baseline ────────────────────────────────────────────────
def pf_envelope_max(c_query, pf_points, complexity_cols):
    """For candidate at c_query, return  max{y' : ∀axis c'≥c_query} on PF.
    This is the highest loss among PF points dominating-or-equal-to candidate
    in complexity (i.e., they use no-fewer-bits/no-less-dim than candidate).
    Candidate ε-violates iff y_actual < return-value − ε, since then the
    candidate strictly dominates that PF point (better loss with less or
    equal complexity).
    """
    mask = np.all(pf_points[:, complexity_cols] >= c_query[None, :], axis=1)
    if not mask.any():
        return np.nan
    return float(np.max(pf_points[mask, 0]))

# ─── Load evaluation results ────────────────────────────────────────────────
with open(EVAL_JSON) as f:
    data = json.load(f)
results = [r for r in data['results'] if 'y_actual' in r]
print(f"Loaded {len(results)} evaluated candidates")

# ─── Recompute baseline + violations in 3D and 2D ───────────────────────────
# Also classify candidates as in-cloud (Pareto baseline defined) vs out-of-cloud
# (no PF point has higher-or-equal complexity in every axis ⇒ candidate is at a
#  region not covered by training PF; cannot be a Pareto violator by definition,
#  it can only be a new Pareto-extension point).
y_pf_min = float(np.min(PF_3D[:, 0]))
for r in results:
    c_w  = r['wbits']; c_kv = r['kvbits']; c_kd = r['kvdim']
    eff  = r['eff_kvbits']
    base_3d = pf_envelope_max(np.array([c_w, c_kv, c_kd]), PF_3D, [1, 2, 3])
    base_2d = pf_envelope_max(np.array([c_w, eff]),       PF_2D, [1, 2])
    r['pf_baseline_3d'] = base_3d
    r['pf_baseline_2d'] = base_2d
    r['pf_baseline_1d'] = r['pf_baseline']  # keep original for comparison
    r['in_cloud_3d'] = bool(not np.isnan(base_3d))
    r['in_cloud_2d'] = bool(not np.isnan(base_2d))
    y = r['y_actual']
    # Strict Pareto violation: only meaningful when in-cloud
    for tag, base in [('3d', base_3d), ('2d', base_2d), ('1d', r['pf_baseline_1d'])]:
        for eps in (0.005, 0.01, 0.02, 0.05):
            r[f'violation_{tag}_eps_{eps}'] = bool(
                not np.isnan(base) and y < base - eps)
    # Pareto extension (out-of-cloud): does candidate beat the training PF's min loss?
    # If yes, candidate is a NEW Pareto point at higher complexity (not a violator).
    r['extends_pf'] = bool(y < y_pf_min)
    r['delta_vs_pf_min'] = float(y - y_pf_min)

# ─── Statistics ─────────────────────────────────────────────────────────────
def rule_of_three_upper(violations, n, alpha=0.05):
    if n == 0: return 1.0
    if violations == 0: return -np.log(alpha) / n
    return float(scipy_beta.ppf(1 - alpha, violations + 1, n - violations))

def viol_table(results, baseline_key, eps_list=(0.005, 0.01, 0.02, 0.05)):
    n = len(results); rows = []
    for eps in eps_list:
        # Use the matching violation flag stored on records
        tag = baseline_key.split('_')[-1]
        col = f'violation_{tag}_eps_{eps}'
        v = sum(1 for r in results if r.get(col, False))
        rows.append((eps, v, v/n if n else 0, rule_of_three_upper(v, n)))
    return rows

base_3d_arr = np.array([r['pf_baseline_3d'] for r in results])
base_2d_arr = np.array([r['pf_baseline_2d'] for r in results])
base_1d_arr = np.array([r['pf_baseline_1d'] for r in results])
y_arr       = np.array([r['y_actual']      for r in results])

# Residuals
resid_3d = y_arr - base_3d_arr
resid_2d = y_arr - base_2d_arr
resid_1d = y_arr - base_1d_arr

# ─── Write summary ──────────────────────────────────────────────────────────
lines = []
lines.append(f"Falsification w/ proper Pareto-envelope baseline  (n={len(results)})")
lines.append(f"=" * 75)
lines.append(f"")
lines.append(f"Three baselines compared (all from same 200 AWQ-measured PF samples):")
lines.append(f"  1d: f*(total_c)         where total_c = wbits + eff_kvbits   [old, projection]")
lines.append(f"  2d: f*(wbits, eff_kvbits)                                     [eng-relevant]")
lines.append(f"  3d: f*(wbits, kvbits, kvdim)                                  [search-space-native]")
lines.append(f"")
lines.append(f"For 2D/3D: baseline at (c_w,...) = max{{ y_PF : every axis c'≥c }}.")
lines.append(f"  Candidate ε-violates iff y_actual < baseline − ε  ⇒  candidate")
lines.append(f"  Pareto-dominates a PF point (lower loss with no-greater complexity).")
lines.append(f"")
in_cloud_3d = np.array([r['in_cloud_3d'] for r in results])
in_cloud_2d = np.array([r['in_cloud_2d'] for r in results])
extends    = np.array([r['extends_pf']  for r in results])
n_extends  = int(np.sum(extends & ~in_cloud_3d))
lines.append(f"Baseline coverage:")
lines.append(f"  3d defined (in-cloud): {int(in_cloud_3d.sum())}/{len(results)}")
lines.append(f"  2d defined (in-cloud): {int(in_cloud_2d.sum())}/{len(results)}")
lines.append(f"  out-of-cloud (3d):     {int((~in_cloud_3d).sum())}/{len(results)}  "
             f"(no PF point dominates these in every axis ⇒")
lines.append(f"                          cannot be ε-violators by definition;")
lines.append(f"                          they sit at PF cloud's upper-corner)")
lines.append(f"  PF extensions:         {n_extends}/{len(results)}  "
             f"(out-of-cloud AND y_actual < min(training PF y) = {y_pf_min:.5f})")
lines.append(f"")
lines.append(f"y_actual − baseline (negative ⇒ Pareto-dominates a PF point):")
for tag, arr in [('1d', resid_1d), ('2d', resid_2d), ('3d', resid_3d)]:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0: continue
    lines.append(f"  {tag}: range=[{valid.min():+.5f}, {valid.max():+.5f}]  "
                 f"mean={valid.mean():+.5f}  p05={np.percentile(valid,5):+.5f}  "
                 f"p95={np.percentile(valid,95):+.5f}")
lines.append(f"")
n_in = int(in_cloud_3d.sum())
lines.append(f"ε-violations (y_actual < baseline − ε):")
lines.append(f"  Violations only meaningful for in-cloud candidates.")
lines.append(f"  In-cloud denominator: 3d={n_in}, 2d={int(in_cloud_2d.sum())}, "
             f"1d={len(results)} (1D extrapolates everywhere).")
lines.append(f"")
lines.append(f"   ε        1d count       2d count       3d count       "
             f"1d-CI95   2d-CI95   3d-CI95")
for eps in (0.005, 0.01, 0.02, 0.05):
    rs = []
    for tag, denom in [('1d', len(results)),
                        ('2d', int(in_cloud_2d.sum())),
                        ('3d', n_in)]:
        col = f'violation_{tag}_eps_{eps}'
        v = sum(1 for r in results if r.get(col, False))
        ub = rule_of_three_upper(v, denom) if denom else float('nan')
        rs.append((v, denom, ub))
    lines.append(f"  {eps:>5.3f}    "
                 f"{rs[0][0]:>3d}/{rs[0][1]:<3d}        "
                 f"{rs[1][0]:>3d}/{rs[1][1]:<3d}        "
                 f"{rs[2][0]:>3d}/{rs[2][1]:<3d}        "
                 f"{100*rs[0][2]:>5.2f}%   "
                 f"{100*rs[1][2]:>5.2f}%   "
                 f"{100*rs[2][2]:>5.2f}%")
# Out-of-cloud detail
lines.append(f"")
lines.append(f"Out-of-cloud candidates (sit at PF training corner):")
lines.append(f"  {'sel':>4s} {'wbits':>7s} {'kvbits':>7s} {'kvdim':>5s} "
             f"{'y_act':>8s} {'min_PF':>8s} {'Δ':>9s}  extends?")
for r in results:
    if r['in_cloud_3d']: continue
    lines.append(f"  {r['sel_idx']:>4d} {r['wbits']:>7.3f} {r['kvbits']:>7.3f} "
                 f"{r['kvdim']:>5.0f} {r['y_actual']:>8.5f} "
                 f"{y_pf_min:>8.5f} {r['delta_vs_pf_min']:>+9.5f}  "
                 f"{'YES' if r['extends_pf'] else 'no'}")

with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(lines))
print('\n' + '\n'.join(lines))

# ─── Save JSON + CSV ───────────────────────────────────────────────────────
data['results_with_3d'] = results
with open(OUT_JSON, 'w') as f:
    json.dump({'results': results}, f, indent=2,
              default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
print(f"\nsaved {OUT_JSON}")

with open(OUT_CSV, 'w', newline='') as f:
    cols = ['sel_idx', 'src', 'wbits', 'kvbits', 'kvdim', 'eff_kvbits',
            'total_c', 'pf_baseline_1d', 'pf_baseline_2d', 'pf_baseline_3d',
            'y_actual', 'pred_mu', 'pred_sigma', 'prob_violator',
            'violation_3d_eps_0.005', 'violation_3d_eps_0.01',
            'violation_3d_eps_0.02', 'violation_3d_eps_0.05']
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for r in results: w.writerow(r)
print(f"saved {OUT_CSV}")

# ─── Figure: comparison of baselines and residuals ──────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# (a-c) Residuals vs each baseline
for ax, (tag, base, resid) in zip(axes[0],
        [('1d', base_1d_arr, resid_1d),
         ('2d', base_2d_arr, resid_2d),
         ('3d', base_3d_arr, resid_3d)]):
    ax.scatter(base, y_arr, s=28, c='steelblue', edgecolor='black',
               linewidth=0.3, alpha=0.85)
    lo = np.nanmin(np.concatenate([base, y_arr]))
    hi = np.nanmax(np.concatenate([base, y_arr]))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, label='y=baseline')
    for eps, ls_, col in [(0.01, '-', '#c0392b'), (0.005, ':', '#9b59b6')]:
        ax.plot([lo, hi], [lo - eps, hi - eps], col, ls=ls_, lw=1.0,
                label=f'ε={eps}')
    ax.set_xlabel(f'pf_baseline_{tag}'); ax.set_ylabel('y_actual')
    n_v = int(np.sum((y_arr < base - 0.01) & ~np.isnan(base)))
    ax.set_title(f'baseline {tag}  |  ε=0.01 viol={n_v}/{len(results)}',
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (d) Residual histograms — 3 baselines overlaid
ax = axes[1, 0]
for tag, resid, c in [('1d', resid_1d, '#888'), ('2d', resid_2d, '#3498db'),
                      ('3d', resid_3d, '#e74c3c')]:
    valid = resid[~np.isnan(resid)]
    ax.hist(valid, bins=25, alpha=0.5, color=c, label=f'{tag}', edgecolor='black')
ax.axvline(0, color='black', lw=0.8)
ax.axvline(-0.01, color='red', ls='--', lw=0.7, label='−ε=0.01')
ax.set_xlabel('y_actual − baseline'); ax.set_ylabel('count')
ax.set_title('residual distribution (3 baselines)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (e) Per-candidate baseline_3d − baseline_1d (how 1D over/underestimates)
ax = axes[1, 1]
diff = base_3d_arr - base_1d_arr
total_c = np.array([r['total_c'] for r in results])
ax.scatter(total_c, diff, s=28, c='purple', edgecolor='black', linewidth=0.3)
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('total_c (sum)'); ax.set_ylabel('baseline_3d − baseline_1d')
ax.set_title('how much 1D was off vs 3D\n(+ ⇒ 1D was tighter, − ⇒ 1D was looser)',
             fontweight='bold')
ax.grid(True, alpha=0.3)

# (f) Scatter: candidate placement in (wbits, eff_kvbits) colored by residual_3d
ax = axes[1, 2]
wb = np.array([r['wbits'] for r in results])
ek = np.array([r['eff_kvbits'] for r in results])
sc = ax.scatter(wb, ek, c=resid_3d, cmap='RdBu_r', s=42, edgecolor='black',
                linewidth=0.3, vmin=-max(0.005, np.nanmax(np.abs(resid_3d))),
                vmax=max(0.005, np.nanmax(np.abs(resid_3d))))
plt.colorbar(sc, ax=ax, fraction=0.04, label='y − baseline_3d')
# Overlay PF points for reference
ax.scatter(wbits, eff_kvbits, s=8, c='gray', alpha=0.4, label='200 PF samples')
ax.set_xlabel('wbits'); ax.set_ylabel('eff_kvbits')
ax.set_title('candidates in 2D complexity\n(red ⇒ violator-side)',
             fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle(f'Falsification w/ proper baselines (n={len(results)})',
             fontweight='bold', y=1.0)
plt.tight_layout()
os.makedirs(os.path.dirname(FIG), exist_ok=True)
plt.savefig(FIG, dpi=150, bbox_inches='tight'); plt.close()
print(f"saved {FIG}")
print("Done.")
