"""08_phase8_analysis.py — falsification analysis on Phase-7 measurements.

Six endpoints:
  1. 3D envelope residual           r_3D = y_actual − f_C*(c)
  2. Paired projection margin       d(a) = y(a) − y(π(a))  (from B3 pairs)
  3. Rank-depth breakdown            violation rate by axis-layer combo
  4. Calibration                     P_ε bin reliability + Brier + ECE
  5. Support-distance audit          Mahalanobis d_train(a) to RS50
  6. Zero-violation Clopper-Pearson  upper bound at α=0.05

Per-model figure with all 6 panels + JSON dump.
"""
import os, sys, json, warnings
warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as _beta, wilcoxon

from _common import load_csv, extract_xy, PATHS

BASE = '/NAS/SJ/actquant/search'
OUT  = f'{BASE}/analysis/v4'
FIGDIR = f'{OUT}/figures'
os.makedirs(FIGDIR, exist_ok=True)

EPS_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def clopper_pearson(k, n, alpha=0.05):
    if n == 0: return (0.0, 1.0)
    if k == 0: return (0.0, 1 - alpha ** (1.0 / n))
    if k == n: return (alpha ** (1.0 / n), 1.0)
    return (float(_beta.ppf(alpha/2, k, n-k+1)),
            float(_beta.ppf(1-alpha/2, k+1, n-k)))


def mahalanobis_to_rs50(X_test, X_rs):
    """Robust Mahalanobis distance to RS50 train set (per-axis)."""
    mu = X_rs.mean(0)
    cov = np.cov(X_rs.T) + 1e-6 * np.eye(X_rs.shape[1])
    inv = np.linalg.inv(cov)
    diff = X_test - mu
    return np.sqrt(np.einsum('ni,ij,nj->n', diff, inv, diff))


def fC_star_3D_lookup(c_query, pf_pts):
    """Reproduces the Phase-6 baseline interpolant."""
    out = np.full(len(c_query), pf_pts[:, 0].max(), dtype=float)
    for i in range(len(out)):
        wb, kb, kd = c_query[i]
        cover = ((pf_pts[:, 1] <= wb) & (pf_pts[:, 2] <= kb) & (pf_pts[:, 3] <= kd))
        if cover.any(): out[i] = pf_pts[cover, 0].min()
    return out


def analyse(tag, pretty):
    acq_path = f'{OUT}/acquired_falsifiers_{tag}.json'
    eval_path = f'{OUT}/eval_falsifiers_{tag}.json'
    if not os.path.exists(eval_path):
        print(f"[{tag}] no eval file yet: {eval_path}"); return None
    with open(acq_path) as f: acq = json.load(f)
    with open(eval_path) as f: ev = json.load(f)
    results = ev['results']
    if not results: return None
    n = len(results)
    print(f"\n=== Phase 8 [{tag}] — {pretty} (n={n}) ===")

    σ_conf = acq['sigma_conf']
    eps_p  = acq['eps_primary']
    pf_pts = np.array(acq['baseline_pf_points'])
    delta_3 = acq['archive_slack_delta_3']

    # Reconstruct fields from records
    y_actual = np.array([r['y_actual'] for r in results])
    mu       = np.array([r['rbf_mu']  for r in results])
    P_eps    = np.array([r['P_eps0p005'] for r in results])
    EVI      = np.array([r['EVI_eps0p005'] for r in results])
    wbits    = np.array([r['actual_wbits']  for r in results])
    kvbits   = np.array([r['actual_kvbits'] for r in results])
    kvdim    = np.array([r['actual_kvdim']  for r in results])
    bucket   = np.array([r['bucket']  for r in results])
    w_layer  = np.array([r['w_layer']  for r in results])
    kv_layer = np.array([r['kv_layer'] for r in results])
    kvd_layer= np.array([r['kvd_layer']for r in results])

    # Re-compute f_C* per actual complexity (records already had it, but actual
    # complexity might differ slightly — use the measured wbits/kvbits/kvdim)
    c_query = np.column_stack([wbits, kvbits, kvdim])
    fC_star = fC_star_3D_lookup(c_query, pf_pts)
    r_3D = y_actual - fC_star

    # ── Endpoint 1: r_3D distribution per ε ─────────────────────────────
    print(f"\n  [1] 3D envelope residual r_3D = y_actual − f_C*(c)")
    e1 = []
    for eps in EPS_GRID:
        n_v = int((r_3D < -eps).sum())
        lo, hi = clopper_pearson(n_v, n)
        e1.append(dict(eps=eps, n_violations=n_v, total=n,
                       cp_lower=lo, cp_upper=hi))
        print(f"    ε={eps:>6.3f}:  violations={n_v:>3d}/{n}  "
              f"CP95% = [{lo*100:.2f} %, {hi*100:.2f} %]")
    print(f"    min r_3D = {r_3D.min():+.5f},  median = {np.median(r_3D):+.5f}")

    # ── Endpoint 2: paired projection margin ────────────────────────────
    print(f"\n  [2] Paired projection margin d(a) = y(a) − y(π(a))")
    pair_map = acq.get('b3_pair_map', [])
    idx_by_pool = {r['pool_idx']: i for i, r in enumerate(results)}
    d_arr = []
    for pm in pair_map:
        if pm['a_pool_idx'] in idx_by_pool and pm['pi_pool_idx'] in idx_by_pool:
            a_i = idx_by_pool[pm['a_pool_idx']]
            pi_i = idx_by_pool[pm['pi_pool_idx']]
            d_arr.append(y_actual[a_i] - y_actual[pi_i])
    d_arr = np.array(d_arr)
    if len(d_arr) >= 2:
        e2 = {}
        for eps in EPS_GRID:
            n_v = int((d_arr < -eps).sum())
            lo, hi = clopper_pearson(n_v, len(d_arr))
            e2[f'eps_{eps}'] = dict(n_violations=n_v, total=len(d_arr),
                                    cp_lower=lo, cp_upper=hi)
            print(f"    ε={eps:>6.3f}:  violations={n_v:>3d}/{len(d_arr)}  "
                  f"CP95% = [{lo*100:.2f} %, {hi*100:.2f} %]")
        # Wilcoxon sign rank: H0 d = 0 vs H1 d < 0 (one-sided)
        if np.any(d_arr != 0):
            stat, p = wilcoxon(d_arr, alternative='less')
            e2['wilcoxon_p_lt_0'] = float(p)
            print(f"    Wilcoxon signed-rank (H1: d<0): p = {p:.4f}")
        e2['min_d'] = float(d_arr.min())
        e2['median_d'] = float(np.median(d_arr))
        e2['n_pairs'] = int(len(d_arr))
    else:
        e2 = dict(n_pairs=int(len(d_arr)), note='too few pairs for stats')
        print(f"    only {len(d_arr)} pairs available")

    # ── Endpoint 3: rank-depth breakdown ────────────────────────────────
    print(f"\n  [3] Rank-depth breakdown (axis-layer combo vs violation rate)")
    layer_max = np.maximum.reduce([w_layer, kv_layer, kvd_layer])
    e3 = []
    for K in (1, 2, 3):
        mask = layer_max == K
        n_g = int(mask.sum()); n_v = int(((r_3D < -eps_p) & mask).sum())
        lo, hi = clopper_pearson(n_v, n_g) if n_g > 0 else (0, 1)
        e3.append(dict(max_layer=K, n=n_g, n_violations=n_v, cp_lower=lo, cp_upper=hi))
        print(f"    max_layer={K}: n={n_g:>3d}  violations(ε={eps_p})={n_v:>2d}  "
              f"CP95% = [{lo*100:.2f} %, {hi*100:.2f} %]")

    # ── Endpoint 4: P_ε calibration (reliability diagram) ───────────────
    print(f"\n  [4] Calibration — P_ε bin reliability")
    bins = [(0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.001)]
    e4 = []
    for lo_b, hi_b in bins:
        mask = (P_eps >= lo_b) & (P_eps < hi_b)
        n_b = int(mask.sum())
        if n_b == 0:
            e4.append(dict(bin=f'[{lo_b:.1f},{hi_b:.1f})', n=0,
                           observed=None, mean_P=None))
            print(f"    P_ε ∈ [{lo_b:.1f},{hi_b:.1f}): empty")
            continue
        n_v = int(((r_3D < -eps_p) & mask).sum())
        obs = n_v / n_b
        mean_P = float(P_eps[mask].mean())
        e4.append(dict(bin=f'[{lo_b:.1f},{hi_b:.1f})', n=n_b,
                       observed_rate=obs, mean_P=mean_P, n_violations=n_v))
        print(f"    P_ε ∈ [{lo_b:.1f},{hi_b:.1f}): n={n_b:>3d}  mean_P={mean_P:.3f}  "
              f"observed violation rate = {obs:.3f}")
    # Brier score (P vs is_violation)
    is_viol = (r_3D < -eps_p).astype(float)
    brier = float(np.mean((P_eps - is_viol) ** 2))
    # ECE
    bin_edges = np.linspace(0, 1, 11); ece = 0.0
    for k in range(10):
        m = (P_eps >= bin_edges[k]) & (P_eps < bin_edges[k+1])
        if m.sum() == 0: continue
        ece += (m.sum() / n) * abs(P_eps[m].mean() - is_viol[m].mean())
    print(f"    Brier = {brier:.4f}   ECE = {ece:.4f}")

    # ── Endpoint 5: support-distance audit ──────────────────────────────
    print(f"\n  [5] Support-distance audit (Mahalanobis to RS50)")
    X_rs, _, _, _ = extract_xy(load_csv(PATHS[f'{tag}_rs50']))
    # Use z-coords (RBF input) — read from arch dict via record sub_loss_*
    X_cand = np.column_stack([[r['sub_loss_W'] for r in results],
                              [r['sub_loss_KV'] for r in results],
                              [r['sub_loss_KVD'] for r in results]])
    d_train = mahalanobis_to_rs50(X_cand, X_rs)
    threshold = float(np.quantile(d_train, 0.5))  # median split
    in_support = d_train <= threshold
    n_in = int(in_support.sum()); n_out = int((~in_support).sum())
    v_in = int(((r_3D < -eps_p) & in_support).sum())
    v_out = int(((r_3D < -eps_p) & ~in_support).sum())
    e5 = dict(threshold_q50=threshold,
              n_in=n_in, n_out=n_out, viol_in=v_in, viol_out=v_out,
              mean_d_train=float(d_train.mean()),
              max_d_train=float(d_train.max()))
    print(f"    d_train median = {threshold:.3f}  range [{d_train.min():.2f}, {d_train.max():.2f}]")
    print(f"    in-support  (≤ median): n={n_in:>3d}  violations={v_in}")
    print(f"    out-support (>  median): n={n_out:>3d}  violations={v_out}")

    # ── Endpoint 6: zero-violation Clopper-Pearson (overall) ────────────
    print(f"\n  [6] Zero-violation Clopper-Pearson 95% upper bound")
    e6 = {}
    for eps in EPS_GRID:
        n_v = int((r_3D < -eps).sum())
        lo, hi = clopper_pearson(n_v, n)
        e6[f'eps_{eps}'] = dict(k=n_v, n=n, cp_upper=hi)
        print(f"    ε={eps:>6.3f}  k={n_v}/{n}  upper bound = {hi*100:.2f} %")

    # ── Figure (6 panels) ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # P1: r_3D histogram
    ax = axes[0, 0]
    ax.hist(r_3D, bins=40, color='C0', edgecolor='k', linewidth=0.3, alpha=0.85)
    ax.axvline(0, color='k', lw=0.8)
    for eps, col in zip([0.005, 0.01, 0.05], ['C3', 'C1', 'gray']):
        ax.axvline(-eps, color=col, ls='--', lw=0.7, label=f'ε={eps}')
    ax.set_xlabel('r_3D = y_actual − f_C*(c)')
    ax.set_ylabel('count')
    ax.set_title(f'[1] residual distribution (n={n})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # P2: paired d(a)
    ax = axes[0, 1]
    if len(d_arr) >= 1:
        ax.hist(d_arr, bins=15, color='C2', edgecolor='k', linewidth=0.3, alpha=0.85)
        ax.axvline(0, color='k', lw=0.8)
        ax.axvline(-eps_p, color='C3', ls='--', lw=0.8, label=f'ε={eps_p}')
        ax.set_xlabel('d(a) = y(a) − y(π(a))')
        ax.set_title(f'[2] paired projection margin (n={len(d_arr)})')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'no paired data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('[2] paired projection (empty)')
    ax.grid(alpha=0.3)

    # P3: rank-depth bars
    ax = axes[0, 2]
    Ks = [d['max_layer'] for d in e3]
    rates = [d['n_violations'] / max(d['n'], 1) for d in e3]
    n_ks = [d['n'] for d in e3]
    bars = ax.bar(Ks, rates, color=['C0', 'C1', 'C3'], edgecolor='k', linewidth=0.4)
    for k, r, n_k in zip(Ks, rates, n_ks):
        ax.text(k, r + 0.005, f'{n_k}', ha='center', fontsize=8)
    ax.set_xticks(Ks); ax.set_xticklabels([f'K={k}' for k in Ks])
    ax.set_ylabel(f'observed violation rate (ε={eps_p})')
    ax.set_title(f'[3] rank-depth breakdown')
    ax.grid(alpha=0.3, axis='y')

    # P4: reliability
    ax = axes[1, 0]
    obs = [d.get('observed_rate', 0) for d in e4 if d['n'] > 0]
    mp  = [d.get('mean_P', 0) for d in e4 if d['n'] > 0]
    ns  = [d['n'] for d in e4 if d['n'] > 0]
    ax.plot([0, 1], [0, 1], 'k--', lw=0.6, alpha=0.5, label='perfect')
    ax.scatter(mp, obs, s=[30 + 5*x for x in ns], color='C3', edgecolor='k', linewidth=0.4)
    for x, y, n_v in zip(mp, obs, ns): ax.text(x, y + 0.02, f'n={n_v}', fontsize=8)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('mean predicted P_ε'); ax.set_ylabel('observed violation rate')
    ax.set_title(f'[4] reliability  Brier={brier:.4f}  ECE={ece:.4f}')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # P5: support distance scatter
    ax = axes[1, 1]
    ax.scatter(d_train[in_support],  r_3D[in_support],  s=20, color='C0',
               label=f'in-support (≤ q50, n={n_in})', edgecolor='k', linewidth=0.3)
    ax.scatter(d_train[~in_support], r_3D[~in_support], s=20, color='C3',
               label=f'out-support (> q50, n={n_out})', edgecolor='k', linewidth=0.3)
    ax.axhline(0, color='k', lw=0.6)
    ax.axhline(-eps_p, color='C3', ls='--', lw=0.7, label=f'-ε={eps_p}')
    ax.set_xlabel('Mahalanobis d_train (to RS50)')
    ax.set_ylabel('r_3D')
    ax.set_title('[5] support-distance audit')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # P6: Clopper-Pearson upper bounds across ε grid
    ax = axes[1, 2]
    eps_x = [d['k'] for k, d in e6.items()]   # not used
    cp_y = [e6[f'eps_{eps}']['cp_upper'] * 100 for eps in EPS_GRID]
    ax.semilogx(EPS_GRID, cp_y, 'o-', color='C3', lw=1.4)
    for eps, y in zip(EPS_GRID, cp_y):
        k = e6[f'eps_{eps}']['k']
        ax.text(eps, y + 0.3, f'k={k}', fontsize=8, ha='center')
    ax.set_xlabel('ε (log)'); ax.set_ylabel('95% CP upper bound on viol. rate (%)')
    ax.set_title(f'[6] zero-violation CP upper bounds (n={n})')
    ax.grid(alpha=0.3, which='both')

    plt.suptitle(f"Phase 8 — {pretty}  (σ_conf={σ_conf:.4f}, ε_primary={eps_p})", fontsize=11)
    plt.tight_layout()
    fig_path = f'{FIGDIR}/v4_fig8_falsification_{tag}.png'
    plt.savefig(fig_path, dpi=140, bbox_inches='tight'); plt.close()
    print(f"\n  saved figure: {fig_path}")

    return dict(
        model=pretty, n=n, sigma_conf=σ_conf, eps_primary=eps_p,
        archive_slack_delta_3=delta_3,
        endpoints=dict(
            E1_residual_envelope=e1,
            E2_paired_projection=e2,
            E3_rank_depth=e3,
            E4_calibration=dict(bins=e4, brier=brier, ECE=ece),
            E5_support_distance=e5,
            E6_CP_upper_bound=e6,
        ),
        summary=dict(
            min_r_3D=float(r_3D.min()),
            median_r_3D=float(np.median(r_3D)),
            violations_eps_0p005=int((r_3D < -eps_p).sum()),
            cp_upper_eps_0p005=clopper_pearson(int((r_3D < -eps_p).sum()), n)[1],
        ),
    )


if __name__ == '__main__':
    out = {}
    for tag, pretty in [('llama', 'Llama-3.1-8B-Instruct'),
                        ('qwen',  'Qwen2.5-7B-Instruct')]:
        r = analyse(tag, pretty)
        if r: out[tag] = r
    with open(f'{OUT}/phase8_falsification_results.json', 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
    print(f"\nsaved phase8_falsification_results.json")
    print("Done (Phase 8).")
