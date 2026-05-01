"""
Interaction Terms + Low-wbits Correction + Hierarchical vs Direct Calibration
==============================================================================
Follow-up analysis:
  (1) Effect of adding interaction terms to the additive model
  (2) Correction strategies for low wbits region (worst residuals)
  (3) Direct 3-way LSQ vs Hierarchical 2-way→3-way calibration
"""

import os, csv, json
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.simplefilter("ignore")

BASE = '/NAS/SJ/actquant/search/save/result'
EXPERIMENTS = {
    'awq_w_kv':       '2604162010_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr',
    'awq_w_kvdim':    '2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim',
    'awq_w_kv_kvdim': '2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim',
    'awq_kv_kvdim':   '2604162013_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_kv_expr_kv_dim',
}
COMP_KEYS = ['wbits', 'kvbits', 'kbits', 'vbits', 'kvdim', 'kdim', 'vdim',
             'eff_kvbits', 'eff_kbits', 'eff_vbits', 'memory', 'n_token']

def load_csv(name):
    data = []
    with open(os.path.join(BASE, EXPERIMENTS[name], 'results.csv')) as f:
        for row in csv.reader(f):
            if row:
                data.append([float(x) for x in row])
    return data

def parse(name):
    d = load_csv(name)
    return {'comp': {k: np.array(d[i]) for i, k in enumerate(COMP_KEYS)},
            'jsd': np.array(d[len(COMP_KEYS)])}

def lsq(X, y, intercept=True):
    if X.ndim == 1: X = X.reshape(-1, 1)
    Xa = np.column_stack([X, np.ones(len(y))]) if intercept else X
    th, _, _, _ = np.linalg.lstsq(Xa, y, rcond=None)
    yp = Xa @ th
    ss_r = np.sum((y - yp)**2)
    ss_t = np.sum((y - np.mean(y))**2)
    return th, 1 - ss_r/ss_t, yp

def cv_lsq(X, y, k=5, intercept=True):
    if X.ndim == 1: X = X.reshape(-1, 1)
    n = len(y)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    fold = n // k
    r2s, sps, mses = [], [], []
    for f in range(k):
        te = idx[f*fold:(f+1)*fold]
        tr = np.concatenate([idx[:f*fold], idx[(f+1)*fold:]])
        Xtr = np.column_stack([X[tr], np.ones(len(tr))]) if intercept else X[tr]
        Xte = np.column_stack([X[te], np.ones(len(te))]) if intercept else X[te]
        th, _, _, _ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        yp = Xte @ th
        ss_r = np.sum((y[te] - yp)**2)
        ss_t = np.sum((y[te] - np.mean(y[te]))**2)
        r2s.append(1 - ss_r/ss_t if ss_t > 0 else 0)
        sps.append(stats.spearmanr(yp, y[te])[0])
        mses.append(np.mean((y[te]-yp)**2))
    return np.mean(r2s), np.std(r2s), np.mean(sps), np.std(sps), np.mean(mses)


def main():
    e = parse('awq_w_kv_kvdim')
    y = e['jsd']
    wb = e['comp']['wbits']
    kvb = e['comp']['kvbits']
    kvd = e['comp']['kvdim']

    # Standardized for stability
    wb_s = (wb - wb.mean()) / wb.std()
    kvb_s = (kvb - kvb.mean()) / kvb.std()
    kvd_s = (kvd - kvd.mean()) / kvd.std()
    # Also collect component F arrays for 2-way experiments (same random seed → same combos)
    # We'll use 3-way complexity metrics as proxy for all.

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: INTERACTION TERM ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 90)
    print("PART 1: INTERACTION TERM ANALYSIS")
    print("=" * 90)

    # Build feature sets
    fsets = {
        'M0: Linear additive':
            np.column_stack([wb_s, kvb_s, kvd_s]),
        'M1: + W×KV':
            np.column_stack([wb_s, kvb_s, kvd_s, wb_s*kvb_s]),
        'M2: + W×KVdim':
            np.column_stack([wb_s, kvb_s, kvd_s, wb_s*kvd_s]),
        'M3: + KV×KVdim':
            np.column_stack([wb_s, kvb_s, kvd_s, kvb_s*kvd_s]),
        'M4: + all 2-way inter.':
            np.column_stack([wb_s, kvb_s, kvd_s, wb_s*kvb_s, wb_s*kvd_s, kvb_s*kvd_s]),
        'M5: + quadratics':
            np.column_stack([wb_s, kvb_s, kvd_s, wb_s**2, kvb_s**2, kvd_s**2]),
        'M6: + W² only':
            np.column_stack([wb_s, kvb_s, kvd_s, wb_s**2]),
        'M7: quad + all inter.':
            np.column_stack([wb_s, kvb_s, kvd_s, wb_s**2, kvb_s**2, kvd_s**2,
                             wb_s*kvb_s, wb_s*kvd_s, kvb_s*kvd_s]),
        'M8: W² + W×KV + W×KVdim':
            np.column_stack([wb_s, kvb_s, kvd_s, wb_s**2, wb_s*kvb_s, wb_s*kvd_s]),
        'M9: cubic W + lin kv+kvdim':
            np.column_stack([wb_s, kvb_s, kvd_s, wb_s**2, wb_s**3]),
    }

    print(f"\n{'Model':<32} {'p':>4} {'R²':>7} {'CV R²':>14} {'CV ρ':>14} {'CV MSE':>9}")
    print("-" * 82)
    model_results = {}
    for name, X in fsets.items():
        th, r2, yp = lsq(X, y)
        r2_cv, r2_cv_std, sp_cv, sp_cv_std, mse_cv = cv_lsq(X, y)
        p_count = X.shape[1] + 1
        model_results[name] = (r2, r2_cv, sp_cv, mse_cv, yp)
        print(f"{name:<32} {p_count:>4} {r2:>7.4f} {r2_cv:>6.4f}±{r2_cv_std:.4f} {sp_cv:>6.4f}±{sp_cv_std:.4f} {mse_cv:>9.6f}")

    # Partial F-test for each interaction
    print(f"\n  Partial F-test vs linear additive (M0):")
    base_X = fsets['M0: Linear additive']
    _, _, yp_base = lsq(base_X, y)
    ss_base = np.sum((y - yp_base)**2)
    n = len(y)
    p_base = base_X.shape[1] + 1
    for name in ['M1: + W×KV', 'M2: + W×KVdim', 'M3: + KV×KVdim', 'M4: + all 2-way inter.',
                  'M5: + quadratics', 'M6: + W² only', 'M8: W² + W×KV + W×KVdim']:
        X = fsets[name]
        _, _, yp_alt = lsq(X, y)
        ss_alt = np.sum((y - yp_alt)**2)
        p_alt = X.shape[1] + 1
        d_p = p_alt - p_base
        if d_p == 0: continue
        f_stat = ((ss_base - ss_alt) / d_p) / (ss_alt / (n - p_alt))
        f_p = 1 - stats.f.cdf(f_stat, d_p, n - p_alt)
        sig = '**' if f_p < 0.01 else ('*' if f_p < 0.05 else '')
        print(f"    {name:<32} F={f_stat:>7.3f}, p={f_p:.2e}  {sig}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: LOW-WBITS CORRECTION STRATEGIES
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("PART 2: LOW-WBITS CORRECTION STRATEGIES")
    print("=" * 90)

    # Low wbits region = [2.0, 2.75) → 107 samples (from prev. analysis)
    low_mask = wb < 2.75
    hi_mask = ~low_mask
    print(f"\n  Low wbits [<2.75): {low_mask.sum()} samples, JSD μ={y[low_mask].mean():.4f}, σ={y[low_mask].std():.4f}")
    print(f"  High wbits [≥2.75): {hi_mask.sum()} samples, JSD μ={y[hi_mask].mean():.4f}, σ={y[hi_mask].std():.4f}")

    # Strategy A: Baseline linear additive
    th_A, r2_A, yp_A = lsq(np.column_stack([wb, kvb, kvd]), y)
    resid_A = y - yp_A
    low_mae_A = np.mean(np.abs(resid_A[low_mask]))
    hi_mae_A = np.mean(np.abs(resid_A[hi_mask]))

    # Strategy B: Log(y) transform
    y_log = np.log(y + 1e-6)
    th_B, _, yp_B_log = lsq(np.column_stack([wb, kvb, kvd]), y_log)
    yp_B = np.exp(yp_B_log)
    resid_B = y - yp_B
    r2_B = 1 - np.sum(resid_B**2) / np.sum((y - y.mean())**2)

    # Strategy C: Quadratic on wbits only
    th_C, r2_C, yp_C = lsq(np.column_stack([wb, wb**2, kvb, kvd]), y)
    resid_C = y - yp_C

    # Strategy D: Exponential transform on wbits (2^(-wbits))
    wb_exp = 2.0 ** (-wb)  # rough information-theoretic quantization noise proxy
    th_D, r2_D, yp_D = lsq(np.column_stack([wb_exp, kvb, kvd]), y)
    resid_D = y - yp_D

    # Strategy E: Piecewise linear — separate low/high fits
    X_pw_low = np.column_stack([wb[low_mask], kvb[low_mask], kvd[low_mask]])
    th_pw_lo, _, yp_pw_lo = lsq(X_pw_low, y[low_mask])
    X_pw_hi = np.column_stack([wb[hi_mask], kvb[hi_mask], kvd[hi_mask]])
    th_pw_hi, _, yp_pw_hi = lsq(X_pw_hi, y[hi_mask])
    yp_E = np.zeros_like(y)
    yp_E[low_mask] = yp_pw_lo
    yp_E[hi_mask] = yp_pw_hi
    resid_E = y - yp_E
    r2_E = 1 - np.sum(resid_E**2) / np.sum((y - y.mean())**2)

    # Strategy F: Weighted LSQ — upweight low-wbits samples
    weights = np.where(low_mask, 2.0, 1.0)
    W = np.diag(weights)
    X_F = np.column_stack([wb, kvb, kvd, np.ones(n)])
    th_F = np.linalg.solve(X_F.T @ W @ X_F, X_F.T @ W @ y)
    yp_F = X_F @ th_F
    resid_F = y - yp_F
    r2_F = 1 - np.sum(resid_F**2) / np.sum((y - y.mean())**2)

    # Strategy G: Quadratic + all interactions (best from Part 1)
    X_G = np.column_stack([wb_s, kvb_s, kvd_s, wb_s**2, kvb_s**2, kvd_s**2,
                           wb_s*kvb_s, wb_s*kvd_s, kvb_s*kvd_s])
    th_G, r2_G, yp_G = lsq(X_G, y)
    resid_G = y - yp_G

    # Strategy H: 2^(-wbits) + interactions
    wb_exp_s = (wb_exp - wb_exp.mean()) / wb_exp.std()
    X_H = np.column_stack([wb_exp_s, kvb_s, kvd_s, wb_exp_s*kvb_s, wb_exp_s*kvd_s])
    th_H, r2_H, yp_H = lsq(X_H, y)
    resid_H = y - yp_H

    strategies = [
        ('A: Linear additive (baseline)',          r2_A, resid_A, 3),
        ('B: log(y) + linear x',                   r2_B, resid_B, 3),
        ('C: wbits + wbits² (W²)',                 r2_C, resid_C, 4),
        ('D: 2^(-wbits) + kv + kvdim',             r2_D, resid_D, 3),
        ('E: Piecewise (low/hi split at 2.75)',    r2_E, resid_E, 7),
        ('F: Weighted LSQ (low wbits ×2)',         r2_F, resid_F, 3),
        ('G: Quad + all 2-way interactions',       r2_G, resid_G, 9),
        ('H: 2^(-wbits) × kv/kvdim inter.',        r2_H, resid_H, 5),
    ]

    print(f"\n{'Strategy':<38} {'p':>3} {'R²':>7} {'MAE_low':>9} {'MAE_hi':>9} {'σ_low':>8} {'σ_hi':>8} {'max|r|_low':>11}")
    print("-" * 100)
    for label, r2s, rs, p in strategies:
        mae_lo = np.mean(np.abs(rs[low_mask]))
        mae_hi = np.mean(np.abs(rs[hi_mask]))
        sd_lo = np.std(rs[low_mask])
        sd_hi = np.std(rs[hi_mask])
        mx_lo = np.max(np.abs(rs[low_mask]))
        print(f"{label:<38} {p:>3} {r2s:>7.4f} {mae_lo:>9.5f} {mae_hi:>9.5f} {sd_lo:>8.5f} {sd_hi:>8.5f} {mx_lo:>11.5f}")

    # CV for each strategy
    print(f"\n  5-fold CV performance (on test folds):")
    print(f"{'Strategy':<38} {'CV R²':>14} {'CV ρ':>14} {'CV MSE':>10}")
    print("-" * 80)
    cv_feats = {
        'A: Linear additive': np.column_stack([wb, kvb, kvd]),
        'C: + wbits²': np.column_stack([wb, wb**2, kvb, kvd]),
        'D: 2^(-wbits)': np.column_stack([wb_exp, kvb, kvd]),
        'G: Quad + interactions': X_G,
        'H: 2^(-wbits) + inter.': X_H,
    }
    for label, X in cv_feats.items():
        r2_cv, r2_cv_sd, sp_cv, sp_cv_sd, mse_cv = cv_lsq(X, y)
        print(f"{label:<38} {r2_cv:>6.4f}±{r2_cv_sd:.4f} {sp_cv:>6.4f}±{sp_cv_sd:.4f} {mse_cv:>10.6f}")

    # Rank correlation within low-wbits region specifically
    print(f"\n  Spearman ρ within LOW wbits (<2.75, n={low_mask.sum()}):")
    for label, r2s, rs, p in strategies:
        yp_s = y - rs
        sp = stats.spearmanr(yp_s[low_mask], y[low_mask])[0]
        print(f"    {label:<40} ρ={sp:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: DIRECT 3-WAY vs HIERARCHICAL CALIBRATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("PART 3: DIRECT 3-WAY vs HIERARCHICAL CALIBRATION")
    print("=" * 90)

    # Direct 3-way LSQ
    X_direct = np.column_stack([wb, kvb, kvd])
    th_D, r2_D, yp_D = lsq(X_direct, y)
    cv_D = cv_lsq(X_direct, y)

    # Direct with quadratic (best full model)
    X_direct_q = np.column_stack([wb, kvb, kvd, wb**2, kvb**2, kvd**2])
    th_Dq, r2_Dq, yp_Dq = lsq(X_direct_q, y)
    cv_Dq = cv_lsq(X_direct_q, y)

    # Hierarchical strategies:
    # Step 1: calibrate (A+B) on (training data for just those two components)
    # Step 2: use calibrated_AB as feature with C

    # But each pair's calibration must use same samples: we can't use awq_w_kv
    # data for (W+KV) calibration and then apply to 3-way samples, because they
    # are different random samples.
    # So simulate hierarchy within the 3-way sample set:
    # Hier-path P12 (WKV first): fit yp_12 = f(wb, kvb), then y = g(yp_12, kvd)
    # This is algebraically equivalent to linear additive in (wb, kvb, kvd) because
    # linear(linear) = linear. BUT if we allow nonlinearity in step 1 or step 2, it differs.

    # Hierarchical with nonlinear composition: calibrate each step with quadratic
    def hierarchical(xA, xB, xC, y):
        """Step1: yp_AB = quadratic fit on (xA, xB). Step2: quadratic fit (yp_AB, xC)."""
        X1 = np.column_stack([xA, xB, xA**2, xB**2, xA*xB])
        th1, r2_1, yp1 = lsq(X1, y)
        X2 = np.column_stack([yp1, xC, yp1**2, xC**2, yp1*xC])
        th2, r2_2, yp2 = lsq(X2, y)
        return r2_1, r2_2, yp2

    # Direct 3-way with same model complexity as hierarchical final
    def hierarchical_cv(xA, xB, xC, y, k=5):
        n = len(y)
        rng = np.random.RandomState(42)
        idx_perm = rng.permutation(n)
        fold = n // k
        r2s, sps = [], []
        for f in range(k):
            te = idx_perm[f*fold:(f+1)*fold]
            tr = np.concatenate([idx_perm[:f*fold], idx_perm[(f+1)*fold:]])
            # Step 1 on tr
            X1_tr = np.column_stack([xA[tr], xB[tr], xA[tr]**2, xB[tr]**2, xA[tr]*xB[tr], np.ones(len(tr))])
            th1, _, _, _ = np.linalg.lstsq(X1_tr, y[tr], rcond=None)
            yp1_tr = X1_tr @ th1
            X1_te = np.column_stack([xA[te], xB[te], xA[te]**2, xB[te]**2, xA[te]*xB[te], np.ones(len(te))])
            yp1_te = X1_te @ th1
            # Step 2
            X2_tr = np.column_stack([yp1_tr, xC[tr], yp1_tr**2, xC[tr]**2, yp1_tr*xC[tr], np.ones(len(tr))])
            th2, _, _, _ = np.linalg.lstsq(X2_tr, y[tr], rcond=None)
            X2_te = np.column_stack([yp1_te, xC[te], yp1_te**2, xC[te]**2, yp1_te*xC[te], np.ones(len(te))])
            yp2_te = X2_te @ th2
            ss_r = np.sum((y[te] - yp2_te)**2)
            ss_t = np.sum((y[te] - np.mean(y[te]))**2)
            r2s.append(1 - ss_r/ss_t if ss_t > 0 else 0)
            sps.append(stats.spearmanr(yp2_te, y[te])[0])
        return np.mean(r2s), np.std(r2s), np.mean(sps), np.std(sps)

    # Three paths
    paths = [
        ('WKV→KVdim', wb, kvb, kvd),
        ('WKVdim→KV', wb, kvd, kvb),
        ('KVKVdim→W', kvb, kvd, wb),
    ]

    print(f"\nLinear vs Quadratic comparison:")
    print(f"{'Strategy':<40} {'R²':>7} {'CV R²':>14} {'CV ρ':>14}")
    print("-" * 80)
    print(f"{'Direct linear (y = αw + βkv + γkvd)':<40} {r2_D:>7.4f} {cv_D[0]:>6.4f}±{cv_D[1]:.4f} {cv_D[2]:>6.4f}±{cv_D[3]:.4f}")
    print(f"{'Direct quadratic (+ squared terms)':<40} {r2_Dq:>7.4f} {cv_Dq[0]:>6.4f}±{cv_Dq[1]:.4f} {cv_Dq[2]:>6.4f}±{cv_Dq[3]:.4f}")

    print(f"\nHierarchical quadratic composition (2 step quadratic fits):")
    print(f"{'Path':<30} {'Step1 R²':>9} {'Step2 R²':>9} {'CV R²':>14} {'CV ρ':>14}")
    print("-" * 85)
    for pname, xA, xB, xC in paths:
        r2_1, r2_2, yp_h = hierarchical(xA, xB, xC, y)
        cv_h = hierarchical_cv(xA, xB, xC, y)
        print(f"{pname:<30} {r2_1:>9.4f} {r2_2:>9.4f} {cv_h[0]:>6.4f}±{cv_h[1]:.4f} {cv_h[2]:>6.4f}±{cv_h[3]:.4f}")

    # Why do they differ?
    print(f"\n  Key analysis:")
    print(f"    - Linear (no power) hierarchical = linear additive direct (identical fit)")
    print(f"    - Quadratic hierarchical gives DIFFERENT fit than direct quadratic because")
    print(f"      the interaction terms differ: hierarchical builds xA·xB and (xA+xB)·xC,")
    print(f"      skipping xA·xC, xB·xC as separate terms (they're folded into step-2 quad).")

    # Direct with FULL quadratic (all features)
    X_full = np.column_stack([wb, kvb, kvd, wb**2, kvb**2, kvd**2,
                              wb*kvb, wb*kvd, kvb*kvd])
    th_full, r2_full, yp_full = lsq(X_full, y)
    cv_full = cv_lsq(X_full, y)
    print(f"\n{'Direct FULL quadratic (main+quad+all inter.)':<40} {r2_full:>7.4f} {cv_full[0]:>6.4f}±{cv_full[1]:.4f} {cv_full[2]:>6.4f}±{cv_full[3]:.4f}")

    # Parameter count for each strategy
    print(f"\n  Parameter count (including intercept):")
    print(f"    Direct linear:       4 params")
    print(f"    Direct quadratic:    7 params")
    print(f"    Direct full quad:   10 params")
    print(f"    Hier quadratic:      6 + 6 = 12 params  (more flexible but regularization harder)")

    # Nonlinear intermediate comparison ─ when is hierarchy better?
    # Scenario: limited calibration budget. Simulate N_cal ∈ {10, 15, 20, 27, 40}
    print(f"\n  Limited-budget calibration comparison (simulating n_cal = 27):")
    n_trials = 200
    results_budget = {}
    for n_cal in [10, 15, 20, 27, 40, 60]:
        r2_direct_lin, r2_direct_q, r2_hier_best = [], [], []
        sp_direct_lin, sp_direct_q, sp_hier_best = [], [], []
        for t in range(n_trials):
            rng = np.random.RandomState(t)
            cal_idx = rng.choice(n, n_cal, replace=False)
            test_idx = np.array([i for i in range(n) if i not in set(cal_idx)])
            y_cal, y_te = y[cal_idx], y[test_idx]

            # Direct linear
            X_lin_cal = np.column_stack([wb[cal_idx], kvb[cal_idx], kvd[cal_idx], np.ones(n_cal)])
            X_lin_te = np.column_stack([wb[test_idx], kvb[test_idx], kvd[test_idx], np.ones(len(test_idx))])
            try:
                th, _, _, _ = np.linalg.lstsq(X_lin_cal, y_cal, rcond=None)
                yp = X_lin_te @ th
                ss_r = np.sum((y_te - yp)**2); ss_t = np.sum((y_te - y_te.mean())**2)
                r2_direct_lin.append(1 - ss_r/ss_t if ss_t > 0 else 0)
                sp_direct_lin.append(stats.spearmanr(yp, y_te)[0])
            except: pass

            # Direct quadratic (if enough samples)
            if n_cal >= 8:
                X_q_cal = np.column_stack([wb[cal_idx], kvb[cal_idx], kvd[cal_idx],
                                           wb[cal_idx]**2, kvb[cal_idx]**2, kvd[cal_idx]**2, np.ones(n_cal)])
                X_q_te = np.column_stack([wb[test_idx], kvb[test_idx], kvd[test_idx],
                                          wb[test_idx]**2, kvb[test_idx]**2, kvd[test_idx]**2, np.ones(len(test_idx))])
                try:
                    th, _, _, _ = np.linalg.lstsq(X_q_cal, y_cal, rcond=None)
                    yp = X_q_te @ th
                    ss_r = np.sum((y_te - yp)**2); ss_t = np.sum((y_te - y_te.mean())**2)
                    r2_direct_q.append(1 - ss_r/ss_t if ss_t > 0 else 0)
                    sp_direct_q.append(stats.spearmanr(yp, y_te)[0])
                except: pass

            # Hierarchical (quadratic WKV→KVdim)
            if n_cal >= 8:
                X1c = np.column_stack([wb[cal_idx], kvb[cal_idx], wb[cal_idx]**2,
                                       kvb[cal_idx]**2, wb[cal_idx]*kvb[cal_idx], np.ones(n_cal)])
                try:
                    th1, _, _, _ = np.linalg.lstsq(X1c, y_cal, rcond=None)
                    yp1_cal = X1c @ th1
                    X1t = np.column_stack([wb[test_idx], kvb[test_idx], wb[test_idx]**2,
                                           kvb[test_idx]**2, wb[test_idx]*kvb[test_idx], np.ones(len(test_idx))])
                    yp1_te = X1t @ th1
                    X2c = np.column_stack([yp1_cal, kvd[cal_idx], yp1_cal**2,
                                           kvd[cal_idx]**2, yp1_cal*kvd[cal_idx], np.ones(n_cal)])
                    th2, _, _, _ = np.linalg.lstsq(X2c, y_cal, rcond=None)
                    X2t = np.column_stack([yp1_te, kvd[test_idx], yp1_te**2,
                                           kvd[test_idx]**2, yp1_te*kvd[test_idx], np.ones(len(test_idx))])
                    yp_h = X2t @ th2
                    ss_r = np.sum((y_te - yp_h)**2); ss_t = np.sum((y_te - y_te.mean())**2)
                    r2_hier_best.append(1 - ss_r/ss_t if ss_t > 0 else 0)
                    sp_hier_best.append(stats.spearmanr(yp_h, y_te)[0])
                except: pass

        results_budget[n_cal] = {
            'dir_lin': (np.mean(r2_direct_lin), np.std(r2_direct_lin), np.mean(sp_direct_lin)),
            'dir_q': (np.mean(r2_direct_q) if r2_direct_q else np.nan,
                      np.std(r2_direct_q) if r2_direct_q else np.nan,
                      np.mean(sp_direct_q) if sp_direct_q else np.nan),
            'hier': (np.mean(r2_hier_best) if r2_hier_best else np.nan,
                     np.std(r2_hier_best) if r2_hier_best else np.nan,
                     np.mean(sp_hier_best) if sp_hier_best else np.nan),
        }

    print(f"\n{'n_cal':>6} | {'Direct Linear':>22} | {'Direct Quadratic':>23} | {'Hierarchical Quad':>24}")
    print(f"{'':6} | {'R²':>7} {'σ':>6} {'ρ':>6} | {'R²':>7} {'σ':>6} {'ρ':>6} | {'R²':>7} {'σ':>6} {'ρ':>6}")
    print("-" * 90)
    for n_cal, r in results_budget.items():
        dl = r['dir_lin']; dq = r['dir_q']; hi = r['hier']
        print(f"{n_cal:>6} | {dl[0]:>7.4f} {dl[1]:>6.4f} {dl[2]:>6.4f} | "
              f"{dq[0]:>7.4f} {dq[1]:>6.4f} {dq[2]:>6.4f} | "
              f"{hi[0]:>7.4f} {hi[1]:>6.4f} {hi[2]:>6.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 4: COMBINED BEST MODEL
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("PART 4: RECOMMENDED CALIBRATION MODEL")
    print("=" * 90)

    # Best model: add wbits² and wbits×(kv/kvdim) interactions (M8)
    X_rec = np.column_stack([wb, kvb, kvd, wb**2, wb*kvb, wb*kvd])
    th_rec, r2_rec, yp_rec = lsq(X_rec, y)
    cv_rec = cv_lsq(X_rec, y)

    print(f"\n  Recommended: y = β0 + β1·w + β2·kv + β3·kvd + β4·w² + β5·w·kv + β6·w·kvd")
    print(f"    β0 (intercept): {th_rec[-1]:+.6f}")
    print(f"    β1 (w):         {th_rec[0]:+.6f}")
    print(f"    β2 (kv):        {th_rec[1]:+.6f}")
    print(f"    β3 (kvdim):     {th_rec[2]:+.6f}")
    print(f"    β4 (w²):        {th_rec[3]:+.6f}")
    print(f"    β5 (w·kv):      {th_rec[4]:+.6f}")
    print(f"    β6 (w·kvdim):   {th_rec[5]:+.6f}")
    print(f"    R²={r2_rec:.4f}, CV R²={cv_rec[0]:.4f}±{cv_rec[1]:.4f}, CV ρ={cv_rec[2]:.4f}±{cv_rec[3]:.4f}")

    # Residuals in low wbits
    resid_rec = y - yp_rec
    print(f"\n  Residuals with recommended model:")
    print(f"    Low wbits (<2.75):  MAE={np.mean(np.abs(resid_rec[low_mask])):.5f}, "
          f"max|r|={np.max(np.abs(resid_rec[low_mask])):.5f}")
    print(f"    High wbits (≥2.75): MAE={np.mean(np.abs(resid_rec[hi_mask])):.5f}, "
          f"max|r|={np.max(np.abs(resid_rec[hi_mask])):.5f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 5: VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Row 1: Model comparison
    models_to_plot = [('A: Linear', yp_A), ('C: +W²', yp_C), ('G: Full quad+inter', yp_G), ('Rec: W²+W·kv+W·kvd', yp_rec)]
    for i, (label, yp_m) in enumerate(models_to_plot):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(yp_m[low_mask], y[low_mask], alpha=0.6, s=15, c='crimson', label=f'low wbits (<2.75)')
        ax.scatter(yp_m[hi_mask], y[hi_mask], alpha=0.6, s=15, c='steelblue', label=f'high wbits (≥2.75)')
        lim = [0, max(y.max(), yp_m.max()) * 1.05]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_xlabel('Predicted JSD')
        ax.set_ylabel('Actual JSD')
        r2 = 1 - np.sum((y - yp_m)**2) / np.sum((y - y.mean())**2)
        ax.set_title(f'{label}\nR²={r2:.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 2: Residual distributions
    for i, (label, yp_m) in enumerate(models_to_plot):
        ax = fig.add_subplot(gs[1, i])
        rs = y - yp_m
        ax.scatter(wb, rs, alpha=0.5, s=15, c=np.where(low_mask, 'crimson', 'steelblue'))
        ax.axhline(0, color='black', ls='--', lw=1)
        ax.axvline(2.75, color='gray', ls=':', lw=1)
        ax.set_xlabel('wbits')
        ax.set_ylabel('Residual')
        ax.set_title(f'{label}: Resid vs wbits')
        ax.grid(True, alpha=0.3)

    # Row 3: Budget comparison
    ax = fig.add_subplot(gs[2, 0:2])
    ns = list(results_budget.keys())
    dl_r2 = [results_budget[n]['dir_lin'][0] for n in ns]
    dq_r2 = [results_budget[n]['dir_q'][0] for n in ns]
    hi_r2 = [results_budget[n]['hier'][0] for n in ns]
    dl_sd = [results_budget[n]['dir_lin'][1] for n in ns]
    dq_sd = [results_budget[n]['dir_q'][1] for n in ns]
    hi_sd = [results_budget[n]['hier'][1] for n in ns]
    ax.errorbar(ns, dl_r2, yerr=dl_sd, marker='o', label='Direct Linear', capsize=4)
    ax.errorbar(ns, dq_r2, yerr=dq_sd, marker='s', label='Direct Quadratic', capsize=4)
    ax.errorbar(ns, hi_r2, yerr=hi_sd, marker='^', label='Hierarchical Quad', capsize=4)
    ax.axvline(27, color='red', ls=':', alpha=0.5, label='Target 27 samples')
    ax.set_xlabel('Calibration samples')
    ax.set_ylabel('Test R²')
    ax.set_title('Calibration budget vs Test R² (200 trials)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2:4])
    dl_sp = [results_budget[n]['dir_lin'][2] for n in ns]
    dq_sp = [results_budget[n]['dir_q'][2] for n in ns]
    hi_sp = [results_budget[n]['hier'][2] for n in ns]
    ax.plot(ns, dl_sp, marker='o', label='Direct Linear')
    ax.plot(ns, dq_sp, marker='s', label='Direct Quadratic')
    ax.plot(ns, hi_sp, marker='^', label='Hierarchical Quad')
    ax.axvline(27, color='red', ls=':', alpha=0.5)
    ax.set_xlabel('Calibration samples')
    ax.set_ylabel('Test Spearman ρ')
    ax.set_title('Calibration budget vs Test Spearman')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 4: Low wbits correction strategies
    strategies_plot = [
        ('A: Linear', resid_A),
        ('C: + W²', resid_C),
        ('D: 2^(-w)', resid_D),
        ('Rec: W²+W·kv+W·kvd', resid_rec),
    ]
    for i, (label, rs) in enumerate(strategies_plot):
        ax = fig.add_subplot(gs[3, i])
        ax.scatter(wb, rs, alpha=0.5, s=15, c='purple')
        ax.axhline(0, color='black', ls='--', lw=1)
        mae_lo = np.mean(np.abs(rs[low_mask]))
        mae_hi = np.mean(np.abs(rs[hi_mask]))
        ax.set_xlabel('wbits')
        ax.set_ylabel('Residual')
        ax.set_title(f'{label}\nMAE_lo={mae_lo:.4f}, MAE_hi={mae_hi:.4f}')
        ax.grid(True, alpha=0.3)

    save_path = '/NAS/SJ/actquant/search/save/result/analysis_interaction_hierarchy.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {save_path}")
    plt.close()

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)

if __name__ == '__main__':
    main()
