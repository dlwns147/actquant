"""
Pareto Frontier Combination Analysis
=====================================
Analyzes the additive approximation of combining independent optimization
method Pareto frontiers (W quantization, KV quantization, KV dimension pruning).

Uses 200 random samples from each combination with actual JSD measurements.
"""

import os, csv, json
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.simplefilter("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
BASE = '/NAS/SJ/actquant/search/save/result'

EXPERIMENTS = {
    'awq_w_kv':        '2604162010_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr',
    'awq_w_kvdim':     '2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim',
    'awq_w_kv_kvdim':  '2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim',
    'awq_kv_kvdim':    '2604162013_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_kv_expr_kv_dim',
    'hqq_w_kv_kvdim':  '2604162019_Llama-3.1-8B-Instruct__0_0_hqq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim',
    'hqq_w_kv':        '2604162034_Llama-3.1-8B-Instruct__0_0_hqq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr',
    'hqq_kv_kvdim':    '2604162045_Llama-3.1-8B-Instruct__0_0_hqq_kivi_wikitext2_1_kv_scale_0seed_kv_expr_kv_dim',
    'hqq_w_kvdim':     '2604162045_Llama-3.1-8B-Instruct__0_0_hqq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim',
}

COMP_KEYS = ['wbits', 'kvbits', 'kbits', 'vbits', 'kvdim', 'kdim', 'vdim',
             'eff_kvbits', 'eff_kbits', 'eff_vbits', 'memory', 'n_token']
N_COMP = len(COMP_KEYS)

def load_csv(name):
    path = os.path.join(BASE, EXPERIMENTS[name], 'results.csv')
    data = []
    with open(path, 'r') as f:
        for row in csv.reader(f):
            if row:  # skip empty rows
                data.append([float(x) for x in row])
    return data

def parse_experiment(name):
    """Parse CSV: 12 complexity rows + 1 actual JSD row (remaining rows are empty/ignored)."""
    data = load_csv(name)
    result = {}
    result['comp'] = {}
    for i, key in enumerate(COMP_KEYS):
        if i < len(data):
            result['comp'][key] = np.array(data[i])
    result['jsd_actual'] = np.array(data[N_COMP])  # row 12
    result['n_samples'] = len(result['jsd_actual'])
    return result

def corr_analysis(x, y):
    pearson_r, _ = stats.pearsonr(x, y)
    spearman_r, _ = stats.spearmanr(x, y)
    kendall_tau, _ = stats.kendalltau(x, y)
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    return {
        'pearson': pearson_r, 'spearman': spearman_r, 'kendall': kendall_tau,
        'r2': r_value**2, 'slope': slope, 'intercept': intercept,
    }

def lsq_fit(X, y, intercept=True):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if intercept:
        X = np.column_stack([X, np.ones(len(y))])
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ theta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    return theta, r2, y_pred

def cv_lsq(X, y, k=5, intercept=True):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(y)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    fold = n // k
    r2s, sps = [], []
    y_pred = np.zeros(n)
    for f in range(k):
        te = idx[f*fold:(f+1)*fold]
        tr = np.concatenate([idx[:f*fold], idx[(f+1)*fold:]])
        Xtr = np.column_stack([X[tr], np.ones(len(tr))]) if intercept else X[tr]
        Xte = np.column_stack([X[te], np.ones(len(te))]) if intercept else X[te]
        th, _, _, _ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        yp = Xte @ th
        y_pred[te] = yp
        ss_r = np.sum((y[te] - yp)**2)
        ss_t = np.sum((y[te] - np.mean(y[te]))**2)
        r2s.append(1 - ss_r / ss_t if ss_t > 0 else 0)
        sps.append(stats.spearmanr(yp, y[te])[0])
    return np.mean(r2s), np.std(r2s), np.mean(sps), np.std(sps), y_pred

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 90)
    print("LOADING EXPERIMENTS")
    print("=" * 90)

    exps = {}
    for name in EXPERIMENTS:
        exps[name] = parse_experiment(name)
        e = exps[name]
        print(f"  {name}: {e['n_samples']} samples, "
              f"JSD=[{e['jsd_actual'].min():.4f}, {e['jsd_actual'].max():.4f}], "
              f"wbits=[{e['comp']['wbits'].min():.2f}, {e['comp']['wbits'].max():.2f}], "
              f"kvdim=[{e['comp']['kvdim'].min():.1f}, {e['comp']['kvdim'].max():.1f}]")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1: COMPLEXITY → JSD PREDICTABILITY
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 1: COMPLEXITY METRICS → JSD PREDICTABILITY")
    print("  Testing: how well do complexity metrics predict actual JSD?")
    print("=" * 90)

    for name in ['awq_w_kv', 'awq_w_kvdim', 'awq_kv_kvdim', 'awq_w_kv_kvdim']:
        e = exps[name]
        y = e['jsd_actual']
        short = name.replace('awq_', '').upper().replace('_', '+')
        print(f"\n[{short}]")

        # Individual complexity vs JSD
        for key in ['wbits', 'kvbits', 'eff_kvbits', 'kvdim', 'memory']:
            x = e['comp'][key]
            if np.std(x) < 1e-10:  # constant
                continue
            r = corr_analysis(x, y)
            print(f"  {key:>12s}: Pearson={r['pearson']:.4f}, Spearman={r['spearman']:.4f}, R²={r['r2']:.4f}")

        # Multivariate: all varying complexity metrics → JSD
        varying = [k for k in ['wbits', 'kvbits', 'kvdim'] if np.std(e['comp'][k]) > 1e-10]
        if len(varying) > 0:
            X = np.column_stack([e['comp'][k] for k in varying])
            th, r2, yp = lsq_fit(X, y)
            cv_r2, cv_r2_std, cv_sp, cv_sp_std, _ = cv_lsq(X, y)
            print(f"  {'Multi':>12s} ({'+'.join(varying)}): R²={r2:.4f}, CV R²={cv_r2:.4f}±{cv_r2_std:.4f}, CV ρ={cv_sp:.4f}±{cv_sp_std:.4f}")
            for i, k in enumerate(varying):
                print(f"    β_{k} = {th[i]:.6f}")
            print(f"    intercept = {th[-1]:.6f}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2: ADDITIVITY OF LOSS (SEPARABILITY TEST)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 2: ADDITIVITY / SEPARABILITY TEST")
    print("  Question: Is JSD(w,kv,kvdim) ≈ f(wbits) + g(kvbits) + h(kvdim)?")
    print("  We test with polynomial features and interaction terms.")
    print("=" * 90)

    if 'awq_w_kv_kvdim' in exps:
        e = exps['awq_w_kv_kvdim']
        y = e['jsd_actual']
        wb = e['comp']['wbits']
        kvb = e['comp']['kvbits']  # or eff_kvbits
        kvd = e['comp']['kvdim']

        # Standardize for numerical stability
        wb_s = (wb - wb.mean()) / wb.std()
        kvb_s = (kvb - kvb.mean()) / kvb.std()
        kvd_s = (kvd - kvd.mean()) / kvd.std()

        # Model 1: Pure additive (linear)
        X1 = np.column_stack([wb_s, kvb_s, kvd_s])
        th1, r2_1, yp1 = lsq_fit(X1, y)
        cv1 = cv_lsq(X1, y)

        # Model 2: Additive with quadratics
        X2 = np.column_stack([wb_s, kvb_s, kvd_s, wb_s**2, kvb_s**2, kvd_s**2])
        th2, r2_2, yp2 = lsq_fit(X2, y)
        cv2 = cv_lsq(X2, y)

        # Model 3: Linear + 2nd-order interactions
        X3 = np.column_stack([wb_s, kvb_s, kvd_s, wb_s*kvb_s, wb_s*kvd_s, kvb_s*kvd_s])
        th3, r2_3, yp3 = lsq_fit(X3, y)
        cv3 = cv_lsq(X3, y)

        # Model 4: Full quadratic (additive + interactions)
        X4 = np.column_stack([wb_s, kvb_s, kvd_s, wb_s**2, kvb_s**2, kvd_s**2,
                              wb_s*kvb_s, wb_s*kvd_s, kvb_s*kvd_s])
        th4, r2_4, yp4 = lsq_fit(X4, y)
        cv4 = cv_lsq(X4, y)

        print(f"\n{'Model':<40} {'R²':>7} {'CV R²':>12} {'CV ρ':>12}")
        print("-" * 75)
        print(f"{'Linear additive':<40} {r2_1:>7.4f} {cv1[0]:>6.4f}±{cv1[1]:.4f} {cv1[2]:>6.4f}±{cv1[3]:.4f}")
        print(f"{'Additive + quadratics':<40} {r2_2:>7.4f} {cv2[0]:>6.4f}±{cv2[1]:.4f} {cv2[2]:>6.4f}±{cv2[3]:.4f}")
        print(f"{'Linear + interactions':<40} {r2_3:>7.4f} {cv3[0]:>6.4f}±{cv3[1]:.4f} {cv3[2]:>6.4f}±{cv3[3]:.4f}")
        print(f"{'Full quadratic':<40} {r2_4:>7.4f} {cv4[0]:>6.4f}±{cv4[1]:.4f} {cv4[2]:>6.4f}±{cv4[3]:.4f}")

        # F-test: interactions vs additive
        n = len(y)
        p_add, p_inter = 4, 7
        ss_add = np.sum((y - yp1)**2)
        ss_inter = np.sum((y - yp3)**2)
        f_stat = ((ss_add - ss_inter) / (p_inter - p_add)) / (ss_inter / (n - p_inter))
        f_p = 1 - stats.f.cdf(f_stat, p_inter - p_add, n - p_inter)

        print(f"\nF-test (interactions vs additive):")
        print(f"  F={f_stat:.4f}, p={f_p:.2e}, ΔR²={r2_3 - r2_1:.6f}")
        print(f"  → Interactions {'SIGNIFICANT' if f_p < 0.05 else 'NOT significant'}")
        if f_p < 0.05 and (r2_3 - r2_1) < 0.02:
            print(f"    BUT practical gain is small (ΔR²<0.02): additive is sufficient for ranking")

        # Interaction coefficients
        print(f"\nInteraction coefficients (standardized):")
        labels_inter = ['W×KV', 'W×KVdim', 'KV×KVdim']
        for i, lbl in enumerate(labels_inter):
            print(f"  β_{lbl} = {th3[3+i]:.6f}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3: VARIANCE DECOMPOSITION
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 3: VARIANCE DECOMPOSITION (ANOVA-style)")
    print("  How much JSD variance is explained by each optimization method?")
    print("=" * 90)

    if 'awq_w_kv_kvdim' in exps:
        e = exps['awq_w_kv_kvdim']
        y = e['jsd_actual']
        wb = e['comp']['wbits'].reshape(-1, 1)
        kvb = e['comp']['kvbits'].reshape(-1, 1)
        kvd = e['comp']['kvdim'].reshape(-1, 1)

        _, r2_w, _ = lsq_fit(wb, y)
        _, r2_kv, _ = lsq_fit(kvb, y)
        _, r2_kvd, _ = lsq_fit(kvd, y)
        _, r2_wkv, _ = lsq_fit(np.column_stack([wb, kvb]), y)
        _, r2_wkvd, _ = lsq_fit(np.column_stack([wb, kvd]), y)
        _, r2_kvkvd, _ = lsq_fit(np.column_stack([kvb, kvd]), y)
        _, r2_all, _ = lsq_fit(np.column_stack([wb, kvb, kvd]), y)

        print(f"\n  Individual R²:")
        print(f"    W alone:      {r2_w:.4f}  ({r2_w*100:.1f}%)")
        print(f"    KV alone:     {r2_kv:.4f}  ({r2_kv*100:.1f}%)")
        print(f"    KVdim alone:  {r2_kvd:.4f}  ({r2_kvd*100:.1f}%)")

        print(f"\n  Pairwise R²:")
        print(f"    W+KV:         {r2_wkv:.4f}  ({r2_wkv*100:.1f}%)")
        print(f"    W+KVdim:      {r2_wkvd:.4f}  ({r2_wkvd*100:.1f}%)")
        print(f"    KV+KVdim:     {r2_kvkvd:.4f}  ({r2_kvkvd*100:.1f}%)")

        print(f"\n  Full model R²:")
        print(f"    W+KV+KVdim:   {r2_all:.4f}  ({r2_all*100:.1f}%)")

        # Semi-partial R² (unique contribution)
        dw = r2_all - r2_kvkvd
        dkv = r2_all - r2_wkvd
        dkvd = r2_all - r2_wkv
        shared = r2_all - dw - dkv - dkvd

        print(f"\n  Unique variance (semi-partial R²):")
        print(f"    ΔR²(W | KV,KVdim)     = {dw:.4f}  — W 고유 기여")
        print(f"    ΔR²(KV | W,KVdim)     = {dkv:.4f}  — KV 고유 기여")
        print(f"    ΔR²(KVdim | W,KV)     = {dkvd:.4f}  — KVdim 고유 기여")
        print(f"    Shared variance        = {shared:.4f}")
        print(f"    Unexplained            = {1 - r2_all:.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4: 2-WAY vs 3-WAY COMPARISON AT EQUAL COMPLEXITY
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 4: 2-WAY vs 3-WAY — QUALITY AT COMPARABLE COMPLEXITY")
    print("=" * 90)

    pairs = [
        ('awq_w_kv', 'W+KV (kvdim=128)'),
        ('awq_w_kvdim', 'W+KVdim (kvbits=4)'),
        ('awq_kv_kvdim', 'KV+KVdim (wbits≈4.25)'),
        ('awq_w_kv_kvdim', 'W+KV+KVdim (3-way)'),
    ]

    print(f"\n{'Combination':<28} {'JSD mean':>9} {'JSD std':>8} {'JSD min':>8} {'JSD Q25':>8} {'JSD med':>8} {'JSD Q75':>8} {'JSD max':>8}")
    print("-" * 95)
    for name, label in pairs:
        e = exps[name]
        y = e['jsd_actual']
        q25, q50, q75 = np.percentile(y, [25, 50, 75])
        print(f"{label:<28} {y.mean():>9.4f} {y.std():>8.4f} {y.min():>8.4f} {q25:>8.4f} {q50:>8.4f} {q75:>8.4f} {y.max():>8.4f}")

    # Memory distribution comparison
    print(f"\n{'Combination':<28} {'Mem mean(GB)':>12} {'Mem std':>10} {'Mem min':>10} {'Mem max':>10}")
    print("-" * 75)
    for name, label in pairs:
        e = exps[name]
        mem = e['comp']['memory'] / 1e9
        print(f"{label:<28} {mem.mean():>12.3f} {mem.std():>10.3f} {mem.min():>10.3f} {mem.max():>10.3f}")

    # For comparable memory ranges, which gives better JSD?
    if 'awq_w_kv' in exps and 'awq_w_kv_kvdim' in exps:
        e2 = exps['awq_w_kv']
        e3 = exps['awq_w_kv_kvdim']
        # Find overlapping memory range
        mem2 = e2['comp']['memory']
        mem3 = e3['comp']['memory']
        lo = max(mem2.min(), mem3.min())
        hi = min(mem2.max(), mem3.max())
        mask2 = (mem2 >= lo) & (mem2 <= hi)
        mask3 = (mem3 >= lo) & (mem3 <= hi)
        if mask2.sum() > 0 and mask3.sum() > 0:
            print(f"\n  Memory overlap [{lo/1e9:.3f}, {hi/1e9:.3f}] GB:")
            print(f"    W+KV:       n={mask2.sum()}, JSD mean={e2['jsd_actual'][mask2].mean():.4f}")
            print(f"    W+KV+KVdim: n={mask3.sum()}, JSD mean={e3['jsd_actual'][mask3].mean():.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5: HQQ SEARCH → AWQ BENCHMARK TRANSFER
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 5: HQQ (SEARCH) vs AWQ (BENCHMARK) — METHOD TRANSFER")
    print("  Search uses HQQ. Benchmark uses AWQ. How well does rank transfer?")
    print("=" * 90)

    hqq_awq = [
        ('hqq_w_kv', 'awq_w_kv', 'W+KV'),
        ('hqq_w_kvdim', 'awq_w_kvdim', 'W+KVdim'),
        ('hqq_kv_kvdim', 'awq_kv_kvdim', 'KV+KVdim'),
        ('hqq_w_kv_kvdim', 'awq_w_kv_kvdim', 'W+KV+KVdim'),
    ]

    print(f"\n{'Combo':<12} {'HQQ JSD':>9} {'AWQ JSD':>9} {'HQQ std':>9} {'AWQ std':>9} {'Ratio':>7}")
    print("-" * 60)
    for hname, aname, label in hqq_awq:
        h = exps[hname]; a = exps[aname]
        ratio = h['jsd_actual'].mean() / a['jsd_actual'].mean()
        print(f"{label:<12} {h['jsd_actual'].mean():>9.4f} {a['jsd_actual'].mean():>9.4f} "
              f"{h['jsd_actual'].std():>9.4f} {a['jsd_actual'].std():>9.4f} {ratio:>7.3f}")

    # Within each method, complexity→JSD prediction quality
    print(f"\n{'Combo':<12} {'Method':>6} {'R²(wbits)':>10} {'R²(kvbits)':>11} {'R²(kvdim)':>10} {'R²(all)':>8}")
    print("-" * 65)
    for hname, aname, label in hqq_awq:
        for method, ename in [('HQQ', hname), ('AWQ', aname)]:
            e = exps[ename]
            y = e['jsd_actual']
            r2s = {}
            for key in ['wbits', 'kvbits', 'kvdim']:
                x = e['comp'][key]
                if np.std(x) < 1e-10:
                    r2s[key] = '  const'
                else:
                    _, r2, _ = lsq_fit(x.reshape(-1, 1), y)
                    r2s[key] = f'{r2:>7.4f}'

            varying = [k for k in ['wbits', 'kvbits', 'kvdim'] if np.std(e['comp'][k]) > 1e-10]
            if len(varying) > 0:
                X = np.column_stack([e['comp'][k] for k in varying])
                _, r2_all, _ = lsq_fit(X, y)
                r2_all_s = f'{r2_all:.4f}'
            else:
                r2_all_s = '   N/A'
            print(f"{label:<12} {method:>6} {r2s['wbits']:>10} {r2s['kvbits']:>11} {r2s['kvdim']:>10} {r2_all_s:>8}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 6: RANK PREDICTION ACROSS METHODS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 6: COMPLEXITY-BASED RANK PREDICTION")
    print("  If we rank by additive complexity, how well does it match actual JSD rank?")
    print("=" * 90)

    for name, label in [('awq_w_kv_kvdim', 'W+KV+KVdim (AWQ)'), ('hqq_w_kv_kvdim', 'W+KV+KVdim (HQQ)')]:
        e = exps[name]
        y = e['jsd_actual']

        # Various surrogate rankings
        rankings = {}

        # 1. Memory (lower = fewer bits = more quant = higher JSD)
        rankings['memory'] = e['comp']['memory']

        # 2. eff_kvbits
        if np.std(e['comp']['eff_kvbits']) > 1e-10:
            rankings['eff_kvbits'] = e['comp']['eff_kvbits']

        # 3. wbits + kvbits
        if np.std(e['comp']['wbits']) > 1e-10 and np.std(e['comp']['kvbits']) > 1e-10:
            rankings['wbits+kvbits'] = e['comp']['wbits'] + e['comp']['kvbits']

        # 4. wbits + eff_kvbits
        if np.std(e['comp']['wbits']) > 1e-10 and np.std(e['comp']['eff_kvbits']) > 1e-10:
            rankings['wbits+eff_kvbits'] = e['comp']['wbits'] + e['comp']['eff_kvbits']

        # 5. LSQ-calibrated
        varying = [k for k in ['wbits', 'kvbits', 'kvdim'] if np.std(e['comp'][k]) > 1e-10]
        if len(varying) > 0:
            X = np.column_stack([e['comp'][k] for k in varying])
            _, _, yp = lsq_fit(X, y)
            rankings[f'LSQ({"+".join(varying)})'] = yp

        print(f"\n[{label}]")
        print(f"  {'Ranking Method':<30} {'Spearman ρ':>11} {'Kendall τ':>10} {'Top-10 overlap':>15}")
        print("  " + "-" * 70)
        for rname, rvals in rankings.items():
            sp = stats.spearmanr(rvals, y)[0]
            kt = stats.kendalltau(rvals, y)[0]
            # Sign: lower complexity → lower JSD (negative corr) or higher → higher (positive)
            # We want the absolute direction to be correct
            # For complexity metrics, lower values = more aggressive quant = higher JSD
            # So correlation should be negative: as complexity decreases, JSD increases
            rank_r = stats.rankdata(-rvals)  # rank by decreasing complexity
            rank_a = stats.rankdata(y)  # rank by increasing JSD
            top10_r = set(np.argsort(rvals)[:10])
            top10_a = set(np.argsort(y)[:10])
            overlap = len(top10_r & top10_a)
            print(f"  {rname:<30} {sp:>11.4f} {kt:>10.4f} {overlap:>10d}/10")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 7: CALIBRATION FOR 27-SAMPLE DESIGN (3×3×3)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 7: CALIBRATION DESIGN — 3×3×3 = 27 SAMPLES")
    print("  Simulating: sample 3 per Pareto frontier, measure 27, calibrate.")
    print("=" * 90)

    if 'awq_w_kv_kvdim' in exps:
        e = exps['awq_w_kv_kvdim']
        y = e['jsd_actual']
        wb = e['comp']['wbits']
        kvb = e['comp']['kvbits']
        kvd = e['comp']['kvdim']

        # Simulate: select 27 samples using quantile-like sampling, use rest for testing
        # Sort by each complexity metric and pick 3 quantile positions
        n = len(y)
        np.random.seed(42)

        n_trials = 100
        test_spearman_naive = []
        test_spearman_cal = []
        test_r2_cal = []

        for trial in range(n_trials):
            # Select ~27 calibration samples via stratified sampling
            # Tercile stratification on wbits and eff_kvbits
            wb_terciles = np.percentile(wb, [33.3, 66.7])
            kvb_terciles = np.percentile(kvb, [33.3, 66.7])
            kvd_terciles = np.percentile(kvd, [33.3, 66.7])

            cal_indices = set()
            for wb_lo, wb_hi in [(wb.min()-1, wb_terciles[0]), (wb_terciles[0], wb_terciles[1]), (wb_terciles[1], wb.max()+1)]:
                for kvb_lo, kvb_hi in [(kvb.min()-1, kvb_terciles[0]), (kvb_terciles[0], kvb_terciles[1]), (kvb_terciles[1], kvb.max()+1)]:
                    for kvd_lo, kvd_hi in [(kvd.min()-1, kvd_terciles[0]), (kvd_terciles[0], kvd_terciles[1]), (kvd_terciles[1], kvd.max()+1)]:
                        mask = (wb >= wb_lo) & (wb < wb_hi) & (kvb >= kvb_lo) & (kvb < kvb_hi) & (kvd >= kvd_lo) & (kvd < kvd_hi)
                        candidates = np.where(mask)[0]
                        if len(candidates) > 0:
                            choice = np.random.choice(candidates, min(1, len(candidates)), replace=False)
                            cal_indices.update(choice)

            cal_idx = np.array(sorted(cal_indices))
            test_idx = np.array([i for i in range(n) if i not in cal_indices])

            if len(cal_idx) < 5 or len(test_idx) < 10:
                continue

            # Naive: use complexity sum as proxy
            proxy = wb + kvb - kvd/128 * 4  # rough: lower kvdim → lower effective bits → higher JSD
            sp_naive = stats.spearmanr(proxy[test_idx], y[test_idx])[0]
            test_spearman_naive.append(sp_naive)

            # Calibrated: fit on cal samples, predict test
            X_all = np.column_stack([wb, kvb, kvd])
            X_cal = X_all[cal_idx]
            y_cal = y[cal_idx]
            X_test = X_all[test_idx]
            y_test = y[test_idx]

            X_cal_aug = np.column_stack([X_cal, np.ones(len(cal_idx))])
            X_test_aug = np.column_stack([X_test, np.ones(len(test_idx))])
            theta_cal, _, _, _ = np.linalg.lstsq(X_cal_aug, y_cal, rcond=None)
            y_pred_test = X_test_aug @ theta_cal

            sp_cal = stats.spearmanr(y_pred_test, y_test)[0]
            test_spearman_cal.append(sp_cal)

            ss_r = np.sum((y_test - y_pred_test)**2)
            ss_t = np.sum((y_test - np.mean(y_test))**2)
            test_r2_cal.append(1 - ss_r / ss_t)

        if test_spearman_naive:
            print(f"\n  Simulation ({n_trials} trials, ~27 cal / ~173 test):")
            print(f"    Naive (complexity sum) Spearman on test: {np.mean(test_spearman_naive):.4f}±{np.std(test_spearman_naive):.4f}")
            print(f"    Calibrated (LSQ)      Spearman on test: {np.mean(test_spearman_cal):.4f}±{np.std(test_spearman_cal):.4f}")
            print(f"    Calibrated (LSQ)      R² on test:       {np.mean(test_r2_cal):.4f}±{np.std(test_r2_cal):.4f}")
            print(f"    Improvement from calibration: Δρ={np.mean(test_spearman_cal)-np.mean(test_spearman_naive):+.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 8: RESIDUAL ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 8: RESIDUAL ANALYSIS — WHERE DOES ADDITIVITY FAIL?")
    print("=" * 90)

    if 'awq_w_kv_kvdim' in exps:
        e = exps['awq_w_kv_kvdim']
        y = e['jsd_actual']
        wb = e['comp']['wbits']
        kvb = e['comp']['kvbits']
        kvd = e['comp']['kvdim']

        X = np.column_stack([wb, kvb, kvd])
        _, _, yp = lsq_fit(X, y)
        resid = y - yp

        print(f"\n  Calibrated residuals (3-way, AWQ):")
        print(f"    Mean:    {resid.mean():.6f}")
        print(f"    Std:     {resid.std():.6f}")
        print(f"    Skew:    {stats.skew(resid):.4f}")
        print(f"    Kurtosis:{stats.kurtosis(resid):.4f}")

        _, p_shapiro = stats.shapiro(resid[:50])
        print(f"    Shapiro-Wilk p={p_shapiro:.4f} ({'normal' if p_shapiro > 0.05 else 'non-normal'})")

        # Residuals by region
        print(f"\n  Residuals by wbits region:")
        for lo, hi, label in [(2.0, 2.75, 'Low wbits [2,2.75)'),
                               (2.75, 3.5, 'Mid wbits [2.75,3.5)'),
                               (3.5, 4.5, 'High wbits [3.5,4.5)')]:
            mask = (wb >= lo) & (wb < hi)
            if mask.sum() == 0: continue
            print(f"    {label:<25} n={mask.sum():>3}, μ_res={resid[mask].mean():>+.6f}, σ_res={resid[mask].std():.6f}, μ_JSD={y[mask].mean():.4f}")

        print(f"\n  Residuals by kvdim region:")
        kvd_unique = np.unique(kvd)
        kvd_q = np.percentile(kvd, [33, 67])
        for lo, hi, label in [(kvd.min()-1, kvd_q[0], f'Low kvdim [~{kvd.min():.0f},{kvd_q[0]:.0f})'),
                               (kvd_q[0], kvd_q[1], f'Mid kvdim [{kvd_q[0]:.0f},{kvd_q[1]:.0f})'),
                               (kvd_q[1], kvd.max()+1, f'High kvdim [{kvd_q[1]:.0f},{kvd.max():.0f}]')]:
            mask = (kvd >= lo) & (kvd < hi)
            if mask.sum() == 0: continue
            print(f"    {label:<25} n={mask.sum():>3}, μ_res={resid[mask].mean():>+.6f}, σ_res={resid[mask].std():.6f}, μ_JSD={y[mask].mean():.4f}")

        # Identify worst-predicted samples
        worst_idx = np.argsort(np.abs(resid))[-5:]
        print(f"\n  Top 5 worst-predicted samples:")
        for i in worst_idx:
            print(f"    idx={i:>3}: JSD={y[i]:.4f}, pred={yp[i]:.4f}, resid={resid[i]:+.4f}, "
                  f"wbits={wb[i]:.2f}, kvbits={kvb[i]:.2f}, kvdim={kvd[i]:.1f}, mem={e['comp']['memory'][i]/1e9:.3f}GB")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 9: COMBINED SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("SECTION 9: COMBINED SUMMARY TABLE")
    print("=" * 90)

    print(f"\n{'Experiment':<18} {'Quant':>5} {'n':>4} {'R²(comp→JSD)':>13} {'Spearman(comp→JSD)':>19} {'JSD μ':>7} {'JSD σ':>7}")
    print("-" * 80)
    for name, label in [('awq_w_kv', 'W+KV'), ('awq_w_kvdim', 'W+KVdim'),
                         ('awq_kv_kvdim', 'KV+KVdim'), ('awq_w_kv_kvdim', 'W+KV+KVdim'),
                         ('hqq_w_kv', 'W+KV'), ('hqq_w_kvdim', 'W+KVdim'),
                         ('hqq_kv_kvdim', 'KV+KVdim'), ('hqq_w_kv_kvdim', 'W+KV+KVdim')]:
        e = exps[name]
        y = e['jsd_actual']
        quant = 'AWQ' if 'awq' in name else 'HQQ'
        varying = [k for k in ['wbits', 'kvbits', 'kvdim'] if np.std(e['comp'][k]) > 1e-10]
        if varying:
            X = np.column_stack([e['comp'][k] for k in varying])
            _, r2, yp = lsq_fit(X, y)
            sp = stats.spearmanr(yp, y)[0]
        else:
            r2, sp = 0, 0
        print(f"{label:<18} {quant:>5} {len(y):>4} {r2:>13.4f} {sp:>19.4f} {y.mean():>7.4f} {y.std():>7.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(28, 24))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Row 1: JSD vs complexity for each AWQ 2-way
    for idx, (name, label) in enumerate([
        ('awq_w_kv', 'W+KV'), ('awq_w_kvdim', 'W+KVdim'),
        ('awq_kv_kvdim', 'KV+KVdim'), ('awq_w_kv_kvdim', 'W+KV+KVdim')]):
        ax = fig.add_subplot(gs[0, idx])
        e = exps[name]
        y = e['jsd_actual']
        mem = e['comp']['memory'] / 1e9
        sc = ax.scatter(mem, y, alpha=0.5, s=15, c=e['comp']['wbits'], cmap='coolwarm', vmin=2.0, vmax=4.5)
        ax.set_xlabel('Memory (GB)')
        ax.set_ylabel('Actual JSD')
        ax.set_title(f'{label}')
        plt.colorbar(sc, ax=ax, label='wbits')
        ax.grid(True, alpha=0.3)

    # Row 2: Predicted vs Actual for each AWQ combo
    for idx, (name, label) in enumerate([
        ('awq_w_kv', 'W+KV'), ('awq_w_kvdim', 'W+KVdim'),
        ('awq_kv_kvdim', 'KV+KVdim'), ('awq_w_kv_kvdim', 'W+KV+KVdim')]):
        ax = fig.add_subplot(gs[1, idx])
        e = exps[name]
        y = e['jsd_actual']
        varying = [k for k in ['wbits', 'kvbits', 'kvdim'] if np.std(e['comp'][k]) > 1e-10]
        if not varying: continue
        X = np.column_stack([e['comp'][k] for k in varying])
        _, r2, yp = lsq_fit(X, y)
        sp = stats.spearmanr(yp, y)[0]
        ax.scatter(yp, y, alpha=0.5, s=15, c='steelblue')
        lim = [min(yp.min(), y.min())*0.9, max(yp.max(), y.max())*1.1]
        ax.plot(lim, lim, 'r--', lw=1.5, label='Perfect')
        ax.set_xlabel('LSQ Predicted JSD')
        ax.set_ylabel('Actual JSD')
        ax.set_title(f'{label}\nR²={r2:.4f}, ρ={sp:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 3: HQQ vs AWQ comparison
    for idx, (hname, aname, label) in enumerate([
        ('hqq_w_kv', 'awq_w_kv', 'W+KV'),
        ('hqq_w_kvdim', 'awq_w_kvdim', 'W+KVdim'),
        ('hqq_kv_kvdim', 'awq_kv_kvdim', 'KV+KVdim'),
        ('hqq_w_kv_kvdim', 'awq_w_kv_kvdim', 'W+KV+KVdim')]):
        ax = fig.add_subplot(gs[2, idx])
        h = exps[hname]; a = exps[aname]
        bins = np.linspace(min(h['jsd_actual'].min(), a['jsd_actual'].min()),
                          max(h['jsd_actual'].max(), a['jsd_actual'].max()), 40)
        ax.hist(a['jsd_actual'], bins=bins, alpha=0.5, color='tab:blue', label=f'AWQ (μ={a["jsd_actual"].mean():.3f})')
        ax.hist(h['jsd_actual'], bins=bins, alpha=0.5, color='tab:orange', label=f'HQQ (μ={h["jsd_actual"].mean():.3f})')
        ax.set_xlabel('Actual JSD')
        ax.set_ylabel('Count')
        ax.set_title(f'{label}: HQQ vs AWQ')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 4: Residual analysis for 3-way AWQ
    if 'awq_w_kv_kvdim' in exps:
        e = exps['awq_w_kv_kvdim']
        y = e['jsd_actual']
        wb = e['comp']['wbits']; kvb = e['comp']['kvbits']; kvd = e['comp']['kvdim']
        X = np.column_stack([wb, kvb, kvd])
        _, _, yp = lsq_fit(X, y)
        resid = y - yp

        # Residual histogram
        ax = fig.add_subplot(gs[3, 0])
        ax.hist(resid, bins=30, color='salmon', edgecolor='darkred', alpha=0.7)
        ax.axvline(0, color='black', ls='--', lw=1)
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')
        ax.set_title(f'3-way Residuals\nσ={resid.std():.4f}')
        ax.grid(True, alpha=0.3)

        # Residual vs wbits
        ax = fig.add_subplot(gs[3, 1])
        ax.scatter(wb, resid, alpha=0.5, s=15, c='purple')
        ax.axhline(0, color='black', ls='--', lw=1)
        ax.set_xlabel('wbits')
        ax.set_ylabel('Residual')
        ax.set_title('Resid vs wbits')
        ax.grid(True, alpha=0.3)

        # Residual vs kvbits
        ax = fig.add_subplot(gs[3, 2])
        ax.scatter(kvb, resid, alpha=0.5, s=15, c='teal')
        ax.axhline(0, color='black', ls='--', lw=1)
        ax.set_xlabel('kvbits')
        ax.set_ylabel('Residual')
        ax.set_title('Resid vs kvbits')
        ax.grid(True, alpha=0.3)

        # Residual vs kvdim
        ax = fig.add_subplot(gs[3, 3])
        ax.scatter(kvd, resid, alpha=0.5, s=15, c='orange')
        ax.axhline(0, color='black', ls='--', lw=1)
        ax.set_xlabel('kvdim')
        ax.set_ylabel('Residual')
        ax.set_title('Resid vs kvdim')
        ax.grid(True, alpha=0.3)

    save_path = '/NAS/SJ/actquant/search/save/result/analysis_split_combine.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {save_path}")
    plt.close()

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == '__main__':
    main()
