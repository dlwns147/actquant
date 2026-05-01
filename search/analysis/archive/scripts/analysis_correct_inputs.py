"""
Comprehensive analysis of additive loss approximation for combined Pareto frontiers.
CORRECTED: Uses per-method HQQ JSD as regression inputs (not architecture bits).

Pipeline:
1. Load search archives (.stats files) for W, KV, KVDIM methods
2. Compute Pareto fronts (matching post_search_split.py logic)
3. Match AWQ CSV samples to per-method Pareto points via comp values
4. Build dataset: X = [JSD_W_hqq, JSD_KV_hqq, JSD_KVdim_hqq], y = JSD_AWQ
5. Run all analyses: additive model, full quadratic, ANOVA, 27-grid, PF recovery, monotonicity
"""

import sys, json, csv, os
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.optimize import curve_fit

# ─── Pareto front computation (2D: minimize both objectives) ───────────────────
def pareto_front_2d(F):
    """Return indices of non-dominated points in 2D (minimize both dims)."""
    n = len(F)
    dominated = np.zeros(n, dtype=bool)
    order = np.argsort(F[:, 0])
    F_sorted = F[order]
    min_so_far = np.inf
    is_nd = np.zeros(n, dtype=bool)
    for i in range(n):
        if F_sorted[i, 1] < min_so_far:
            is_nd[i] = True
            min_so_far = F_sorted[i, 1]
    nd_in_sorted = np.where(is_nd)[0]
    return order[nd_in_sorted]

def load_archive_pareto(stats_path, comp_key, config, group_size):
    """Load archive+candidates, compute Pareto front, return F[:,0]=metric, F[:,1]=comp."""
    from utils.func import get_net_info
    with open(stats_path) as f:
        data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs   = [v[0] for v in archive]
    metrics = np.array([v[1] for v in archive])
    comps   = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
    F = np.column_stack((metrics, comps))
    # Sort by metric (matching load_expr logic)
    sort_idx = np.argsort(metrics)
    F = F[sort_idx]
    archs_sorted = [archs[i] for i in sort_idx]
    # Pareto filter (expr_front=True in shell script)
    nd_idx = pareto_front_2d(F)
    return F[nd_idx], [archs_sorted[i] for i in nd_idx]

# ─── Match comp value to Pareto front → return metric ─────────────────────────
def match_comp_to_metric(comp_val, pf_F, tol=1e-6):
    """Find Pareto point with matching comp value, return its metric (JSD_hqq)."""
    diffs = np.abs(pf_F[:, 1] - comp_val)
    idx = np.argmin(diffs)
    if diffs[idx] > tol:
        return None, diffs[idx]  # no match within tolerance
    return pf_F[idx, 0], diffs[idx]

# ─── Load AWQ CSV → (N_samples × 13 rows) matrix ─────────────────────────────
def load_awq_csv(csv_path):
    """Return matrix where rows[i] = metric row i, columns = samples."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]  # skip empty rows
    # Convert to float, pad if columns differ
    max_cols = max(len(r) for r in rows)
    mat = np.full((len(rows), max_cols), np.nan)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            try:
                mat[i, j] = float(v)
            except ValueError:
                pass
    return mat

# ─── Setup paths ─────────────────────────────────────────────────────────────
BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'

AWQ_3WAY  = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
AWQ_W_KV  = f'{BASE}/save/result/awq/2604162010_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr/results.csv'
AWQ_W_KVD = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim/results.csv'
AWQ_KV_KVD= f'{BASE}/save/result/awq/2604162013_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_kv_expr_kv_dim/results.csv'

CONFIG_PATH = f'{BASE}/config/llama.json'

print("Loading config...")
with open(CONFIG_PATH) as f:
    config_all = json.load(f)
config = config_all['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

# ─── Step 1: Load Pareto fronts ───────────────────────────────────────────────
print("Loading W Pareto front...")
pf_W, archs_W = load_archive_pareto(W_STATS, 'wbits', config, group_size)
print(f"  W PF: {len(pf_W)} points, metric [{pf_W[:,0].min():.4f}, {pf_W[:,0].max():.4f}], wbits [{pf_W[:,1].min():.4f}, {pf_W[:,1].max():.4f}]")

print("Loading KV Pareto front...")
pf_KV, archs_KV = load_archive_pareto(KV_STATS, 'kvbits', config, group_size)
print(f"  KV PF: {len(pf_KV)} points, metric [{pf_KV[:,0].min():.4f}, {pf_KV[:,0].max():.4f}], kvbits [{pf_KV[:,1].min():.4f}, {pf_KV[:,1].max():.4f}]")

print("Loading KVDIM Pareto front...")
pf_KVDIM, archs_KVDIM = load_archive_pareto(KVDIM_STATS, 'kvdim', config, group_size)
print(f"  KVDIM PF: {len(pf_KVDIM)} points, metric [{pf_KVDIM[:,0].min():.4f}, {pf_KVDIM[:,0].max():.4f}], kvdim [{pf_KVDIM[:,1].min():.4f}, {pf_KVDIM[:,1].max():.4f}]")

# ─── Step 2: Load AWQ CSV and extract per-method JSD ─────────────────────────
print("\nLoading 3-way AWQ CSV...")
mat_3way = load_awq_csv(AWQ_3WAY)
print(f"  Shape: {mat_3way.shape}")

N = mat_3way.shape[1]
print(f"  N_samples = {N}")

# CSV row indices from get_net_info key order:
# 0: wbits, 1: kvbits, 2: kbits, 3: vbits, 4: kvdim, 5: kdim, 6: vdim,
# 7: eff_kvbits, 8: eff_kbits, 9: eff_vbits, 10: memory, 11: n_token, 12: JSD_AWQ
CSV_WBITS  = 0
CSV_KVBITS = 1
CSV_KVDIM  = 4
CSV_JSD    = 12

wbits_csv  = mat_3way[CSV_WBITS,  :N]
kvbits_csv = mat_3way[CSV_KVBITS, :N]
kvdim_csv  = mat_3way[CSV_KVDIM,  :N]
jsd_awq    = mat_3way[CSV_JSD,    :N]

print(f"  wbits range:  [{wbits_csv.min():.4f}, {wbits_csv.max():.4f}]")
print(f"  kvbits range: [{kvbits_csv.min():.4f}, {kvbits_csv.max():.4f}]")
print(f"  kvdim range:  [{kvdim_csv.min():.4f}, {kvdim_csv.max():.4f}]")
print(f"  JSD_AWQ range:[{jsd_awq.min():.4f}, {jsd_awq.max():.4f}]")

# ─── Step 3: Match CSV comp values to Pareto front metrics ───────────────────
print("\nMatching CSV samples to Pareto fronts...")

jsd_W_hqq    = np.full(N, np.nan)
jsd_KV_hqq   = np.full(N, np.nan)
jsd_KVDIM_hqq = np.full(N, np.nan)
match_errors = {'W': [], 'KV': [], 'KVDIM': []}

for i in range(N):
    m_w, e_w = match_comp_to_metric(wbits_csv[i], pf_W)
    m_kv, e_kv = match_comp_to_metric(kvbits_csv[i], pf_KV)
    m_kvd, e_kvd = match_comp_to_metric(kvdim_csv[i], pf_KVDIM)
    if m_w is not None: jsd_W_hqq[i] = m_w
    if m_kv is not None: jsd_KV_hqq[i] = m_kv
    if m_kvd is not None: jsd_KVDIM_hqq[i] = m_kvd
    match_errors['W'].append(e_w)
    match_errors['KV'].append(e_kv)
    match_errors['KVDIM'].append(e_kvd)

for key in ['W', 'KV', 'KVDIM']:
    errs = np.array(match_errors[key])
    n_matched = np.sum(errs < 1e-6)
    print(f"  {key}: {n_matched}/{N} exact matches, max_err={errs.max():.6f}, mean_err={errs.mean():.6f}")

# Use tolerance 0.01 (floating point rounding in get_net_info)
tol = 0.01
valid_mask = (
    (np.array(match_errors['W']) < tol) &
    (np.array(match_errors['KV']) < tol) &
    (np.array(match_errors['KVDIM']) < tol) &
    (~np.isnan(jsd_awq))
)
print(f"\n  Valid samples (all 3 matched): {valid_mask.sum()}/{N}")

# ─── If matching fails, try with larger tolerance and nearest-neighbor ────────
if valid_mask.sum() < 50:
    print("  Low match count — using nearest-neighbor matching (no tolerance threshold)")
    for i in range(N):
        # W: nearest by wbits
        idx_w = np.argmin(np.abs(pf_W[:, 1] - wbits_csv[i]))
        jsd_W_hqq[i] = pf_W[idx_w, 0]
        # KV: nearest by kvbits
        idx_kv = np.argmin(np.abs(pf_KV[:, 1] - kvbits_csv[i]))
        jsd_KV_hqq[i] = pf_KV[idx_kv, 0]
        # KVDIM: nearest by kvdim
        idx_kvd = np.argmin(np.abs(pf_KVDIM[:, 1] - kvdim_csv[i]))
        jsd_KVDIM_hqq[i] = pf_KVDIM[idx_kvd, 0]
    valid_mask = ~np.isnan(jsd_awq)
    print(f"  After NN matching: {valid_mask.sum()} valid samples")

# ─── Extract valid dataset ────────────────────────────────────────────────────
X_W   = jsd_W_hqq[valid_mask]
X_KV  = jsd_KV_hqq[valid_mask]
X_KVD = jsd_KVDIM_hqq[valid_mask]
y     = jsd_awq[valid_mask]
N_valid = len(y)
print(f"\n--- Dataset summary (N={N_valid}) ---")
print(f"JSD_W_hqq:    [{X_W.min():.4f}, {X_W.max():.4f}], mean={X_W.mean():.4f}")
print(f"JSD_KV_hqq:   [{X_KV.min():.4f}, {X_KV.max():.4f}], mean={X_KV.mean():.4f}")
print(f"JSD_KVDIM_hqq:[{X_KVD.min():.4f}, {X_KVD.max():.4f}], mean={X_KVD.mean():.4f}")
print(f"JSD_AWQ:       [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")

# Correlation matrix
R = np.corrcoef(np.stack([X_W, X_KV, X_KVD, y]))
print(f"\nCorrelation with JSD_AWQ: W={R[0,3]:.4f}, KV={R[1,3]:.4f}, KVD={R[2,3]:.4f}")

# ─── Helper: build polynomial features ────────────────────────────────────────
def build_features(X_W, X_KV, X_KVD, mode='linear'):
    """Build feature matrix. mode: linear, quadratic, full_quad"""
    n = len(X_W)
    ones = np.ones(n)
    if mode == 'linear':  # [1, w, kv, kvd]
        return np.column_stack([ones, X_W, X_KV, X_KVD])
    elif mode == 'quadratic':  # [1, w, kv, kvd, w², kv², kvd²]
        return np.column_stack([ones, X_W, X_KV, X_KVD, X_W**2, X_KV**2, X_KVD**2])
    elif mode == 'full_quad':  # [1, w, kv, kvd, w², kv², kvd², w*kv, w*kvd, kv*kvd]
        return np.column_stack([ones, X_W, X_KV, X_KVD, X_W**2, X_KV**2, X_KVD**2,
                                X_W*X_KV, X_W*X_KVD, X_KV*X_KVD])

def fit_model(Phi, y):
    """OLS. Returns (coef, y_pred, R2, RMSE)."""
    coef, res, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
    y_pred = Phi @ coef
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / len(y))
    return coef, y_pred, r2, rmse

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

models = {}
for mode in ['linear', 'quadratic', 'full_quad']:
    Phi = build_features(X_W, X_KV, X_KVD, mode)
    coef, y_pred, r2, rmse = fit_model(Phi, y)
    n_params = Phi.shape[1]
    # AIC: n*ln(RSS/n) + 2k
    rss = np.sum((y - y_pred)**2)
    aic = N_valid * np.log(rss / N_valid) + 2 * n_params
    models[mode] = {'coef': coef, 'y_pred': y_pred, 'r2': r2, 'rmse': rmse, 'aic': aic, 'n_params': n_params}
    print(f"\n{mode.upper()} (k={n_params}): R²={r2:.6f}, RMSE={rmse:.6f}, AIC={aic:.2f}")
    if mode == 'linear':
        print(f"  y = {coef[0]:.4f} + {coef[1]:.4f}·JSD_W + {coef[2]:.4f}·JSD_KV + {coef[3]:.4f}·JSD_KVD")
    elif mode == 'full_quad':
        labels = ['1', 'W', 'KV', 'KVD', 'W²', 'KV²', 'KVD²', 'W·KV', 'W·KVD', 'KV·KVD']
        terms = [f"{c:+.4f}·{l}" for c, l in zip(coef, labels)]
        print(f"  y = {' '.join(terms)}")

# Best model by AIC
best_mode = min(models, key=lambda m: models[m]['aic'])
print(f"\nBest model by AIC: {best_mode}")

# ─── ANOVA / Hoeffding variance decomposition ─────────────────────────────────
print("\n" + "="*60)
print("VARIANCE DECOMPOSITION (Hoeffding ANOVA)")
print("="*60)

# Use full_quad model: decompose variance by term group
fq = models['full_quad']
coef = fq['coef']
# Compute contribution of each term to variance of y_pred
# Feature groups: main effects (W,KV,KVD), quadratic (W²,KV²,KVD²), interaction (W*KV,W*KVD,KV*KVD)
Phi_fq = build_features(X_W, X_KV, X_KVD, 'full_quad')
term_groups = {
    'W_main': [1], 'KV_main': [2], 'KVD_main': [3],
    'W_quad': [4], 'KV_quad': [5], 'KVD_quad': [6],
    'W_KV_int': [7], 'W_KVD_int': [8], 'KV_KVD_int': [9]
}
var_y = np.var(fq['y_pred'])
print(f"Var(ŷ) = {var_y:.6f}")
for gname, idxs in term_groups.items():
    contribution = np.var(np.sum([coef[i] * Phi_fq[:, i] for i in idxs], axis=0))
    print(f"  {gname:15s}: Var = {contribution:.6f} ({100*contribution/var_y:.2f}%)")

# Method-level variance
for mname, idxs in [('W_total', [1,4]), ('KV_total', [2,5]), ('KVD_total', [3,6]),
                    ('interactions', [7,8,9]), ('intercept', [0])]:
    contribution = np.var(np.sum([coef[i] * Phi_fq[:, i] for i in idxs], axis=0))
    print(f"  {mname:15s}: Var = {contribution:.6f} ({100*contribution/var_y:.2f}%)")

# ─── 27-grid analysis ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("27-GRID CALIBRATION ANALYSIS")
print("="*60)

# Compute [0.1, 0.5, 0.9] quantiles for each method's JSD
q10_W, q50_W, q90_W    = np.quantile(X_W, [0.1, 0.5, 0.9])
q10_KV, q50_KV, q90_KV = np.quantile(X_KV, [0.1, 0.5, 0.9])
q10_KVD, q50_KVD, q90_KVD = np.quantile(X_KVD, [0.1, 0.5, 0.9])

print(f"W quantiles  [0.1,0.5,0.9]: [{q10_W:.4f}, {q50_W:.4f}, {q90_W:.4f}]")
print(f"KV quantiles [0.1,0.5,0.9]: [{q10_KV:.4f}, {q50_KV:.4f}, {q90_KV:.4f}]")
print(f"KVD quantiles[0.1,0.5,0.9]: [{q10_KVD:.4f}, {q50_KVD:.4f}, {q90_KVD:.4f}]")

# Build 27-grid points
grid_pts = np.array([[w, kv, kvd]
                     for w in [q10_W, q50_W, q90_W]
                     for kv in [q10_KV, q50_KV, q90_KV]
                     for kvd in [q10_KVD, q50_KVD, q90_KVD]])

# Find AWQ values for 27-grid points: nearest neighbor in dataset
def find_nearest(pt, X_all, y_all, n_neighbors=3):
    diffs = np.sqrt(((X_all - pt)**2).sum(axis=1))
    idx = np.argsort(diffs)[:n_neighbors]
    return y_all[idx[0]], diffs[idx[0]]

X_all = np.column_stack([X_W, X_KV, X_KVD])
y_grid = np.array([find_nearest(pt, X_all, y)[0] for pt in grid_pts])

print(f"\n27-grid JSD values: [{y_grid.min():.4f}, {y_grid.max():.4f}], mean={y_grid.mean():.4f}")

# Fit full_quad model on 27-grid points
grid_W, grid_KV, grid_KVD = grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2]
Phi_27 = build_features(grid_W, grid_KV, grid_KVD, 'full_quad')
coef_27, y_pred_27, r2_27, rmse_27 = fit_model(Phi_27, y_grid)
print(f"\n27-grid full_quad model: R²={r2_27:.6f}, RMSE={rmse_27:.6f}")

# Predict on full dataset
Phi_full = build_features(X_W, X_KV, X_KVD, 'full_quad')
y_pred_grid_on_full = Phi_full @ coef_27
r2_grid_full = 1 - np.sum((y - y_pred_grid_on_full)**2) / np.sum((y - y.mean())**2)
rmse_grid_full = np.sqrt(np.mean((y - y_pred_grid_on_full)**2))
print(f"27-grid model on full data: R²={r2_grid_full:.6f}, RMSE={rmse_grid_full:.6f}")
print(f"Full-data model on full data: R²={models['full_quad']['r2']:.6f}, RMSE={models['full_quad']['rmse']:.6f}")

# Compare: 27-grid vs random 27 (bootstrap)
n_boot = 200
r2_rand_list, rmse_rand_list = [], []
for _ in range(n_boot):
    idx_rand = np.random.choice(N_valid, 27, replace=False)
    Phi_rand = build_features(X_W[idx_rand], X_KV[idx_rand], X_KVD[idx_rand], 'full_quad')
    if np.linalg.matrix_rank(Phi_rand) < 10:
        continue
    coef_rand, _, _, _ = fit_model(Phi_rand, y[idx_rand])
    y_pred_rand = Phi_full @ coef_rand
    r2_r = 1 - np.sum((y - y_pred_rand)**2) / np.sum((y - y.mean())**2)
    rmse_r = np.sqrt(np.mean((y - y_pred_rand)**2))
    r2_rand_list.append(r2_r)
    rmse_rand_list.append(rmse_r)

r2_rand = np.array(r2_rand_list)
print(f"\n27-grid vs Random-27 (bootstrap n={n_boot}):")
print(f"  27-grid R²     = {r2_grid_full:.6f}")
print(f"  Random-27 R²   = {np.median(r2_rand):.6f} (median), [{np.percentile(r2_rand,10):.4f}, {np.percentile(r2_rand,90):.4f}]")
print(f"  27-grid beats Random-27 {np.mean(r2_grid_full > r2_rand)*100:.1f}% of the time")

# ─── Monotonicity analysis ────────────────────────────────────────────────────
print("\n" + "="*60)
print("MONOTONICITY ANALYSIS")
print("="*60)
# Condition: ∂F/∂JSD_W ≥ 0, ∂F/∂JSD_KV ≥ 0, ∂F/∂JSD_KVD ≥ 0
# Using full_quad model: coef = [β0, βW, βKV, βKVD, βWW, βKVKV, βKVDKVD, βWKV, βWKVD, βKVKVD]
c = models['full_quad']['coef']
# ∂F/∂W   = βW + 2βWW·W + βWKV·KV + βWKVD·KVD
# ∂F/∂KV  = βKV + 2βKVKV·KV + βWKV·W + βKVKVD·KVD
# ∂F/∂KVD = βKVD + 2βKVDKVD·KVD + βWKVD·W + βKVKVD·KV

dF_dW   = c[1] + 2*c[4]*X_W   + c[7]*X_KV   + c[8]*X_KVD
dF_dKV  = c[2] + 2*c[5]*X_KV  + c[7]*X_W    + c[9]*X_KVD
dF_dKVD = c[3] + 2*c[6]*X_KVD + c[8]*X_W    + c[9]*X_KV

print(f"\n∂F/∂JSD_W:   min={dF_dW.min():.4f}, max={dF_dW.max():.4f}, mean={dF_dW.mean():.4f}")
print(f"  Fraction ≥ 0: {(dF_dW >= 0).mean()*100:.1f}%  (should be ~100%)")
print(f"  Fraction < 0 (violations): {(dF_dW < 0).mean()*100:.1f}%")

print(f"\n∂F/∂JSD_KV:  min={dF_dKV.min():.4f}, max={dF_dKV.max():.4f}, mean={dF_dKV.mean():.4f}")
print(f"  Fraction ≥ 0: {(dF_dKV >= 0).mean()*100:.1f}%")
print(f"  Fraction < 0 (violations): {(dF_dKV < 0).mean()*100:.1f}%")

print(f"\n∂F/∂JSD_KVD: min={dF_dKVD.min():.4f}, max={dF_dKVD.max():.4f}, mean={dF_dKVD.mean():.4f}")
print(f"  Fraction ≥ 0: {(dF_dKVD >= 0).mean()*100:.1f}%")
print(f"  Fraction < 0 (violations): {(dF_dKVD < 0).mean()*100:.1f}%")

# Where do violations occur?
viol_mask = (dF_dW < 0) | (dF_dKV < 0) | (dF_dKVD < 0)
print(f"\nAny violation: {viol_mask.sum()}/{N_valid} points ({viol_mask.mean()*100:.1f}%)")
if viol_mask.sum() > 0:
    print(f"  At violations: JSD_W=[{X_W[viol_mask].min():.4f},{X_W[viol_mask].max():.4f}], "
          f"JSD_KV=[{X_KV[viol_mask].min():.4f},{X_KV[viol_mask].max():.4f}], "
          f"JSD_KVD=[{X_KVD[viol_mask].min():.4f},{X_KVD[viol_mask].max():.4f}]")

# ─── PF Recovery analysis ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("PARETO FRONT RECOVERY ANALYSIS")
print("="*60)

def pareto_front_2d_from_arrays(metrics, comps):
    """Given (metrics, comps), return Pareto front indices (minimize both)."""
    F = np.column_stack([metrics, comps])
    order = np.argsort(F[:, 1])  # sort by comp
    F_sorted = F[order]
    min_metric = np.inf
    nd = []
    for i in range(len(F_sorted)):
        if F_sorted[i, 0] < min_metric:
            nd.append(i)
            min_metric = F_sorted[i, 0]
    return order[nd]

# Complexity: use wbits_csv + kvbits_csv + kvdim_csv (range-normalized)
def range_norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-10)

C_joint = (range_norm(wbits_csv[valid_mask]) +
           range_norm(kvbits_csv[valid_mask]) +
           range_norm(kvdim_csv[valid_mask]))

# True PF from actual AWQ measurements
true_pf_idx = pareto_front_2d_from_arrays(y, C_joint)
n_true_pf = len(true_pf_idx)

# Surrogate PF from full_quad model predictions (trained on full data)
y_pred_full = models['full_quad']['y_pred']
surr_pf_idx = pareto_front_2d_from_arrays(y_pred_full, C_joint)
n_surr_pf = len(surr_pf_idx)

# Recovery: how many true PF points are in surrogate PF?
true_pf_set = set(true_pf_idx)
surr_pf_set = set(surr_pf_idx)
n_recovered = len(true_pf_set & surr_pf_set)
print(f"True PF size: {n_true_pf}, Surrogate PF size: {n_surr_pf}")
print(f"Exact recovery: {n_recovered} points ({n_recovered/n_true_pf*100:.1f}% of true PF)")

# Hypervolume comparison
def hypervolume_2d(pf_metrics, pf_comps, ref_metric=None, ref_comp=None):
    """2D hypervolume with reference point."""
    if ref_metric is None: ref_metric = max(pf_metrics) * 1.1
    if ref_comp is None: ref_comp = max(pf_comps) * 1.1
    # Sort by comp
    order = np.argsort(pf_comps)
    m = pf_metrics[order]
    c = pf_comps[order]
    hv = 0
    for i in range(len(m)):
        w = (ref_comp if i == len(m)-1 else c[i+1]) - c[i]
        if w < 0: w = 0
        hv += w * (ref_metric - m[i])
    return hv

ref_m = y.max() * 1.1
ref_c = C_joint.max() * 1.1

hv_true = hypervolume_2d(y[true_pf_idx], C_joint[true_pf_idx], ref_m, ref_c)
hv_surr = hypervolume_2d(y[surr_pf_idx], C_joint[surr_pf_idx], ref_m, ref_c)
# For surrogate PF, use actual y values to compute "realized" HV
hv_surr_realized = hypervolume_2d(y[surr_pf_idx], C_joint[surr_pf_idx], ref_m, ref_c)

print(f"\nHypervolume (2D, range-normalized comp):")
print(f"  True PF HV:      {hv_true:.6f}")
print(f"  Surrogate PF HV: {hv_surr:.6f}")
print(f"  HV ratio (surr/true): {hv_surr/hv_true:.4f}")

# ─── Direct vs Hierarchical calibration comparison ────────────────────────────
print("\n" + "="*60)
print("DIRECT vs HIERARCHICAL CALIBRATION")
print("="*60)

# Load 2-way CSVs for hierarchical analysis
def load_2way(csv_path, comp_row1, comp_row2, pf1, pf2, method1, method2):
    """Load 2-way AWQ CSV and extract per-method JSD + actual JSD."""
    mat = load_awq_csv(csv_path)
    n = mat.shape[1]
    comp1 = mat[comp_row1, :n]
    comp2 = mat[comp_row2, :n]
    jsd_actual = mat[CSV_JSD, :n]

    jsd1 = np.array([match_comp_to_metric(c, pf1)[0] or pf1[np.argmin(np.abs(pf1[:,1]-c)), 0]
                     for c in comp1])
    jsd2 = np.array([match_comp_to_metric(c, pf2)[0] or pf2[np.argmin(np.abs(pf2[:,1]-c)), 0]
                     for c in comp2])
    valid = ~np.isnan(jsd_actual)
    return jsd1[valid], jsd2[valid], jsd_actual[valid]

print("\nLoading 2-way CSV data...")
jsd_W2, jsd_KV2, jsd_WKV = load_2way(AWQ_W_KV,   CSV_WBITS, CSV_KVBITS, pf_W, pf_KV,    'W',  'KV')
jsd_W3, jsd_KVD2, jsd_WKVD = load_2way(AWQ_W_KVD, CSV_WBITS, CSV_KVDIM,  pf_W, pf_KVDIM, 'W',  'KVD')
jsd_KV3, jsd_KVD3, jsd_KVKVD = load_2way(AWQ_KV_KVD, CSV_KVBITS, CSV_KVDIM, pf_KV, pf_KVDIM, 'KV', 'KVD')

print(f"W+KV 2-way: N={len(jsd_WKV)}, JSD range [{jsd_WKV.min():.4f},{jsd_WKV.max():.4f}]")
print(f"W+KVD 2-way: N={len(jsd_WKVD)}, JSD range [{jsd_WKVD.min():.4f},{jsd_WKVD.max():.4f}]")
print(f"KV+KVD 2-way: N={len(jsd_KVKVD)}, JSD range [{jsd_KVKVD.min():.4f},{jsd_KVKVD.max():.4f}]")

# 3-way Direct: fit full_quad on all 3 methods with N=27 budget
# Split: use 27 points for fitting (grid), evaluate on remainder
n_train = min(27, N_valid)
train_idx = np.linspace(0, N_valid-1, n_train, dtype=int)
test_idx  = np.setdiff1d(np.arange(N_valid), train_idx)

Phi_train = build_features(X_W[train_idx], X_KV[train_idx], X_KVD[train_idx], 'full_quad')
coef_direct, _, r2_train, _ = fit_model(Phi_train, y[train_idx])
Phi_test = build_features(X_W[test_idx], X_KV[test_idx], X_KVD[test_idx], 'full_quad')
y_pred_direct = Phi_test @ coef_direct
r2_direct = 1 - np.sum((y[test_idx]-y_pred_direct)**2) / np.sum((y[test_idx]-y[test_idx].mean())**2)
rmse_direct = np.sqrt(np.mean((y[test_idx]-y_pred_direct)**2))
print(f"\nDirect 3-way (n_train={n_train}): R²={r2_direct:.4f}, RMSE={rmse_direct:.6f}")

# 2-way model on W+KV, then add KVD
n2 = len(jsd_W2)
n2_train = min(27, n2)
train2_idx = np.linspace(0, n2-1, n2_train, dtype=int)
test2_idx  = np.setdiff1d(np.arange(n2), train2_idx)
Phi2_train = np.column_stack([np.ones(n2_train), jsd_W2[train2_idx], jsd_KV2[train2_idx],
                               jsd_W2[train2_idx]**2, jsd_KV2[train2_idx]**2,
                               jsd_W2[train2_idx]*jsd_KV2[train2_idx]])
coef2, _, r2_2way, _ = fit_model(Phi2_train, jsd_WKV[train2_idx])
print(f"2-way W+KV model: R²_train={r2_2way:.4f}")

print("\n[Note: Full hierarchical analysis requires matched 2-way+3-way samples.]")
print("[Using direct comparison between n=27 calibration budgets.]")

# ─── Save results for figure generation ─────────────────────────────────────
results = {
    'X_W': X_W, 'X_KV': X_KV, 'X_KVD': X_KVD, 'y': y, 'N_valid': N_valid,
    'models': {k: {'coef': v['coef'], 'y_pred': v['y_pred'], 'r2': v['r2'], 'rmse': v['rmse']}
               for k, v in models.items()},
    'pf_W': pf_W, 'pf_KV': pf_KV, 'pf_KVDIM': pf_KVDIM,
    'C_joint': C_joint,
    'true_pf_idx': true_pf_idx, 'surr_pf_idx': surr_pf_idx,
    'dF_dW': dF_dW, 'dF_dKV': dF_dKV, 'dF_dKVD': dF_dKVD,
    'grid_pts': grid_pts, 'y_grid': y_grid, 'coef_27': coef_27,
    'r2_grid_full': r2_grid_full, 'r2_rand': r2_rand,
}
np.save(f'{BASE}/save/result/figures/analysis_results.npy', results, allow_pickle=True)

# ─── Generate Figures ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60)
os.makedirs(f'{BASE}/save/result/figures', exist_ok=True)

COLORS = {'W': '#E64B35', 'KV': '#4DBBD5', 'KVD': '#00A087', 'AWQ': '#3C5488', 'highlight': '#F39B7F'}
FONT = {'family': 'sans-serif', 'size': 11}
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
                     'legend.fontsize': 10, 'figure.dpi': 150})

# ── Fig 1: Pareto Frontiers for each method ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (pf, label, color, xlabel) in zip(axes, [
    (pf_W,    'W Quant (HQQ)',    COLORS['W'],   'Avg Weight Bits'),
    (pf_KV,   'KV Cache Quant',   COLORS['KV'],  'Avg KV Bits'),
    (pf_KVDIM,'KV Cache Pruning', COLORS['KVD'], 'Avg KV Dim'),
]):
    order = np.argsort(pf[:, 1])
    ax.plot(pf[order, 1], pf[order, 0], 'o-', color=color, lw=2, ms=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('JSD (HQQ)')
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
plt.suptitle('Per-Method Pareto Frontiers (HQQ Search Archives)', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/save/result/figures/fig1_pareto_frontiers.png', bbox_inches='tight')
plt.close()
print("  Saved fig1_pareto_frontiers.png")

# ── Fig 2: Correlation scatter plots ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (Xi, label, color) in zip(axes, [
    (X_W,  'JSD_W (HQQ)', COLORS['W']),
    (X_KV, 'JSD_KV (HQQ)', COLORS['KV']),
    (X_KVD,'JSD_KVD (HQQ)', COLORS['KVD']),
]):
    ax.scatter(Xi, y, c=color, alpha=0.5, s=15, edgecolors='none')
    # Regression line
    m, b, r, p, _ = stats.linregress(Xi, y)
    x_line = np.linspace(Xi.min(), Xi.max(), 100)
    ax.plot(x_line, m*x_line + b, 'k-', lw=2)
    ax.set_xlabel(label)
    ax.set_ylabel('JSD_AWQ')
    ax.set_title(f'r = {r:.3f}')
    ax.grid(True, alpha=0.3)
plt.suptitle('Per-Method JSD vs Combined AWQ JSD', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/save/result/figures/fig2_correlation.png', bbox_inches='tight')
plt.close()
print("  Saved fig2_correlation.png")

# ── Fig 3: Model comparison (R², RMSE, AIC) ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
model_names = ['linear\n(k=4)', 'quadratic\n(k=7)', 'full_quad\n(k=10)']
r2_vals  = [models[m]['r2']   for m in ['linear','quadratic','full_quad']]
rmse_vals= [models[m]['rmse'] for m in ['linear','quadratic','full_quad']]
aic_vals = [models[m]['aic']  for m in ['linear','quadratic','full_quad']]

bar_colors = [COLORS['W'], COLORS['KV'], COLORS['KVD']]
for ax, vals, ylabel, title in zip(axes,
    [r2_vals, rmse_vals, aic_vals],
    ['R²', 'RMSE', 'AIC'],
    ['R² (higher=better)', 'RMSE (lower=better)', 'AIC (lower=better)']):
    bars = ax.bar(model_names, vals, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{v:.4f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
plt.suptitle('Model Comparison: Additive Approximation Quality', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/save/result/figures/fig3_model_comparison.png', bbox_inches='tight')
plt.close()
print("  Saved fig3_model_comparison.png")

# ── Fig 4: Variance decomposition bar chart ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
c = models['full_quad']['coef']
Phi_fq = build_features(X_W, X_KV, X_KVD, 'full_quad')
groups = {
    'W_main':   ([1], COLORS['W']),
    'W_quad':   ([4], COLORS['W']),
    'KV_main':  ([2], COLORS['KV']),
    'KV_quad':  ([5], COLORS['KV']),
    'KVD_main': ([3], COLORS['KVD']),
    'KVD_quad': ([6], COLORS['KVD']),
    'W·KV':     ([7], '#9B59B6'),
    'W·KVD':    ([8], '#E67E22'),
    'KV·KVD':   ([9], '#1ABC9C'),
}
var_y_fq = np.var(models['full_quad']['y_pred'])
labels, heights, colors_bar = [], [], []
for gname, (idxs, col) in groups.items():
    v = np.var(np.sum([c[i] * Phi_fq[:, i] for i in idxs], axis=0)) / var_y_fq * 100
    labels.append(gname)
    heights.append(v)
    colors_bar.append(col)

bars = ax.bar(labels, heights, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
for bar, h in zip(bars, heights):
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f}%',
            ha='center', va='bottom', fontsize=9, rotation=45)
ax.set_ylabel('% of Var(ŷ)')
ax.set_title('Hoeffding ANOVA Variance Decomposition\n(Full-Quadratic Surrogate)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{BASE}/save/result/figures/fig4_variance_decomposition.png', bbox_inches='tight')
plt.close()
print("  Saved fig4_variance_decomposition.png")

# ── Fig 5: Fit quality scatter: predicted vs actual ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, mode, label in zip(axes, ['linear', 'full_quad'], ['Linear (Additive)', 'Full Quadratic']):
    yp = models[mode]['y_pred']
    ax.scatter(y, yp, c=COLORS['AWQ'], alpha=0.4, s=15, edgecolors='none')
    lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='y=x')
    ax.set_xlabel('JSD_AWQ (actual)')
    ax.set_ylabel('JSD (predicted)')
    ax.set_title(f'{label}\nR²={models[mode]["r2"]:.4f}, RMSE={models[mode]["rmse"]:.6f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle('Additive Approximation Fit Quality', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/save/result/figures/fig5_fit_quality.png', bbox_inches='tight')
plt.close()
print("  Saved fig5_fit_quality.png")

# ── Fig 6: 27-grid vs Random-27 comparison ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(r2_rand, bins=30, color=COLORS['KV'], alpha=0.7, edgecolor='black', linewidth=0.3)
ax.axvline(r2_grid_full, color=COLORS['W'], lw=2.5, label=f'27-grid R²={r2_grid_full:.4f}')
ax.axvline(np.median(r2_rand), color='black', lw=1.5, ls='--', label=f'Median Random-27={np.median(r2_rand):.4f}')
ax.set_xlabel('R² on full dataset')
ax.set_ylabel('Count')
ax.set_title('27-Grid vs Random-27 Calibration\n(Bootstrap n=200)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
# 3D structure of 27-grid
from itertools import product
grid_W_pts = np.array([q10_W, q50_W, q90_W])
grid_KV_pts = np.array([q10_KV, q50_KV, q90_KV])
grid_KVD_pts = np.array([q10_KVD, q50_KVD, q90_KVD])
for i, w in enumerate(grid_W_pts):
    for j, kv in enumerate(grid_KV_pts):
        sc = ax.scatter([w]*3, grid_KVD_pts, s=30+20*j,
                       c=[COLORS['W'], COLORS['KV'], COLORS['KVD']][i],
                       alpha=0.7, zorder=5)
ax.set_xlabel('JSD_W [0.1,0.5,0.9]')
ax.set_ylabel('JSD_KVD [0.1,0.5,0.9]')
ax.set_title('27-Grid Structure\n([0.1,0.5,0.9]³ quantile design)')
ax.grid(True, alpha=0.3)

plt.suptitle('Calibration Efficiency: 27-Grid vs Random Sampling', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/save/result/figures/fig6_27grid_analysis.png', bbox_inches='tight')
plt.close()
print("  Saved fig6_27grid_analysis.png")

# ── Fig 7: Pareto Front Recovery ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(C_joint, y, c='lightgray', s=10, alpha=0.4, label='All samples', zorder=1)
ax.scatter(C_joint[true_pf_idx], y[true_pf_idx], c=COLORS['AWQ'], s=40,
           marker='D', label=f'True PF (n={n_true_pf})', zorder=3)
ax.scatter(C_joint[surr_pf_idx], y[surr_pf_idx], c=COLORS['W'], s=20,
           marker='^', label=f'Surrogate PF (n={n_surr_pf})', zorder=2, alpha=0.7)
recovered = list(true_pf_set & surr_pf_set)
if recovered:
    ax.scatter(C_joint[recovered], y[recovered], c='green', s=60,
               marker='*', label=f'Recovered (n={n_recovered})', zorder=4)
ax.set_xlabel('C_joint (range-normalized sum)')
ax.set_ylabel('JSD_AWQ')
ax.set_title('Pareto Front Recovery')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
# HV comparison
categories = ['True PF', 'Surrogate PF']
hv_vals = [hv_true, hv_surr]
bars = ax.bar(categories, hv_vals, color=[COLORS['AWQ'], COLORS['W']], alpha=0.8,
              edgecolor='black', linewidth=0.5)
for bar, v in zip(bars, hv_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.4f}',
            ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Hypervolume (2D)')
ax.set_title(f'HV Comparison\n(Ratio: {hv_surr/hv_true:.4f})')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Pareto Front Recovery from Surrogate Model', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/save/result/figures/fig7_pf_recovery.png', bbox_inches='tight')
plt.close()
print("  Saved fig7_pf_recovery.png")

# ── Fig 8: Monotonicity ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (grad, Xi, xlabel, color) in zip(axes, [
    (dF_dW,   X_W,  'JSD_W (HQQ)',   COLORS['W']),
    (dF_dKV,  X_KV, 'JSD_KV (HQQ)',  COLORS['KV']),
    (dF_dKVD, X_KVD,'JSD_KVD (HQQ)', COLORS['KVD']),
]):
    c_arr = np.where(grad >= 0, color, 'red')
    ax.scatter(Xi, grad, c=c_arr, alpha=0.5, s=15, edgecolors='none')
    ax.axhline(0, color='black', lw=1.5, ls='--')
    frac = (grad < 0).mean() * 100
    ax.set_xlabel(xlabel)
    ax.set_ylabel('∂F/∂(input)')
    ax.set_title(f'Violations: {frac:.1f}%')
    ax.grid(True, alpha=0.3)
plt.suptitle('Monotonicity Check: ∂F/∂JSD ≥ 0 (red = violation)', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/save/result/figures/fig8_monotonicity.png', bbox_inches='tight')
plt.close()
print("  Saved fig8_monotonicity.png")

# ── Fig 9: Combined summary figure ───────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.4)

# Top-left: scatter plot (linear model)
ax_scatter = fig.add_subplot(gs[0, :2])
yp_lin = models['linear']['y_pred']
ax_scatter.scatter(y, yp_lin, c=COLORS['AWQ'], alpha=0.5, s=15)
lo, hi = min(y.min(), yp_lin.min()), max(y.max(), yp_lin.max())
ax_scatter.plot([lo, hi], [lo, hi], 'k--', lw=1.5)
ax_scatter.set_xlabel('JSD_AWQ (actual)')
ax_scatter.set_ylabel('Predicted (additive)')
ax_scatter.set_title(f'Additive Model: R²={models["linear"]["r2"]:.4f}')
ax_scatter.grid(True, alpha=0.3)

# Top-right: model comparison R²
ax_bar = fig.add_subplot(gs[0, 2:])
ax_bar.bar(['Linear\nAdditive', 'Quadratic', 'Full Quad\n+Inter'],
           [models['linear']['r2'], models['quadratic']['r2'], models['full_quad']['r2']],
           color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax_bar.set_ylabel('R²')
ax_bar.set_title('Model Comparison')
ax_bar.set_ylim([0, 1.05])
ax_bar.grid(True, alpha=0.3, axis='y')
for i, (n, r) in enumerate(zip(['linear','quadratic','full_quad'], r2_vals)):
    ax_bar.text(i, models[n]['r2'] + 0.01, f'{models[n]["r2"]:.4f}', ha='center', fontsize=9)

# Bottom-left: 27-grid histogram
ax_hist = fig.add_subplot(gs[1, :2])
ax_hist.hist(r2_rand, bins=20, color=COLORS['KV'], alpha=0.7, edgecolor='black', linewidth=0.3)
ax_hist.axvline(r2_grid_full, color=COLORS['W'], lw=2.5, label=f'27-grid={r2_grid_full:.4f}')
ax_hist.set_xlabel('R²')
ax_hist.set_ylabel('Count')
ax_hist.set_title('27-Grid vs Random Sampling')
ax_hist.legend()
ax_hist.grid(True, alpha=0.3)

# Bottom-right: PF recovery
ax_pf = fig.add_subplot(gs[1, 2:])
ax_pf.scatter(C_joint, y, c='lightgray', s=8, alpha=0.3)
ax_pf.scatter(C_joint[true_pf_idx], y[true_pf_idx], c=COLORS['AWQ'], s=30, marker='D', label='True PF')
ax_pf.scatter(C_joint[surr_pf_idx], y[surr_pf_idx], c=COLORS['W'], s=20, marker='^', label='Surrogate PF', alpha=0.7)
if recovered:
    ax_pf.scatter(C_joint[recovered], y[recovered], c='green', s=50, marker='*', label='Recovered')
ax_pf.set_xlabel('Complexity')
ax_pf.set_ylabel('JSD_AWQ')
ax_pf.set_title(f'PF Recovery: {n_recovered}/{n_true_pf} ({n_recovered/n_true_pf*100:.0f}%)')
ax_pf.legend(fontsize=9)
ax_pf.grid(True, alpha=0.3)

plt.suptitle('Additive Pareto Combination: Summary Analysis\n(Input: per-method HQQ JSD)',
             fontweight='bold', fontsize=14)
plt.savefig(f'{BASE}/save/result/figures/fig9_summary.png', bbox_inches='tight')
plt.close()
print("  Saved fig9_summary.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nKey results with CORRECT inputs (per-method HQQ JSD):")
print(f"  Linear (additive) model R² = {models['linear']['r2']:.4f}")
print(f"  Full-quadratic model R²    = {models['full_quad']['r2']:.4f}")
print(f"  Δ(full_quad - linear) R²   = {models['full_quad']['r2'] - models['linear']['r2']:.4f}")
print(f"  27-grid calibration R²     = {r2_grid_full:.4f}")
print(f"  PF recovery rate           = {n_recovered}/{n_true_pf} ({n_recovered/n_true_pf*100:.0f}%)")
print(f"  Monotonicity violations    = {viol_mask.sum()}/{N_valid} ({viol_mask.mean()*100:.1f}%)")
