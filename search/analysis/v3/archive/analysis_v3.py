"""analysis_v3.py — Extended additive-loss approximation analysis.

Sections (per user request):
  1. Math approximations (M0/M1/M8/M9/M10) on 3-way N=200
  2. RBF (predictor/rbf.py + pySOT) with sorted-index input
  3. Hierarchical 2-then-1 combination (math models + RBF)
  4. 27-grid + N_train sample-efficiency scan
  5. 3-tier (low/mid/high) bit-regime split

Inputs are per-method HQQ JSD values (recovered from per-method PF archives),
matched to AWQ measurement CSVs (3-way and three 2-way runs, N=200 each).
"""
import sys, os, json, csv
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.func import get_net_info
from predictor.rbf import RBF

# ─── Data loading ─────────────────────────────────────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]
    min2 = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < min2:
            nd.append(i); min2 = F_s[i, 1]
    return order[nd]

def load_archive_pareto(stats_path, comp_key, config, group_size):
    with open(stats_path) as f:
        data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs = [v[0] for v in archive]
    metrics = np.array([v[1] for v in archive])
    comps = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
    F = np.column_stack((metrics, comps))
    return F[pareto_front_2d(F)]

def load_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    max_cols = max(len(r) for r in rows)
    mat = np.full((len(rows), max_cols), np.nan)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            try: mat[i, j] = float(v)
            except: pass
    return mat

def match_metric(comp_vals, pf):
    return np.array([pf[np.argmin(np.abs(pf[:, 1] - c)), 0] for c in comp_vals])

def match_index(comp_vals, pf_metric_sorted):
    return np.array([np.argmin(np.abs(pf_metric_sorted[:, 1] - c)) for c in comp_vals])

# ─── Setup ────────────────────────────────────────────────────────────────────
BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY  = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
AWQ_W_KV  = f'{BASE}/save/result/awq/2604162010_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr/results.csv'
AWQ_W_KVD = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim/results.csv'
AWQ_KV_KVD= f'{BASE}/save/result/awq/2604162013_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_kv_expr_kv_dim/results.csv'

with open(f'{BASE}/config/llama.json') as f:
    config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

print("Loading per-method Pareto fronts...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
# PF sorted by metric ascending (best first) for sorted-index lookups
pf_W_m     = pf_W    [np.argsort(pf_W    [:, 0])]
pf_KV_m    = pf_KV   [np.argsort(pf_KV   [:, 0])]
pf_KVDIM_m = pf_KVDIM[np.argsort(pf_KVDIM[:, 0])]
print(f"  pf_W: {len(pf_W)} pts | pf_KV: {len(pf_KV)} pts | pf_KVDIM: {len(pf_KVDIM)} pts")

# Default JSD = lowest-metric end of each PF (max comp = least quantization)
JSD_W_def   = pf_W_m   [0, 0]
JSD_KV_def  = pf_KV_m  [0, 0]
JSD_KVD_def = pf_KVDIM_m[0, 0]
print(f"  Default JSD (max-comp): W={JSD_W_def:.4f}, KV={JSD_KV_def:.4f}, KVD={JSD_KVD_def:.4f}")

# 3-way data
mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y3   = mat3[12, :N0]
v3   = ~np.isnan(y3)
xW3  = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3 = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD3= match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y3   = y3[v3]
iW3   = match_index(mat3[0, :N0][v3], pf_W_m   ).astype(float)
iKV3  = match_index(mat3[1, :N0][v3], pf_KV_m  ).astype(float)
iKVD3 = match_index(mat3[4, :N0][v3], pf_KVDIM_m).astype(float)
N3 = len(y3)
print(f"\n3-way data: N={N3}")

# 2-way data
def load_2way(path, has_W, has_KV, has_KVD):
    mat = load_csv(path); N = mat.shape[1]
    y = mat[12, :N]; valid = ~np.isnan(y)
    xW   = match_metric(mat[0, :N], pf_W   ) if has_W   else np.full(N, JSD_W_def)
    xKV  = match_metric(mat[1, :N], pf_KV  ) if has_KV  else np.full(N, JSD_KV_def)
    xKVD = match_metric(mat[4, :N], pf_KVDIM) if has_KVD else np.full(N, JSD_KVD_def)
    return xW[valid], xKV[valid], xKVD[valid], y[valid]

xW_WK, xKV_WK, xKVD_WK, y_WK = load_2way(AWQ_W_KV,  True,  True,  False)
xW_WD, xKV_WD, xKVD_WD, y_WD = load_2way(AWQ_W_KVD, True,  False, True )
xW_KD, xKV_KD, xKVD_KD, y_KD = load_2way(AWQ_KV_KVD,False, True,  True )
print(f"2-way data: W+KV N={len(y_WK)}, W+KVD N={len(y_WD)}, KV+KVD N={len(y_KD)}")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def features(xw, xkv, xkvd, mode):
    n = len(xw); o = np.ones(n)
    if mode == 'M0' : return np.column_stack([xw, xkv, xkvd])
    if mode == 'M1' : return np.column_stack([o, xw, xkv, xkvd])
    if mode == 'M8' : return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2])
    if mode == 'M9' : return np.column_stack([o, xw, xkv, xkvd, xw*xkv, xw*xkvd, xkv*xkvd])
    if mode == 'M10': return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2,
                                              xw*xkv, xw*xkvd, xkv*xkvd])
    raise ValueError(mode)

def fit_ols(Phi, y):
    coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    yp = Phi @ coef
    ss_r = np.sum((y - yp)**2); ss_t = np.sum((y - y.mean())**2)
    r2 = 1 - ss_r / max(ss_t, 1e-30)
    rmse = np.sqrt(ss_r / len(y))
    aic = len(y) * np.log(max(ss_r/len(y), 1e-30)) + 2 * Phi.shape[1]
    return coef, yp, r2, rmse, aic

def loocv_rmse(Phi, y):
    n = len(y); errs = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        try:
            coef, *_ = np.linalg.lstsq(Phi[m], y[m], rcond=None)
            errs[i] = y[i] - Phi[i] @ coef
        except Exception:
            errs[i] = np.nan
    return np.sqrt(np.nanmean(errs**2))

def kfold_cv(X, y, fit_predict, k=5, seed=0):
    """Generic K-fold CV. fit_predict(X_tr, y_tr, X_te) → y_pred."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    yp = np.full_like(y, np.nan, dtype=float)
    for f in range(k):
        te = folds[f]; tr = np.setdiff1d(idx, te)
        try:
            yp[te] = fit_predict(X[tr], y[tr], X[te])
        except Exception:
            yp[te] = y[tr].mean()
    ok = ~np.isnan(yp)
    ss_r = np.sum((y[ok] - yp[ok])**2); ss_t = np.sum((y[ok] - y[ok].mean())**2)
    return 1 - ss_r/max(ss_t,1e-30), np.sqrt(np.mean((y[ok] - yp[ok])**2)), yp

def rbf_fit_predict_factory(kernel='cubic', tail='linear', lb=None, ub=None):
    def fp(X_tr, y_tr, X_te):
        m = RBF(kernel=kernel, tail=tail, lb=lb, ub=ub)
        m.fit(X_tr, y_tr)
        return m.predict(X_te).ravel()
    return fp

def ols_fit_predict_factory(mode):
    def fp(X_tr, y_tr, X_te):
        Phi_tr = features(X_tr[:,0], X_tr[:,1], X_tr[:,2], mode)
        Phi_te = features(X_te[:,0], X_te[:,1], X_te[:,2], mode)
        coef, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
        return Phi_te @ coef
    return fp

# =============================================================================
# SECTION 1: Math approximations on 3-way N=200
# =============================================================================
print("\n" + "="*70)
print("SECTION 1 — Math models (M0/M1/M8/M9/M10) on 3-way N=200")
print("="*70)
sec1 = {}
X3_jsd = np.column_stack([xW3, xKV3, xKVD3])
for m in ['M0','M1','M8','M9','M10']:
    Phi = features(xW3, xKV3, xKVD3, m)
    coef, yp, r2, rmse, aic = fit_ols(Phi, y3)
    rmse_loo = loocv_rmse(Phi, y3)
    r2_kf, rmse_kf, _ = kfold_cv(X3_jsd, y3, ols_fit_predict_factory(m), k=5, seed=0)
    sec1[m] = dict(coef=coef, yp=yp, r2=r2, rmse=rmse, aic=aic,
                   rmse_loo=rmse_loo, r2_kf=r2_kf, k=Phi.shape[1])
    print(f"  {m:3s} (k={Phi.shape[1]:2d}): R²={r2:.4f}  RMSE={rmse:.5f}  "
          f"LOOCV-RMSE={rmse_loo:.5f}  5-fold R²={r2_kf:.4f}  AIC={aic:.1f}")

print("\n  Coefficients of linear-additive models:")
c0 = sec1['M0']['coef']
c1 = sec1['M1']['coef']
print(f"    M0 (no-int): y = {c0[0]:.4f}·JSD_W + {c0[1]:.4f}·JSD_KV + {c0[2]:.4f}·JSD_KVD")
print(f"    M1 (w/ int): y = {c1[0]:.4f} + {c1[1]:.4f}·JSD_W + {c1[2]:.4f}·JSD_KV + {c1[3]:.4f}·JSD_KVD")

# Monotonicity (full quad)
c10 = sec1['M10']['coef']
dW   = c10[1] + 2*c10[4]*xW3   + c10[7]*xKV3   + c10[8]*xKVD3
dKV  = c10[2] + 2*c10[5]*xKV3  + c10[7]*xW3    + c10[9]*xKVD3
dKVD = c10[3] + 2*c10[6]*xKVD3 + c10[8]*xW3    + c10[9]*xKV3
print(f"  Monotonicity (M10): "
      f"W min={dW.min():+.3f}({(dW>=0).mean()*100:.0f}%)  "
      f"KV min={dKV.min():+.3f}({(dKV>=0).mean()*100:.0f}%)  "
      f"KVD min={dKVD.min():+.3f}({(dKVD>=0).mean()*100:.0f}%)")

# =============================================================================
# SECTION 2: RBF predictor with sorted-index input
# =============================================================================
print("\n" + "="*70)
print("SECTION 2 — RBF predictor (pySOT), input = [i_W, i_KV, i_KVD]")
print("="*70)
X3_idx = np.column_stack([iW3, iKV3, iKVD3])
lb_idx = np.array([0., 0., 0.])
ub_idx = np.array([len(pf_W_m)-1, len(pf_KV_m)-1, len(pf_KVDIM_m)-1], dtype=float)
print(f"  index ranges: i_W∈[0,{ub_idx[0]:.0f}]  i_KV∈[0,{ub_idx[1]:.0f}]  i_KVD∈[0,{ub_idx[2]:.0f}]")

sec2 = {}
for kernel in ['cubic', 'tps', 'linear']:
    for tail in ['linear', 'constant']:
        try:
            mfit = RBF(kernel=kernel, tail=tail, lb=lb_idx, ub=ub_idx)
            mfit.fit(X3_idx, y3)
            yp_train = mfit.predict(X3_idx).ravel()
            r2_tr = 1 - np.sum((y3-yp_train)**2)/np.sum((y3-y3.mean())**2)
            r2_cv, rmse_cv, yp_cv = kfold_cv(
                X3_idx, y3, rbf_fit_predict_factory(kernel, tail, lb_idx, ub_idx), k=5, seed=0)
            sec2[(kernel, tail)] = dict(r2_tr=r2_tr, r2_cv=r2_cv, rmse_cv=rmse_cv, yp_cv=yp_cv)
            print(f"  RBF({kernel:6s}+{tail:8s}): train R²={r2_tr:.4f}  5-fold CV R²={r2_cv:.4f}  RMSE_cv={rmse_cv:.5f}")
        except Exception as e:
            print(f"  RBF({kernel}+{tail}): FAILED — {e}")

# Also RBF on JSD (3-D real-valued) input for baseline comparison
print("\n  RBF on JSD-valued input (real-coord vs sorted-idx):")
lb_jsd = X3_jsd.min(0); ub_jsd = X3_jsd.max(0)
r2_cv_jsd, rmse_cv_jsd, _ = kfold_cv(
    X3_jsd, y3, rbf_fit_predict_factory('cubic', 'linear', lb_jsd, ub_jsd), k=5, seed=0)
print(f"  RBF(cubic+linear, JSD-input): 5-fold CV R²={r2_cv_jsd:.4f}  RMSE_cv={rmse_cv_jsd:.5f}")

# =============================================================================
# SECTION 3: Hierarchical 2-then-1 (math + RBF)
# =============================================================================
print("\n" + "="*70)
print("SECTION 3 — Hierarchical 2-then-1 combination")
print("="*70)
pair_data = {
    'WK': (xW_WK, xKV_WK, xKVD_WK, y_WK, ('W','KV')),
    'WD': (xW_WD, xKV_WD, xKVD_WD, y_WD, ('W','KVD')),
    'KD': (xW_KD, xKV_KD, xKVD_KD, y_KD, ('KV','KVD')),
}

def features_pair(xw, xkv, xkvd, vars_, mode):
    fmap = {'W':xw,'KV':xkv,'KVD':xkvd}
    a, b = fmap[vars_[0]], fmap[vars_[1]]
    n = len(a); o = np.ones(n)
    if mode == 'M1' : return np.column_stack([o, a, b])
    if mode == 'M10': return np.column_stack([o, a, b, a**2, b**2, a*b])

# Step 1: fit pair models on 2-way data
pair_models = {}
print("\n  [Step 1] Fit 2-way pair models on each pair-N=200 dataset:")
for key, (xw, xkv, xkvd, y, vars_) in pair_data.items():
    pair_models[key] = {'vars': vars_}
    for m in ['M1','M10']:
        Phi = features_pair(xw, xkv, xkvd, vars_, m)
        coef, _, r2, rmse, _ = fit_ols(Phi, y)
        pair_models[key][m] = dict(coef=coef, r2=r2, rmse=rmse)
        print(f"    {key} ({'+'.join(vars_):7s}) {m:3s}: train R²={r2:.4f}  RMSE={rmse:.5f}")
    fmap = {'W':xw,'KV':xkv,'KVD':xkvd}
    X2 = np.column_stack([fmap[vars_[0]], fmap[vars_[1]]])
    lb2 = X2.min(0); ub2 = X2.max(0)
    rbf2 = RBF(kernel='cubic', tail='linear', lb=lb2, ub=ub2)
    rbf2.fit(X2, y)
    yp2 = rbf2.predict(X2).ravel()
    r2_2 = 1 - np.sum((y-yp2)**2)/np.sum((y-y.mean())**2)
    pair_models[key]['RBF'] = dict(model=rbf2, lb=lb2, ub=ub2, r2=r2_2)
    print(f"    {key} ({'+'.join(vars_):7s}) RBF: train R²={r2_2:.4f}")

# Step 2: predict 3-way using pair model + 1D residual w/ third var
print("\n  [Step 2] Apply pair → residual model on 3-way N=200:")
fmap3 = {'W':xW3, 'KV':xKV3, 'KVD':xKVD3}
sec3 = {}
for key, (_,_,_,_, vars_) in pair_data.items():
    third = [v for v in ['W','KV','KVD'] if v not in vars_][0]
    a3, b3, t3 = fmap3[vars_[0]], fmap3[vars_[1]], fmap3[third]

    for m in ['M1','M10','RBF']:
        # Pair-only prediction
        if m == 'RBF':
            X3p = np.column_stack([a3, b3])
            yp_pair = pair_models[key]['RBF']['model'].predict(X3p).ravel()
        else:
            o3 = np.ones(N3)
            if m == 'M1':
                Phi3 = np.column_stack([o3, a3, b3])
            else:
                Phi3 = np.column_stack([o3, a3, b3, a3**2, b3**2, a3*b3])
            yp_pair = Phi3 @ pair_models[key][m]['coef']

        r2_pair_only = 1 - np.sum((y3-yp_pair)**2)/np.sum((y3-y3.mean())**2)
        residual = y3 - yp_pair

        # Linear residual (in third var)
        c_lin = np.polyfit(t3, residual, 1)
        yp_tot_lin = yp_pair + np.polyval(c_lin, t3)
        r2_lin = 1 - np.sum((y3-yp_tot_lin)**2)/np.sum((y3-y3.mean())**2)
        rmse_lin = np.sqrt(np.mean((y3-yp_tot_lin)**2))

        # 1D RBF residual (cross-validated to avoid in-sample overfit)
        try:
            X1 = t3.reshape(-1, 1); lb1 = np.array([t3.min()]); ub1 = np.array([t3.max()])
            r2_rbf_cv, rmse_rbf_cv, yp_resid_cv = kfold_cv(
                X1, residual, rbf_fit_predict_factory('cubic','linear',lb1,ub1), k=5, seed=0)
            yp_tot_rbf = yp_pair + yp_resid_cv
            r2_tot_rbf = 1 - np.sum((y3-yp_tot_rbf)**2)/np.sum((y3-y3.mean())**2)
            rmse_tot_rbf = np.sqrt(np.mean((y3-yp_tot_rbf)**2))
        except Exception:
            r2_tot_rbf, rmse_tot_rbf = np.nan, np.nan

        sec3[(key, m)] = dict(
            r2_pair_only=r2_pair_only,
            r2_hier_lin=r2_lin, rmse_hier_lin=rmse_lin,
            r2_hier_rbf=r2_tot_rbf, rmse_hier_rbf=rmse_tot_rbf,
        )
        print(f"    pair={key}({'+'.join(vars_):7s}) third={third:3s}  pair-model={m:3s}: "
              f"pair-only R²={r2_pair_only:+.3f}  +lin-res R²={r2_lin:.4f}  "
              f"+RBF-res(CV) R²={r2_tot_rbf:.4f}")

# Compare to direct 3-way (Section 1, 2)
print("\n  Direct 3-way reference:")
print(f"    M1   train R²={sec1['M1']['r2']:.4f}, 5-fold R²={sec1['M1']['r2_kf']:.4f}")
print(f"    M10  train R²={sec1['M10']['r2']:.4f}, 5-fold R²={sec1['M10']['r2_kf']:.4f}")
best_rbf = max(sec2.items(), key=lambda x: x[1]['r2_cv'])
print(f"    RBF (best CV={best_rbf[0]}): 5-fold R²={best_rbf[1]['r2_cv']:.4f}")

# =============================================================================
# SECTION 4: 27-grid + N_train sample-efficiency scan
# =============================================================================
print("\n" + "="*70)
print("SECTION 4 — 27-grid + sample-efficiency scan")
print("="*70)
qs = [0.1, 0.5, 0.9]
qW   = np.quantile(xW3,   qs)
qKV  = np.quantile(xKV3,  qs)
qKVD = np.quantile(xKVD3, qs)
print(f"  Quantiles [0.1,0.5,0.9]: W={qW.round(4)}  KV={qKV.round(4)}  KVD={qKVD.round(4)}")

scale = X3_jsd.std(0) + 1e-10
X3n = X3_jsd / scale
grid27 = np.array([[w,kv,kvd] for w in qW for kv in qKV for kvd in qKVD]) / scale
grid_idx = np.array([np.argmin(np.sum((X3n - g)**2, axis=1)) for g in grid27])
grid_used = np.unique(grid_idx)
test_idx  = np.setdiff1d(np.arange(N3), grid_used)
print(f"  27-grid → {len(grid_used)} unique data points (test on {len(test_idx)} excluded)")

print("\n  Train on 27-grid, test on the rest:")
sec4_grid = {}
for m in ['M1','M10','RBF']:
    if m == 'RBF':
        rbf_g = RBF(kernel='cubic', tail='linear', lb=lb_idx, ub=ub_idx)
        rbf_g.fit(X3_idx[grid_used], y3[grid_used])
        yp_te = rbf_g.predict(X3_idx[test_idx]).ravel()
    else:
        Phi_tr = features(xW3[grid_used], xKV3[grid_used], xKVD3[grid_used], m)
        Phi_te = features(xW3[test_idx],  xKV3[test_idx],  xKVD3[test_idx],  m)
        coef, *_ = np.linalg.lstsq(Phi_tr, y3[grid_used], rcond=None)
        yp_te = Phi_te @ coef
    r2 = 1 - np.sum((y3[test_idx]-yp_te)**2)/np.sum((y3[test_idx]-y3[test_idx].mean())**2)
    rmse = np.sqrt(np.mean((y3[test_idx]-yp_te)**2))
    sec4_grid[m] = dict(r2=r2, rmse=rmse)
    print(f"    {m:3s}: test R²={r2:.4f}  RMSE={rmse:.5f}")

print("\n  Sample-efficiency scan (median over 30 random seeds):")
n_seed = 30
scan_sizes = [27, 50, 75, 100, 125, 150, 175]
scan_res = {m: {n: [] for n in scan_sizes} for m in ['M1','M10','RBF']}
for seed in range(n_seed):
    rng = np.random.RandomState(seed)
    for n_tr in scan_sizes:
        if n_tr >= N3: continue
        tr = rng.choice(N3, n_tr, replace=False)
        te = np.setdiff1d(np.arange(N3), tr)
        for m in ['M1','M10']:
            try:
                Phi_tr = features(xW3[tr], xKV3[tr], xKVD3[tr], m)
                Phi_te = features(xW3[te], xKV3[te], xKVD3[te], m)
                coef, *_ = np.linalg.lstsq(Phi_tr, y3[tr], rcond=None)
                yp_te = Phi_te @ coef
                r2 = 1 - np.sum((y3[te]-yp_te)**2)/np.sum((y3[te]-y3[te].mean())**2)
                scan_res[m][n_tr].append(r2)
            except Exception: pass
        try:
            rbf_s = RBF(kernel='cubic', tail='linear', lb=lb_idx, ub=ub_idx)
            rbf_s.fit(X3_idx[tr], y3[tr])
            yp_te = rbf_s.predict(X3_idx[te]).ravel()
            r2 = 1 - np.sum((y3[te]-yp_te)**2)/np.sum((y3[te]-y3[te].mean())**2)
            scan_res['RBF'][n_tr].append(r2)
        except Exception: pass

for n_tr in scan_sizes:
    if n_tr >= N3: continue
    parts = [f"n={n_tr:3d}"]
    for m in ['M1','M10','RBF']:
        rs = scan_res[m][n_tr]
        if rs: parts.append(f"{m}: med={np.median(rs):.4f} [10%,90%]=[{np.percentile(rs,10):.4f},{np.percentile(rs,90):.4f}]")
    print("    " + "  ".join(parts))

# =============================================================================
# SECTION 5: 3-tier (low/mid/high) bit-regime split
# =============================================================================
print("\n" + "="*70)
print("SECTION 5 — 3-tier bit-regime split (low / mid / high JSD)")
print("="*70)
tW   = np.quantile(xW3,   [1/3, 2/3])
tKV  = np.quantile(xKV3,  [1/3, 2/3])
tKVD = np.quantile(xKVD3, [1/3, 2/3])
print(f"  Tercile cuts: W={tW.round(4)}  KV={tKV.round(4)}  KVD={tKVD.round(4)}")
print(f"  Convention: low JSD ⇔ high-bit (mild quant); high JSD ⇔ low-bit (aggressive)")

def regime(vals, ts): return np.searchsorted(ts, vals)
regW   = regime(xW3,   tW)
regKV  = regime(xKV3,  tKV)
regKVD = regime(xKVD3, tKVD)

print("\n  [A] Per-method conditional fits — fix one method's regime, fit M1 on others:")
sec5_cond = {}
for cond_name, cond_reg in [('W', regW), ('KV', regKV), ('KVD', regKVD)]:
    for tier in [0, 1, 2]:
        mask = cond_reg == tier
        n = int(mask.sum())
        if n < 8:
            print(f"    cond={cond_name} tier={tier} N={n} — skip"); continue
        Phi = features(xW3[mask], xKV3[mask], xKVD3[mask], 'M1')
        coef, _, r2, rmse, _ = fit_ols(Phi, y3[mask])
        sec5_cond[(cond_name, tier)] = dict(r2=r2, rmse=rmse, coef=coef, n=n)
        tn = ['low(high-bit)','mid','high(low-bit)'][tier]
        print(f"    cond={cond_name:3s} tier={tier} {tn:14s} N={n:3d}: "
              f"R²={r2:.4f}  α_W={coef[1]:+.3f}  α_KV={coef[2]:+.3f}  α_KVD={coef[3]:+.3f}")

print("\n  [B] 27-octant cell counts:")
oct_counts = np.zeros((3,3,3), int)
for w in range(3):
    for kv in range(3):
        for kvd in range(3):
            oct_counts[w,kv,kvd] = int(((regW==w)&(regKV==kv)&(regKVD==kvd)).sum())
print(f"  Total 27 cells: {oct_counts.sum()}, ≥10: {(oct_counts>=10).sum()}, ≥6: {(oct_counts>=6).sum()}")
print("  Counts (printed as W=0..2 blocks of KV×KVD matrix):")
for w in range(3):
    print(f"    W={w}: " + str(oct_counts[w].tolist()))

print("\n  [C] M10 (full quad) R² split by single method's tier:")
sec5_split = {}
for name, reg in [('W', regW), ('KV', regKV), ('KVD', regKVD)]:
    line = f"    split by {name:3s}: "
    for tier in [0,1,2]:
        mask = reg == tier; n = int(mask.sum())
        if n < 12:
            line += f"t{tier}({['lo','md','hi'][tier]}):N={n} "; continue
        try:
            Phi = features(xW3[mask], xKV3[mask], xKVD3[mask], 'M10')
            if np.linalg.matrix_rank(Phi) < Phi.shape[1]:
                line += f"t{tier}:rank-def "; continue
            coef, _, r2, rmse, _ = fit_ols(Phi, y3[mask])
            sec5_split[(name, tier)] = dict(r2=r2, rmse=rmse, n=n)
            line += f"t{tier}({['lo','md','hi'][tier]}): R²={r2:.3f}(N={n}) "
        except Exception as e:
            line += f"t{tier}:err "
    print(line)

print("\n  [D] Additivity ratio per tier — (interaction R² gain over M1):")
print("       Δ = R²_M10 − R²_M1; high Δ ⇒ stronger non-additivity")
for name, reg in [('W', regW), ('KV', regKV), ('KVD', regKVD)]:
    line = f"    by {name:3s}: "
    for tier in [0,1,2]:
        mask = reg == tier; n = int(mask.sum())
        if n < 12:
            line += f"t{tier}:N={n} "; continue
        Phi1  = features(xW3[mask], xKV3[mask], xKVD3[mask], 'M1')
        Phi10 = features(xW3[mask], xKV3[mask], xKVD3[mask], 'M10')
        try:
            _,_,r1,*_ = fit_ols(Phi1, y3[mask])
            if np.linalg.matrix_rank(Phi10) < Phi10.shape[1]:
                line += f"t{tier}:rank-def "; continue
            _,_,r10,*_ = fit_ols(Phi10, y3[mask])
            line += f"t{tier}({['lo','md','hi'][tier]}): Δ={r10-r1:+.3f} (R²={r1:.3f}→{r10:.3f}) "
        except Exception:
            line += f"t{tier}:err "
    print(line)

# =============================================================================
# FIGURES
# =============================================================================
print("\nGenerating figures...")
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT = dict(dpi=170, bbox_inches='tight')
C_W='#E64B35'; C_KV='#4DBBD5'; C_KVD='#00A087'; C_AWQ='#3C5488'

# Fig v3-1: Section 1 model comparison
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
mods = ['M0','M1','M8','M9','M10']
for ax, (vals, ylab, fmt) in zip(axes, [
    ([sec1[m]['r2']      for m in mods], 'Train R²',      '.4f'),
    ([sec1[m]['rmse_loo']for m in mods], 'LOOCV-RMSE',   '.5f'),
    ([sec1[m]['r2_kf']   for m in mods], '5-fold CV R²', '.4f'),
]):
    bars = ax.bar(mods, vals, color=[C_W,C_AWQ,C_KV,C_KVD,'#9B59B6'],
                  alpha=0.85, edgecolor='#333', width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), format(v,fmt),
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_ylabel(ylab); ax.grid(True, alpha=0.25, lw=0.5, axis='y')
plt.suptitle('Section 1 — Math model comparison on 3-way N=200', fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/v3_fig1_math_models.png', **PLT); plt.close()

# Fig v3-2: RBF kernel/tail comparison
fig, ax = plt.subplots(figsize=(8, 3.8))
keys = list(sec2.keys())
labels = [f"{k}+{t}" for k,t in keys]
r2_tr = [sec2[k]['r2_tr']  for k in keys]
r2_cv = [sec2[k]['r2_cv']  for k in keys]
x = np.arange(len(keys))
ax.bar(x-0.18, r2_tr, 0.36, label='train R²', color=C_KV, alpha=0.8, edgecolor='#333')
ax.bar(x+0.18, r2_cv, 0.36, label='5-fold CV R²', color=C_W, alpha=0.85, edgecolor='#333')
ax.axhline(sec1['M1' ]['r2_kf'], color='black', ls='--', lw=1, label=f'M1 5-fold={sec1["M1"]["r2_kf"]:.3f}')
ax.axhline(sec1['M10']['r2_kf'], color='gray',  ls='--', lw=1, label=f'M10 5-fold={sec1["M10"]["r2_kf"]:.3f}')
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel('R²'); ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25, lw=0.5, axis='y')
ax.set_title('Section 2 — RBF predictor on sorted-index input (3-way N=200)', fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/v3_fig2_rbf.png', **PLT); plt.close()

# Fig v3-3: Hierarchical 2-then-1 R² heatmap
fig, ax = plt.subplots(figsize=(9, 4))
rows = ['WK','WD','KD']; cols = ['M1','M10','RBF']
mat = np.zeros((len(rows), len(cols)))
mat_lin = np.zeros_like(mat)
for i,r in enumerate(rows):
    for j,c in enumerate(cols):
        mat[i,j]    = sec3[(r,c)]['r2_hier_rbf']
        mat_lin[i,j]= sec3[(r,c)]['r2_hier_lin']
im = ax.imshow(mat, cmap='viridis', vmin=0.85, vmax=1.0, aspect='auto')
for i in range(len(rows)):
    for j in range(len(cols)):
        ax.text(j, i, f"+RBF res:\n{mat[i,j]:.4f}\n(+lin: {mat_lin[i,j]:.4f})",
                ha='center', va='center', color='white' if mat[i,j]<0.93 else 'black', fontsize=8.5)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"pair-{c}" for c in cols])
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"pair={r}" for r in rows])
ax.set_title('Section 3 — Hierarchical 2-then-1: R² on 3-way (rows: pair, cols: pair-model)', fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/v3_fig3_hierarchical.png', **PLT); plt.close()

# Fig v3-4: Sample-efficiency scan
fig, ax = plt.subplots(figsize=(8, 4.5))
for m, color in zip(['M1','M10','RBF'], [C_W, C_KV, C_KVD]):
    xs, ys, lo, hi = [], [], [], []
    for n in scan_sizes:
        if n >= N3: continue
        rs = scan_res[m][n]
        if not rs: continue
        xs.append(n); ys.append(np.median(rs))
        lo.append(np.percentile(rs, 10)); hi.append(np.percentile(rs, 90))
    ax.plot(xs, ys, '-o', color=color, label=m, lw=2, ms=6)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
ax.axvline(27, color='black', ls='--', alpha=0.5, label='27-grid')
ax.scatter([27], [sec4_grid['M1' ]['r2']], marker='^', color=C_W,  s=80, zorder=5, label='M1 27-grid')
ax.scatter([27], [sec4_grid['M10']['r2']], marker='s', color=C_KV, s=80, zorder=5, label='M10 27-grid')
ax.scatter([27], [sec4_grid['RBF']['r2']], marker='*', color=C_KVD, s=120, zorder=5, label='RBF 27-grid')
ax.set_xlabel('Number of training samples'); ax.set_ylabel('Out-of-sample R²')
ax.set_title('Section 4 — Sample-efficiency (median, [10,90]% band over 30 seeds)', fontweight='bold')
ax.legend(fontsize=8.5, ncol=2); ax.grid(True, alpha=0.25, lw=0.5)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/v3_fig4_sample_efficiency.png', **PLT); plt.close()

# Fig v3-5: 3-tier bit-regime — slopes & R² across tiers (per condition method)
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
methods = ['W','KV','KVD']
for col, cond in enumerate(methods):
    # Top: slopes by tier
    ax = axes[0, col]
    tiers = [0,1,2]
    aW = [sec5_cond.get((cond, t), {'coef':[0,0,0,0]})['coef'][1] for t in tiers]
    aKV = [sec5_cond.get((cond, t), {'coef':[0,0,0,0]})['coef'][2] for t in tiers]
    aKVD = [sec5_cond.get((cond, t), {'coef':[0,0,0,0]})['coef'][3] for t in tiers]
    x = np.arange(3); w = 0.27
    ax.bar(x-w, aW,  w, label='α_W',  color=C_W,  alpha=0.85, edgecolor='#333')
    ax.bar(x,   aKV, w, label='α_KV', color=C_KV, alpha=0.85, edgecolor='#333')
    ax.bar(x+w, aKVD,w, label='α_KVD',color=C_KVD,alpha=0.85, edgecolor='#333')
    ax.axhline(0, color='black', lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(['low\n(high-bit)','mid','high\n(low-bit)'])
    ax.set_ylabel('M1 slope'); ax.set_title(f'Condition on {cond} regime', fontweight='bold')
    if col == 0: ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, lw=0.5, axis='y')
    # Bottom: M10 vs M1 R² with Δ
    ax = axes[1, col]
    r1  = [sec5_cond.get((cond,t), {'r2':np.nan})['r2'] for t in tiers]
    r10 = [sec5_split.get((cond,t), {'r2':np.nan})['r2'] for t in tiers]
    ax.plot(['low','mid','high'], r1,  '-o', color=C_W,  label='M1', lw=2, ms=8)
    ax.plot(['low','mid','high'], r10, '-s', color=C_KVD, label='M10', lw=2, ms=8)
    for i,(a,b) in enumerate(zip(r1, r10)):
        if not (np.isnan(a) or np.isnan(b)):
            ax.text(i, max(a,b)+0.005, f'Δ={b-a:+.3f}', ha='center', fontsize=8)
    ax.set_ylabel('R²'); ax.set_xlabel(f'{cond} tier')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, lw=0.5)
plt.suptitle('Section 5 — 3-tier bit-regime split: slopes (top) and additivity gap (bottom)', fontweight='bold', y=1.005)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/v3_fig5_regime_split.png', **PLT); plt.close()

# Fig v3-6: Combined summary
fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

# (0,0) Section 1 R²
ax = fig.add_subplot(gs[0, 0])
r2s = [sec1[m]['r2'] for m in mods]; r2cv = [sec1[m]['r2_kf'] for m in mods]
x = np.arange(len(mods)); w=0.35
ax.bar(x-w/2, r2s,  w, label='train', color=C_AWQ, alpha=0.85, edgecolor='#333')
ax.bar(x+w/2, r2cv, w, label='5-fold CV', color=C_W, alpha=0.85, edgecolor='#333')
ax.set_xticks(x); ax.set_xticklabels(mods); ax.set_ylim([0.93, 1.0])
ax.set_ylabel('R²'); ax.set_title('(a) Math models', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.25, lw=0.5, axis='y')

# (0,1) RBF best
ax = fig.add_subplot(gs[0, 1])
keys = list(sec2.keys()); labels = [f"{k}+{t}" for k,t in keys]
ax.bar(labels, [sec2[k]['r2_cv'] for k in keys], color=C_KV, alpha=0.85, edgecolor='#333')
ax.set_ylabel('5-fold CV R²'); ax.set_title('(b) RBF kernel/tail', fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
ax.grid(True, alpha=0.25, lw=0.5, axis='y')

# (0,2) Hierarchical R²
ax = fig.add_subplot(gs[0, 2])
hier_rows = []
for r in rows:
    for c in cols: hier_rows.append((f"{r}-{c}", sec3[(r,c)]['r2_hier_rbf']))
ax.bar([h[0] for h in hier_rows], [h[1] for h in hier_rows],
       color=C_KVD, alpha=0.85, edgecolor='#333')
ax.axhline(sec1['M10']['r2_kf'], color='black', ls='--', lw=1.2, label=f'direct M10 CV={sec1["M10"]["r2_kf"]:.3f}')
ax.set_ylabel('R² on 3-way (RBF residual)'); ax.set_title('(c) Hierarchical 2-then-1', fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
ax.legend(fontsize=8); ax.grid(True, alpha=0.25, lw=0.5, axis='y')

# (1,0) Sample efficiency
ax = fig.add_subplot(gs[1, 0])
for m, color in zip(['M1','M10','RBF'], [C_W, C_KV, C_KVD]):
    xs, ys = [], []
    for n in scan_sizes:
        if n>=N3: continue
        rs = scan_res[m][n]
        if rs: xs.append(n); ys.append(np.median(rs))
    ax.plot(xs, ys, '-o', color=color, label=m, lw=2, ms=5)
ax.set_xlabel('N_train'); ax.set_ylabel('Median test R²')
ax.set_title('(d) Sample efficiency', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.25, lw=0.5)

# (1,1) 3-tier R² heatmap (cond × tier)
ax = fig.add_subplot(gs[1, 1])
M = np.full((3,3), np.nan)
for i, cond in enumerate(methods):
    for t in [0,1,2]:
        M[i,t] = sec5_cond.get((cond,t), {'r2':np.nan})['r2']
im = ax.imshow(M, cmap='RdYlGn', vmin=0.7, vmax=1.0, aspect='auto')
for i in range(3):
    for t in range(3):
        ax.text(t, i, f"{M[i,t]:.3f}" if not np.isnan(M[i,t]) else 'N/A',
                ha='center', va='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels(['low','mid','high'])
ax.set_yticks(range(3)); ax.set_yticklabels([f'cond {m}' for m in methods])
ax.set_title('(e) M1 R² by conditioning regime', fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.04)

# (1,2) Additivity gap (M10−M1)
ax = fig.add_subplot(gs[1, 2])
G = np.full((3,3), np.nan)
for i, cond in enumerate(methods):
    for t in [0,1,2]:
        a = sec5_cond.get((cond,t), {}).get('r2', np.nan)
        b = sec5_split.get((cond,t), {}).get('r2', np.nan)
        if not (np.isnan(a) or np.isnan(b)): G[i,t] = b - a
im = ax.imshow(G, cmap='magma', aspect='auto')
for i in range(3):
    for t in range(3):
        if not np.isnan(G[i,t]):
            ax.text(t, i, f"{G[i,t]:+.3f}", ha='center', va='center',
                    color='white' if G[i,t]<G[~np.isnan(G)].mean() else 'black', fontsize=10, fontweight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels(['low','mid','high'])
ax.set_yticks(range(3)); ax.set_yticklabels([f'cond {m}' for m in methods])
ax.set_title('(f) Additivity gap Δ = R²(M10) − R²(M1)', fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.04)

fig.suptitle('analysis_v3 — Extended additive-loss approximation analysis (Llama-3.1-8B-Inst, AWQ N=200)',
             fontweight='bold', fontsize=13, y=1.01)
plt.savefig(f'{FIG_DIR}/v3_fig6_summary.png', **PLT); plt.close()

print(f"\nAll figures saved to {FIG_DIR}/v3_fig*.png")
print("Done.\n")
