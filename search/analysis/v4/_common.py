"""_common.py — shared helpers for v4 analysis on 260510 CSV files.

CSV row layout (see get_net_info({}, ..) for first 12 keys, then datasets, then pf metrics):
    rows 0..11  : complexity (wbits, kvbits, kbits, vbits, kvdim, kdim, vdim,
                  eff_kvbits, eff_kbits, eff_vbits, memory, n_token)
    row  12     : measured metric on first dataset (wikitext2 ppl/cross-entropy here)
    row  13     : combined predicted metric  (== sum of per-method z's for 3-way)
    rows 14..   : per-method z values  — order matches expr_keys
                  3-way: row14=z_W, row15=z_KV, row16=z_KVD
                  2-way: row14=z_first, row15=z_second
"""
import csv, json
import numpy as np

BASE = '/NAS/SJ/actquant/search'
DATA = f'{BASE}/save/result/260510'

PATHS = {
    # 3-way joint
    'llama_qs50':  f'{DATA}/2605101952_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_qs_w159_kv159_kvdim159_rs23/results.csv',
    'llama_rs50':  f'{DATA}/2605101952_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_rs50/results.csv',
    'llama_rs200': f'{DATA}/2605101953_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_rs200/results.csv',
    'qwen_qs50':   f'{DATA}/2605101953_Qwen2.5-7B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_qs_w159_kv159_kvdim159_rs23/results.csv',
    'qwen_rs50':   f'{DATA}/2605112224_Qwen2.5-7B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_rs50/results.csv',
    'qwen_rs200':  f'{DATA}/2605101953_Qwen2.5-7B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_rs200/results.csv',
    # 2-axis QS (hierarchical pair builds)
    'llama_wk':    f'{DATA}/2605121123_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_qs_w159_kv159_rs41/results.csv',
    'llama_kvkd':  f'{DATA}/2605121124_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_kv_expr_kvdim_expr_qs_kv159_kvdim159_rs41/results.csv',
    'llama_wkd':   f'{DATA}/2605121136_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kvdim_expr_qs_w159_kvdim159_rs41/results.csv',
}

AXIS_NAMES_3 = ('W', 'KV', 'KVD')


def load_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    nc = max(len(r) for r in rows)
    M = np.full((len(rows), nc), np.nan)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            try: M[i, j] = float(v)
            except ValueError: pass
    return M


def extract_xy(mat, n_axes=3):
    """Return X (per-method z's, shape (N, n_axes)), y (actual metric), y_add (additive prediction).

    Skips columns where the actual metric (row 12) is NaN.
    """
    ncol = mat.shape[1]
    y_all = mat[12, :ncol]
    v = ~np.isnan(y_all)
    X = mat[14:14 + n_axes, :ncol].T[v]
    y = y_all[v]
    y_add = mat[13, :ncol][v] if mat.shape[0] > 13 else X.sum(1)
    comp = {
        'wbits':  mat[0, :ncol][v],
        'kvbits': mat[1, :ncol][v],
        'kvdim':  mat[4, :ncol][v],
        'eff_kvbits': mat[7, :ncol][v],
        'memory': mat[10, :ncol][v],
    }
    return X, y, y_add, comp


def r2(y, yp):
    ss_r = np.sum((y - yp) ** 2)
    ss_t = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_r / max(ss_t, 1e-30))


def rmse(y, yp):
    return float(np.sqrt(np.mean((y - yp) ** 2)))


def eps_inf(y, yp):
    return float(np.max(np.abs(y - yp)))


# ─── Surrogates ───────────────────────────────────────────────────────────────
def features_M1(X):
    n = len(X); o = np.ones(n)
    cols = [o] + [X[:, i] for i in range(X.shape[1])]
    return np.column_stack(cols)


def features_M_full_quad(X):
    """M10 (3-axis): intercept + 3 main + 3 sq + 3 pair. Generalises to D-axis: 1 + D + D + D*(D-1)/2."""
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d): cols.append(X[:, i])
    for i in range(d): cols.append(X[:, i] ** 2)
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)


def fit_ols(Phi_tr, y_tr, Phi_te):
    coef, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
    return Phi_tr @ coef, Phi_te @ coef, coef


def fit_rbf(X_tr, y_tr, X_te, kernel='tps'):
    import sys
    sys.path.insert(0, BASE)
    from predictor.rbf import RBF as PySOTRBF
    lb = np.minimum(X_tr.min(0), X_te.min(0))
    ub = np.maximum(X_tr.max(0), X_te.max(0))
    m = PySOTRBF(kernel=kernel, tail='linear', lb=lb, ub=ub)
    m.fit(X_tr, y_tr)
    return m.predict(X_tr).ravel(), m.predict(X_te).ravel(), m


def fit_ard_gp(X_tr, y_tr, X_te, kernel='matern32', with_noise=True, n_restarts=30):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (RBF as SKRBF, ConstantKernel as C,
                                                  WhiteKernel, Matern, RationalQuadratic)
    d = X_tr.shape[1]; ls0 = [1.0] * d; lsb = (1e-4, 1e4)
    base = C(1.0, (1e-4, 1e2))
    if kernel == 'rbf':       core = SKRBF(ls0, lsb)
    elif kernel == 'matern52':core = Matern(ls0, lsb, nu=2.5)
    elif kernel == 'matern32':core = Matern(ls0, lsb, nu=1.5)
    elif kernel == 'rq':      core = RationalQuadratic(1.0, 1.0, lsb, (1e-2, 1e2))
    else: raise ValueError(kernel)
    k = base * core
    if with_noise:
        k = k + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2))
        alpha = 1e-8
    else:
        alpha = 1e-10
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=n_restarts, alpha=alpha)
    gp.fit(X_tr, y_tr)
    return gp.predict(X_tr), gp.predict(X_te), gp


def get_ard_lengthscales(gp, with_noise=True, d=3):
    k = gp.kernel_
    if with_noise: k = k.k1
    core = k.k2
    ls = getattr(core, 'length_scale', None)
    if ls is None: return None
    arr = np.atleast_1d(np.asarray(ls, dtype=float))
    if arr.size == 1: arr = np.full(d, arr[0])
    return arr


def all_surrogate_fits(X_tr, y_tr, X_te, ard_kernel='matern32', n_restarts=30):
    """Run M1, M_full_quad, RBF cubic, RBF tps, ARD-GP. Returns dict {name: {yp_tr, yp_te, model}}."""
    out = {}
    Phi_tr1, Phi_te1 = features_M1(X_tr), features_M1(X_te)
    ytr1, yte1, c1 = fit_ols(Phi_tr1, y_tr, Phi_te1)
    out['M1 linear additive'] = dict(yp_tr=ytr1, yp_te=yte1, coef=c1)

    Phi_trQ, Phi_teQ = features_M_full_quad(X_tr), features_M_full_quad(X_te)
    ytrQ, yteQ, cQ = fit_ols(Phi_trQ, y_tr, Phi_teQ)
    out['M_quad full quadratic'] = dict(yp_tr=ytrQ, yp_te=yteQ, coef=cQ)

    yptr, ypte, _ = fit_rbf(X_tr, y_tr, X_te, kernel='cubic')
    out['RBF cubic+linear'] = dict(yp_tr=yptr, yp_te=ypte)

    yptr, ypte, _ = fit_rbf(X_tr, y_tr, X_te, kernel='tps')
    out['RBF tps+linear']   = dict(yp_tr=yptr, yp_te=ypte)

    yptr, ypte, gp = fit_ard_gp(X_tr, y_tr, X_te, kernel=ard_kernel, with_noise=True, n_restarts=n_restarts)
    out[f'ARD-GP ({ard_kernel}+noise)'] = dict(yp_tr=yptr, yp_te=ypte, gp=gp)
    return out


def pareto_front_2d(F):
    """Return indices of Pareto front in F (minimise both columns)."""
    order = np.argsort(F[:, 0]); F_s = F[order]; m = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < m: nd.append(i); m = F_s[i, 1]
    return order[nd]
