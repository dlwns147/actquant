"""03_test_lsbounds.py — quick sensitivity test of length_scale_bounds in the ARD-GP kernel sweep.

Reuses Phase-1 train (50 QS) / test (200 AWQ) split. Reruns the same kernel sweep with
length_scale_bounds = (1e-3, 1e3) and compares to the (1e-4, 1e4) baseline (Phase 1).
"""
import sys, os, json, csv, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF as SKRBF, ConstantKernel as C,
                                                WhiteKernel, Matern, RationalQuadratic)
from utils.func import get_net_info
from predictor.rbf import RBF as PySOTRBF

# ─── Helpers (copy of Phase 1) ────────────────────────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]; min2 = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < min2: nd.append(i); min2 = F_s[i, 1]
    return order[nd]
def load_archive_pareto(stats_path, comp_key, config, group_size):
    with open(stats_path) as f: data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs = [v[0] for v in archive]
    metrics = np.array([v[1] for v in archive])
    comps = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
    F = np.column_stack((metrics, comps))
    return F[pareto_front_2d(F)]
def load_csv(path):
    with open(path) as f: rows = [r for r in csv.reader(f) if r]
    max_cols = max(len(r) for r in rows)
    mat = np.full((len(rows), max_cols), np.nan)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            try: mat[i, j] = float(v)
            except: pass
    return mat
def match_metric(comp_vals, pf):
    return np.array([pf[np.argmin(np.abs(pf[:, 1] - c)), 0] for c in comp_vals])
def r2(y_t, y_p):
    ss_r = np.sum((y_t - y_p)**2); ss_t = np.sum((y_t - y_t.mean())**2)
    return 1 - ss_r / max(ss_t, 1e-30)
def rmse(y_t, y_p): return float(np.sqrt(np.mean((y_t - y_p)**2)))

def make_kernel(name, n_dim, ls_bnd):
    base_C = C(1.0, (1e-4, 1e2))
    ls = [1.0] * n_dim
    if name == 'rbf':       core = SKRBF(ls, ls_bnd)
    elif name == 'matern52':core = Matern(ls, ls_bnd, nu=2.5)
    elif name == 'matern32':core = Matern(ls, ls_bnd, nu=1.5)
    elif name == 'rq':      core = RationalQuadratic(length_scale=1.0, alpha=1.0,
                                                     length_scale_bounds=ls_bnd,
                                                     alpha_bounds=(1e-2, 1e2))
    else: raise ValueError(name)
    return base_C * core

def fit_variant(X_tr, y_tr, X_te, kernel_name, with_noise, ls_bnd, n_restarts=50):
    n_dim = X_tr.shape[1]
    k = make_kernel(kernel_name, n_dim, ls_bnd)
    if with_noise:
        k = k + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2))
        alpha = 1e-8
    else:
        alpha = 1e-10
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=alpha)
    gp.fit(X_tr, y_tr)
    return gp.predict(X_tr), gp.predict(X_te), gp

# ─── Load data (same as Phase 1) ──────────────────────────────────────────────
BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
QS_3WAY  = f'{BASE}/save/result/260506/llama_3.1_8b_inst_quantile_sample/2605060818_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kvdim_expr_qs_w159_kv159_kvdim159_rs23/results.csv'

with open(f'{BASE}/config/llama.json') as f:
    config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)

mat_qs = load_csv(QS_3WAY); n_qs = mat_qs.shape[1]
y_tr_raw = mat_qs[12, :n_qs]; v_tr = ~np.isnan(y_tr_raw)
xW_tr = match_metric(mat_qs[0,:n_qs], pf_W)[v_tr]
xKV_tr= match_metric(mat_qs[1,:n_qs], pf_KV)[v_tr]
xKVD_tr=match_metric(mat_qs[4,:n_qs], pf_KVDIM)[v_tr]
y_tr = y_tr_raw[v_tr]; X_tr = np.column_stack([xW_tr,xKV_tr,xKVD_tr])

mat_aw = load_csv(AWQ_3WAY); n_aw = mat_aw.shape[1]
y_te_raw = mat_aw[12, :n_aw]; v_te = ~np.isnan(y_te_raw)
xW_te = match_metric(mat_aw[0,:n_aw], pf_W)[v_te]
xKV_te= match_metric(mat_aw[1,:n_aw], pf_KV)[v_te]
xKVD_te=match_metric(mat_aw[4,:n_aw], pf_KVDIM)[v_te]
y_te = y_te_raw[v_te]; X_te = np.column_stack([xW_te,xKV_te,xKVD_te])

# ─── Sweep both bounds ────────────────────────────────────────────────────────
kernel_variants = [('rbf', True), ('rbf', False),
                   ('matern52', True), ('matern52', False),
                   ('matern32', True), ('matern32', False),
                   ('rq', True), ('rq', False)]
bounds_to_test = [('1e-4..1e4', (1e-4, 1e4)),
                  ('1e-3..1e3', (1e-3, 1e3))]

rows = {}
for tag, ls_bnd in bounds_to_test:
    print(f"\n=== length_scale_bounds = {tag} ===")
    print(f"  {'kernel':22s}  {'LMML':>10s}  {'R²_test':>10s}  {'RMSE':>10s}  {'eps_∞':>10s}  {'l_W,l_KV,l_KVD':24s}")
    print("  " + "-"*100)
    for kn, wn in kernel_variants:
        name = f"ARD-{kn}{' +noise' if wn else ' nonoise'}"
        yp_tr_v, yp_te_v, gp_v = fit_variant(X_tr, y_tr, X_te, kn, wn, ls_bnd, n_restarts=50)
        lmml = float(gp_v.log_marginal_likelihood_value_)
        r2_te = r2(y_te, yp_te_v); rm = rmse(y_te, yp_te_v)
        eps_inf = float(np.max(np.abs(y_te - yp_te_v)))
        # Extract length scales
        kk = gp_v.kernel_
        if wn: kk = kk.k1
        ls_attr = getattr(kk.k2, 'length_scale', None)
        if ls_attr is not None:
            ls = np.atleast_1d(np.asarray(ls_attr, dtype=float))
            if ls.size == 1: ls = np.full(3, ls[0])
            ls_str = f"{ls[0]:.3f},{ls[1]:.3f},{ls[2]:.3f}"
        else:
            ls_str = "-"
        rows[(tag, name)] = dict(lmml=lmml, r2_test=float(r2_te), rmse=rm, eps_inf=eps_inf,
                                  length_scales=ls_str)
        print(f"  {name:22s}  {lmml:10.3f}  {r2_te:10.4f}  {rm:10.5f}  {eps_inf:10.5f}  {ls_str:24s}")

# ─── Side-by-side comparison ──────────────────────────────────────────────────
print("\n" + "="*100)
print(f"{'Kernel':22s}  {'1e-4..1e4':>34s}  {'1e-3..1e3':>34s}  {'Δ R²':>10s}")
print(f"{'':22s}  {'LMML':>10s} {'R²':>10s} {'RMSE':>10s}  {'LMML':>10s} {'R²':>10s} {'RMSE':>10s}")
print("-"*116)
for kn, wn in kernel_variants:
    name = f"ARD-{kn}{' +noise' if wn else ' nonoise'}"
    a = rows[('1e-4..1e4', name)]; b = rows[('1e-3..1e3', name)]
    dR2 = b['r2_test'] - a['r2_test']
    print(f"{name:22s}  "
          f"{a['lmml']:10.3f} {a['r2_test']:10.4f} {a['rmse']:10.5f}  "
          f"{b['lmml']:10.3f} {b['r2_test']:10.4f} {b['rmse']:10.5f}  "
          f"{dR2:+10.4f}")

# Best per bound by LMML
print("\nBest by training LMML:")
for tag, _ in bounds_to_test:
    best = max(((kn, wn) for kn, wn in kernel_variants),
               key=lambda x: rows[(tag, f"ARD-{x[0]}{' +noise' if x[1] else ' nonoise'}")]['lmml'])
    name = f"ARD-{best[0]}{' +noise' if best[1] else ' nonoise'}"
    r = rows[(tag, name)]
    print(f"  bounds={tag}: {name}  (LMML={r['lmml']:.3f}, R²_test={r['r2_test']:.4f}, "
          f"RMSE={r['rmse']:.5f}, ls={r['length_scales']})")

# Save JSON
out = {'bounds_tested': [t for t, _ in bounds_to_test],
       'rows': {f"{t}|{n}": v for (t, n), v in rows.items()}}
with open(f'{BASE}/analysis/v4/lsbounds_test.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {BASE}/analysis/v4/lsbounds_test.json")
