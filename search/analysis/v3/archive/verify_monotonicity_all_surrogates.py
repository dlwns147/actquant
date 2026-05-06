"""verify_monotonicity_all_surrogates.py

Theorem 3 (local PF Cartesian product coverage) requires the surrogate
G(z_W, z_KV, z_KVD) to be coordinate-wise nondecreasing in z_i on the
candidate-relevant region D_z. This script audits this condition for each
surrogate family by computing min_z ∂G/∂z_i and the fraction of points where
the partial derivative is nonneg.

Surrogates audited:
  M0   : β_W z_W + β_KV z_KV + β_KVD z_KVD             — coefficient signs
  M1   : β_0 + β_W z_W + β_KV z_KV + β_KVD z_KVD       — coefficient signs
  M10  : full quadratic (affine ∂G/∂z_i)               — 8-corner check + dense
  RBF cubic+linear (JSD)                                — dense gradient audit
  RBF tps+linear   (JSD)                                — dense gradient audit
  ARD-GP                                                — dense gradient audit

Audit sets:
  D_meas : 200 measured joint samples (candidate-relevant proxy)
  D_box  : 5000 LH fill of [z_min, z_max]^3 (extrapolation included)
  D_corn : 8 box corners (M10 affine derivative check)
"""
import sys, os, json, csv, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SKRBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import linear_sum_assignment

from utils.func import get_net_info
from predictor.rbf import RBF as PySOTRBF

# ─── Data loading ───────────────────────────────────────────────────────────
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

BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY  = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'

with open(f'{BASE}/config/llama.json') as f: config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w':128, 'k':[128,128], 'v':[128,128]}

print("Loading data...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y_all = mat3[12, :N0]; v3 = ~np.isnan(y_all)
xW3   = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3  = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD  = match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y_all = y_all[v3]
N = len(y_all)
X3 = np.column_stack([xW3, xKV3, xKVD])
bounds_jsd = np.array([[xW3.min(), xW3.max()],
                       [xKV3.min(), xKV3.max()],
                       [xKVD.min(), xKVD.max()]])
print(f"  N={N}  bounds: W=[{bounds_jsd[0,0]:.3f},{bounds_jsd[0,1]:.3f}], "
      f"KV=[{bounds_jsd[1,0]:.3f},{bounds_jsd[1,1]:.3f}], "
      f"KVD=[{bounds_jsd[2,0]:.3f},{bounds_jsd[2,1]:.3f}]")

# ─── Same fixed split as 02 ─────────────────────────────────────────────────
qs = [0.1, 0.5, 0.9]
qW_, qKV_, qKVD_ = np.quantile(xW3, qs), np.quantile(xKV3, qs), np.quantile(xKVD, qs)
grid27 = np.array([[w,kv,kvd] for w in qW_ for kv in qKV_ for kvd in qKVD_])
scale  = X3.std(0) + 1e-10
X3n    = X3 / scale
grid_n = grid27 / scale
cost   = np.zeros((27, N))
for j in range(27): cost[j] = np.sum((X3n - grid_n[j])**2, axis=1)
_, col_ind = linear_sum_assignment(cost)
grid_samples = col_ind.astype(int)
remaining = np.setdiff1d(np.arange(N), grid_samples)
extras = []; sel = list(grid_samples)
for _ in range(23):
    sel_arr = np.array(sel, dtype=int)
    d = np.array([np.min(np.sum((X3n[r] - X3n[sel_arr])**2, axis=1)) for r in remaining])
    far = int(np.argmax(d))
    extras.append(int(remaining[far])); sel.append(int(remaining[far]))
    remaining = np.delete(remaining, far)
extras = np.array(extras, dtype=int)
train_pool = np.concatenate([grid_samples, extras])
test_set   = np.setdiff1d(np.arange(N), train_pool)
print(f"  Fixed split: {len(train_pool)} train pool / {len(test_set)} test")

X_tr, y_tr = X3[train_pool], y_all[train_pool]
X_te, y_te = X3[test_set],   y_all[test_set]

# ─── Audit sets ─────────────────────────────────────────────────────────────
# (1) measured joint samples (candidate-relevant proxy)
D_meas = X3
# (2) Latin-hypercube fill (extrapolation included)
rng = np.random.RandomState(0)
n_lh = 5000
u = (rng.permutation(np.arange(n_lh).reshape(-1,1).repeat(3, axis=1).T).T + rng.uniform(0,1,(n_lh,3))) / n_lh
D_box = bounds_jsd[:,0] + u * (bounds_jsd[:,1] - bounds_jsd[:,0])
# (3) 8 corners (for M10 affine derivative)
D_corn = np.array([[bounds_jsd[0,a], bounds_jsd[1,b], bounds_jsd[2,c]]
                   for a in range(2) for b in range(2) for c in range(2)])

audit_sets = {'D_meas (200 measured)': D_meas, 'D_box (5000 LH)': D_box}

def epsilon_inf(y, yp): return float(np.max(np.abs(y - yp)))
def r2(y, yp):
    ss = max(np.sum((y - y.mean())**2), 1e-30)
    return 1 - np.sum((y-yp)**2)/ss

# ─── Surrogate fits (50-train) ──────────────────────────────────────────────
print("\n" + "="*88)
print("Fitting surrogates on 50-train pool")
print("="*88)

# M0: no intercept (force through origin)
m0 = LinearRegression(fit_intercept=False).fit(X_tr, y_tr)
beta_m0 = m0.coef_
y_m0_te = m0.predict(X_te)
print(f"\nM0 coefficients: β_W={beta_m0[0]:+.4f}, β_KV={beta_m0[1]:+.4f}, β_KVD={beta_m0[2]:+.4f}")
print(f"  R²={r2(y_te, y_m0_te):.4f},  ε̂_∞={epsilon_inf(y_te, y_m0_te):.4f}")

# M1: with intercept
m1 = LinearRegression(fit_intercept=True).fit(X_tr, y_tr)
beta_m1 = m1.coef_
print(f"\nM1 coefficients: β_0={m1.intercept_:+.4f}, β_W={beta_m1[0]:+.4f}, "
      f"β_KV={beta_m1[1]:+.4f}, β_KVD={beta_m1[2]:+.4f}")
y_m1_te = m1.predict(X_te)
print(f"  R²={r2(y_te, y_m1_te):.4f},  ε̂_∞={epsilon_inf(y_te, y_m1_te):.4f}")

# M10: full quadratic (β0 + Σβ_i z_i + Σq_i z_i² + Σγ_ij z_i z_j)
poly = PolynomialFeatures(degree=2, include_bias=True)
Xtr_q = poly.fit_transform(X_tr)
m10 = LinearRegression(fit_intercept=False).fit(Xtr_q, y_tr)
y_m10_te = m10.predict(poly.transform(X_te))
print(f"\nM10 R²={r2(y_te, y_m10_te):.4f},  ε̂_∞={epsilon_inf(y_te, y_m10_te):.4f}")
# Extract coefficients: features = [1, zW, zKV, zKVD, zW², zW·zKV, zW·zKVD, zKV², zKV·zKVD, zKVD²]
fnames = poly.get_feature_names_out(['zW','zKV','zKVD'])
coef_dict = dict(zip(fnames, m10.coef_))
b0  = coef_dict['1']
bW  = coef_dict['zW'];  bKV = coef_dict['zKV']; bKVD = coef_dict['zKVD']
qW  = coef_dict['zW^2']; qKV = coef_dict['zKV^2']; qKVD = coef_dict['zKVD^2']
gWK = coef_dict['zW zKV']; gWD = coef_dict['zW zKVD']; gKD = coef_dict['zKV zKVD']
print(f"  M10 quadratic coefficients:")
print(f"    intercept  β_0 = {b0:+.4f}")
print(f"    linear     β_W={bW:+.4f}, β_KV={bKV:+.4f}, β_KVD={bKVD:+.4f}")
print(f"    quadratic  q_W={qW:+.4f}, q_KV={qKV:+.4f}, q_KVD={qKVD:+.4f}")
print(f"    interaction γ_WK={gWK:+.4f}, γ_WD={gWD:+.4f}, γ_KD={gKD:+.4f}")

# RBF cubic+linear (JSD input)
lb_jsd = X3.min(0); ub_jsd = X3.max(0)
m_rbfc = PySOTRBF(kernel='cubic', tail='linear', lb=lb_jsd, ub=ub_jsd)
m_rbfc.fit(X_tr, y_tr)
y_rbfc_te = m_rbfc.predict(X_te).ravel()
print(f"\nRBF cubic+linear: R²={r2(y_te, y_rbfc_te):.4f},  ε̂_∞={epsilon_inf(y_te, y_rbfc_te):.4f}")

# RBF tps+linear (JSD input)
m_rbft = PySOTRBF(kernel='tps', tail='linear', lb=lb_jsd, ub=ub_jsd)
m_rbft.fit(X_tr, y_tr)
y_rbft_te = m_rbft.predict(X_te).ravel()
print(f"RBF tps+linear:   R²={r2(y_te, y_rbft_te):.4f},  ε̂_∞={epsilon_inf(y_te, y_rbft_te):.4f}")

# ARD-GP
def fit_ard_gp(X, y, n_dim, n_restarts=20):
    kernel = (C(1.0,(1e-4,1e2)) *
              SKRBF(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-4,1e4)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9,1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y); return gp
gp = fit_ard_gp(X_tr, y_tr, 3)
y_gp_te = gp.predict(X_te)
print(f"ARD-GP:           R²={r2(y_te, y_gp_te):.4f},  ε̂_∞={epsilon_inf(y_te, y_gp_te):.4f}")

# ─── Gradient computation ───────────────────────────────────────────────────
def grad_finite_diff(predict_fn, X, dim, h=None):
    if h is None: h = (bounds_jsd[dim,1] - bounds_jsd[dim,0]) * 1e-3
    Xp = X.copy(); Xp[:, dim] += h
    Xm = X.copy(); Xm[:, dim] -= h
    return (predict_fn(Xp) - predict_fn(Xm)) / (2*h)

def m10_predict(X):  return m10.predict(poly.transform(X))
def rbfc_predict(X): return m_rbfc.predict(X).ravel()
def rbft_predict(X): return m_rbft.predict(X).ravel()
def gp_predict(X):   return gp.predict(X)

# ─── Audit per surrogate ────────────────────────────────────────────────────
print("\n" + "="*88)
print("Coordinate-wise monotonicity audit (Theorem 3 condition C2)")
print("="*88)

results = {}
def audit(name, predict_fn, audit_set_dict, is_linear=False, beta_vec=None):
    """For linear surrogates use beta_vec directly; otherwise use finite difference."""
    res = {}
    for set_name, Xs in audit_set_dict.items():
        ax = {}
        for d, axn in enumerate(['W','KV','KVD']):
            if is_linear:
                # Constant gradient = beta_d
                grad = np.full(len(Xs), beta_vec[d])
            else:
                grad = grad_finite_diff(predict_fn, Xs, d)
            frac_pos = float((grad >= -1e-8).mean())
            ax[axn] = dict(min=float(grad.min()), max=float(grad.max()),
                           mean=float(grad.mean()), frac_pos=frac_pos)
        res[set_name] = ax
    results[name] = res

# M0/M1: linear → constant gradient
audit('M0',  None, audit_sets, is_linear=True, beta_vec=beta_m0)
audit('M1',  None, audit_sets, is_linear=True, beta_vec=beta_m1)
# M10: also do corner check
audit('M10', m10_predict, audit_sets)
# Add corner audit for M10
m10_corn = {}
for d, axn in enumerate(['W','KV','KVD']):
    g_corn = grad_finite_diff(m10_predict, D_corn, d)
    m10_corn[axn] = dict(min=float(g_corn.min()), max=float(g_corn.max()),
                         mean=float(g_corn.mean()),
                         frac_pos=float((g_corn >= -1e-8).mean()))
results['M10']['D_corn (8)'] = m10_corn

audit('RBF cubic', rbfc_predict, audit_sets)
audit('RBF tps',   rbft_predict, audit_sets)
audit('ARD-GP',    gp_predict,   audit_sets)

# ─── Print ─────────────────────────────────────────────────────────────────
print(f"\n{'surrogate':12s}  {'audit set':24s}  {'axis':4s}  {'min':>9s}  {'mean':>9s}  "
      f"{'max':>9s}  {'frac ≥ 0':>10s}")
print("─"*100)
for name in ['M0','M1','M10','RBF cubic','RBF tps','ARD-GP']:
    for set_name, ax in results[name].items():
        for axn in ['W','KV','KVD']:
            r_ = ax[axn]
            print(f"{name:12s}  {set_name:24s}  {axn:4s}  {r_['min']:>+9.4f}  "
                  f"{r_['mean']:>+9.4f}  {r_['max']:>+9.4f}  {r_['frac_pos']:>10.4f}")
        print("─"*100)

# ─── Verdicts ──────────────────────────────────────────────────────────────
print("\n" + "="*88)
print("Theorem 3 (C2) verdict per surrogate")
print("="*88)
def verdict(r_, set_name):
    f_W = r_[set_name]['W']['frac_pos']
    f_K = r_[set_name]['KV']['frac_pos']
    f_D = r_[set_name]['KVD']['frac_pos']
    f_min = min(f_W, f_K, f_D)
    if f_min == 1.0:    return f"CONFIRMED ✓  (all 3 axes 100% on {set_name})"
    elif f_min >= 0.99: return f"≥99% ✓ on {set_name}  (min frac = {f_min*100:.1f}%)"
    elif f_min >= 0.90: return f"MOSTLY  on {set_name}  (min frac = {f_min*100:.1f}%)"
    else:               return f"VIOLATED on {set_name} (min frac = {f_min*100:.1f}%)"

for name in ['M0','M1','M10','RBF cubic','RBF tps','ARD-GP']:
    r_ = results[name]
    print(f"\n{name}:")
    if name in ('M0','M1'):
        coefs = beta_m0 if name == 'M0' else beta_m1
        if all(c >= 0 for c in coefs):
            print(f"  All coefficients nonneg → GLOBAL coord-wise monotone ✓")
        else:
            neg = [['β_W','β_KV','β_KVD'][i] for i,c in enumerate(coefs) if c < 0]
            print(f"  Negative coefficients: {neg} → C2 VIOLATED globally ✗")
    elif name == 'M10':
        print(f"  D_corn (8 box corners): {verdict(r_, 'D_corn (8)')}")
        print(f"  D_meas (200 measured) : {verdict(r_, 'D_meas (200 measured)')}")
        print(f"  D_box  (5000 LH fill) : {verdict(r_, 'D_box (5000 LH)')}")
    else:
        print(f"  D_meas (candidate-relevant proxy) : {verdict(r_, 'D_meas (200 measured)')}")
        print(f"  D_box  (extrapolation included)   : {verdict(r_, 'D_box (5000 LH)')}")

# ─── Plot ───────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

fig, axes = plt.subplots(6, 3, figsize=(14, 17))
COL = {'W':'#E64B35', 'KV':'#4DBBD5', 'KVD':'#00A087'}
surrogate_list = ['M0','M1','M10','RBF cubic','RBF tps','ARD-GP']

for r_idx, name in enumerate(surrogate_list):
    r_ = results[name]
    for c_idx, axn in enumerate(['W','KV','KVD']):
        ax = axes[r_idx, c_idx]
        # Plot histogram of gradient on D_meas (candidate-relevant proxy)
        # Recompute to have full vector
        if name in ('M0','M1'):
            beta = beta_m0 if name == 'M0' else beta_m1
            grad = np.full(len(D_meas), beta[c_idx])
        elif name == 'M10':
            grad = grad_finite_diff(m10_predict, D_meas, c_idx)
        elif name == 'RBF cubic':
            grad = grad_finite_diff(rbfc_predict, D_meas, c_idx)
        elif name == 'RBF tps':
            grad = grad_finite_diff(rbft_predict, D_meas, c_idx)
        elif name == 'ARD-GP':
            grad = grad_finite_diff(gp_predict, D_meas, c_idx)

        if grad.std() < 1e-12:  # constant gradient (M0/M1)
            ax.axvline(grad[0], color=COL[axn], lw=3)
            ax.set_xlim(min(grad[0]*1.5, -0.1), max(grad[0]*1.5, 1.0))
        else:
            ax.hist(grad, bins=50, color=COL[axn], alpha=0.85, edgecolor='#333')
        ax.axvline(0, color='black', lw=1.2, ls='-')
        fp = (grad >= -1e-8).mean()
        ax.set_xlabel(f'∂G/∂z_{axn}  (D_meas)')
        if c_idx == 0:
            ax.set_ylabel(f'{name}\n# points')
        ax.set_title(f'{axn}: {fp*100:.1f}% ≥ 0  '
                     f'[min={grad.min():+.3f}, max={grad.max():+.3f}]',
                     fontsize=9, color=COL[axn])
        ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle('Theorem 3 (C2) audit — coordinate-wise monotonicity ∂G/∂z_i on D_meas '
             '(candidate-relevant proxy)', fontweight='bold', y=1.001)
plt.tight_layout()
out = f'{FIG_DIR}/verify_monotonicity_all.png'
plt.savefig(out, dpi=160, bbox_inches='tight'); plt.close()
print(f"\nFigure: {out}")
print("Done.")
