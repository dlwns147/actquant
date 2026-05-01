"""03_pareto_combination.py — Section 6 of the narrative.

2ε-Pareto Theorem (statement only — direct PF/HV computation is too sparse to verify):

    ‖y_actual − y_add‖_∞ = ε  ⇒  Pareto(y_add) ⊆ 2ε-Pareto(y_actual)

Empirical analysis is restricted to **150 actual hold-out test** measurements:
  • Surrogate comparison (actual vs prediction scatter, R², ε bounds)
  • ε bounds from y_actual − y_pred for each surrogate
  • Theorem statement using the additive surrogate's ε

Surrogates compared (50-train / 150-test):
  • Math: M1 (linear additive), M10 (full quadratic)
  • RBF: cubic+linear, tps+linear (both JSD input)
  • Hierarchical: Hier-WD-TPS-linear (n_3way=18 + n_pair=32 = 50 total)
  • ARD-GP (full, anisotropic kernel)
  • Additive y_add (Hoeffding decomposition from ARD-GP)
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
from scipy.optimize import linear_sum_assignment
from utils.func import get_net_info
from predictor.rbf import RBF as PySOTRBF

# ─── Data loading ─────────────────────────────────────────────────────────────
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
def match_metric(comp_vals, pf): return np.array([pf[np.argmin(np.abs(pf[:, 1] - c)), 0] for c in comp_vals])

BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY  = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
AWQ_W_KVD = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim/results.csv'

with open(f'{BASE}/config/llama.json') as f: config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w':128, 'k':[128,128], 'v':[128,128]}

print("Loading data + WD pair...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y_all = mat3[12, :N0]; v3 = ~np.isnan(y_all)
xW3   = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3  = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD  = match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y_all = y_all[v3]; N = len(y_all)
X3 = np.column_stack([xW3, xKV3, xKVD])
HEAD_DIM = 128
bounds_jsd = [[xW3.min(), xW3.max()], [xKV3.min(), xKV3.max()], [xKVD.min(), xKVD.max()]]

wbits_all  = mat3[0, :N0][v3]
kvbits_all = mat3[1, :N0][v3]
kvdim_all  = mat3[4, :N0][v3]
eff_kv_all = kvbits_all * kvdim_all / HEAD_DIM

mat_wd = load_csv(AWQ_W_KVD); N_wd = mat_wd.shape[1]
y_WD_full = mat_wd[12, :N_wd]; v_wd = ~np.isnan(y_WD_full)
xW_WD   = match_metric(mat_wd[0, :N_wd], pf_W   )[v_wd]
xKVD_WD = match_metric(mat_wd[4, :N_wd], pf_KVDIM)[v_wd]
y_WD    = y_WD_full[v_wd]
print(f"  N={N} 3-way; WD pair N={len(y_WD)}")

# ─── Fixed 50/150 split ──────────────────────────────────────────────────────
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
print(f"  Fixed split: {len(train_pool)} train / {len(test_set)} test")

# ─── Surrogate fits on 50-train ──────────────────────────────────────────────
def fit_ard_gp(X, y, n_restarts=20):
    kernel = (C(1.0,(1e-4,1e2)) *
              SKRBF(length_scale=[1.0]*X.shape[1], length_scale_bounds=(1e-3,1e3)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9,1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y); return gp
def features_M(xw, xkv, xkvd, mode):
    n = len(xw); o = np.ones(n)
    if mode == 'M1' : return np.column_stack([o, xw, xkv, xkvd])
    if mode == 'M10': return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2,
                                              xw*xkv, xw*xkvd, xkv*xkvd])

print("\nFitting all surrogates on 50-train pool...")
X_tr = X3[train_pool]; y_tr = y_all[train_pool]
X_te = X3[test_set];   y_actual = y_all[test_set]
wbits_te = wbits_all[test_set]; eff_kv_te = eff_kv_all[test_set]

# (1) ARD-GP
gp = fit_ard_gp(X_tr, y_tr, n_restarts=20)
y_ard = gp.predict(X_te)

# (2) Hoeffding additive y_add
rng = np.random.RandomState(0); n_mc = 5000
X_mc = np.array([rng.uniform(b[0], b[1], n_mc) for b in bounds_jsd]).T
y_mc = gp.predict(X_mc); f0 = float(y_mc.mean())
def cmean(x_query, dim, h_factor=0.15):
    h = X_mc[:, dim].std() * h_factor
    out = np.zeros(len(x_query))
    for k, xq in enumerate(x_query):
        w = np.exp(-0.5 * ((X_mc[:, dim] - xq) / h)**2)
        out[k] = (w * y_mc).sum() / w.sum()
    return out
y_add = f0 + (cmean(X_te[:,0], 0) - f0) + (cmean(X_te[:,1], 1) - f0) + (cmean(X_te[:,2], 2) - f0)

# (3) M1
Phi_tr1 = features_M(X_tr[:,0], X_tr[:,1], X_tr[:,2], 'M1')
Phi_te1 = features_M(X_te[:,0], X_te[:,1], X_te[:,2], 'M1')
coef_M1, *_ = np.linalg.lstsq(Phi_tr1, y_tr, rcond=None)
y_M1 = Phi_te1 @ coef_M1

# (4) M10
Phi_tr10 = features_M(X_tr[:,0], X_tr[:,1], X_tr[:,2], 'M10')
Phi_te10 = features_M(X_te[:,0], X_te[:,1], X_te[:,2], 'M10')
coef_M10, *_ = np.linalg.lstsq(Phi_tr10, y_tr, rcond=None)
y_M10 = Phi_te10 @ coef_M10

# (5) RBF cubic+linear (JSD)
lb_jsd = X3.min(0); ub_jsd = X3.max(0)
m_rbf_c = PySOTRBF(kernel='cubic', tail='linear', lb=lb_jsd, ub=ub_jsd)
m_rbf_c.fit(X_tr, y_tr); y_rbf_c = m_rbf_c.predict(X_te).ravel()

# (6) RBF tps+linear (JSD)
m_rbf_t = PySOTRBF(kernel='tps', tail='linear', lb=lb_jsd, ub=ub_jsd)
m_rbf_t.fit(X_tr, y_tr); y_rbf_t = m_rbf_t.predict(X_te).ravel()

# (7) Hier-WD-TPS-linear (50-budget: 18 3way + 32 pair)
n_3way_h = 18; n_pair_h = 32
tr3_hier = grid_samples[:n_3way_h]
rng_p = np.random.RandomState(0)
pair_idx = rng_p.choice(len(y_WD), n_pair_h, replace=False)
xa_p = xW_WD[pair_idx]; xb_p = xKVD_WD[pair_idx]; y_p = y_WD[pair_idx]
X2_p = np.column_stack([xa_p, xb_p])
m_pair = PySOTRBF(kernel='tps', tail='linear', lb=X2_p.min(0), ub=X2_p.max(0))
m_pair.fit(X2_p, y_p)
yp_pair_tr = m_pair.predict(np.column_stack([X3[tr3_hier, 0], X3[tr3_hier, 2]])).ravel()
yp_pair_te = m_pair.predict(np.column_stack([X_te[:, 0],     X_te[:, 2]])).ravel()
r_tr_h = y_all[tr3_hier] - yp_pair_tr
c_h = np.polyfit(X3[tr3_hier, 1], r_tr_h, 1)
y_hier = yp_pair_te + np.polyval(c_h, X_te[:, 1])

# ─── ε bounds on 150 actual test ────────────────────────────────────────────
def r2(y, yp):
    ss = max(np.sum((y - y.mean())**2), 1e-30); return 1 - np.sum((y-yp)**2)/ss
def rmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)))

surrogates = [
    ('M1 (linear additive)',       y_M1,   '#3C5488'),
    ('M10 (full quadratic)',       y_M10,  '#9B59B6'),
    ('RBF cubic+linear (JSD)',     y_rbf_c,'#00A087'),
    ('RBF tps+linear (JSD)',       y_rbf_t,'#F1C40F'),
    ('Hier-WD-TPS-linear',         y_hier, '#E64B35'),
    ('ARD-GP (full)',              y_ard,  '#16A085'),
    ('Additive y_add (Hoeffding)', y_add,  '#7F7F7F'),
]

print("\n" + "="*100)
print("ε bounds on 150 actual test  (residual = y_actual − y_pred)")
print("="*100)
print(f"  {'Surrogate':30s}  {'R²':8s}  {'RMSE':8s}  {'ε_∞':8s}  {'ε_2':8s}  {'ε_95':8s}")
print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
results = {}
for name, ypred, _ in surrogates:
    res = y_actual - ypred
    e_inf = float(np.max(np.abs(res)))
    e_2   = float(np.sqrt(np.mean(res**2)))
    e_95  = float(np.percentile(np.abs(res), 95))
    r2v = r2(y_actual, ypred); rm = rmse(y_actual, ypred)
    results[name] = dict(r2=r2v, rmse=rm, e_inf=e_inf, e_2=e_2, e_95=e_95, ypred=ypred, res=res)
    print(f"  {name:30s}  {r2v:.4f}    {rm:.4f}    {e_inf:.4f}    {e_2:.4f}    {e_95:.4f}")

# ─── Theorem statement (using additive y_add ε) ─────────────────────────────
eps_inf_add = results['Additive y_add (Hoeffding)']['e_inf']
eps_2_add   = results['Additive y_add (Hoeffding)']['e_2']
print("\n" + "="*100)
print("THEOREM (statement, no PF/HV computation)")
print("="*100)
print(f"  ‖y_actual − y_add‖_∞ = {eps_inf_add:.4f}  ⇒  "
      f"Pareto(y_add) ⊆ 2ε-Pareto(y_actual)")
print(f"  2ε_∞ corridor = {2*eps_inf_add:.4f}")
print(f"  Tighter L²:  2ε_2 = {2*eps_2_add:.4f}")

# ─── Figures ─────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')

# Fig 1: actual/prediction scatter for all 7 surrogates (2x4 grid, last cell unused)
fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
ax_list = axes.flatten()
color_ax = wbits_te + eff_kv_te
for ax, (name, ypred, col) in zip(ax_list, surrogates):
    res = results[name]
    sc = ax.scatter(y_actual, ypred, c=color_ax, cmap='viridis', s=22, alpha=0.85,
                    edgecolor='black', linewidth=0.3)
    lo = min(y_actual.min(), ypred.min()); hi = max(y_actual.max(), ypred.max())
    ax.plot([lo,hi],[lo,hi],'k--',lw=1.0,label='y_pred = y_actual')
    xx = np.array([lo, hi])
    ax.fill_between(xx, xx-res['e_inf'], xx+res['e_inf'], color='gray', alpha=0.10,
                    label=f'±ε_∞={res["e_inf"]:.3f}')
    ax.fill_between(xx, xx-res['e_2'], xx+res['e_2'], color='gray', alpha=0.22,
                    label=f'±ε_2={res["e_2"]:.3f}')
    ax.set_title(f'{name}\nR²={res["r2"]:.4f}, ε_2={res["e_2"]:.4f}',
                 fontweight='bold', fontsize=9.5, color=col)
    ax.set_xlabel('y_actual'); ax.set_ylabel('y_pred')
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_xlim([lo*0.95, hi*1.05]); ax.set_ylim([lo*0.95, hi*1.05])
ax_list[-1].set_visible(False)
plt.colorbar(sc, ax=ax_list[-1], fraction=0.6, label='wbits + eff_kvbits', location='left')
plt.suptitle(f'Actual vs Prediction on 150 test (color = wbits + eff_kvbits)',
             fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_pareto_combination_pfronts.png', **PLT_KW); plt.close()

# Fig 2: ε bars + R² bars + residual hist (all surrogates, 150 test)
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.3)

ax = fig.add_subplot(gs[0, :2])
names = [s[0] for s in surrogates]
eps_inf_vals = [results[n]['e_inf'] for n in names]
eps_2_vals   = [results[n]['e_2']   for n in names]
colors_lst = [s[2] for s in surrogates]
x_pos = np.arange(len(names))
ax.bar(x_pos - 0.18, eps_inf_vals, 0.36, color=colors_lst, alpha=0.6,
       edgecolor='#333', label='ε_∞')
ax.bar(x_pos + 0.18, eps_2_vals,   0.36, color=colors_lst, alpha=1.0,
       edgecolor='#333', label='ε_2')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.split(' (')[0] for n in names], rotation=30, ha='right', fontsize=8.5)
ax.set_ylabel('ε  (residual norm on 150 test)')
ax.set_title('ε bounds across surrogates (light: ε_∞, dark: ε_2)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25, lw=0.5, axis='y')

ax = fig.add_subplot(gs[0, 2:])
r2_vals = [results[n]['r2'] for n in names]
ax.bar(x_pos, r2_vals, color=colors_lst, alpha=0.85, edgecolor='#333')
for i, v in enumerate(r2_vals):
    ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.split(' (')[0] for n in names], rotation=30, ha='right', fontsize=8.5)
ax.set_ylabel('Test R² on 150 hold-out')
ax.set_title('R² across surrogates', fontweight='bold')
ax.set_ylim([min(r2_vals)*0.99, 1.0])
ax.grid(True, alpha=0.25, lw=0.5, axis='y')

ax = fig.add_subplot(gs[1, :])
for name, _, col in surrogates:
    res = results[name]['res']
    ax.hist(res, bins=25, color=col, alpha=0.40, edgecolor='#333', linewidth=0.3,
            label=f'{name.split(" (")[0]} (ε_∞={results[name]["e_inf"]:.3f})')
ax.axvline(0, color='black', lw=1, ls=':')
ax.set_xlabel('Residual r = y_actual − y_pred'); ax.set_ylabel('# samples')
ax.set_title('Residual distributions (all surrogates, 150 test)', fontweight='bold')
ax.legend(fontsize=8, ncol=4, loc='upper right')
ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle(f'ε analysis across 7 surrogate families on 150 test  (50-measurement budget)',
             fontweight='bold', y=1.005)
plt.savefig(f'{FIG_DIR}/03_pareto_combination_HV.png', **PLT_KW); plt.close()

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n" + "="*100)
print("SUMMARY (only 150 actual test ε; no PF/HV; no oracle/candidate analysis)")
print("="*100)
best_r2  = max(results.items(), key=lambda x: x[1]['r2'])
best_eps = min(results.items(), key=lambda x: x[1]['e_inf'])
print(f"\n  Best R²:  {best_r2[0]:30s}  R²  = {best_r2[1]['r2']:.4f}")
print(f"  Best ε_∞: {best_eps[0]:30s}  ε_∞ = {best_eps[1]['e_inf']:.4f}")
print(f"\n  Theorem applied with additive y_add residual on 150 test:")
print(f"    Pareto(y_add) ⊆ 2ε-Pareto(y_actual),  2ε_∞ = {2*eps_inf_add:.4f},  2ε_2 = {2*eps_2_add:.4f}")

print(f"\nFigures saved:")
print(f"  03_pareto_combination_pfronts.png  — actual vs prediction scatter (7 surrogates)")
print(f"  03_pareto_combination_HV.png       — ε bars + R² bars + residual hists")
print("Done.\n")
