"""analysis_v3_lowN.py — Sections 1+2 in realistic low-sample regime.

Re-runs the math-model and RBF analyses with only 27~50 train samples
(matching the practical setting where ~27 actual JSD measurements are taken).
For each train size and 50 random seeds, fits all models and reports held-out
test R²/RMSE on the remaining samples.
"""
import sys, os, json, csv
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.func import get_net_info
from predictor.rbf import RBF

# ─── Data loading (mirrors analysis_v3.py) ────────────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]
    min2 = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < min2:
            nd.append(i); min2 = F_s[i, 1]
    return order[nd]

def load_archive_pareto(stats_path, comp_key, config, group_size):
    with open(stats_path) as f: data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs   = [v[0] for v in archive]
    metrics = np.array([v[1] for v in archive])
    comps   = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
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
def match_index(comp_vals, pf_metric_sorted):
    return np.array([np.argmin(np.abs(pf_metric_sorted[:, 1] - c)) for c in comp_vals])

BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'

with open(f'{BASE}/config/llama.json') as f:
    config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

print("Loading PFs and 3-way data...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
pf_W_m     = pf_W    [np.argsort(pf_W    [:, 0])]
pf_KV_m    = pf_KV   [np.argsort(pf_KV   [:, 0])]
pf_KVDIM_m = pf_KVDIM[np.argsort(pf_KVDIM[:, 0])]

mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y3   = mat3[12, :N0]
v3   = ~np.isnan(y3)
xW3   = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3  = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD3 = match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y3    = y3[v3]
iW3   = match_index(mat3[0, :N0][v3], pf_W_m   ).astype(float)
iKV3  = match_index(mat3[1, :N0][v3], pf_KV_m  ).astype(float)
iKVD3 = match_index(mat3[4, :N0][v3], pf_KVDIM_m).astype(float)
N = len(y3)
print(f"  Total available samples: N={N}")

X_jsd = np.column_stack([xW3, xKV3, xKVD3])
X_idx = np.column_stack([iW3, iKV3, iKVD3])
lb_idx = np.array([0., 0., 0.])
ub_idx = np.array([len(pf_W_m)-1, len(pf_KV_m)-1, len(pf_KVDIM_m)-1], dtype=float)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def features(xw, xkv, xkvd, mode):
    n = len(xw); o = np.ones(n)
    if mode == 'M0' : return np.column_stack([xw, xkv, xkvd])
    if mode == 'M1' : return np.column_stack([o, xw, xkv, xkvd])
    if mode == 'M8' : return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2])
    if mode == 'M9' : return np.column_stack([o, xw, xkv, xkvd, xw*xkv, xw*xkvd, xkv*xkvd])
    if mode == 'M10': return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2,
                                              xw*xkv, xw*xkvd, xkv*xkvd])

def fit_pred_ols(mode, x_tr, y_tr, x_te):
    Phi_tr = features(x_tr[:,0], x_tr[:,1], x_tr[:,2], mode)
    Phi_te = features(x_te[:,0], x_te[:,1], x_te[:,2], mode)
    coef, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
    return Phi_tr @ coef, Phi_te @ coef

def r2(y_t, y_p):
    ss_r = np.sum((y_t - y_p)**2); ss_t = np.sum((y_t - y_t.mean())**2)
    return 1 - ss_r / max(ss_t, 1e-30)

def rmse(y_t, y_p):
    return float(np.sqrt(np.mean((y_t - y_p)**2)))

# ─── Low-N scan ───────────────────────────────────────────────────────────────
N_TRS = [27, 30, 35, 40, 45, 50]
N_SEED = 50
math_models = ['M0', 'M1', 'M8', 'M9', 'M10']
rbf_variants = [('cubic','linear'), ('tps','linear'), ('linear','linear'), ('linear','constant')]
rbf_names = [f'RBF-{k}+{t}' for k,t in rbf_variants]
all_models = math_models + rbf_names

# storage[(model, n_tr)] = list of (r2_train, r2_test, rmse_test)
storage = {}

print(f"\nRunning {N_SEED} seeds × {len(N_TRS)} train sizes...")
for seed in range(N_SEED):
    rng = np.random.RandomState(seed)
    for n_tr in N_TRS:
        idx = rng.permutation(N)
        tr, te = idx[:n_tr], idx[n_tr:]

        for m in math_models:
            try:
                yp_tr, yp_te = fit_pred_ols(m, X_jsd[tr], y3[tr], X_jsd[te])
                storage.setdefault((m, n_tr), []).append(
                    (r2(y3[tr], yp_tr), r2(y3[te], yp_te), rmse(y3[te], yp_te)))
            except Exception:
                pass

        for kernel, tail in rbf_variants:
            name = f'RBF-{kernel}+{tail}'
            try:
                m_rbf = RBF(kernel=kernel, tail=tail, lb=lb_idx, ub=ub_idx)
                m_rbf.fit(X_idx[tr], y3[tr])
                yp_tr = m_rbf.predict(X_idx[tr]).ravel()
                yp_te = m_rbf.predict(X_idx[te]).ravel()
                storage.setdefault((name, n_tr), []).append(
                    (r2(y3[tr], yp_tr), r2(y3[te], yp_te), rmse(y3[te], yp_te)))
            except Exception:
                pass

# ─── Reporting ────────────────────────────────────────────────────────────────
def stat_block(model_list, title):
    print("\n" + "="*100)
    print(title)
    print("="*100)
    hdr = f"  {'Model':24s}"
    for n_tr in N_TRS: hdr += f"   N={n_tr:<3d}      "
    print(hdr)
    for m in model_list:
        line = f"  {m:24s}"
        for n_tr in N_TRS:
            rs = storage.get((m, n_tr), [])
            r2_te = [r[1] for r in rs]
            if r2_te:
                med = np.median(r2_te)
                lo = np.percentile(r2_te, 10); hi = np.percentile(r2_te, 90)
                line += f"  {med:.3f}[{lo:.2f},{hi:.2f}] "
            else:
                line += f"  {'N/A':14s} "
        print(line)
    print("\n  (cell format: median[10%, 90%] over 50 seeds; N_test = N_total - N_train)")

stat_block(math_models, "SECTION 1 (low-N) — Math models, test R²")
stat_block(rbf_names,   "SECTION 2 (low-N) — RBF on sorted-index input, test R²")

# Test RMSE table
def rmse_block(model_list, title):
    print("\n" + "="*100)
    print(title)
    print("="*100)
    hdr = f"  {'Model':24s}"
    for n_tr in N_TRS: hdr += f"   N={n_tr:<3d}      "
    print(hdr)
    for m in model_list:
        line = f"  {m:24s}"
        for n_tr in N_TRS:
            rs = storage.get((m, n_tr), [])
            rm_te = [r[2] for r in rs]
            if rm_te:
                med = np.median(rm_te); lo = np.percentile(rm_te, 10); hi = np.percentile(rm_te, 90)
                line += f"  {med:.4f}({hi:.3f}) "
            else:
                line += f"  {'N/A':14s} "
        print(line)
    print("  (cell format: median RMSE (90%-percentile RMSE))")

rmse_block(math_models, "SECTION 1 (low-N) — Math models, test RMSE")
rmse_block(rbf_names,   "SECTION 2 (low-N) — RBF, test RMSE")

# ─── Figures ──────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')

math_colors = {'M0':'#E64B35', 'M1':'#3C5488', 'M8':'#4DBBD5', 'M9':'#F39B7F', 'M10':'#9B59B6'}
rbf_colors  = {'RBF-cubic+linear':'#E64B35', 'RBF-tps+linear':'#4DBBD5',
               'RBF-linear+linear':'#00A087', 'RBF-linear+constant':'#9B59B6'}

def collect(name):
    xs, med, lo, hi = [], [], [], []
    for n_tr in N_TRS:
        rs = storage.get((name, n_tr), [])
        r2_te = [r[1] for r in rs]
        if r2_te:
            xs.append(n_tr); med.append(np.median(r2_te))
            lo.append(np.percentile(r2_te, 10)); hi.append(np.percentile(r2_te, 90))
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)

# Fig 1: math models + RBF side-by-side (test R² vs N_tr)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
for m in math_models:
    xs, med, lo, hi = collect(m)
    ax.plot(xs, med, '-o', color=math_colors[m], label=m, lw=2, ms=6)
    ax.fill_between(xs, lo, hi, color=math_colors[m], alpha=0.13)
ax.set_xlabel('N_train'); ax.set_ylabel('Test R²')
ax.set_title('Section 1 — Math models', fontweight='bold')
ax.set_xticks(N_TRS); ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.25, lw=0.5)

ax = axes[1]
for r_name in rbf_names:
    xs, med, lo, hi = collect(r_name)
    ax.plot(xs, med, '-o', color=rbf_colors[r_name], label=r_name.replace('RBF-',''), lw=2, ms=6)
    ax.fill_between(xs, lo, hi, color=rbf_colors[r_name], alpha=0.13)
ax.set_xlabel('N_train'); ax.set_ylabel('Test R²')
ax.set_title('Section 2 — RBF (sorted-index)', fontweight='bold')
ax.set_xticks(N_TRS); ax.legend(fontsize=9); ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle('Low-sample regime: held-out test R² (median + [10%, 90%] band over 50 seeds)',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/v3_lowN_sections12.png', **PLT_KW); plt.close()

# Fig 2: best math vs best RBF on one axis
fig, ax = plt.subplots(figsize=(9, 4.8))
for m in math_models:
    xs, med, _, _ = collect(m)
    ax.plot(xs, med, '-o', color=math_colors[m], label=m, lw=2, ms=5)
for r_name in rbf_names:
    xs, med, _, _ = collect(r_name)
    ax.plot(xs, med, '--s', color=rbf_colors[r_name],
            label=r_name.replace('RBF-',''), lw=2, ms=5, alpha=0.85)
ax.set_xlabel('N_train'); ax.set_ylabel('Median test R²')
ax.set_title('All models in low-N regime (math: solid, RBF: dashed)', fontweight='bold')
ax.set_xticks(N_TRS); ax.legend(fontsize=8, ncol=3, loc='lower right')
ax.grid(True, alpha=0.25, lw=0.5)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/v3_lowN_all_models.png', **PLT_KW); plt.close()

# Fig 3: train-test gap (overfitting check) for M10 and RBF-cubic at each N_tr
fig, ax = plt.subplots(figsize=(9, 4.5))
for m, color in [('M0', math_colors['M0']), ('M1', math_colors['M1']),
                 ('M10', math_colors['M10']),
                 ('RBF-cubic+linear', rbf_colors['RBF-cubic+linear']),
                 ('RBF-linear+linear', rbf_colors['RBF-linear+linear'])]:
    xs_tr, ys_tr, xs_te, ys_te = [], [], [], []
    for n_tr in N_TRS:
        rs = storage.get((m, n_tr), [])
        if not rs: continue
        r2_tr = [r[0] for r in rs]; r2_te = [r[1] for r in rs]
        xs_tr.append(n_tr); ys_tr.append(np.median(r2_tr))
        xs_te.append(n_tr); ys_te.append(np.median(r2_te))
    ax.plot(xs_tr, ys_tr, '--', color=color, lw=1.5, alpha=0.6)
    ax.plot(xs_te, ys_te, '-o', color=color, lw=2, ms=6, label=m)
ax.set_xlabel('N_train'); ax.set_ylabel('R² (train: dashed, test: solid)')
ax.set_title('Train vs Test R² — overfitting check (low-N regime)', fontweight='bold')
ax.set_xticks(N_TRS); ax.legend(fontsize=9); ax.grid(True, alpha=0.25, lw=0.5)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/v3_lowN_overfitting.png', **PLT_KW); plt.close()

print(f"\nFigures saved: v3_lowN_sections12.png, v3_lowN_all_models.png, v3_lowN_overfitting.png")
print("Done.\n")
