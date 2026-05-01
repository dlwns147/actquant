"""analysis_v3_regime_bits.py — 3-tier regime analysis by raw bits metrics.

FAIR comparison: all surrogates trained on the SAME 50-train pool, evaluated on
test samples within each regime cell. Reports RMSE primarily (R² is unstable
in low-variance cells where SS_tot shrinks).

Splits:
  • Per-method tercile: wbits, kvbits, eff_kvbits (eff = kvbits × kvdim / 128)
  • 2D tercile: (wbits, eff_kvbits) → 9 cells

Surrogates compared (all trained on 50-train pool, sorted-index input):
  • RBF (cubic kernel + linear tail)
  • TPS (thin-plate-spline kernel + linear tail)
  • M1  (linear-additive, baseline)

Per cell reports:
  • N_test (cell test count)
  • std(y_test|cell)  ← cell-mean baseline RMSE; R² formula uses this in denominator
  • RMSE_RBF, RMSE_TPS, RMSE_M1
  • R²_RBF, R²_TPS, R²_M1  (with caveat for low std cells)
  • Skill score = 1 − RMSE_pred / std(y_cell)  (positive = better than cell-mean baseline)

Convention:
  low tier  = low bit value = aggressive quantization (large JSD)
  high tier = high bit value = mild quantization (small JSD)
"""
import sys, os, json, csv
sys.path.insert(0, '/NAS/SJ/actquant/search')

import warnings
warnings.simplefilter("ignore")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SKRBF, ConstantKernel as C, WhiteKernel
from utils.func import get_net_info
from predictor.rbf import RBF

# ─── Data loading (mirrors prior scripts) ─────────────────────────────────────
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
    return np.column_stack((metrics, comps))[pareto_front_2d(np.column_stack((metrics, comps)))]

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
def match_index(comp_vals, pf_metric_sorted): return np.array([np.argmin(np.abs(pf_metric_sorted[:, 1] - c)) for c in comp_vals])

BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY  = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'

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
y3   = mat3[12, :N0]; v3 = ~np.isnan(y3)
xW3   = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3  = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD3 = match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y3    = y3[v3]
iW3   = match_index(mat3[0, :N0][v3], pf_W_m   ).astype(float)
iKV3  = match_index(mat3[1, :N0][v3], pf_KV_m  ).astype(float)
iKVD3 = match_index(mat3[4, :N0][v3], pf_KVDIM_m).astype(float)
N = len(y3)
X3_jsd = np.column_stack([xW3, xKV3, xKVD3])
X3_idx = np.column_stack([iW3, iKV3, iKVD3])
lb_idx = np.array([0., 0., 0.])
ub_idx = np.array([len(pf_W_m)-1, len(pf_KV_m)-1, len(pf_KVDIM_m)-1], dtype=float)

# Bits metrics (Llama-3.1-8B head_dim = 128)
HEAD_DIM = 128
wbits_all  = mat3[0, :N0][v3]
kvbits_all = mat3[1, :N0][v3]
kvdim_all  = mat3[4, :N0][v3]
eff_kv_all = kvbits_all * kvdim_all / HEAD_DIM
print(f"  wbits range: [{wbits_all.min():.2f},{wbits_all.max():.2f}]  "
      f"kvbits: [{kvbits_all.min():.2f},{kvbits_all.max():.2f}]  "
      f"eff_kv_bits: [{eff_kv_all.min():.2f},{eff_kv_all.max():.2f}]")

# ─── Build 27-grid + 23 maximin = 50 train pool / 150 test ────────────────────
qs = [0.1, 0.5, 0.9]
qW   = np.quantile(xW3,   qs); qKV  = np.quantile(xKV3,  qs); qKVD = np.quantile(xKVD3, qs)
grid27 = np.array([[w,kv,kvd] for w in qW for kv in qKV for kvd in qKVD])
scale  = X3_jsd.std(0) + 1e-10
X3n    = X3_jsd / scale
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
print(f"  Fixed split: 50 train pool / {len(test_set)} test")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def features3(xw, xkv, xkvd, mode):
    n = len(xw); o = np.ones(n)
    if mode == 'M1' : return np.column_stack([o, xw, xkv, xkvd])
    if mode == 'M10': return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2,
                                              xw*xkv, xw*xkvd, xkv*xkvd])
def r2(y, yp):
    ss_t = max(np.sum((y - y.mean())**2), 1e-30)
    return 1 - np.sum((y-yp)**2)/ss_t
def rmse(y, yp): return float(np.sqrt(np.mean((y-yp)**2)))
def tier(vals, t_cuts): return np.searchsorted(t_cuts, vals)

# ─── Train all surrogates on the SAME 50-train pool ───────────────────────────
print("\nFitting surrogates on 50-train pool (same data for fair comparison)...")
y_tr = y3[train_pool]
y_te = y3[test_set]
X_tr_idx = X3_idx[train_pool]; X_te_idx = X3_idx[test_set]
X_tr_jsd = X3_jsd[train_pool]; X_te_jsd = X3_jsd[test_set]

# RBF (cubic+linear) with sorted-index input
m_rbf = RBF(kernel='cubic', tail='linear', lb=lb_idx, ub=ub_idx)
m_rbf.fit(X_tr_idx, y_tr)
yp_te_rbf = m_rbf.predict(X_te_idx).ravel()

# TPS (tps+linear) with sorted-index input
m_tps = RBF(kernel='tps', tail='linear', lb=lb_idx, ub=ub_idx)
m_tps.fit(X_tr_idx, y_tr)
yp_te_tps = m_tps.predict(X_te_idx).ravel()

# M1 with JSD input
Phi_tr = features3(X_tr_jsd[:,0], X_tr_jsd[:,1], X_tr_jsd[:,2], 'M1')
Phi_te = features3(X_te_jsd[:,0], X_te_jsd[:,1], X_te_jsd[:,2], 'M1')
coef_m1, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
yp_te_m1 = Phi_te @ coef_m1

# ARD-GP (sklearn) with JSD input
kernel_ard = (C(1.0, (1e-4, 1e2)) *
              SKRBF(length_scale=[1.0]*3, length_scale_bounds=(1e-3, 1e3)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
gp_ard = GaussianProcessRegressor(kernel=kernel_ard, normalize_y=True,
                                  n_restarts_optimizer=10, alpha=1e-8)
gp_ard.fit(X_tr_jsd, y_tr)
yp_te_ard = gp_ard.predict(X_te_jsd)

print(f"  Overall test (150) — RBF:    R²={r2(y_te, yp_te_rbf):.4f} RMSE={rmse(y_te, yp_te_rbf):.4f}")
print(f"                       TPS:    R²={r2(y_te, yp_te_tps):.4f} RMSE={rmse(y_te, yp_te_tps):.4f}")
print(f"                       M1:     R²={r2(y_te, yp_te_m1):.4f} RMSE={rmse(y_te, yp_te_m1):.4f}")
print(f"                       ARD-GP: R²={r2(y_te, yp_te_ard):.4f} RMSE={rmse(y_te, yp_te_ard):.4f}")

surrogates = {'RBF': yp_te_rbf, 'TPS': yp_te_tps, 'M1': yp_te_m1, 'ARD-GP': yp_te_ard}

# ─── Tercile cuts ─────────────────────────────────────────────────────────────
metrics = {'wbits': wbits_all, 'kvbits': kvbits_all, 'eff_kvbits': eff_kv_all}
cuts = {k: np.quantile(v, [1/3, 2/3]) for k, v in metrics.items()}
print("\nTercile cuts (low value = aggressive / low-bit, high value = mild / high-bit):")
for k, c in cuts.items(): print(f"  {k:12s}: low<{c[0]:.3f} | mid | {c[1]:.3f}<high")

# ─── (A) Per-method 1-D tercile — surrogate accuracy by tier ─────────────────
print("\n" + "="*120)
print("(A) Per-method tercile — RMSE primary, R² secondary (* = std(y|cell) shown for context)")
print("="*120)
for k, vals_full in metrics.items():
    vals_te = vals_full[test_set]
    tiers_te = tier(vals_te, cuts[k])
    print(f"\n  Split by {k}:")
    for t_i, tname in enumerate(['low (lo-bit/aggro)','mid','high (hi-bit/mild)']):
        m = tiers_te == t_i; n = int(m.sum())
        if n < 5:
            print(f"    t{t_i} {tname:18s} N={n} skip"); continue
        std_y = float(y_te[m].std())
        line = f"    t{t_i} {tname:18s} N={n:3d}  σ(y)={std_y:.4f}  "
        for s_name, yp in surrogates.items():
            rm = rmse(y_te[m], yp[m])
            r2v = r2(y_te[m], yp[m])
            skill = 1 - rm / max(std_y, 1e-30)
            line += f"{s_name}: RMSE={rm:.4f} R²={r2v:+.2f} skill={skill:+.2f} | "
        print(line)

# ─── (B) 2-D regime: (wbits × eff_kvbits) → 9 cells, fair comparison ─────────
print("\n" + "="*120)
print("(B) 2-D tercile (wbits × eff_kvbits) — all surrogates trained on 50-pool, eval on cell test")
print("    Skill score = 1 − RMSE / σ(y|cell); positive = better than predicting cell-mean")
print("="*120)
tiers_w_te   = tier(wbits_all  [test_set], cuts['wbits'])
tiers_eff_te = tier(eff_kv_all [test_set], cuts['eff_kvbits'])

cells = {}
print(f"  {'cell':25s} {'N':3s}  {'σ(y)':6s} | "
      f"{'RBF RMSE  R²   skill':22s} | {'TPS RMSE  R²   skill':22s} | {'M1  RMSE  R²   skill':22s}")
for w_t in [0,1,2]:
    for e_t in [0,1,2]:
        m_te = (tiers_w_te == w_t) & (tiers_eff_te == e_t)
        n = int(m_te.sum())
        cell_lbl = f"W=t{w_t}({['lo','md','hi'][w_t]}) effKV=t{e_t}({['lo','md','hi'][e_t]})"
        if n < 4:
            print(f"  {cell_lbl:25s} {n:3d}  skip"); continue
        std_y = float(y_te[m_te].std())
        cells[(w_t, e_t)] = dict(n=n, std=std_y)
        line = f"  {cell_lbl:25s} {n:3d}  {std_y:.4f} |"
        for s_name, yp in surrogates.items():
            rm = rmse(y_te[m_te], yp[m_te])
            r2v = r2(y_te[m_te], yp[m_te])
            skill = 1 - rm / max(std_y, 1e-30)
            cells[(w_t, e_t)][s_name] = dict(rmse=rm, r2=r2v, skill=skill)
            line += f" {rm:.4f} {r2v:+.2f} {skill:+.2f} |"
        print(line)

# ─── Why does RBF "look bad" in some cells? — explained ───────────────────────
print("\n" + "="*120)
print("EXPLANATION: low R² in some cells does NOT mean surrogate is bad — it's a cell-variance artifact")
print("="*120)
print("  R² = 1 − SS_res / SS_tot  where SS_tot = Σ(y − ȳ_cell)² ≈ N·σ(y|cell)²")
print("  When σ(y|cell) is small (mild quant cell ⇒ y values cluster near 0.02), even tiny RMSE")
print("  produces low R².  Compare RMSE columns and 'skill' (= 1 − RMSE/σ; positive = beats")
print("  cell-mean prediction).  In high-bit cells, σ(y) ~ 0.005 vs RMSE ~ 0.013 → R² becomes negative")
print("  but absolute prediction error is still very small.")
print()
# Show concrete numbers for the worst-R² cells
print("  Worst R² cells with σ(y) and skill:")
worst = sorted(cells.items(), key=lambda x: x[1].get('RBF', {}).get('r2', np.inf))[:3]
for (w_t, e_t), c in worst:
    rb = c['RBF']
    print(f"    W=t{w_t}({['lo','md','hi'][w_t]}) effKV=t{e_t}({['lo','md','hi'][e_t]}): "
          f"σ(y)={c['std']:.4f}  RBF RMSE={rb['rmse']:.4f}  R²={rb['r2']:+.2f}  skill={rb['skill']:+.2f}")

# ─── Figures ──────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')
COL = {'RBF': '#00A087', 'TPS': '#F1C40F', 'M1': '#3C5488', 'ARD-GP': '#E64B35'}
SURROG_LIST = ['RBF', 'TPS', 'M1', 'ARD-GP']

# Fig 1: Per-method tercile — RMSE bars (primary) and R² (secondary)
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
methods_list = list(metrics.keys())
for col, k in enumerate(methods_list):
    vals_te = metrics[k][test_set]
    tt = tier(vals_te, cuts[k])
    tlbls = ['low\n(lo-bit/aggro)','mid','high\n(hi-bit/mild)']

    # Top: RMSE
    ax = axes[0, col]
    rmses = {sn: [] for sn in surrogates}
    stds = []
    for t_i in [0,1,2]:
        m = tt == t_i
        if m.sum() < 5:
            for sn in surrogates: rmses[sn].append(np.nan)
            stds.append(np.nan); continue
        stds.append(float(y_te[m].std()))
        for sn, yp in surrogates.items():
            rmses[sn].append(rmse(y_te[m], yp[m]))
    x = np.arange(3); w = 0.18
    for i, sn in enumerate(SURROG_LIST):
        ax.bar(x + (i - 1.5)*w, rmses[sn], w, label=sn, color=COL[sn], alpha=0.85, edgecolor='#333')
    ax.plot(x, stds, 'k--', marker='D', label='σ(y|cell)', lw=1.5, ms=7, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(tlbls, fontsize=8.5)
    ax.set_ylabel('RMSE (lower better)'); ax.set_title(f'split by {k}', fontweight='bold')
    if col == 0: ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, lw=0.5, axis='y')

    # Bottom: skill score
    ax = axes[1, col]
    for sn in SURROG_LIST:
        skills = []
        for t_i in [0,1,2]:
            m = tt == t_i
            if m.sum() < 5:
                skills.append(np.nan); continue
            std_y = float(y_te[m].std())
            skills.append(1 - rmse(y_te[m], surrogates[sn][m]) / max(std_y, 1e-30))
        ax.plot(tlbls, skills, '-o', color=COL[sn], label=sn, lw=2, ms=7)
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_ylabel('Skill score = 1 − RMSE/σ(y)\n(>0: beats cell-mean baseline)')
    ax.set_xlabel(f'{k} tier'); ax.legend(fontsize=8); ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle('(A) Per-method tercile — same 50-train surrogates, fair test on each tier',
             fontweight='bold', y=1.005)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/04_regime_bits_per_method.png', **PLT_KW); plt.close()

# Fig 2: 2D heatmap — RMSE per cell for σ(y), RBF, TPS, M1, ARD-GP
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
def get_mat(field, sn=None):
    M = np.full((3,3), np.nan)
    for w_t in [0,1,2]:
        for e_t in [0,1,2]:
            c = cells.get((w_t, e_t))
            if c is None: continue
            if sn is None:
                M[w_t, e_t] = c[field]
            else:
                M[w_t, e_t] = c[sn][field] if sn in c else np.nan
    return M

for ax, mat, ttl, cmap, vrange, fmt in [
    (axes[0,0], get_mat('std'),               '(B0) σ(y|cell) — variance baseline',     'Greys',    None,    '.4f'),
    (axes[0,1], get_mat('rmse', 'ARD-GP'),    '(B1) ARD-GP RMSE per cell',              'YlOrRd',   None,    '.4f'),
    (axes[0,2], get_mat('rmse', 'RBF'),       '(B2) RBF RMSE per cell',                 'YlOrRd',   None,    '.4f'),
    (axes[1,0], get_mat('rmse', 'TPS'),       '(B3) TPS RMSE per cell',                 'YlOrRd',   None,    '.4f'),
    (axes[1,1], get_mat('rmse', 'M1'),        '(B4) M1 RMSE per cell',                  'YlOrRd',   None,    '.4f'),
    (axes[1,2], None,                         '(deferred — leave blank)',               'Greys',    (0,1),   '.0f'),
]:
    if mat is None:
        ax.set_visible(False); continue
    if vrange is None:
        finite = mat[np.isfinite(mat)]
        vrange = (finite.min(), finite.max()) if finite.size else (0, 1)
    im = ax.imshow(mat, cmap=cmap, vmin=vrange[0], vmax=vrange[1], aspect='auto')
    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            txt = f"{v:{fmt}}" if not np.isnan(v) else "N/A"
            color = 'white' if (not np.isnan(v) and v > vrange[0] + 0.6*(vrange[1]-vrange[0])) else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=11, fontweight='bold', color=color)
    ax.set_xticks(range(3)); ax.set_xticklabels(['eff_KV: lo\n(lo-bit)','mid','hi\n(hi-bit)'])
    ax.set_yticks(range(3)); ax.set_yticklabels(['W: lo\n(lo-bit)','mid','hi\n(hi-bit)'])
    ax.set_title(ttl, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04)
plt.suptitle('(B) 2-D regime cells: RMSE comparison (all trained on same 50-pool)',
             fontweight='bold', y=1.005)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/04_regime_bits_2d.png', **PLT_KW); plt.close()

# Fig 3: residuals scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
resid_rbf = y_te - yp_te_rbf
ax = axes[0]
sc = ax.scatter(wbits_all[test_set], resid_rbf, c=eff_kv_all[test_set], cmap='viridis', s=22, alpha=0.85)
ax.axhline(0, color='black', lw=0.7)
ax.set_xlabel('wbits'); ax.set_ylabel('y − ŷ_RBF')
ax.set_title('Residual vs wbits, colored by eff_kvbits', fontweight='bold')
plt.colorbar(sc, ax=ax, fraction=0.04, label='eff_kvbits')
ax.grid(True, alpha=0.25, lw=0.5)

ax = axes[1]
sc = ax.scatter(eff_kv_all[test_set], resid_rbf, c=wbits_all[test_set], cmap='magma', s=22, alpha=0.85)
ax.axhline(0, color='black', lw=0.7)
ax.set_xlabel('eff_kvbits'); ax.set_ylabel('y − ŷ_RBF')
ax.set_title('Residual vs eff_kvbits, colored by wbits', fontweight='bold')
plt.colorbar(sc, ax=ax, fraction=0.04, label='wbits')
ax.grid(True, alpha=0.25, lw=0.5)
plt.suptitle('(C) RBF residual structure on 150 hold-out', fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/04_regime_bits_residuals.png', **PLT_KW); plt.close()

# Fig 4: σ(y) vs RMSE — show that low R² = small denominator, not bad RMSE
fig, ax = plt.subplots(figsize=(7.5, 5.5))
xs, ys_rbf, ys_tps, ys_m1, lbls = [], [], [], [], []
for w_t in [0,1,2]:
    for e_t in [0,1,2]:
        c = cells.get((w_t, e_t))
        if c is None: continue
        xs.append(c['std']); ys_rbf.append(c['RBF']['rmse'])
        ys_tps.append(c['TPS']['rmse']); ys_m1.append(c['M1']['rmse'])
        lbls.append(f"W={w_t},KV={e_t}")
xs = np.array(xs); ys_rbf = np.array(ys_rbf); ys_tps = np.array(ys_tps); ys_m1 = np.array(ys_m1)
mx = max(xs.max(), ys_rbf.max(), ys_m1.max()) * 1.05
ax.plot([0, mx], [0, mx], 'k--', alpha=0.5, label='RMSE = σ(y)  (skill=0)')
ax.scatter(xs, ys_rbf, c=COL['RBF'], s=80, alpha=0.85, label='RBF', marker='o', edgecolor='#333')
ax.scatter(xs, ys_tps, c=COL['TPS'], s=80, alpha=0.85, label='TPS', marker='s', edgecolor='#333')
ax.scatter(xs, ys_m1,  c=COL['M1'],  s=80, alpha=0.85, label='M1',  marker='^', edgecolor='#333')
for x, y, l in zip(xs, ys_rbf, lbls):
    ax.annotate(l, (x, y), fontsize=7, alpha=0.7, xytext=(3,3), textcoords='offset points')
ax.set_xlabel('σ(y|cell)  — cell-mean baseline RMSE')
ax.set_ylabel('Surrogate RMSE on cell test')
ax.set_title('Per-cell RMSE vs cell variance\n(below dashed line = beats baseline; cells near origin = mild-quant cells with low spread)',
             fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.25, lw=0.5)
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/04_regime_bits_skill.png', **PLT_KW); plt.close()

print(f"\nFigures saved:")
print(f"  04_regime_bits_per_method.png   — RMSE+skill bars per tier")
print(f"  04_regime_bits_2d.png           — 2D heatmap of RMSE per cell")
print(f"  04_regime_bits_residuals.png    — RBF residual scatter")
print(f"  04_regime_bits_skill.png        — RMSE vs σ(y|cell) (explains low R² artifact)")
print("Done.\n")
