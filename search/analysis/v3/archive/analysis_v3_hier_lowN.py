"""analysis_v3_hier_lowN.py — Direct-3way vs Hierarchical 2-then-1 in low-N regime.

Uses the same fixed-split design as analysis_v3_lowN_fixed.py:
  - 27 base train = closest distinct samples to [0.1,0.5,0.9]³ quantile grid (Hungarian)
  - 23 extras    = farthest-point (maximin) samples
  - Train pool   = 50, Test = 150 (deterministic)

For each N_tr ∈ {27, 30, 35, 40, 45, 50}, compare:
  • Direct  : fit M0/M1/M8/M9/M10/RBF directly on N_tr 3-way samples
  • Hierarchical : pre-fit pair model on full 200 2-way (W+KV / W+KVD / KV+KVD),
                   then fit 1D residual (linear or RBF) on N_tr 3-way samples,
                   predict y = pair_pred + residual_pred on 150 hold-out
"""
import sys, os, json, csv
sys.path.insert(0, '/NAS/SJ/actquant/search')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from utils.func import get_net_info
from predictor.rbf import RBF

# ─── Data loading ─────────────────────────────────────────────────────────────
def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]
    min2 = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < min2: nd.append(i); min2 = F_s[i, 1]
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

def match_metric(comp_vals, pf): return np.array([pf[np.argmin(np.abs(pf[:, 1] - c)), 0] for c in comp_vals])
def match_index(comp_vals, pf_metric_sorted): return np.array([np.argmin(np.abs(pf_metric_sorted[:, 1] - c)) for c in comp_vals])

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

print("Loading PFs and all datasets...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
pf_W_m     = pf_W    [np.argsort(pf_W    [:, 0])]
pf_KV_m    = pf_KV   [np.argsort(pf_KV   [:, 0])]
pf_KVDIM_m = pf_KVDIM[np.argsort(pf_KVDIM[:, 0])]
JSD_W_def, JSD_KV_def, JSD_KVD_def = pf_W_m[0,0], pf_KV_m[0,0], pf_KVDIM_m[0,0]

def load_3way(path):
    mat = load_csv(path); N0 = mat.shape[1]
    y = mat[12, :N0]; v = ~np.isnan(y)
    return (match_metric(mat[0,:N0], pf_W   )[v],
            match_metric(mat[1,:N0], pf_KV  )[v],
            match_metric(mat[4,:N0], pf_KVDIM)[v],
            y[v])

def load_2way(path, has_W, has_KV, has_KVD):
    mat = load_csv(path); N0 = mat.shape[1]
    y = mat[12, :N0]; valid = ~np.isnan(y)
    xW   = match_metric(mat[0, :N0], pf_W   ) if has_W   else np.full(N0, JSD_W_def)
    xKV  = match_metric(mat[1, :N0], pf_KV  ) if has_KV  else np.full(N0, JSD_KV_def)
    xKVD = match_metric(mat[4, :N0], pf_KVDIM) if has_KVD else np.full(N0, JSD_KVD_def)
    return xW[valid], xKV[valid], xKVD[valid], y[valid]

xW3, xKV3, xKVD3, y3 = load_3way(AWQ_3WAY); N = len(y3)
mat3 = load_csv(AWQ_3WAY)
v3mask = ~np.isnan(mat3[12, :mat3.shape[1]])
iW3   = match_index(mat3[0, :mat3.shape[1]][v3mask], pf_W_m   ).astype(float)
iKV3  = match_index(mat3[1, :mat3.shape[1]][v3mask], pf_KV_m  ).astype(float)
iKVD3 = match_index(mat3[4, :mat3.shape[1]][v3mask], pf_KVDIM_m).astype(float)
X3_jsd = np.column_stack([xW3, xKV3, xKVD3])
X3_idx = np.column_stack([iW3, iKV3, iKVD3])
lb_idx = np.array([0., 0., 0.])
ub_idx = np.array([len(pf_W_m)-1, len(pf_KV_m)-1, len(pf_KVDIM_m)-1], dtype=float)

xW_WK, xKV_WK, xKVD_WK, y_WK = load_2way(AWQ_W_KV,  True,  True,  False)
xW_WD, xKV_WD, xKVD_WD, y_WD = load_2way(AWQ_W_KVD, True,  False, True )
xW_KD, xKV_KD, xKVD_KD, y_KD = load_2way(AWQ_KV_KVD,False, True,  True )
print(f"  3-way N={N}; 2-way: WK={len(y_WK)}, WD={len(y_WD)}, KD={len(y_KD)}")

# ─── 27-grid + maximin extras (3-way only; 150 test fixed) ────────────────────
qs = [0.1, 0.5, 0.9]
qW   = np.quantile(xW3,   qs); qKV  = np.quantile(xKV3,  qs); qKVD = np.quantile(xKVD3, qs)
grid27 = np.array([[w,kv,kvd] for w in qW for kv in qKV for kvd in qKVD])
scale  = X3_jsd.std(0) + 1e-10
X3n    = X3_jsd / scale
grid_n = grid27 / scale
cost   = np.zeros((27, N))
for j in range(27):
    cost[j] = np.sum((X3n - grid_n[j])**2, axis=1)
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
assert len(test_set) == 150
print(f"  3-way fixed split: 50 train pool / 150 test")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def features3(xw, xkv, xkvd, mode):
    n = len(xw); o = np.ones(n)
    if mode == 'M0' : return np.column_stack([xw, xkv, xkvd])
    if mode == 'M1' : return np.column_stack([o, xw, xkv, xkvd])
    if mode == 'M8' : return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2])
    if mode == 'M9' : return np.column_stack([o, xw, xkv, xkvd, xw*xkv, xw*xkvd, xkv*xkvd])
    if mode == 'M10': return np.column_stack([o, xw, xkv, xkvd, xw**2, xkv**2, xkvd**2,
                                              xw*xkv, xw*xkvd, xkv*xkvd])

def features2(a, b, mode):
    n = len(a); o = np.ones(n)
    if mode == 'M1' : return np.column_stack([o, a, b])
    if mode == 'M10': return np.column_stack([o, a, b, a**2, b**2, a*b])

def fit_pred3(mode, x_tr, y_tr, x_te):
    Phi_tr = features3(x_tr[:,0], x_tr[:,1], x_tr[:,2], mode)
    Phi_te = features3(x_te[:,0], x_te[:,1], x_te[:,2], mode)
    coef, *_ = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)
    return Phi_tr @ coef, Phi_te @ coef

def r2(y_t, y_p):
    ss_r = np.sum((y_t - y_p)**2); ss_t = np.sum((y_t - y_t.mean())**2)
    return 1 - ss_r / max(ss_t, 1e-30)
def rmse(y_t, y_p): return float(np.sqrt(np.mean((y_t - y_p)**2)))

# ─── Pre-fit pair models on full 200 pair data ────────────────────────────────
print("\nPre-fitting 2-way pair models on full N=200 pair datasets...")
pair_data = {
    'WK': dict(xa=xW_WK, xb=xKV_WK, y=y_WK, vars=('W','KV')),
    'WD': dict(xa=xW_WD, xb=xKVD_WD, y=y_WD, vars=('W','KVD')),
    'KD': dict(xa=xKV_KD, xb=xKVD_KD, y=y_KD, vars=('KV','KVD')),
}
pair_models = {}
for key, d in pair_data.items():
    pm = {'vars': d['vars']}
    # OLS pair models
    for m in ['M1','M10']:
        Phi = features2(d['xa'], d['xb'], m)
        coef, *_ = np.linalg.lstsq(Phi, d['y'], rcond=None)
        yp = Phi @ coef
        pm[m] = dict(coef=coef, r2=r2(d['y'], yp))
    # RBF pair model
    X2 = np.column_stack([d['xa'], d['xb']])
    lb2, ub2 = X2.min(0), X2.max(0)
    rbf_pm = RBF(kernel='cubic', tail='linear', lb=lb2, ub=ub2)
    rbf_pm.fit(X2, d['y'])
    yp_r = rbf_pm.predict(X2).ravel()
    pm['RBF'] = dict(model=rbf_pm, r2=r2(d['y'], yp_r))
    pair_models[key] = pm
    a, b = d['vars']
    print(f"  Pair {key}({a}+{b}) train R²: M1={pm['M1']['r2']:.4f}  M10={pm['M10']['r2']:.4f}  RBF={pm['RBF']['r2']:.4f}")

def pair_predict(pair_key, pair_model, xa, xb):
    pm = pair_models[pair_key][pair_model]
    if pair_model == 'RBF':
        return pm['model'].predict(np.column_stack([xa, xb])).ravel()
    Phi = features2(xa, xb, pair_model)
    return Phi @ pm['coef']

# ─── Helper: fit residual model and combine ───────────────────────────────────
def hierarchical_predict(pair_key, pair_model, resid_kind, x3_tr, y3_tr, x3_te):
    """x3_tr/te are (N, 3) arrays with cols (W, KV, KVD).
    Returns y_pred_test."""
    fmap = {'W': 0, 'KV': 1, 'KVD': 2}
    a_idx, b_idx = fmap[pair_models[pair_key]['vars'][0]], fmap[pair_models[pair_key]['vars'][1]]
    third_idx = [i for i in range(3) if i not in (a_idx, b_idx)][0]

    yp_pair_tr = pair_predict(pair_key, pair_model, x3_tr[:, a_idx], x3_tr[:, b_idx])
    yp_pair_te = pair_predict(pair_key, pair_model, x3_te[:, a_idx], x3_te[:, b_idx])
    r_tr = y3_tr - yp_pair_tr
    t_tr, t_te = x3_tr[:, third_idx], x3_te[:, third_idx]

    if resid_kind == 'linear':
        c = np.polyfit(t_tr, r_tr, 1)
        r_te = np.polyval(c, t_te)
    elif resid_kind == 'rbf':
        # 1D RBF
        try:
            mr = RBF(kernel='cubic', tail='linear',
                     lb=np.array([t_tr.min()]), ub=np.array([t_tr.max()]))
            mr.fit(t_tr.reshape(-1,1), r_tr)
            r_te = mr.predict(t_te.reshape(-1,1)).ravel()
        except Exception:
            c = np.polyfit(t_tr, r_tr, 1)
            r_te = np.polyval(c, t_te)
    elif resid_kind == 'none':
        r_te = np.zeros_like(t_te)
    return yp_pair_te + r_te

# ─── Evaluation loop ──────────────────────────────────────────────────────────
N_TRS = [27, 30, 35, 40, 45, 50]
N_SEED = 50
direct_models  = ['M0','M1','M8','M9','M10','RBF']
pair_keys      = ['WK','WD','KD']
pair_model_set = ['M1','M10','RBF']
resid_kinds    = ['none','linear','rbf']
hier_labels    = [(pk, pm, rk) for pk in pair_keys for pm in pair_model_set for rk in resid_kinds]

X_te = X3_jsd[test_set]; y_te = y3[test_set]; X_te_idx = X3_idx[test_set]
storage = {}

print(f"\nRunning Direct vs Hierarchical evaluation (50 seeds for N_tr ∈ {{30,35,40,45}})...")
for n_tr in N_TRS:
    n_extra = n_tr - 27
    seeds = [None] if n_extra in (0, 23) else list(range(N_SEED))

    for seed in seeds:
        if n_extra == 0:
            tr_idx = grid_samples
        elif n_extra == 23:
            tr_idx = train_pool
        else:
            rng = np.random.RandomState(seed)
            tr_idx = np.concatenate([grid_samples, extras[rng.choice(23, n_extra, replace=False)]])

        x3_tr = X3_jsd[tr_idx]; y_tr = y3[tr_idx]; x3_tr_idx = X3_idx[tr_idx]

        # Direct
        for m in direct_models:
            try:
                if m == 'RBF':
                    mr = RBF(kernel='cubic', tail='linear', lb=lb_idx, ub=ub_idx)
                    mr.fit(x3_tr_idx, y_tr)
                    yp_te = mr.predict(X_te_idx).ravel()
                else:
                    _, yp_te = fit_pred3(m, x3_tr, y_tr, X_te)
                storage.setdefault(('Direct', m, n_tr), []).append((r2(y_te, yp_te), rmse(y_te, yp_te)))
            except Exception:
                pass

        # Hierarchical
        for pk, pm, rk in hier_labels:
            try:
                yp_te = hierarchical_predict(pk, pm, rk, x3_tr, y_tr, X_te)
                storage.setdefault(('Hier', f'{pk}-{pm}-{rk}', n_tr), []).append(
                    (r2(y_te, yp_te), rmse(y_te, yp_te)))
            except Exception:
                pass

# ─── Reporting ────────────────────────────────────────────────────────────────
def fmt_cell(rs, idx=0):
    if not rs: return f"{'N/A':14s}"
    vals = [r[idx] for r in rs]
    if len(vals) == 1: return f"{vals[0]:.4f}        "
    return f"{np.median(vals):.3f}[{np.percentile(vals,10):.2f},{np.percentile(vals,90):.2f}]"

print("\n" + "="*120)
print("DIRECT 3-way (test R² on 150 hold-out)")
print("="*120)
hdr = f"  {'Model':10s}"
for n_tr in N_TRS:
    hdr += f"  {'*' if n_tr in (27,50) else ' '}N={n_tr:<3d}        "
print(hdr + "    (* = deterministic)")
for m in direct_models:
    line = f"  {m:10s}"
    for n_tr in N_TRS:
        line += f"  {fmt_cell(storage.get(('Direct', m, n_tr), []), 0):16s}"
    print(line)

print("\n" + "="*120)
print("HIERARCHICAL 2-then-1 (test R² on 150 hold-out)")
print("  pair model trained on full N=200 2-way; residual fitted on N_tr 3-way samples")
print("="*120)
for pk in pair_keys:
    a, b = pair_models[pk]['vars']
    third = [v for v in ['W','KV','KVD'] if v not in (a,b)][0]
    print(f"\n  Pair {pk} ({a}+{b}); residual on {third}:")
    for pm in pair_model_set:
        for rk in resid_kinds:
            label = f'{pk}-{pm}-{rk}'
            line = f"    pair={pm:3s} resid={rk:6s}"
            for n_tr in N_TRS:
                line += f"  {fmt_cell(storage.get(('Hier', label, n_tr), []), 0):16s}"
            print(line)

# ─── Top-3 at each N_tr (combined ranking) ────────────────────────────────────
print("\n" + "="*120)
print("Top-3 configurations at each N_tr (by median test R²)")
print("="*120)
for n_tr in N_TRS:
    cands = []
    for m in direct_models:
        rs = storage.get(('Direct', m, n_tr), [])
        if rs: cands.append((f'Direct-{m}', np.median([r[0] for r in rs])))
    for pk, pm, rk in hier_labels:
        rs = storage.get(('Hier', f'{pk}-{pm}-{rk}', n_tr), [])
        if rs: cands.append((f'Hier-{pk}-{pm}-{rk}', np.median([r[0] for r in rs])))
    cands.sort(key=lambda x: -x[1])
    print(f"  N={n_tr}: " + " | ".join([f"{c[0]:30s} {c[1]:.4f}" for c in cands[:3]]))

# ─── Direct-best vs Hier-best summary ──────────────────────────────────────────
print("\n" + "="*120)
print("Direct-best vs Hierarchical-best at each N_tr")
print("="*120)
print(f"  {'N_tr':6s}  {'Direct best':35s}  {'Hier best':45s}  {'Δ (hier−direct)':12s}")
for n_tr in N_TRS:
    d_best = max([(m, np.median([r[0] for r in storage.get(('Direct', m, n_tr), [])]))
                  for m in direct_models if storage.get(('Direct', m, n_tr))], key=lambda x: x[1])
    h_best = max([(f'{pk}-{pm}-{rk}',
                   np.median([r[0] for r in storage.get(('Hier', f'{pk}-{pm}-{rk}', n_tr), [])]))
                  for pk, pm, rk in hier_labels if storage.get(('Hier', f'{pk}-{pm}-{rk}', n_tr))],
                 key=lambda x: x[1])
    delta = h_best[1] - d_best[1]
    print(f"  N={n_tr}    Direct-{d_best[0]:8s} R²={d_best[1]:.4f}    "
          f"Hier-{h_best[0]:25s} R²={h_best[1]:.4f}    Δ={delta:+.4f}")

# ─── Figures ──────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')

direct_colors = {'M0':'#E64B35','M1':'#3C5488','M8':'#4DBBD5','M9':'#F39B7F','M10':'#9B59B6','RBF':'#00A087'}
pair_colors   = {'WK':'#E64B35','WD':'#4DBBD5','KD':'#9B59B6'}

def collect(key_tuple):
    xs, med, lo, hi = [], [], [], []
    for n_tr in N_TRS:
        rs = storage.get(key_tuple + (n_tr,), [])
        if not rs: continue
        v = [r[0] for r in rs]
        xs.append(n_tr); med.append(np.median(v))
        lo.append(np.percentile(v, 10) if len(v) > 1 else v[0])
        hi.append(np.percentile(v, 90) if len(v) > 1 else v[0])
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)

# Fig 1: Direct vs Hier-best per pair (best resid) — clean comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

ax = axes[0]
for m in direct_models:
    xs, med, lo, hi = collect(('Direct', m))
    ax.plot(xs, med, '-o', color=direct_colors[m], label=f'Direct-{m}', lw=2, ms=5)
    if len(med) and lo[0] != hi[0]:
        ax.fill_between(xs, lo, hi, color=direct_colors[m], alpha=0.10)
ax.set_xlabel('N_train (3-way)'); ax.set_ylabel('Test R² on 150 hold-out')
ax.set_title('Direct 3-way', fontweight='bold')
ax.set_xticks(N_TRS); ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.25, lw=0.5)

ax = axes[1]
# best hier per pair (we'll plot RBF pair model + best resid)
for pk in pair_keys:
    # Pick best (pair_model, resid) combination at each N_tr (averaged across N_tr)
    avg_r2 = {}
    for pm in pair_model_set:
        for rk in resid_kinds:
            rs_all = []
            for n_tr in N_TRS:
                rs = storage.get(('Hier', f'{pk}-{pm}-{rk}', n_tr), [])
                if rs: rs_all.extend([r[0] for r in rs])
            if rs_all: avg_r2[(pm, rk)] = np.mean(rs_all)
    best_pm, best_rk = max(avg_r2, key=avg_r2.get)
    xs, med, lo, hi = collect(('Hier', f'{pk}-{best_pm}-{best_rk}'))
    a, b = pair_models[pk]['vars']; third = [v for v in ['W','KV','KVD'] if v not in (a,b)][0]
    ax.plot(xs, med, '-s', color=pair_colors[pk],
            label=f'Hier-{pk}({a}+{b}) pair={best_pm}, resid={best_rk}', lw=2, ms=5)
    if len(med) and lo[0] != hi[0]:
        ax.fill_between(xs, lo, hi, color=pair_colors[pk], alpha=0.12)

# Overlay the best Direct (RBF) for reference
xs, med, _, _ = collect(('Direct', 'RBF'))
ax.plot(xs, med, '--', color='black', lw=1.8, label='Direct-RBF (ref)', alpha=0.7)
ax.set_xlabel('N_train (3-way)'); ax.set_ylabel('Test R² on 150 hold-out')
ax.set_title('Best Hierarchical per pair (vs Direct-RBF)', fontweight='bold')
ax.set_xticks(N_TRS); ax.legend(fontsize=8); ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle('Direct 3-way vs Hierarchical 2-then-1 (fixed 150 hold-out, low-N regime)',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/v3_hier_lowN_summary.png', **PLT_KW); plt.close()

# Fig 2: 3-panel detail per pair (all 9 hier configs per pair)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
pm_styles = {'M1': 'o', 'M10': 's', 'RBF': '^'}
rk_styles = {'none': ':', 'linear': '--', 'rbf': '-'}

for ax, pk in zip(axes, pair_keys):
    a, b = pair_models[pk]['vars']
    third = [v for v in ['W','KV','KVD'] if v not in (a,b)][0]
    for pm in pair_model_set:
        for rk in resid_kinds:
            xs, med, _, _ = collect(('Hier', f'{pk}-{pm}-{rk}'))
            ax.plot(xs, med, marker=pm_styles[pm], linestyle=rk_styles[rk],
                    label=f'pair={pm}, res={rk}', ms=5, lw=1.5, alpha=0.85)
    # Direct RBF reference
    xs, med, _, _ = collect(('Direct', 'RBF'))
    ax.plot(xs, med, '-', color='black', lw=2, label='Direct-RBF', zorder=10)
    ax.set_xlabel('N_train (3-way)'); ax.set_ylabel('Test R²')
    ax.set_title(f'Pair {pk} ({a}+{b})  →  residual on {third}', fontweight='bold')
    ax.set_xticks(N_TRS); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle('Hierarchical 2-then-1: all (pair-model × residual-kind) configurations',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/v3_hier_lowN_detail.png', **PLT_KW); plt.close()

# Fig 3: Δ heatmap (hier_best − direct_best)
fig, ax = plt.subplots(figsize=(8, 4.6))
labels_dir = direct_models
labels_hier = [f'{pk}-{pm}-{rk}' for pk in pair_keys for pm in pair_model_set for rk in resid_kinds]
M = np.full((len(N_TRS), 1 + 3), np.nan)
mat_data = []
for i, n_tr in enumerate(N_TRS):
    d = max([np.median([r[0] for r in storage.get(('Direct', m, n_tr), [])])
             for m in direct_models if storage.get(('Direct', m, n_tr))], default=np.nan)
    row = [d]
    for pk in pair_keys:
        h = max([np.median([r[0] for r in storage.get(('Hier', f'{pk}-{pm}-{rk}', n_tr), [])])
                 for pm in pair_model_set for rk in resid_kinds
                 if storage.get(('Hier', f'{pk}-{pm}-{rk}', n_tr))], default=np.nan)
        row.append(h)
    mat_data.append(row)
mat_data = np.array(mat_data)  # (n_n_trs, 4)
im = ax.imshow(mat_data, cmap='viridis', aspect='auto', vmin=0.93, vmax=1.0)
ax.set_xticks(range(4)); ax.set_xticklabels(['Direct-best','Hier-WK-best','Hier-WD-best','Hier-KD-best'])
ax.set_yticks(range(len(N_TRS))); ax.set_yticklabels([f'N={n}' for n in N_TRS])
for i in range(len(N_TRS)):
    for j in range(4):
        if not np.isnan(mat_data[i,j]):
            ax.text(j, i, f"{mat_data[i,j]:.4f}", ha='center', va='center',
                    color='white' if mat_data[i,j] < 0.97 else 'black', fontsize=10, fontweight='bold')
ax.set_title('Best test R² per category (Direct vs Hierarchical-by-pair)', fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/v3_hier_lowN_heatmap.png', **PLT_KW); plt.close()

print(f"\nFigures saved: v3_hier_lowN_summary.png, v3_hier_lowN_detail.png, v3_hier_lowN_heatmap.png")
print("Done.\n")
