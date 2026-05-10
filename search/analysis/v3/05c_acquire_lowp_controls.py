"""05c_acquire_lowp_controls.py — Generate 25 low-P(violator) control candidates
for predictor calibration validation.

Purpose: User concern that ARD-GP's high-P prediction failures could mean
predictor is broken (not that PF is robust). Counter-test: evaluate candidates
that ARD-GP says are NOT violators (P < 1%). If those also turn out to be
not-violators, predictor's "non-violator" predictions are validated → strengthens
"PF robust" conclusion. If some are actual violators, predictor missed them →
PF robustness conclusion weakened.

Output:
  analysis/v3/acquired_lowP_controls_25.json   (25 archs, sel_idx 100..124)
  analysis/v3/figures/05c_lowp_controls.png

Run:
  python analysis/v3/05c_acquire_lowp_controls.py
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
from scipy.stats import norm

from utils.func import get_net_info
from utils.select import build_arch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

BASE = '/NAS/SJ/actquant/search'
W_STATS  = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_wbits_loss_w_hqq_kv_kivi_iter_200_n_iter_50_w234kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_vdim0_obj_2_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_200.stats'
KV_STATS = f'{BASE}/save/search/think/2603271708_Llama-3.1-8B-Instruct_kvbits_loss_w_hqq_kv_kivi_iter_150_n_iter_30_w4kv234bits_w128kv3264128x3gs0kdim0vdim_128res_len_k_channel_v_token_kdim0_vdim0_obj_1_5_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
KVDIM_STATS = f'{BASE}/save/search/think/2603251553_Llama-3.1-8B-Instruct_kvdim_loss_w_hqq_kv_think_iter_150_n_iter_30_w4kv4bits_w128kv128gs_128res_len_k_channel_v_token_kdim0_16_32_48_64_vdim0_obj_0_128_jsd_co_0.9_mut_0.1_wikitext2_1bs_128sample_2048seq_0token_rbf_512stride/iter_150.stats'
AWQ_3WAY = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
CONFIG_PATH = f'{BASE}/config/llama.json'
MODEL_NAME  = 'Llama-3.1-8B-Instruct'

OUT_DIR = f'{BASE}/analysis/v3'
FIG_DIR = f'{OUT_DIR}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

N_CONTROL    = 25         # control set size
SEL_IDX_BASE = 100        # avoid collision with existing 100 sel_idx 0..99
TOP_K_PER_AXIS = 50
HEAD_DIM = 128
RNG = np.random.RandomState(7)  # different seed from 05b for variety

# ─── Reuse helpers from 05b ──────────────────────────────────────────────────
def load_csv(p):
    with open(p) as f: rows = [r for r in csv.reader(f) if r]
    M = max(len(r) for r in rows); m = np.full((len(rows), M), np.nan)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            try: m[i, j] = float(v)
            except: pass
    return m

def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]; min2 = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < min2: nd.append(i); min2 = F_s[i, 1]
    return order[nd]

def pareto_layers(losses, comps, n_layers=3):
    F = np.column_stack([losses, comps])
    layers, remaining = [], np.arange(len(F))
    for _ in range(n_layers):
        if len(remaining) == 0: break
        front = NonDominatedSorting().do(F[remaining], only_non_dominated_front=True)
        layers.append(remaining[front])
        remaining = np.setdiff1d(remaining, remaining[front])
    return layers, remaining

def select_axis_pool(losses, comps, K_pf=20, K_subpf=20, K_dom=10):
    layers, dominated = pareto_layers(losses, comps, n_layers=3)
    pf, sub = layers[0], np.concatenate(layers[1:]) if len(layers) > 1 else np.array([], dtype=int)
    if len(pf) <= K_pf:
        pf_sel = pf
    else:
        sort_idx = np.argsort(comps[pf])
        pick = np.linspace(0, len(pf) - 1, K_pf, dtype=int)
        pf_sel = pf[sort_idx[pick]]
    sub_sel = sub if len(sub) <= K_subpf else RNG.choice(sub, K_subpf, replace=False)
    dom_sel = dominated if len(dominated) <= K_dom else RNG.choice(dominated, K_dom, replace=False)
    return np.unique(np.concatenate([pf_sel, sub_sel, dom_sel]))

def load_archive(stats_path, comp_key, config, group_size):
    with open(stats_path) as f: data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs = [v[0] for v in archive]
    losses = np.array([v[1] for v in archive])
    comps  = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
    return archs, losses, comps

def match_metric(comp_vals, pf_arr):
    return np.array([pf_arr[np.argmin(np.abs(pf_arr[:, 1] - c)), 0] for c in comp_vals])

def loc_pf(losses, comps):
    F = np.column_stack([losses, comps])
    return F[pareto_front_2d(F)]

def fit_ard_gp(X, y, n_dim=3, n_restarts=20):
    kernel = (C(1.0, (1e-4, 1e2)) *
              SKRBF(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-4, 1e4)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y); return gp

# ─── Build the same pool + ARD-GP as 05b ────────────────────────────────────
print("Loading + setup (same as 05b) ...")
with open(CONFIG_PATH) as f:
    config = json.load(f)[MODEL_NAME]
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

W_archs,  W_losses,  W_comps  = load_archive(W_STATS,    'wbits',  config, group_size)
KV_archs, KV_losses, KV_comps = load_archive(KV_STATS,   'kvbits', config, group_size)
KVD_archs,KVD_losses,KVD_comps= load_archive(KVDIM_STATS,'kvdim',  config, group_size)

mat = load_csv(AWQ_3WAY); v3 = ~np.isnan(mat[12, :mat.shape[1]])
y_pf  = mat[12, :][v3]
wb_pf = mat[0,  :][v3]; kvb_pf = mat[1, :][v3]; kvd_pf = mat[4, :][v3]
eff_pf = kvb_pf * kvd_pf / HEAD_DIM

pf_W = loc_pf(W_losses, W_comps)
pf_KV = loc_pf(KV_losses, KV_comps)
pf_KVD = loc_pf(KVD_losses, KVD_comps)

X_train = np.column_stack([
    match_metric(wb_pf,  pf_W),
    match_metric(kvb_pf, pf_KV),
    match_metric(kvd_pf, pf_KVD),
])
gp = fit_ard_gp(X_train, y_pf)

# Per-axis pools (same RNG as 05b would give same pools — but use different seed
# is fine since we want broad coverage. Use 05b's RNG seed)
RNG_pool = np.random.RandomState(0)  # match 05b
def select_axis_pool_seeded(losses, comps, K_pf=20, K_subpf=20, K_dom=10):
    layers, dominated = pareto_layers(losses, comps, n_layers=3)
    pf, sub = layers[0], np.concatenate(layers[1:]) if len(layers) > 1 else np.array([], dtype=int)
    if len(pf) <= K_pf: pf_sel = pf
    else:
        sort_idx = np.argsort(comps[pf])
        pick = np.linspace(0, len(pf) - 1, K_pf, dtype=int)
        pf_sel = pf[sort_idx[pick]]
    sub_sel = sub if len(sub) <= K_subpf else RNG_pool.choice(sub, K_subpf, replace=False)
    dom_sel = dominated if len(dominated) <= K_dom else RNG_pool.choice(dominated, K_dom, replace=False)
    return np.unique(np.concatenate([pf_sel, sub_sel, dom_sel]))

W_pool   = select_axis_pool_seeded(W_losses, W_comps)[:TOP_K_PER_AXIS]
KV_pool  = select_axis_pool_seeded(KV_losses, KV_comps)[:TOP_K_PER_AXIS]
KVD_pool = select_axis_pool_seeded(KVD_losses, KVD_comps)[:TOP_K_PER_AXIS]
print(f"  pools: {len(W_pool)}/{len(KV_pool)}/{len(KVD_pool)}")

NW, NKV, NKVD = len(W_pool), len(KV_pool), len(KVD_pool)
i_arr, j_arr, k_arr = np.meshgrid(np.arange(NW), np.arange(NKV), np.arange(NKVD), indexing='ij')
i_flat, j_flat, k_flat = i_arr.ravel(), j_arr.ravel(), k_arr.ravel()
X_cand = np.column_stack([W_losses[W_pool][i_flat],
                           KV_losses[KV_pool][j_flat],
                           KVD_losses[KVD_pool][k_flat]])
cW_cand   = W_comps[W_pool][i_flat]
cKV_cand  = KV_comps[KV_pool][j_flat]
cKVD_cand = KVD_comps[KVD_pool][k_flat]
eff_cand  = cKV_cand * cKVD_cand / HEAD_DIM
N_pool = len(i_flat)

# Predict
mu_cand    = np.empty(N_pool)
sigma_cand = np.empty(N_pool)
for s in range(0, N_pool, 50000):
    e = min(s + 50000, N_pool)
    mu_cand[s:e], sigma_cand[s:e] = gp.predict(X_cand[s:e], return_std=True)

# 3D Pareto-envelope baseline (vectorised loop)
f_star = np.full(N_pool, np.nan)
for i in range(N_pool):
    mask = ((wb_pf >= cW_cand[i] - 1e-9) &
            (kvb_pf >= cKV_cand[i] - 1e-9) &
            (kvd_pf >= cKVD_cand[i] - 1e-9))
    if mask.any():
        f_star[i] = y_pf[mask].max()

in_cloud = ~np.isnan(f_star)
gap   = f_star - mu_cand
sigma_safe = np.maximum(sigma_cand, 1e-6)
prob_3d = np.where(in_cloud, norm.cdf(gap / sigma_safe), np.nan)

# Off-Cartesian-PF mask (same as 05b)
def is_on_pf(losses_pool, comps_pool, pf_arr, atol=1e-9):
    on = np.zeros(len(losses_pool), dtype=bool)
    for i, (l, c) in enumerate(zip(losses_pool, comps_pool)):
        nearest = np.argmin(np.abs(pf_arr[:, 1] - c))
        on[i] = abs(pf_arr[nearest, 0] - l) < atol
    return on
W_on   = is_on_pf(W_losses[W_pool],  W_comps[W_pool],  pf_W)
KV_on  = is_on_pf(KV_losses[KV_pool], KV_comps[KV_pool], pf_KV)
KVD_on = is_on_pf(KVD_losses[KVD_pool],KVD_comps[KVD_pool],pf_KVD)
cart_mask = W_on[i_flat] & KV_on[j_flat] & KVD_on[k_flat]
eligible_low = in_cloud & ~cart_mask & (prob_3d < 0.01)
print(f"  in-cloud + off-Cart + P<1%: {int(eligible_low.sum()):,}")

# Stratified low-P selection: 5×5 buckets, top by 1−P (i.e., highest "non-violator
# confidence") within each bucket. So we pick candidates ARD-GP is most confident
# are NOT violators. If any of these turn out to be violators → predictor failed.
NB_W, NB_KV = 5, 5
elig_idx = np.where(eligible_low)[0]
w_edges  = np.quantile(cW_cand[elig_idx], np.linspace(0, 1, NB_W  + 1))
kv_edges = np.quantile(eff_cand[elig_idx], np.linspace(0, 1, NB_KV + 1))
w_edges[0] -= 1e-9;  w_edges[-1]  += 1e-9
kv_edges[0] -= 1e-9; kv_edges[-1] += 1e-9
def bid(cw, ckv):
    bw  = max(0, min(NB_W -1, int(np.searchsorted(w_edges,  cw,  side='right') - 1)))
    bkv = max(0, min(NB_KV-1, int(np.searchsorted(kv_edges, ckv, side='right') - 1)))
    return bw * NB_KV + bkv
bucket_lists = [[] for _ in range(NB_W * NB_KV)]
for idx in elig_idx:
    bucket_lists[bid(cW_cand[idx], eff_cand[idx])].append(idx)
# top by lowest P (ARD-GP says STRONGLY not a violator)
for b in range(len(bucket_lists)):
    bucket_lists[b].sort(key=lambda i: prob_3d[i])
selected = []
for round_idx in range(5):  # at most 5 rounds × 25 buckets
    for b in range(len(bucket_lists)):
        if round_idx < len(bucket_lists[b]):
            selected.append(bucket_lists[b][round_idx])
        if len(selected) >= N_CONTROL: break
    if len(selected) >= N_CONTROL: break
selected = selected[:N_CONTROL]
print(f"  selected {len(selected)} low-P controls (target {N_CONTROL})")

# Build arch dicts
n_block = config['n_block']; w_linears = config['linear']
default_arch = {
    'q': {'w': {linear: [4]*n_block for linear in w_linears},
          'k': [[4, 128]]*n_block, 'v': [[4, 128]]*n_block},
    'p': {'k': [0]*n_block, 'v': [0]*n_block},
}
esm = {
    'w':     np.array([W_archs[i]   for i in W_pool],   dtype=object),
    'kv':    np.array([KV_archs[i]  for i in KV_pool],  dtype=object),
    'kvdim': np.array([KVD_archs[i] for i in KVD_pool], dtype=object),
}
expr_keys = ['w', 'kv', 'kvdim']

archs, records = [], []
for s, idx in enumerate(selected):
    iw, jkv, kkvd = int(i_flat[idx]), int(j_flat[idx]), int(k_flat[idx])
    arch = build_arch(default_arch, expr_keys, esm, np.array([iw, jkv, kkvd]))
    info = get_net_info(arch, config, group_size)
    eff_kv = float(info.get('eff_kvbits', info.get('kvbits', 0)))
    rec = {
        'sel_idx': SEL_IDX_BASE + s,
        'pool_idx': int(idx),
        'src': 'lowP_control',
        'pred_mu': float(mu_cand[idx]),
        'pred_sigma': float(sigma_cand[idx]),
        'pf_baseline': float(f_star[idx]),
        'pred_gap': float(gap[idx]),
        'prob_violator': float(prob_3d[idx]),
        'pf_baseline_3d': float(f_star[idx]),
        'pred_gap_3d': float(gap[idx]),
        'prob_violator_3d': float(prob_3d[idx]),
        'wbits': float(info['wbits']),
        'kvbits': float(info.get('kvbits', np.nan)),
        'kvdim': float(info.get('kvdim', 0)),
        'eff_kvbits': eff_kv,
        'total_c': float(info['wbits']) + eff_kv,
        'on_cart_pf': bool(cart_mask[idx]),
        'in_cloud_3d': True,
        'sub_idx': {'w': iw, 'kv': jkv, 'kvdim': kkvd},
    }
    archs.append(arch); records.append(rec)

# Save
out_path = f'{OUT_DIR}/acquired_lowP_controls_25.json'
with open(out_path, 'w') as f:
    json.dump({'archs': archs, 'records': records,
               'config': {'role': 'low_P_control', 'n': len(records),
                          'pool_size': N_pool,
                          'eligible_low_P_count': int(eligible_low.sum())}},
              f, indent=2,
              default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
print(f"\nSaved {out_path}")

# Diagnostics
P_vals = np.array([r['prob_violator'] for r in records])
print(f"\nControl set:")
print(f"  P range:  [{P_vals.min():.6f}, {P_vals.max():.6f}]")
print(f"  P mean:   {P_vals.mean():.6f}  median: {np.median(P_vals):.6f}")
print(f"  pred_gap range: [{min(r['pred_gap'] for r in records):+.4f}, "
      f"{max(r['pred_gap'] for r in records):+.4f}]")
print(f"  wbits range:    [{min(r['wbits'] for r in records):.2f}, "
      f"{max(r['wbits'] for r in records):.2f}]")
print(f"  eff_kv range:   [{min(r['eff_kvbits'] for r in records):.2f}, "
      f"{max(r['eff_kvbits'] for r in records):.2f}]")

# Quick figure: where are these in (wb, eff_kv) vs PF training
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
sel_arr = np.array(selected)
ax.scatter(wb_pf, eff_pf, s=14, c='lightgray', alpha=0.7, label=f'PF train (n={len(y_pf)})')
ax.scatter([r['wbits'] for r in records], [r['eff_kvbits'] for r in records],
           s=42, c='blue', edgecolor='black', linewidth=0.3, label=f'low-P controls (n={N_CONTROL})')
# Also show high-P 100 from acquired_offsurface_100
try:
    with open(f'{OUT_DIR}/acquired_offsurface_100.json') as f:
        main_recs = json.load(f)['records']
    ax.scatter([r['wbits'] for r in main_recs], [r['eff_kvbits'] for r in main_recs],
               s=20, c='red', alpha=0.4, label=f'high-P main (n={len(main_recs)})')
except Exception:
    pass
ax.set_xlabel('wbits'); ax.set_ylabel('eff_kvbits')
ax.set_title('control vs main candidates in 2D complexity', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(prob_3d[eligible_low], bins=30, alpha=0.4, color='gray', label='eligible low-P pool')
ax.hist(P_vals, bins=10, alpha=0.85, color='blue', label='selected (n=25)')
ax.set_xlabel('P_3d (violator) — note these are <1%')
ax.set_ylabel('count'); ax.set_yscale('log')
ax.set_title('low-P control selection', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = f'{FIG_DIR}/05c_lowp_controls.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved {fig_path}")
print("Done.")
