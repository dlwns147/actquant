"""05b_acquire_offsurface_3d.py — Acquire 100 off-Cartesian-PF samples for AWQ
falsification using **3D Pareto-envelope baseline** in (wbits, kvbits, kvdim).

Key fix vs 05: original used 1D `total_c = wbits + eff_kvbits` for the f*(c)
baseline, which collapsed (wbits=2,eff=4) and (wbits=4,eff=2) into the same
point and mis-ranked candidates. Diagnostic [08_check_selection_bias.py] showed
top-K overlap between 1D-rank and 3D-rank was 0% at K=30 and 25/25 of 1D-top-25
were out-of-cloud (cannot be ε-violators by definition).

This script:
  1. Loads AWQ-measured 200-sample training set as ground-truth PF.
  2. For each candidate (in 125k Cartesian product of local archive sub-arches),
     computes 3D Pareto-envelope baseline:
        f*_3D(c) = max{y_PF : ∀axis c_PF ≥ c}     (NaN if no PF dominates ⇒ out-of-cloud)
  3. Discards out-of-cloud candidates (cannot be Pareto-violators by definition).
  4. Computes P(violator) = Φ((f*_3D − μ_GP)/σ_GP) using the trained ARD-GP.
  5. Stratifies by 2D bucket (wbits × eff_kvbits, 5×5=25) — picks top-K
     P(violator) per bucket; fills with high-σ extras (P>1%).
  6. Saves 100 in-cloud, well-stratified candidates for AWQ evaluation.

Outputs (overwrite previous 05 outputs since user requested fresh restart):
  acquired_offsurface_100.json
  acquired_offsurface_100.txt
  figures/05b_acquire_overview.png
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

# ─── Config ──────────────────────────────────────────────────────────────────
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

N_BUDGET    = 100
N_BUCKETS_W  = 5  # wbits buckets
N_BUCKETS_KV = 5  # eff_kvbits buckets
N_PER_BUCKET = 3   # → 75 main, 25 σ-extras

TOP_K_PER_AXIS = 50  # pool size per axis

HEAD_DIM = 128
RNG = np.random.RandomState(0)

# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path) as f: rows = [r for r in csv.reader(f) if r]
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

# ─── Step 1: load data ──────────────────────────────────────────────────────
print("=" * 80)
print("Step 1: Loading config, archives, and AWQ training set")
print("=" * 80)
with open(CONFIG_PATH) as f:
    config = json.load(f)[MODEL_NAME]
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

W_archs,   W_losses,   W_comps   = load_archive(W_STATS,    'wbits',  config, group_size)
KV_archs,  KV_losses,  KV_comps  = load_archive(KV_STATS,   'kvbits', config, group_size)
KVD_archs, KVD_losses, KVD_comps = load_archive(KVDIM_STATS,'kvdim',  config, group_size)
print(f"  W: n={len(W_archs)}, KV: n={len(KV_archs)}, KVDIM: n={len(KVD_archs)}")

# AWQ training set (used as ground-truth Pareto front in 3D)
mat = load_csv(AWQ_3WAY)
N0 = mat.shape[1]
v3 = ~np.isnan(mat[12, :N0])
y_pf      = mat[12, :N0][v3]
wb_pf     = mat[0,  :N0][v3]
kvb_pf    = mat[1,  :N0][v3]
kvd_pf    = mat[4,  :N0][v3]
eff_pf    = kvb_pf * kvd_pf / HEAD_DIM
PF_TR     = np.column_stack([y_pf, wb_pf, kvb_pf, kvd_pf])    # (y, c_w, c_kv, c_kd)
print(f"  AWQ-measured PF training: N={len(y_pf)}")
print(f"    wbits=[{wb_pf.min():.2f},{wb_pf.max():.2f}]  "
      f"kvbits=[{kvb_pf.min():.2f},{kvb_pf.max():.2f}]  "
      f"kvdim=[{kvd_pf.min():.0f},{kvd_pf.max():.0f}]")
print(f"    y_pf range: [{y_pf.min():.5f}, {y_pf.max():.5f}]")

# ─── Step 2: Fit ARD-GP on AWQ training set ────────────────────────────────
# ARD-GP input is 3D loss values from local PFs (matching 02_ard_gp_analysis.py)
def match_metric(comp_vals, pf_arr):
    return np.array([pf_arr[np.argmin(np.abs(pf_arr[:, 1] - c)), 0] for c in comp_vals])

# Local PFs (sorted by comp)
def loc_pf(losses, comps):
    F = np.column_stack([losses, comps])
    return F[pareto_front_2d(F)]

pf_W   = loc_pf(W_losses,   W_comps)
pf_KV  = loc_pf(KV_losses,  KV_comps)
pf_KVD = loc_pf(KVD_losses, KVD_comps)

xW_train  = match_metric(wb_pf,  pf_W  )
xKV_train = match_metric(kvb_pf, pf_KV )
xKVD_tr   = match_metric(kvd_pf, pf_KVD)
X_train   = np.column_stack([xW_train, xKV_train, xKVD_tr])
print(f"\nFitting ARD-GP on N={len(y_pf)} (3D loss inputs from local PFs)...")
def fit_ard_gp(X, y, n_dim=3, n_restarts=20):
    kernel = (C(1.0, (1e-4, 1e2)) *
              SKRBF(length_scale=[1.0]*n_dim, length_scale_bounds=(1e-4, 1e4)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y); return gp
gp = fit_ard_gp(X_train, y_pf)
ls = np.array(gp.kernel_.k1.k2.length_scale)
print(f"  length scales: W={ls[0]:.4f}, KV={ls[1]:.4f}, KVD={ls[2]:.4f}")

# ─── Step 3: Build per-axis pools ───────────────────────────────────────────
print("\n" + "=" * 80)
print("Step 3: Per-axis pools")
print("=" * 80)
W_pool   = select_axis_pool(W_losses,   W_comps,   K_pf=20, K_subpf=20, K_dom=10)[:TOP_K_PER_AXIS]
KV_pool  = select_axis_pool(KV_losses,  KV_comps,  K_pf=20, K_subpf=20, K_dom=10)[:TOP_K_PER_AXIS]
KVD_pool = select_axis_pool(KVD_losses, KVD_comps, K_pf=20, K_subpf=20, K_dom=10)[:TOP_K_PER_AXIS]
print(f"  |W|={len(W_pool)} × |KV|={len(KV_pool)} × |KVD|={len(KVD_pool)} = "
      f"{len(W_pool)*len(KV_pool)*len(KVD_pool):,}")

# ─── Step 4: Vectorised candidate construction ──────────────────────────────
NW, NKV, NKVD = len(W_pool), len(KV_pool), len(KVD_pool)
i_arr, j_arr, k_arr = np.meshgrid(np.arange(NW), np.arange(NKV), np.arange(NKVD), indexing='ij')
i_flat, j_flat, k_flat = i_arr.ravel(), j_arr.ravel(), k_arr.ravel()
N_pool = len(i_flat)
xW_p   = W_losses[W_pool];   cW_p   = W_comps[W_pool]
xKV_p  = KV_losses[KV_pool]; cKV_p  = KV_comps[KV_pool]
xKVD_p = KVD_losses[KVD_pool]; cKVD_p = KVD_comps[KVD_pool]
X_cand = np.column_stack([xW_p[i_flat], xKV_p[j_flat], xKVD_p[k_flat]])
cW_cand   = cW_p[i_flat]
cKV_cand  = cKV_p[j_flat]
cKVD_cand = cKVD_p[k_flat]
eff_cand  = cKV_cand * cKVD_cand / HEAD_DIM

# ─── Step 5: ARD-GP batch prediction ────────────────────────────────────────
print("\n" + "=" * 80)
print(f"Step 5: ARD-GP prediction on {N_pool:,} candidates")
print("=" * 80)
mu_cand    = np.empty(N_pool)
sigma_cand = np.empty(N_pool)
batch = 50000
for s in range(0, N_pool, batch):
    e = min(s + batch, N_pool)
    mu_cand[s:e], sigma_cand[s:e] = gp.predict(X_cand[s:e], return_std=True)
print(f"  μ:[{mu_cand.min():.4f},{mu_cand.max():.4f}]  σ:[{sigma_cand.min():.4f},{sigma_cand.max():.4f}]")

# ─── Step 6: 3D Pareto-envelope baseline (vectorised) ───────────────────────
print("\n" + "=" * 80)
print("Step 6: 3D Pareto-envelope baseline f*_3D(c_w, c_kv, c_kd)")
print("       = max{y_PF : ∀axis c_PF ≥ c_candidate}")
print("=" * 80)
# For each candidate, find max y_PF over PF points dominating c_candidate.
# Vectorised: candidate[i] dominator-set mask = (wb_pf >= cW_cand[i]) & (kvb_pf >= cKV_cand[i]) & (kvd_pf >= cKVD_cand[i])
# Then f_star[i] = max(y_pf[mask])  or NaN if no mask.
# Loop over candidates is 125k × 200 = 25M ops, fast.
f_star_3d = np.full(N_pool, np.nan)
for i in range(N_pool):
    mask = ((wb_pf >= cW_cand[i] - 1e-9) &
            (kvb_pf >= cKV_cand[i] - 1e-9) &
            (kvd_pf >= cKVD_cand[i] - 1e-9))
    if mask.any():
        f_star_3d[i] = y_pf[mask].max()

in_cloud = ~np.isnan(f_star_3d)
print(f"  in-cloud candidates: {in_cloud.sum():,} / {N_pool:,}  "
      f"({100*in_cloud.mean():.1f}%)")

# ─── Step 7: P(violator) using 3D baseline ──────────────────────────────────
sigma_safe = np.maximum(sigma_cand, 1e-6)
gap_3d   = f_star_3d - mu_cand
prob_3d  = np.where(in_cloud, norm.cdf(gap_3d / sigma_safe), np.nan)

# Identify off-Cartesian-PF candidates (at least one sub-arch not on local PF)
def is_on_pf(losses_pool, comps_pool, pf_arr, atol=1e-9):
    on = np.zeros(len(losses_pool), dtype=bool)
    for i, (l, c) in enumerate(zip(losses_pool, comps_pool)):
        nearest = np.argmin(np.abs(pf_arr[:, 1] - c))
        on[i] = abs(pf_arr[nearest, 0] - l) < atol
    return on
W_on   = is_on_pf(xW_p,   cW_p,   pf_W  )
KV_on  = is_on_pf(xKV_p,  cKV_p,  pf_KV )
KVD_on = is_on_pf(xKVD_p, cKVD_p, pf_KVD)
cart_mask = W_on[i_flat] & KV_on[j_flat] & KVD_on[k_flat]

# Eligible: in-cloud AND off-Cartesian-PF
eligible = in_cloud & ~cart_mask
elig_idx = np.where(eligible)[0]
print(f"  eligible (in-cloud & off-Cartesian-PF): {len(elig_idx):,} / {N_pool:,}")
print(f"  P_3d on eligible: max={prob_3d[elig_idx].max():.4f},  "
      f"mean={prob_3d[elig_idx].mean():.4f},  "
      f"P>0.5={int((prob_3d[elig_idx] > 0.5).sum())},  "
      f"P>0.1={int((prob_3d[elig_idx] > 0.1).sum())}")

# ─── Step 8: 2D bucketing (wbits × eff_kvbits) ──────────────────────────────
print("\n" + "=" * 80)
print(f"Step 8: 2D bucketing ({N_BUCKETS_W} wbits × {N_BUCKETS_KV} eff_kvbits)")
print("=" * 80)
w_edges  = np.quantile(cW_cand[eligible],   np.linspace(0, 1, N_BUCKETS_W + 1))
kv_edges = np.quantile(eff_cand[eligible],  np.linspace(0, 1, N_BUCKETS_KV + 1))
w_edges[0] -= 1e-9;  w_edges[-1]  += 1e-9
kv_edges[0] -= 1e-9; kv_edges[-1] += 1e-9
print(f"  wbits edges:      {[f'{e:.2f}' for e in w_edges]}")
print(f"  eff_kvbits edges: {[f'{e:.2f}' for e in kv_edges]}")

def bucket_id_2d(cw, ckv):
    bw  = int(np.searchsorted(w_edges,  cw,  side='right') - 1)
    bkv = int(np.searchsorted(kv_edges, ckv, side='right') - 1)
    bw  = max(0, min(N_BUCKETS_W  - 1, bw))
    bkv = max(0, min(N_BUCKETS_KV - 1, bkv))
    return bw * N_BUCKETS_KV + bkv

bucket_ids = np.array([bucket_id_2d(cW_cand[i], eff_cand[i]) for i in elig_idx])

# ─── Step 9: Stratified P(violator) selection ───────────────────────────────
sel_main = []
for b in range(N_BUCKETS_W * N_BUCKETS_KV):
    in_b = elig_idx[bucket_ids == b]
    if len(in_b) == 0: continue
    # top by P_3d
    sel_main.extend(in_b[np.argsort(-prob_3d[in_b])[:N_PER_BUCKET]].tolist())
sel_main = list(dict.fromkeys(sel_main))
print(f"\nStep 9: stratified P_3d picks: {len(sel_main)} unique")

# σ-extras: high σ in P>1% region
plausible = elig_idx[prob_3d[elig_idx] > 0.01]
remaining = np.setdiff1d(plausible, np.array(sel_main, dtype=int))
n_extra = N_BUDGET - len(sel_main)
extras = remaining[np.argsort(-sigma_cand[remaining])[:n_extra]].tolist()
if len(extras) < n_extra:
    fill_pool = np.setdiff1d(elig_idx, np.array(sel_main + extras, dtype=int))
    fill = fill_pool[np.argsort(-prob_3d[fill_pool])[:n_extra - len(extras)]].tolist()
    extras = extras + fill
print(f"        σ-extras (P>1%): {len(extras)}")
selected = list(dict.fromkeys(sel_main + extras))[:N_BUDGET]
print(f"        total: {len(selected)} (target {N_BUDGET})")

# ─── Step 10: Build merged arch dicts ───────────────────────────────────────
n_block = config['n_block']; w_linears = config['linear']
default_arch = {
    'q': {
        'w': {linear: [4]*n_block for linear in w_linears},
        'k': [[4, 128]]*n_block, 'v': [[4, 128]]*n_block,
    },
    'p': {'k': [0]*n_block, 'v': [0]*n_block},
}
esm = {
    'w':     np.array([W_archs[i]   for i in W_pool],   dtype=object),
    'kv':    np.array([KV_archs[i]  for i in KV_pool],  dtype=object),
    'kvdim': np.array([KVD_archs[i] for i in KVD_pool], dtype=object),
}
expr_keys = ['w', 'kv', 'kvdim']

selected_archs, records = [], []
for s, idx in enumerate(selected):
    iw, jkv, kkvd = int(i_flat[idx]), int(j_flat[idx]), int(k_flat[idx])
    arch = build_arch(default_arch, expr_keys, esm, np.array([iw, jkv, kkvd]))
    info = get_net_info(arch, config, group_size)
    eff_kv = float(info.get('eff_kvbits', info.get('kvbits', 0)))
    rec = {
        'sel_idx':   int(s),
        'pool_idx':  int(idx),
        'src':       'P_3d' if s < len(sel_main) else 'sigma|P_3d>1%',
        'pred_mu':   float(mu_cand[idx]),
        'pred_sigma':float(sigma_cand[idx]),
        # canonical keys (3D baseline) — used by downstream 06_evaluate_awq_100.py
        'pf_baseline':    float(f_star_3d[idx]),
        'pred_gap':       float(gap_3d[idx]),
        'prob_violator':  float(prob_3d[idx]),
        # explicit 3D names (kept for clarity)
        'pf_baseline_3d': float(f_star_3d[idx]),
        'pred_gap_3d':    float(gap_3d[idx]),
        'prob_violator_3d': float(prob_3d[idx]),
        'wbits':     float(info['wbits']),
        'kvbits':    float(info.get('kvbits', np.nan)),
        'kvdim':     float(info.get('kvdim', 0)),
        'eff_kvbits':eff_kv,
        'total_c':   float(info['wbits']) + eff_kv,  # for 06's bucketing fallback
        'on_cart_pf':bool(cart_mask[idx]),
        'in_cloud_3d': True,
        'sub_idx':   {'w': iw, 'kv': jkv, 'kvdim': kkvd},
    }
    selected_archs.append(arch)
    records.append(rec)

# ─── Step 11: Save ──────────────────────────────────────────────────────────
out_json = f'{OUT_DIR}/acquired_offsurface_100.json'
with open(out_json, 'w') as f:
    json.dump({'archs': selected_archs, 'records': records,
               'config': {'model_name': MODEL_NAME, 'group_size': group_size,
                          'n_budget': N_BUDGET,
                          'n_buckets_w': N_BUCKETS_W,
                          'n_buckets_kv': N_BUCKETS_KV,
                          'n_per_bucket': N_PER_BUCKET,
                          'top_k_per_axis': TOP_K_PER_AXIS,
                          'pool_size': int(N_pool),
                          'in_cloud_pool': int(in_cloud.sum()),
                          'eligible_pool': int(eligible.sum()),
                          'gp_length_scales': ls.tolist(),
                          'pf_train_n': int(len(y_pf))}},
              f, indent=2,
              default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
print(f"\nSaved {out_json}")

# Summary text
gaps_3d  = np.array([r['pred_gap_3d'] for r in records])
sigmas   = np.array([r['pred_sigma'] for r in records])
probs_3d = np.array([r['prob_violator_3d'] for r in records])
wbits_v  = np.array([r['wbits'] for r in records])
eff_v    = np.array([r['eff_kvbits'] for r in records])
src_cnt  = {'P_3d': sum(1 for r in records if r['src']=='P_3d'),
            'sigma|P_3d>1%': sum(1 for r in records if r['src']=='sigma|P_3d>1%')}

summary = []
summary.append(f"Off-surface acquisition (3D baseline, n={len(records)})")
summary.append(f"=" * 70)
summary.append(f"")
summary.append(f"Sources:")
summary.append(f"  W:    {len(W_archs)} / pooled {len(W_pool)}")
summary.append(f"  KV:   {len(KV_archs)} / pooled {len(KV_pool)}")
summary.append(f"  KVDIM:{len(KVD_archs)} / pooled {len(KVD_pool)}")
summary.append(f"  AWQ-measured PF train: {len(y_pf)}")
summary.append(f"")
summary.append(f"ARD-GP length scales: W={ls[0]:.4f}, KV={ls[1]:.4f}, KVD={ls[2]:.4f}")
summary.append(f"")
summary.append(f"Pool composition:")
summary.append(f"  candidates total:   {N_pool:,}")
summary.append(f"  in-cloud (3D):      {int(in_cloud.sum()):,}  ({100*in_cloud.mean():.1f}%)")
summary.append(f"  off-Cartesian-PF:   {int((~cart_mask).sum()):,}")
summary.append(f"  eligible (both):    {int(eligible.sum()):,}")
summary.append(f"")
summary.append(f"Selection:")
summary.append(f"  P_3d-stratified:    {src_cnt['P_3d']}  "
               f"({N_BUCKETS_W*N_BUCKETS_KV} buckets × top-{N_PER_BUCKET})")
summary.append(f"  σ-extras (P_3d>1%): {src_cnt['sigma|P_3d>1%']}")
summary.append(f"")
summary.append(f"Predicted P(violator) [Φ((f*_3D − μ)/σ)]:")
summary.append(f"  max:        {probs_3d.max():.4f}")
summary.append(f"  mean:       {probs_3d.mean():.4f}")
summary.append(f"  P>10%:      {int((probs_3d > 0.10).sum())}/{len(records)}")
summary.append(f"  P>30%:      {int((probs_3d > 0.30).sum())}/{len(records)}")
summary.append(f"  P>50%:      {int((probs_3d > 0.50).sum())}/{len(records)}")
summary.append(f"")
summary.append(f"Predicted gap (f*_3D − μ):  range=[{gaps_3d.min():+.4f}, {gaps_3d.max():+.4f}]")
summary.append(f"")
summary.append(f"Complexity coverage:")
summary.append(f"  wbits:      [{wbits_v.min():.2f}, {wbits_v.max():.2f}]")
summary.append(f"  eff_kvbits: [{eff_v.min():.2f}, {eff_v.max():.2f}]")

out_txt = f'{OUT_DIR}/acquired_offsurface_100.txt'
with open(out_txt, 'w') as f: f.write('\n'.join(summary))
print(f"Saved {out_txt}")
print('\n' + '\n'.join(summary))

# ─── Step 12: Diagnostic figure ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
sel_arr = np.array(selected, dtype=int)
mask_main = np.isin(sel_arr, np.array(sel_main, dtype=int))
mask_sig  = np.isin(sel_arr, np.array(extras, dtype=int))

# (a) selection in 2D complexity, color by P_3d
ax = axes[0, 0]
sub = RNG.choice(elig_idx, min(15000, len(elig_idx)), replace=False)
ax.scatter(cW_cand[sub], eff_cand[sub], s=3, alpha=0.18, c='lightgray',
           label='eligible pool (sub)')
ax.scatter(wb_pf, eff_pf, s=12, c='black', alpha=0.5, label=f'200 PF train')
sc = ax.scatter(wbits_v, eff_v, c=probs_3d, cmap='YlOrRd', s=40,
                edgecolor='black', linewidth=0.3, vmin=0, vmax=1, label='selected')
plt.colorbar(sc, ax=ax, fraction=0.04, label='P_3d (violator)')
for ed in w_edges[1:-1]:  ax.axvline(ed, color='gray', lw=0.3, alpha=0.5)
for ed in kv_edges[1:-1]: ax.axhline(ed, color='gray', lw=0.3, alpha=0.5)
ax.set_xlabel('wbits'); ax.set_ylabel('eff_kvbits')
ax.set_title('(a) selection in 2D, P_3d colored', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (b) P_3d distribution
ax = axes[0, 1]
ax.hist(prob_3d[elig_idx], bins=50, alpha=0.4, color='gray', label='eligible pool')
ax.hist(probs_3d, bins=20, alpha=0.85, color='red', label='selected (n=100)')
ax.axvline(0.10, color='black', ls='--', lw=0.8, label='P=10%')
ax.axvline(0.30, color='black', ls=':',  lw=0.8, label='P=30%')
ax.set_xlabel('P_3d (violator)'); ax.set_ylabel('count'); ax.set_yscale('log')
ax.set_title('(b) P_3d distribution', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (c) gap_3d vs σ, with iso-P contours
ax = axes[1, 0]
ax.scatter(gap_3d[sub], sigma_cand[sub], s=3, alpha=0.25, c='lightgray',
           label='eligible (sub)')
ax.scatter(gaps_3d[mask_main], sigmas[mask_main], s=22, c='red',
           edgecolor='black', linewidth=0.3, label='P_3d-pick')
ax.scatter(gaps_3d[mask_sig], sigmas[mask_sig], s=22, c='orange',
           edgecolor='black', linewidth=0.3, label='σ-extras')
sg = np.linspace(sigma_cand.min(), sigma_cand.max(), 50)
for P, ls_ in [(0.05, ':'), (0.10, '-.'), (0.50, '--')]:
    ax.plot(norm.ppf(P) * sg, sg, color='black', ls=ls_, lw=0.7, label=f'P={P}')
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('predicted gap (f*_3D − μ)')
ax.set_ylabel('σ')
ax.set_title('(c) acquisition landscape (iso-P)', fontweight='bold')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

# (d) bucket coverage
ax = axes[1, 1]
sel_buck = np.array([bucket_id_2d(wbits_v[i], eff_v[i]) for i in range(len(records))])
counts = np.bincount(sel_buck, minlength=N_BUCKETS_W*N_BUCKETS_KV)
heatmap = counts.reshape(N_BUCKETS_W, N_BUCKETS_KV)
im = ax.imshow(heatmap, origin='lower', cmap='YlOrRd', aspect='auto')
for i in range(N_BUCKETS_W):
    for j in range(N_BUCKETS_KV):
        ax.text(j, i, str(int(heatmap[i, j])), ha='center', va='center',
                color='black', fontweight='bold', fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.04, label='# selected')
ax.set_xticks(range(N_BUCKETS_KV)); ax.set_yticks(range(N_BUCKETS_W))
ax.set_xlabel('eff_kvbits bucket (low→high)')
ax.set_ylabel('wbits bucket (low→high)')
ax.set_title(f'(d) coverage: {N_BUCKETS_W}×{N_BUCKETS_KV} buckets', fontweight='bold')

plt.suptitle(f'Off-surface acquisition w/ 3D baseline (n={len(records)})',
             fontweight='bold', y=1.0)
plt.tight_layout()
fig_path = f'{FIG_DIR}/05b_acquire_overview.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nSaved {fig_path}")
print("Done.")
