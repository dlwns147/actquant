"""05_acquire_offsurface_100.py — Acquire 100 off-Cartesian-PF samples for AWQ
evaluation, to empirically falsify "Cartesian-combined PF ≈ true full PF".

Pipeline:
  1. Load local archives (W, KV, KVDIM stats).
  2. Build candidate pool = product of per-axis (PF + Pareto layer 2-3 + diverse
     sub-Pareto). Each candidate has its own measured sub-loss from local search,
     so 3D coord x = (sub_loss_W, sub_loss_KV, sub_loss_KVD) is well-defined.
  3. Refit ARD-GP on existing AWQ-evaluated samples (Cartesian-PF tuples, 200 pts)
     using sub-arch losses as input — same form as 02_ard_gp_analysis.py.
  4. Predict (μ, σ) for the entire pool.
  5. Build Cartesian-PF interpolant f*(c_total) where c_total = wbits + eff_kvbits.
  6. Acquisition: stratified across complexity buckets via Expected Improvement
     (EI), plus high-σ exploration extras.  Total = 100.
  7. Save merged arch dicts + selection diagnostics for downstream AWQ eval
     (compatible with post_search_split.py's --random_sample_path workflow).

Outputs:
  - acquired_offsurface_100.json   (arch dicts + per-arch metadata)
  - acquired_offsurface_100.txt    (selection summary)
  - figures/05_acquire_overview.png

Run:
  cd /NAS/SJ/actquant/search
  python analysis/v3/05_acquire_offsurface_100.py
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

# Acquisition budget
N_BUDGET       = 100
N_BUCKETS      = 25     # complexity strata
N_PER_BUCKET   = 3      # EI top-K per bucket → 75 EI-stratified
# remaining 25 → high-σ exploration

# Pool size: top-K per axis (PF + sub-Pareto)
TOP_K_PER_AXIS = 50     # 50^3 = 125k candidates, fits in memory

HEAD_DIM = 128
RNG = np.random.RandomState(0)

# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    max_cols = max(len(r) for r in rows)
    mat = np.full((len(rows), max_cols), np.nan)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            try:
                mat[i, j] = float(v)
            except Exception:
                pass
    return mat

def pareto_front_2d(F):
    order = np.argsort(F[:, 0]); F_s = F[order]; min2 = np.inf; nd = []
    for i in range(len(F_s)):
        if F_s[i, 1] < min2:
            nd.append(i); min2 = F_s[i, 1]
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

def select_axis_pool(losses, comps, archs, K_total=50, K_pf=20, K_subpf=20, K_dom=10):
    """
    Pool composition:
      K_pf:   diverse PF arches (complexity-stratified)
      K_subpf:random from Pareto layers 2-3
      K_dom:  random from dominated set (broader off-surface coverage)
    """
    layers, dominated = pareto_layers(losses, comps, n_layers=3)
    pf, sub = layers[0], np.concatenate(layers[1:]) if len(layers) > 1 else np.array([], dtype=int)

    # PF: take all if small, else complexity-diverse subset
    if len(pf) <= K_pf:
        pf_sel = pf
    else:
        sort_idx = np.argsort(comps[pf])
        pick = np.linspace(0, len(pf) - 1, K_pf, dtype=int)
        pf_sel = pf[sort_idx[pick]]

    # Sub-Pareto layers
    sub_sel = sub if len(sub) <= K_subpf else RNG.choice(sub, K_subpf, replace=False)

    # Dominated (deeper off-surface)
    dom_sel = dominated if len(dominated) <= K_dom else RNG.choice(dominated, K_dom, replace=False)

    sel = np.concatenate([pf_sel, sub_sel, dom_sel])
    sel = np.unique(sel)[:K_total]
    return sel

def load_archive(stats_path, comp_key, config, group_size):
    with open(stats_path) as f:
        data = json.load(f)
    archive = data['archive'] + data['candidates']
    archs = [v[0] for v in archive]
    losses = np.array([v[1] for v in archive])
    comps  = np.array([get_net_info(a, config, group_size)[comp_key] for a in archs])
    return archs, losses, comps

# ─── Step 1: load configuration & archives ──────────────────────────────────
print("=" * 80)
print("Loading config + local archives...")
print("=" * 80)

with open(CONFIG_PATH) as f:
    config = json.load(f)[MODEL_NAME]
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

W_archs,   W_losses,   W_comps   = load_archive(W_STATS,    'wbits',  config, group_size)
KV_archs,  KV_losses,  KV_comps  = load_archive(KV_STATS,   'kvbits', config, group_size)
KVD_archs, KVD_losses, KVD_comps = load_archive(KVDIM_STATS,'kvdim',  config, group_size)
print(f"  W archive:    n={len(W_archs)}")
print(f"  KV archive:   n={len(KV_archs)}")
print(f"  KVDIM archive:n={len(KVD_archs)}")

# Local PFs (used for the f* baseline interpolant)
pf_W   = np.column_stack([W_losses,   W_comps  ]); pf_W   = pf_W  [pareto_front_2d(pf_W  )]
pf_KV  = np.column_stack([KV_losses,  KV_comps ]); pf_KV  = pf_KV [pareto_front_2d(pf_KV )]
pf_KVD = np.column_stack([KVD_losses, KVD_comps]); pf_KVD = pf_KVD[pareto_front_2d(pf_KVD)]

# ─── Step 2: load existing AWQ training samples + fit ARD-GP ────────────────
print("\nLoading AWQ training set + fitting ARD-GP...")
mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y_all = mat3[12, :N0]; v3 = ~np.isnan(y_all)

def match_metric(comp_vals, pf):
    return np.array([pf[np.argmin(np.abs(pf[:, 1] - c)), 0] for c in comp_vals])

xW_train  = match_metric(mat3[0, :N0], pf_W  )[v3]
xKV_train = match_metric(mat3[1, :N0], pf_KV )[v3]
xKVD_tr   = match_metric(mat3[4, :N0], pf_KVD)[v3]
y_train   = y_all[v3]
X_train   = np.column_stack([xW_train, xKV_train, xKVD_tr])
N_tr = len(y_train)
print(f"  AWQ training samples: N={N_tr}")

def fit_ard_gp(X, y, n_dim=3, n_restarts=20):
    kernel = (C(1.0, (1e-4, 1e2)) *
              SKRBF(length_scale=[1.0] * n_dim, length_scale_bounds=(1e-4, 1e4)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y)
    return gp

gp = fit_ard_gp(X_train, y_train)
ls = np.array(gp.kernel_.k1.k2.length_scale)
sigma_f2 = float(gp.kernel_.k1.k1.constant_value)
sigma_n2 = float(gp.kernel_.k2.noise_level)
print(f"  ARD-GP length scales: W={ls[0]:.4f}, KV={ls[1]:.4f}, KVD={ls[2]:.4f}")
print(f"  σ_f²={sigma_f2:.4f}, σ_n²={sigma_n2:.6f}")

# ─── Step 3: per-axis pools (PF + sub-Pareto + dominated) ───────────────────
print("\n" + "=" * 80)
print("Building per-axis pools (PF + sub-Pareto + dominated)...")
print("=" * 80)
W_pool   = select_axis_pool(W_losses,   W_comps,   W_archs,   K_total=TOP_K_PER_AXIS)
KV_pool  = select_axis_pool(KV_losses,  KV_comps,  KV_archs,  K_total=TOP_K_PER_AXIS)
KVD_pool = select_axis_pool(KVD_losses, KVD_comps, KVD_archs, K_total=TOP_K_PER_AXIS)
print(f"  |W pool|     = {len(W_pool)}")
print(f"  |KV pool|    = {len(KV_pool)}")
print(f"  |KVDIM pool| = {len(KVD_pool)}")
print(f"  Cartesian pool size = {len(W_pool) * len(KV_pool) * len(KVD_pool):,}")

# ─── Step 4: build flat candidate arrays (vectorised) ───────────────────────
NW, NKV, NKVD = len(W_pool), len(KV_pool), len(KVD_pool)
i_arr, j_arr, k_arr = np.meshgrid(np.arange(NW), np.arange(NKV), np.arange(NKVD), indexing='ij')
i_flat, j_flat, k_flat = i_arr.ravel(), j_arr.ravel(), k_arr.ravel()
N_pool = len(i_flat)

xW_p   = W_losses[W_pool];   cW_p   = W_comps[W_pool]
xKV_p  = KV_losses[KV_pool]; cKV_p  = KV_comps[KV_pool]
xKVD_p = KVD_losses[KVD_pool]; cKVD_p = KVD_comps[KVD_pool]

# 3D ARD-GP input: per-arch sub-losses (extends training distribution upward)
X_cand = np.column_stack([xW_p[i_flat], xKV_p[j_flat], xKVD_p[k_flat]])
# Per-arch complexities
cW_cand   = cW_p[i_flat]
cKV_cand  = cKV_p[j_flat]
cKVD_cand = cKVD_p[k_flat]
eff_kvbits_cand = cKV_cand * cKVD_cand / HEAD_DIM   # effective kv bits if pruning used
# 1D summary complexity (used for bucketing + PF interpolation)
total_c_cand = cW_cand + eff_kvbits_cand

# ─── Step 5: ARD-GP batch prediction (μ, σ) ─────────────────────────────────
print("\n" + "=" * 80)
print(f"ARD-GP prediction on {N_pool:,} candidates...")
print("=" * 80)
mu_cand    = np.empty(N_pool)
sigma_cand = np.empty(N_pool)
batch = 50000
for s in range(0, N_pool, batch):
    e = min(s + batch, N_pool)
    mu_cand[s:e], sigma_cand[s:e] = gp.predict(X_cand[s:e], return_std=True)
print(f"  μ:     [{mu_cand.min():.4f}, {mu_cand.max():.4f}]")
print(f"  σ:     [{sigma_cand.min():.4f}, {sigma_cand.max():.4f}]")

# ─── Step 6: Cartesian PF interpolant f*(total_c) ───────────────────────────
# Identify which pool axes are on the local PF
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
off_mask  = ~cart_mask
print(f"\nPool composition:")
print(f"  Cartesian-PF tuples: {cart_mask.sum():,}")
print(f"  Off-surface:         {off_mask.sum():,}")

# Build PF interpolant from Cartesian-PF predicted (μ, total_c)
mu_cart = mu_cand[cart_mask]; tc_cart = total_c_cand[cart_mask]
F_pf = np.column_stack([mu_cart, tc_cart])
front = pareto_front_2d(F_pf)
pf_x = np.sort(tc_cart[front])
pf_y = mu_cart[front][np.argsort(tc_cart[front])]
# Make monotone non-increasing in y (PF property): right-to-left running min
pf_y_mono = np.minimum.accumulate(pf_y[::-1])[::-1]

def pf_interp(c):
    return np.interp(c, pf_x, pf_y_mono, left=pf_y_mono[0], right=pf_y_mono[-1])

f_star_cand = pf_interp(total_c_cand)

# ─── Step 7: acquisition (P(violator)) + stratified selection ───────────────
# Score: probability that actual y < f*(c) under ARD-GP posterior.
#   prob_violator = Φ(gap / σ)  where gap = f* − μ
# Higher score ⇒ more likely PF violator under predictor uncertainty.
# (EI ranks by σ φ(z) when gap << 0, picking far-from-PF high-σ points.
#  P(violator) instead picks the "closest-to-PF, high-σ" points — exactly
#  what falsification wants.)
gap_cand = f_star_cand - mu_cand   # predicted PF improvement (− if μ above PF)
sigma_safe = np.maximum(sigma_cand, 1e-6)
prob_violator = norm.cdf(gap_cand / sigma_safe)
ucb_cand = gap_cand + 2.0 * sigma_cand   # 95% UCB for "is a violator"

# Also classical EI for reporting only
def expected_improvement(mu, sigma, f_star):
    sigma = np.maximum(sigma, 1e-12)
    z = (f_star - mu) / sigma
    return (f_star - mu) * norm.cdf(z) + sigma * norm.pdf(z)
ei_cand = expected_improvement(mu_cand, sigma_cand, f_star_cand)

print("\n" + "=" * 80)
print("Acquisition (off-surface only)...")
print("=" * 80)
off_idx_arr = np.where(off_mask)[0]
print(f"  gap (f*−μ): max={gap_cand[off_idx_arr].max():+.5f}, "
      f"mean={gap_cand[off_idx_arr].mean():+.5f}")
print(f"  P(violator): max={prob_violator[off_idx_arr].max():.4f}, "
      f"mean={prob_violator[off_idx_arr].mean():.4f}, "
      f"P>5% count={int(np.sum(prob_violator[off_idx_arr] > 0.05))}, "
      f"P>10% count={int(np.sum(prob_violator[off_idx_arr] > 0.10))}")
print(f"  UCB (gap+2σ): max={ucb_cand[off_idx_arr].max():+.5f}, "
      f"%>0={100*np.mean(ucb_cand[off_idx_arr] > 0):.1f}%")

# 7a. Stratified P(violator): N_BUCKETS × N_PER_BUCKET (the main selection)
print(f"\n  Stratified P(violator): {N_BUCKETS} buckets × top-{N_PER_BUCKET}")
edges = np.quantile(total_c_cand[off_mask], np.linspace(0, 1, N_BUCKETS + 1))
edges[0] -= 1e-9; edges[-1] += 1e-9
sel_main = []
for b in range(N_BUCKETS):
    in_b = off_idx_arr[(total_c_cand[off_idx_arr] >= edges[b]) &
                       (total_c_cand[off_idx_arr] <  edges[b + 1])]
    if len(in_b) == 0: continue
    sel_main.extend(in_b[np.argsort(-prob_violator[in_b])[:N_PER_BUCKET]].tolist())
sel_main = list(dict.fromkeys(sel_main))
print(f"    -> {len(sel_main)} unique candidates")

# 7b. High-σ extras restricted to plausible-violator candidates (P > 1%)
#     so we don't waste budget on far-from-PF high-σ points.
plausible = off_idx_arr[prob_violator[off_idx_arr] > 0.01]
remaining = np.setdiff1d(plausible, np.array(sel_main, dtype=int))
n_extra = N_BUDGET - len(sel_main)
extras = remaining[np.argsort(-sigma_cand[remaining])[:n_extra]].tolist()
# Fallback: if we don't have enough plausible high-σ, fill with top-prob from all off-surface
if len(extras) < n_extra:
    extra_remaining = np.setdiff1d(off_idx_arr,
                                   np.array(sel_main + extras, dtype=int))
    extra_fill = extra_remaining[np.argsort(-prob_violator[extra_remaining])
                                  [:n_extra - len(extras)]].tolist()
    extras = extras + extra_fill
print(f"  High-σ extras (P>1%): {len(extras)}")
selected = sel_main + extras
selected = list(dict.fromkeys(selected))[:N_BUDGET]
print(f"  Total selected: {len(selected)} (target {N_BUDGET})")
sel_ei = sel_main  # alias for downstream record-keeping

# ─── Step 8: build merged arch dicts ────────────────────────────────────────
print("\n" + "=" * 80)
print("Building merged arch dicts...")
print("=" * 80)

# default_arch (used by build_arch as the base; we override all three axes)
n_block = config['n_block']; w_linears = config['linear']
default_arch = {
    'q': {
        'w': {linear: [4] * n_block for linear in w_linears},
        'k': [[4, 128]] * n_block,
        'v': [[4, 128]] * n_block,
    },
    'p': {'k': [0] * n_block, 'v': [0] * n_block},
}

# Per-arch axis arrays for build_arch
esm = {
    'w':     np.array([W_archs[i]   for i in W_pool],   dtype=object),
    'kv':    np.array([KV_archs[i]  for i in KV_pool],  dtype=object),
    'kvdim': np.array([KVD_archs[i] for i in KVD_pool], dtype=object),
}
expr_keys = ['w', 'kv', 'kvdim']

selected_archs, records = [], []
for s, idx in enumerate(selected):
    iw, jkv, kkvd = int(i_flat[idx]), int(j_flat[idx]), int(k_flat[idx])
    nd_idx_row = np.array([iw, jkv, kkvd])
    arch = build_arch(default_arch, expr_keys, esm, nd_idx_row)
    info = get_net_info(arch, config, group_size)
    rec = {
        'sel_idx':   int(s),
        'pool_idx':  int(idx),
        'src':       'P(violator)' if s < len(sel_main) else 'sigma|P>1%',
        'pred_mu':   float(mu_cand[idx]),
        'pred_sigma':float(sigma_cand[idx]),
        'pf_baseline':float(f_star_cand[idx]),
        'pred_gap':  float(gap_cand[idx]),
        'prob_violator': float(prob_violator[idx]),
        'ucb':       float(ucb_cand[idx]),
        'ei':        float(ei_cand[idx]),
        'wbits':     float(info['wbits']),
        'kvbits':    float(info.get('kvbits', np.nan)),
        'kvdim':     float(info.get('kvdim', 0)),
        'eff_kvbits':float(info.get('eff_kvbits', info.get('kvbits', 0))),
        'total_c':   float(total_c_cand[idx]),
        'on_cart_pf':bool(cart_mask[idx]),
        'sub_idx':   {'w': iw, 'kv': jkv, 'kvdim': kkvd},
    }
    selected_archs.append(arch)
    records.append(rec)

# ─── Step 9: save outputs ───────────────────────────────────────────────────
out_json = f'{OUT_DIR}/acquired_offsurface_100.json'
with open(out_json, 'w') as f:
    json.dump({'archs': selected_archs, 'records': records,
               'config': {'model_name': MODEL_NAME, 'group_size': group_size,
                          'n_budget': N_BUDGET, 'n_buckets': N_BUCKETS,
                          'n_per_bucket': N_PER_BUCKET,
                          'top_k_per_axis': TOP_K_PER_AXIS,
                          'pool_size': int(N_pool),
                          'off_surface_pool': int(off_mask.sum()),
                          'cartesian_pool':   int(cart_mask.sum()),
                          'gp_length_scales': ls.tolist(),
                          'gp_sigma_f2': sigma_f2, 'gp_sigma_n2': sigma_n2}},
              f, indent=2,
              default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
print(f"\nSaved arch list → {out_json}")

# Selection summary
gaps = np.array([r['pred_gap'] for r in records])
sigmas = np.array([r['pred_sigma'] for r in records])
probs = np.array([r['prob_violator'] for r in records])
totals = np.array([r['total_c'] for r in records])
src_counts = {'P(violator)': sum(1 for r in records if r['src'] == 'P(violator)'),
              'sigma|P>1%':  sum(1 for r in records if r['src'] == 'sigma|P>1%')}
n_predicted_violator     = int(np.sum(gaps > 0.01))
n_predicted_violator_005 = int(np.sum(gaps > 0.005))
n_high_prob_5  = int(np.sum(probs > 0.05))
n_high_prob_10 = int(np.sum(probs > 0.10))

summary = []
summary.append(f"Off-surface acquisition for AWQ falsification (n={len(records)})")
summary.append(f"=" * 70)
summary.append(f"")
summary.append(f"Sources:")
summary.append(f"  W archive:    {len(W_archs)}, top-{TOP_K_PER_AXIS} pooled")
summary.append(f"  KV archive:   {len(KV_archs)}, top-{TOP_K_PER_AXIS} pooled")
summary.append(f"  KVDIM archive:{len(KVD_archs)}, top-{TOP_K_PER_AXIS} pooled")
summary.append(f"  AWQ train:    {N_tr} samples for ARD-GP fit")
summary.append(f"")
summary.append(f"ARD-GP:")
summary.append(f"  length scales: W={ls[0]:.4f}  KV={ls[1]:.4f}  KVD={ls[2]:.4f}")
summary.append(f"  σ_f²={sigma_f2:.4f}, σ_n²={sigma_n2:.6f}")
summary.append(f"")
summary.append(f"Pool: {N_pool:,}  (Cartesian {cart_mask.sum():,} / off-surface {off_mask.sum():,})")
summary.append(f"")
summary.append(f"Selection:")
summary.append(f"  P(violator)-stratified: {src_counts['P(violator)']}  "
               f"({N_BUCKETS} buckets × top-{N_PER_BUCKET})")
summary.append(f"  σ-extras (P>1%):        {src_counts['sigma|P>1%']}")
summary.append(f"")
summary.append(f"P(violator) = Φ((f*−μ)/σ)   (under ARD-GP posterior):")
summary.append(f"  max:               {probs.max():.4f}")
summary.append(f"  mean:              {probs.mean():.4f}")
summary.append(f"  n_P > 0.05:        {n_high_prob_5}/{len(records)}")
summary.append(f"  n_P > 0.10:        {n_high_prob_10}/{len(records)}")
summary.append(f"")
summary.append(f"Predicted PF gap (= f*(c) − μ):")
summary.append(f"  max:               {gaps.max():+.5f}")
summary.append(f"  mean:              {gaps.mean():+.5f}")
summary.append(f"  n_pred_gap > 0.005:{n_predicted_violator_005}/{len(records)}")
summary.append(f"  n_pred_gap > 0.01: {n_predicted_violator}/{len(records)}")
summary.append(f"")
summary.append(f"Sigma:")
summary.append(f"  σ mean: {sigmas.mean():.4f},  σ max: {sigmas.max():.4f}")
summary.append(f"")
summary.append(f"Complexity coverage (wbits + eff_kvbits):")
summary.append(f"  range:       [{totals.min():.3f}, {totals.max():.3f}]")
summary.append(f"  q10/q50/q90: {np.quantile(totals,0.1):.3f} / "
               f"{np.quantile(totals,0.5):.3f} / {np.quantile(totals,0.9):.3f}")
summary.append(f"")
summary.append(f"How to evaluate (downstream):")
summary.append(f"  1. Read archs from {out_json}")
summary.append(f"  2. Feed into post_search_split.py-style AWQ eval loop:")
summary.append(f"     for arch in archs:")
summary.append(f"         model = evaluator.sample(arch)")
summary.append(f"         metric = evaluator.eval(arch=arch, model=model, ...)")
summary.append(f"  3. Compare measured loss vs record['pf_baseline'] for ε-violation")

out_txt = f'{OUT_DIR}/acquired_offsurface_100.txt'
with open(out_txt, 'w') as f:
    f.write('\n'.join(summary))
print(f"Saved summary → {out_txt}")
print()
print('\n'.join(summary))

# ─── Step 10: diagnostic figure ─────────────────────────────────────────────
sel_arr   = np.array(selected, dtype=int)
main_set  = set(sel_main)
sigma_set = set(extras)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) Selected vs Cartesian PF in (total_c, μ)
ax = axes[0, 0]
sub = RNG.choice(np.where(off_mask)[0], min(20000, off_mask.sum()), replace=False)
ax.scatter(total_c_cand[sub], mu_cand[sub], s=3, alpha=0.18, c='lightgray',
           label='off-surface pool (sub)')
ax.scatter(total_c_cand[cart_mask], mu_cand[cart_mask], s=4, alpha=0.55,
           c='steelblue', label='Cartesian PF tuples')
mask_main = np.isin(sel_arr, list(main_set))
mask_sig  = np.isin(sel_arr, list(sigma_set))
ax.scatter(total_c_cand[sel_arr[mask_main]], mu_cand[sel_arr[mask_main]], s=22, c='red',
           edgecolor='black', linewidth=0.3, label=f'P(violator) ({mask_main.sum()})')
ax.scatter(total_c_cand[sel_arr[mask_sig]],  mu_cand[sel_arr[mask_sig]],  s=22, c='orange',
           edgecolor='black', linewidth=0.3, label=f'σ-extras ({mask_sig.sum()})')
ax.plot(pf_x, pf_y_mono, 'k-', lw=1.5, label='Cartesian PF interp f*(c)')
ax.set_xlabel('wbits + eff_kvbits'); ax.set_ylabel('ARD-GP predicted μ')
ax.set_title('(a) Selection vs Cartesian-PF baseline', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (b) P(violator) distribution
ax = axes[0, 1]
ax.hist(prob_violator[off_mask], bins=60, alpha=0.4, color='gray',
        label='off-surface pool')
ax.hist(prob_violator[sel_arr], bins=25, alpha=0.85, color='red',
        label='selected')
ax.axvline(0.05, color='black', ls='--', lw=0.8, label='P=5%')
ax.axvline(0.10, color='black', ls=':', lw=0.8, label='P=10%')
ax.set_xlabel('P(violator) = Φ((f*−μ)/σ)')
ax.set_ylabel('count'); ax.set_yscale('log')
ax.set_title('(b) P(violator) — selected vs pool', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (c) σ vs predicted gap, colored by selection
ax = axes[1, 0]
ax.scatter(gap_cand[sub], sigma_cand[sub], s=3, alpha=0.25, c='lightgray',
           label='off-surface (sub)')
ax.scatter(gap_cand[sel_arr[mask_main]], sigma_cand[sel_arr[mask_main]], s=22, c='red',
           edgecolor='black', linewidth=0.3, label='P(violator)')
ax.scatter(gap_cand[sel_arr[mask_sig]],  sigma_cand[sel_arr[mask_sig]],  s=22, c='orange',
           edgecolor='black', linewidth=0.3, label='σ-extras')
# overlay iso-P contours: P = const ⇔ gap = Φ⁻¹(P) · σ (lines through origin)
sig_grid = np.linspace(sigma_cand.min(), sigma_cand.max(), 50)
for P, ls_ in [(0.05, ':'), (0.10, '-.'), (0.25, '--')]:
    ax.plot(norm.ppf(P) * sig_grid, sig_grid, color='black', ls=ls_, lw=0.7,
            label=f'P={P}')
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('predicted gap (f*−μ)  [+ ⇒ predicted PF violator]')
ax.set_ylabel('σ')
ax.set_title('(c) Acquisition landscape (iso-P lines)', fontweight='bold')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

# (d) Bucket coverage
ax = axes[1, 1]
ax.hist(total_c_cand[sel_arr], bins=N_BUCKETS, color='red', alpha=0.7,
        edgecolor='black')
for ed in edges[1:-1]:
    ax.axvline(ed, color='gray', lw=0.4, alpha=0.6)
ax.set_xlabel('wbits + eff_kvbits'); ax.set_ylabel('# selected')
ax.set_title(f'(d) Coverage across {N_BUCKETS} complexity buckets', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Off-surface acquisition for AWQ falsification (n={len(records)})',
             fontweight='bold', y=1.0)
plt.tight_layout()
fig_path = f'{FIG_DIR}/05_acquire_overview.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nSaved figure → {fig_path}")
print("Done.")
