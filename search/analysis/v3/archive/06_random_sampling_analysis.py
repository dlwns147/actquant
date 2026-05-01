"""06_random_sampling_analysis.py — Why does random sampling sometimes hit R²≈0.99?

Run 100 random seeds at N_train ∈ {27, 30, 35} for ARD-GP.
Record sample composition + R² per seed.
Identify what distinguishes "lucky" random samples (R² > 0.985) from "unlucky" ones.

Design metrics computed per seed:
  • Per-dim coverage:  ratio (max - min) / range_full
  • Per-dim std:       sample std (informativeness)
  • Quantile-cell hits: of the 27 cells [0.1,0.5,0.9]³ how many have ≥1 sample
  • Extreme-corner hits: # samples in extreme W (top/bot 10% of W)
  • Distance to structured 27-grid samples (mean min-dist)
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

# ─── Data loading ────────────────────────────────────────────────────────────
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
AWQ_3WAY = f'{BASE}/save/result/awq/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'
with open(f'{BASE}/config/llama.json') as f: config = json.load(f)['Llama-3.1-8B-Instruct']
group_size = {'w': 128, 'k': [128, 128], 'v': [128, 128]}

print("Loading...")
pf_W     = load_archive_pareto(W_STATS,    'wbits',  config, group_size)
pf_KV    = load_archive_pareto(KV_STATS,   'kvbits', config, group_size)
pf_KVDIM = load_archive_pareto(KVDIM_STATS,'kvdim',  config, group_size)
mat3 = load_csv(AWQ_3WAY); N0 = mat3.shape[1]
y3   = mat3[12, :N0]; v3 = ~np.isnan(y3)
xW3  = match_metric(mat3[0, :N0], pf_W   )[v3]
xKV3 = match_metric(mat3[1, :N0], pf_KV  )[v3]
xKVD = match_metric(mat3[4, :N0], pf_KVDIM)[v3]
y3   = y3[v3]; N = len(y3)
X3 = np.column_stack([xW3, xKV3, xKVD])

range_jsd = np.array([np.ptp(xW3), np.ptp(xKV3), np.ptp(xKVD)])
mins_jsd  = np.array([xW3.min(), xKV3.min(), xKVD.min()])

# Structured 27-grid + 23 maximin (for distance reference)
qs = [0.1, 0.5, 0.9]
qW_  = np.quantile(xW3, qs); qKV_ = np.quantile(xKV3, qs); qKVD_ = np.quantile(xKVD, qs)
grid27 = np.array([[w,kv,kvd] for w in qW_ for kv in qKV_ for kvd in qKVD_])
scale  = X3.std(0) + 1e-10
X3n    = X3 / scale
grid_n = grid27 / scale
cost   = np.zeros((27, N))
for j in range(27): cost[j] = np.sum((X3n - grid_n[j])**2, axis=1)
_, col_ind = linear_sum_assignment(cost)
struct_grid_idx = col_ind.astype(int)

def fit_ard_gp(X, y, n_restarts=8):
    kernel = (C(1.0, (1e-4, 1e2)) *
              SKRBF(length_scale=[1.0]*X.shape[1], length_scale_bounds=(1e-3, 1e3)) +
              WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, alpha=1e-8)
    gp.fit(X, y); return gp

def r2(y, yp): return 1 - np.sum((y-yp)**2) / max(np.sum((y-y.mean())**2), 1e-30)

def design_metrics(idx):
    """Compute design quality metrics for sample indices."""
    pts = X3[idx]
    pts_n = pts / scale
    # Per-dim coverage / std
    cov = np.array([(pts[:, d].max() - pts[:, d].min()) / range_jsd[d] for d in range(3)])
    stds = pts.std(axis=0) / scale[0:3]  # normalized std
    # Quantile cell hits (3³ = 27 cells)
    q_low  = np.array([np.quantile(xW3, 1/3),  np.quantile(xKV3, 1/3),  np.quantile(xKVD, 1/3)])
    q_high = np.array([np.quantile(xW3, 2/3),  np.quantile(xKV3, 2/3),  np.quantile(xKVD, 2/3)])
    cells = set()
    for p in pts:
        cell = tuple((p[d] >= q_low[d]) + (p[d] >= q_high[d]) for d in range(3))
        cells.add(cell)
    cell_hit = len(cells)  # max 27
    # Extreme-W corner inclusion (top/bot 10% of W)
    w_lo = np.quantile(xW3, 0.10); w_hi = np.quantile(xW3, 0.90)
    n_w_lo = int((pts[:, 0] <= w_lo).sum())
    n_w_hi = int((pts[:, 0] >= w_hi).sum())
    # Mean min-distance to structured grid samples (in normalized space)
    grid_pts_n = X3[struct_grid_idx] / scale
    dmin = np.mean([np.min(np.linalg.norm(grid_pts_n - p, axis=1)) for p in pts_n])
    return dict(cov=cov, std=stds, cell_hit=cell_hit,
                n_w_lo=n_w_lo, n_w_hi=n_w_hi, dmin_to_grid=dmin)

# ─── Run experiments ─────────────────────────────────────────────────────────
N_TRS = [27, 30, 35]
N_SEEDS = 100
print(f"\nRunning {N_SEEDS} random seeds × {len(N_TRS)} N_train values...")

all_results = {}  # (N_tr) -> list of (seed, idx, r2, metrics)
for N_tr in N_TRS:
    print(f"\n  N_train = {N_tr}")
    results = []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, N_tr, replace=False)
        te_idx = np.setdiff1d(np.arange(N), idx)
        try:
            gp = fit_ard_gp(X3[idx], y3[idx], n_restarts=5)
            yp = gp.predict(X3[te_idx])
            r2_v = r2(y3[te_idx], yp)
            results.append((seed, idx, r2_v, design_metrics(idx)))
        except Exception:
            continue
    all_results[N_tr] = results
    r2s = [r[2] for r in results]
    print(f"    R² distribution: median={np.median(r2s):.4f}, mean={np.mean(r2s):.4f}")
    print(f"    Top: max={max(r2s):.4f}, P90={np.percentile(r2s, 90):.4f}, P95={np.percentile(r2s, 95):.4f}")
    print(f"    Bot: min={min(r2s):.4f}, P10={np.percentile(r2s, 10):.4f}")

# ─── Analysis: top vs bottom seeds ───────────────────────────────────────────
print("\n" + "="*100)
print("DESIGN METRICS: top-10 vs bottom-10 random seeds")
print("="*100)

for N_tr in N_TRS:
    results = sorted(all_results[N_tr], key=lambda x: x[2])  # asc by R²
    bot10 = results[:10]; top10 = results[-10:]

    def avg_metric(grp, key):
        return np.mean([r[3][key] for r in grp], axis=0) if isinstance(grp[0][3][key], np.ndarray) else \
               np.mean([r[3][key] for r in grp])

    print(f"\n  N_train = {N_tr}:")
    print(f"  {'metric':30s}  {'TOP-10 avg':18s}  {'BOT-10 avg':18s}  {'top R² range':18s}")
    print(f"  {'cov_W (W coverage ratio)':30s}  {avg_metric(top10,'cov')[0]:.3f}              {avg_metric(bot10,'cov')[0]:.3f}              {top10[0][2]:.3f}~{top10[-1][2]:.3f}")
    print(f"  {'cov_KV':30s}  {avg_metric(top10,'cov')[1]:.3f}              {avg_metric(bot10,'cov')[1]:.3f}")
    print(f"  {'cov_KVD':30s}  {avg_metric(top10,'cov')[2]:.3f}              {avg_metric(bot10,'cov')[2]:.3f}")
    print(f"  {'std_W (norm)':30s}  {avg_metric(top10,'std')[0]:.3f}              {avg_metric(bot10,'std')[0]:.3f}")
    print(f"  {'std_KV (norm)':30s}  {avg_metric(top10,'std')[1]:.3f}              {avg_metric(bot10,'std')[1]:.3f}")
    print(f"  {'std_KVD (norm)':30s}  {avg_metric(top10,'std')[2]:.3f}              {avg_metric(bot10,'std')[2]:.3f}")
    print(f"  {'cell_hit (max 27)':30s}  {avg_metric(top10,'cell_hit'):.1f}                {avg_metric(bot10,'cell_hit'):.1f}")
    print(f"  {'n_w_low (W bot-10%)':30s}  {avg_metric(top10,'n_w_lo'):.1f}                 {avg_metric(bot10,'n_w_lo'):.1f}")
    print(f"  {'n_w_high (W top-10%)':30s}  {avg_metric(top10,'n_w_hi'):.1f}                 {avg_metric(bot10,'n_w_hi'):.1f}")
    print(f"  {'dmin_to_struct_grid (norm)':30s}  {avg_metric(top10,'dmin_to_grid'):.3f}              {avg_metric(bot10,'dmin_to_grid'):.3f}")

# ─── Predictor of success: correlation between metrics and R² ────────────────
print("\n" + "="*100)
print("Pearson correlation of design metrics with R² (across all 100 seeds per N_tr)")
print("="*100)
from scipy.stats import pearsonr

for N_tr in N_TRS:
    results = all_results[N_tr]
    r2s = np.array([r[2] for r in results])
    print(f"\n  N_train = {N_tr}:")
    for key in ['cov_W', 'cov_KV', 'cov_KVD', 'std_W', 'std_KV', 'std_KVD',
                'cell_hit', 'n_w_lo', 'n_w_hi', 'dmin_to_grid']:
        if key.startswith('cov_'):
            d = key[-1]; di = {'W':0,'V':1,'D':2}[d] if d in 'W' else (1 if d=='V' else 2)
            di = {'W':0, 'V':1, 'D':2}.get(d, 0)
            vals = np.array([r[3]['cov'][di if d=='W' else (1 if 'KV' in key else 2)] for r in results])
        elif key.startswith('std_'):
            di = 0 if key=='std_W' else (1 if key=='std_KV' else 2)
            vals = np.array([r[3]['std'][di] for r in results])
        else:
            vals = np.array([r[3][key] for r in results])
        try:
            rho, p = pearsonr(vals, r2s)
            sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))
            print(f"    {key:18s}: r = {rho:+.3f}  (p={p:.4f}) {sig}")
        except Exception:
            pass

# ─── Figures ─────────────────────────────────────────────────────────────────
FIG_DIR = f'{BASE}/analysis/v3/figures'
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
PLT_KW = dict(dpi=170, bbox_inches='tight')

# Fig 1: R² histogram + design metric scatter
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for col, N_tr in enumerate(N_TRS):
    results = all_results[N_tr]
    r2s = np.array([r[2] for r in results])
    cov_W = np.array([r[3]['cov'][0] for r in results])
    cell_hit = np.array([r[3]['cell_hit'] for r in results])

    # Top: histogram
    ax = axes[0, col]
    ax.hist(r2s, bins=25, color='#4DBBD5', alpha=0.85, edgecolor='#333')
    ax.axvline(np.median(r2s), color='black', ls='--', lw=1.5, label=f'median={np.median(r2s):.4f}')
    ax.axvline(np.percentile(r2s, 90), color='#E64B35', ls='-', lw=1.2, label=f'P90={np.percentile(r2s,90):.4f}')
    ax.set_xlabel('Test R² (random sample)'); ax.set_ylabel('# seeds')
    ax.set_title(f'N_train = {N_tr}', fontweight='bold'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, lw=0.5)

    # Bot: scatter R² vs cov_W (key metric)
    ax = axes[1, col]
    sc = ax.scatter(cov_W, r2s, c=cell_hit, cmap='viridis', s=18, alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('W coverage = (max W − min W) / range_W')
    ax.set_ylabel('Test R²')
    ax.set_title(f'R² vs W-coverage  (color = cell_hits)', fontweight='bold')
    plt.colorbar(sc, ax=ax, fraction=0.04)
    ax.grid(True, alpha=0.25, lw=0.5)

plt.suptitle('Random sampling at small N_train: R² distribution + design quality predictors',
             fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_random_sampling_R2_dist.png', **PLT_KW); plt.close()

# Fig 2: 3D scatter showing top-3 vs bot-3 sample distributions for N=27
N_show = 27
results = sorted(all_results[N_show], key=lambda x: x[2])
top3 = results[-3:]; bot3 = results[:3]

fig, axes = plt.subplots(2, 3, figsize=(15, 9), subplot_kw={'projection': '3d'})
for col, (label, group) in enumerate([('TOP', top3[::-1]), ('BOT', bot3)]):
    pass
for row, (label, group) in enumerate([('TOP-3', top3[::-1]), ('BOT-3', bot3)]):
    for col, (seed, idx, r2v, _) in enumerate(group):
        ax = axes[row, col]
        # All test points faded
        te = np.setdiff1d(np.arange(N), idx)
        ax.scatter(X3[te, 0], X3[te, 1], X3[te, 2], c='lightgray', s=4, alpha=0.3)
        ax.scatter(X3[idx, 0], X3[idx, 1], X3[idx, 2],
                   c=('#E64B35' if 'TOP' in label else '#3C5488'),
                   s=42, edgecolor='black', linewidth=0.4)
        ax.set_xlabel('JSD_W'); ax.set_ylabel('JSD_KV'); ax.set_zlabel('JSD_KVD')
        ax.set_title(f'{label} seed={seed}\nR²={r2v:.4f}', fontweight='bold', fontsize=10)
        ax.view_init(elev=18, azim=45)
plt.suptitle(f'Top vs bottom random samples at N_train={N_show}\n(top: R²≈0.99 lucky picks; bot: R²<0.95)',
             fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_random_sampling_top_vs_bot.png', **PLT_KW); plt.close()

# Fig 3: Compare structured vs top random vs bot random
fig, ax = plt.subplots(figsize=(11, 5))
# Structured 27-grid base (deterministic baseline)
ax.scatter(X3[:, 0], X3[:, 1], c='lightgray', s=10, alpha=0.4, label='all 200 samples')
ax.scatter(X3[struct_grid_idx, 0], X3[struct_grid_idx, 1], c='#9B59B6', s=50, marker='*',
           edgecolor='black', linewidth=0.5, label='structured 27-grid (R²=0.964)')
top_seed_idx = top3[-1][1]
ax.scatter(X3[top_seed_idx, 0], X3[top_seed_idx, 1], c='#E64B35', s=42,
           marker='^', edgecolor='black', linewidth=0.4,
           label=f'top random (seed={top3[-1][0]}, R²={top3[-1][2]:.3f})')
bot_seed_idx = bot3[0][1]
ax.scatter(X3[bot_seed_idx, 0], X3[bot_seed_idx, 1], c='#3C5488', s=42,
           marker='s', edgecolor='black', linewidth=0.4,
           label=f'bot random (seed={bot3[0][0]}, R²={bot3[0][2]:.3f})')
ax.set_xlabel('JSD_W'); ax.set_ylabel('JSD_KV')
ax.set_title('JSD_W × JSD_KV projection: structured vs top/bot random', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25, lw=0.5)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_random_sampling_layouts.png', **PLT_KW); plt.close()

print(f"\nFigures saved:")
print(f"  06_random_sampling_R2_dist.png      — R² histogram + design predictor scatter")
print(f"  06_random_sampling_top_vs_bot.png   — Top-3 vs bot-3 3D layouts")
print(f"  06_random_sampling_layouts.png      — Structured vs top/bot 2D projection")
print("Done.\n")
