"""
Grid 27 analysis:
  - Sample 3 points per method at quantiles [0.1, 0.5, 0.9]
  - Form 3x3x3 = 27 combinations
  - Fit full-quadratic + all-interactions model on these 27 points
  - Evaluate held-out prediction quality
  - Compare vs random-27, structured grid robustness
"""
import os, csv, numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

BASE = '/NAS/SJ/actquant/search/save/result'
F3 = f'{BASE}/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv'

def load(fp):
    data = []
    with open(fp) as f:
        for row in csv.reader(f):
            if row and any(x.strip() for x in row):
                try: data.append([float(x) for x in row])
                except ValueError: pass
    return pd.DataFrame(dict(
        w=np.array(data[0]),
        kv=np.array(data[1]),
        kvd=np.array(data[4]),
        y=np.array(data[12]),
    ))

df = load(F3).dropna().reset_index(drop=True)
print("="*92)
print(f"SETUP: 3-way AWQ samples  n={len(df)}")
print(f"  wbits   : [{df.w.min():.3f}, {df.w.max():.3f}] μ={df.w.mean():.3f}")
print(f"  kvbits  : [{df.kv.min():.3f}, {df.kv.max():.3f}] μ={df.kv.mean():.3f}")
print(f"  kvdim   : [{df.kvd.min():.1f}, {df.kvd.max():.1f}] μ={df.kvd.mean():.1f}")
print(f"  JSD     : [{df.y.min():.4f}, {df.y.max():.4f}] μ={df.y.mean():.4f}")
print("="*92)

# ------------------------------------------------------------------------
# PART 1: 27 GRID POINTS AT QUANTILES [0.1, 0.5, 0.9]
# ------------------------------------------------------------------------
Q = [0.1, 0.5, 0.9]
qw  = np.quantile(df.w,   Q)
qkv = np.quantile(df.kv,  Q)
qkd = np.quantile(df.kvd, Q)

print(f"\n  Per-method quantile anchors:")
print(f"    wbits   Q10/50/90: {qw[0]:.3f} / {qw[1]:.3f} / {qw[2]:.3f}")
print(f"    kvbits  Q10/50/90: {qkv[0]:.3f} / {qkv[1]:.3f} / {qkv[2]:.3f}")
print(f"    kvdim   Q10/50/90: {qkd[0]:.1f} / {qkd[1]:.1f} / {qkd[2]:.1f}")

# For each of the 27 target grid points, find nearest 3-way sample in 200-set.
# Distance normalized by per-method std.
sw, skv, skd = df.w.std(), df.kv.std(), df.kvd.std()
grid = []
nearest_idx = []
for w in qw:
    for kv in qkv:
        for kd in qkd:
            d = ((df.w - w)/sw)**2 + ((df.kv - kv)/skv)**2 + ((df.kvd - kd)/skd)**2
            i = int(np.argmin(d.values))
            grid.append((w, kv, kd))
            nearest_idx.append(i)

grid_df = df.iloc[nearest_idx].reset_index(drop=True)
print(f"\n  27 grid anchor points (nearest 3-way sample picked per anchor):")
print(f"  {'anchor (w,kv,kvd)':<32}{'nearest (w,kv,kvd)':<32}{'JSD':>10}")
for (w,kv,kd), i in zip(grid, nearest_idx):
    row = df.iloc[i]
    print(f"   ({w:5.2f}, {kv:5.2f}, {kd:5.1f})        ({row.w:5.2f}, {row.kv:5.2f}, {row.kvd:5.1f})        {row.y:.4f}")

# Dedup check: nearest-neighbor can collapse anchors
unique_idx = set(nearest_idx)
print(f"\n  Unique nearest samples: {len(unique_idx)} / 27")

# ------------------------------------------------------------------------
# PART 2: FIT FULL-QUAD + ALL-INTERACTIONS ON 27 GRID POINTS
# ------------------------------------------------------------------------
def design(df_, cols=('w','kv','kvd')):
    xs = [df_[c].values for c in cols]
    X = []
    for v in xs: X += [v, v**2]
    X += [xs[0]*xs[1], xs[0]*xs[2], xs[1]*xs[2]]
    return np.c_[np.ones(len(df_)), np.array(X).T]

def fit_eval(train_df, test_df):
    Xtr = design(train_df); ytr = train_df.y.values
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    Xte = design(test_df); yte = test_df.y.values
    yhat = Xte @ beta
    sst = ((yte - yte.mean())**2).sum()
    R2 = 1 - ((yte - yhat)**2).sum()/sst if sst>0 else np.nan
    rho = stats.spearmanr(yte, yhat)[0]
    mae = np.mean(np.abs(yte - yhat))
    rmse = np.sqrt(np.mean((yte - yhat)**2))
    return beta, yhat, R2, rho, mae, rmse

# Training fit on the 27 anchors
beta, yhat27, R2_train, rho_train, mae_train, rmse_train = fit_eval(grid_df, grid_df)
print(f"\n  Training fit on 27 grid points:")
print(f"    R² = {R2_train:.4f},  Spearman ρ = {rho_train:.4f},  MAE = {mae_train:.4f}")

print("\n  Coefficients (y = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + Σβᵢⱼxᵢxⱼ):")
names = ['β0(intercept)','β_w','β_w²','β_kv','β_kv²','β_kvd','β_kvd²',
         'β_w·kv','β_w·kvd','β_kv·kvd']
for nm, b in zip(names, beta):
    print(f"    {nm:<18} = {b:+.6f}")

# Held-out: all 3-way samples NOT picked
held_mask = np.ones(len(df), bool)
for i in set(nearest_idx):
    held_mask[i] = False
held_df = df[held_mask].reset_index(drop=True)
_, yhat_held, R2_held, rho_held, mae_held, rmse_held = fit_eval(grid_df, held_df)
print(f"\n  Held-out evaluation on {len(held_df)} remaining 3-way samples:")
print(f"    R² = {R2_held:.4f},  Spearman ρ = {rho_held:.4f},  MAE = {mae_held:.4f},  RMSE = {rmse_held:.4f}")

# Low-wbits subset
low = held_df[held_df.w < 2.75]
hi  = held_df[held_df.w >= 2.75]
if len(low):
    _, _, R2_low, rho_low, mae_low, _ = fit_eval(grid_df, low)
    print(f"    low-wbits (<2.75, n={len(low)}):  R²={R2_low:.4f}  ρ={rho_low:.4f}  MAE={mae_low:.4f}")
if len(hi):
    _, _, R2_hi, rho_hi, mae_hi, _ = fit_eval(grid_df, hi)
    print(f"    hi-wbits  (≥2.75, n={len(hi)}): R²={R2_hi:.4f}  ρ={rho_hi:.4f}  MAE={mae_hi:.4f}")

# ------------------------------------------------------------------------
# PART 3: BASELINE COMPARISON — 27 random points vs 27 grid points
# ------------------------------------------------------------------------
print("\n" + "="*92)
print("PART 3: STRUCTURED 27-GRID vs 27-RANDOM BASELINE  (held-out on same 173 pts)")
print("="*92)

rng = np.random.default_rng(0)
TRIALS = 500
R2_rand, rho_rand, mae_rand = [], [], []
for _ in range(TRIALS):
    idx = rng.choice(len(df), size=27, replace=False)
    tr = df.iloc[idx]; te = df.drop(idx)
    _, _, R2, rho, mae, _ = fit_eval(tr, te)
    R2_rand.append(R2); rho_rand.append(rho); mae_rand.append(mae)
R2_rand = np.array(R2_rand); rho_rand = np.array(rho_rand); mae_rand = np.array(mae_rand)

print(f"\n  Method                       R²                ρ                MAE")
print("  " + "-"*80)
print(f"  Grid Q[0.1,0.5,0.9]   {R2_held:>8.4f}           {rho_held:>8.4f}          {mae_held:>8.4f}")
print(f"  Random 27 (mean±σ)    {R2_rand.mean():>8.4f}±{R2_rand.std():.4f}  "
      f"{rho_rand.mean():>8.4f}±{rho_rand.std():.4f}  {mae_rand.mean():>8.4f}±{mae_rand.std():.4f}")
print(f"  Random 27 (median)    {np.median(R2_rand):>8.4f}           "
      f"{np.median(rho_rand):>8.4f}          {np.median(mae_rand):>8.4f}")
print(f"  Random 27 (90% CI)   [{np.quantile(R2_rand,0.05):.4f},{np.quantile(R2_rand,0.95):.4f}]  "
      f"[{np.quantile(rho_rand,0.05):.4f},{np.quantile(rho_rand,0.95):.4f}]  "
      f"[{np.quantile(mae_rand,0.05):.4f},{np.quantile(mae_rand,0.95):.4f}]")

# Percentile of grid result inside random distribution
pct_R2  = (R2_rand < R2_held).mean()*100
pct_rho = (rho_rand < rho_held).mean()*100
pct_mae = (mae_rand > mae_held).mean()*100  # lower is better
print(f"\n  Grid-27 beats random-27 in:")
print(f"    R²  : better than {pct_R2:.1f}% of random trials")
print(f"    ρ   : better than {pct_rho:.1f}% of random trials")
print(f"    MAE : better than {pct_mae:.1f}% of random trials")

# ------------------------------------------------------------------------
# PART 4: QUANTILE-SCHEME SENSITIVITY
# ------------------------------------------------------------------------
print("\n" + "="*92)
print("PART 4: QUANTILE-SCHEME SENSITIVITY")
print("="*92)
schemes = [
    ('[0.10, 0.50, 0.90]', [0.10, 0.50, 0.90]),
    ('[0.05, 0.50, 0.95]', [0.05, 0.50, 0.95]),
    ('[0.20, 0.50, 0.80]', [0.20, 0.50, 0.80]),
    ('[0.15, 0.50, 0.85]', [0.15, 0.50, 0.85]),
    ('[0.10, 0.40, 0.70]', [0.10, 0.40, 0.70]),
    ('[0.25, 0.50, 0.75]', [0.25, 0.50, 0.75]),
    ('[min, median, max]',  None),    # special: use extrema
]

print(f"\n  {'Scheme':<24}{'R² (held)':>12}{'ρ (held)':>12}{'MAE':>10}{'low R²':>10}{'low ρ':>10}")
print("  " + "-"*80)
rows = []
for name, qset in schemes:
    if qset is None:
        qw_ = [df.w.min(), df.w.median(), df.w.max()]
        qkv_ = [df.kv.min(), df.kv.median(), df.kv.max()]
        qkd_ = [df.kvd.min(), df.kvd.median(), df.kvd.max()]
    else:
        qw_ = np.quantile(df.w, qset); qkv_ = np.quantile(df.kv, qset); qkd_ = np.quantile(df.kvd, qset)
    idxs = []
    for w_ in qw_:
        for kv_ in qkv_:
            for kd_ in qkd_:
                d = ((df.w-w_)/sw)**2 + ((df.kv-kv_)/skv)**2 + ((df.kvd-kd_)/skd)**2
                idxs.append(int(np.argmin(d.values)))
    train = df.iloc[idxs]; test = df.drop(list(set(idxs)))
    _, _, R2h, rhoh, maeh, _ = fit_eval(train, test)
    low_ = test[test.w < 2.75]
    if len(low_):
        _, _, R2l, rhol, _, _ = fit_eval(train, low_)
    else:
        R2l, rhol = np.nan, np.nan
    rows.append((name, R2h, rhoh, maeh, R2l, rhol))
    print(f"  {name:<24}{R2h:>12.4f}{rhoh:>12.4f}{maeh:>10.4f}{R2l:>10.4f}{rhol:>10.4f}")

# ------------------------------------------------------------------------
# PART 5: ANCHOR-EXACT FIT (if the 27 grid points were evaluated exactly)
# ------------------------------------------------------------------------
print("\n" + "="*92)
print("PART 5: IDEAL GRID (if JSD measured EXACTLY at [Q10,Q50,Q90]^3)")
print("="*92)
print("""
  Current 'grid' uses nearest-neighbor from the 200 random samples.
  If per-method Pareto frontier sampling produces the exact anchor values,
  the 27 cartesian points would be on-grid. We simulate this using the
  full-quadratic fit on all 200 samples as the 'ground truth' and then
  predict JSD at the 27 exact anchors.
""")
# Fit on ALL 200 as ground-truth proxy, use it to create the 27 anchor labels
beta_gt, *_ = fit_eval(df, df)  # gt model
# Predict the 27 exact anchor JSDs
anchor_rows = pd.DataFrame(grid, columns=['w','kv','kvd'])
# fake y column via the ground-truth model
anchor_rows['y'] = design(anchor_rows) @ beta_gt
beta_ideal, yh_ideal, R2_anc, rho_anc, mae_anc, _ = fit_eval(anchor_rows, anchor_rows)
_, _, R2_ext, rho_ext, mae_ext, _ = fit_eval(anchor_rows, df)
print(f"  Ideal-grid training  : R²={R2_anc:.4f} (should be ~1.0 since full-quad fits 10 params perfectly on 27 pts)")
print(f"  Ideal-grid vs real 200: R²={R2_ext:.4f}  ρ={rho_ext:.4f}  MAE={mae_ext:.4f}")
print(f"  Realistic (nearest-NN): R²={R2_held:.4f}  ρ={rho_held:.4f}  MAE={mae_held:.4f}")
print(f"\n  ⇒ Gap = {R2_ext - R2_held:.4f} in R² — measures NN-substitution noise penalty")

# ------------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax = axes[0,0]
ax.scatter(df.y, np.zeros(len(df))+0.05, s=5, alpha=0.3, color='gray', label='all 200')
ax.scatter(grid_df.y, np.zeros(len(grid_df))+0.15, s=30, color='C3', label='27 grid (NN)')
ax.axvline(df.y.mean(), color='k', lw=0.6, ls='--')
ax.set_xlabel('JSD'); ax.set_yticks([])
ax.set_title('JSD coverage: 27-grid vs 200-samples')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0,1]
ax.scatter(df.w, df.kv, s=8, alpha=0.3, color='gray')
ax.scatter(grid_df.w, grid_df.kv, s=60, color='C3', edgecolor='k')
for w_ in qw: ax.axvline(w_, color='C0', lw=0.4, alpha=0.5)
for kv_ in qkv: ax.axhline(kv_, color='C2', lw=0.4, alpha=0.5)
ax.set_xlabel('wbits'); ax.set_ylabel('kvbits')
ax.set_title('27 grid anchors in (W, KV) space')
ax.grid(alpha=0.3)

ax = axes[0,2]
ax.scatter(df.w, df.kvd, s=8, alpha=0.3, color='gray')
ax.scatter(grid_df.w, grid_df.kvd, s=60, color='C3', edgecolor='k')
for w_ in qw: ax.axvline(w_, color='C0', lw=0.4, alpha=0.5)
for kd_ in qkd: ax.axhline(kd_, color='C2', lw=0.4, alpha=0.5)
ax.set_xlabel('wbits'); ax.set_ylabel('kvdim')
ax.set_title('27 grid anchors in (W, KVdim) space')
ax.grid(alpha=0.3)

ax = axes[1,0]
y_full = held_df.y.values
_, ypred, *_ = fit_eval(grid_df, held_df)
ax.scatter(y_full, ypred, s=10, alpha=0.6)
lo, hi_ = min(y_full.min(), ypred.min()), max(y_full.max(), ypred.max())
ax.plot([lo,hi_],[lo,hi_],'k--',lw=0.6)
ax.set_xlabel('true JSD'); ax.set_ylabel('grid-27 prediction')
ax.set_title(f'Held-out (n={len(held_df)})  R²={R2_held:.3f}, ρ={rho_held:.3f}')
ax.grid(alpha=0.3)

ax = axes[1,1]
ax.hist(R2_rand, bins=40, color='C0', alpha=0.7, edgecolor='k', label='Random-27')
ax.axvline(R2_held, color='C3', lw=2, label=f'Grid Q[.1,.5,.9]={R2_held:.3f}')
ax.set_xlabel('Held-out R²'); ax.set_ylabel('count')
ax.set_title(f'Grid-27 vs 500 random-27 trials  (better than {pct_R2:.0f}%)')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1,2]
names = [r[0] for r in rows]; R2s = [r[1] for r in rows]; R2ls = [r[4] for r in rows]
x = np.arange(len(names))
ax.bar(x-0.2, R2s, 0.4, label='overall')
ax.bar(x+0.2, R2ls, 0.4, label='low-wbits')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('held-out R²')
ax.set_title('Quantile scheme sensitivity')
ax.legend(); ax.grid(alpha=0.3)

out = f'{BASE}/analysis_grid27.png'
plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
print(f"\nSaved plot to {out}")
print("="*92)
