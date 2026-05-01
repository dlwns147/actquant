"""
Generate publication-quality figures summarizing all analyses:
  fig1_variance_decomposition.png    — Hoeffding/ANOVA decomposition
  fig2_model_comparison.png          — Interaction-term & quadratic comparison
  fig3_low_wbits.png                 — Low-wbits correction strategies
  fig4_hier_vs_direct.png            — Calibration budget study
  fig5_grid27.png                    — 27-grid vs random-27
  fig6_quantile_scheme.png           — Quantile scheme sensitivity
  fig7_recommended_model.png         — Final recommended calibration fit
  fig8_summary.png                   — One-pager summary
"""
import os, csv, numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from itertools import combinations

plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

BASE = '/NAS/SJ/actquant/search/save/result'
OUT = f'{BASE}/figures'
os.makedirs(OUT, exist_ok=True)

FILES = {
    'w_kv':       f'{BASE}/2604162010_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr/results.csv',
    'w_kvdim':    f'{BASE}/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_dim/results.csv',
    'kv_kvdim':   f'{BASE}/2604162013_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_kv_expr_kv_dim/results.csv',
    'w_kv_kvdim': f'{BASE}/2604162012_Llama-3.1-8B-Instruct__0_0_awq_kivi_wikitext2_1_kv_scale_0seed_w_expr_kv_expr_kv_dim/results.csv',
}

def load(fp):
    data = []
    with open(fp) as f:
        for row in csv.reader(f):
            if row and any(x.strip() for x in row):
                try: data.append([float(x) for x in row])
                except ValueError: pass
    return pd.DataFrame(dict(
        w=np.array(data[0]), kv=np.array(data[1]),
        kvd=np.array(data[4]), y=np.array(data[12])))

df3 = load(FILES['w_kv_kvdim']).dropna().reset_index(drop=True)
dfwkv   = load(FILES['w_kv']).dropna(subset=['w','kv','y']).reset_index(drop=True)
dfwkd   = load(FILES['w_kvdim']).dropna(subset=['w','kvd','y']).reset_index(drop=True)
dfkvkd  = load(FILES['kv_kvdim']).dropna(subset=['kv','kvd','y']).reset_index(drop=True)

def regress(X, y):
    X1 = np.c_[np.ones(len(y)), X]
    beta,*_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    ss_reg = ((yhat - y.mean())**2).sum()
    ss_res = ((y - yhat)**2).sum()
    return beta, yhat, ss_reg, ss_res

def std(v): return (v - v.mean()) / v.std()

zw, zkv, zkd = std(df3.w).values, std(df3.kv).values, std(df3.kvd).values
y = df3.y.values
SST = ((y - y.mean())**2).sum()

# =============================================================================
# FIG 1: VARIANCE DECOMPOSITION
# =============================================================================
bases = [
    ('L_W (lin+quad)', np.c_[zw, zw**2]),
    ('L_KV (lin+quad)', np.c_[zkv, zkv**2]),
    ('L_KVdim (lin+quad)', np.c_[zkd, zkd**2]),
    ('L_W,KV (pure)', np.c_[zw*zkv]),
    ('L_W,KVdim (pure)', np.c_[zw*zkd]),
    ('L_KV,KVdim (pure)', np.c_[zkv*zkd]),
    ('L_W,KV,KVdim (pure)', np.c_[zw*zkv*zkd]),
]
cumX = np.zeros((len(y), 0))
prev = 0
decomp = []
for name, B in bases:
    cumX = np.c_[cumX, B]
    _,_,ssr,_ = regress(cumX, y)
    decomp.append((name, (ssr-prev)/SST*100))
    prev = ssr

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: per-component
ax = axes[0]
labels = [d[0].split('(')[0].strip() for d in decomp]
pcts = [d[1] for d in decomp]
colors = ['#2E86AB','#A23B72','#F18F01'] + ['#6B8E23','#6B8E23','#6B8E23'] + ['#8B0000']
bars = ax.barh(labels, pcts, color=colors)
ax.invert_yaxis()
for b, p in zip(bars, pcts):
    ax.text(p+1.5, b.get_y()+b.get_height()/2, f'{p:.2f}%', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('% of total variance (SST)')
ax.set_title('Hoeffding / ANOVA decomposition  (AWQ, n=200)', fontweight='bold')
ax.axvline(0, color='k', lw=0.5)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, max(pcts)*1.15)

# Right: aggregated
ax = axes[1]
cats = ['Additive\nmains', '2-way\ninteractions', '3-way\ninteraction', 'Residual\n(noise)']
vals = [sum(pcts[:3]), sum(pcts[3:6]), pcts[6], 100-sum(pcts)]
colors2 = ['#2E86AB', '#F18F01', '#8B0000', '#999999']
bars = ax.bar(cats, vals, color=colors2)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+1.5, f'{v:.2f}%', ha='center', fontweight='bold')
ax.set_ylabel('% of SST')
ax.set_title('Additivity dominance: 95.94% from mains alone', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(vals)*1.15)

plt.suptitle('Fig. 1 — Variance decomposition of JSD loss L(xW, xKV, xKVdim)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_variance_decomposition.png', bbox_inches='tight')
plt.close()
print('Saved fig1_variance_decomposition.png')

# =============================================================================
# FIG 2: MODEL COMPARISON (interaction/quadratic)
# =============================================================================
w_s, kv_s, kvd_s = zw, zkv, zkd
models = {
    'M0:\nlin.add.':          np.c_[w_s, kv_s, kvd_s],
    'M1:\n+W·KV':             np.c_[w_s, kv_s, kvd_s, w_s*kv_s],
    'M2:\n+W·KVdim':          np.c_[w_s, kv_s, kvd_s, w_s*kvd_s],
    'M3:\n+KV·KVdim':         np.c_[w_s, kv_s, kvd_s, kv_s*kvd_s],
    'M4:\nall 2-way':         np.c_[w_s, kv_s, kvd_s, w_s*kv_s, w_s*kvd_s, kv_s*kvd_s],
    'M6:\n+W²':               np.c_[w_s, kv_s, kvd_s, w_s**2],
    'M8:\n+W²+W·(KV,KVdim)':  np.c_[w_s, kv_s, kvd_s, w_s**2, w_s*kv_s, w_s*kvd_s],
    'M5:\nall quad.':         np.c_[w_s, w_s**2, kv_s, kv_s**2, kvd_s, kvd_s**2],
    'M7:\nquad+all inter.':   np.c_[w_s, w_s**2, kv_s, kv_s**2, kvd_s, kvd_s**2,
                                    w_s*kv_s, w_s*kvd_s, kv_s*kvd_s],
}
def cv_r2(X, y, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    fold = len(y)//k
    r2s, rhos = [], []
    for f in range(k):
        te = idx[f*fold:(f+1)*fold]
        tr = np.concatenate([idx[:f*fold], idx[(f+1)*fold:]])
        X1 = np.c_[np.ones(len(tr)), X[tr]]
        beta,*_ = np.linalg.lstsq(X1, y[tr], rcond=None)
        X1te = np.c_[np.ones(len(te)), X[te]]
        yhat = X1te @ beta
        sst = ((y[te]-y[te].mean())**2).sum()
        r2s.append(1 - ((y[te]-yhat)**2).sum()/sst)
        rhos.append(stats.spearmanr(y[te], yhat)[0])
    return np.mean(r2s), np.std(r2s), np.mean(rhos)

cv_r2s, cv_stds, cv_rhos, R2_trains, ps = [], [], [], [], []
for name, X in models.items():
    _,yhat,ssr,ssres = regress(X, y)
    R2_trains.append(1 - ssres/SST)
    mr, sr, mrho = cv_r2(X, y)
    cv_r2s.append(mr); cv_stds.append(sr); cv_rhos.append(mrho)
    ps.append(X.shape[1]+1)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
x = np.arange(len(models))
ax.bar(x-0.2, R2_trains, 0.4, label='Train R²', color='#2E86AB')
ax.errorbar(x+0.2, cv_r2s, yerr=cv_stds, fmt='s', color='#C73E1D', markersize=8,
            capsize=3, label='5-fold CV R² ± σ', lw=1.5)
ax.set_xticks(x); ax.set_xticklabels(models.keys(), fontsize=9)
ax.set_ylabel('R²')
ax.set_ylim(0.83, 0.97)
ax.set_title('(a) Train vs CV R² for 9 model families', fontweight='bold')
ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.3)
for xi, p in zip(x, ps):
    ax.text(xi, 0.832, f'p={p}', ha='center', fontsize=8, color='gray')

ax = axes[1]
ax.scatter(ps, cv_r2s, s=120, c=range(len(ps)), cmap='viridis', edgecolor='k', zorder=3)
for xi, yi, nm in zip(ps, cv_r2s, models.keys()):
    ax.annotate(nm.replace('\n',' '), (xi, yi), xytext=(5, 5),
                textcoords='offset points', fontsize=8)
ax.set_xlabel('# parameters')
ax.set_ylabel('CV R²')
ax.set_title('(b) Parsimony-accuracy trade-off', fontweight='bold')
ax.grid(alpha=0.3)
ax.axhline(max(cv_r2s), color='red', lw=0.8, ls='--', alpha=0.5)
ax.text(max(ps)*0.6, max(cv_r2s)+0.001, f'best CV R²={max(cv_r2s):.4f}', color='red', fontsize=9)

plt.suptitle('Fig. 2 — Interaction & quadratic model comparison', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_model_comparison.png', bbox_inches='tight')
plt.close()
print('Saved fig2_model_comparison.png')

# =============================================================================
# FIG 3: LOW-WBITS CORRECTION
# =============================================================================
lo = df3[df3.w < 2.75]; hi = df3[df3.w >= 2.75]
zwl, zkvl, zkdl = std(lo.w).values, std(lo.kv).values, std(lo.kvd).values
yl = lo.y.values

strategies = {
    'A: linear add.':              np.c_[zw, zkv, zkd],
    'B: log y + lin.':             np.c_[zw, zkv, zkd],  # y=log below
    'C: lin + W²':                 np.c_[zw, zw**2, zkv, zkd],
    'D: 2^(-W) basis':             np.c_[2**(-df3.w.values), zkv, zkd],
    'E: full quad.':               np.c_[zw, zw**2, zkv, zkv**2, zkd, zkd**2],
    'F: quad + inter.':            np.c_[zw, zw**2, zkv, zkv**2, zkd, zkd**2,
                                         zw*zkv, zw*zkd, zkv*zkd],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
names = []
mae_low, mae_hi = [], []
R2s = []
for name, X in strategies.items():
    if 'log' in name:
        ylog = np.log(y+1e-6)
        _,yhat,_,ssres = regress(X, ylog)
        yhat_orig = np.exp(yhat)-1e-6
    else:
        _,yhat,_,ssres = regress(X, y)
        yhat_orig = yhat
    R2s.append(1 - ssres/SST)
    mask_lo = df3.w.values < 2.75
    mae_low.append(np.mean(np.abs(y[mask_lo] - yhat_orig[mask_lo])))
    mae_hi.append(np.mean(np.abs(y[~mask_lo] - yhat_orig[~mask_lo])))
    names.append(name)

x = np.arange(len(names))
ax.bar(x-0.2, mae_low, 0.4, label='low-wbits (<2.75, n=107)', color='#C73E1D')
ax.bar(x+0.2, mae_hi,  0.4, label='hi-wbits  (≥2.75, n=93)', color='#2E86AB')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('MAE of fitted JSD')
ax.set_title('(a) Low-wbits regime dominates residual error', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)

ax = axes[1]
# Residual scatter: baseline vs W² added
_, yhatA, *_ = regress(strategies['A: linear add.'], y)
_, yhatC, *_ = regress(strategies['C: lin + W²'], y)
_, yhatF, *_ = regress(strategies['F: quad + inter.'], y)
ax.scatter(df3.w, y - yhatA, s=15, alpha=0.5, label=f'A: linear        σ={np.std(y-yhatA):.4f}', color='#999')
ax.scatter(df3.w, y - yhatC, s=15, alpha=0.6, label=f'C: +W²           σ={np.std(y-yhatC):.4f}', color='#2E86AB')
ax.scatter(df3.w, y - yhatF, s=15, alpha=0.7, label=f'F: quad+inter.  σ={np.std(y-yhatF):.4f}', color='#C73E1D')
ax.axhline(0, color='k', lw=0.6)
ax.axvline(2.75, color='red', lw=1, ls='--', alpha=0.6, label='low/hi split')
ax.set_xlabel('wbits'); ax.set_ylabel('residual (y − ŷ)')
ax.set_title('(b) Residual shrinks with W² term', fontweight='bold')
ax.legend(fontsize=9, loc='upper right'); ax.grid(alpha=0.3)

plt.suptitle('Fig. 3 — Low-wbits correction strategies', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_low_wbits.png', bbox_inches='tight')
plt.close()
print('Saved fig3_low_wbits.png')

# =============================================================================
# FIG 4: HIERARCHICAL vs DIRECT (budget curve)
# =============================================================================
def design_full(df_, cols=('w','kv','kvd')):
    xs = [df_[c].values for c in cols]
    X = []
    for v in xs: X += [v, v**2]
    X += [xs[0]*xs[1], xs[0]*xs[2], xs[1]*xs[2]]
    return np.c_[np.ones(len(df_)), np.array(X).T]

def direct_trial(n_total, trials=100, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        idx = rng.choice(len(df3), size=n_total, replace=False)
        tr = df3.iloc[idx]; te = df3.drop(idx)
        Xtr = design_full(tr); beta,*_ = np.linalg.lstsq(Xtr, tr.y.values, rcond=None)
        yhat = design_full(te) @ beta
        sst = ((te.y.values - te.y.mean())**2).sum()
        out.append(1 - ((te.y.values - yhat)**2).sum()/sst)
    return np.array(out)

def direct_lin_trial(n_total, trials=100, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        idx = rng.choice(len(df3), size=n_total, replace=False)
        tr = df3.iloc[idx]; te = df3.drop(idx)
        Xtr = np.c_[np.ones(len(tr)), tr.w.values, tr.kv.values, tr.kvd.values]
        beta,*_ = np.linalg.lstsq(Xtr, tr.y.values, rcond=None)
        Xte = np.c_[np.ones(len(te)), te.w.values, te.kv.values, te.kvd.values]
        yhat = Xte @ beta
        sst = ((te.y.values - te.y.mean())**2).sum()
        out.append(1 - ((te.y.values - yhat)**2).sum()/sst)
    return np.array(out)

def hier_trial(pair_df, pair_cols, third, n2, n3, trials=100, seed=0):
    rng = np.random.default_rng(seed+10)
    out = []
    for _ in range(trials):
        ix2 = rng.choice(len(pair_df), size=min(n2, len(pair_df)), replace=False)
        d2 = pair_df.iloc[ix2]
        # step 1: quad+inter on pair
        a,b = pair_cols
        xs = [d2[a].values, d2[b].values]
        Xp = np.c_[np.ones(len(d2)), xs[0], xs[0]**2, xs[1], xs[1]**2, xs[0]*xs[1]]
        beta_p,*_ = np.linalg.lstsq(Xp, d2.y.values, rcond=None)
        # step 2
        ix3 = rng.choice(len(df3), size=n3, replace=False)
        d3 = df3.iloc[ix3]
        xs3 = [d3[a].values, d3[b].values]
        pp = np.c_[np.ones(len(d3)), xs3[0], xs3[0]**2, xs3[1], xs3[1]**2, xs3[0]*xs3[1]] @ beta_p
        resid = d3.y.values - pp
        zC = d3[third].values
        Xr = np.c_[np.ones(len(zC)), zC, zC**2]
        coef,*_ = np.linalg.lstsq(Xr, resid, rcond=None)
        # eval
        held = df3.drop(ix3)
        xsh = [held[a].values, held[b].values]
        pph = np.c_[np.ones(len(held)), xsh[0], xsh[0]**2, xsh[1], xsh[1]**2, xsh[0]*xsh[1]] @ beta_p
        zCh = held[third].values
        corr = np.c_[np.ones(len(zCh)), zCh, zCh**2] @ coef
        yhat = pph + corr
        sst = ((held.y.values - held.y.mean())**2).sum()
        out.append(1 - ((held.y.values - yhat)**2).sum()/sst)
    return np.array(out)

budgets = [15, 20, 27, 40, 60, 80, 100, 127]
direct_r2 = [direct_trial(min(b, 190)).mean() for b in budgets]
direct_std = [direct_trial(min(b, 190)).std() for b in budgets]
directlin_r2 = [direct_lin_trial(min(b, 190)).mean() for b in budgets]

hier_configs = [
    ('WKV→KVdim', dfwkv,  ('w','kv'),  'kvd'),
    ('WKVdim→KV', dfwkd,  ('w','kvd'), 'kv'),
    ('KVKVdim→W', dfkvkd, ('kv','kvd'),'w'),
]
hier_data = {}
for name, pd_, cols, third in hier_configs:
    means = []
    for b in budgets:
        n2 = max(int(b*0.65), 10); n3 = b - n2
        if n3 < 5: n3 = 5; n2 = b - n3
        means.append(hier_trial(pd_, cols, third, n2, n3).mean())
    hier_data[name] = means

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.plot(budgets, directlin_r2, 'o--', label='Direct linear', color='#999', lw=2)
ax.plot(budgets, direct_r2, 's-', label='Direct full quad.', color='#2E86AB', lw=2.5, markersize=9)
cols_hier = {'WKV→KVdim':'#A23B72','WKVdim→KV':'#F18F01','KVKVdim→W':'#8B0000'}
for name, vals in hier_data.items():
    ax.plot(budgets, vals, 'd-', label=f'Hier {name}', color=cols_hier[name], lw=1.5, alpha=0.85)
ax.axvline(27, color='red', lw=1, ls=':', alpha=0.6)
ax.text(27.5, 0.72, '27 samples\n(3×3×3)', color='red', fontsize=9)
ax.set_xlabel('Total AWQ samples')
ax.set_ylabel('Held-out R²')
ax.set_title('(a) Calibration budget curves', fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
# pair fit quality
pairs = [('(W,KV)', dfwkv, ['w','kv']), ('(W,KVdim)', dfwkd, ['w','kvd']), ('(KV,KVdim)', dfkvkd, ['kv','kvd'])]
pair_R2 = []
for _, pd_, cols in pairs:
    xs = [pd_[c].values for c in cols]
    X = np.c_[xs[0], xs[0]**2, xs[1], xs[1]**2, xs[0]*xs[1]]
    _,_,_, res = regress(X, pd_.y.values)
    sst = ((pd_.y.values - pd_.y.mean())**2).sum()
    pair_R2.append(1 - res/sst)
x_ = np.arange(3)
bars = ax.bar(x_, pair_R2, color=['#2E86AB','#A23B72','#8B0000'])
for b, r in zip(bars, pair_R2):
    ax.text(b.get_x()+b.get_width()/2, r+0.01, f'{r:.3f}', ha='center', fontweight='bold')
ax.set_xticks(x_); ax.set_xticklabels([p[0] for p in pairs])
ax.set_ylabel('Step-1 R² (quad + interaction)')
ax.set_ylim(0.65, 1.0)
ax.set_title('(b) Step-1 pair fit quality', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.axhline(0.9, color='red', lw=0.8, ls='--', alpha=0.5)

plt.suptitle('Fig. 4 — Hierarchical vs Direct calibration (equal AWQ budget)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_hier_vs_direct.png', bbox_inches='tight')
plt.close()
print('Saved fig4_hier_vs_direct.png')

# =============================================================================
# FIG 5: 27-GRID vs RANDOM-27
# =============================================================================
Q = [0.1, 0.5, 0.9]
qw, qkv, qkd = np.quantile(df3.w, Q), np.quantile(df3.kv, Q), np.quantile(df3.kvd, Q)
sw, skv, skd = df3.w.std(), df3.kv.std(), df3.kvd.std()
grid_idx = []
for w in qw:
    for kv in qkv:
        for kd in qkd:
            d = ((df3.w-w)/sw)**2 + ((df3.kv-kv)/skv)**2 + ((df3.kvd-kd)/skd)**2
            grid_idx.append(int(np.argmin(d.values)))
grid_df = df3.iloc[grid_idx]

def fit_eval(tr, te):
    Xtr = design_full(tr); beta,*_ = np.linalg.lstsq(Xtr, tr.y.values, rcond=None)
    yhat = design_full(te) @ beta
    sst = ((te.y.values - te.y.mean())**2).sum()
    R2 = 1 - ((te.y.values - yhat)**2).sum()/sst if sst>0 else np.nan
    rho = stats.spearmanr(te.y.values, yhat)[0]
    return yhat, R2, rho

held = df3.drop(list(set(grid_idx)))
yhat_grid, R2_grid, rho_grid = fit_eval(grid_df, held)

rng = np.random.default_rng(0)
R2_rand = []
for _ in range(500):
    idx = rng.choice(len(df3), size=27, replace=False)
    tr = df3.iloc[idx]; te = df3.drop(idx)
    _, R2, _ = fit_eval(tr, te)
    R2_rand.append(R2)
R2_rand = np.array(R2_rand)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(df3.w, df3.kv, s=14, alpha=0.35, color='#bbb', label='200 samples')
ax.scatter(grid_df.w, grid_df.kv, s=100, color='#C73E1D', edgecolor='k', lw=1, label='27-grid NN')
for q in qw: ax.axvline(q, color='#2E86AB', lw=0.6, alpha=0.5)
for q in qkv: ax.axhline(q, color='#A23B72', lw=0.6, alpha=0.5)
ax.set_xlabel('wbits'); ax.set_ylabel('kvbits')
ax.set_title('(a) 27-grid anchors in (W, KV)', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.scatter(df3.w, df3.kvd, s=14, alpha=0.35, color='#bbb')
ax.scatter(grid_df.w, grid_df.kvd, s=100, color='#C73E1D', edgecolor='k', lw=1)
for q in qw: ax.axvline(q, color='#2E86AB', lw=0.6, alpha=0.5)
for q in qkd: ax.axhline(q, color='#F18F01', lw=0.6, alpha=0.5)
ax.set_xlabel('wbits'); ax.set_ylabel('kvdim')
ax.set_title('(b) 27-grid anchors in (W, KVdim)', fontweight='bold')
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 2])
ax.hist(R2_rand, bins=40, color='#2E86AB', alpha=0.7, edgecolor='k')
ax.axvline(R2_grid, color='#C73E1D', lw=3, label=f'Grid = {R2_grid:.3f}')
ax.axvline(R2_rand.mean(), color='k', lw=1, ls='--', label=f'Rand μ = {R2_rand.mean():.3f}')
pct = (R2_rand < R2_grid).mean()*100
ax.set_xlabel('Held-out R²'); ax.set_ylabel('count')
ax.set_title(f'(c) Grid-27 beats {pct:.0f}% of random-27 (500 trials)', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(held.y, yhat_grid, s=14, alpha=0.6, color='#2E86AB')
lo_, hi_ = held.y.min(), held.y.max()
ax.plot([lo_, hi_], [lo_, hi_], 'k--', lw=0.8)
ax.set_xlabel('true JSD'); ax.set_ylabel('grid-27 prediction')
ax.set_title(f'(d) Held-out fit  R²={R2_grid:.3f}  ρ={rho_grid:.3f}', fontweight='bold')
ax.grid(alpha=0.3)

# Residual vs wbits
ax = fig.add_subplot(gs[1, 1])
resid = held.y.values - yhat_grid
ax.scatter(held.w, resid, s=14, alpha=0.6, color='#A23B72')
ax.axhline(0, color='k', lw=0.6)
ax.axvline(2.75, color='red', lw=0.8, ls='--', alpha=0.6)
ax.set_xlabel('wbits'); ax.set_ylabel('residual')
ax.set_title(f'(e) Residual pattern (MAE={np.mean(np.abs(resid)):.4f})', fontweight='bold')
ax.grid(alpha=0.3)

# Quantile scheme sensitivity
ax = fig.add_subplot(gs[1, 2])
schemes = [
    ('.05/.50/.95', [.05,.5,.95]),
    ('.10/.50/.90', [.10,.5,.90]),
    ('.15/.50/.85', [.15,.5,.85]),
    ('.20/.50/.80', [.20,.5,.80]),
    ('.25/.50/.75', [.25,.5,.75]),
]
scheme_R2 = []
for nm, qset in schemes:
    qw_, qkv_, qkd_ = np.quantile(df3.w, qset), np.quantile(df3.kv, qset), np.quantile(df3.kvd, qset)
    idxs = []
    for w_ in qw_:
        for kv_ in qkv_:
            for kd_ in qkd_:
                d = ((df3.w-w_)/sw)**2 + ((df3.kv-kv_)/skv)**2 + ((df3.kvd-kd_)/skd)**2
                idxs.append(int(np.argmin(d.values)))
    tr = df3.iloc[idxs]; te = df3.drop(list(set(idxs)))
    _, R2, _ = fit_eval(tr, te)
    scheme_R2.append(R2)
x_ = np.arange(len(schemes))
bars = ax.bar(x_, scheme_R2, color=['#1f77b4' if s!='.10/.50/.90' else '#C73E1D' for s,_ in schemes])
for b, r in zip(bars, scheme_R2):
    ax.text(b.get_x()+b.get_width()/2, r+0.002, f'{r:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x_); ax.set_xticklabels([s for s,_ in schemes], rotation=15)
ax.set_ylim(min(scheme_R2)-0.02, max(scheme_R2)+0.01)
ax.set_ylabel('Held-out R²')
ax.set_title('(f) Quantile-scheme sensitivity', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Fig. 5 — 27-Grid (3×3×3) vs Random-27 calibration', fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_grid27.png', bbox_inches='tight')
plt.close()
print('Saved fig5_grid27.png')

# =============================================================================
# FIG 6: RECOMMENDED MODEL FIT & COEFFICIENTS
# =============================================================================
# Final recommended: y = β0 + β1·w + β2·kv + β3·kvd + β4·w² + β5·w·kv + β6·w·kvd
xs = [df3.w.values, df3.kv.values, df3.kvd.values]
X_rec = np.c_[xs[0], xs[1], xs[2], xs[0]**2, xs[0]*xs[1], xs[0]*xs[2]]
beta_rec, yhat_rec, ss_reg_rec, ss_res_rec = regress(X_rec, y)
R2_rec = 1 - ss_res_rec/SST

# Full quad+inter on 27-grid (alternative final)
Xg = design_full(grid_df); beta_g,*_ = np.linalg.lstsq(Xg, grid_df.y.values, rcond=None)
yhat_full = design_full(df3) @ beta_g

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0,0]
coef_names = ['β₀','β_w','β_kv','β_kvd','β_w²','β_w·kv','β_w·kvd']
coefs = beta_rec
colors = ['#999','#2E86AB','#A23B72','#F18F01','#C73E1D','#6B8E23','#6B8E23']
bars = ax.barh(coef_names, coefs, color=colors)
ax.invert_yaxis()
ax.axvline(0, color='k', lw=0.6)
for b, c in zip(bars, coefs):
    ax.text(c + (0.01 if c>=0 else -0.01), b.get_y()+b.get_height()/2,
            f'{c:+.4f}', va='center',
            ha='left' if c>=0 else 'right', fontsize=9, fontweight='bold')
ax.set_xlabel('coefficient (unstandardized)')
ax.set_title('(a) Recommended model: ŷ = β₀ + Σβᵢxᵢ + β·w² + β·w·kv + β·w·kvd',
             fontweight='bold', fontsize=11)
ax.grid(axis='x', alpha=0.3)

ax = axes[0,1]
ax.scatter(y, yhat_rec, s=14, alpha=0.5, label=f'Recommended  R²={R2_rec:.3f}', color='#2E86AB')
ax.scatter(y, yhat_full, s=14, alpha=0.5, label=f'Grid-27 full  R²={1 - ((y-yhat_full)**2).sum()/SST:.3f}', color='#C73E1D')
lo_, hi_ = y.min(), y.max()
ax.plot([lo_, hi_], [lo_, hi_], 'k--', lw=0.6)
ax.set_xlabel('true JSD'); ax.set_ylabel('predicted JSD')
ax.set_title('(b) Predicted vs true', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

# 2D slice: vary w, kvd; fix kv=median
ax = axes[1,0]
kv_med = df3.kv.median()
ww = np.linspace(df3.w.min(), df3.w.max(), 60)
kd = np.linspace(df3.kvd.min(), df3.kvd.max(), 60)
WW, KD = np.meshgrid(ww, kd)
Xs = np.c_[WW.ravel(), np.full(WW.size, kv_med), KD.ravel(),
           WW.ravel()**2, WW.ravel()*kv_med, WW.ravel()*KD.ravel()]
zz = (np.c_[np.ones(Xs.shape[0]), Xs] @ beta_rec).reshape(WW.shape)
cs = ax.contourf(WW, KD, zz, levels=20, cmap='viridis')
plt.colorbar(cs, ax=ax, label='predicted JSD')
ax.scatter(df3.w, df3.kvd, s=6, alpha=0.4, color='white', edgecolor='k', lw=0.3)
ax.set_xlabel('wbits'); ax.set_ylabel('kvdim')
ax.set_title(f'(c) Surface at kv={kv_med:.2f} (median)', fontweight='bold')

ax = axes[1,1]
# residual histogram
resid_rec = y - yhat_rec
ax.hist(resid_rec, bins=40, color='#2E86AB', alpha=0.7, edgecolor='k')
ax.axvline(0, color='k', lw=1)
ax.axvline(resid_rec.mean(), color='red', lw=1, ls='--', label=f'mean={resid_rec.mean():.4f}')
ax.set_xlabel('residual (y − ŷ)'); ax.set_ylabel('count')
ax.set_title(f'(d) Residual dist.  σ={resid_rec.std():.4f},  MAE={np.mean(np.abs(resid_rec)):.4f}',
             fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('Fig. 6 — Recommended calibration model', fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_recommended_model.png', bbox_inches='tight')
plt.close()
print('Saved fig6_recommended_model.png')

# =============================================================================
# FIG 7: ONE-PAGER SUMMARY
# =============================================================================
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

# 1. variance pie
ax = fig.add_subplot(gs[0, 0])
sizes = [95.94, 0.09, 0.06, 4.06 - 0.09 - 0.06]  # rough noise
labels = ['Additive\nmains\n95.94%', '2-way\n0.09%', '3-way\n0.06%', 'noise\n3.91%']
colors_p = ['#2E86AB','#F18F01','#8B0000','#999999']
ax.pie(sizes, labels=labels, colors=colors_p, startangle=90, textprops=dict(fontsize=9))
ax.set_title('1. Variance decomposition\n(Hoeffding)', fontweight='bold')

# 2. model R²
ax = fig.add_subplot(gs[0, 1])
short = ['M0\nlin', 'M6\n+W²', 'M5\n+all quad', 'M7\n+inter', '27-grid\nfull']
vals_ = [0.848, 0.940, 0.959, 0.960, 0.943]
colors3 = ['#999','#2E86AB','#A23B72','#F18F01','#C73E1D']
bars = ax.bar(short, vals_, color=colors3)
for b, v in zip(bars, vals_):
    ax.text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylim(0.82, 0.98)
ax.set_ylabel('CV R² (200 samples)')
ax.set_title('2. Model family ranking', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 3. budget curve
ax = fig.add_subplot(gs[0, 2])
ax.plot(budgets, direct_r2, 's-', label='Direct full quad.', color='#2E86AB', lw=2.5)
ax.plot(budgets, hier_data['WKV→KVdim'], 'd-', label='Hier WKV→KVdim', color='#A23B72', lw=1.5)
ax.plot(budgets, hier_data['KVKVdim→W'], 'd-', label='Hier KVKVdim→W', color='#8B0000', lw=1.5)
ax.axvline(27, color='red', ls=':', lw=1)
ax.set_xlabel('AWQ samples'); ax.set_ylabel('Held-out R²')
ax.set_title('3. Budget curve', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 4. grid vs random histogram
ax = fig.add_subplot(gs[0, 3])
ax.hist(R2_rand, bins=30, color='#2E86AB', alpha=0.7, edgecolor='k')
ax.axvline(R2_grid, color='#C73E1D', lw=3, label=f'27-grid={R2_grid:.3f}')
ax.set_xlabel('Held-out R² (n_cal=27)')
ax.set_ylabel('count')
ax.set_title(f'4. Grid vs random-27\n(grid in top {100-pct:.0f}%)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 5. scatter: recommended model
ax = fig.add_subplot(gs[1, 0])
ax.scatter(y, yhat_rec, s=10, alpha=0.5, color='#2E86AB')
lo_, hi_ = y.min(), y.max()
ax.plot([lo_, hi_], [lo_, hi_], 'k--', lw=0.6)
ax.set_xlabel('true JSD'); ax.set_ylabel('predicted JSD')
ax.set_title(f'5. Recommended fit\nR²={R2_rec:.3f}', fontweight='bold')
ax.grid(alpha=0.3)

# 6. residual vs W
ax = fig.add_subplot(gs[1, 1])
ax.scatter(df3.w, y - yhat_rec, s=8, alpha=0.5, color='#A23B72')
ax.axhline(0, color='k', lw=0.6)
ax.axvline(2.75, color='red', ls='--', lw=0.8, alpha=0.7)
ax.set_xlabel('wbits'); ax.set_ylabel('residual')
ax.set_title('6. Residual vs wbits', fontweight='bold')
ax.grid(alpha=0.3)

# 7. pair fit quality
ax = fig.add_subplot(gs[1, 2])
bars = ax.bar(['(W,KV)','(W,KVd)','(KV,KVd)'], pair_R2, color=['#2E86AB','#A23B72','#8B0000'])
for b, v in zip(bars, pair_R2):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontweight='bold')
ax.set_ylim(0.6, 1.0)
ax.set_ylabel('Step-1 R²')
ax.set_title('7. Pair fit quality\n(W-containing pairs >> KV+KVdim)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 8. summary text
ax = fig.add_subplot(gs[1, 3])
ax.axis('off')
summary = """KEY FINDINGS

• Additive mains → 95.9% of Var[L]
• 2-way + 3-way inter. → 0.15%

• W² alone lifts R² : 0.85 → 0.94
• Full quad+inter : R² = 0.96

• n_cal = 27: direct > hier
  (equal budget, all paths)

• 27-grid Q[.1,.5,.9]:
  held-out R² = 0.943
  beats 80% of random-27

• Pair order WITHIN stage = irrelevant
  (polynomial basis symmetric)

• KVKVdim→W hier path unstable
  at small n (σ×47 vs WKV)

⇒ RECOMMENDATION
  Direct 27-grid Q[.1,.5,.9]
  with full quad + all inter.
"""
ax.text(0, 0.98, summary, fontsize=10, va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f7ff', edgecolor='#2E86AB'))

plt.suptitle('Fig. 7 — Summary: Additive Pareto-frontier combination for LLM quantization NAS',
             fontsize=14, fontweight='bold', y=1.00)
plt.savefig(f'{OUT}/fig7_summary.png', bbox_inches='tight')
plt.close()
print('Saved fig7_summary.png')

print(f'\nAll figures saved to: {OUT}/')
print('  fig1_variance_decomposition.png')
print('  fig2_model_comparison.png')
print('  fig3_low_wbits.png')
print('  fig4_hier_vs_direct.png')
print('  fig5_grid27.png')
print('  fig6_recommended_model.png')
print('  fig7_summary.png')
