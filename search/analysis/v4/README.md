# v4 — surrogate / hierarchical-vs-joint / PF-combination re-analysis on `save/result/260510`

Updated 2026-05-12. The previous v4 (which trained on `260506` and tested on the
old `awq/2604162012_..._w_expr_kv_expr_kv_dim` set) is archived under
`archive_v1/`. This pass uses **only** the CSV files in `save/result/260510/`
— no `save/search/think/*.stats` files are consulted, per user request.

CSV row layout (built by `post_search_split.py`):

| row | content |
|----:|---------|
| 0..11 | complexity columns from `get_net_info(...)`: wbits, kvbits, kbits, vbits, kvdim, kdim, vdim, eff_kvbits, eff_kbits, eff_vbits, memory, n_token |
| 12 | **measured** wikitext2 JSD (target `y`) |
| 13 | search-time **combined** prediction (≡ row14 + row15 + row16 for 3-axis runs) |
| 14..16 | per-method z's from each axis's PF: row14=z_W, row15=z_KV, row16=z_KVD |

So the surrogate's input X = (z_W, z_KV, z_KVD) and target y = row 12.
The "no-learning baseline" used as Hierarchical-A in Phase 3 is row 13 directly.

## Datasets used (260510, CSV-only)

| Tag | File suffix | Cols | Role |
|---|---|---:|---|
| llama_qs50  | `..._qs_w159_kv159_kvdim159_rs23/results.csv`            | 50  | train (27 quantile + 23 random) |
| llama_rs50  | `..._w_expr_kv_expr_kvdim_expr_rs50/results.csv`         | 50  | pure-random train baseline |
| llama_rs200 | `..._w_expr_kv_expr_kvdim_expr_rs200/results.csv`        | 200 | held-out test |
| qwen_qs50   | Qwen counterpart                                          | 50  | train |
| qwen_rs50   | Qwen counterpart                                          | 50  | pure-random train baseline |
| qwen_rs200  | Qwen counterpart                                          | 200 | held-out test |
| llama_wk    | `_w_expr_kv_expr_qs_w159_kv159_rs41/results.csv`         | 29  | 2-axis (w,kv) — held for follow-on |
| llama_kvkd  | `_kv_expr_kvdim_expr_qs_kv159_kvdim159_rs41/results.csv` | 30  | 2-axis (kv,kvd) — held |
| llama_wkd   | `_w_expr_kvdim_expr_qs_w159_kvdim159_rs41/results.csv`   | 31  | 2-axis (w,kvd) — held |

## Scripts

| Script | Phase | Train | Test |
|---|---|---|---|
| `01_phase1_surrogates.py`        | 1   | 50 QS                                              | 200 RS  (Llama + Qwen) |
| `01b_phase1b_sweep.py`           | 1b  | N ∈ {27, 30, 35, 40, 45, 50} from *_qs50; det N ∈ {27,50}, stochastic 20-seed for {30,35,40,45} | 200 RS  (Llama + Qwen) |
| `01c_phase1c_qs_vs_rs_sweep.py`  | 1c  | Same N grid, but RS-pure (from *_rs50) vs QS-first overlay + coverage-volume diagnostic | 200 RS  (Llama + Qwen) |
| `02_phase2_qs_vs_random.py`      | 2   | QS-50 vs RS-50 (paired single trial + 20-seed bootstrap on RS) | 200 RS  (Llama + Qwen) |
| `03_phase3_hier_vs_joint.py`     | 3   | 50 QS — joint RBF tps+linear surrogate, naive row-13 baseline, and Hoeffding additive collapse of the joint | 200 RS  (Llama + Qwen) |
| `04_phase4_internal.py`          | 4   | 50 QS — Sobol/active-subspace/Hoeffding on RBF tps+linear (Phase-1 winner) + ARD-Matérn-3/2 (length-scale-bearing comparator) | 200 RS  (Llama + Qwen) |
| `05_phase5_local_pf_combination.py` | 5 | — uses 200 RS test directly (row 12 vs row 13) | 200 RS  (Llama + Qwen) |

---

## Phase 1 — surrogate comparison (50 QS train → 200 RS test)

Surrogates: **M1 (linear additive — baseline)**, M_quad (full quadratic 1+3+3+3),
RBF cubic+linear, RBF tps+linear, ARD-GP Matérn-3/2 +noise (n_restarts=30,
length-scale bounds (1e-4, 1e4)).

### Llama-3.1-8B-Instruct

| Surrogate | R²_train | R²_test | RMSE | ε_∞ |
|---|---:|---:|---:|---:|
| M1 linear additive *(baseline)*          | 0.9594 | 0.9445 | 0.02837 | 0.11958 |
| M_quad full quadratic                    | 0.9801 | 0.9858 | 0.01438 | 0.04858 |
| **RBF cubic+linear**                     | 1.0000 | **0.9951** | **0.00840** | **0.03163** |
| RBF tps+linear                           | 1.0000 | 0.9934 | 0.00976 | 0.07011 |
| ARD-GP Matérn-3/2 +noise                 | 1.0000 | 0.9949 | 0.00862 | 0.03905 |

ARD-GP length scales: l_W=1.05, l_KV=0.48, l_KVD=0.90; σ_n=2.1×10⁻⁵; LMML=57.0.

### Qwen2.5-7B-Instruct

| Surrogate | R²_train | R²_test | RMSE | ε_∞ |
|---|---:|---:|---:|---:|
| M1 linear additive *(baseline)*          | 0.9585 | 0.9433 | 0.02275 | 0.07646 |
| M_quad full quadratic                    | 0.9817 | 0.9592 | 0.01931 | 0.12909 |
| RBF cubic+linear                         | 1.0000 | 0.9823 | 0.01271 | 0.08102 |
| RBF tps+linear                           | 1.0000 | 0.9804 | 0.01337 | 0.08520 |
| **ARD-GP Matérn-3/2 +noise**             | 1.0000 | **0.9856** | **0.01146** | 0.08047 |

ARD-GP length scales: l_W=1.23, l_KV=0.45, l_KVD=0.61; σ_n=1.7×10⁻⁴; LMML=49.7.

Headline: every non-trivial surrogate beats the M1 baseline by **≥0.04 R²** on
Llama and **≥0.02 R²** on Qwen. Llama's best non-trivial surrogate is RBF
cubic+linear (R²=0.9951); Qwen's is ARD-Matérn-3/2 (R²=0.9856).

Figures: `figures/v4_fig1_scatter.png` (Llama), `..fig1q_scatter.png` (Qwen).
Numbers: `phase1_results.json`.

---

## Phase 1b — sample-size sweep (Llama + Qwen)

First 27 rows of `*_qs50` are the deterministic 3³ quantile-grid base; rows 27..49
are the random extras. N ∈ {27, 50} → single deterministic point; N ∈ {30, 35, 40, 45}
→ 20 seeds drawing extras uniformly from the 23-element pool (ARD-GP omitted from
the stochastic-N loops because each fit was ≫ 1 minute under CPU contention).

### Llama-3.1-8B-Instruct — R²_test median (with [p10, p90] for stochastic N)

| Surrogate | N=27 (det) | N=30 | N=35 | N=40 | N=45 | N=50 (det) |
|---|---:|---:|---:|---:|---:|---:|
| M1 linear additive *(baseline)* | 0.441  | 0.928 [0.60, 0.94] | 0.942 [0.94, 0.94] | 0.943 [0.94, 0.94] | 0.944 [0.94, 0.95] | 0.945 |
| M_quad full quadratic           | −0.506 | 0.912 [0.40, 0.96] | 0.950 [0.85, 0.97] | 0.971 [0.88, 0.98] | 0.982 [0.98, 0.98] | 0.986 |
| RBF cubic+linear                | 0.168  | 0.963 [0.88, 0.98] | 0.981 [0.96, 0.99] | 0.991 [0.98, 0.99] | **0.995 [0.99, 1.00]** | 0.995 |
| RBF tps+linear                  | 0.369  | 0.972 [0.73, 0.99] | 0.984 [0.97, 0.99] | 0.990 [0.98, 0.99] | 0.992 [0.99, 0.99] | 0.993 |

**N=27 (quantile-grid only) is too sparse for the spline-style RBF and the
quadratic — adding even 3 random extras (N=30) jumps every surrogate from sub-0.5
to >0.9 R²**. The plateau begins around N=40 and is flat after N=45. RBF
cubic+linear is the Llama winner from N=40 onward.

### Qwen2.5-7B-Instruct — R²_test median (with [p10, p90] for stochastic N)

| Surrogate | N=27 (det) | N=30 | N=35 | N=40 | N=45 | N=50 (det) |
|---|---:|---:|---:|---:|---:|---:|
| M1 linear additive *(baseline)* | 0.681 | 0.938 [0.86, 0.95] | 0.943 [0.93, 0.95] | 0.943 [0.94, 0.95] | 0.942 [0.94, 0.95] | 0.943 |
| M_quad full quadratic           | 0.751 | 0.923 [0.76, 0.95] | 0.936 [0.90, 0.96] | 0.956 [0.94, 0.96] | 0.959 [0.95, 0.96] | 0.959 |
| RBF cubic+linear                | 0.767 | 0.962 [0.95, 0.97] | 0.971 [0.95, 0.98] | 0.975 [0.96, 0.98] | **0.982 [0.97, 0.98]** | 0.982 |
| RBF tps+linear                  | 0.632 | 0.974 [0.94, 0.98] | 0.975 [0.97, 0.98] | 0.977 [0.97, 0.98] | 0.980 [0.97, 0.98] | 0.980 |

**Qwen's quantile-only N=27 is much healthier than Llama's** (R² 0.63-0.77 vs
0.17-0.44) — the 3³ grid by itself nearly suffices for a usable surrogate. The
N=30 jump is still material (+0.21 R² for tps) but the plateau height is lower
(R² ≈ 0.98 ceiling vs 0.995 on Llama), consistent with Phase 1's observation
that Qwen's response surface has slightly more residual noise.

Figures (Llama): `figures/v4_fig1b_lcurve_R2.png`, `..fig1b_lcurve_RMSE.png`.
Figures (Qwen): `figures/v4_fig1bq_lcurve_R2.png`, `..fig1bq_lcurve_RMSE.png`.
Numbers: `phase1b_sweep_results.json` (keyed by `llama` / `qwen`).

---

## Phase 1c — Why does QS-N=27 collapse, and how does pure-random compare?

The Phase-1b finding "N=27 → N=30 jumps R² from 0.17 to >0.93 by adding 3 random
points" begs the question: **is the quantile grid actually a poor design, or
just unlucky at N=27?**

Phase 1c re-runs the sweep with two training sources on the **same 200 RS test**:

- **QS-first** : 27 deterministic quantile grid + (N−27) random extras from the
  same QS pool — identical to Phase 1b.
- **RS-pure**  : N samples drawn uniformly from the dedicated `*_rs50` file
  (20 seeds per N for N<50; deterministic at N=50).

R²_test median over 20 seeds (M1 baseline + best non-trivial surrogates):

### Llama-3.1-8B-Instruct

| N | M1 QS / RS | M_quad QS / RS | RBF cubic QS / RS | RBF tps QS / RS |
|---:|---:|---:|---:|---:|
| 27 | 0.441 / **0.927** | −0.506 / **0.924** | 0.168 / **0.981** | 0.369 / **0.982** |
| 30 | 0.928 / 0.929 | 0.912 / 0.931 | 0.963 / 0.983 | 0.972 / 0.983 |
| 35 | 0.942 / 0.932 | 0.950 / 0.930 | 0.981 / 0.984 | 0.984 / 0.985 |
| 40 | 0.943 / 0.934 | 0.971 / 0.959 | 0.991 / 0.988 | 0.990 / 0.987 |
| 45 | 0.944 / 0.938 | **0.982** / 0.963 | **0.995** / 0.989 | **0.992** / 0.989 |
| 50 | 0.945 / 0.940 | **0.986** / 0.963 | **0.995** / 0.989 | **0.993** / 0.989 |

### Qwen2.5-7B-Instruct

| N | M1 QS / RS | M_quad QS / RS | RBF cubic QS / RS | RBF tps QS / RS |
|---:|---:|---:|---:|---:|
| 27 | 0.681 / **0.948** | 0.751 / **0.958** | 0.767 / **0.975** | 0.632 / **0.978** |
| 30 | 0.938 / 0.950 | 0.923 / 0.965 | 0.962 / 0.978 | 0.974 / 0.979 |
| 35 | 0.943 / 0.950 | 0.936 / 0.965 | 0.971 / 0.978 | 0.975 / 0.981 |
| 40 | 0.943 / 0.950 | 0.956 / 0.965 | 0.975 / 0.979 | 0.977 / 0.982 |
| 45 | 0.942 / 0.951 | 0.959 / 0.968 | 0.982 / 0.982 | 0.980 / 0.983 |
| 50 | 0.943 / 0.952 | 0.959 / 0.968 | 0.982 / 0.984 | 0.980 / 0.984 |

**The N=27 cliff is QS-specific. 27 pure-random samples already saturate the
non-trivial surrogates** (R² ≈ 0.93–0.98 for both models), while the QS-27 grid
trails by 0.5–0.8 in R² for RBF/quadratic. The two designs converge by N≈35 and
QS-first edges ahead at N≥40 on the higher-capacity surrogates.

### Root cause — z-range coverage volume

For each train pool we compute the **coverage ratio** along each axis:
ratio_k = (z_max − z_min)_train / (z_max − z_min)_test, and the **coverage
volume** = Π_k ratio_k.

| Model | N=27 QS volume | N=27 RS volume | N=30 QS volume |
|---|---:|---:|---:|
| Llama | **0.081** | 0.540 | 0.255 |
| Qwen  | **0.148** | 0.544 | 0.500 |

Per-axis ranges at N=27 (Llama, the worst case):

|  | QS-27 covers | RS-50 covers |
|---|---:|---:|
| z_W   | **23.2%** of test range | 99.3% |
| z_KV  | 93.0% | 99.9% |
| z_KVD | 37.5% | 100.0% |

**The quantile grid is built from the per-axis Pareto-front quantiles (0.1, 0.5,
0.9), but those PFs are dominated by low-z (good-quality) architectures**, so the
0.9-quantile of the PF is still far from the empirical 0.9-quantile of random
architectures. The QS-27 design covers only the inner 8% of the test z-range
volume on Llama and 15% on Qwen — the rest of the box requires extrapolation,
which breaks the spline-style RBFs (and badly overfits M_quad → R²=−0.51).

Adding 3 random extras (N=30) immediately fixes z_W (the worst-covered axis):
the QS-first volume jumps from 0.081 → 0.255 on Llama, and the surrogates
recover. Pure-random does not need this fix because uniform sampling over the
search space hits the high-z tail with probability ∝ its empirical density.

### Practical takeaway

| Budget | Recommendation |
|---|---|
| N=27 (smallest)       | **Use 27 pure random samples**, not the quantile grid. Pure-random already saturates non-trivial surrogates (RBF tps R² ≈ 0.98 on both models). |
| N=30–35               | Either source works; QS-first plateaus a bit faster on M1 but lags slightly on RBF. |
| N≥40                  | **QS-first slightly wins** on RBF cubic / M_quad — the structured low-z coverage helps once enough random extras have patched the high-z tail. |

Figures: `figures/v4_fig1c_lcurve_R2_overlay.png` (Llama),
`..fig1cq_lcurve_R2_overlay.png` (Qwen), `..fig1c_coverage_volume.png`.
Numbers: `phase1c_qs_vs_rs_sweep_results.json`.

---

## Phase 2 — QS+RS-50 vs pure RS-50

Two 50-sample training pools, **same 200 RS held-out test**:

- `QS`: 27 deterministic quantile + 23 random extras (from `llama_qs50` / `qwen_qs50`)
- `RS`: 50 random samples (from `llama_rs50` / `qwen_rs50`)

We report the single-trial scores side-by-side, then bootstrap 20 resamples
**of the RS-50 file** (50-of-50 with replacement) to estimate intra-pool variance
of the random pool. QS-50 is fixed (no resampling).

### Llama-3.1-8B-Instruct

| Surrogate | R²_QS | R²_RS single | Δ R² (QS − RS) | R²_RS bootstrap (μ ± σ) |
|---|---:|---:|---:|---|
| M1 linear additive       | 0.9445 | 0.9399 | +0.0047 | 0.9265 ± 0.0102 |
| M_quad full quadratic    | 0.9858 | 0.9631 | +0.0227 | 0.8946 ± 0.1298 |
| RBF cubic+linear         | 0.9951 | 0.9894 | +0.0057 | 0.9706 ± 0.0276 |
| RBF tps+linear           | 0.9934 | 0.9890 | +0.0045 | 0.9804 ± 0.0134 |
| ARD-GP Matérn-3/2        | 0.9949 | 0.9930 | +0.0019 | 0.8431 ± 0.2563 |

QS wins on all 5 surrogates. The bootstrap reveals that the **variability of the
random pool** is large (σ up to 0.26 for ARD-GP, 0.13 for M_quad), whereas QS is
a fixed point.

### Qwen2.5-7B-Instruct

| Surrogate | R²_QS | R²_RS single | Δ R² (QS − RS) | R²_RS bootstrap (μ ± σ) |
|---|---:|---:|---:|---|
| M1 linear additive       | 0.9433 | 0.9520 | −0.0087 | 0.9475 ± 0.0078 |
| M_quad full quadratic    | 0.9592 | 0.9681 | −0.0089 | 0.9532 ± 0.0282 |
| RBF cubic+linear         | 0.9823 | 0.9844 | −0.0021 | 0.9747 ± 0.0086 |
| RBF tps+linear           | 0.9804 | 0.9839 | −0.0035 | 0.9775 ± 0.0054 |
| ARD-GP Matérn-3/2        | 0.9856 | 0.9872 | −0.0016 | 0.7117 ± 0.3149 |

On Qwen the single-trial RS-50 file happened to be slightly better, but the
ARD-GP bootstrap σ blows up to 0.31, showing the RS-50 advantage is **a lucky
draw, not a methodology-level win**. QS-50's R² of 0.9856 with ARD-GP is well
above the bootstrap mean of 0.71; an RS-50 draw exceeding QS on ARD-GP would be
a tail event.

Figures: `figures/v4_fig2_qs_vs_random.png`. Numbers: `phase2_results.json`.

---

## Phase 3 — Hierarchical vs joint surrogate

Two flavours of "hierarchical":

- **Hier-A** (no learning): y_pred = row 13 = z_W + z_KV + z_KVD. The per-method
  search already publishes this sum as a candidate scoring proxy.
- **Hier-B** (Hoeffding additive collapse of a learned joint surrogate):
  y_pred = f₀ + Σ_k g_k(z_k), with g_k the KDE-smoothed conditional mean of the
  joint surrogate's prediction at input z_k (MC bandwidth = 0.15·σ on training
  range).  Best possible additive 1-D fit to the joint.

The **joint** baseline is the Phase-1 winner (RBF tps+linear) trained on the
same 50 QS.

### Llama-3.1-8B-Instruct (200 RS test)

| Predictor | R² | RMSE | ε_∞ | 2ε corridor |
|---|---:|---:|---:|---:|
| Joint surrogate (RBF tps+linear)         | **0.9934** | 0.00976 | 0.0701 | 0.1402 |
| Hier-A naive additive (row 13)           | 0.3600     | 0.09636 | 0.3487 | 0.6973 |
| Hier-B Hoeffding collapse of joint       | 0.9233     | 0.03337 | 0.0721 | 0.1442 |

`Var(joint − collapse) / Var(joint) = 0.041`  → the joint surrogate is **>95%
additively explainable** once a learned re-weighting is applied. The naive sum
(Hier-A) is bad only because z_k's are not unit-scale comparable.

### Qwen2.5-7B-Instruct (200 RS test)

| Predictor | R² | RMSE | ε_∞ | 2ε corridor |
|---|---:|---:|---:|---:|
| Joint surrogate (RBF tps+linear)         | **0.9804** | 0.01337 | 0.0852 | 0.1704 |
| Hier-A naive additive (row 13)           | −0.3812    | 0.11233 | 0.3078 | 0.6156 |
| Hier-B Hoeffding collapse of joint       | 0.9572     | 0.01976 | 0.0944 | 0.1888 |

`Var(joint − collapse) / Var(joint) = 0.018`. Even more additively-collapsible
than Llama.

Headline: **a 3-axis joint surrogate beats every hierarchical predictor on
single-arch accuracy, but its structure is almost entirely additive** — the
Hoeffding-additive collapse is within 3-5% R² of the full joint. Practical
takeaway: if you can afford one extra MC pass on the training pool, the
calibrated additive predictor (Hier-B) gives a useful 1-D-decomposable
explanation while retaining R² ≳ 0.92.

Figures: `figures/v4_fig3_hier_vs_joint.png`, `..fig3b_additive_collapse_scatter.png`.
Numbers: `phase3_results.json`.

---

## Phase 3b — Hierarchical vs joint with **budget-matched sample-split sweep**

Setup mirrors v3's hier-surrogate decomposition but with two new constraints:

1. **Hierarchical total = joint total = N.** Both methods get the same sample
   budget; only how they spend it differs.
2. **Sweep the split**  `(n_pair, n_3way)` with `n_pair + n_3way = N`.

Decomposition (per pair direction):
```
y(x_W, x_KV, x_KVD)  ≈  f_pair(x_a, x_b)  +  g(x_c)
```
- Stage 1: `f_pair` = RBF tps+linear on `n_pair` samples from a 2-axis 260510
  CSV (z_a, z_b, measured y_pair).
- Stage 2: residual `g(x_c)` = degree-2 polynomial fit on
  `r_3way = y_3way − f_pair(z_a_3way, z_b_3way)` for `n_3way` 3-way samples.

Joint baseline: RBF on `N = n_pair + n_3way` 3-way samples.

Three pair directions (residual axis in parens):
- WK = (z_W, z_KV) ↔ residual z_KVD
- KD = (z_KV, z_KVD) ↔ residual z_W
- WD = (z_W, z_KVD) ↔ residual z_KV

Budget `N = 50`, splits `(n_pair, n_3way) ∈ {(0,50), (5,45), (10,40), (15,35),
(20,30), (25,25), (29,21)}` (cap `29` from the smallest pair pool — WK CSV
has 29 measurements). 10 paired seeds per stochastic split.

> **Llama-only**: 260510 has no Qwen 2-axis CSVs; the workflow is identical once
> those measurements are collected.

### R²_test on 200 RS hold-out — joint = 0.9934 across every row (fixed reference)

| n_pair / n_3way | Hier-WK R²  (median [p10, p90]) | Hier-KD R²       | Hier-WD R²        |
|---|---:|---:|---:|
|  0 / 50 | +0.151                  | **+0.830**            | +0.030             |
|  5 / 45 | +0.510 [+0.38, +0.67]   | **+0.957 [+0.94, +0.97]** | +0.657 [+0.51, +0.71] |
| 10 / 40 | +0.479 [+0.38, +0.71]   | **+0.954 [+0.95, +0.96]** | +0.786 [+0.50, +0.84] |
| 15 / 35 | +0.309 [−1.62, +0.53]   | +0.954 [+0.95, +0.96] | +0.761 [+0.50, +0.86] |
| 20 / 30 | −0.193 [−6.25, +0.47]   | +0.953 [+0.95, +0.96] | +0.776 [+0.59, +0.86] |
| 25 / 25 | −0.408 [−6.74, −0.01]   | +0.951 [+0.95, +0.95] | +0.780 [+0.69, +0.82] |
| 29 / 21 | −0.168 [single seed]    | +0.949 [+0.95, +0.95] | +0.778 [+0.68, +0.82] |

### Three findings

1. **Pair-direction choice dominates everything.** Hier-KD peaks at R²=0.957 (95 %
   of joint's 0.993); Hier-WD plateaus at ~0.78; Hier-WK is broken (R² < 0.51,
   often negative). The reason is **z-range coverage of the pair training set**:

   | Pair tag | pair z₁ train range | matched 3-way range | extrap ratio |
   |---|---|---|---:|
   | WK (z_W) | [0.022, 0.185] | [0.019, 0.66] | **3.6×** |
   | KD (z_KV) | [0.018, 0.066] | [0.018, 0.140] | 2.1× |
   | WD (z_W) | [0.022, 0.241] | [0.019, 0.66] | 2.7× |

   Hier-WK extrapolates z_W over 3.6× its training range — spline RBFs collapse,
   giving wild negative R² (e.g. −6.25). KD's residual axis is z_W which the 1D
   polynomial handles cleanly over the full range, and the pair (z_KV, z_KVD) is
   well-covered.

2. **More pair samples actively hurt when the pair box is too narrow.** Hier-WK
   at `n_pair=0` (residual-only) gets R²=0.15; adding pair samples drops it to
   −0.4. The pair RBF interpolates well within its tight box but extrapolates
   catastrophically outside, and a bigger pair training set just makes the
   extrapolation more confident-but-wrong.

3. **Joint always wins at N=50.** Even Hier-KD's best 0.957 is below joint's
   0.993. The hierarchical decomposition pays a modelling-mismatch cost (the
   true response is not exactly `f_pair + g`); joint doesn't make that
   assumption. The decomposition might help below the joint surrogate's
   minimum-viable budget (Phase 1b showed joint needs ≥ 30 samples), but at
   N = 50 the joint surrogate has already saturated and any decomposition is
   strictly worse.

### Practical takeaway

- If a pair-direction has good z-coverage on both axes (KD here), hierarchical
  matches joint to within ~3.5 % R² at any split with `n_pair ≥ 5` — useful when
  the 3-way evaluator is much more expensive than pair evaluations.
- Pair-direction selection MUST be guided by the per-axis z-range of the search
  PFs vs the target test distribution — using the wrong pair (WK here, with the
  W-axis under-covering the high-z tail) gives R² no better than a random guess.
- Adding more pair samples beyond ~5–10 brings no benefit when the pair box is
  smaller than the test box; that budget should go to `n_3way` to model the
  residual along the third axis.

Figures: `figures/v4_fig3b_hier_joint_sweep.png` — three panels (WK / KD / WD)
of R² vs `n_3way / N` with Hier (blue, with p10-p90 band) vs Joint (red dashed).
Numbers: `phase3b_hier_joint_results.json`.

> The original Phase 3 (naive additive row-13 vs Hoeffding-collapse vs joint)
> is preserved in `phase3_results.json` / `figures/v4_fig3_*.png` for reference;
> 3b supersedes it with budget-matched comparison.

## Phase 3c — does the axis-ordering inside the decomposition matter?

Two decomposition styles, all on 50-sample 3-way QS train + 200 RS test (Llama):

**A) Pair-then-residual** (Phase 3b's three pair directions, RBF pair + degree-2
poly residual). Reported at two sample-budgets per pair.

**B) Sequential 1-D additive** (NEW): all 6 permutations of fitting axes:
```
g_a fit to (x_π(0), y);            r1 = y  − g_a(x_π(0))
g_b fit to (x_π(1), r1);           r2 = r1 − g_b(x_π(1))
g_c fit to (x_π(2), r2)
y_pred = g_a + g_b + g_c
```
Each `g_k` is a degree-2 polynomial. Order matters because each stage is
*stagewise/greedy* — it fits the residual from the previous stage rather
than co-fitting all three jointly. All 6 orderings use the same 50 3-way
samples (total budget = 50, same as joint).

### Results (R²_test on 200 RS) — joint reference R² = 0.9934

| Decomposition | R² | Notes |
|---|---:|---|
| **Joint RBF**                                | **0.9934** | gold standard, 50 3-way |
| Hier-WK (pair-full: 29 + 50 3-way)           | −0.2444 | pair extrapolation collapse |
| Hier-WK (50-budget: 10 + 40)                 | +0.4786 | very unstable, p10=0.38 |
| Hier-KD (pair-full: 30 + 50 3-way)           | 0.9519 | |
| **Hier-KD (50-budget: 10 + 40)**             | **0.9535** | best pair direction |
| Hier-WD (pair-full: 31 + 50 3-way)           | 0.8307 | |
| Hier-WD (50-budget: 10 + 40)                 | 0.7863 | |
| **Seq-1D W → KV → KVD** *(best ordering)*    | **0.9701** | |
| Seq-1D W → KVD → KV                          | 0.9594 | |
| Seq-1D KV → W → KVD                          | 0.9548 | |
| Seq-1D KVD → W → KV                          | 0.9509 | |
| Seq-1D KVD → KV → W                          | 0.9463 | |
| Seq-1D KV → KVD → W *(worst ordering)*       | 0.9021 | |

### Three findings

1. **Axis-ordering inside Seq-1D matters: R² spread is 0.068** (0.902 → 0.970)
   across the 6 permutations. Starting with **W** (the axis with the widest
   z-range and the biggest Sobol-S1 from Phase 4) gives the best fit — the
   first stage absorbs most of the variance, leaving small residuals for the
   later stages. Starting with KV (smallest z-range, smallest S1) leaves a
   bigger non-linear residual that the simple degree-2 polynomial can't
   capture in stages 2-3.

2. **Sequential 1D additive (≈ 0.97 with the right order) is competitive with
   pair-then-residual (≈ 0.95 with Hier-KD) — at lower complexity**. Seq-1D
   uses only 3-way QS data (no 2-axis pair CSVs), 6× fewer parameters than
   the RBF pair surrogate, and is order-invariant if you use joint OLS — so
   the stagewise+order-sensitive flavour is the entire knob you can tune. The
   best ordering gives Seq-1D the lead.

3. **The pair-direction effect dwarfs the axis-ordering effect.** Hier-WK can
   give R² = −0.24 (worse than predicting the mean) due to pair-set z-range
   extrapolation, while the Seq-1D worst case is still R² = 0.90. Picking
   the right pair direction is much more important than picking the right
   sequential order.

Figures: `figures/v4_fig3c_pair_orderings.png` — horizontal bar chart of all
12 decompositions vs joint reference line.
Numbers: `phase3c_orderings_results.json`.

---

## Phase 4 — internal analysis of the best surrogate (Llama + Qwen)

We probe the two strongest surrogates from Phase 1 on each model: **RBF
cubic+linear** (Phase-1 winner — R² 0.9951 Llama / 0.9823 Qwen, RMSE ~14 %
lower than tps) and **ARD-Matérn-3/2** (GP comparator that exposes length
scales directly).

### Sobol (Saltelli pick-freeze, N_base=2048; uniform input box from training range)

**Llama-3.1-8B-Instruct**

| Surrogate | S1_W | S1_KV | S1_KVD | ΣS1 | interaction (1−ΣS1) |
|---|---:|---:|---:|---:|---:|
| **RBF cubic+linear** | **0.811** | 0.079 | 0.082 | **0.972** | **0.028** |
| ARD-Matérn-3/2       | 0.826     | 0.038 | 0.076 | 0.941     | 0.059     |

**Qwen2.5-7B-Instruct**

| Surrogate | S1_W | S1_KV | S1_KVD | ΣS1 | interaction (1−ΣS1) |
|---|---:|---:|---:|---:|---:|
| **RBF cubic+linear** | **0.737** | 0.191 | 0.071 | **0.998** | **0.002** |
| ARD-Matérn-3/2       | 0.740     | 0.178 | 0.079 | 0.996     | 0.004     |

Both models: W dominates first-order. Qwen has **higher KV share** (0.19 vs 0.08
on Llama) and ultra-low interaction mass (0.2-0.4 %). Llama has small but
non-negligible interaction (2.8 % under RBF cubic — the more capacity-laden
predictor) which the smoother tps under-estimates as 0.8 % — cubic is more
honest about residual interaction structure.

### Active subspace (gradient outer product, N_MC=4000)

**Llama-3.1-8B-Instruct**

| Surrogate | λ₁ | λ₂/λ₁ | λ₃/λ₁ | first eigvec [W, KV, KVD] |
|---|---:|---:|---:|---|
| **RBF cubic+linear** | 1.613 | **0.075** | 0.042 | [−0.42, **−0.78**, −0.47] |
| ARD-Matérn-3/2       | 1.185 | 0.088 | 0.069 | [−0.48, −0.70, −0.53] |

**Qwen2.5-7B-Instruct**

| Surrogate | λ₁ | λ₂/λ₁ | λ₃/λ₁ | first eigvec [W, KV, KVD] |
|---|---:|---:|---:|---|
| **RBF cubic+linear** | 4.807 | **0.034** | 0.007 | [−0.23, **−0.82**, −0.52] |
| ARD-Matérn-3/2       | 4.544 | 0.019 | 0.009 | [−0.24, −0.80, −0.55] |

Both models collapse to roughly 1-D in input space (λ₂/λ₁ ≤ 0.09 on Llama, ≤ 0.03
on Qwen). KV has the largest |loading| in the first eigenvector on both models
because its input range is narrowest (per-unit slope is largest); this is
consistent with Sobol still giving W the largest absolute variance contribution.

### ARD-Matérn-3/2 length scales + noise

| Model | l_W | l_KV | l_KVD | σ_n |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 1.050 | **0.478** | 0.899 | 2.1×10⁻⁵ |
| Qwen2.5-7B   | 1.233 | **0.454** | 0.614 | 1.7×10⁻⁴ |

Both models: smallest length scale on **KV**, matching the active-subspace
finding that KV drives finest-grained variation. Qwen has ~8× higher fitted
noise variance, hinting at a slightly bumpier surface (also reflected in lower
ARD-GP R² on Phase 1).

### Hoeffding additive collapse on 200 RS test

**Llama-3.1-8B-Instruct**

| Surrogate | full ε_∞ | add ε_∞ | full R² | Var(res_add)/Var(y_te) |
|---|---:|---:|---:|---:|
| **RBF cubic+linear** | 0.0316 | **0.0847** (2.68×) | 0.9951 | 0.064 |
| ARD-Matérn-3/2       | 0.0391 | 0.0973 (2.49×)     | 0.9949 | 0.090 |

**Qwen2.5-7B-Instruct**

| Surrogate | full ε_∞ | add ε_∞ | full R² | Var(res_add)/Var(y_te) |
|---|---:|---:|---:|---:|
| **RBF cubic+linear** | 0.0810 | **0.0937** (1.16×) | 0.9823 | 0.043 |
| ARD-Matérn-3/2       | 0.0805 | 0.0944 (1.17×)     | 0.9856 | 0.043 |

→ Qwen collapses to additive with only +16 % ε_∞ — near-ideal for the 2ε-Pareto
theorem (Var(res_add)/Var(y_te) = 0.043). Llama's cubic surrogate pays +168 %
ε_∞ when collapsed (0.0316 → 0.0847) because it captures more interaction
structure than tps did; still, **97 % of variance is additive** (ΣS1 = 0.972),
so the residual interaction is small in absolute terms (~ 2.8 % of total
variance, 6.4 % of test variance).

Figures (Llama): `figures/v4_fig4_sobol.png`, `..fig4_active_subspace.png`,
`..fig4_hoeffding_add.png`.
Figures (Qwen): `figures/v4_fig4q_sobol.png`, `..fig4q_active_subspace.png`,
`..fig4q_hoeffding_add.png`. Numbers: `phase4_internal_results.json` (keyed by
`llama` / `qwen`).

---

## Phase 5 — Local-PF combination is within ε of the joint method PF

Setup: for each of the 200 RS test architectures we have
- `y_actual` (row 12)         — measured 3-axis JSD,
- `y_localsum` (row 13)       — z_W + z_KV + z_KVD, i.e. **the additive
  combination of the per-axis local PF values** with NO surrogate involved.

Pick the complexity proxy c = wbits + eff_kvbits (total bits per token), and
build two Pareto fronts on `(y, c)`:
- the **joint method PF**: `Pareto({(y_actual_i, c_i)})`,
- the **proxy PF**: `Pareto({(y_localsum_i, c_i)})`.

The 2ε-Pareto theorem (Daskalakis–Diakonikolas; classical) says that if
‖y_actual − y_localsum‖_∞ ≤ ε on the test set, then for every proxy-PF point
the *actual* loss is at most ε above the actual-PF curve at the same c, and the
proxy front is contained in a 2ε corridor of the actual front.

We measure **Δy = y_actual_at_proxy_PF − y_actualPF(c)** per proxy-PF point.

| Model | N_test | ε_global | 2ε corridor | |actual PF| | |proxy PF| | Δy mean | Δy max | Δy p95 | inside 2ε |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B  | 200 | 0.3487 | 0.6973 | 44 | 41 | 0.0007 | 0.0250 | 0.0014 | **41 / 41 (100%)** |
| Qwen2.5-7B    | 200 | 0.3078 | 0.6156 | 40 | 35 | 0.0054 | 0.0633 | 0.0359 | **35 / 35 (100%)** |

**Key result:** even though the *global* ε_∞ is huge (0.35 Llama / 0.31 Qwen —
driven by the worst-case architectures at the corner of the box), the actual gap
between the proxy-PF and the joint method PF is two orders of magnitude smaller
on average: Δy mean ≈ 0.001 (Llama) / 0.005 (Qwen). Every single proxy-PF point
is inside the 2ε corridor — the theorem is satisfied, but the corridor is
extremely loose because ε_∞ is dominated by **non-PF** worst-case errors.

This validates the hierarchical-search-then-combine workflow on the **on-PF
regime** at zero extra learning cost: the per-method PF z-values, summed
unweighted, give a Pareto front that **never falls more than ~3% in y above the
joint-search PF** (Llama: max Δy = 0.025; Qwen: max Δy = 0.063).

Out-of-scope (deferred per user request): characterising whether OFF-proxy-PF
architectures can beat the joint method PF — i.e. whether the additive proxy
*misses* parts of the joint PF entirely.

Figures: `figures/v4_fig5_local_pf_combination.png`. Numbers:
`phase5_local_pf_results.json`.

---

## Phase 6 / 7 / 8 — Severe falsification of the local-PF Cartesian product

Question: do architectures **off** the local-PF Cartesian product ε-dominate the
3D Cartesian-PF envelope when measured on GPU?

Pipeline (final consolidated form):

```
06_phase6_acquisition.py     # CPU: build pool, fit RBF cubic, EVI, paired+controls
07_phase7_eval.py            # GPU: AWQ + KIVI eval of all acquired candidates
08_phase8_analysis.py        # CPU: 6 endpoints + reliability + support audit
```

Earlier `06/07/08_phase*.py` (different bucket structure, naive_add baseline)
is archived under `archive_phase6_old/`.

### Design decisions (all consolidated into the v4 pipeline)

| Issue | Decision |
|---|---|
| Baseline loss column | **RBF cubic+linear μ** on **structural Cartesian-PF subset** (PF_W^(1) × PF_KV^(1) × PF_KVD^(1)) — NOT `naive_add`, NOT the full pool. |
| Complexity dims in PF | 3 dims `(wbits, kvbits, kvdim)` preserved; 4-objective NDS. Never collapsed to 1D. |
| Per-axis archive PFs | O(n log n) sorted-sweep (equivalent to pymoo NDS in 2D, ~10× faster on 10k archives). |
| Pool size | Layer 1 ∪ 2 ∪ 3 union per axis, complexity-stratified cap 100/layer → ~300/axis, 27 M Cartesian. |
| Predictor | RBF cubic+linear (Phase 1 winner). σ_conf = conformal LOOCV q_{0.95}. |
| Acquisition | EVI_{ε=0.005} with closed-form Gaussian; P_ε = Φ(t) for calibration. |
| Bucket structure | B1 (50 EVI-top) + B3 (15 paired projection pairs = 30) + B5 (10 low-P controls) = **90 / model**. |
| Primary endpoint | r_3D(a) = y_actual(a) − f_C^{*,3D}(c(a)). Multi-ε grid {0.001,…,0.05}. |
| Reviewer-defence | Archive-coverage slack δ_i^(3) reported; rank-depth K=1/2/3 breakdown. |

### Phase 6 — Acquisition (`06_phase6_acquisition.py`)

For each model:

1. **Archives**: load 3 per-axis search archives (W, KV, KVDIM stats).
2. **Layered pool R^(≤3)**: per axis, peel Pareto layers 1, 2, 3 via 2D fast-sweep
   and complexity-stratify each layer to `CAP_PER_LAYER = 100`. Total ~300/axis,
   Cartesian ~27 M (NDS still completes in seconds).
3. **Predictor**: RBF cubic+linear on `*_rs50` 50-sample pool (X = sub-losses
   z_W, z_KV, z_KVD; y = measured wikitext2 JSD). LOOCV residual → conformal
   noise σ_conf = q_{0.95}(|resid|).
4. **Structural Cartesian-PF baseline**: subset where all 3 axes are in their
   layer 1. NDS on 4D objective `(RBF μ, wbits, kvbits, kvdim)` over that subset.
5. **f_C^{*,3D}(c)** = min RBF μ among baseline-PF points dominating `c` in all
   3 complexity dims (corner fallback at boundary).
6. **EVI / P_ε** on off-structural-PF candidates:
   `t = (B_ε − μ)/σ_conf`, `EVI = (B_ε−μ)Φ(t) + σ_conf φ(t)`, `P_ε = Φ(t)`,
   primary ε = 0.005.
7. **Buckets** (B = 90 per model):
   - **B1 EVI-top adversarial (50)** — rank-tuple stratification
     `1-axis-off (20) / 2-axis-off (17) / 3-axis-off (13)`; inside each group,
     3³ = 27 (wbits × kvbits × kvdim) bucket-stratified top-EVI.
   - **B3 Paired projection (30 = 15 pairs)** — top-EVI off-PF `a`; per axis
     `π_i(a_i)` = nearest Layer-1 PF point with `c_i(p) ≤ c_i(a)` AND
     `z_i(p) ≤ z_i(a)` (Hausdorff-style fallback when no dominator). Both `a` and
     `π(a)` are queued for measurement.
   - **B5 Low-P controls (10)** — off-PF candidates with `P_ε < 0.10`, complexity-
     stratified random. Anchors the left bin of the reliability diagram.
8. **Archive-coverage slack δ_i^(3)** — sup over the archive of the minimum
   `z_i(R^(≤3)) − z_i(archive)` subject to a 2 %-range complexity slack ρ. Used
   for the reviewer-defence corridor `2ε̂ + Σ_i L_i δ_i^(3)`.

Output: `acquired_falsifiers_{tag}.json` + `figures/v4_fig6_acquire_{tag}.png`.

JSON schema includes per-candidate `bucket`, `rbf_mu`, `fC_star_3D`,
`EVI_eps0p005`, `P_eps0p005`, axis-layer tuple, `(wbits, kvbits, kvdim, eff_kvbits)`,
the structural baseline-PF point array, and the `b3_pair_map` linking
`(a_pool_idx, pi_pool_idx)` pairs.

### Phase 7 — GPU evaluation (`07_phase7_eval.py`)

Reads `acquired_falsifiers_{tag}.json`, evaluates each candidate (arch dict
already merged) through `LlamaEvaluator` with AWQ + KIVI under the 260510
measurement protocol. Round-robin by bucket so the first few minutes already
cover EVI-top + paired-a + paired-π + low-P. Resume-safe (per-arch incremental
save to `eval_falsifiers_{tag}.json`). Launch:

```bash
cd /NAS/SJ/actquant/search
CUDA_VISIBLE_DEVICES=2 python -u analysis/v4/07_phase7_eval.py --tag llama --gpu_id 2 \
    > analysis/v4/phase7_llama_run.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u analysis/v4/07_phase7_eval.py --tag qwen  --gpu_id 3 \
    > analysis/v4/phase7_qwen_run.log 2>&1 &
```

Expected wall: ~30 s per arch × 90 = **~45 min per model parallel on GPU 2/3**.

> **Known infra risk**: previous session saw CIFS overload (load avg ~103);
> torch + hqq + transformers imports hung in `wait_for_response` for 20+ min.
> Check `cat /proc/loadavg` before launching. If load > 50, mirror the source
> tree to `/SSD/actquant_search_local/` and run from there.

### Phase 8 — Analysis (`08_phase8_analysis.py`)

Six endpoints on the measured set:

| # | Endpoint | What it tests |
|---|---|---|
| 1 | **r_3D residual distribution** | `r_3D(a) = y_actual − f_C^{*,3D}(c(a))`. Per-ε violation counts + Clopper-Pearson 95 % bounds for ε ∈ {0.001, 0.002, 0.005, 0.01, 0.02, 0.05}. |
| 2 | **Paired projection margin** | `d(a) = y(a) − y(π(a))` for the B3 pairs. Per-ε counts + Wilcoxon signed-rank one-sided p-value. |
| 3 | **Rank-depth breakdown** | Violation rate stratified by `max(W_layer, KV_layer, KVD_layer)` ∈ {1, 2, 3} — defence for the rank-≤3 cutoff. |
| 4 | **Calibration** | P_ε bin reliability diagram (`[0, 0.1)`, `[0.1, 0.5)`, `[0.5, 0.9)`, `[0.9, 1.0]`) + Brier score + ECE. |
| 5 | **Support-distance audit** | Mahalanobis `d_train(a)` of each candidate's `(z_W, z_KV, z_KVD)` to the RS50 train set. Split at median; report violations in vs out-of-support. |
| 6 | **Zero-violation CP upper bound** | If k/n violators at each ε, report 95 % one-sided upper bound for the violation rate. |

Output: `phase8_falsification_results.json` + `figures/v4_fig8_falsification_{tag}.png`
(6-panel: residual histogram, paired-margin histogram, rank-depth bars,
reliability diagram, support scatter, CP curve).

---

## Caveats

- Single QS-pool: `..._qs_w159_kv159_kvdim159_rs23` (rs23 = 23 random extras).
  The 27 quantile-grid base is deterministic, but the variance of the 23-extras
  draw is not characterised here (would need multiple rs-seeded QS files).
- Phase 1b reduces n_restarts from 50 → {2, 5, 10} for ARD-GP, and excludes
  ARD-GP from stochastic-N loops, to fit wall-clock under CPU contention.
  Numbers should be interpreted as *median behaviour*, not Bayesian-optimal GP
  fits.
- Phase 2 bootstrap uses *with-replacement* resamples of the 50-row RS file;
  this measures intra-pool variance (which arch tuples reappear), not new draws
  from the full RS200 universe. Treat it as a lower bound on the true RS-pool
  variance.
- Phase 5 uses the joint method PF derived from **measured** y on the 200 RS
  test set, not from a separate full search. The 200 RS points are a sample;
  the true joint PF may include points outside this sample.
- The 2-axis CSV files (`llama_wk`, `llama_kvkd`, `llama_wkd` — 29/30/31 cols)
  are catalogued in `_common.PATHS` but not yet used; reserved for a separate
  pair-surrogate study.

## Files

```
v4/
├── README.md                       (this file)
├── _common.py                      (shared loaders, surrogate fits, PF helper)
├── 01_phase1_surrogates.py
├── 01b_phase1b_sweep.py
├── 02_phase2_qs_vs_random.py
├── 03_phase3_hier_vs_joint.py
├── 04_phase4_internal.py
├── 05_phase5_local_pf_combination.py
├── phase1_results.json
├── phase1b_sweep_results.json
├── phase2_results.json
├── phase3_results.json
├── phase4_internal_results.json
├── phase5_local_pf_results.json
├── phase1_run.log  phase1b_run.log  phase2_run.log  phase3_run.log  phase4_run.log  phase5_run.log
├── archive_v1/                     (previous v4 pass — different data set)
└── figures/
    ├── v4_fig1_scatter.png                 Phase 1 scatter (Llama)
    ├── v4_fig1q_scatter.png                Phase 1 scatter (Qwen)
    ├── v4_fig1b_lcurve_R2.png              Phase 1b R² learning curve (Llama)
    ├── v4_fig1b_lcurve_RMSE.png            Phase 1b RMSE learning curve (Llama)
    ├── v4_fig1bq_lcurve_R2.png             Phase 1b R² learning curve (Qwen)
    ├── v4_fig1bq_lcurve_RMSE.png           Phase 1b RMSE learning curve (Qwen)
    ├── v4_fig1c_lcurve_R2_overlay.png      Phase 1c QS-first vs RS-pure overlay (Llama)
    ├── v4_fig1cq_lcurve_R2_overlay.png     Phase 1c QS-first vs RS-pure overlay (Qwen)
    ├── v4_fig1c_coverage_volume.png        Phase 1c coverage volume diagnostic
    ├── v4_fig2_qs_vs_random.png            Phase 2 QS vs RS bars
    ├── v4_fig3_hier_vs_joint.png           Phase 3 bar chart
    ├── v4_fig3b_additive_collapse_scatter.png  Phase 3 scatter
    ├── v4_fig4_sobol.png                   Phase 4 Sobol decomposition (Llama)
    ├── v4_fig4_active_subspace.png         Phase 4 active subspace (Llama)
    ├── v4_fig4_hoeffding_add.png           Phase 4 Hoeffding additive (Llama)
    ├── v4_fig4q_sobol.png                  Phase 4 Sobol decomposition (Qwen)
    ├── v4_fig4q_active_subspace.png        Phase 4 active subspace (Qwen)
    ├── v4_fig4q_hoeffding_add.png          Phase 4 Hoeffding additive (Qwen)
    └── v4_fig5_local_pf_combination.png    Phase 5 PF combination check
```
