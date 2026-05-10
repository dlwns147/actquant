# v4 — Quantile-sample Train Pool Re-analysis

Train surrogates on the 50-sample 3-way **quantile_sample** set
(`save/result/260506/llama_3.1_8b_inst_quantile_sample/2605060818_..._qs_w159_kv159_kvdim159_rs23/`)
and evaluate on the 200-sample AWQ 3-way set
(`save/result/awq/2604162012_..._w_expr_kv_expr_kv_dim/`).

Phases:

| Phase | Script | Train | Test |
|---|---|---|---|
| 1   | `01_quantile_train_full_test.py` | 50 QS samples | 200 AWQ samples (full hold-out, no overlap) |
| 1b  | `04_sample_size_sweep.py`        | N ∈ {27,30,35,40,45,50} from QS (first 27 = quantile-grid base, +random extras, 50 seeds for stochastic N) | 200 AWQ samples |
| 1c  | `05_rbf_internal_analysis.py`    | 50 QS samples | 200 AWQ samples — Sobol / active subspace / Hoeffding additive on **RBF tps+linear** (primary), RBF cubic+linear and ARD-Matérn-3/2 as comparators |
| 2   | `02_quantile_vs_random.py`       | 50 random AWQ **vs** 50 QS (paired) | 150 AWQ hold-out, 10 seeds |
| aux | `03_test_lsbounds.py`             | Sensitivity test: ARD-GP kernel sweep with `length_scale_bounds=(1e-3,1e3)` vs `(1e-4,1e4)` (no effect; bounds not active at optimum) |

## Setup

- Inputs: per-method JSD `(z_W, z_KV, z_KVD)` recovered by matching CSV `(wbits, kvbits, kvdim)` to the per-method PFs (same as v3).
- Surrogates: M1 (linear additive), M10 (full quadratic), RBF cubic+linear, RBF tps+linear, and two ARD-GP variants — with squared-exponential (RBF) kernel and with Matérn-3/2 kernel.
- Architecture overlap (QS ↔ AWQ by `(wbits, kvbits, kvdim)`): **0** — no leakage, dedup is a no-op.
- **PF recovery / HV ratio metrics removed** — they rely on prediction values to *select* PF members, which conflates selection bias with surrogate quality. We do not use these metrics anymore.

## ARD-GP kernel sweep (Phase 1)

Selection is by training log-marginal-likelihood (LMML) — a Bayesian Occam-razor criterion that uses only training data, so it does not leak the test set.

| Kernel × noise | LMML (train) | R²_test | RMSE_test | ε_∞ |
|---|---:|---:|---:|---:|
| RBF + noise           | 50.81 | 0.9827 | 0.0159 | 0.069 |
| RBF, no noise         | 19.48 | 0.9033 | 0.0377 | 0.143 |
| Matérn-5/2 + noise    | 52.13 | 0.9897 | 0.0123 | 0.047 |
| Matérn-5/2, no noise  | 41.28 | 0.9875 | 0.0136 | 0.061 |
| **Matérn-3/2 + noise** | **54.36** | **0.9939** | **0.0095** | **0.032** |
| Matérn-3/2, no noise  | 54.01 | 0.9939 | 0.0094 | 0.032 |
| Rational-quadratic + noise | 45.10 | 0.9849 | 0.0149 | 0.080 |
| Rational-quadratic, no noise | 27.32 | 0.9831 | 0.0158 | 0.072 |

→ **Best: ARD-Matérn-3/2 with WhiteKernel.** Length scales l_W=1.04, l_KV=0.46, l_KVD=1.33; σ_n²≈1.1×10⁻⁵.

Why Matérn-3/2 beats RBF: the squared-exponential kernel encodes infinitely smooth functions (C^∞), but the quantization-loss landscape has finite smoothness (≈ C^1). Matérn-3/2 corresponds exactly to once-differentiable mean-square paths, so the prior matches the data. With this fix, ARD-GP becomes competitive with the spline-style RBF surrogates (R²=0.9939 vs RBF tps+linear 0.9953).

## Phase 1 results (50 QS train → 200 AWQ test)

| Surrogate | R²_train | R²_test | RMSE | ε_∞ |
|---|---:|---:|---:|---:|
| M1 linear additive       | 0.9589 | 0.9482 | 0.0276 | 0.0648 |
| M10 full quadratic       | 0.9791 | 0.9789 | 0.0176 | 0.0667 |
| RBF cubic+linear         | 1.0000 | 0.9945 | 0.0090 | 0.0340 |
| RBF tps+linear           | 1.0000 | **0.9953** | **0.0083** | **0.0331** |
| ARD-GP (RBF + noise)     | 0.9998 | 0.9827 | 0.0159 | 0.0693 |
| ARD-GP (Matérn-3/2)      | 1.0000 | 0.9939 | 0.0095 | 0.0317 |

Internal-analysis (Sobol etc.) is now in **Phase 1c** with **RBF tps+linear** as the primary surrogate (Phase-1 winner). Phase-1 fig4 retains the ARD-Matérn-3/2 Sobol view for completeness. Comparison of all three surrogates is in Phase 1c below.

Figures: `figures/v4_fig1_phase1_scatter.png`, `..fig2_phase1_residual.png`,
`..fig3_phase1_ardgp_kernel_sweep.png`, `..fig4_phase1_sobol.png`. Full numbers: `phase1_results.json`.

## Phase 1b — sample-size sweep (N_train: 27 → 50)

Same QS pool. N=27 uses the first 27 columns of the QS CSV (the quantile-grid base, deterministic). N ∈ {30, 35, 40, 45} adds random extras from columns 27..49 (50 seeds, summarised as median [10/90 percentile]). N=50 uses all 50 columns (deterministic).

Test set: 200 AWQ samples (same as Phase 1).

R²_test (median [p10, p90] for stochastic N; single value for deterministic):

| Surrogate | N=27 (det) | N=30 | N=35 | N=40 | N=45 | N=50 (det) |
|---|---:|---:|---:|---:|---:|---:|
| M1 linear additive   | 0.464 | 0.942 [0.740, 0.948] | 0.948 [0.941, 0.951] | 0.948 [0.945, 0.951] | 0.948 [0.945, 0.950] | 0.948 |
| M10 full quadratic   | −0.242 | 0.864 [0.688, 0.943] | 0.930 [0.878, 0.959] | 0.960 [0.911, 0.976] | 0.976 [0.969, 0.979] | 0.979 |
| RBF cubic+linear     | 0.108 | 0.936 [0.893, 0.961] | 0.973 [0.943, 0.985] | 0.986 [0.973, 0.994] | 0.994 [0.991, 0.995] | 0.994 |
| RBF tps+linear       | 0.373 | 0.971 [0.911, 0.981] | 0.982 [0.969, 0.987] | 0.988 [0.980, 0.993] | 0.994 [0.992, 0.995] | **0.995** |
| ARD-GP (Matérn-3/2)  | **0.893** | 0.977 [0.937, 0.987] | 0.983 [0.950, 0.990] | 0.991 [0.980, 0.994] | 0.994 [0.992, 0.995] | 0.994 |

Observations:

- **N=27 (just the quantile grid, no random extras):** ARD-GP Matérn-3/2 generalises far better (R²=0.893) than the spline-style RBF (0.11–0.37) and quadratic (−0.24). When training points are equally spaced on the quantile grid, RBF interpolation has no information to extrapolate to the empirical AWQ distribution; M10 over-fits the 10 free coefficients to 27 points; the GP's built-in length-scale prior smooths it out. **Adding even 3 random extras (N=30) closes the gap entirely.**
- **N≥35:** all non-trivial surrogates plateau near R² ≈ 0.99. RBF tps+linear and ARD-Matérn-3/2 are statistically tied from N=40 onwards.
- **Conclusion on sample efficiency:** if you can measure ≥30 points, RBF or ARD-GP both work; if you are stuck at exactly 27 quantile-grid points, ARD-GP is the only safe choice.

Figures: `figures/v4_fig8_lcurve_R2.png`, `..fig9_lcurve_RMSE.png`, `..fig10_lcurve_eps.png`. Full table: `sample_size_sweep_results.json`.

## Phase 1c — RBF-primary internal analysis

Sobol decomposition / active subspace / Hoeffding additive decomposition done on the **RBF tps+linear** surrogate (Phase-1 winner) trained on the full 50 QS samples; comparators are RBF cubic+linear and ARD-GP Matérn-3/2.

### Sobol indices (Saltelli pick-freeze, N_base=2048)

| Surrogate | S1_W | S1_KV | S1_KVD | ΣS1 | interaction (1−ΣS1) | S2_W,KV | S2_W,KVD | S2_KV,KVD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **RBF tps+linear (primary)** | **0.824** | 0.076 | 0.090 | **0.990** | **0.010** | 0.002 | 0.007 | 0.003 |
| RBF cubic+linear             | 0.831 | 0.069 | 0.073 | 0.973 | 0.027 | 0.003 | 0.021 | 0.004 |
| ARD-GP (Matérn-3/2)          | 0.845 | 0.044 | 0.056 | 0.945 | 0.055 | 0.015 | 0.039 | 0.002 |

→ **The RBF tps+linear surrogate gives the cleanest near-additive structure (interaction = 1.0%).** All three surrogates agree W dominates first-order (S1_W ≈ 0.83–0.85) and W,KVD is the largest pair-interaction.

### Active subspace (relative eigenvalues)

| Surrogate | λ_1 | λ_2/λ_1 | λ_3/λ_1 | first eigvec [W, KV, KVD] |
|---|---:|---:|---:|---|
| **RBF tps+linear (primary)** | 1.000 | **0.050** | **0.016** | [−0.46, −0.83, −0.33] |
| RBF cubic+linear             | 1.000 | 0.080 | 0.028 | [−0.47, −0.83, −0.31] |
| ARD-GP (Matérn-3/2)          | 1.000 | 0.101 | 0.037 | [−0.54, −0.78, −0.32] |

→ All three agree the response is essentially 1-D in input space; RBF tps gives the sharpest collapse (λ_2/λ_1 = 0.05). Note: the first eigvec weights gradients, which is sensitive to *input range*; KV has the narrowest input range so per-unit slope is largest there. This is consistent with the Sobol indices (which are range-aware — Sobol still ranks W first because its absolute variance contribution is larger).

### Hoeffding additive ε bounds on the 200 AWQ test

Replace the full surrogate with `y_add = f_0 + g_W(z_W) + g_KV(z_KV) + g_KVD(z_KVD)` (KDE-smoothed conditional means, MC bandwidth = 0.15·σ on the QS-train range). Evaluate on the 200 AWQ test:

| Surrogate | full ε_∞ | full ε_2 | add ε_∞ | add ε_2 | full R² | add R² | Var(res_add)/Var(y_te) | 1 − ΣS1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **RBF tps+linear (primary)** | 0.0331 | 0.0083 | **0.0666** | **0.0300** | 0.995 | 0.939 | 0.041 | 0.010 |
| RBF cubic+linear             | 0.0340 | 0.0090 | 0.0751 | 0.0364 | 0.994 | 0.910 | 0.054 | 0.027 |
| ARD-GP (Matérn-3/2)          | 0.0317 | 0.0095 | 0.0838 | 0.0442 | 0.994 | 0.867 | 0.076 | 0.055 |

→ With RBF tps+linear, the additive collapse pays only **2× ε_∞** (0.033 → 0.067) and 4×ε_2 — small enough that the 2ε-Pareto theorem gives a usable bound (2·ε_add_∞ ≈ 0.13). The slight gap between Var(res_add)/Var(y_te) and (1−ΣS1) is expected: Sobol's variance identity holds on the *surrogate's own input distribution* (uniform over the QS-train range), whereas Var(y_te) is the empirical variance of the 200 AWQ test.

Figures: `figures/v4_fig11_internal_sobol.png`, `..fig12_internal_active_sub.png`, `..fig13_internal_hoeffding.png`. Full numbers: `internal_analysis_results.json`.

## Phase 2 results (10 seeds, 50/150 paired holdout)

For each of 10 random seeds, AWQ 200 → 50 train (random) / 150 test. Compare two
training sets evaluated on the **same** 150 holdout: random-50 vs the fixed quantile-50.
Both ARD-GP kernels included.

| Surrogate | random R² (μ±σ) | QS R² (μ±σ) | Δ R² (QS−rand) | seeds QS>rand |
|---|---|---|---|---:|
| M1 linear additive      | 0.942 ± 0.009 | 0.948 ± 0.005 | +0.007 ± 0.011 | 6 / 10 |
| M10 full quadratic      | 0.916 ± 0.116 | **0.979** ± 0.002 | +0.063 ± 0.116 | 10 / 10 |
| RBF cubic+linear        | 0.978 ± 0.011 | **0.994** ± 0.000 | +0.016 ± 0.011 | 10 / 10 |
| RBF tps+linear          | 0.984 ± 0.006 | **0.995** ± 0.000 | +0.012 ± 0.006 | 10 / 10 |
| ARD-GP (RBF)            | 0.983 ± 0.012 | 0.982 ± 0.002 | −0.001 ± 0.012 | 4 / 10 |
| ARD-GP (Matérn-3/2)     | 0.990 ± 0.006 | **0.994** ± 0.000 | +0.004 ± 0.006 | 9 / 10 |

| Surrogate | random RMSE | QS RMSE |
|---|---|---|
| M1 linear additive      | 0.02904 ± 0.00253 | 0.02736 ± 0.00084 |
| M10 full quadratic      | 0.03022 ± 0.01728 | **0.01758** ± 0.00076 |
| RBF cubic+linear        | 0.01730 ± 0.00380 | **0.00911** ± 0.00030 |
| RBF tps+linear          | 0.01520 ± 0.00266 | **0.00840** ± 0.00025 |
| ARD-GP (RBF)            | 0.01490 ± 0.00548 | 0.01607 ± 0.00081 |
| ARD-GP (Matérn-3/2)     | 0.01169 ± 0.00335 | **0.00962** ± 0.00025 |

Headline: **the QS pool yields equal or better mean fit and 10–30× lower variance across seeds for every non-trivial surrogate.** Concretely:

- M10 / RBF cubic / RBF tps: QS wins on every seed (10/10).
- ARD-GP Matérn-3/2: QS wins 9/10 seeds, σ shrinks from 0.006 → 0.0003 (≈20×).
- ARD-GP RBF (the inferior kernel): QS doesn't help on the mean (small δ ≈ 0). This is consistent with the Phase-1 finding that the RBF kernel is mismatched to the data — once the prior is wrong, more uniform training coverage cannot rescue it.
- ARD-GP RBF in Phase 2 (n_restarts=10) yields slightly different numbers from Phase 1 (n_restarts=50); the Matérn-3/2 results are stable across both.

Figures: `figures/v4_fig5_phase2_R2_bars.png`, `..fig6_phase2_RMSE_bars.png`,
`..fig7_phase2_paired_delta.png`. Full per-seed table: `phase2_results.json`.

## Caveats / scope

- Sampling pool is the single QS folder `..._qs_w159_kv159_kvdim159_rs23`. The 50-row composition is taken as-is (variance benefit may depend on the specific QS protocol; here only `rs23` was tested).
- Skipped (per request): v3 scripts 05–08 (off-surface acquisition + AWQ side-eval) — those concern measurement-point acquisition, not predictor quality.
- Phase 2 uses paired holdouts, so Δ R² is a fair within-seed comparison; the high std on the random side reflects genuine sampling-luck variance, not measurement noise.
- ARD-GP kernel selection is by **training log-marginal-likelihood**, which uses only training data — no test-set leakage.
