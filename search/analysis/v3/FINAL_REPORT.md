# Falsification of Cartesian-combined Pareto Front

## Question
Does any off-Cartesian-PF arch (i.e., a combination using sub-Pareto sub-arches
from local archives) Pareto-dominate the Cartesian-combined PF?

## Method (4-stage pipeline)

1. **Acquisition** ([05b_acquire_offsurface_3d.py](05b_acquire_offsurface_3d.py))
   Cartesian product of top-50 sub-arches per axis (W / KV / KVDIM) =
   125,000 candidates. Score each via ARD-GP-predicted (μ, σ) with the
   3D Pareto-envelope baseline f*_3D(c_w, c_kv, c_kd) = max{y_PF :
   ∀axis c_PF ≥ c_q}, computed from 200 AWQ-measured PF samples. Pick 100
   off-Cartesian-PF candidates: 75 stratified across 5×5 = 25 (wbits ×
   eff_kvbits) buckets by P(violator) descending; 25 σ-extras (P>1%, high σ).

2. **Low-P controls** ([05c_acquire_lowp_controls.py](05c_acquire_lowp_controls.py))
   25 additional candidates with **P(violator) = 0** (predictor strongly
   says NOT violator), stratified across same 5×5 buckets. Test predictor
   calibration: if any of these turn out to be actual violators, predictor
   missed them in main 100.

3. **AWQ evaluation** ([06_evaluate_awq_100.py](06_evaluate_awq_100.py))
   125 archs run through AWQ + KIVI on Llama-3.1-8B-Instruct, wikitext2
   JSD, n_sample=128, seqlen=2048. Total time ≈ 18 hours.

4. **3D Pareto-envelope analysis** ([07_recompute_baseline_3d.py](07_recompute_baseline_3d.py))
   For each candidate, compute baseline f*_3D(c) and check
   ε-violation: y_actual < f*_3D − ε.

## Result

```
n = 125 (100 main P-targeting + 25 low-P controls)
all candidates in-cloud (3D baseline defined)

ε-violations (y_actual < f*_3D − ε):
  ε=0.005:  0/125    (95% upper CI on rate: 2.40%)
  ε=0.01:   0/125    (95% upper CI on rate: 2.40%)
  ε=0.02:   0/125    (95% upper CI on rate: 2.40%)
  ε=0.05:   0/125    (95% upper CI on rate: 2.40%)

residual_3d (y_actual − f*_3D):
  range: [+0.00436, +0.44842]   (all positive ⇒ no Pareto-domination)
  min:   +0.00436                (closest to PF: sel_idx=57, total_c=6.92)
  mean:  +0.09234
```

### ARD-GP P(violator) calibration

| Group | n | actual violator rate | predicted P range |
|---|---|---|---|
| Main (P_3d > 50%) | 60 | 0/60 | 0.50–0.97 |
| Main (10% < P_3d ≤ 50%) | 34 | 0/34 | 0.10–0.50 |
| Main (P_3d ≤ 10%) | 6 | 0/6 | < 0.10 |
| Controls (P = 0) | 25 | 0/25 | exactly 0 |

→ ARD-GP's P(violator) is **systematically over-confident in the violator
direction**: even at P=0.97 prediction, actual violation rate is 0/60.
But this bias is *violator-friendly* (predictor under-estimates loss in
off-surface region, biasing toward predicting violators). Under unbiased
predictor, violation rate would be even lower.

### 1D vs 2D vs 3D baseline comparison

In this dataset all 125 candidates are in-cloud, so the 1D PF interpolant
coincides with 3D Pareto envelope (no extrapolation kicked in). Both
correctly report 0 violations. **2D baseline (wbits, eff_kvbits) gives
24/125 spurious "violations" at ε=0.005** because collapsing kvbits and
kvdim into eff_kvbits loses information — 2D under-estimates the true
upper envelope. **3D is the correct strict Pareto test.**

## Conclusion

> **Cartesian-combined PF is a faithful approximation of the true Pareto
> frontier on this search space.** With 95% confidence, the rate at which
> any off-Cartesian-PF candidate ε-Pareto-dominates the Cartesian PF is
> ≤ 2.40% for any ε ∈ {0.005, 0.01, 0.02, 0.05}.

This conclusion is *adversarially robust*: 60 of the 100 main candidates
were specifically chosen by ARD-GP as having ≥50% predicted violator
probability, and 0 of those were actual violators. The 25 low-P controls
verify the predictor's "non-violator" claims are also accurate (0/25
were violators).

## Caveats

The candidate pool is restricted to the **Cartesian product of local
archive sub-arches** (top 50 per axis × 3 axes ⇒ 125k candidates). Sub-
arches outside the local archives' coverage are not tested. To extend
coverage to truly novel sub-arches would require additional local
sub-loss measurements (1 forward pass per axis per new sub-arch) before
ARD-GP scoring.

## Files

- Acquisition (3D): `acquired_offsurface_100.json`, `acquired_lowP_controls_25.json`
- Evaluations:    `eval_offsurface_100.{json,csv,txt}` (125 records)
- Final analysis: `eval_offsurface_baseline3d.{json,csv,txt}`
- Figures:        `figures/05b_acquire_overview.png`, `figures/05c_lowp_controls.png`,
                  `figures/06_eval_offsurface_overview.png`, `figures/07_baseline3d_overview.png`
- Backups (1D):   `*_1d.bak`
