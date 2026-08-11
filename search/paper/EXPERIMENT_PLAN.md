# Evidence and experiment plan

This document separates what the current repository already supports from what
is still required for a defensible submission.  It is intentionally stricter
than a normal project TODO list: every experiment is tied to a paper claim or a
reviewer attack.

## 1. Highest-priority validity checks

### A. Off-front coverage audit (the theorem's main open assumption)

The existing 20x20 grid uses blocks sampled from the Stage-1 fronts.  It does not
cover every role in the sequential-swap proof once Stage 2 mutates a block.

Sample 30 off-front/mutated joint architectures `a=(w,k)`.  For each, find its
Stage-1 front projections `(p_w,p_k)` and measure:

1. `y(w,k)`;
2. `y(w,p_k)`;
3. reuse or measure `y(p_w,p_k)`.

This costs roughly 60 new AWQ evaluations if projected corners are reused.  Report
both swap margins, their maximum, and bootstrap confidence intervals.  If this
fails, narrow the theorem's empirical claim to the audited front product and
treat mutation as an unguaranteed exploration heuristic.

### B. Stage-1 front adequacy

Run independent Stage-1 searches (at least three seeds) for W and effective KV.
For every pair of runs, report:

- mutual epsilon-dominance/coverage;
- hypervolume difference;
- cost-bin best-loss gap;
- structural diversity of tied front blocks.

This estimates the `eta_i` term in the theorem.  A single converged-looking run
cannot establish that its archive approximates the true axis front.

### C. Matched-budget end-to-end baseline

At a fixed AWQ-build budget, compare:

- full-space random/NSGA search;
- product-front search without mutation;
- epsilon-band product plus mutation;
- additive/MCKP-style selection;
- the complete method.

Use the pooled measured union as an empirical reference front and report
hypervolume, IGD, and memory-band top-1 regret.  This is necessary even if the
theorem holds: the theorem is an existence result, not a claim that the current
surrogate/NSGA-III loop finds the covered point.

## 2. HQQ/AWQ proxy boundary

Use the same configurations and measurement protocol for both methods.  Evaluate
agreement at four resolutions:

| Resolution | Question | Statistics |
|---|---|---|
| Per-axis | Can HQQ construct Stage-1 fronts? | Spearman, Kendall, Pareto overlap, violation margin |
| Cross-budget | Does it order coarse memory changes? | rank correlation, calibration slope |
| Fixed 2-D budget cell | Can it select the final allocation? | within-cell rho, discordance, AWQ regret |
| Structurally far, equal-cost pairs | Are failures only near-tie noise? | discordance vs Hamming distance and loss spread |

Also report the recall of the AWQ-best point under HQQ top-q shortlists.  Avoid
claiming that a low correlation *itself* proves different Pareto fronts; a
proxy-dominance witness for an AWQ-front point, or a direct Pareto-set comparison,
is the correct certificate.  The current 4,365-point paired production archive
already provides an in-archive certificate (exact AWQ-front recall 71.9%; with a
`1e-3` JSD tolerance, recall 66.8% and front Jaccard 0.496).  Repeat this audit on
a held-out pool that was not selected by the AWQ search, and report the number
and deployment gaps of target-front points excluded by proxy dominance.

Run this audit on at least Llama-3.1-8B and Qwen2.5-7B.  The current result is
strong enough to motivate AWQ information in Stage 2, but not to claim universal
failure of proxy-only joint search.

## 3. PLS sample-efficiency ablation

The current PLS results are encouraging but several historical comparisons used
different splits, heads, or metrics.  Use one fixed held-out test set and vary
only the input representation.

Inputs:

- ordinal genome;
- full one-hot genome;
- PCA at 4/8/16/32 dimensions;
- HQQ-supervised PLS at 2/4/8/16 components per axis;
- self-PLS fitted only on the AWQ archive;
- exact costs only (negative control).

Keep the surrogate head fixed (`sqrty_ard_gp`, Matérn-3/2), and sweep AWQ training
sizes `N={25,50,100,200,400}` over at least five splits.  Split by **weight
family**, not by individual architecture, so KV companions sharing one AWQ build
cannot leak between train and test.

Report:

- global and budget-band RMSE/Spearman;
- within-cell Spearman and pairwise discordance;
- best-of-pool/top-1 AWQ regret by memory band;
- Pareto hypervolume obtained after one acquisition round;
- fit and acquisition time.

The paper's sample-efficiency claim should be based on the area under the
learning curve or the number of AWQ builds required to reach a target regret,
not on one correlation at `N=100`.

## 4. Multi-KV (`K`) ablation

Define `K` consistently as the **total** number of KV configurations evaluated
per AWQ weight build.  The current code setting `--companion_kv 10` therefore
corresponds to `K=11`.

Sweep `K={1,5,10,20,40}` under three controls:

1. fixed distinct-W build budget;
2. fixed total label budget (`B*K`);
3. fixed measured wall-clock/GPU-hour budget.

For each setting and at checkpoints `B={25,50,100,200}` builds, report:

- hypervolume and IGD to the pooled-union reference front;
- 8x8 weight/KV budget-cell coverage;
- number of distinct W families;
- maximum/mean KV gap inside each W family;
- grouped-CV surrogate RMSE and within-cell rank;
- memory-band top-1 regret;
- actual build, KV-swap, and evaluation time.

Use at least three search seeds.  The expected curve is diminishing returns:
larger K is nearly free in the current 2K-token protocol but can starve independent
W diversity under a fixed label budget, and its evaluation cost may no longer be
negligible at 16K--128K contexts.  Do not choose K from label count alone.

Useful secondary comparisons:

- random KV companions vs stratified cost quantiles vs predicted-front/coverage;
- endpoint forcing on/off;
- identical K for every W vs adaptive K based on posterior uncertainty or the
  current family's largest KV gap.

An adaptive policy is a plausible follow-up: allocate the next companion to the
W family with the largest predicted reduction in worst-cell regret per KV-eval
second.  It should be proposed only after the fixed-K curve is measured.

## 5. Additional paper-strengthening experiments

### Metric and downstream validity

Search with the current answer-phase JSD, then evaluate selected fronts on:

- WikiText-2 and C4 perplexity;
- LongBench/LongBench-E;
- RULER retrieval at multiple context lengths;
- latency, peak memory, and tokens/s with the actual mixed kernels.

Include uniform-bit W/KV, AMQ weight-only, KVTuner-style KV-only, and independent
best-axis combinations.  A lower calibration JSD is not enough if the end-to-end
kernel or long-context task does not improve.

### Context-length robustness

Repeat the cost accounting and a reduced proxy audit at 2K, 8K, 16K, and 64K.
Weight memory is static while KV memory grows with context, so the optimal W/KV
exchange rate must change.  This is a core benefit of a joint method and should
eventually appear as a principal result, not only an appendix ablation.

### Noise and reproducibility

The current AWQ seed check is deterministic for the tested configurations, but
repeat measurements across calibration subsets, not only software seeds.
Estimate near-tie flip probability and put confidence intervals on top-1 regret.

### Hardware realism

Average bits and effective KV bits are useful search coordinates, but the final
paper should also optimize or at least filter by exact bytes and report real
latency.  Mixed per-layer bit-widths can incur kernel dispatch and packing
overheads that an average-bit Pareto plot misses.

## 6. Claims to avoid until the evidence changes

- "The joint loss is additive."  It is approximately additive in a useful band
  but has corner saturation; the coverage theorem does not need additivity.
- "HQQ and AWQ have different correlation, therefore their Pareto fronts are
  different."  Show same-budget reversals/nondominance or direct front overlap.
- "The theorem guarantees the algorithm finds the optimum."  It guarantees that
  a covered point exists under explicit assumptions.
- "K KV labels equal K independent samples."  They share the same W allocation;
  use grouped validation and count distinct builds.
- "PLS improves sample efficiency" based only on Stage-1 reconstruction R2 or
  one historical split.  Use the fixed-test learning-curve experiment above.
