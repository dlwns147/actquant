# Joint second-search top-1 regret audit (2026-08-11)

## Scope

- Joint archive: `save/second_search/2608090529_.../iter_15.stats`
- Benchmark labels: `save/correlation/2607301912_.../correlation.csv`
- Deployment budget: memory 5,315,764,224 bytes, relative band `±0.5%`,
  `n_token=16384`, residual 128, attention sink 8
- Search/calibration bridge metric: `wt2_jsd_pp128_s32`
- Endpoint utility: LongBench-E average and RULER average are standardized
  separately. The primary decision minimizes the worse of the two endpoint
  regrets (minimax), avoiding an arbitrary raw-scale weighted sum.

The analysis is CPU-bound and has only 200 labelled rows. A GPU would reduce no
meaningful uncertainty, so it was not used for the regressions.

## Archive audit

| Item | Result |
|---|---:|
| Measured archive rows | 4,090 |
| Unique full architectures | 4,090 |
| Unique W / KV blocks | 96 / 1,197 |
| In-band rows, old residual-omitting memory | 95 |
| In-band rows, physical residual-128 memory | 108 |
| Physical-memory measured-loss winner | archive index 4041, JSD 0.0549927 |
| Second-best gap | 0.0007324 |
| Fifth-best gap | 0.0040283 |

The memory bug does not change this run's winner, but it changes membership from
95 to 108 rows. Residual cache adds roughly 10--11 MiB in this region.

The symmetric band is not a hard cap. Index 4041 uses 5,341,501,184 bytes,
0.484% above the 5,315,764,224-byte center. If that center is a deployment
maximum rather than an iso-memory comparison point, the upper bound must be the
center itself. Under `[center*(1-0.005), center]`, 63 rows remain and the measured-
loss winner changes to index 4035 (5,313,486,368 bytes, JSD 0.0602417).

The local best was not stable across iterations. It was 0.06061 through
iteration 4, 0.05859 through iteration 14, then improved to 0.05499 at iteration
15. Archive indices 4041, 4047, and 4048 (loss ranks 1, 2, and 4) all arrived in
the final iteration. Thus search-truncation uncertainty is at least as important
as post-hoc benchmark correction.

`stats["candidates"]` is not an unmeasured extra pool in this run. It is exactly
the last measured slice already appended to `archive`; its values equal the next
iteration's archive values. `--second_include_candidates` therefore needs
deduplication, which is now implemented.

The three worker logs contain normal repeated AWQ builds (~7.4 minutes each) and
no traceback/OOM/NaN. They do not contain architecture IDs, metric values, or
replicate measurements, so they cannot estimate JSD noise or top-1 flip
probability.

## Transfer risk from the 200 labelled architectures

There are zero exact full-architecture matches between the 200 labelled rows and
the 4,090-row archive, one exact W-block match, and zero exact KV-block matches.
At the exact target band the labelled set has only four rows (8/12/40 rows at
±1% / ±2% / ±5%). Consequently, random row CV alone is not adequate.

The bridge loss is nevertheless protocol-matched: WikiText-2, 128 samples,
2,048 tokens, stride 32, prefill, last 128 tokens, sink 8. Globally it is very
correlated with LongBench-E/RULER, but the useful question is conditional ranking
inside a memory band:

The old `sample_meta.json` does not persist dtype or quantizer method, so full
provenance cannot be machine-verified from that file alone; AWQ is inferred from
the run name/results. New calibration artifacts should persist both fields.

| Scope | Pearson(loss, LB-E) | Pearson(loss, RULER) |
|---|---:|---:|
| All 200 | -0.988 | -0.906 |
| Target ±5% (n=40) | -0.925 | -0.465 |
| Target ±3% (n=20) | -0.761 | -0.015 |

This is why a monotone loss calibration cannot fix local RULER regret: it cannot
change the ordering. W/KV allocation features are needed, especially for RULER.

## Model and decision validation

Models used only the measured joint loss and 11 aggregate W/KV/memory features.
Raw per-layer profiles were rejected: although they improved random CV, their
structural-cluster holdout balanced/minimax regret increased from
0.037/0.063 (loss baseline) to 0.148/0.212.

### Explicit 100/100 and 150/50 train/test holdouts

The main validation uses the requested independent train/test sizes. Memory was
split into ten quantile strata, then `StratifiedShuffleSplit` was repeated for 50
seeds. Models saw only the train rows. Top-1 and the oracle were computed only
among test rows in fixed ±5% memory windows (at least five test candidates).
Regret units standardize LB-E and RULER separately using train statistics.

| Train / test | Method | Balanced regret | Minimax regret | Paired delta vs loss (95% CI) |
|---|---|---:|---:|---:|
| 100 / 100 | measured loss | 0.09446 | 0.16405 | -- |
| 100 / 100 | unrestricted ridge | 0.03700 | 0.05481 | -0.05746 / -0.10925 |
| 100 / 100 | guarded 4/5 ensemble | **0.08814** | **0.15329** | -0.00633 `[-0.00884,-0.00381]` / -0.01076 `[-0.01504,-0.00648]` |
| 150 / 50 | measured loss | 0.08782 | 0.15826 | -- |
| 150 / 50 | unrestricted ridge | 0.03683 | 0.06043 | -0.05099 / -0.09783 |
| 150 / 50 | guarded 4/5 ensemble | **0.08278** | **0.14991** | -0.00504 `[-0.00783,-0.00224]` / -0.00834 `[-0.01303,-0.00366]` |

The low-dimensional ridge test R² was 0.9906/0.9595 (LB-E/RULER) for 100/100
and 0.9913/0.9603 for 150/50. The guarded ensemble changed the loss winner in
only 7.7% and 3.1% of test-window decisions, respectively; its mean extra JSD
was only 0.000035 and 0.000020. Thus its small regret reduction is not obtained
by routinely sacrificing calibration loss.

At the specific target ±5% window, 100/100 had 20.4 test candidates on average:
guarding changed balanced/minimax regret from 0.01096/0.01365 to
0.00869/0.01145. For 150/50 there were only 9.68 candidates and the change was
0.01711/0.02943 to 0.01678/0.02902. The exact ±0.5% target band is too sparse in
the labelled set for an honest holdout claim (four rows total before splitting).

For comparison, the earlier cross-validation/stress splits gave:

| Split | Loss-only balanced / minimax | Unrestricted correction | Guarded correction (`loss <= best+0.001`) |
|---|---:|---:|---:|
| Repeated random 5-fold | 0.091 / 0.166 | 0.032 / 0.054 | 0.085 / 0.156 |
| Hold out memory quintile | 0.085 / 0.146 | 0.059 / 0.084 | 0.077 / 0.130 |
| Hold out contiguous sample block | 0.075 / 0.137 | 0.078 / 0.128 | 0.071 / 0.130 |

Unrestricted correction often lowers average regret, but on the current archive
it proposes JSD rank 38 (loss 0.06531) and predicts RULER 0.929. Tree and nearest-
neighbor models instead propose ranks 15 or 39. This disagreement is a direct
out-of-distribution warning, not evidence for choosing one aggressive model.

The implemented rule therefore:

1. keeps only measured-loss contenders within `best + 0.001`;
2. predicts LB-E and RULER with five low-dimensional model families;
3. lets each model vote for standardized minimax regret;
4. overrides measured-loss top-1 only with at least 4/5 agreement;
5. otherwise abstains and returns measured-loss top-1.

For this archive only indices 4041 and 4047 pass the loss guard. Votes are:

- 4041: kNN, ExtraTrees, RandomForest (3/5);
- 4047: ridge, histogram GBDT (2/5).

No candidate reaches the 80% override threshold. The defensible current result
is therefore **archive index 4041**, not an aggressively benchmark-corrected
candidate. This is an evidence-based abstention: the 200 labels show that
allocation correction can help, but they do not identify a different current
top-1 with adequate support.

## Remaining uncertainty and recommended next evidence

1. Resume the joint search for several iterations. The best changing on the
   final iteration is evidence that the local archive has not stabilized.
2. Add benchmark-labelled architectures sampled directly from this second-stage
   archive and target memory band. A paired factorial design over a few W
   families and KV allocations is more informative than more global random rows.
3. Repeat calibration subsets (not only software seeds) for the top loss
   neighborhood to estimate the 0.00073 top-two gap's flip probability.
4. Do not claim benchmark-regret reduction from the guarded selector until a
   new, archive-native labelled holdout confirms it. The current result supports
   the fallback decision, not a strong correction claim.
