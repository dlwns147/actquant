# Paper draft

This directory contains a Korean Markdown draft through **Introduction, Related
Work, and Method** for the joint weight/KV-cache search project. `main.md` is
the primary editing version; the English LaTeX source is retained as an
optional typesetting and comparison version.

## Files

- `main.md`: complete Korean paper draft in one Markdown file.
- `main.tex`: optional LaTeX version with section assembly.
- `sections/1_introduction.tex`: motivation, order--value framing, and contributions.
- `sections/2_related_work.tex`: weight PTQ, AMQ/AutoML, KV compression, and positioning.
- `sections/3_method.tex`: formulation, two-stage search, theorems, proxy boundary,
  PLS embedding, multi-KV evaluation, and assumption-audit protocol.
- `references.bib`: references used by the current draft.
- `EXPERIMENT_PLAN.md`: critical evidence audit and prioritized experiments.
- `scripts/proxy_pareto_audit.py`: read-only reproduction of the
  proxy-front overlap, dominance-certificate, and within-budget regret numbers.

The provisional paper name is **ActQuant** only because it matches the repository
name. In `main.md` it appears as plain text; the optional LaTeX version defines
it once as `\method` in `main.tex`.

## Viewing and optional LaTeX build

Open `main.md` in any Markdown previewer. Its display equations use the common
`$...$`/`$$...$$` math syntax, and its two figures use paths relative to the
paper directory.

To build the optional LaTeX version, run from this directory:


```bash
latexmk -pdf main.tex
```

No TeX engine is installed in the current workspace, so the LaTeX version was
checked statically but not rendered here.

## Evidence policy used in the draft

Only the newest implementation and 2026-07/08 analyses were treated as current.
Older `analysis/v3--v5` results were used only as background.  Numerical claims in
the Method section are marked as preliminary and correspond to the current
Llama-3.1-8B, WikiText-2, stride-128, answer-phase-JSD protocol unless noted.

The current evidence supports these scoped statements:

- On the audited Stage-1 front blocks, the paired `20 x 20` AWQ grid has a small
  ordering-violation rate (`20 / 7,600` for stride 128) and worst combined
  violation margin (`0.0103 + 0.0034 = 0.0137`).
- HQQ-to-AWQ order is preserved on the two audited individual axes, but not among
  fine-grained equal-budget joint allocations (median `rho ~= 0.40`; the HQQ
  top pick matches the AWQ best in only about 22% of 183 cells).
- On the 4,365-point paired production archive, the exact HQQ front recalls
  71.9% of the AWQ front. With a `1e-3` JSD tie tolerance, recall is 66.8% and
  front Jaccard is 0.496, versus `0.825 +/- 0.007` under an AWQ noise null.
- The production embedding is an 18-dimensional feature: 8 PLS components per
  axis plus exact weight/KV costs.
- Reusing one AWQ weight build while swapping KV configurations reproduces a
  fresh-build loss in the current GPU correctness test.
- On 200 Llama-3.1-8B joint configurations, the current prefill-plus-stride JSD
  has substantially higher global and within-budget rank correlation with
  LongBench-E and RULER than the conventional single-forward JSD.

The draft deliberately does **not** yet claim:

- global coverage of arbitrary off-front mutations;
- that the measured production archive estimates Pareto mismatch over the
  unobserved full combinatorial space;
- that one Stage-1 run found the true axis fronts;
- that PLS is conclusively better than all raw/one-hot inputs under a clean,
  fixed-test causal ablation;
- that the current companion count is optimal;
- cross-model or downstream-task generality.
- universal superiority of Strided JSD beyond the audited model, configuration
  pool, and long-context benchmarks.

Those gaps are the first items in `EXPERIMENT_PLAN.md`.
