# MODIFY.md

Findings from a full pass over `amq/amq/` (excluding `kernel/` third-party code) after wiring up the new `BenchmarkProxyEvaluator` at `evaluation/evaluator.py`. Items are ordered by blast radius: blocking bugs first, then arg/config drift, then minor cleanups.

---

<!-- ## 1. `search/optimizer.py` — blocking bugs

The optimizer still references the old perplexity-proxy contract and does not match the new `BenchmarkProxyEvaluator` signature.

### 1.1 Typo — `onfig=` (line 58)
```python
self.evaluator = BenchmarkProxyEvaluator(
    onfig=self.config,              # ← typo: should be `config=`
    ...
)
```
`BenchmarkProxyEvaluator.__init__` takes `config` as its first positional-or-keyword argument. Current code raises `TypeError: unexpected keyword argument 'onfig'`. -->

<!-- ### 1.2 Keyword mismatch — `n_sample=` / `fp16_cache_dir=` (lines 65–66)
```python
self.evaluator = BenchmarkProxyEvaluator(
    ...
    n_sample=self.n_sample,             # ← evaluator expects `n=`
    fp16_cache_dir=self.fp16_cache_dir, # ← evaluator expects `cache_dir=`
)
```
`BenchmarkProxyEvaluator.__init__` accepts `n=16` and `cache_dir="fp16_cache"` (see `evaluator.py:367-368`). Rename the kwargs at the call site, or rename the evaluator params if we prefer the longer names (recommend keeping the evaluator's public API and renaming in the caller). -->

### 1.3 Metric dict key mismatch (line 219)
```python
metric_list.append(min(self.max_value, np.nan_to_num(metric[self.dataset], nan=self.max_value)))
```
The new evaluator's `metric_list` is keyed by `"{ds}-generate"` / `"{ds}-JSD"` plus an aggregate `"score"`. It does **not** contain an entry named `self.dataset`. Since the search loop only needs one scalar per architecture, replace with:
```python
metric_list.append(min(self.max_value, np.nan_to_num(metric["score"], nan=self.max_value)))
```

<!-- ### 1.4 Missing arg — `args.dataset` (line 40)
```python
self.dataset = args.dataset
```
The search-mode parser in `utils/args.py` defines `--datasets` (plural, via `add_data_args`), not `--dataset`. This line raises `AttributeError` immediately. See §2.1 for the arg-side fix; this line should consume whatever singular/first-element convention we settle on — or be removed entirely, since after §1.3 the optimizer no longer needs to know a specific dataset name. -->

<!-- ### 1.5 Missing arg — `args.result_file` (line 38, used line 198)
```python
self.result_file = args.result_file
...
with open(os.path.join(self.save_path, self.result_file), 'w') as f:
```
`--result_file` is **commented out** in `utils/args.py:101-102`. Either re-enable the argparse entry (default `"results.json"`) or compute the filename locally. Currently `args.result_file` raises `AttributeError` on line 38. -->

---

<!-- ## 2. `utils/args.py` — arg-side drift

### 2.1 `--datasets` default is a single comma-joined string (line 114)
```python
group.add_argument('--datasets', type=str, nargs='+',
                   default=['gsm8k, livebench'],   # ← one string "gsm8k, livebench"
                   help='Dataset for calibration and evaluation')
```
`nargs='+'` returns a list, but the default is `['gsm8k, livebench']` (one element). Should be `['gsm8k', 'livebench']` so the default gives two datasets.

### 2.2 Missing `--dataset` / `--seqlen` for search mode
- `amq_search.py:54` uses `args.dataset` and `args.seqlen`
- `search/optimizer.py:40` uses `args.dataset`

Neither `add_search_args` nor `add_data_args` defines `--dataset` (singular) or `--seqlen`. Pick one:
- **Option A (recommended, aligned with benchmark-proxy plan):** drop `--dataset`/`--seqlen` from the search path entirely; use `--datasets` everywhere, and make the sensitivity-path string on `amq_search.py:54` compose from `"_".join(args.datasets)` instead of `args.dataset` — and stop threading `args.seqlen` (the benchmark-proxy evaluator doesn't use `seqlen`).
- **Option B:** add `--dataset` and `--seqlen` to `add_data_args` as legacy shims. Cheaper now but leaves two overlapping concepts.

### 2.3 `--result_file` is commented out (lines 101-102)
Re-enable (recommended):
```python
group.add_argument('--result_file', type=str, default='results.json',
                   help='File name to save results summary')
```
Alternatively, delete the reference in `optimizer.py:38, 198` and just hardcode `"results.txt"`.

### 2.4 `add_eval_args` still defaults to `wikitext2` (line 129)
```python
group.add_argument('--eval_dataset', type=str, nargs='+', default=['wikitext2'], ...)
```
`amq_quantization.py:104` forwards this to `evaluator_original.Evaluator`, which is fine for perplexity-mode eval, but it's worth deciding whether final quantized-model eval should also run on GSM8K/LiveBench. If yes, this needs to pivot (and `amq_quantization.py` needs to call `BenchmarkProxyEvaluator` too). If perplexity is intentionally kept for the quant sanity check, leave as-is but add a comment. -->

---

<!-- ## 3. `amq_search.py` — sensitivity-path composition

### 3.1 Line 54 — depends on the §2.2 decision
```python
args.sensitivity_path = f"amq/sensitivity/{args.model_name}_dataset_{args.dataset}_n_sample_{args.n_sample}_seqlen_{args.seqlen}.json"
```
After the §2.2 resolution this becomes, e.g.:
```python
args.sensitivity_path = (
    f"amq/sensitivity/{args.model_name}"
    f"_dataset_{'_'.join(args.datasets)}"
    f"_n_sample_{args.n_sample}.json"
)
```
(Drop `seqlen` if it's no longer a search-time arg.)

Also note the **path separator inconsistency**: `amq_sensitivity.py:72` writes to `args.output_file` (caller-provided), but `amq_search.py:54` hardcodes `amq/sensitivity/...`. Tie them together (either via a single helper or by making `amq_sensitivity.py`'s output path follow the same template) so a sensitivity run and a search run actually find the same file. -->

---

## 4. `evaluation/evaluator.py` (new `BenchmarkProxyEvaluator`) — minor polish

These are not blocking, but worth addressing before the full end-to-end GPU run.

### 4.1 `_ensure_fp16_cache` loads fp16 once per eval call (`evaluator.py:462-475`)
Currently, if JSD is called for both `gsm8k` and `livebench` in the same `eval()` and both have misses, we load fp16 twice. Cheap fix: gather *all* misses across both datasets first, load fp16 once, then iterate. Keeps the "lazy cache" contract but avoids the double load. Not urgent — a single call per iter still dominates.

### 4.2 GSM8K gold answer is `doc["answer"]` (`evaluator.py:287-289`)
```python
if dataset == "gsm8k":
    doc = row.get("doc", {})
    return " " + doc.get("answer", str(row.get("target", "")))
```
The fallback `str(row.get("target", ""))` is a defensive-coding smell the user told us to avoid. The bucket JSONL files always have `row["doc"]["answer"]` — drop the `get`/fallback:
```python
return " " + row["doc"]["answer"]
```
Same for `_build_reference` GSM8K branch (`evaluator.py:167`) — `row["target"]` is always present.

### 4.3 Livebench gold response (`evaluator.py:290`)
```python
return row.get("response", "")
```
Same pattern — LiveBench bucket rows always have `"response"`. Drop the `.get(..., "")` and index directly.

<!-- ### 4.4 `bits_range=(),` default is unusable in search mode
The assertion on `evaluator.py:397-399` will trip if the caller forgets `bits_range`. That's the intended behavior, but the default `()` makes the failure obscure ("0 proxies but 0 bits matches, no assert triggers until…"). Consider making `bits_range` a required arg (remove the default) or asserting non-empty with a clearer error in search mode. -->

### 4.5 `_read_fp16_logits` is O(answer_len × vocab) floats parsed from text (`evaluator.py:324-328`)
Full-vocab text parsing is ~125k floats per token per doc — fine for n=16, but expect ~2-3s per JSD sample at vocab=128k. If iter wall-clock bites later, switch the on-disk format to `np.savez_compressed`(float16) with no API change. Flagging for later, not now.

---

## 5. `utils/data.py` — minor cleanups

### 5.1 `get_trainloaders` is dead (lines 56-61)
Never referenced elsewhere in the tree. Remove.

### 5.2 `get_wikitext2_trainenc` / `get_c4_trainenc` signatures mismatch `get_trainloaders` call
Not a live bug because the caller is dead code. Remove the caller; leave the two `*_trainenc` helpers (they're used by `get_loader`).

---

## 6. `amq_quantization.py` / `amq_sensitivity.py` — correct, just noting

These two already import from `evaluation.evaluator_original` (applied in the rename pass) and still use the perplexity contract. They work as-is. Only touch them if we decide in §2.4 that they should migrate to the benchmark-proxy pipeline.

---

## 7. `amq_speed_benchmark.py` — out of scope for this pass

Self-contained HQQ/FT speed benchmark; doesn't intersect with the evaluator or search loop. No changes needed.

---

## Suggested order of operations

1. Fix `search/optimizer.py` (§1.1 – §1.5). Without these, search won't start.
2. Resolve the arg drift in `utils/args.py` (§2). Decide on `--datasets` vs `--dataset` once — propagates into §1.4 and §3.
3. Run a minimal `amq_search.py` smoke (tiny `n`, tiny `iterations`) to confirm the evaluator integrates end-to-end.
4. Polish `evaluation/evaluator.py` (§4.2, §4.3). Strip the defensive `.get(..., default)` patterns.
5. Remove dead code in `utils/data.py` (§5).
6. Decide on `amq_quantization.py`'s eval mode (§2.4 / §6).
