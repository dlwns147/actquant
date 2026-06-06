# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Neural Architecture Search (NAS) framework for LLM quantization**. It
searches per-layer mixed-precision configurations along several axes
simultaneously, using surrogate-model-guided multi-objective optimization
(NSGA2 + Pareto archive):

- **Weight bits** — per-linear (q/k/v/o, gate/up/down) bit-width (HQQ / AWQ / GPTQ / QEFT)
- **KV-cache bits + group size** — per-layer `(bits, group_size)` for K and V (KIVI backend)
- **KV channel pruning (ThinK)** — per-layer `head_dim` channels removed for K and V

Quality is measured as **JSD** to the FP16 teacher (KLD / cross_entropy / PPL also
supported); complexity objectives (`comp_obj`) are chosen from `wbits`, `kvbits`,
`kbits`, `vbits`, `kvdim`, `kdim`, `vdim`, `eff_kvbits`, `eff_kbits`, `eff_vbits`,
`memory`.

## Running Scripts

All main scripts are launched via `accelerate launch`; see `scripts/` for the
canonical, fully-parameterized shell wrappers (these are the source of truth for
flags and paths). Representative invocations:

```bash
# 1) Per-axis NAS search (search.py / scripts/search.sh)
#    Run once per axis (W, KV-bits, KV-dim) with the matching --comp_obj.
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --num_machines=1 --main_process_port=12345 \
  search.py --gpu_id 0 --model_path /SSD/huggingface/meta-llama --model_name Llama-3.1-8B-Instruct \
  --config config/llama.json \
  --w_method hqq --kv_method kivi think --quant_model_paths "2bit_path 3bit_path 4bit_path" \
  --w_bits 2 3 4 --k_bits 2 3 4 --v_bits 2 3 4 \
  --w_group_size 128 --k_group_size 32 64 128 --v_group_size 32 64 128 \
  --k_quant_scheme channel --v_quant_scheme token \
  --k_pruning_dim 0 16 32 48 64 --v_pruning_dim 0 16 32 48 64 \
  --comp_obj wbits kvbits kvdim --comp_obj_min 2 2 64 --comp_obj_max 4 5 128 \
  --metric loss --loss_func jsd --predictor rbf \
  --iterations 200 --n_doe 100 --n_iter 50 --ga_pop_size 40 \
  --stride 128 --prefill_prompt --last_tokens 512 --save path/to/save

# 2) Two-stage post-search
#    Stage 1 — sample the joint combo space + write results.csv (sample_surrogate.py)
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=12345 \
  sample_surrogate.py --model_path ... --model_name Llama-3.1-8B-Instruct --config config/llama.json \
  --w_expr <W iter_N.stats> --kv_expr <KV iter_N.stats> --kvdim_expr <KVdim iter_N.stats> --expr_front \
  --quantile_sample "metric_w#0.0,0.1,0.5,0.9,0.99 ..." --random_sample 50 \
  --datasets wikitext2 --metric loss --loss_func jsd --save path --results_csv_file results.csv
#    Stage 2 — fit surrogate, pick final arch under COMP_OBJ, evaluate + benchmark (post_search.py)
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=12345 \
  post_search.py --model_path ... --model_name Llama-3.1-8B-Instruct --config config/llama.json \
  --w_expr ... --kv_expr ... --kvdim_expr ... --expr_front \
  --sample_path path/results.csv --surrogate ard_gp --ard_kernel matern32 \
  --comp_obj wbits kvbits kvdim --comp_obj_min ... --comp_obj_max ... -n 5 \
  --datasets wikitext2 --longbench --ruler ...

# 3) Loss/PPL ↔ LongBench/RULER correlation harness (correlation.py)
#    --mode sample writes archs.csv; --mode eval evaluates one --idx; aggregate offline.
```

## Architecture

### Per-axis Search (`search.py`)

`SearchThink` drives the NAS loop (one run per axis; archives combined later):
1. **DOE** — sample `n_doe` archs, evaluate via `LlamaEvaluator.eval`
2. **Surrogate fit** — `_fit_predictor` trains a predictor (default `rbf`) on the
   evaluated `(encoded_arch, metric)` pairs
3. **NSGA2 candidate generation** — `AuxiliarySingleLevelProblemThink` finds
   Pareto-optimal candidates (predicted metric vs. each `comp_obj`), with
   `comp_obj_min/max` enforced as constraints
4. **Subset selection** — `SubsetProblem` GA picks `n_iter` diverse candidates
5. Repeat for `--iterations`; each iter dumps `iter_<it>.stats` (the archive,
   hypervolume, surrogate fit stats, and **per-objective Pareto front-coverage**).
   `--debug` also saves per-iter scatter plots.

`metric/loss_func` knobs: `--metric loss --loss_func jsd` is the standard search
objective; `--stride/--prefill_prompt/--last_tokens` select the answer-phase
JSD measurement (see `analysis/answer_stride` and the v5 protocol below).

### Two-stage Post-search (`sample_surrogate.py` → `post_search.py`)

The final architecture is chosen by combining the per-axis archives, **not** by a
single joint search:
- **`sample_surrogate.py` (stage 1)** — load per-axis archives
  (`--w_expr/--kv_expr/--kvdim_expr/--eff_kv_expr`, optionally `--expr_front` to
  keep only each archive's Pareto frontier), sample the joint combo space
  (`--quantile_sample` anchors + `--random_sample` / coverage-NSGA2 extras),
  evaluate each, and write `results.csv` (the surrogate training data).
- **`post_search.py` (stage 2)** — fit a surrogate from `results.csv`
  (`--surrogate`, see below), re-rank the full combo space, select final arch(s)
  under `--comp_obj_min/max`, then evaluate + run benchmarks. Without
  `--sample_path`, falls back to the additive combined metric or, for separable
  single-axis `comp_obj`, a **per-axis-argmin** combination (no Σ of cross-reference
  JSDs). **NOTE: `--prefer` (ASF) and `--high_tradeoff` final picks are disabled
  in the code**; selection is per-axis-argmin or plain top-`n`.

`results.csv` layout: `n_comp` complexity rows, then one measured-metric row per
`--datasets`, then the combined-predicted row, then `n_axes` per-axis search-metric
rows (see `post_search.load_sample_csv`).

### Correlation harness (`correlation.py`)

Two-mode (`--mode sample` → `archs.csv`; `--mode eval --idx N` → `result_<idx>.json`)
harness measuring how calibration metrics (many JSD/PPL variants over
stride / prefill / key-token / needle / gsm8k-unpadded) correlate with
LongBench / LongBench-E / RULER. Idempotent and incremental per metric key.
See `analysis/corr_keyppl`, `analysis/band_local_proxy_collapse`.

### Evaluation (`evaluator.py`)

`LlamaEvaluator.eval(arch, metric, loss_func, stride, prefill_prompt)` rebuilds the
mixed-precision model and returns `(metric_dict, complexity_dict)`:
- **Weights** — HQQ loads pre-quantized models from `quant_model_paths` and swaps
  per-layer weights; AWQ/GPTQ/QEFT quantize dynamically in `sample()`.
- **KV cache** — `model/replace.py::replace_kv_cache()` injects the KIVI/HQQ cache.
- **JSD/KLD** — compares against precomputed FP16 `dense_logits`. The evaluator
  accepts injected `precomputed_{train,test}_loaders / dense_logits / key_token_list`
  so one FP-teacher pass can serve many archs (used heavily by correlation.py).
- **Complexity** — `get_net_info()` (see ThinK section).

### Search Space (`search_space/`)

- `llama.py` — `LlamaSearchSpace`, `LlamaGroupSizeSearchSpace`
  (KV options are `(bits, group_size)` tuples), `LlamaGroupSizeQEFTSearchSpace`
  (W options are `(bits, n_outlier)` tuples).
- `llama_think.py` — **`LlamaThinKSearchSpace`** (extends `LlamaGroupSizeSearchSpace`),
  the one used by `search.py`. Adds the two ThinK axes
  `k_pruning_dim_option` / `v_pruning_dim_option`. `encode()/decode()` map a flat
  integer array `(n_linear + 4) × n_block` (+4 = k, v, k_prune, v_prune) ↔ the arch
  dict. `pass_module` freezes specified layers at their max option.

### Surrogate Models (`predictor/`)

Factory `get_predictor(model, inputs, targets, device, **kwargs)`. Base predictors
(`predictor.factory.BASE_PREDICTORS`):
`rbf`, `gp`, `mlp`, `carts`, `as` (adaptive switching), `ard_gp` (sklearn ARD-GP;
`--ard_kernel matern32/52/rbf/rq`), `badd_quad` (Bayesian additive quadratic),
`gam` (monotone GAM). Each base can be wrapped with a **target transform** prefix
(`predictor/target_transform.py`): `sqrty_` (sqrt Y — JSD), `logy_` (log Y — PPL),
`logity_` (logit Y — bounded). `sqrty_gp` is a legacy alias for `sqrty_ard_gp`.
`search.py --predictor` defaults to `rbf`; `post_search.py --surrogate` enumerates
all base+transform combos via `all_surrogates()`.

### Quantization (`quant/` and `model/`)

- Weights: `quant/awq.py`, `quant/gptq.py`, `quant/qeft.py` (+ `quant/model.py`
  `get_quantized_model`); HQQ uses pre-quantized models loaded from disk.
- KV cache: `model/replace.py::replace_kv_cache()` patches the model — HQQ path sets
  `generation_config.cache_config`; KIVI/ThinK path builds a `KIVICacheConfig` and
  calls the per-architecture converter (`model/llama_kivi.py`, `qwen2_kivi.py`,
  `mistral_kivi.py`, `gemma3_kivi.py`). `enable_think` is set when `'think'` is in
  `--kv_method`. Caches: `model/KIVICache.py`, `model/HQQCache.py`.

### KV Pruning Dim Convention (ThinK) — read before touching `p`/`kvdim`

Two **deliberately different** representations of the same per-layer KV channel pruning. Do not "unify" them by flipping a sign — see the warning below.

- **prune dim** = number of `head_dim` channels *removed* (`0` = no pruning). This is the physical knob and the on-disk arch format:
  - `arch['p']['k'|'v']` storage, consumed directly by the ThinK kernels
  - search space options `k_pruning_dim_option` / `v_pruning_dim_option`
  - CLI/script knobs: `search.py --k_pruning_dim/--v_pruning_dim`, `search.sh K_PRUNING_DIM/V_PRUNING_DIM`, `compute_mem.py --k_prune_dim/--v_prune_dim/--kv_prune_dim`
  - **Rule: anything named `*prune*` is a removed-channel count.**
- **kvdim / kdim / vdim** = *retained* dim = `head_dim - prune`, mean over k+v layers (`head_dim` = no pruning). Computed at the single flip point [`utils/func.py`](utils/func.py) `get_net_info` (`net_info['kvdim'|'kdim'|'vdim']`). This is the **complexity objective**, used by `comp_obj`, the `comp_obj_min/max` budget filter (`utils/select.py`), NSGA Pareto sorting, `post_search.py`, `sample_surrogate.py`, `.stats` archives, and dir-name `obj_<min>_<max>` tokens.
  - **Rule: `kvdim/kdim/vdim` (no `prune`) is a retained-dim count.**

Why retained, not prune, for the objective: it must be monotone-with-quality like `wbits/kvbits/eff_kvbits/memory` (bigger = more capacity) so Pareto direction and the budget range stay consistent across objectives.

> ⚠️ **Never change `get_net_info`'s prune→retained flip or the `arch['p']` convention.** 781+ `.stats` archives store `kvdim` as retained (e.g. [96,128] for Llama-3.1 head_dim=128, step 0.25), and 80+ analysis scripts, all `COMP_OBJ_MIN/MAX` in scripts, and dir-name semantics depend on it. The dual representation is correct; only naming was ever the confusion (now `compute_mem` uses `*_prune_dim`).

### Decomposition study (`analysis/v5`)

`analysis/v5` is the empirical justification for the per-axis-then-combine design:
it shows the joint space (Llama 4.8e212) decomposes per method (Sobol Σ first-order
≈ 0.99, interactions ≤ 1.1%), that surrogates are monotone (Theorem-3 PF
preservation, 0/30 falsification violations, p≥0.9997), and that **ARD-GP
(Matérn-3/2)** is the most robust surrogate (R² 0.993 Llama / 0.973 Qwen). It also
documents the measurement protocol (`prefill_prompt`, `last_tokens=512`,
`stride=128`, wikitext2 JSD) and the quantile-vs-random training-pool tradeoff
(`rs50` safe default; narrow-axis models like Qwen need maximin hybrid extras).
Other `analysis/` subdirs: `answer_stride`, `kv_proxy_study`,
`batching_investigation`, `corr_keyppl`.

### Config Files (`config/`)

JSON model metadata (layer count, linear shapes, numel, vocab, head dim).
Present: `llama.json`, `qwen2.json`, `mistral.json`, `gemma3.json`
(Llama-2 7B/13B/70B, Meta-Llama-3-8B, Llama-3.1-8B-Instruct, Qwen2.5, Mistral, Gemma3).

## Key Dependencies

- `hqq==0.2.2` — weight quantization backend
- `pymoo==0.6.1.3` — NSGA2, GA, subset selection, hypervolume
- `transformers==4.45.2`
- `accelerate` — multi-GPU / launch wrapper
- `datasets`, `lm_eval` — calibration data and downstream evaluation

## Exploratory experiments (`tests/`) — findings log

Empirical studies on Llama-3.1-8B-Instruct probing extensions to the search.
Each has a runnable script under `tests/`; this log is kept updated as new
results land. Quant primitive hook point: every arch's attention calls
`model/kivi_utils.py::quant_kv_output` → module-level `fake_quant`; the cache
(generation) path calls the same `fake_quant` from `model/KIVICache.py`.
Monkeypatching that ONE function injects rotation/sink into the whole pipeline
with no change to search.py / search_space / surrogate / NSGA2.

### KV Hadamard rotation (RotateKV/QuaRot-style) — search-combinable
`test_rotation_feasibility.py`, `test_rotation_speed_micro.py`,
`test_rotation_speed_longctx.py`, `test_rotation_decode_speed.py`,
`test_rot_kvdim_interaction.py`.
- Drops into `fake_quant` as `unrot(fake_quant(rot(x)))` (Hadamard on head_dim).
  Faithful to deployed RotateKV: H orthonormal ⇒ `(QH)(Quant(KH))ᵀ == Q(Quant(KH)Hᵀ)ᵀ`.
- Quality: 2-bit JSD −22.5%, 3-bit −20.9%, 4-bit −6.6% (monotone-with-bits kept).
- Speed: search-time +1.2% (microbench +2.5ms/forward). Prefill rot is O(T)
  (+5→84ms at 16K→128K) but attention is O(T²) so relative overhead → 0; decode
  rotation is context-independent and within noise (test_rotation_decode_speed:
  16K–65K rot Δ ≈ ±0, 131K OOM at prefill).
- **rot × kvdim(ThinK) interaction**: additive+beneficial in the deployable band
  (prune≤32: 2-bit gain +0.036→+0.043) but FLIPS antagonistic at the catastrophic
  prune=64 (gain −0.032; Hadamard de-concentrates energy so ThinK can't find
  prunable channels). ⇒ make rotation a per-layer searchable axis (NAS turns it
  off in deep-prune layers); surrogate must not extrapolate rot benefit there.

### Weight rotation (QuaRot) — needs matched activation rotation
`test_weight_rotation.py`, `test_weight_rotation_all.py`,
`test_activation_rotation.py`, `test_weight_rotation_overhead.py`.
- Computational invariance `‖Wx−(WHᵀ)(Hx)‖/‖Wx‖≈1.2e-6`. **Rotating the weight
  WITHOUT rotating the activation is BROKEN** (2-bit o_proj err 1.84 ≫ 0.68); the
  matched activation rotation Hx is mandatory AND lowers output error.
- Per-linear output-error reduction (32-layer mean): o_proj 12.8%, v_proj 10.8%,
  k_proj 6.5%, q_proj 5.2%, gate/up ~2%, down_proj 3.5% (block-Hadamard approx;
  full factorized 14336=512×28 Hadamard would be larger). Activation-quant error
  also −30~35% (why QuaRot enables low-bit KV/activations).
- Integration: rotate FP model once, regenerate HQQ banks from rotated weights
  OFFLINE (AWQ/GPTQ rotate at sample() ONLINE); per-linear W-bit search then runs
  unchanged. Absorbable rotations (q/k/v/gate/up input + global hidden) fold into
  weights+RMSscale → 0 runtime; only the online down_proj/o_proj Hadamards cost.
- **End-to-end overhead** (fused `fast_hadamard_transform` kernel; down_proj
  512-block + o_proj per-head 128): **decode ≈ 0% (−0.1% ms/tok @16K ctx)**;
  prefill +12.8%/11.0%/6.3%/2.7% at seqlen 2K/8K/16K/32K (Hadamard O(T·n·log n)
  shrinks vs attention O(T²) as context grows). NOTE: a naive torch FWHT
  (log n kernel launches) mis-measures this as +50–100% — must use the fused
  CUDA kernel.

### Attention-sink protection — cheapest biggest KV win
`test_attention_sink.py`, `test_sink_ruler.py`.
- Keeping the first S tokens of the KV cache in FP (KVSink) — orthogonal to
  `residual_length` which only protects the LAST R tokens.
- 2-bit JSD: sink4 −44%, sink16 −49%, sink64 −54% (bulk from the first 4 tokens,
  0.2% mem). Stacks with rotation (sub-additive).
- RULER end-to-end: niah_multikey_2@16384 2-bit 0.95→1.00 (sink16, = FP);
  harder niah_multivalue 2-bit 0.90→0.95 (sink16). ⇒ add a small sink window to
  KIVIFakeCache/quant_kv_output; likely fixes the cwe/multivalue proxy collapse.

### MCKP / Lagrangian bit allocation vs NSGA-II
`mckp_vs_nsga2.py` (CPU, on the real 10400-config W-axis archive).
- v5 additivity ⇒ optimal allocation is a Multiple-Choice Knapsack solved exactly
  by a Lagrangian sweep (`b_i*(λ)=argmin_b[d_i(b)+λ·bits(b)]`).
- Additive fit TEST R²=0.979; Lagrangian traces the full front in 1.1 ms.
- Matches/dominates the NSGA-II measured front in 2.25–3.4 bits (±0.01–0.02 JSD)
  at ~15× fewer evals (672 vs 10400); overpredicts only at the uniform-2-bit
  extreme where additivity breaks. ⇒ replace `search.py::_next` NSGA2 with the
  closed-form sweep for separable comp_objs; refine the extreme corner by
  measurement (hybrid). Needs per-module 2/3/4-bit curves (extend sensitivity.py).

### Deferred (recorded, not yet run)
M2 Hessian/Fisher sensitivity prior into ARD-GP (BAQ); M4 cross-model surrogate
transfer (RAMP); M5 per-head + K/V-asymmetric rate-distortion (RateQuant); M6
once-for-all quant supernet (One-QuantLLM). Reasoning (M7): ThinKV thought-adaptive
compression, MixKVQ query-aware, outlier-token tracing; add CoT GSM8K/BBH eval.
