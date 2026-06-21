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

- `llama.py` — **the single search-space module** (all former files merged here;
  `llama_think.py` / `llama_qeft.py` are gone). Contents:
  - **`LlamaSearchSpace`** — the unified space used by `search.py`. Covers weights
    + KV `(bits, group_size)` + ThinK KV-pruning + QEFT outlier columns in one
    class. `encode()/decode()` map a flat integer array `(n_linear + 4) × n_block`
    (+4 = k, v, k_prune, v_prune) ↔ the arch dict; `pass_module` freezes specified
    layers at their max option. `initialize()` seeds boundary anchors over
    `w × (k|v paired) × (kdim|vdim paired)` (diagonal pairing, not cartesian) and
    fills the rest with `sample(stratify=True)` — stratified complexity-level
    sampling that spreads the `comp_obj` evenly instead of clustering at the CLT
    mean. Back-compat aliases `LlamaGroupSizeSearchSpace`,
    `LlamaGroupSizeQEFTSearchSpace`, `LlamaThinKSearchSpace` all point at it.
  - **`LlamaQEFTSearchSpace`** (+ `build_w_options`, `DEFAULT_QEFT_COLUMNS`,
    `OUTLIER_LINEARS`) — the legacy QEFT-only weight space with the flat
    `{'w':.., 'k':.., 'v':..}` arch dict / `(n_linear + 2)` encoding (W options are
    `(bits, n_outlier)` tuples). Kept for the QEFT-only weight search and its tests.

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
`test_attention_sink.py`, `test_sink_ruler.py`, `test_sink_per_layer.py`,
`verify_sink_cache.py`, `test_sink_cache_eval.py`.
- Keeping the first S tokens of the KV cache in FP (KVSink) — orthogonal to
  `residual_length` which only protects the LAST R tokens.
- **NOW A REAL IN-MODEL KNOB (not just a monkeypatch):** `KIVICacheConfig.sink`
  (+ `replace_kv_cache(sink=S)`). Implemented in `KIVICache.py::KIVIFakeCache` at
  all 6 fake_quant sites (prefill/stride/lazy_update × K/V) via `_sink_n` (position-
  aware: protects only the GLOBAL first-S tokens — committed-length bookkeeping, so
  stride/decode never mis-protect mid-seq) + `_restore_sink` (overwrite first n with
  FP). Correctness `verify_sink_cache.py`: sink≥seq ⇒ JSD vs FP-cache = 5e-4 (≈0, all
  restored) PASS; sink0 ⇒ 0.080; sink4/16/64 ⇒ ~0.041 (−48%). End-to-end via the
  CONFIG path (no monkeypatch, `test_sink_cache_eval.py --bench ruler|minilongbench`):
  RULER niah_multikey_3@16384 n=20 = FP 1.0 / 2-bit sink0 0.25 / **sink16 0.50** and
  MiniLongBench 2-bit gs128 = sink0 58.34 / **sink16 59.61** (recovers to FP 59.39) —
  both reproduce the monkeypatch rescue.
- **PACKED real-GEMV cache (`KIVIDynamicCache`, packing=True) ALSO implemented**
  (`test_sink_packed.py`). Faithful to KVSink (arXiv 2508.04257) / KIVI-K2V2* (keeps
  initial ~32 tokens FP): a FP FRONT sink buffer (`key/value_states_sink_cache`)
  peeled before quantization (mirror of the back residual) + attended via a new
  `att_qksink` term in `kivi_utils.forward_for_kivi_gemv` (positions stay ordered
  [sink | quant | back-residual]; value reconstruction splits post-softmax weights by
  position). Kernel constraint: K GEMV groups along the token axis, so the FP front
  is rounded UP to a multiple of `residual_length` (`_sink_len`; rl is a multiple of
  k_group_size, V groups along head_dim so unconstrained) — so packed S protects ≥S
  in residual_length-sized steps (harmless over-protection). K_sink is km-centered
  (softmax-invariant). Correctness: packed 2-bit decode, sink≥prompt ⇒ JSD vs FP
  decode 2.5e-4 (all FP via sink buffer through the GEMV path) PASS; sink0 0.532;
  sink128 0.0015 (first-128 FP rescues near-FP in the real packed cache). sink=0 skips
  all sink code (no regression). Stride path also prepends sink.
- **Group EXCLUSION (true KVSink) vs overwrite:** sinks must be EXCLUDED from quant
  groups, not just overwritten after, else their outlier values inflate the group
  min/max and pollute neighbours. Packed peels before quant = true exclusion (K+V).
  FakeCache: V (token scheme, per-token groups along head_dim) overwrite ≡ exclusion
  (no cross-token sharing); K (channel scheme, groups along TOKEN axis) needs true
  exclusion → `_fake_quant_k_excl_sink` quantizes only [n':] and keeps the FP front
  (n' = sink rounded UP to k_group_size). So overwrite-vs-exclude only ever mattered
  for FakeCache-K's first group (small; made the old measurement slightly conservative).
- **WIRED as `--attn_sink` (default 0 = off):** search.py → SearchThink →
  LlamaEvaluator(attn_sink=) → replace_kv_cache(sink=) → KIVICacheConfig.sink; same
  --attn_sink arg + ATTN_SINK script knob in search.py/sample_surrogate.py/post_search.py/
  correlation.py (and all 5 scripts). NOT a searched axis (per-layer S flat) — a FIXED
  global eval-time primitive like residual_length/k_quant_scheme: it is NOT stored in the
  arch dict or .stats archives, so it does NOT propagate from the search run — you must
  pass --attn_sink EXPLICITLY at every post-search stage, and it MUST match the search-time
  value (else the surrogate trains on sink-on JSD but the final pick is eval'd sink-off,
  or vice-versa). Default 0 so existing 781+ archives stay comparable; recommend 8 when
  enabling. Effective S rounds: packed → residual_length multiple, FakeCache-K →
  k_group_size multiple, V exact.
- **Memory accounting** (`utils/func.py`): sink is a THIRD KV token partition (oldest
  S tokens = fp16 at FULL head_dim, never quantized/pruned) alongside the back-residual
  (newest R fp16) and ThinK-residual (newest 32 full-dim). `compute_memory` uses a
  segment sweep over {sink, residual, think} boundaries (handles all orderings);
  `compute_cache_memory_single/_batch` (the post-search `memory` comp_obj budget) add the
  fp16-full sink term and quantize only n_token−S. Threaded via `get_net_info(...,
  attn_sink=)` ← evaluator/search self.attn_sink, and `build_nd`/`_build_lazy_comp` ←
  args.attn_sink; compute_mem.py has --attn_sink. sink=0 reproduces old numbers exactly.
  Since sink is a global constant added equally to every arch, it shifts the memory
  budget but not the relative ranking.
- **Scripts** (`scripts/*.sh`): `ATTN_SINK=0` knob + `--attn_sink ${ATTN_SINK}` in
  search.sh / search_mckp.sh / sample_surrogate.sh / post_search.sh /
  correlation_sample.sh / correlation_eval.sh. SAVE/result dirs get an abbreviated
  `SINK_TAG` = `_sk${ATTN_SINK}` (e.g. `_sk8`), appended ONLY when sink≠0 (PP_TAG
  pattern) so sink=0 dir names stay byte-identical/comparable with existing archives.
  `_sk` not `_s` because correlation dirs already use `_s${SEED}`. correlation_eval
  inherits SAVE from the sample dir so its subdirs carry the tag automatically.
- 2-bit JSD: sink4 −44%, sink16 −49%, sink64 −54% (bulk from the first 4 tokens,
  0.2% mem). Stacks with rotation (sub-additive).
- RULER end-to-end @16384 (n=20), 2-bit KIVI gs128, sink0→sink16→sink16+rot:
  niah_multikey_3 **0.25→0.55→0.85** (FP 1.0; 2-bit collapses, sink+rot rescues),
  ruler_cwe 0.60→0.70→0.80 (FP 0.65), niah_multikey_2 0.95→1.0→1.0,
  niah_multivalue 0.90→0.95, ruler_vt 0.95→1.0, niah_multiquery 0.95→1.0.
  ⇒ on HARD low-bit long-context tasks sink+rotation are essential; add a small
  sink window to KIVIFakeCache/quant_kv_output. (test_sink_ruler.py --task ...)
- MiniLongBench (10 KV-sensitive EN datasets), 2-bit gs128: fp16 59.39, sink0
  58.34, **sink16 59.33 (recovers to FP)**, sink16+rot 58.23 (rotation neutral/
  noisy on aggregate LongBench vs clearly helpful on hard RULER retrieval).
  (test_sink_minilongbench.py --mode fp16|sink0|sink16|sink16rot, 1 cfg/GPU)
- **Per-layer sink axis? NO** (`test_sink_per_layer.py`, layer-aware sink via
  patching `llama_kivi.quant_kv_output` + `kivi_config.sink[layer]`; n=8 wikitext2
  @2048, 2-bit gs128, S=16). Gate before making sink a NAS axis = is the benefit
  layer-localized? *Which* layer needs sink IS layer-dependent (leave-one-in CV=1.29,
  benefit concentrated in EARLY layers 1,2,3 + a few mid 7,9,12,14; layers 15-31 ≈0).
  BUT *how much* S per layer is NOT — the per-layer S∈{4,16,64} JSD curves are FLAT
  (spread ~0.003 = noise floor; argmin scatter is noise). Greedy knee: top-6 layers
  → only 46% of the 49% global gain, no sharp knee. ⇒ sink is near-free (0.2% mem) +
  monotone, so nothing to recover by withholding it from late layers; **bake in a
  fixed global S≈4-8 as a primitive (like residual_length), do NOT add a per-layer
  sink-length search axis.** (modes: profile|loo|ssweep; patches the per-arch kivi
  module's `quant_kv_output` name via `get_arch_module`, so works for llama/qwen2/
  mistral; gemma3 needs the text_config+hybrid-cache harness, not yet run.)
  **Cross-arch confirmed** (Qwen2.5-7B, Mistral-7B-v0.3, same protocol): both
  findings hold everywhere — which-layers IS layer-dependent (CV: Llama 1.29 /
  Mistral 1.47 / Qwen 1.46), per-layer S IS flat (argmin=noise) in all three, no
  sharp knee (top-6 → 46/55/59%). NUANCE: the profile LOCATION is arch-specific —
  Llama & Mistral concentrate in EARLY layers (1-5) but Qwen2.5's strongest is the
  LATE layer 26 (+scattered mid); global gain also varies (Llama 49% > Mistral 37%
  > Qwen 26%). ⇒ use a fixed UNIFORM global S≈8 (arch-agnostic, ~0.2% mem); an
  "early-layers-only" shortcut is fragile (breaks on Qwen).

### 1-bit KV feasibility — NO (2-bit is the floor)
`test_attention_sink.py --bits 1`, `test_kv_1bit_asym.py` (Llama-3.1-8B, wikitext2
@2048, JSD vs FP). Asked: with the full sink+rotation+small-group stack, is sub-2-bit
KV feasible? Answer: no.
- **Symmetric 1-bit collapses** to JSD ~0.61-0.68 (ln2=0.693 = max possible JSD =
  output nearly decorrelated from FP), at BOTH gs128 (0.676) and gs32 (0.635).
  Group size barely helps (gs128→gs32: 2-bit 0.159→0.050 but 1-bit only 0.676→0.635).
- **Sink/rotation give ~0 at 1-bit** (−2 to −5%, vs −44/−54% at 2-bit): they protect
  outliers, but at 1-bit every value snaps to its group min/max so there is no
  within-group dynamic range left to protect.
- **Asymmetric (gs32 sink16 rot): K is the bottleneck, not V.** K2V2 0.0287 → K2V1
  0.214 (7.5×) → K1V2 0.510 (17.8×) → K1V1 0.630 (22×). 1-bit K destroys attention
  (K feeds the softmax exponentially; V is only a linear average). But even the best
  sub-2-bit point K2V1 (eff 1.5b) is 7.5× worse than K2V2 AND worse than coarse
  symmetric 2-bit gs128 (0.159) — not a good trade.
- ⇒ **2-bit is the practical KIVI floor; never drop K below 2-bit.** Scalar group
  quant can't go sub-2-bit; that needs a different quantizer (per-head/token mixed
  precision, or vector/product quant) — out of current scope.

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
- **3-axis post_search baseline** (`mckp_post_search.py`, mirrors post_search.sh:
  same W/KV/KVdim archives + memory budget + build_nd combined space). At the
  5.316 GB budget MCKP/additive (0 sampling, 267 ms over 23M combos) picks
  wbits3.21/kvbits2.95/kvdim123.5 vs the ard_gp surrogate's
  wbits3.19/kvbits2.95/kvdim127.75 — **rank-corr ρ=0.978**, near-identical
  operating point at zero sampling cost. ⇒ MCKP/additive is a strong baseline for
  post_search; the surrogate's value is the small residual interaction + extreme
  corner. Fronts saved to tests/mckp_baseline/.
- **`search_mckp.py` + `scripts/search_mckp.sh`** (Stage-1 per-method search, the
  recommended MCKP locus; mirrors search.sh, same LlamaEvaluator/protocol).
  MEASUREMENT-based (NOT archive-fit): measures the reference (all-max-bit) + every
  per-module marginal d_m(b)=JSD(ref with m:=b)−JSD(ref) with REAL JSD, runs a
  SIZE-WEIGHTED DP-MCKP (cost MUST be numel-weighted — wbits is size-weighted),
  then MEASURES each frontier arch's real JSD → iter_mckp.stats (post_search-
  consumable). Tested on wbits (461 real evals vs NSGA's 10400; DP 67 ms):
  vs the same-protocol NSGA pp512 front on MEASURED JSD — **MCKP MATCHES/BEATS NSGA
  at ≥3.1 bits** (Δ 0 to −0.007, e.g. 3.31bit 0.055 vs 0.062) but is **WORSE below
  ~2.7 bits** (2.31bit 0.660 vs 0.519) where marginal additivity breaks (many
  modules forced to 2-bit at once). HV 0.892×. ⇒ MCKP = near-optimal per-method
  frontier at ~23× fewer evals in the deployable band; hybrid (measure-refine the
  aggressive low-bit corner). Stage analysis: MCKP belongs at Stage 1 (additive +
  separable cost + 3^224 explosion); Stage 2 keeps the surrogate (coupled KV×KVdim
  memory + cross-method interaction). GOTCHAs: n_sample>=128 (wikitext2_trainenc
  joins n_sample text rows → too few = empty/None loader); HQQ banks are bfloat16
  on disk (no float16); pass a COPY of vars(args) to SearchThink (it pops keys).

### Deferred (recorded, not yet run)
M2 Hessian/Fisher sensitivity prior into ARD-GP (BAQ); M4 cross-model surrogate
transfer (RAMP); M5 per-head + K/V-asymmetric rate-distortion (RateQuant); M6
once-for-all quant supernet (One-QuantLLM). Reasoning (M7): ThinKV thought-adaptive
compression, MixKVQ query-aware, outlier-token tracing; add CoT GSM8K/BBH eval.
