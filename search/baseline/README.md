# Baselines

Cloned + smoke-tested baseline repos for the quant-NAS paper (see `../docs/BASELINES.md`
for the full landscape). Each baseline runs in its **own isolated venv** under
`/opt/baseline_envs/<name>` created with `--system-site-packages`, so torch/CUDA are
inherited from the base conda env (no multi-GB re-download) and the base env's
`transformers==4.50.3` is left untouched.

| Baseline | Repo | Env | Smoke test | Status |
|---|---|---|---|---|
| **KVTuner** (ICML'25, mixed-prec KV search) | [cmd2001/KVTuner](https://github.com/cmd2001/KVTuner) | `/opt/baseline_envs/kvtuner` (transformers 4.50.3) | `KVTuner/smoke_test.py` | ✅ PASS |
| **PM-KVQ** (KV ILP allocation) | [thu-nics/PM-KVQ](https://github.com/thu-nics/PM-KVQ) | `/opt/baseline_envs/pm_kvq` (transformers **4.49.0** + cvxpy) | `PM-KVQ/smoke_test.py` | ✅ PASS |
| **WKVQuant** (joint W+KV) | — | — | — | ⛔ **no public code** ([arXiv:2402.12065](https://arxiv.org/abs/2402.12065), PapersWithCode "no code available") |

Smoke tests use **local** models in `/SSD/huggingface` (no downloads): KVTuner →
Llama-3.1-8B-Instruct, PM-KVQ → Qwen2.5-7B-Instruct.

## Run

```bash
# KVTuner  (FP16 vs uniform-KV4 vs KVTuner searched per-layer mixed-precision preset)
source /opt/baseline_envs/kvtuner/bin/activate
cd KVTuner && CUDA_VISIBLE_DEVICES=0 python smoke_test.py

# PM-KVQ  (real: gradient sensitivity -> CVXPY integer-programming allocation -> progressive fake-quant gen)
source /opt/baseline_envs/pm_kvq/bin/activate
cd PM-KVQ && CUDA_VISIBLE_DEVICES=0 python smoke_test.py
```

## Setup notes / gotchas
- **KVTuner**: only depends on `transformers` (loose) → runs on base 4.50.3. Install =
  `pip install -e KVTuner/flexible_quant`. Searched mixed-precision presets ship in
  `KVTuner/calibration_presets/` (Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-3B), loaded via
  `FlexibleQuantizedCacheConfig(per_layer_quant=True, per_layer_config_path=...)`. The two
  git submodules (`lm-evaluation-harness-X`, `GAOKAO-Bench`) use SSH URLs and are only
  needed for full lm-eval/GAOKAO benchmarks — not fetched, not needed for the cache/smoke test.
- **PM-KVQ**: pins `transformers==4.49.0` (≠ base 4.50.3) → **must** be isolated. `cvxpy`
  is required (the ILP solver, `cp.SCIPY`/HiGHS). `pyext` (in requirements.txt) **fails on
  Python 3.11** (`inspect.getargspec` removed) but is only imported by the LiveCodeBench
  evaluator — skipped here; install it (or a 3.11 fork) only if you need LiveCodeBench.
  Full pipeline calibrates on RedPajama (needs `load_dataset(path).select(range(1510))`);
  the smoke test substitutes a tiny in-memory calib set to avoid the download.
  `apply_fake_pmkvq(model, rep_scales, ...)` binds `rep_scales` to the `max_keys` arg of
  `apply_smoothattention_rep` (positional), so pass real rep_scales or **bf16 ones** (no-op);
  float32 scales upcast bf16 weights and crash `q_proj`.

## WKVQuant
No public implementation exists. Options if you need a joint-W+KV baseline:
1. Substitute **QServe (W4A8KV4)** or **Atom (W4A4KV4)** — runnable joint W+(A)+KV systems
   (fixed precision); see `../docs/BASELINES.md` Group 1.
2. Reimplement WKVQuant's three components (Past-Only Quant, 2D KV quant, cross-block
   reconstruction) on top of an OmniQuant codebase — non-trivial.
3. Cite WKVQuant as related work without an empirical comparison (it reports W4KV4 uniform).
