# EXPERIMENTS — status & roadmap

Status tracker for the rotation / attention-sink / MCKP exploration (2026-06).
Numbers + design live in `CLAUDE.md` (Exploratory experiments findings-log) and in
auto-memory; this file is the **DONE / IN-PROGRESS / TODO** board. Scripts are in
`tests/` unless noted.

Legend: ✅ done & measured · 🔄 running/partial · ⬜ planned · ⏸ deferred

---

## A. KV Hadamard rotation (RotateKV/QuaRot-style)
- ✅ **Search-combinable** via one `fake_quant` monkeypatch — `tests/test_rotation_feasibility.py`.
  2-bit JSD −22.5% / 3-bit −20.9% / 4-bit −6.6%; monotone-with-bits preserved.
- ✅ **Speed**: search +1.2%; long-ctx prefill +5→84 ms (16K→128K, O(T) vs attn O(T²));
  decode overhead ≈ 0 (context-independent) — `test_rotation_speed_{micro,longctx}.py`,
  `test_rotation_decode_speed.py`.
- ✅ **rot × kvdim(ThinK) interaction** — `test_rot_kvdim_interaction.py`:
  additive+beneficial at prune≤32, ANTAGONISTIC at the catastrophic prune=64.
- ⬜ Integrate as a **per-layer searchable axis** (or fixed backend) in
  KIVICacheConfig + `model/kivi_utils.py`; NAS turns rotation off in deep-prune layers.
- ⬜ **rotation backend for real deploy**: rotate FP model once, regenerate HQQ
  banks from rotated weights; uses the installed fused `fast_hadamard_transform`.

## B. Weight rotation (QuaRot)
- ✅ **Weight-only is broken; needs matched activation rotation** — `test_activation_rotation.py`.
  FP invariance 1.2e-6. Per-linear output-err reduction (32-layer mean): o_proj 12.8%,
  v_proj 10.8%, k 6.5%, q 5.2%, gate/up ~2%, down_proj 3.5% — `test_weight_rotation{,_all}.py`.
- ✅ **End-to-end overhead** (fused kernel): decode ≈ 0%, prefill +12.8/11/6.3/2.7%
  @2K/8K/16K/32K — `test_weight_rotation_overhead.py` (naive torch FWHT mis-measures +50–100%).
- ⬜ Unify weight+KV rotation (full QuaRot) + regenerate HQQ banks.

## C. Attention-sink (first S tokens FP)  ★ biggest cheap win
- ✅ **JSD**: 2-bit sink4 −44% / sink16 −49% / sink64 −54% (0.2% mem) — `test_attention_sink.py`.
- ✅ **RULER @16384** (2-bit gs128, sink0→sink16→sink16rot) — `test_sink_ruler.py --task`:
  niah_multikey_3 **0.25→0.55→0.85**, ruler_cwe 0.60→0.70→0.80, multikey_2 0.95→1.0,
  multivalue 0.90→0.95, vt 0.95→1.0, multiquery 0.95→1.0, qa_hotpot 0.53→0.567 (=FP).
- ✅ **MiniLongBench** (10 EN, 2-bit gs128): fp16 59.39, sink0 58.34, **sink16 59.33 (=FP)**,
  sink16rot 58.23 (rotation neutral/noisy on aggregate QA; hotpotqa −7.33 was 13-sample
  noise — RULER qa_hotpot neutral) — `test_sink_minilongbench.py`.
- ⬜ **Integrate a sink window** (first S tokens never quantized) into
  `KIVIFakeCache` + `quant_kv_output`, alongside `residual_length`; add to memory model.
  Likely the highest impact/cost ratio — DO THIS FIRST.

## D. MCKP bit allocation (Stage-1 per-method search)
- ✅ **Stage analysis**: MCKP belongs at **Stage 1** (additive + separable cost + 3^224
  explosion). Stage 2 keeps the surrogate (coupled KV×KVdim memory + cross-method
  interaction); MCKP/additive only a baseline there.
- ✅ **Stage-2 baseline** `tests/mckp_post_search.py` (mirrors post_search.sh): MCKP/additive
  vs ard_gp surrogate at the 5.3 GB budget — rank-corr ρ=0.978, 0 sampling.
- ✅ **`search_mckp.py` + `scripts/search_mckp.sh`** (MEASUREMENT-based, mirrors search.sh):
  size-weighted DP-MCKP over measured per-module marginals → measured frontier.
  wbits: 461 real evals (vs NSGA 10400). **vs NSGA measured front: matches/beats at
  ≥3.1 bits, worse below ~2.7 bits** (additivity breaks). HV 0.892×.
- ✅ **Downstream validation on lm_eval** (`tests/mckp_vs_nsga_lmeval.py`; RULER niah is
  binary-collapse so a poor weight-quant discriminator — `tests/mckp_vs_nsga_ruler.py`
  kept but deprioritized). MCKP vs NSGA arch at matched wbits, 9 MC/knowledge tasks
  (arc_c/e, hellaswag, winogrande, piqa, boolq, openbookqa, social_iqa, lambada) + gsm8k:
  **downstream accuracy FAITHFULLY TRACKS JSD** — NSGA wins ≤2.8 bit (9-task Δ −0.075@2.5,
  −0.046@2.8; gsm8k 0.07 vs 0.01@2.8), MCKP wins/ties ≥3.1 bit (+0.010@3.1; gsm8k 0.51 vs
  0.47@3.4). The JSD-frontier ordering transfers to real benchmarks. gsm8k (reasoning) is
  extra per-layer-allocation-sensitive (NSGA-leaning in the 2.8–3.1 mid band). ⇒ MCKP is
  downstream-competitive/superior at ≥3 bit (≈23× cheaper); refine the <2.8-bit corner by
  measurement. Also: lm_eval≈tracks JSD validates the framework's JSD objective.
  CAVEAT: `nearest` wbits match is ±0.05–0.09; JSD (measured on the actual arch) is the
  confound-free axis.
- ⬜ Run **kvbits / kvdim axes** with `scripts/search_mckp.sh` (COMP_OBJ switch).
- ⬜ **Hybrid**: measure-refine only the aggressive low-bit (<2.7 bit) corner where
  additivity breaks; production `--n_sample 128 --mckp_front_points 30-50` run per axis.
- ⬜ Extend `sensitivity.py` to per-module 2/3/4-bit curves (independent MCKP input).

## E. Reasoning / evaluation (M7)
- ⬜ Add CoT GSM8K/BBH/GPQA to the correlation harness (capture long-generation error
  accumulation the JSD proxy misses).
- ⬜ Try ThinKV (thought-adaptive) / MixKVQ (query-aware) / outlier-token tracing.

## F. Deferred methodology (⏸ recorded, revisit later — memory: project-deferred-methodology)
- ⏸ M2 Hessian/Fisher sensitivity prior into ARD-GP (BAQ).
- ⏸ M4 cross-model surrogate transfer (RAMP) — warm-start Qwen from Llama.
- ⏸ M5 per-head + K/V-asymmetric rate-distortion (RateQuant) — pair with MCKP.
- ⏸ M6 once-for-all quant supernet (One-QuantLLM).

---

## Recommended next order
1. **C: integrate the sink window** (cheapest, biggest, generalizes across tasks).
2. **A: rotation as a per-layer searchable axis** (front shifts down, decode ~0 cost).
3. **D: MCKP hybrid + kvbits/kvdim axes + production run** (≈23× cheaper Stage-1).
4. **E: reasoning eval**, then **F** deferred methods.

## GOTCHAs (bit us already)
- `get_wikitext2_trainenc` joins `n_sample` text rows → `n_sample<128` gives a too-short /
  empty / None loader. Use ≥128.
- HQQ banks on disk are **bfloat16 only** (no float16) for Llama-3.1-8B.
- `SearchThink.__init__` **pops keys out of the kwargs dict** → pass `dict(vars(args))`
  (a copy) and capture `save`/etc. before constructing it.
- MCKP DP cost MUST be **numel-weighted** (wbits/memory are size-weighted; uniform
  bit-cost mis-ranks).
- Naive torch FWHT mis-measures rotation overhead by 10–40×; use the fused CUDA kernel.
