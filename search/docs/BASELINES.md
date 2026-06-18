# Baseline Landscape — Quantization-NAS for LLMs

Reference map of baselines for positioning **this framework**: per-layer mixed-precision
**weight** bits (HQQ/AWQ/GPTQ/QEFT) + per-layer **KV-cache** bits & group size (KIVI) +
**KV channel pruning** (ThinK), searched with a **learned surrogate + NSGA-II** over a
**JSD-to-FP16 vs memory/bits** Pareto front. Eval: WikiText2 PPL, LongBench, RULER.

Built from two adversarially-verified deep-research passes (2026-06-14). Confidence legend:
- **[V]** = claim adversarially verified this session (3/3 vote vs primary source).
- **[L]** = well-established from literature, fetched but not re-verified this pass; structural
  facts (what/how/granularity/code) are reliable, **specific PPL/bit numbers are paper-self-reported**.

The decisive takeaway is the **ingredient matrix** at the bottom: **no single baseline combines
all five of this framework's ingredients** — each prior work has at most one or two.

---

## Group 1 — Weight + KV-cache JOINT optimization  (top priority)

All "joint" methods use **fixed/heuristic** precision, not a per-layer search. This is the
framework's strongest novelty wedge.

| Method | Quantizes | Bit-allocation | Granularity | Code | Conf |
|---|---|---|---|---|---|
| **QServe / QoQ** ([2405.04532](https://arxiv.org/abs/2405.04532), MLSys'25) | W+A+KV (W4A8KV4) | **Fixed uniform**; progressive group quant, per-head asym INT4 KV, SmoothAttention. *Explicitly rejects mixed precision* | per-channel/group/head, fixed | [omniserve](https://github.com/mit-han-lab/omniserve) | [V] |
| **COMET** ([2410.12168](https://arxiv.org/abs/2410.12168), ASPLOS'25) | W+A+KV (W4A4KV4) | Mixed-prec on the **activation** axis only (FMPQ: ~84% W4A4 / ~16% W4A8) | block-wise k=128 (act.) | (ASPLOS'25) | [V] |
| **Atom** ([2310.19102](https://arxiv.org/abs/2310.19102), MLSys'24) | W+A+KV (W4A4KV4) | **Fixed heuristic**: 128 top-square-sum channels→INT8, rest INT4 gs128 (~4.25 eff bits), uniform | per-channel outlier + gs128 | [efeslab/Atom](https://github.com/efeslab/Atom) | [V] |
| **WKVQuant** ([2402.12065](https://arxiv.org/abs/2402.12065)) | **W + KV** (no act.) | **Fixed uniform W4KV4**; Past-Only Quant + 2D KV quant + cross-block reconstruction (OmniQuant-style learnable clip) | per-channel/token, fixed | — | [L] |
| **"KV Pareto"** ([2512.01953](https://arxiv.org/abs/2512.01953), Dec'25) | W (AWQ4) + KV {int2/4/8} | **Predefined-combo benchmark**, not a search (its "joint optimization frontier" framing was *refuted*) | per-token/tensor/block | — | [V] |

> **WKVQuant** is the single *genuine* joint-W+KV PTQ method (not a serving system) — and even it
> uses **uniform W4KV4**. So across all of Group 1, the W/KV bit-widths are hand-set or heuristic;
> none searches them per-layer. → **Closest baselines: QServe, Atom, WKVQuant** (joint, but fixed).

---

## Group 2 — KV-cache optimization

### 2a — Mixed-precision / any-size KV  (your stated MAIN sub-priority)

| Method | Allocation strategy | Granularity | Searched? | Code | Conf |
|---|---|---|---|---|---|
| **KVTuner** ([2502.04420](https://arxiv.org/abs/2502.04420), ICML'25) — **exemplar** | Sensitivity-aware **multi-objective Pareto SEARCH** (offline) over per-layer K/V precision **pairs**; intra-layer pair-pruning + inter-layer clustering; K > V | **Layer-wise** (head/token/channel configurable) | ✅ **Pareto** | [cmd2001/KVTuner](https://github.com/cmd2001/KVTuner) | [V] |
| **PM-KVQ** ([2505.18610](https://arxiv.org/abs/2505.18610)) | **Integer Programming** (CVXPY) over block sensitivity + progressive within-block bit-lowering; long-CoT reasoning | per-**block** | ✅ **ILP** | [thu-nics/PM-KVQ](https://github.com/thu-nics/PM-KVQ) | [V] |
| **KVmix** ([2506.08018](https://arxiv.org/pdf/2506.08018), AAAI) | **Gradient sensitivity** (L2 of loss-grad wrt K/V proj); top-20% layers high-bit, rest aggressive | per-layer | ❌ sensitivity | [LfLab-AI/KVmix](https://github.com/LfLab-AI/KVmix) | [V] |
| **KVQuant** ([2401.18079](https://arxiv.org/abs/2401.18079), NeurIPS'24) | **Per-channel Key + pre-RoPE + non-uniform (nuq)** + dense-and-sparse outliers; Fisher-weighted k-means; optional one-shot mixed-prec heuristic | per-channel (K) / per-token (V) | ❌ sensitivity | [SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant) | [V] |
| **CQ (Coupled Quant)** ([2405.03917](https://arxiv.org/abs/2405.03917), NeurIPS'24) | **Couples channel groups** into shared codebook; Fisher-info-guided centroids; **uniform fixed config** (down to 1 bit/channel, 16×) | channel-group | ❌ Fisher codebook | — | [V] |
| **ZipCache** ([2405.14256](https://arxiv.org/abs/2405.14256), NeurIPS'24) | **Per-token saliency heuristic** (normalized attn score, FlashAttn-compatible); salient 4-bit / rest 2-bit | per-token | ❌ heuristic | [ThisisBillhe/ZipCache](https://github.com/ThisisBillhe/ZipCache) | [V] |
| **MiKV** ([2402.18096](https://arxiv.org/abs/2402.18096)) | **Heavy-hitter importance** (H2O/Scissorhands); important pairs FP16, evicted pairs low-bit | per-token | ❌ heuristic | — | [V] |
| **QAQ** ([2403.04643](https://arxiv.org/abs/2403.04643)) | Quality-adaptive; **separate K vs V** strategies (proven distinct sensitivities); ~10× | per-KV-pair, K/V-asym | ❌ sensitivity | — | [V] |
| **SKVQ** ([2405.06219](https://arxiv.org/abs/2405.06219), COLM'24) | **Technique-driven**: channel reorder + clipped dynamic group quant + hi-prec window; **2-bit K / 1.5-bit V** | group, K/V-asym | ❌ fixed config | [cat538/SKVQ](https://github.com/cat538/SKVQ) | [V] |
| **PrefixQuant** ([2410.05265](https://arxiv.org/abs/2410.05265)) | **Static** training-free outlier-token prefixing (W4A4KV4 target); not searched | static rule | ❌ static | — | [V] |

> **KVTuner is your single closest direct baseline** (offline multi-objective Pareto search,
> per-layer KV). **PM-KVQ** is the closest on *allocation machinery* (it formalizes KV bits as an
> optimization problem — but ILP, not evolutionary, and per-block). Your differentiators vs both:
> (1) you also search **group size**, (2) a **learned surrogate** (they search/solve directly),
> (3) **JSD-to-FP16** objective, (4) co-search **weights + ThinK pruning** (both are KV-only).
> KVmix/KVQuant/CQ/QAQ = sensitivity/Fisher row; ZipCache/MiKV = per-token heuristic row.

### 2b — Fixed-precision (uniform) KV  (secondary)

| Method | Idea | Code | Conf |
|---|---|---|---|
| **KIVI** ([2402.02750](https://arxiv.org/abs/2402.02750), ICML'24) — **your backend** | 2-bit, per-channel Key + per-token Value, asym group-wise, FP residual window for recent tokens; tuning-free | [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) | [L] |
| **GEAR** ([2403.05527](https://arxiv.org/abs/2403.05527)) | Uniform quant **+ low-rank residual-error matrix + sparse outliers**; near-lossless 2-bit | [opengear-project/GEAR](https://github.com/opengear-project/GEAR) | [L] |
| **IntactKV** ([2403.01241](https://arxiv.org/abs/2403.01241)) | Keep **pivot / attention-sink tokens lossless**; composable with any KV quant | (HanlinTang) | [L] |
| **FlexGen** (int4 KV) | Throughput offloading; group-wise 4-bit KV + weights on single GPU | [FMInference/FlexGen](https://github.com/FMInference/FlexgenGen) | [L] |

> Uniform-bit; these are what your per-layer search should *beat at equal memory*. KIVI is the
> backend, so its **uniform 2/4-bit configs are your natural same-backend ablation**.

---

## Group 3 — KV channel-pruning / token-eviction / low-rank  (your ThinK axis)

Three distinct compression axes — note which they touch:

| Method | Compresses | Strategy | Searched? | Code | Conf |
|---|---|---|---|---|---|
| **ThinK** ([2407.21018](https://arxiv.org/abs/2407.21018), ICLR'25) — **your axis** | **Key CHANNELS** (head_dim) | Greedy query-driven score `‖Q[:,j]K[:,j]ᵀ‖_F`, **fixed pruning ratio** | ❌ heuristic | [SalesforceAIResearch/ThinK](https://github.com/SalesforceAIResearch/ThinK) | [V] |
| **Palu** ([2407.21118](https://arxiv.org/abs/2407.21118), ICLR'25) | **RANK** (hidden dim, SVD latent) | **Fisher-info automatic rank-search** across proj matrices; quant-compatible (fused Hadamard) | ✅ rank search | [shadowpa0327/Palu](https://github.com/shadowpa0327/Palu) | [V] |
| **H2O** ([2306.14048](https://arxiv.org/abs/2306.14048), NeurIPS'23) | **TOKENS** (evict) | Heavy-Hitter Oracle (attn-score) + recent; dynamic submodular | ❌ heuristic | [FMInference/H2O](https://github.com/FMInference/H2O) | [V] |
| **SnapKV** ([2404.14469](https://arxiv.org/abs/2404.14469), NeurIPS'24) | **TOKENS** (prefill select) | Observation-window voting per head + pooling | ❌ heuristic | [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV) | [L] |
| **PyramidKV** ([2406.02069](https://arxiv.org/abs/2406.02069)) | **TOKENS** (per-layer budget) | **Layer-wise pyramid budget** (more in low layers); attention-concentration based | ⚠ layer-budget rule | [Zefan-Cai/KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory) | [L] |
| **StreamingLLM** ([2309.17453](https://arxiv.org/abs/2309.17453), ICLR'24) | **TOKENS** (window) | **Attention sink** (first tokens) + sliding recent window | ❌ fixed | [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) | [L] |
| **Scissorhands** ([2305.17118](https://arxiv.org/abs/2305.17118), NeurIPS'23) | **TOKENS** (evict) | Persistence-of-importance hypothesis | ❌ heuristic | — | [L] |
| **FastGen** ([2310.01801](https://arxiv.org/abs/2310.01801), ICLR'24) | **TOKENS** (per-head policy) | **Profiles each head**, picks from {special/punct/locality/frequency} policy set | ⚠ per-head adaptive | — | [L] |

> ThinK (channels) is *your* axis — you replace its **fixed-ratio heuristic** with a **searched
> per-layer** allocation. Palu (rank) and the token-eviction family are **orthogonal/complementary**
> axes — strong as "compose-with" baselines and as a fourth potential search axis (Palu's rank
> search is the closest in spirit to making pruning learned). **Closest baselines: ThinK (same axis,
> heuristic), Palu (searched, different axis), H2O/SnapKV (token axis).**

---

## Group 4 — Weight-only mixed-precision & NAS-for-quantization  (your surrogate/MCKP/NSGA design)

### 4a — Weight quantization backends & weight mixed-precision

| Method | Idea | Allocation | Conf |
|---|---|---|---|
| **GPTQ** ([2210.17323](https://arxiv.org/abs/2210.17323)) | One-shot OBQ/Hessian error-compensation PTQ | uniform per-layer | [L] |
| **AWQ** ([2306.00978](https://arxiv.org/abs/2306.00978), MLSys'24) — *your backend* | Activation-aware per-channel scaling, protect ~1% salient | uniform group-wise | [L] |
| **OWQ** ([2306.02272](https://arxiv.org/abs/2306.02272)) — *your QEFT axis kin* | Outlier-aware: keep outlier columns in higher precision | mixed (outlier cols) | [L] |
| **SpQR** ([2306.03078](https://arxiv.org/abs/2306.03078)) | Sparse-quantized: outliers as sparse FP, rest 3-4 bit | mixed (sparse) | [L] |
| **SqueezeLLM** ([2306.07629](https://arxiv.org/abs/2306.07629)) | **Sensitivity (Fisher) non-uniform k-means** + dense-and-sparse | sensitivity, non-uniform | [L] |
| **QuIP / QuIP#** ([2307.13304](https://arxiv.org/abs/2307.13304) / [2402.04396](https://arxiv.org/abs/2402.04396)) | Incoherence processing (Hadamard) + lattice codebooks → 2-bit | uniform | [L] |
| **SliM-LLM** ([2405.14917](https://arxiv.org/abs/2405.14917), ICML'25) | **Salience-driven group-wise mixed-precision** weights; KL double-pointer search (~16 it) | group-wise, weight-only | [V] |

### 4b — NAS-for-quantization (the framework's methodological ancestors)

| Method | Search engine | Surrogate? | Bit-alloc | Domain | Conf |
|---|---|---|---|---|---|
| **HAQ** ([CVPR'19](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf)) | **RL (DDPG)**, hardware-in-loop | ❌ | per-layer mixed | vision | [L] |
| **HAWQ / V2** ([1905.03696](https://arxiv.org/abs/1905.03696) / [1911.03852](https://arxiv.org/abs/1911.03852)) | Hessian top-eigenvalue / avg-trace **sensitivity ordering** + Pareto | ❌ | per-layer mixed | vision | [L] |
| **HAWQ-V3** ([2011.10680](https://arxiv.org/abs/2011.10680), ICML'21) | **Integer Linear Programming (ILP)** under HW constraints; integer-only | ❌ | per-layer mixed | vision | [L] |
| **APQ** ([2006.08509](https://arxiv.org/abs/2006.08509), CVPR'20) | **Evolutionary search + quant-aware ACCURACY PREDICTOR (surrogate)** over a once-for-all net; joint arch+prune+quant | ✅ **predictor** | per-layer mixed | vision | [L] |
| **BatchQuant** ([2105.08952](https://arxiv.org/abs/2105.08952)) | Robust quantizer enabling a **once-for-all quantized supernet** | (supernet) | per-layer mixed | vision | [L] |

> **APQ is your closest methodological ancestor** — surrogate (accuracy predictor) + evolutionary
> search for *joint* compression — but it's vision-era and never touches KV cache or a JSD objective.
> **HAWQ-V3 (ILP)** and your own **MCKP/Lagrangian** are the closed-form-allocation cousins;
> **HAQ (RL)** is the search-based cousin. Your framework is the **first to bring this
> surrogate+NSGA-II NAS-for-quant machinery to the LLM joint W+KV+channel-pruning setting with a JSD
> objective**.

---

## ⭐ Ingredient matrix — why no single baseline matches this framework

Five ingredients this framework combines simultaneously:

| Baseline | NSGA/Pareto/evo search | Learned surrogate | JSD-to-FP16 obj | Joint **W+KV** bits | Channel pruning |
|---|:--:|:--:|:--:|:--:|:--:|
| QServe / Atom / COMET | — | — | — | ✓ (fixed) | — |
| WKVQuant | — | — | — | ✓ (fixed) | — |
| **KVTuner** | ✓ | — | — | KV-only | — |
| PM-KVQ | ✓ (ILP) | — | — | KV-only | — |
| KVQuant / CQ / Palu | — | (Fisher prior) | — | KV-only | Palu=rank |
| ZipCache / MiKV / H2O / SnapKV | — | — | — | KV-only | tokens |
| **ThinK** | — | — | — | — | ✓ (heuristic) |
| SliM-LLM | (KL pointer) | — | — | W-only | — |
| **APQ** (vision) | ✓ (evo) | ✓ | — | — | (prune) |
| HAWQ-V3 / HAQ (vision) | ✓ (ILP/RL) | — | — | W-only | — |
| **THIS FRAMEWORK** | ✅ | ✅ | ✅ | ✅ **searched** | ✅ **searched** |

**No prior method holds more than two of the five columns**, and none combines a *searched* joint
W+KV bit allocation with a *learned surrogate* and a *JSD* objective. That conjunction is the
contribution.

---

## Closest direct baselines to compare against (per group)

- **G1 (joint W+KV):** QServe (W4A8KV4), Atom (W4A4KV4), **WKVQuant (W4KV4, the genuine joint PTQ)** —
  all fixed; you beat them on *per-layer flexibility at equal memory*.
- **G2a (mixed-prec KV):** **KVTuner** (must-have, same Pareto-search spirit) + **PM-KVQ** (ILP
  allocation) + KVQuant (canonical per-channel/pre-RoPE recipe) as the sensitivity row.
- **G2b (fixed KV):** **KIVI uniform 2/4-bit** (same-backend ablation) + GEAR.
- **G3 (pruning/rank):** **ThinK** (same channel axis, heuristic ratio — your direct ablation) +
  Palu (searched rank) + H2O/SnapKV (token axis).
- **G4 (weight / NAS-quant):** SliM-LLM (weight mixed-prec) + **APQ / HAWQ-V3** as the NAS-for-quant
  method ancestors you generalize to LLMs.

---

### Sources & method
Two adversarially-verified deep-research passes (2026-06-14), 45 verified claims (3/3 vote vs primary
arXiv/venue/repo). Refuted along the way: "KV Pareto" as a joint-search framework (0–3); PrefixQuant
as a searched joint method (1–2). Items marked **[L]** (KIVI/GEAR, SnapKV/PyramidKV/StreamingLLM/
Scissorhands/FastGen, GPTQ/AWQ/OWQ/SpQR/SqueezeLLM/QuIP, HAWQ/HAQ/APQ/BatchQuant, WKVQuant) are
canonical and fetched but not re-verified this pass — **double-check exact bit/PPL numbers against the
primary source before citing in the paper.**
