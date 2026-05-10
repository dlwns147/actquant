# KV-cache search-proxy fidelity study

How well do cheap "proxy" KV-cache simulations approximate the answer-token
loss the model would actually emit during real decoding? Search uses the
proxy to rank thousands of mixed-precision architectures, so proxy ↔ real-decode
agreement determines whether the search converges to good architectures.

## Proxies compared

All proxies use the **same** quantised KIVI model (residual_length=128,
k_quant_scheme=channel, v_quant_scheme=token) and CE/JSD against the same
answer-token positions. Only the KV-cache simulation differs.

| Proxy              | use_cache | quant_kv_output | residual | forwards |
|--------------------|:---------:|:---------------:|:--------:|:--------:|
| L_gen              | True      | False           | 128      | 1 + 128 (prefill + token-by-token) |
| L_stride S         | True      | False           | 128      | ⌈(seq+128)/S⌉ chunks (whole sequence) |
| L_prefill_stride S | True      | False           | 128      | 1 (prefill prompt) + ⌈128/S⌉ (answer chunks) |
| L_single_0         | False     | True            | 0        | 1 |
| L_single_R         | False     | True            | 128 (residual-aware) | 1 |

`L_gen` is the ground-truth measurement — what real `model.generate()` produces.

## Layout

```
01_dump_fp16.py        Dump fp16 reference: per-sample L_fp16 (CE) + answer logits.
02_measure_ce.py       CE sweep over (cfg × seqlen × stride).
03_measure_jsd.py      JSD-vs-fp16 sweep with the same loop.
04_measure_speed.py    Wall-clock benchmark of every proxy.
05_plot.py             Renders the main figures from data/.
06_measure_prefill_stride.py
                       L_prefill_stride sweep (CE + JSD), supports
                       --answer_len for longer-answer experiments.
07_plot_long_answer.py Plots prefill+stride bias as a function of stride
                       and answer_len (reads prefill_stride_jsd.json).
data/
  fp16_logits/             Per-(seq, answer_len, sample) [A,V] fp16 logits.
  fp16_baseline.json       Per-(prompt_len, answer_len) L_fp16 (CE) per sample.
  ce_results.json          Whole-sequence stride sweep CE.
  jsd_results.json         Whole-sequence stride sweep JSD.
  prefill_stride_ce.json   Prefill+answer-stride sweep CE.
  prefill_stride_jsd.json  Prefill+answer-stride sweep JSD.
  speed_results.json       Per-(seq,proxy) wall-clock ms.
figures/
  ce_quant_cost.png                 ΔL_proxy vs ΔL_gen scatter (fp16-relative).
  jsd_quant_cost.png                L_proxy_jsd vs L_gen_jsd scatter.
  proxy_correlation.png             Pearson/Spearman bars (CE row + JSD row).
  speed_vs_correlation.png          Wall-clock vs JSD-Pearson Pareto.
  prefill_stride_bias.png           Δ vs answer-stride (CE row + JSD row, A=128).
  prefill_stride_long_answer.png    Δ vs stride, faceted by answer_len.
  prefill_stride_bias_collapse.png  Bias collapses on stride/R ratio.
```

## How to reproduce

```bash
cd /NAS/SJ/actquant/search

# 0. Sanity check the residual-aware quant_kv_output() patch
python tests/test_quant_kv_output_residual.py

# 1. Dump fp16 reference (≈ 30 s)
CUDA_VISIBLE_DEVICES=0 python analysis/kv_proxy_study/01_dump_fp16.py \
  --model_path /SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct \
  --seqlens 2048,4096,8192 --answer_len 128 --n_samples 4

# 2. CE sweep (≈ 25 min for 9 configs × 3 seqlens × 4 strides)
for SEQ in 2048 4096 8192; do
  for CFG in 2:2 2:3 2:4 3:2 3:3 3:4 4:2 4:3 4:4; do
    CUDA_VISIBLE_DEVICES=0 python analysis/kv_proxy_study/02_measure_ce.py \
      --model_path /SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct \
      --residual_length 128 --strides 128,256,512,1024 \
      --n_samples 4 --prompt_len $SEQ --answer_len 128 --configs $CFG
  done
done

# 3. JSD sweep (same shape, ≈ 25 min)
for SEQ in 2048 4096 8192; do
  for CFG in 2:2 2:3 2:4 3:2 3:3 3:4 4:2 4:3 4:4; do
    CUDA_VISIBLE_DEVICES=0 python analysis/kv_proxy_study/03_measure_jsd.py \
      --model_path /SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct \
      --residual_length 128 --strides 128,256,512,1024 \
      --n_samples 4 --prompt_len $SEQ --answer_len 128 --configs $CFG
  done
done

# 4. Prefill+answer-stride sweep (≈ 25 min)
for SEQ in 2048 4096 8192; do
  for CFG in 2:2 2:3 2:4 3:2 3:3 3:4 4:2 4:3 4:4; do
    CUDA_VISIBLE_DEVICES=0 python analysis/kv_proxy_study/06_measure_prefill_stride.py \
      --model_path /SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct \
      --residual_length 128 --strides 2,4,8,16,32,64,128 \
      --n_samples 4 --prompt_len $SEQ --answer_len 128 --configs $CFG
  done
done

# 5. Speed benchmark (≈ 5 min)
for SEQ in 2048 4096 8192; do
  CUDA_VISIBLE_DEVICES=0 python analysis/kv_proxy_study/04_measure_speed.py \
    --model_path /SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct \
    --prompt_len $SEQ --answer_len 128 \
    --residual_length 128 --strides 128,256,512,1024 --kbits 2 --vbits 2
done

# 6. Long-answer prefill-stride sweep (≈ 30 min)
#    Same script as step 4 but iterating over multiple --answer_len values.
PROMPT=2048
for A in 128 256 512 1024; do
  CUDA_VISIBLE_DEVICES=0 python analysis/kv_proxy_study/01_dump_fp16.py \
    --model_path /SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct \
    --seqlens $PROMPT --answer_len $A --n_samples 4
  for CFG in 2:2 2:4 3:3 4:4; do
    CUDA_VISIBLE_DEVICES=0 python analysis/kv_proxy_study/06_measure_prefill_stride.py \
      --model_path /SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct \
      --residual_length 128 --strides 2,8,32,128,256,512,1024 \
      --n_samples 4 --prompt_len $PROMPT --answer_len $A --configs $CFG
  done
done

# 7. Render all final figures
python analysis/kv_proxy_study/05_plot.py
python analysis/kv_proxy_study/07_plot_long_answer.py
```

Steps 1-5 are resume-friendly: each writes to its JSON incrementally and
skips already-completed (cfg × seq) cells if rerun.

## Findings (Llama-3.1-8B-Instruct, WikiText-2 raw test, n_samples=4)

JSD config-level Pearson over 9 ≤4-bit configs:
```
seq     sgl0     sglR     s128     s256     s512    s1024
2048   0.951   0.970   0.991   0.996   0.994   0.996
4096   0.984   0.986   0.991   0.994   0.996   0.995
8192   0.991   0.997   0.992   0.990   0.997   0.995
```
- Stride proxies reach Pearson 0.99+ at all seqlens.
- single_R is essentially as accurate (0.97–1.00) and uses 1 forward → fastest.
- single_0 (legacy R=0) is worst at every seqlen.
- Stride values 128/256/512/1024 are within noise of each other → pick the
  largest (=fastest) stride.
- Spearman is 1.0 for all proxies at all seqlens → ranking-based search is
  insensitive to which proxy you pick within {stride*, single_R}.

Recommended: **single_R** (or stride=1024 if extra robustness wanted on K=2
configs). Avoid: single_0, stride=128.
