# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Neural Architecture Search (NAS) framework for LLM quantization**. It searches for optimal per-layer bit-width configurations (weight, key-cache, value-cache quantization) across transformer model layers using surrogate-model-guided multi-objective optimization.

## Running Scripts

All main scripts are launched via `accelerate launch`:

```bash
# Linear sensitivity analysis
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --num_machines=1 --main_process_port=12345 \
  linear_sensitivity.py --gpu_id 0 --model_path /path/to/models --model_name Llama2-7b-hf \
  --method hqq --quant_model_paths "2bit_path 3bit_path 4bit_path" --quant_model_bits "2 3 4" \
  --n_sample 128 --loss_csv_file path/to/loss --config config/llama.json --loss_func jsd

# NAS search
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --num_machines=1 --main_process_port=12345 \
  search.py --gpu_id 0 --model_path /path/to/models --model_name Llama2-7b-hf \
  --method hqq --quant_model_paths "2bit_path 3bit_path 4bit_path" --quant_model_bits "2 3 4" \
  --sec_obj bits --predictor mlp --save path/to/save \
  --iterations 300 --n_doe 250 --n_iter 50 --ga_pop_size 200 \
  --config config/llama.json --loss_func jsd

# Post-search evaluation
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --num_machines=1 --main_process_port=12345 \
  post_search.py --model_path /path/to/models --model_name Llama2-7b-hf \
  --config config/llama.json --method hqq \
  --quant_model_paths "2bit_path 3bit_path 4bit_path" --quant_model_bits "2 3 4" \
  --sec_obj bits -n 5 --save path/to/save --expr path/to/search/results \
  --prefer "metric#0.0 bits#3.0" --datasets wikitext2 --target_bits_range 2.995 3.005
```

See `scripts/` for full shell script examples.

## Architecture

### Search Pipeline (`search.py`)

The `Search` class drives the full NAS loop:
1. **DOE (Design of Experiment)**: Samples `n_doe` random architectures and evaluates them via `LlamaEvaluator`
2. **Surrogate fitting**: Trains a predictor (RBF, MLP, GP, CART) on evaluated samples
3. **NSGA2 candidate generation**: Uses PyMOO's `AuxiliarySingleLevelProblem` to find Pareto-optimal candidates balancing metric vs. complexity
4. **Subset selection**: GA-based diversity filtering to select `n_iter` candidates for evaluation
5. Repeat for `--iterations` cycles; save Pareto archive to `results.json`

### Evaluation (`evaluator.py`)

`LlamaEvaluator.eval(arch)` takes an architecture config (dict of layer→bit-width), reconstructs the mixed-precision model by loading pre-quantized weights from `quant_model_paths`, applies KV cache quantization via `replace_kv_cache()`, and returns a scalar metric (JSD/KLD/cross_entropy/PPL) plus complexity (effective bits).

The evaluator expects pre-quantized models already saved to disk at paths in `--quant_model_paths`. It does not run quantization itself during search.

### Search Space (`search_space/llama.py`)

`LlamaGroupSizeSearchSpace` encodes per-layer bit assignments as integer arrays. Each position maps to a layer+projection (q/k/v/o projections and MLP gates/ups/downs). The `encode()`/`decode()` methods convert between dict configs and flat integer arrays consumed by NSGA2. `--pass_linear_list` marks layers that are frozen at their default bit-width.

### Surrogate Models (`predictor/`)

Factory: `get_predictor(type, n_obj, n_var, bounds)`. Supported types: `rbf`, `mlp`, `gp`, `cart`, `adaptive`. The predictor is trained incrementally on all evaluated `(arch_encoding, metric)` pairs and used to cheaply score the large NSGA2 population before expensive GPU evaluation.

### Quantization (`quant/`)

- `quant/hqq.py`, `quant/awq.py`, `quant/gptq.py` — per-method weight quantization wrappers
- `model/kv_cache.py` — `replace_kv_cache()` patches `LlamaAttention` to use quantized KV (HQQ or KIVI backends)
- KV cache configs: `HQQCache` / `KIVICache` with per-layer group sizes and bit-widths

### Config Files (`config/`)

JSON files (e.g., `config/llama.json`) store model metadata: layer count, linear layer shapes, model numel, vocab size, head dimensions. Required at runtime; currently supports Llama-2 (7B/13B/70B), Meta-Llama-3-8B, Llama-3.1-8B-Instruct, Mistral, Qwen2.

## Key Dependencies

- `hqq==0.2.2` — quantization backend
- `pymoo==0.6.1.3` — NSGA2, GA, subset selection
- `transformers==4.45.2`
- `accelerate` — multi-GPU / launch wrapper
- `datasets`, `lm_eval` — calibration data and downstream evaluation
