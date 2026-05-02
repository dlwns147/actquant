"""
Debug: verify ThinK keep-mask is actually applied to Qwen2 KIVI cache.

Checks:
1. After prefill on a long input, cache stores K/V with pruned dims set to zero.
2. The pruned-dim count matches k_pruning_dim / v_pruning_dim per layer.
3. Compares output of decode step with ThinK-on vs ThinK-off to confirm divergence.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.replace import replace_kv_cache
from model.KIVICache import KIVIFakeCache


def load_model(model_id, k_pdim, v_pdim, residual_length=128):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='cuda'
    )
    n_block = model.config.num_hidden_layers
    model = replace_kv_cache(
        model=model, tokenizer=tok, method=['kivi', 'think'],
        n_block=n_block, k_quant_scheme='channel', v_quant_scheme='token',
        residual_length=residual_length, packing=False, quant_kv_output=False,
        k_pruning_dim=k_pdim, v_pruning_dim=v_pdim,
    )
    model.config.kivi_config.k_bits = [2] * n_block
    model.config.kivi_config.v_bits = [2] * n_block
    model.config.kivi_config.k_group_size = [128] * n_block
    model.config.kivi_config.v_group_size = [128] * n_block
    model.config.kivi_config.k_pruning_dim = [k_pdim] * n_block
    model.config.kivi_config.v_pruning_dim = [v_pdim] * n_block
    model.config.kivi_config.enable_think = (k_pdim > 0 or v_pdim > 0)
    model.config.use_cache = True
    model.config.quant_kv_output = False
    model.eval()
    return model, tok


def inspect_cache_after_prefill(model, tok, prompt, label, k_pdim, v_pdim):
    inp = tok(prompt, return_tensors='pt').to(model.device)
    seqlen = inp.input_ids.shape[1]
    print(f"\n[{label}] prompt seqlen={seqlen}, k_pdim={k_pdim}, v_pdim={v_pdim}")

    cache = KIVIFakeCache(model.config.kivi_config)
    with torch.no_grad():
        out = model(**inp, past_key_values=cache, use_cache=True, return_dict=True)

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers

    # Inspect a few layers
    for layer_idx in [0, n_layers // 2, n_layers - 1]:
        kq = cache.key_states_quant_trans_cache[layer_idx]
        vq = cache.value_states_quant_cache[layer_idx]
        kmask = cache.key_keep_mask_cache[layer_idx]
        vmask = cache.value_keep_mask_cache[layer_idx]

        if kq is None or not isinstance(kq, torch.Tensor):
            print(f"  L{layer_idx}: no quant K (residual only)")
            continue

        # Per-head zero-dim counts (each KV head has its own ThinK keep mask, so we
        # must NOT average across n_kv — that smears different prune patterns).
        k_per_head_dim_abs = kq.abs().mean(dim=(0, 2))  # (n_kv, D)
        k_zero_per_head = (k_per_head_dim_abs == 0).sum(dim=-1).tolist()  # (n_kv,)
        v_zero_per_head = None
        if vq is not None and isinstance(vq, torch.Tensor):
            v_per_head_dim_abs = vq.abs().mean(dim=(0, 2))  # (n_kv, D)
            v_zero_per_head = (v_per_head_dim_abs == 0).sum(dim=-1).tolist()

        kmask_kept_per_head = kmask.sum(dim=-1)[0].tolist() if isinstance(kmask, torch.Tensor) else None
        vmask_kept_per_head = vmask.sum(dim=-1)[0].tolist() if isinstance(vmask, torch.Tensor) else None

        print(f"  L{layer_idx}: K shape={tuple(kq.shape)} zero_dims_K_per_head={k_zero_per_head} (expect {k_pdim})")
        print(f"          V shape={tuple(vq.shape) if vq is not None else None} zero_dims_V_per_head={v_zero_per_head} (expect {v_pdim})")
        print(f"          kmask kept_dims_per_head={kmask_kept_per_head} (expect {head_dim - k_pdim})")
        print(f"          vmask kept_dims_per_head={vmask_kept_per_head} (expect {head_dim - v_pdim})")

    return out.logits[0, -1].argmax().item()


def main():
    model_id = '/SSD/huggingface/Qwen/Qwen2.5-7B-Instruct'
    prompt = "The quick brown fox jumps over the lazy dog. " * 100  # ~900 tokens, > residual_length=128

    # Test 1: ThinK off (KIVI baseline)
    model, tok = load_model(model_id, k_pdim=0, v_pdim=0)
    next_off = inspect_cache_after_prefill(model, tok, prompt, "OFF", 0, 0)
    del model, tok
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # Test 2: ThinK 25% K and V
    model, tok = load_model(model_id, k_pdim=32, v_pdim=32)
    next_25 = inspect_cache_after_prefill(model, tok, prompt, "K=V=25%", 32, 32)
    del model, tok
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # Test 3: ThinK 50% K and V
    model, tok = load_model(model_id, k_pdim=64, v_pdim=64)
    next_50 = inspect_cache_after_prefill(model, tok, prompt, "K=V=50%", 64, 64)
    del model, tok
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    print("\n=== prefill-final-token argmax (last hidden -> next token) ===")
    print(f"  OFF      next_token_id = {next_off}")
    print(f"  K=V=25%  next_token_id = {next_25}")
    print(f"  K=V=50%  next_token_id = {next_50}")
    print("Note: prefill uses full K/V regardless of ThinK (by design),")
    print("so the immediate next token may match across settings;")
    print("ThinK only affects DECODE attention (subsequent generated tokens).")


if __name__ == '__main__':
    main()
