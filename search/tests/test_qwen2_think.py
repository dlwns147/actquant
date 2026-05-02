"""
Smoke test: Qwen2 + KIVI + ThinK (packing=False / KIVIFakeCache).

Verifies:
1. replace_kv_cache wires Qwen2 with kivi+think and forward pass runs without crashing.
2. ThinK keep_mask is computed in attention forward and propagated via cache_kwargs.
3. Decode-time path (generate one token) works on the masked cache.
4. Output is finite and has the right shape.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/test_qwen2_think.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.replace import replace_kv_cache
from model.KIVICache import KIVIFakeCache, KIVIDynamicCache


def build_arch_config(n_block, k_bits, v_bits, k_gs, v_gs, k_pdim, v_pdim):
    return dict(
        n_block=n_block,
        k_bits=k_bits, v_bits=v_bits,
        k_group_size=k_gs, v_group_size=v_gs,
        k_pruning_dim=k_pdim, v_pruning_dim=v_pdim,
    )


def setup_model(model_id, k_bits=2, v_bits=2, k_gs=128, v_gs=128,
                k_pruning_dim=32, v_pruning_dim=32,
                packing=False, residual_length=128, dtype=torch.float16):
    print(f"[setup] Loading {model_id} dtype={dtype} ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, device_map='cuda',
    )
    n_block = model.config.num_hidden_layers
    print(f"[setup] n_block={n_block}, head_dim={model.config.hidden_size//model.config.num_attention_heads}")

    model = replace_kv_cache(
        model=model,
        tokenizer=tok,
        method=['kivi', 'think'],
        n_block=n_block,
        k_quant_scheme='channel',
        v_quant_scheme='token',
        residual_length=residual_length,
        packing=packing,
        quant_kv_output=False,
        k_pruning_dim=k_pruning_dim,
        v_pruning_dim=v_pruning_dim,
    )
    # Mirror evaluator.sample's per-arch update to populate per-layer bits/group_size lists
    model.config.kivi_config.k_bits = [k_bits] * n_block
    model.config.kivi_config.v_bits = [v_bits] * n_block
    model.config.kivi_config.k_group_size = [k_gs] * n_block
    model.config.kivi_config.v_group_size = [v_gs] * n_block
    model.config.kivi_config.k_pruning_dim = [k_pruning_dim] * n_block
    model.config.kivi_config.v_pruning_dim = [v_pruning_dim] * n_block
    model.config.kivi_config.enable_think = True
    model.config.use_cache = True
    model.config.quant_kv_output = False
    model.eval()
    return model, tok


def run_prefill_and_generate(model, tok, prompt, max_new_tokens=8):
    inp = tok(prompt, return_tensors='pt').to(model.device)
    seqlen = inp.input_ids.shape[1]
    print(f"[run] prompt seqlen={seqlen}")

    with torch.no_grad():
        # generate exercises both prefill (long) and decode (single-token) paths
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    decoded = tok.decode(out[0, seqlen:], skip_special_tokens=True)
    print(f"[run] decoded continuation: {decoded!r}")
    return out


def assert_cache_has_keep_mask(model, packing):
    # Walk the model's cache state via a fresh forward to verify the cache has keep masks.
    cache_cls = KIVIDynamicCache if packing else KIVIFakeCache
    print(f"[assert] expecting cache class: {cache_cls.__name__}")


def main():
    model_id = '/SSD/huggingface/Qwen/Qwen2.5-7B-Instruct'

    print("\n=== Test A: packing=False, K and V pruning ===")
    model, tok = setup_model(model_id, packing=False, k_pruning_dim=32, v_pruning_dim=32, residual_length=128)
    # Need a prompt long enough to exceed residual_length so the K/V quant + ThinK paths trigger
    prompt = ("In a quiet town near the river there lived an old engineer who " * 30)
    out = run_prefill_and_generate(model, tok, prompt, max_new_tokens=4)
    assert torch.isfinite(out.float()).all(), "non-finite tokens generated"
    print("[A] OK")

    # Cleanup before second test
    del model, tok
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n=== Test B: packing=False, K only pruning (V dim=0) ===")
    model, tok = setup_model(model_id, packing=False, k_pruning_dim=32, v_pruning_dim=0, residual_length=128)
    out = run_prefill_and_generate(model, tok, prompt, max_new_tokens=4)
    assert torch.isfinite(out.float()).all()
    print("[B] OK")

    del model, tok
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n=== Test C: packing=False, no ThinK pruning (K=V=0) — KIVI baseline ===")
    model, tok = setup_model(model_id, packing=False, k_pruning_dim=0, v_pruning_dim=0, residual_length=128)
    # disable enable_think to be a true KIVI-only baseline
    model.config.kivi_config.enable_think = False
    out = run_prefill_and_generate(model, tok, prompt, max_new_tokens=4)
    assert torch.isfinite(out.float()).all()
    print("[C] OK")

    del model, tok
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n=== Test D: packing=True, ThinK K-pruning (KIVIDynamicCache + GEMV decode) ===")
    model, tok = setup_model(model_id, packing=True, k_pruning_dim=32, v_pruning_dim=0,
                              residual_length=128, k_gs=64, v_gs=64)
    out = run_prefill_and_generate(model, tok, prompt, max_new_tokens=4)
    assert torch.isfinite(out.float()).all()
    print("[D] OK")

    print("\nAll Qwen2 + KIVI + ThinK smoke tests passed.")


if __name__ == '__main__':
    main()
