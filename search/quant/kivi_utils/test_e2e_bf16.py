"""
End-to-end bf16 KV-cache smoke test using the real KIVIDynamicCache.

Exercises the full integration:
  1. Build a KIVICacheConfig (per-layer bits/group_size, residual_length).
  2. Push a bf16 prefill chunk through `cache._update_prefill` — this
     calls `triton_quantize_and_pack_along_last_dim` on bf16 tensors and
     stores bf16 scale/min in the cache.
  3. Verify stored cache tensor dtypes are bf16.
  4. Run the same decode-time GEMV the model does
     (`cuda_bmm_fA_qB_outer` against the cached packed K/V) and compare
     to an fp32 reference dequantising the same packed ints.
  5. Repeat for fp16 (regression).

Run:  cd /NAS/SJ/actquant/search && python quant/kivi_utils/test_e2e_bf16.py
"""
import os, sys, torch

# repo root, so `model.KIVICache` resolves
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from model.KIVICache import KIVICacheConfig, KIVIDynamicCache  # noqa: E402
from quant.kivi_utils.new_pack import unpack_tensor  # noqa: E402
from quant.kivi_utils.matmul import cuda_bmm_fA_qB_outer  # noqa: E402


def _dequant_fp32(code, scale, mn, group_size, bits):
    unpacked = unpack_tensor(code, bits, pack_dim=code.ndim - 1)
    shape = unpacked.shape
    ng = shape[-1] // group_size
    grouped = unpacked.view(*shape[:-1], ng, group_size).float()
    s = scale.float().unsqueeze(-1)
    m = mn.float().unsqueeze(-1)
    return (grouped * s + m).view(shape)


def _err(out, ref):
    diff = (out.float() - ref.float()).abs()
    return (diff / ref.float().abs().clamp_min(1e-3)).mean().item()


def run(dtype: torch.dtype, label: str):
    print(f"\n=== e2e: {label} ===")
    torch.manual_seed(0)
    B, n_h, n_kv, D = 2, 8, 8, 128
    n_layer = 2
    residual_length = 128
    T_prefill = 1024 + residual_length   # quantised part + residual
    bits, gs = 4, 64

    cfg = KIVICacheConfig(
        k_bits=[bits] * n_layer, v_bits=[bits] * n_layer,
        k_group_size=[gs] * n_layer, v_group_size=[gs] * n_layer,
        residual_length=residual_length, packing=True,
    )
    cache = KIVIDynamicCache(cfg)

    # Prefill both layers
    for layer in range(n_layer):
        K = torch.randn(B, n_kv, T_prefill, D, device="cuda", dtype=dtype)
        V = torch.randn(B, n_kv, T_prefill, D, device="cuda", dtype=dtype)
        cache.update(K.clone(), V.clone(), layer_idx=layer)

    # Verify stored dtypes match input
    for layer in range(n_layer):
        code_k = cache.key_states_quant_trans_cache[layer]
        sc_k = cache.key_scale_trans_cache[layer]
        mn_k = cache.key_mn_trans_cache[layer]
        code_v = cache.value_states_quant_cache[layer]
        sc_v = cache.value_scale_cache[layer]
        mn_v = cache.value_mn_cache[layer]
        assert sc_k.dtype == dtype and mn_k.dtype == dtype, \
            f"layer {layer} K scale dtype {sc_k.dtype} != {dtype}"
        assert sc_v.dtype == dtype and mn_v.dtype == dtype, \
            f"layer {layer} V scale dtype {sc_v.dtype} != {dtype}"
        assert code_k.dtype == torch.int32 and code_v.dtype == torch.int32
        print(f"  layer {layer}: K code {tuple(code_k.shape)} scale dtype={sc_k.dtype} | "
              f"V code {tuple(code_v.shape)} scale dtype={sc_v.dtype}")

    # Decode-time GEMV against the cached packed K/V (the path
    # `forward_for_kivi_gemv` runs in `model/kivi_utils.py`).
    for layer in range(n_layer):
        # Q @ K^T
        Q = torch.randn(B, n_h, 1, D, device="cuda", dtype=dtype)
        code = cache.key_states_quant_trans_cache[layer]
        sc = cache.key_scale_trans_cache[layer]
        mn = cache.key_mn_trans_cache[layer]
        out_k = cuda_bmm_fA_qB_outer(gs, Q, code, sc, mn, bits)
        ref_k = torch.matmul(Q.float(), _dequant_fp32(code, sc, mn, gs, bits)).to(dtype)
        # attn @ V — V-cache excludes the residual_length tail; use its T.
        code_v = cache.value_states_quant_cache[layer]
        T_v_quant = code_v.shape[-2]
        attn = torch.randn(B, n_h, 1, T_v_quant, device="cuda", dtype=dtype)
        sc_v = cache.value_scale_cache[layer]
        mn_v = cache.value_mn_cache[layer]
        out_v = cuda_bmm_fA_qB_outer(gs, attn, code_v, sc_v, mn_v, bits)
        ref_v = torch.matmul(attn.float(), _dequant_fp32(code_v, sc_v, mn_v, gs, bits)).to(dtype)

        assert out_k.dtype == dtype and out_v.dtype == dtype
        ek, ev = _err(out_k, ref_k), _err(out_v, ref_v)
        print(f"  layer {layer}: decode K-path mean-rel={ek:.3e}  V-path mean-rel={ev:.3e}")
        tol = 5e-3 if dtype == torch.float16 else 2e-2
        assert ek < tol and ev < tol, f"layer {layer} {label} kernel drift too high"

    print(f"  -> {label} e2e PASS")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0))
    run(torch.float16, "fp16 (regression)")
    run(torch.bfloat16, "bf16 (new)")
    print("\nAll e2e tests passed.")
