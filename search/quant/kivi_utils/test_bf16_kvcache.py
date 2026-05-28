"""
bf16 KV-cache GEMV kernel correctness + speed test.

Exercises `kivi_gemv.gemv_forward_cuda_outer_dim` (the kernel used by
`cuda_bmm_fA_qB_outer` in `matmul.py`, i.e. the path called by every
`*_kivi.py` model file for the attn-weight @ V-cache and Q @ K-cache
quantised products) for both fp16 (regression) and bf16 (new).

Run:
    cd /NAS/SJ/actquant/search/quant/kivi_utils
    python test_bf16_kvcache.py
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from new_pack import triton_quantize_and_pack_along_last_dim, unpack_tensor  # noqa: E402
from matmul import cuda_bmm_fA_qB_outer  # noqa: E402


def _set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dequant_from_packed_fp32(code, scale, mn, group_size, bits):
    """Dequantise the *same* packed integers the kernel will read, in fp32.

    The CUDA kernel converts scale/zero from {fp16,bf16}→fp32 before the
    inner multiply (fp32 accumulator). To compare apples-to-apples, the
    reference dequant must also happen in fp32; otherwise bf16 rounding
    in the reference dominates the measured "error".
    """
    unpacked = unpack_tensor(code, bits, pack_dim=code.ndim - 1)
    shape = unpacked.shape
    ng = shape[-1] // group_size
    grouped = unpacked.view(*shape[:-1], ng, group_size).float()
    s = scale.float().unsqueeze(-1)
    m = mn.float().unsqueeze(-1)
    return (grouped * s + m).view(shape)  # fp32


def _quant_then_cuda_bmm(fA, K_or_V, group_size, bits, *, key_path: bool):
    """Quantise K/V the way the model code does, then run the CUDA bmm
    kernel. Returns (kernel_out, reference_out).

    Reference dequantises the **same packed integers** the kernel reads,
    so the only difference measured is kernel-vs-pytorch matmul drift
    (not quantisation-rounding drift, which would be unfairly amplified
    in bf16 due to its coarser mantissa).
    """
    if key_path:
        K = K_or_V
        Kt = K.transpose(2, 3).contiguous()  # (B, nh, D, T)
        code, scale, mn = triton_quantize_and_pack_along_last_dim(Kt, group_size, bits)
        dequant_Kt = _dequant_from_packed_fp32(code, scale, mn, group_size, bits)
        # Match kernel: inputs upcast to fp32 for the dot, then cast result back.
        ref = torch.matmul(fA.float(), dequant_Kt).to(fA.dtype)
    else:
        V = K_or_V
        code, scale, mn = triton_quantize_and_pack_along_last_dim(V, group_size, bits)
        dequant_V = _dequant_from_packed_fp32(code, scale, mn, group_size, bits)
        ref = torch.matmul(fA.float(), dequant_V).to(fA.dtype)

    out = cuda_bmm_fA_qB_outer(group_size, fA, code, scale, mn, bits)
    return out, ref


def _max_abs_relative_error(out, ref):
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-3)
    return (diff / denom).mean().item(), diff.max().item()


def test_correctness(dtype: torch.dtype, label: str):
    print(f"\n=== correctness: {label} ===")
    _set_seed(0)
    B, nh, nh_kv, T, D = 2, 8, 8, 1024, 128
    group_size = 64
    fA = torch.randn((B, nh, 1, D), device="cuda", dtype=dtype)

    # K-path (Q @ K^T)
    K = torch.randn((B, nh_kv, T, D), device="cuda", dtype=dtype)
    # V-path (attn @ V)
    V = torch.randn((B, nh_kv, T, D), device="cuda", dtype=dtype)
    attn = torch.randn((B, nh, 1, T), device="cuda", dtype=dtype)

    for bits in [2, 4, 8]:
        out_k, ref_k = _quant_then_cuda_bmm(fA, K, group_size, bits, key_path=True)
        out_v, ref_v = _quant_then_cuda_bmm(attn, V, group_size, bits, key_path=False)
        assert out_k.dtype == dtype, f"K-path out dtype {out_k.dtype} != {dtype}"
        assert out_v.dtype == dtype, f"V-path out dtype {out_v.dtype} != {dtype}"
        assert not torch.isnan(out_k).any(), f"NaN in K-path at {bits}b"
        assert not torch.isnan(out_v).any(), f"NaN in V-path at {bits}b"

        rk_mean, rk_max = _max_abs_relative_error(out_k, ref_k)
        rv_mean, rv_max = _max_abs_relative_error(out_v, ref_v)
        print(
            f"  bits={bits} | K-path mean-rel={rk_mean:.4e} max-abs={rk_max:.3e}"
            f" || V-path mean-rel={rv_mean:.4e} max-abs={rv_max:.3e}"
        )
        # Both kernel and reference dequantise the **same** packed ints,
        # so this only measures matmul-accumulator drift. fp32 accumulator
        # in both → very tight bound.
        tol = 5e-3 if dtype == torch.float16 else 2e-2
        assert rk_mean < tol, f"K-path mean rel err {rk_mean} too high for {label} {bits}b"
        assert rv_mean < tol, f"V-path mean rel err {rv_mean} too high for {label} {bits}b"
    print(f"  -> {label} correctness PASS")


def _bench(fn, n_warmup=10, n_iter=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iter  # ms/iter


def test_speed(dtype: torch.dtype, label: str):
    print(f"\n=== speed: {label} ===")
    _set_seed(0)
    B, nh, nh_kv, T, D = 8, 32, 32, 4096, 128
    group_size = 64
    fA = torch.randn((B, nh, 1, D), device="cuda", dtype=dtype)
    K = torch.randn((B, nh_kv, T, D), device="cuda", dtype=dtype)
    for bits in [2, 4, 8]:
        code, scale, mn = triton_quantize_and_pack_along_last_dim(
            K.transpose(2, 3).contiguous(), group_size, bits
        )
        ms = _bench(
            lambda: cuda_bmm_fA_qB_outer(group_size, fA, code, scale, mn, bits)
        )
        print(
            f"  bits={bits}  shape B={B} nh={nh} T={T} D={D}  cuda_bmm: {ms:.3f} ms/iter"
        )

    # fp dense baseline for ratio
    Kdense = torch.randn((B, nh_kv, T, D), device="cuda", dtype=dtype)
    ms_ref = _bench(lambda: torch.matmul(fA, Kdense.transpose(2, 3)))
    print(f"  baseline torch.matmul (dense {label}): {ms_ref:.3f} ms/iter")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    assert torch.cuda.is_bf16_supported(), "BF16 not supported on this GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")

    test_correctness(torch.float16, "fp16 (regression)")
    test_correctness(torch.bfloat16, "bf16 (new)")

    test_speed(torch.float16, "fp16")
    test_speed(torch.bfloat16, "bf16")

    print("\nAll tests passed.")
