"""
3-bit KV-cache via the 4-bit CUDA kernel.

Idea: quantise to 3-bit (values in [0,7], scale = (mx-mn)/7), then store
each 3-bit value in a 4-bit slot with the high bit always 0. The 4-bit
kernel's dequant `scale * (packed & 0xF) + zero` is mathematically
identical to true 3-bit dequant, because:
    - the high bit is always 0 → `& 0xF` is a no-op on 3-bit values
    - scale was already derived from the 3-bit range
Cost: same memory as 4-bit packed (NOT true 3-bit packed), but the
numerical fidelity is 3-bit.

Run:  python test_3bit_via_4bit.py
"""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import triton
from new_pack import _minmax_along_last_dim, _pack_along_last_dim, unpack_tensor  # noqa: E402
from matmul import cuda_bmm_fA_qB_outer  # noqa: E402
import numpy as np


def quant_3bit_pack_as_4bit(data: torch.Tensor, group_size: int):
    """3-bit quantise along last dim, pack as 4-bit (8 vals / int32).

    Mirrors `triton_quantize_and_pack_along_last_dim` but with max_int=7
    (3-bit range) while using feat_per_int=8 (4-bit packing layout).
    """
    assert data.dim() == 4
    B, nh, D, T = data.shape
    assert T % group_size == 0
    num_groups = T // group_size
    scale_mn_shape = (B, nh, D, num_groups)
    bit_quant = 3            # quantisation range
    bit_pack = 4             # storage layout
    max_int = (1 << bit_quant) - 1   # = 7
    feat_per_int = 32 // bit_pack    # = 8

    data = data.reshape(B * nh * D, num_groups, group_size)
    mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
    BLOCK_SIZE_N = 128
    grid = lambda meta: (triton.cdiv(data.shape[0] * data.shape[1], BLOCK_SIZE_N),)
    with torch.cuda.device(data.device):
        _minmax_along_last_dim[grid](
            data, mn, mx,
            data.numel(), data.shape[0], num_groups, group_size,
            BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8,
        )
    scale = (mx - mn) / max_int  # 3-bit scale
    data = data - mn.unsqueeze(-1)
    data.div_(scale.unsqueeze(-1))
    data = data.clamp_(0, max_int).round_().to(torch.int32)  # values ∈ [0,7]
    data = data.view(-1, T)

    # Pack as 4-bit: 8 vals / int32, high bit of each slot stays 0
    packshape = (B * nh * D, T // feat_per_int)
    code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
    with torch.cuda.device(data.device):
        _pack_along_last_dim[grid](
            bit_pack, data, code, data.shape[0],
            data.shape[1], feat_per_int,
            BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8,
        )
    return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)


def _dequant_from_packed_fp32(code, scale, mn, group_size, pack_bits):
    """Dequant the same packed int32 the kernel reads, in fp32."""
    unpacked = unpack_tensor(code, pack_bits, pack_dim=code.ndim - 1)
    shape = unpacked.shape
    ng = shape[-1] // group_size
    grouped = unpacked.view(*shape[:-1], ng, group_size).float()
    s = scale.float().unsqueeze(-1)
    m = mn.float().unsqueeze(-1)
    return (grouped * s + m).view(shape)


def _max_abs_relative_error(out, ref):
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-3)
    return (diff / denom).mean().item(), diff.max().item()


def _check_high_bit_unused(code):
    """Verify every 4-bit slot has its high bit = 0 (i.e. values ∈ [0,7])."""
    unpacked = unpack_tensor(code, 4, pack_dim=code.ndim - 1)
    return int(unpacked.max().item()), int((unpacked > 7).sum().item())


def test_correctness(dtype: torch.dtype, label: str):
    print(f"\n=== correctness: 3-bit via 4-bit kernel, {label} ===")
    torch.manual_seed(0)
    B, nh, nh_kv, T, D = 2, 8, 8, 1024, 128
    gs = 64
    fA = torch.randn((B, nh, 1, D), device="cuda", dtype=dtype)

    # K path (Q @ K^T)
    K = torch.randn((B, nh_kv, T, D), device="cuda", dtype=dtype)
    Kt = K.transpose(2, 3).contiguous()
    code, scale, mn = quant_3bit_pack_as_4bit(Kt, gs)
    max_val, n_bad = _check_high_bit_unused(code)
    print(f"  packed value range max={max_val} (must be ≤7), out-of-range={n_bad}")
    assert max_val <= 7 and n_bad == 0, "3-bit clamp invariant broken"
    dequant_Kt = _dequant_from_packed_fp32(code, scale, mn, gs, pack_bits=4)
    ref = torch.matmul(fA.float(), dequant_Kt).to(dtype)
    # Use the 4-bit kernel
    out = cuda_bmm_fA_qB_outer(gs, fA, code, scale, mn, bits=4)
    rk_mean, rk_max = _max_abs_relative_error(out, ref)
    print(f"  K-path  mean-rel={rk_mean:.3e}  max-abs={rk_max:.3e}")

    # V path (attn @ V)
    V = torch.randn((B, nh_kv, T, D), device="cuda", dtype=dtype)
    attn = torch.randn((B, nh, 1, T), device="cuda", dtype=dtype)
    code, scale, mn = quant_3bit_pack_as_4bit(V, gs)
    max_val, n_bad = _check_high_bit_unused(code)
    assert max_val <= 7 and n_bad == 0
    dequant_V = _dequant_from_packed_fp32(code, scale, mn, gs, pack_bits=4)
    ref = torch.matmul(attn.float(), dequant_V).to(dtype)
    out = cuda_bmm_fA_qB_outer(gs, attn, code, scale, mn, bits=4)
    rv_mean, rv_max = _max_abs_relative_error(out, ref)
    print(f"  V-path  mean-rel={rv_mean:.3e}  max-abs={rv_max:.3e}")

    tol = 5e-3 if dtype == torch.float16 else 2e-2
    assert rk_mean < tol and rv_mean < tol, "3-via-4 kernel produced wrong result"
    print(f"  -> {label} PASS")


def test_compare_quality(dtype: torch.dtype):
    """Sanity-check: 3-via-4 should be NOISIER than true 4-bit, less noisy
    than 2-bit. Measured against the *true unquantised* dense matmul.
    V-path: attn (B,nh,1,T) @ V (B,nh,T,D) → (B,nh,1,D).
    """
    print(f"\n=== quality (vs unquantised dense): {dtype} ===")
    torch.manual_seed(0)
    B, nh, T, D = 2, 8, 1024, 128
    gs = 64
    attn = torch.randn((B, nh, 1, T), device="cuda", dtype=dtype)
    V = torch.randn((B, nh, T, D), device="cuda", dtype=dtype)
    ref_dense = torch.matmul(attn.float(), V.float()).to(dtype)

    from new_pack import triton_quantize_and_pack_along_last_dim
    for bits in [2, 4, 8]:
        code, scale, mn = triton_quantize_and_pack_along_last_dim(V, gs, bits)
        out = cuda_bmm_fA_qB_outer(gs, attn, code, scale, mn, bits)
        m, _ = _max_abs_relative_error(out, ref_dense)
        print(f"  V-path true {bits}-bit kernel:        mean-rel vs dense = {m:.4e}")

    code, scale, mn = quant_3bit_pack_as_4bit(V, gs)
    out = cuda_bmm_fA_qB_outer(gs, attn, code, scale, mn, bits=4)
    m, _ = _max_abs_relative_error(out, ref_dense)
    print(f"  V-path 3-bit-quant via 4-bit kernel:  mean-rel vs dense = {m:.4e}")


def test_speed(dtype: torch.dtype):
    print(f"\n=== speed: {dtype} (B=8 nh=32 T=4096 D=128) ===")
    torch.manual_seed(0)
    B, nh, T, D = 8, 32, 4096, 128
    gs = 64
    fA = torch.randn((B, nh, 1, D), device="cuda", dtype=dtype)
    K = torch.randn((B, nh, T, D), device="cuda", dtype=dtype)
    Kt = K.transpose(2, 3).contiguous()

    code3, scale3, mn3 = quant_3bit_pack_as_4bit(Kt, gs)

    from new_pack import triton_quantize_and_pack_along_last_dim
    code4, scale4, mn4 = triton_quantize_and_pack_along_last_dim(Kt, gs, 4)

    def bench(fn, n=50, warm=10):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / n

    ms3 = bench(lambda: cuda_bmm_fA_qB_outer(gs, fA, code3, scale3, mn3, 4))
    ms4 = bench(lambda: cuda_bmm_fA_qB_outer(gs, fA, code4, scale4, mn4, 4))
    print(f"  3-bit-via-4-bit kernel: {ms3:.3f} ms/iter")
    print(f"  true 4-bit kernel:      {ms4:.3f} ms/iter")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0))
    test_correctness(torch.float16, "fp16")
    test_correctness(torch.bfloat16, "bf16")
    test_compare_quality(torch.bfloat16)
    test_speed(torch.bfloat16)
    print("\nAll 3-bit-via-4-bit tests passed.")
