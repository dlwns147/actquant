"""
bf16 KV-cache kernel — long-context (large T) stress test.

Existing test_bf16_kvcache.py covers T={1024, 4096}. RULER 16K and
post-search benchmarks routinely push the cache to T=16384–32768. This
script sweeps T from 1024 → 65536 and measures:
  • bit-exact / mean-rel error vs. a fp32 reference that dequantises the
    *same* packed integers the kernel reads
    (so we only measure kernel/numerical drift, not quant rounding).
  • throughput (ms/iter) for both fp16 and bf16 at each T

Hypothesis being checked: bf16's 8-bit mantissa is enough at 16K+ even
though the dot-product length on the V-path grows with T. Drift should
stay flat (kernel uses fp32 accumulator).

Run:
    cd /NAS/SJ/actquant/search/quant/kivi_utils
    python test_bf16_long_token.py
"""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from new_pack import triton_quantize_and_pack_along_last_dim, unpack_tensor  # noqa: E402
from matmul import cuda_bmm_fA_qB_outer  # noqa: E402


def _dequant_fp32(code, scale, mn, group_size, bits):
    """Dequant the kernel's exact packed int32 in fp32."""
    unpacked = unpack_tensor(code, bits, pack_dim=code.ndim - 1)
    shape = unpacked.shape
    ng = shape[-1] // group_size
    grouped = unpacked.view(*shape[:-1], ng, group_size).float()
    s = scale.float().unsqueeze(-1)
    m = mn.float().unsqueeze(-1)
    return (grouped * s + m).view(shape)


def _err(out, ref):
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-3)
    return (diff / denom).mean().item(), diff.max().item()


def _bench(fn, n_warm=5, n_iter=20):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iter


def run_for_T(T, dtype, bits, group_size, B, nh, D, *, do_speed=True):
    """One (T, dtype, bits) configuration — correctness + speed."""
    torch.manual_seed(0)
    fA = torch.randn((B, nh, 1, D), device="cuda", dtype=dtype)

    # K-path: Q (B,nh,1,D) @ K^T where K is (B,nh,T,D)
    K = torch.randn((B, nh, T, D), device="cuda", dtype=dtype)
    Kt = K.transpose(2, 3).contiguous()  # (B, nh, D, T) → quant along T
    codeK, scK, mnK = triton_quantize_and_pack_along_last_dim(Kt, group_size, bits)
    refK = torch.matmul(fA.float(), _dequant_fp32(codeK, scK, mnK, group_size, bits)).to(dtype)
    outK = cuda_bmm_fA_qB_outer(group_size, fA, codeK, scK, mnK, bits)
    kmean, kmax = _err(outK, refK)

    # V-path: attn (B,nh,1,T) @ V where V is (B,nh,T,D), V quantised along D
    attn = torch.randn((B, nh, 1, T), device="cuda", dtype=dtype)
    V = torch.randn((B, nh, T, D), device="cuda", dtype=dtype)
    codeV, scV, mnV = triton_quantize_and_pack_along_last_dim(V, group_size, bits)
    refV = torch.matmul(attn.float(), _dequant_fp32(codeV, scV, mnV, group_size, bits)).to(dtype)
    outV = cuda_bmm_fA_qB_outer(group_size, attn, codeV, scV, mnV, bits)
    vmean, vmax = _err(outV, refV)

    msK = msV = float("nan")
    if do_speed:
        msK = _bench(lambda: cuda_bmm_fA_qB_outer(group_size, fA, codeK, scK, mnK, bits))
        msV = _bench(lambda: cuda_bmm_fA_qB_outer(group_size, attn, codeV, scV, mnV, bits))

    del K, Kt, codeK, scK, mnK, refK, outK
    del V, codeV, scV, mnV, refV, outV
    torch.cuda.empty_cache()
    return dict(kmean=kmean, kmax=kmax, vmean=vmean, vmax=vmax,
                ms_k=msK, ms_v=msV)


def main():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print("GPU:", torch.cuda.get_device_name(0))

    # Shape: 1 sample, 32 heads, head_dim=128 — same as Llama-3.1-8B's KV-head
    # geometry post-GQA expand (n_kv * groups = 32). Single-batch keeps memory
    # down at 32K/64K while still exercising the long-dot-product code path.
    B, nh, D = 1, 32, 128
    group_size = 32   # matches the 2-bit gs=32 KV used in the RULER comparison
    Ts = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    bits_list = [2, 4, 8]

    # Tolerance comes from: kernel uses fp32 accumulator over T terms;
    # reference also does fp32. Drift should be ~ machine-epsilon × T plus
    # bf16/fp16 cast on the final write. ~5e-3 (fp16) / 5e-2 (bf16) is loose.
    fail = []
    for dtype, label in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
        print(f"\n=== {label} | B={B} nh={nh} D={D} gs={group_size} ===")
        header = (f"  {'T':>6}  {'bits':>4}  "
                  f"{'K mean-rel':>12}  {'K max-abs':>11}  "
                  f"{'V mean-rel':>12}  {'V max-abs':>11}  "
                  f"{'cuda_bmm K (ms)':>16}  {'cuda_bmm V (ms)':>16}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for T in Ts:
            for bits in bits_list:
                r = run_for_T(T, dtype, bits, group_size, B, nh, D, do_speed=True)
                tol = 5e-3 if dtype == torch.float16 else 5e-2
                bad = (r['kmean'] > tol) or (r['vmean'] > tol)
                marker = "  !!" if bad else ""
                print(f"  {T:>6}  {bits:>4}  "
                      f"{r['kmean']:>12.3e}  {r['kmax']:>11.3e}  "
                      f"{r['vmean']:>12.3e}  {r['vmax']:>11.3e}  "
                      f"{r['ms_k']:>16.3f}  {r['ms_v']:>16.3f}{marker}")
                if bad:
                    fail.append((label, T, bits, r['kmean'], r['vmean']))

    if fail:
        print("\nFAILURES:")
        for f in fail:
            print(" ", f)
        raise SystemExit(1)
    print("\nAll long-token bf16 / fp16 kernel checks PASSED.")


if __name__ == "__main__":
    main()
