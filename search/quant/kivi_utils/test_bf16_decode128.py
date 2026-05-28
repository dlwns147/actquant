"""
bf16 KV-cache kernel — 128-step decode benchmark across prefill sizes.

Each row simulates: "prompt of length T already packed, generate 128
tokens one-at-a-time." Per step the hot path is exactly two
cuda_bmm_fA_qB_outer calls (Q@K^T, then attn@V) against the cached
packed K/V. We hold T fixed across the 128 steps — this matches the
KIVI runtime, where new tokens land in the fp16/bf16 residual buffer
(residual_length=128, so a 128-step decode never triggers a re-pack)
and only the residual portion grows. The quantised cache the kernel
touches stays at length T.

Reports per (T, bits, dtype):
  • mean-rel error vs fp32 reference (same packed ints — kernel drift only)
  • per-step ms
  • 128-step cumulative ms (= what the user feels during generation)
  • tokens/sec the kernel path can sustain

Run:  python test_bf16_decode128.py
"""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from new_pack import triton_quantize_and_pack_along_last_dim, unpack_tensor  # noqa: E402
from matmul import cuda_bmm_fA_qB_outer  # noqa: E402


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


def run_decode128(T, dtype, bits, gs, B, nh, D, n_decode=128,
                  n_warm=2, n_iter=3):
    """Pack a length-T cache once, then time n_decode×(Q@K + attn@V) loops."""
    torch.manual_seed(0)
    fA = torch.randn((B, nh, 1, D), device="cuda", dtype=dtype)
    attn = torch.randn((B, nh, 1, T), device="cuda", dtype=dtype)

    K = torch.randn((B, nh, T, D), device="cuda", dtype=dtype)
    Kt = K.transpose(2, 3).contiguous()
    codeK, scK, mnK = triton_quantize_and_pack_along_last_dim(Kt, gs, bits)
    V = torch.randn((B, nh, T, D), device="cuda", dtype=dtype)
    codeV, scV, mnV = triton_quantize_and_pack_along_last_dim(V, gs, bits)

    # ---- correctness (single step, both paths) ----
    refK = torch.matmul(fA.float(), _dequant_fp32(codeK, scK, mnK, gs, bits)).to(dtype)
    outK = cuda_bmm_fA_qB_outer(gs, fA, codeK, scK, mnK, bits)
    err_k = _err(outK, refK)

    refV = torch.matmul(attn.float(), _dequant_fp32(codeV, scV, mnV, gs, bits)).to(dtype)
    outV = cuda_bmm_fA_qB_outer(gs, attn, codeV, scV, mnV, bits)
    err_v = _err(outV, refV)

    # ---- bench: full 128-step decode wall time ----
    def one_decode():
        for _ in range(n_decode):
            _ = cuda_bmm_fA_qB_outer(gs, fA,   codeK, scK, mnK, bits)
            _ = cuda_bmm_fA_qB_outer(gs, attn, codeV, scV, mnV, bits)

    for _ in range(n_warm):
        one_decode()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        one_decode()
    torch.cuda.synchronize()
    ms_total = (time.perf_counter() - t0) * 1000.0 / n_iter  # ms / 128 steps
    ms_per_step = ms_total / n_decode
    toks_per_s = n_decode / (ms_total / 1000.0)

    del K, Kt, codeK, scK, mnK, V, codeV, scV, mnV, refK, outK, refV, outV
    torch.cuda.empty_cache()
    return err_k, err_v, ms_per_step, ms_total, toks_per_s


def main():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print("GPU:", torch.cuda.get_device_name(0))

    B, nh, D = 1, 32, 128
    gs = 32
    Ts = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    bits_list = [2, 4, 8]
    n_decode = 128

    fail = []
    for dtype, label in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
        print(f"\n=== {label} | B={B} nh={nh} D={D} gs={gs} | decode {n_decode} tokens ===")
        hdr = (f"  {'T':>6}  {'bits':>4}  "
               f"{'K mean-rel':>11}  {'V mean-rel':>11}  "
               f"{'ms/step':>9}  {'ms × 128':>10}  {'tok/s':>9}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for T in Ts:
            for bits in bits_list:
                ek, ev, ms_step, ms_total, tps = run_decode128(
                    T, dtype, bits, gs, B, nh, D, n_decode=n_decode)
                tol = 5e-3 if dtype == torch.float16 else 5e-2
                bad = (ek > tol) or (ev > tol)
                mark = "  !!" if bad else ""
                print(f"  {T:>6}  {bits:>4}  "
                      f"{ek:>11.3e}  {ev:>11.3e}  "
                      f"{ms_step:>9.4f}  {ms_total:>10.2f}  {tps:>9.1f}{mark}")
                if bad:
                    fail.append((label, T, bits, ek, ev))

    if fail:
        print("\nFAILURES:", fail)
        raise SystemExit(1)
    print("\nAll 128-step decode benchmarks PASSED.")


if __name__ == "__main__":
    main()
