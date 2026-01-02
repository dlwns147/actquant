import torch
import triton
from new_pack import triton_quantize_and_pack_along_last_dim, triton_quantize_and_pack_along_last_dim_3bit, unpack_tensor_3bit, fake_quant
import time
import numpy as np
from matmul import cuda_bmm_fA_qB_outer, cuda_bmm_fA_qB_outer_3bit

"""
    Test the correctness of the matrix multiplication kernel.
    
    CUDA_VISIBLE_DEVICES=0 python matmul_debug.py --bit 2 # 2-bit matmul
    CUDA_VISIBLE_DEVICES=0 python matmul_debug.py --bit 3 # 3-bit matmul
    CUDA_VISIBLE_DEVICES=0 python matmul_debug.py --bit 4 # 4-bit matmul

    num_tests = 5
    verbose = True
    bit = 3
    test_matmul_nbit_correctness(num_tests, verbose, bit)
"""

def test_matmul_nbit_correctness(num_tests=5, verbose=True, bit=3):
    """
    Compute the matrix multiplication C = query x key.
    Where key is quantized into 3-bit values.

    fA is of shape (B, nh, M, K) float16
    qB is of shape (B, nh, K, N // feat_per_int) int32
    scales is of shape (B, nh, K, G) float16
    zeros is of shape (B, nh, K, G) float16

    groupsize is the number of outer dimensions in each group.
    G = N // groupsize

    Returns C of shape (B, nh, M, N) float16
    """    
    cuda_time = 0
    ref_time = 0

    # 테스트 파라미터
    # B, nh, M, K = query_states.shape 
    B = 4
    nh = 32
    nh_kv = 32
    M = 1  # GEMV
    K = 128
    # N = 128  # 출력 채널, residual length
    N = 131072
    group_size = 128

    for test_idx in range(num_tests):
        if verbose:
            print(f"\n=== Test {test_idx + 1}/{num_tests} ===")
        
        # 1. 입력 데이터 생성
        fA = torch.randn(B, nh, M, K, dtype=torch.float16, device='cuda')
        
        # 2. 가중치 생성 및 패킹
        # 가중치를 먼저 생성하고 패킹
        weight = torch.randn(B, nh_kv, K, N, dtype=torch.float16, device='cuda')
        
        # Quantize and pack
        qB, scales, zeros = pack_func(weight, group_size, bit)
        
        fake_quantized_weight = fake_quant(weight, group_size, bit, along='token')

        if nh > nh_kv:
            fake_quantized_weight = fake_quantized_weight[:, :, None, :, :].expand(B, nh_kv, nh//nh_kv, K, N)
            fake_quantized_weight = fake_quantized_weight.reshape(B, nh_kv * (nh//nh_kv), K, N)

        # 3. CUDA 커널로 계산
        torch.cuda.synchronize()
        start_time = time.time()
        cuda_result = gemv_func(
            group_size, fA, qB, scales, zeros, bit
        )
        torch.cuda.synchronize()
        cuda_time += time.time() - start_time
        
        # 4. 참조 구현으로 계산
        torch.cuda.synchronize()
        start_time = time.time()
        ref = torch.matmul(fA, fake_quantized_weight)
        torch.cuda.synchronize()
        ref_time += time.time() - start_time

        # import code; code.interact('matmul_debug.py:71', local=dict(globals(), **locals()))
        
        # 상대 오차 계산
        max_diff = torch.max(torch.abs(cuda_result - ref))
        mean_diff = torch.mean(torch.abs(cuda_result - ref))
        relative_error = max_diff / (torch.max(torch.abs(ref)) + 1e-8)
        
        if verbose:
            print(f"Max diff: {max_diff.item():.6f}")
            print(f"Mean diff: {mean_diff.item():.6f}")
            print(f"Relative error: {relative_error.item():.6f}")
            print(f"CUDA result shape: {cuda_result.shape}")
            print(f"Ref result shape: {ref.shape}")
        
        # 허용 오차: FP16 정밀도 고려 (약 1e-3)
        tolerance = 1e-1
        if max_diff.item() > tolerance or relative_error.item() > tolerance:
            print(f"❌ Test {test_idx + 1} FAILED")
            print(f"   Max diff: {max_diff.item():.6f} > {tolerance}")
            print(f"   Relative error: {relative_error.item():.6f} > {tolerance}")
            
            # 첫 번째 불일치 위치 찾기
            diff_mask = torch.abs(cuda_result - ref) > tolerance
            if torch.any(diff_mask):
                indices = torch.nonzero(diff_mask)[0]
                print(f"   First mismatch at: {indices.tolist()}")
                print(f"   CUDA value: {cuda_result[tuple(indices)].item():.6f}")
                print(f"   Ref value: {ref[tuple(indices)].item():.6f}")
            return False

    print(f"\n✅ All {num_tests} tests passed!")
    print(f"Average CUDA time: {cuda_time / num_tests * 1000:.3f} ms")
    print(f"Average ref time: {ref_time / num_tests * 1000:.3f} ms")
    print(f"Speedup: {ref_time / cuda_time:.2f}x")

    return True

if __name__ == "__main__":
    print("Testing n-bit matmul kernel correctness...")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bit', type=int, default=3)
    args = parser.parse_args()  # type: ignore

    if args.bit == 3:
        pack_func = triton_quantize_and_pack_along_last_dim_3bit
        gemv_func = cuda_bmm_fA_qB_outer_3bit
    else:
        pack_func = triton_quantize_and_pack_along_last_dim
        gemv_func = cuda_bmm_fA_qB_outer

    success = test_matmul_nbit_correctness(num_tests=5, verbose=True, bit=args.bit)
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
