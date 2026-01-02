import torch
import triton
from new_pack_pv import _pack_along_last_dim_3bit, _pack_along_last_dim_3bit_old, _pack_along_last_dim_3bit_optimized
import time


def print_binary(x, row, is_packed=True):
	row_vals = x[row].tolist()

	if is_packed:
		row_vals = [v & 0xFFFFFFFF for v in row_vals]
		fmt = "{:032b}".format
		print(" ".join(fmt(v) for v in row_vals))
	else:
		row_vals = [format(v & 0xFFFFFFFF, '03b') for v in row_vals]
		for col in range(x.shape[1] // 32):
			print(f"{row_vals[col * 32 + 10][1:]}", end="")
			for i in range(0, 10):
				print(f"{row_vals[col * 32 + 9 - i]}", end="")
			print(" ", end="")

			print(f"{row_vals[col * 32 + 21][2:]}", end="")
			for i in range(11, 21):
				print(f"{row_vals[col * 32 + 31 - i]}", end="")
			print(f"{row_vals[col * 32 + 10][0]}", end="")
			print(" ", end="")

			for i in range(22, 32):
				print(f"{row_vals[col * 32 + 53 - i]}", end="")
			print(f"{row_vals[col * 32 + 21][0:2]}", end="")
			print(" ", end="")
		print()


def pack_3bit_ref(x):
	"""
	x: (N, num_feats) int tensor, values in [0,7]
	return: (N, num_feats // 32 * 3) int32
	"""
	N = x.shape[0]
	num_feats = x.shape[1]
	out = torch.zeros((N, num_feats // 32 * 3), dtype=torch.int32)

	for n in range(N):
		for col in range(num_feats // 32):
			for k in range(0, 10):
				out[n, col * 3] |= (x[n, col * 32 + k] & 0x7) << (3 * k)
			out[n, col * 3] |= (x[n, col * 32 + 10] & 0x7) << 30
			out[n, col * 3 + 1] |= (x[n, col * 32 + 10] & 0x7) >> 2
			for k in range(11, 21):
				out[n, col * 3 + 1] |= (x[n, col * 32 + k] & 0x7) << (3 * (k - 11) + 1)
			out[n, col * 3 + 1] |= (x[n, col * 32 + 21] & 0x7) << 31
			out[n, col * 3 + 2] |= (x[n, col * 32 + 21] & 0x7) >> 1
			for k in range(22, 32):
				out[n, col * 3 + 2] |= (x[n, col * 32 + k] & 0x7) << (3 * (k - 22) + 2)

	return out


def test_shift_correctness(num_tests=10):
    cpu_time = 0
    kernel_time = 0
    kernel_old_time = 0
    kernel_optimized_time = 0
    
    for _ in range(num_tests):
        # 랜덤 3-bit 값
        N = 4096
        BLOCK_SIZE_N = 128
        num_feats = 128
        x = torch.randint(0, 8, (N, num_feats), dtype=torch.int32)
        
        start_time_torch = time.time()
        ref = pack_3bit_ref(x)
        end_time_torch = time.time()
        cpu_time += end_time_torch - start_time_torch

        bits = 3
        data = x.clone().detach().to('cuda')
        
        def grid(meta): return (triton.cdiv(
            data.shape[0], BLOCK_SIZE_N), data.shape[1] // 32,)
        
        # 현재 버전 테스트
        encoded = torch.zeros((N, num_feats // 32 * 3),
                              dtype=torch.int32, device='cuda').contiguous()
        with torch.cuda.device(data.device):
            torch.cuda.synchronize()
            start_time = time.time()
            _pack_along_last_dim_3bit_optimized[grid](
                bits, data, encoded, data.shape[0], data.shape[1], 10, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8)
            torch.cuda.synchronize()
            end_time = time.time()
            kernel_time += end_time - start_time
        encoded = encoded.to('cpu')

        if not torch.equal(ref, encoded):
            print("❌ MISMATCH FOUND (current version)")
            print("input:", x[0].tolist())
            print("ref  :", ref[0].tolist())
            print("encoded:", encoded[0].tolist())
            diff = ref ^ encoded
            print("xor  :", diff[0].tolist())
            return False

    print("✅ All tests passed — shift logic is correct")
    
    print(f"Average cpu time: {cpu_time / num_tests * 1000} ms")
    print(f"Average kernel old time: {kernel_old_time / num_tests * 1000} ms")
    print(f"Average current kernel time: {kernel_time / num_tests * 1000} ms")
    print(f"Average optimized kernel time: {kernel_optimized_time / num_tests * 1000} ms")
    
    return True


print(test_shift_correctness())
# import code; code.interact('debug.py line 72', local=dict(globals(), **locals()))
