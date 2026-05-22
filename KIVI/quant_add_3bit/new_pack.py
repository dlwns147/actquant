import triton
import triton.language as tl
import random
import numpy as np
import torch


def quant_and_pack_kcache(k: torch.FloatTensor, group_size: int, bits: int):
	assert len(k.shape) == 4
	shape = k.shape
	B, nh, T, D = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B, nh, num_groups, group_size, D)
	# Quantize
	max_int = 2 ** bits - 1
	data = k.view(new_shape)
	mn = torch.min(data, dim=-2, keepdim=True)[0]
	mx = torch.max(data, dim=-2, keepdim=True)[0]
	scale =  (mx - mn) / max_int
	data = data - mn
	data.div_(scale)
	data = data.clamp_(0, max_int).round_().to(torch.int32)
	data = data.view(shape)
	code = pack_tensor(data, bits, pack_dim=2)
	return code, scale, mn


def quant_and_pack_vcache(v: torch.FloatTensor, group_size: int, bits: int):
	shape = v.shape
	assert len(shape) == 4
	assert v.shape[-1] % group_size == 0
	num_groups = shape[-1] // group_size
	new_shape = (shape[:-1] + (num_groups, group_size))
	# Quantize
	max_int = 2 ** bits - 1
	data = v.view(new_shape)
	mn = torch.min(data, dim=-1, keepdim=True)[0]
	mx = torch.max(data, dim=-1, keepdim=True)[0]
	scale = (mx - mn) / max_int
	data = data - mn
	data.div_(scale)
	data = data.clamp_(0, max_int).round_().to(torch.int32)
	data = data.view(shape)
	# Pack
	code = pack_tensor(data, bits, pack_dim=3)
	return code, scale, mn


def unpack_and_dequant_kcache(k_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	pack_dim = 2
	assert bits in [2, 4, 8]
	assert len(k_code.shape) == 4
	data = unpack_tensor(k_code, bits, pack_dim=pack_dim)
	shape = data.shape
	num_groups = shape[pack_dim] // group_size
	data = data.view(shape[:pack_dim] + (num_groups, group_size,) + shape[pack_dim+1:])
	data = data.to(torch.float16)
	data = data * scale + mn 
	return data.view(shape)

	
def unpack_and_dequant_vcache(v_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	assert bits in [2, 4, 8]
	assert len(v_code.shape) == 4
	data = unpack_tensor(v_code, bits, pack_dim=3)
	shape = data.shape
	num_groups = shape[-1] // group_size
	data = data.view(shape[:-1] + (num_groups, group_size,))
	data = data.to(torch.float16)
	data = data * scale + mn 
	return data.view(shape)


def pack_tensor(data, bits, pack_dim):
	# Pack
	shape = data.shape
	feat_per_int = 32 // bits
	assert bits in [2,4,8], "Only 2, 4, 8 bits are supported"
	assert shape[pack_dim] % feat_per_int == 0, "Dimension length must be divisible by number of features per int"
	# BS, nh, T, nd // 16 # 16 is for 2bit
	code = torch.zeros(shape[:pack_dim] + (shape[pack_dim] // feat_per_int,)+shape[pack_dim+1:], 
					dtype=torch.int32, 
					device=data.device)
	i = 0
	row = 0
	unpacked_indices = [slice(None)] * len(data.shape)
	packed_indices = [slice(None)] * len(data.shape)
	while row < code.shape[pack_dim]:
		packed_indices[pack_dim] = row
		for j in range(i, i + (32 // bits)):
			unpacked_indices[pack_dim] = j
			code[packed_indices] |= data[unpacked_indices] << (bits * (j - i))
		i += 32 // bits
		row += 1
	return code


def unpack_tensor(v_code: torch.FloatTensor, 
				  bits: int, 
				  pack_dim: int):
	assert bits in [2,4,8]
	shape = v_code.shape
	feat_per_int = 32 // bits
	new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim+1:]
	unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
	i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
	j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
	num = 0xFF >> (8 - bits)
	packed_indices = [slice(None)] * len(new_shape)
	packed_indices[pack_dim] = i
	if pack_dim == 2:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)) & num
	elif pack_dim == 3:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
	else:
		raise NotImplementedError
	return unpacked_v_code


def unpack_tensor_3bit(qB, scales, zeros, group_size):
    """
    qB: (B, nh, K, N*3//32) int32 - packed
    scales: (B, nh, K, N//group_size) float16
    zeros: (B, nh, K, N//group_size) float16
    returns: (B, nh, K, N) float16 - dequantized
    """
    B, nh, K, packed_N = qB.shape
    N = packed_N * 32 // 3
    
    # Unpack
    unpacked = torch.zeros(B, nh, K, N, dtype=torch.int32, device=qB.device)
    
    for b in range(B):
        for h in range(nh):
            for k in range(K):
                for col in range(N // 32):
                    pack_0 = qB[b, h, k, col * 3].item()
                    pack_1 = qB[b, h, k, col * 3 + 1].item()
                    pack_2 = qB[b, h, k, col * 3 + 2].item()
                    
                    # v0~v9
                    for j in range(10):
                        unpacked[b, h, k, col * 32 + j] = (pack_0 >> (3 * j)) & 0x7
                    # v10
                    unpacked[b, h, k, col * 32 + 10] = ((pack_0 >> 30) & 0x3) | ((pack_1 & 0x1) << 2)
                    # v11~v20
                    for j in range(10):
                        unpacked[b, h, k, col * 32 + 11 + j] = (pack_1 >> (3 * j + 1)) & 0x7
                    # v21
                    unpacked[b, h, k, col * 32 + 21] = ((pack_1 >> 31) & 0x1) | ((pack_2 & 0x3) << 1)
                    # v22~v31
                    for j in range(10):
                        unpacked[b, h, k, col * 32 + 22 + j] = (pack_2 >> (3 * j + 2)) & 0x7
    
    # Dequantize
    unpacked = unpacked.to(torch.float16)
    num_groups = N // group_size
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        unpacked[:, :, :, start:end] = unpacked[:, :, :, start:end] * scales[:, :, :, g:g+1] + zeros[:, :, :, g:g+1]
    
    return unpacked


@triton.jit
def _pack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,
	code_ptr,
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	num_int_per_y_dim = num_feats // feat_per_int
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int # offset of the first element at current tile
	packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
	for i in range(feat_per_int):
		ptr = block_start + i
		element = tl.load(ptr, mask=offs_N<N, other=0.)
		element = element << (i * bits)
		# Combine the value using bitwise OR
		packed = packed | element
	tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)


@triton.jit
def _pack_along_last_dim_3bit(
	bits: tl.constexpr,		# 3bit
	intensor_ptr,
	code_ptr,		# N * (ceil(T /// 32 * 3)) -> T will be 128(Residual length of KIVI)
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,		# 10
	BLOCK_SIZE_N: tl.constexpr
):
	tl.static_assert(bits == 3, "bits must be 3")
	tl.static_assert(feat_per_int == 10, "feat_per_int must be 10")
	
	num_int_per_3y_dim = num_feats // 32 * 3
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	row_mask = offs_N < N
	
	# Remove unnecessary calculations by pre-calculating pointers
	block_start = intensor_ptr + offs_N * num_feats + yid * 32
	output_base = code_ptr + offs_N * num_int_per_3y_dim + 3 * yid
	
	# Initialize
	pack_0 = tl.zeros((BLOCK_SIZE_N,), dtype=tl.uint32)
	pack_1 = tl.zeros((BLOCK_SIZE_N,), dtype=tl.uint32)
	pack_2 = tl.zeros((BLOCK_SIZE_N,), dtype=tl.uint32)
	
	# pack_0: v0~v9 (bits 0-29) + v10's lower 2 bits (bits 30-31)
	for j in tl.static_range(10):
		element = tl.load(block_start + j, mask=row_mask, other=0).to(tl.uint32) & 0x7
		pack_0 |= element << (3 * j)
	
	# v10: 3 bits' lower 2 bits are packed into pack_0, upper 1 bit is packed into pack_1
	v10 = tl.load(block_start + 10, mask=row_mask, other=0).to(tl.uint32) & 0x7
	pack_0 |= v10 << 30
	pack_1 |= v10 >> 2  # unnecessary & 1 removal (3 bits value)
	
	# pack_1: v11~v20 (bits 1-30) + v21's lowest 1 bit (bit 31)
	for j in tl.static_range(10):
		element = tl.load(block_start + 11 + j, mask=row_mask, other=0).to(tl.uint32) & 0x7
		pack_1 |= element << (3 * j + 1)
	
	# v21: 3 bits' lowest 1 bit is packed into pack_1, upper 2 bits are packed into pack_2
	v21 = tl.load(block_start + 21, mask=row_mask, other=0).to(tl.uint32) & 0x7
	pack_1 |= v21 << 31
	pack_2 |= v21 >> 1  # unnecessary & 0x3 removal
	
	# pack_2: v22~v31 (bits 2-31)
	for j in tl.static_range(10):
		element = tl.load(block_start + 22 + j, mask=row_mask, other=0).to(tl.uint32) & 0x7
		pack_2 |= element << (3 * j + 2)
	
	# Store pre-calculated pointers
	tl.store(output_base, pack_0, mask=row_mask)
	tl.store(output_base + 1, pack_1, mask=row_mask)
	tl.store(output_base + 2, pack_2, mask=row_mask)



@triton.jit
def _minmax_along_last_dim(
	x_ptr,
	mn_ptr, mx_ptr,
	total_elements: tl.constexpr, 
	N: tl.constexpr,
	num_groups: tl.constexpr, 
	group_size: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	bid = tl.program_id(axis=0)		# CUDA에서 block Idx와 같은 역할. 현재는 1차원만 사용함.
	offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)		# 1차원 벡터가 생성
	offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]		# 2차원 벡터가 생성, Triton에서는 [:, None] -> (N, 1)과 [None, :] -> (1, N)을 사용하여 2차원 벡터를 생성할 수 있음.
	mask = offsets < total_elements
	x = tl.load(x_ptr + offsets, mask=mask)
	mx_val = tl.max(x, axis=1)
	mn_val = tl.min(x, axis=1)
	# tl.device_print('shape', mn_val[:, None].shape)
	tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
	tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)
	

def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	scale_mn_shape = B, nh, D, num_groups
	# Quantize
	data = data.reshape(new_shape)
	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
	with torch.cuda.device(data.device):
		_minmax_along_last_dim[grid](data, mn, mx,
							 data.numel(), data.shape[0], num_groups, group_size,
							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
	scale = (mx - mn) / (2 ** bit - 1)
	data = data - mn.unsqueeze(-1)
	data.div_(scale.unsqueeze(-1))
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
	data = data.view(-1, T)		# data.shape = (B * nh * D, T)
	feat_per_int = 32 // bit
	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)		# packshape = (B * nh * D, T // feat_per_int)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	with torch.cuda.device(data.device):
		_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)

def triton_quantize_and_pack_along_last_dim_3bit(data: torch.Tensor, group_size: int, bit: int):
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	scale_mn_shape = B, nh, D, num_groups
	# Quantize
	data = data.reshape(new_shape)
	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
	with torch.cuda.device(data.device):
		_minmax_along_last_dim[grid](data, mn, mx,
							 data.numel(), data.shape[0], num_groups, group_size,
							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
	scale = (mx - mn) / (2 ** bit - 1)
	data = data - mn.unsqueeze(-1)
	data.div_(scale.unsqueeze(-1))
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
	data = data.view(-1, T)		# data.shape = (B * nh * D, T)
	packshape = (np.prod(shape[:-1]), shape[-1] // 32 * 3,)		# packshape = (B * nh * D, T // feat_per_int)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	feat_per_int = 32 // 3
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // 32,)
	with torch.cuda.device(data.device):
		_pack_along_last_dim_3bit[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)


def fake_quant(inp: torch.FloatTensor, group_size: int, bits: int, along='channel'):
	if along == 'channel':
		assert len(inp.shape) == 4
		shape = inp.shape
		B, nh, T, D = shape
		# ================== Get Scale & Zeros ===============
		assert T % group_size == 0
		num_groups = T // group_size
		new_shape = (B, nh, num_groups, group_size, D)
		# Quantize
		max_int = 2 ** bits - 1
		data = inp.view(new_shape)
		mn = torch.min(data, dim=-2, keepdim=True)[0]
		mx = torch.max(data, dim=-2, keepdim=True)[0]
		scale =  (mx - mn) / max_int
		data = data - mn
		data.div_(scale)
		data = data.clamp_(0, max_int).round_()
		data = data * scale + mn 
		return data.view(shape)
	
	elif along == 'token':
		shape = inp.shape
		assert len(shape) == 4
		assert inp.shape[-1] % group_size == 0
		num_groups = shape[-1] // group_size
		new_shape = (shape[:-1] + (num_groups, group_size))
		# Quantize
		max_int = 2 ** bits - 1
		data = inp.view(new_shape)
		mn = torch.min(data, dim=-1, keepdim=True)[0]
		mx = torch.max(data, dim=-1, keepdim=True)[0]
		scale = (mx - mn) / max_int
		data = data - mn
		data.div_(scale)
		data = data.clamp_(0, max_int).round_()
		data = data * scale + mn 
		return data.view(shape)
	
	else:
		raise NotImplementedError