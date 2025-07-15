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
	offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]		# 2차원 벡터가 생성, Triton에서는 [:, None]과 [None, :]을 사용하여 2차원 벡터를 생성할 수 있음.
	mask = offsets < total_elements
	x = tl.load(x_ptr + offsets, mask=mask)
	mx_val = tl.max(x, axis=1)
	mn_val = tl.min(x, axis=1)
	# tl.device_print('shape', mn_val[:, None].shape)
	tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
	tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)


# def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
# 	assert len(data.shape) == 4
# 	shape = data.shape
# 	B, nh, D, T = shape
# 	# ================== Get Scale & Zeros ===============
# 	assert T % group_size == 0
# 	num_groups = T // group_size
# 	new_shape = (B * nh * D, num_groups, group_size)
# 	scale_mn_shape = B, nh, D, num_groups
# 	# Quantize
# 	max_int = 2 ** bit - 1
# 	data = data.view(new_shape)
# 	mn = torch.min(data, dim=-1, keepdim=True)[0]
# 	mx = torch.max(data, dim=-1, keepdim=True)[0]
# 	# B, nh, D, T // group_size, 1
# 	scale = (mx - mn) / max_int
# 	data = data - mn
# 	data.div_(scale)
# 	data = data.clamp_(0, max_int).round_().to(torch.int32)
# 	scale, mn = scale.squeeze(-1), mn.squeeze(-1)
# 	data = data.view(-1, T)
# 	feat_per_int = 32 // bit
# 	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
# 	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
# 	if B <= 4:
# 		BLOCK_SIZE_N = 32
# 	else:
# 		BLOCK_SIZE_N = 128
# 	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
# 	_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
# 								data.shape[1], feat_per_int, 
# 								BLOCK_SIZE_N=BLOCK_SIZE_N, 
# 								num_warps=8)
# 	return code.view(B, nh, D, -1), scale.view(scale_mn_shape), mn.view(scale_mn_shape)
	
	

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
	

# def triton_fake_quantize_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
# 	assert len(data.shape) == 4
# 	shape = data.shape
# 	B, nh, D, T = shape
# 	# ================== Get Scale & Zeros ===============
# 	assert T % group_size == 0
# 	num_groups = T // group_size
# 	new_shape = (B * nh * D, num_groups, group_size)
# 	scale_mn_shape = B, nh, D, num_groups
# 	# Quantize
# 	data = data.reshape(new_shape)
# 	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
# 	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
# 	BLOCK_SIZE_N = 128
# 	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
# 	with torch.cuda.device(data.device):
# 		_minmax_along_last_dim[grid](data, mn, mx,
# 							 data.numel(), data.shape[0], num_groups, group_size,
# 							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
# 	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
# 	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
# 	scale = (mx - mn) / (2 ** bit - 1)
# 	data = data - mn.unsqueeze(-1)
# 	data.div_(scale.unsqueeze(-1))
# 	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
# 	data = data.view(-1, T)
# 	feat_per_int = 32 // bit
# 	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
# 	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
# 	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
# 	with torch.cuda.device(data.device):
# 		_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
# 								data.shape[1], feat_per_int, 
# 								BLOCK_SIZE_N=BLOCK_SIZE_N, 
# 								num_warps=8)
# 	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)


# def fake_quant_kcache(k: torch.FloatTensor, group_size: int, bits: int):
# 	assert len(k.shape) == 4
# 	shape = k.shape
# 	B, nh, T, D = shape
# 	# ================== Get Scale & Zeros ===============
# 	assert T % group_size == 0
# 	num_groups = T // group_size
# 	new_shape = (B, nh, num_groups, group_size, D)
# 	# Quantize
# 	max_int = 2 ** bits - 1
# 	data = k.view(new_shape)
# 	mn = torch.min(data, dim=-2, keepdim=True)[0]
# 	mx = torch.max(data, dim=-2, keepdim=True)[0]
# 	scale =  (mx - mn) / max_int
# 	data = data - mn
# 	data.div_(scale)
# 	data = data.clamp_(0, max_int).round_()
# 	data = data * scale + mn 
# 	return data.view(shape)


# def fake_quant_vcache(v: torch.FloatTensor, group_size: int, bits: int):
# 	shape = v.shape
# 	assert len(shape) == 4
# 	assert v.shape[-1] % group_size == 0
# 	num_groups = shape[-1] // group_size
# 	new_shape = (shape[:-1] + (num_groups, group_size))
# 	# Quantize
# 	max_int = 2 ** bits - 1
# 	data = v.view(new_shape)
# 	mn = torch.min(data, dim=-1, keepdim=True)[0]
# 	mx = torch.max(data, dim=-1, keepdim=True)[0]
# 	scale = (mx - mn) / max_int
# 	data = data - mn
# 	data.div_(scale)
# 	data = data.clamp_(0, max_int).round_()
# 	data = data * scale + mn 
# 	return data.view(shape)


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






import triton
import triton.language as tl

@triton.jit
def _pack_along_last_dim_3bit(
    bits:        tl.constexpr,  # bit width: 2,3,4,8
    input_ptr,                    # pointer to input int tensor (flattened 2D: N x num_feats)
    output_ptr,                   # pointer to packed output (N x num_ints)
    N:           tl.constexpr,   # number of rows
    num_feats:  tl.constexpr,   # features per row (orig last dim)
    BLOCK_SIZE: tl.constexpr     # threads per block along N
):
    # compute how many packed ints per row
    if bits == 2:
        feat_per_int = 32 // 2    # 16
    elif bits == 4:
        feat_per_int = 32 // 4    # 8
    elif bits == 8:
        feat_per_int = 32 // 8    # 4
    elif bits == 3:
        # 3-bit uses 10 values (30 bits) + spill logic for MSB/LSB across words
        feat_per_int = 10
    else:
        # unsupported bit width
        return

    # number of 32-bit ints per row (ceil division)
    num_ints = tl.cdiv(num_feats, feat_per_int)

    # compute block and thread indices
    bid = tl.program_id(axis=0)            # block id along rows
    yid = tl.program_id(axis=1)            # block id along int-groups
    offs = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_row = offs < N

    # compute base pointer for this tile and this group
    row_offset = offs * num_feats
    group_offset = yid * feat_per_int
    base_ptr = input_ptr + row_offset + group_offset

    packed = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    # common case for 2/4/8-bit: simple pack
    if bits != 3:
        for i in range(feat_per_int):
            valid = mask_row & ((group_offset + i) < num_feats)
            v = tl.load(base_ptr + i, mask=valid, other=0)
            packed |= tl.cast(v, tl.int32) << (i * bits)
    else:
        # 3-bit special: pack 10 values then spill bits across words
        # first 10 into word0 at offsets 0..27
        for i in range(10):
            idx = base_ptr + i
            valid = mask_row & ((group_offset + i) < num_feats)
            v = tl.load(idx, mask=valid, other=0)
            packed |= tl.cast(v & 0x7, tl.int32) << (i * 3)
        # handle j=10: 30->31 in word0, 32nd bit to word1 LSB
        idx10 = base_ptr + 10
        valid10 = mask_row & ((group_offset + 10) < num_feats)
        v10 = tl.load(idx10, mask=valid10, other=0)
        low10 = tl.cast(v10 & 0x7, tl.int32) << 30
        packed |= low10
        high10 = (tl.cast(v10 & 0x7, tl.int32) >> 2)
        # store overflow bit to next word at bit 0
        tl.store(output_ptr + offs * num_ints + yid + 1, high10, mask=valid10)
        # subsequent values j=11..20 pack into word1 at offsets 1..30
        for i in range(11, 21):
            idx = input_ptr + offs * num_feats + i + yid * feat_per_int
            valid = mask_row & ((group_offset + i) < num_feats)
            v = tl.load(idx, mask=valid, other=0)
            packed1 = tl.load(output_ptr + offs * num_ints + yid + 1, mask=mask_row, other=0)
            packed1 |= tl.cast(v & 0x7, tl.int32) << ((i - 10) * 3 + 1)
            tl.store(output_ptr + offs * num_ints + yid + 1, packed1, mask=mask_row)
    # store packed 0th word
    tl.store(output_ptr + offs * num_ints + yid, packed, mask=mask_row)
