import math
import torch
import numpy as np

import time

try:
    # from autogptq_cuda_256 import vecquant2matmul, vecquant3matmul, vecquant4matmul, vecquant2matmul_faster_old, vecquant3matmul_faster_old, vecquant4matmul_faster_old
    import autogptq_cuda_64
    import autogptq_cuda_256

    import custom_gptq

    _autogptq_cuda_available = True
except ImportError:
    # logger.warning("CUDA extension not installed.")
    print("GPTQ : CUDA extension not installed.")
    autogptq_cuda_256 = None
    autogptq_cuda_64 = None
    _autogptq_cuda_available = False

class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel()//x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)


class KIVILinear(torch.nn.Module):
    def __init__(self, 
                 bits,
                 group_size,
                 in_features, 
                 out_features, 
                 bias=False, 
                 kernel_switch_threshold=128,
                 use_cuda_fp16=True,
                 dtype=torch.float16):
        '''
        Asymmetric 2, 3, 4-bit Linear Layer.
        '''
        super().__init__()
        
        if bits not in [2, 3, 4]:
            raise NotImplementedError('Only 2, 3, and 4-bit Linear layers are supported.')
        
        self.bits = bits
        self.group_size = group_size if (group_size is not None and group_size != -1) else in_features
        self.in_features = in_features
        self.out_features = out_features
        
        # self.register_buffer('weight_scales',
        #                      torch.zeros((self.out_features, 1), requires_grad=False))
        # self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
        #                                                      # SubByte weight
        #                                                      dtype=torch.uint8, requires_grad=False)))
        
        self.register_buffer(
            "qweight",
            torch.zeros((in_features // 32 * self.bits, out_features), dtype=torch.int32),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(
                (
                    math.ceil(in_features / self.group_size),
                    out_features,
                ),
                dtype=torch.float,
                ## TODO : why float?
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(in_features / self.group_size), out_features),
                dtype=torch.float,
            ),
        )
        
        
        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
            
        self.half_indim = self.in_features // 2
        
        self.use_cuda_fp16 = use_cuda_fp16 if bits != 8 else False

        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12)

        self.kernel_switch_threshold = kernel_switch_threshold
        self.autogptq_cuda_available = _autogptq_cuda_available
        self.autogptq_cuda = autogptq_cuda_256
        if in_features % 256 != 0 or out_features % 256 != 0:
            self.autogptq_cuda = autogptq_cuda_64
        if in_features % 64 != 0 or out_features % 64 != 0:
            self.autogptq_cuda_available = False

        self.custom_gptq = custom_gptq
            
    def pack(self, W, scales, zeros):
        scale_zeros = zeros * scales
        self.scales = scales.t().contiguous().clone().to(dtype=torch.float)
        self.zeros = scale_zeros.t().contiguous().clone().to(dtype=torch.float)

        num_interleave = 1 if self.group_size == self.in_features else self.group_size
        scales_interleave = torch.repeat_interleave(scales, num_interleave, dim=1)
        scale_zeros_interleave = torch.repeat_interleave(scale_zeros, num_interleave, dim=1)
        
        intweight = torch.round((W + scale_zeros_interleave) / scales_interleave).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.cpu().numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight).to(scales.device)
        
    def forward(self, x):
        x_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        # if (
        #     x.device.type == "cuda"
        #     and self.autogptq_cuda_available is True
        #     and (self.kernel_switch_threshold is False or x.shape[0] < self.kernel_switch_threshold)
        # ):
        if x.shape[0] < self.kernel_switch_threshold:
            out = torch.zeros(x.shape[0], out_shape[-1], dtype=torch.float, device=x.device)
            if self.use_cuda_fp16:
                if x_dtype != torch.float16:
                    print(
                        f"The cuda-old kernel for GPTQ with use_cuda_fp16=True requires a float16 input activation, while {x_dtype} was passed. Casting to float16.\nMake sure you loaded your model with torch_dtype=torch.float16, that the model definition does not inadvertently cast to float32, or disable AMP Autocast that may produce float32 intermediate activations in the model."
                    )
                if self.bits == 2:
                    self.custom_gptq.vecquant2matmul_faster_old(
                        x,
                        self.qweight,
                        out,
                        self.scales,
                        self.zeros,
                        self.group_size,
                        self.half_indim,
                    )
                    # nvtx.pop_range()
                elif self.bits == 3:
                    # self.autogptq_cuda.vecquant3matmul_faster_old(
                    self.custom_gptq.vecquant3matmul_faster_old(
                        x,
                        self.qweight,
                        out,
                        self.scales,
                        self.zeros,
                        self.group_size,
                        self.half_indim,
                    )
                elif self.bits == 4:
                    # self.autogptq_cuda.vecquant4matmul_faster_old(
                    self.custom_gptq.vecquant4matmul_faster_old(
                        x,
                        self.qweight,
                        out,
                        self.scales,
                        self.zeros,
                        self.group_size,
                        self.half_indim,
                    )

                else:
                    raise NotImplementedError("Only 2,3,4 bits are supported.")
                # print(f"{(time.perf_counter() - start) * 1000:.5f}", end =' ')
            else:
                raise NotImplementedError("Only use_cuda_fp16=True is supported.")
                x = x.to(torch.float32)  # This is required for autocast compatibility.
                if self.bits == 2:
                    self.autogptq_cuda.vecquant2matmul_old(
                        x,
                        self.qweight,
                        out,
                        self.scales.float(),
                        self.qzeros,
                        self.group_size,
                    )
                elif self.bits == 3:
                    self.autogptq_cuda.vecquant3matmul_old(
                        x,
                        self.qweight,
                        out,
                        self.scales.float(),
                        self.qzeros,
                        self.group_size,
                    )
                elif self.bits == 4:
                    self.autogptq_cuda.vecquant4matmul_old(
                        x,
                        self.qweight,
                        out,
                        self.scales.float(),
                        self.qzeros,
                        self.group_size,
                    )
                elif self.bits == 8:
                    self.autogptq_cuda.vecquant8matmul_old(
                        x,
                        self.qweight,
                        out,
                        self.scales.float(),
                        self.qzeros,
                        self.group_size,
                    )
                else:
                    raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        else:
            if self.wf.device != self.zeros.device:
                self.wf = self.wf.to(self.zeros.device)

            if self.bits in [2, 4, 8]:
                zeros = self.zeros.half()
                zeros = zeros.reshape(-1, 1, zeros.shape[1])

                scales = self.scales.half()
                scales = scales.reshape(-1, 1, scales.shape[-1])

                weight = torch.bitwise_right_shift(
                    torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                    self.wf.unsqueeze(-1),
                ).to(torch.int16 if self.bits == 8 else torch.int8)
                weight = torch.bitwise_and(weight, (2**self.bits) - 1)
                weight = weight.reshape(-1, self.group_size, weight.shape[2])
            elif self.bits == 3:
                zeros = self.zeros.half()
                zeros = zeros.reshape(-1, 1, zeros.shape[-1])

                scales = self.scales.half()
                scales = scales.reshape(-1, 1, scales.shape[-1])

                weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(
                    -1, -1, 12, -1
                )
                weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
                weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
                weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
                weight = weight & 0x7
                weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
                weight = weight.reshape(-1, self.group_size, weight.shape[2])
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

            weight = scales * weight - zeros
            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
            out = torch.matmul(x, weight)
        out = out.to(dtype=x_dtype).reshape(
            out_shape
        )  # A cast is needed here as for some reason the vecquant2matmul_faster_old still allocate a float32 output.
        out = out + self.bias if self.bias is not None else out
        return out
        

    @staticmethod
    def from_float(module: torch.nn.Linear, quantizers=None, config=None):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        assert quantizers is not None, 'quantizers should not be None'
        
        device = module.weight.device
        
        w_bits = config.w_bits
        assert w_bits in [2, 3, 4], 'w_bits should be 2, 3 or 4'
        
        group_size = config.w_groupsize
        w_asym = config.w_asym
        w_clip = config.w_clip
        
        in_features = module.in_features
        out_features = module.out_features
        
        weight_matrix = module.weight.data
        weight_scales = quantizers.scale
        weight_zeros = quantizers.zero
        
        int_module = KIVILinear(
            bits = w_bits,
            group_size = group_size,
            in_features = in_features,
            out_features = out_features,
            bias = module.bias is not None,
            dtype = weight_matrix.dtype,
        ).to(device)
        
        # import code; code.interact('linear line 345', local=dict(globals(), **locals()))
        int_module.pack(weight_matrix, weight_scales.to(device), weight_zeros.to(device))
        
        del weight_matrix
        del weight_scales
        del weight_zeros
        torch.cuda.empty_cache()
         
        return int_module
    
    
    @staticmethod
    def from_quantized(qweight, scales, zeros, config, dev = 'cuda', bias = None):
        w_bits = config.w_bits
        assert w_bits in [2, 3, 4], 'w_bits should be 2, 3 or 4'
        
        group_size = config.w_groupsize
        w_asym = config.w_asym
        w_clip = config.w_clip
        
        in_features = qweight.shape[0] * 32 // w_bits
        out_features = qweight.shape[1]
        
        int_module = KIVILinear(
            bits = w_bits,
            group_size = group_size,
            in_features = in_features,
            out_features = out_features,
            bias = bias is not None,
        ).to(dev)
        
        int_module.qweight = qweight
        int_module.scales = scales
        int_module.zeros = zeros
        
        if bias is not None:
            int_module.bias = bias
            
        torch.cuda.empty_cache()
        
        return int_module       
        
        