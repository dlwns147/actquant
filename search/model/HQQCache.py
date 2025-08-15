import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from packaging import version

from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_6

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import is_hqq_available, is_optimum_quanto_available, is_torch_greater_or_equal, logging
from transformers.cache_utils import CacheConfig, DynamicCache


if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

logger = logging.get_logger(__name__)


@dataclass
class QuantizedCacheConfig(CacheConfig):
    """
    Configuration class for quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        residual_length (`Optional[int]`, *optional*, defaults to 128):
            Length of the residual cache which will always be stored in original precision.
            Defaults to 128.
        compute_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to perform computations, should be same as the model's device.
    """

    def __init__(
        self,
        backend: str = "HQQ",
        k_bits : Optional[List[int]] = [],
        v_bits : Optional[List[int]] = [],
        k_group_size: Optional[List[int]] = [],
        v_group_size: Optional[List[int]] = [],

        k_quant_scheme: Optional[str] = 'channel',
        v_quant_scheme: Optional[str] = 'token',
        residual_length: Optional[int] = 128,

        compute_dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[str] = "cuda",
    ):
        self.backend = backend
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.k_group_size = k_group_size
        self.v_group_size = v_group_size

        self.axis_key = 0 if k_quant_scheme == 'token' else 1
        self.axis_value = 0 if v_quant_scheme == 'token' else 1
        self.residual_length = residual_length

        self.compute_dtype = compute_dtype
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if all([x not in [1, 2, 3, 4, 8] for x in self.k_bits]) and all([x not in [1, 2, 3, 4, 8] for x in self.v_bits]):
            raise ValueError(
                incorrect_arg_msg.format(
                    key="k_bits or v_bits",
                    correct_value="2 or 4 or 8",
                    found_value=self.k_bits or self.v_bits,
                ),
            )
        if all([x <= 0 for x in self.k_group_size]) and all([x <= 0 for x in self.v_group_size]):
            raise ValueError(
                incorrect_arg_msg.format(
                    key="k_group_size or v_group_size",
                    correct_value="a positive integer",
                    found_value=self.k_group_size or self.v_group_size,
                ),
            )
        if self.residual_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="residual_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )


class QuantizedCache(DynamicCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`
    """

    def __init__(self, cache_config: QuantizedCacheConfig) -> None:
        super().__init__()
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.k_bits = cache_config.k_bits
        self.v_bits = cache_config.v_bits
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value

        self.k_group_size = cache_config.k_group_size
        self.v_group_size = cache_config.v_group_size
        self.residual_length = cache_config.residual_length
        self.compute_dtype = cache_config.compute_dtype
        self.device = cache_config.device

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            self._quantized_key_cache.append(self._quantize(key_states.contiguous(), axis=self.axis_key, nbits=self.k_bits[layer_idx], q_group_size=self.k_group_size[layer_idx]))
            self._quantized_value_cache.append(self._quantize(value_states.contiguous(), axis=self.axis_value, nbits=self.v_bits[layer_idx], q_group_size=self.v_group_size[layer_idx]))
            self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            keys_to_return, values_to_return = key_states, value_states
        else:
            dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(keys_to_return.contiguous(), axis=self.axis_key, nbits=self.k_bits[layer_idx], q_group_size=self.k_group_size[layer_idx])
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), axis=self.axis_value, nbits=self.v_bits[layer_idx], q_group_size=self.v_group_size[layer_idx]
                )
                self.key_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")


class HQQQuantizedCache(QuantizedCache):
    """
    Quantized Cache class that uses `HQQ` as a backend to perform quantization. Current implementation supports `int2`, `int4`, `int8` dtypes.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

    Example:

        ```python
        >>> # Run pip install hqq first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HQQQuantizedCache, QuantizedCacheConfig

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4, axis_key=1, axis_value=1)
        >>> past_key_values = HQQQuantizedCache(cache_config=cache_config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HQQQuantizedCache()
        ```
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        if all([x not in [1, 2, 3, 4, 8] for x in self.k_bits]) and all([x not in [1, 2, 3, 4, 8] for x in self.v_bits]):
            raise ValueError(
                f"`k_bits or v_bits` for `HQQ` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {self.k_bits or self.v_bits}"
            )

        if self.axis_key not in [0, 1]:
            raise ValueError(f"`axis_key` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_key}")

        if self.axis_value not in [0, 1]:
            raise ValueError(f"`axis_value` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_value}")

        self.quantizer = HQQQuantizer

    def _quantize(self, tensor, axis, nbits, q_group_size):
        qtensor, meta = self.quantizer.quantize(
            tensor,
            axis=axis,
            device=self.device,
            compute_dtype=self.compute_dtype,
            nbits=nbits,
            group_size=q_group_size,
        )
        meta["compute_dtype"] = self.compute_dtype
        self.quantizer.cuda(qtensor, meta=meta, device=self.device)  # Move to device and cast to dtype
        meta["scale"] = meta["scale"].to(qtensor.device)
        meta["zero"] = meta["zero"].to(qtensor.device)
        return qtensor, meta

    def _dequantize(self, qtensor):
        quant_tensor, meta = qtensor
        tensor = self.quantizer.dequantize(quant_tensor, meta)
        return tensor
