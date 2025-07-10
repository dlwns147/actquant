import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.cache_utils import DynamicCache, CacheConfig

from quant.new_pack import triton_quantize_and_pack_along_last_dim

# TODO: layer마다 bit, group_size 다르게 하는 코드도 추가

@dataclass
class KIVICacheConfig(CacheConfig):
    cache_implementation = "kivi"

    def __init__(
        self,
        nbits: Optional[int] = 4,
        nbits_key: Optional[int] = 0,
        nbits_value: Optional[int] = 0,
        q_group_size: Optional[int] = 128,
        residual_length: Optional[int] = 128,
        per_layer_quant: Optional[bool] = False,
        per_layer_config: Optional[Dict[str, Any]] = None,
        per_layer_config_path: Optional[str] = None,
    ):
        self.nbits = nbits
        self.nbits_key = nbits_key if nbits_key else nbits
        self.nbits_value = nbits_value if nbits_value else nbits
        self.q_group_size = q_group_size
        self.residual_length = residual_length

        self.per_layer_quant = per_layer_quant
        if per_layer_quant:
            if per_layer_config is not None:
                self.per_layer_config = per_layer_config
            elif per_layer_config_path is not None:
                import yaml
                with open(per_layer_config_path, 'r') as f:
                    self.per_layer_config = yaml.safe_load(f)
            else:
                raise ValueError("per_layer_quant is set to True but per_layer_config or per_layer_config_path is not provided.")


class KIVIDynamicCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self, cache_config: KIVICacheConfig) -> None:
        super().__init__()

        self.is_prefill = True

        self.key_states_quant_trans_cache: List[torch.Tensor] = []
        self.key_states_full_cache: List[torch.Tensor] = []
        self.key_scale_trans_cache: List[torch.Tensor] = []
        self.key_mn_trans_cache: List[torch.Tensor] = []
        self.value_states_quant_cache: List[torch.Tensor] = []
        self.value_states_full_cache: List[torch.Tensor] = []
        self.value_scale_cache: List[torch.Tensor] = []
        self.value_mn_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.nbits_key = cache_config.nbits_key
        self.nbits_value = cache_config.nbits_value
        self.q_group_size = cache_config.q_group_size
        self.residual_length = cache_config.residual_length
        self.per_layer_quant = cache_config.per_layer_quant
        if self.per_layer_quant:
            self.per_layer_config = cache_config.per_layer_config

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_states_quant_trans_cache[layer_idx], self.key_states_full_cache[layer_idx], self.key_scale_trans_cache[layer_idx], self.key_mn_trans_cache[layer_idx], 
                self.value_states_quant_cache[layer_idx], self.value_states_full_cache[layer_idx], self.value_scale_cache[layer_idx], self.value_mn_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_states_quant_trans_cache[layer_idx], self.key_states_full_cache[layer_idx], self.key_scale_trans_cache[layer_idx], self.key_mn_trans_cache[layer_idx],
                self.value_states_quant_cache[layer_idx], self.value_states_full_cache[layer_idx], self.value_scale_cache[layer_idx], self.value_mn_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_states_quant_trans_cache)

    def update(
        self,
        # key_states_quant_trans: torch.Tensor,
        # key_states_full: torch.Tensor,
        # key_scale_trans: torch.Tensor,
        # key_mn_trans: torch.Tensor,
        # value_states_quant: torch.Tensor,
        # value_states_full: torch.Tensor,
        # value_scale: torch.Tensor,
        # value_mn: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self.is_prefill = (len(self) <= layer_idx)
            self._seen_tokens += key_states.shape[-2] if self.is_prefill else 1

        # prefill
        if self.is_prefill:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self), layer_idx):
                self.key_states_quant_trans_cache.append([])
                self.key_states_full_cache.append([])
                self.key_scale_trans_cache.append([])
                self.key_mn_trans_cache.append([])
                self.value_states_quant_cache.append([])
                self.value_states_full_cache.append([])
                self.value_scale_cache.append([])
                self.value_mn_cache.append([])
            
            self._update_prefill(key_states, value_states, layer_idx)
        # generation
        else:
            self._update_generation(key_states, value_states, layer_idx)

        # return key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.value_states_quant_cache) + len(self.value_states_full_cache) == 0  # no cache in any layer
            or len(self.value_states_quant_cache) + len(self.value_states_full_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or ((0 if self.value_states_quant_cache[layer_idx] is None else len(self.value_states_quant_cache[layer_idx])) + (0 if self.value_states_full_cache[layer_idx] is None else len(self.value_states_full_cache[layer_idx])) == 0)  # the layer has no cache
        )
        len_value_quant = 0 if is_empty_layer or self.value_states_quant_cache[layer_idx] is None else self.value_states_quant_cache[layer_idx].shape[-2]
        len_value_full = 0 if is_empty_layer or self.value_states_full_cache[layer_idx] is None else self.value_states_full_cache[layer_idx].shape[-2]
        layer_seq_length = len_value_quant + len_value_full if not is_empty_layer else 0
        return layer_seq_length

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        raise NotImplementedError
        # legacy_cache = ()
        # for layer_idx in range(len(self)):
        #     legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        # return legacy_cache

    @classmethod
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, num_hidden_layers: int = None
    ) -> "KIVIDynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        raise NotImplementedError
        # cache = cls()
        # if past_key_values is not None:
        #     for layer_idx in range(len(past_key_values)):
        #         key_states, value_states = past_key_values[layer_idx]
        #         cache.update(key_states, value_states, layer_idx)
        # return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        raise NotImplementedError
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: int = None
    ) -> List["KIVIDynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        raise NotImplementedError
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = KIVIDynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.key_scale_cache = [tensor[i : i + split_size] for tensor in self.key_scale_cache]
            current_split.key_mn_cache = [tensor[i : i + split_size] for tensor in self.key_mn_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            current_split.value_scale_cache = [tensor[i : i + split_size] for tensor in self.value_scale_cache]
            current_split.value_mn_cache = [tensor[i : i + split_size] for tensor in self.value_mn_cache]
            out.append(current_split)
        return out

    @classmethod
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def from_batch_splits(cls, splits: List["DynamicCache"], num_hidden_layers: int = None) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        raise NotImplementedError
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
            value_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        raise NotImplementedError
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        raise NotImplementedError
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]

    # TODO: full_cache랑 full이랑 지금 좀 어긋남. 수정이 필요함.
    # TODO: if 안에 안 들어갔을 때 처리해야 함.
    def _update_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        if key_states.shape[-2] % self.residual_length != 0:
            if key_states.shape[-2] < self.residual_length:
                key_states_quant = None
                key_states_full = key_states
            else:
                key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
        else:
            key_states_quant = key_states
            key_states_full = None

        # quantize key states
        if key_states_quant is not None:
            key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(
                key_states_quant.transpose(2, 3).contiguous(), self.q_group_size, self.nbits_key if not self.per_layer_quant else self.per_layer_config[layer_idx]["nbits_key"]
                )
        else:
            key_states_quant_trans = None
            key_scale_trans = None
            key_mn_trans = None

        # quantize value states
        if value_states.shape[-2] <= self.residual_length:
            value_states_quant = None
            value_states_full = value_states
            value_scale = None
            value_mn = None
        else:
            value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
            value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
            value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                            self.q_group_size, 
                                                                                            self.nbits_value if not self.per_layer_quant else self.per_layer_config[layer_idx]["nbits_value"])

        self.key_states_quant_trans_cache.append(key_states_quant_trans)
        self.key_states_full_cache.append(key_states_full)
        self.key_scale_trans_cache.append(key_scale_trans)
        self.key_mn_trans_cache.append(key_mn_trans)
        self.value_states_quant_cache.append(value_states_quant)
        self.value_states_full_cache.append(value_states_full)
        self.value_scale_cache.append(value_scale)
        self.value_mn_cache.append(value_mn)


    def _update_generation(self, key_states_full: torch.Tensor, value_states_full: torch.Tensor, layer_idx: int):
        # expected shape: (B, nh, M, K)

        if key_states_full.shape[-2] == self.residual_length:
            assert self.residual_length % self.q_group_size == 0
            key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                        self.q_group_size, 
                                                                                                                        self.nbits_key if not self.per_layer_quant else self.per_layer_config[layer_idx]["nbits_key"])
            key_states_full = None
            if self.key_states_quant_trans_cache[layer_idx] is not None:
                self.key_states_quant_trans_cache[layer_idx] = torch.cat([self.key_states_quant_trans_cache[layer_idx], key_states_quant_trans_new], dim=3)
                self.key_scale_trans_cache[layer_idx] = torch.cat([self.key_scale_trans_cache[layer_idx], key_scale_trans_new], dim=3)
                self.key_mn_trans_cache[layer_idx] = torch.cat([self.key_mn_trans_cache[layer_idx], key_mn_trans_new], dim=3)
            else:
                self.key_states_quant_trans_cache[layer_idx] = key_states_quant_trans_new
                self.key_scale_trans_cache[layer_idx] = key_scale_trans_new
                self.key_mn_trans_cache[layer_idx] = key_mn_trans_new
        
        value_full_length = value_states_full.shape[-2]
        if value_full_length > self.residual_length:
            assert value_full_length == self.residual_length + 1
            value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                            self.q_group_size, 
                                                                                            self.nbits_value if not self.per_layer_quant else self.per_layer_config[layer_idx]["nbits_value"])
            value_states_full = value_states_full[:, :, 1:, :].contiguous()
            if self.value_states_quant_cache[layer_idx] is not None:
                self.value_states_quant_cache[layer_idx] = torch.cat([self.value_states_quant_cache[layer_idx], value_states_quant_new], dim=2)
                self.value_scale_cache[layer_idx] = torch.cat([self.value_scale_cache[layer_idx], scale], dim=2)
                self.value_mn_cache[layer_idx] = torch.cat([self.value_mn_cache[layer_idx], mn], dim=2)
            else:
                self.value_states_quant_cache[layer_idx] = value_states_quant_new
                self.value_scale_cache[layer_idx] = scale
                self.value_mn_cache[layer_idx] = mn

        self.key_states_full_cache[layer_idx] = key_states_full
        self.value_states_full_cache[layer_idx] = value_states_full