import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.cache_utils import DynamicCache

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

    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self.key_states_quant_trans_cache: List[torch.Tensor] = []
        self.key_states_full_cache: List[torch.Tensor] = []
        self.key_scale_trans_cache: List[torch.Tensor] = []
        self.key_mn_trans_cache: List[torch.Tensor] = []
        self.value_states_quant_cache: List[torch.Tensor] = []
        self.value_states_full_cache: List[torch.Tensor] = []
        self.value_scale_cache: List[torch.Tensor] = []
        self.value_mn_cache: List[torch.Tensor] = []

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
        key_states_quant_trans: torch.Tensor,
        key_states_full: torch.Tensor,
        key_scale_trans: torch.Tensor,
        key_mn_trans: torch.Tensor,
        value_states_quant: torch.Tensor,
        value_states_full: torch.Tensor,
        value_scale: torch.Tensor,
        value_mn: torch.Tensor,
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
            self._seen_tokens += (0 if value_states_quant is None else value_states_quant.shape[-2] + 0 if value_states_full is None else value_states_full.shape[-2])

        # Update the cache
        if len(self.key_states_quant_trans_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_states_quant_trans_cache), layer_idx):
                self.key_states_quant_trans_cache.append([])
                self.key_states_full_cache.append([])
                self.key_scale_trans_cache.append([])
                self.key_mn_trans_cache.append([])
                self.value_states_quant_cache.append([])
                self.value_states_full_cache.append([])
                self.value_scale_cache.append([])
                self.value_mn_cache.append([])

            self.key_states_quant_trans_cache.append(key_states_quant_trans)
            self.key_states_full_cache.append(key_states_full)
            self.key_scale_trans_cache.append(key_scale_trans)
            self.key_mn_trans_cache.append(key_mn_trans)
            self.value_states_quant_cache.append(value_states_quant)
            self.value_states_full_cache.append(value_states_full)
            self.value_scale_cache.append(value_scale)
            self.value_mn_cache.append(value_mn)
        else:           
            self.key_states_quant_trans_cache[layer_idx] = key_states_quant_trans
            self.key_states_full_cache[layer_idx] = key_states_full
            self.key_scale_trans_cache[layer_idx] = key_scale_trans
            self.key_mn_trans_cache[layer_idx] = key_mn_trans
            self.value_states_quant_cache[layer_idx] = value_states_quant
            self.value_states_full_cache[layer_idx] = value_states_full
            self.value_scale_cache[layer_idx] = value_scale
            self.value_mn_cache[layer_idx] = value_mn

        # elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
        #     self.key_states_quant_trans_cache[layer_idx] = key_states_quant_trans
        #     self.key_states_full_cache[layer_idx] = key_states_full
        #     self.key_scale_trans_cache[layer_idx] = key_scale_trans
        #     self.key_mn_trans_cache[layer_idx] = key_mn_trans
        #     self.value_states_quant_cache[layer_idx] = value_states_quant
        #     self.value_states_full_cache[layer_idx] = value_states_full
        #     self.value_scale_cache[layer_idx] = value_scale
        #     self.value_mn_cache[layer_idx] = value_mn

        # else:
        #     self.key_states_quant_trans_cache[layer_idx] = torch.cat([self.key_states_quant_trans_cache[layer_idx], key_states_quant_trans], dim=-1)
        #     self.key_states_full_cache[layer_idx] =  torch.cat([self.key_states_full_cache[layer_idx], key_states_full], dim=-2)
        #     self.key_scale_trans_cache[layer_idx] = torch.cat([self.key_scale_trans_cache[layer_idx], key_scale_trans], dim=-1)
        #     self.key_mn_trans_cache[layer_idx] = torch.cat([self.key_mn_trans_cache[layer_idx], key_mn_trans], dim=-1)
        #     self.value_states_quant_cache[layer_idx] = torch.cat([self.value_states_quant_cache[layer_idx], value_states_quant], dim=-2)
        #     self.value_states_full_cache[layer_idx] = torch.cat([self.value_states_full_cache[layer_idx], value_states_full], dim=-2)
        #     self.value_scale_cache[layer_idx] = torch.cat([self.value_scale_cache[layer_idx], value_scale], dim=-2)
        #     self.value_mn_cache[layer_idx] = torch.cat([self.value_mn_cache[layer_idx], value_mn], dim=-2)

        return self.key_states_quant_trans_cache[layer_idx], self.key_states_full_cache[layer_idx], self.key_scale_trans_cache[layer_idx], self.key_mn_trans_cache[layer_idx], \
                self.value_states_quant_cache[layer_idx], self.value_states_full_cache[layer_idx], self.value_scale_cache[layer_idx], self.value_mn_cache[layer_idx]
        # return self.key_cache[layer_idx], self.value_cache[layer_idx]

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

