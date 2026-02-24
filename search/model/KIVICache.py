import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.cache_utils import DynamicCache, CacheConfig
from transformers.configuration_utils import PretrainedConfig

from quant.kivi_utils.new_pack import triton_quantize_and_pack_along_last_dim, fake_quant, unpack_and_dequant_kcache, unpack_and_dequant_vcache

try:
    from .think_utils import (
        compute_k_channel_scores_efficient,
        compute_v_channel_scores,
        get_top_k_indices,
        prune_kv_by_indices,
        DEFAULT_S_OBS,
    )
    _THINK_AVAILABLE = True
except ImportError:
    _THINK_AVAILABLE = False
    DEFAULT_S_OBS = 256

@dataclass
class KIVICacheConfig(CacheConfig):
    """
    KIVI cache config
    We didn't provide single config(per model config) for now.

    Args:
        # per layer config
        k_bits: k_bits for key
        v_bits: v_bits for value
        k_group_size: k_group_size for key
        v_group_size: v_group_size for value

        # quantization config
        k_quant_per: k_quant_per for key
        v_quant_per: v_quant_per for value
        residual_length: residual_length for key and value
    """

    cache_implementation = "kivi"

    def __init__(
        self,

        # per layer config
        k_bits : Optional[List[int]] = [],
        v_bits : Optional[List[int]] = [],
        k_group_size: Optional[List[int]] = [],
        v_group_size: Optional[List[int]] = [],

        # pruning config (per layer retention ratio for ThinK / Think_kivi)
        k_prune: Optional[List[float]] = None,
        v_prune: Optional[List[float]] = None,

        # quantization config
        k_quant_scheme: Optional[str] = 'channel',
        v_quant_scheme: Optional[str] = 'token',
        residual_length: Optional[int] = 128,

        packing: Optional[bool] = False,
    ):
        assert len(k_bits) > 0 and len(v_bits) > 0, "k_bits, v_bits must be provided"

        # per layer config
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.k_group_size = k_group_size
        self.v_group_size = v_group_size

        # quantization config
        self.k_quant_scheme = k_quant_scheme
        self.v_quant_scheme = v_quant_scheme
        self.residual_length = residual_length

        # pruning config
        # By default, no pruning (retention ratio 1.0 for all layers)
        if k_prune is None:
            self.k_prune = [1.0] * len(self.k_bits)
        else:
            assert len(k_prune) == len(self.k_bits), "k_prune length must match k_bits"
            self.k_prune = k_prune

        if v_prune is None:
            self.v_prune = [1.0] * len(self.v_bits)
        else:
            assert len(v_prune) == len(self.v_bits), "v_prune length must match v_bits"
            self.v_prune = v_prune

        # packing config
        self.packing = packing


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

    def __init__(self, cache_config: "KIVICacheConfig") -> None:
        super().__init__()

        self.is_prefill = True
        self.kivi_config = cache_config

        self.key_states_quant_trans_cache: List[torch.Tensor] = []
        self.key_states_full_cache: List[torch.Tensor] = []
        self.key_scale_trans_cache: List[torch.Tensor] = []
        self.key_mn_trans_cache: List[torch.Tensor] = []
        self.value_states_quant_cache: List[torch.Tensor] = []
        self.value_states_full_cache: List[torch.Tensor] = []
        self.value_scale_cache: List[torch.Tensor] = []
        self.value_mn_cache: List[torch.Tensor] = []

        # for sageattention
        self.km: List[torch.Tensor] = []

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
            self._seen_tokens += key_states.shape[-2]

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

                # for sageattention
                self.km.append([])

            # for sageattention
            self.km.append(key_states.mean(dim=2).unsqueeze(2))
            key_states -= self.km[layer_idx]
            # key_states.sub_(self.km[layer_idx])
            
            self._update_prefill(key_states, value_states, layer_idx)
        # generation
        else:
            # for sageattention
            key_states -= self.km[layer_idx]
            # key_states.sub_(self.km[layer_idx])

            self._update_generation(key_states, value_states, layer_idx)

        return key_states, value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_states_quant_trans_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

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

    def _update_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        if key_states.shape[-2] % self.kivi_config.residual_length != 0:
            if key_states.shape[-2] < self.kivi_config.residual_length:
                key_states_quant = None
                key_states_full = key_states
            else:
                key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.kivi_config.residual_length), :].contiguous()
                key_states_full = key_states[:, :, -(key_states.shape[-2] % self.kivi_config.residual_length):, :].contiguous()
        else:
            key_states_quant = key_states
            key_states_full = None

        # quantize key states
        if key_states_quant is not None:
            key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(
                key_states_quant.transpose(2, 3).contiguous(), self.kivi_config.k_group_size[layer_idx], self.kivi_config.k_bits[layer_idx]
                )
        else:
            key_states_quant_trans = None
            key_scale_trans = None
            key_mn_trans = None

        # quantize value states
        if value_states.shape[-2] <= self.kivi_config.residual_length:
            value_states_quant = None
            value_states_full = value_states
            value_scale = None
            value_mn = None
        else:
            value_states_quant = value_states[:, :, :-self.kivi_config.residual_length, :].contiguous()
            value_states_full = value_states[:, :, -self.kivi_config.residual_length:, :].contiguous()
            value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                            self.kivi_config.v_group_size[layer_idx], 
                                                                                            self.kivi_config.v_bits[layer_idx])

        self.key_states_quant_trans_cache.append(key_states_quant_trans)
        self.key_states_full_cache.append(key_states_full)
        self.key_scale_trans_cache.append(key_scale_trans)
        self.key_mn_trans_cache.append(key_mn_trans)
        self.value_states_quant_cache.append(value_states_quant)
        self.value_states_full_cache.append(value_states_full)
        self.value_scale_cache.append(value_scale)
        self.value_mn_cache.append(value_mn)

    def _update_generation(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        if self.key_states_full_cache[layer_idx] is not None:
            self.key_states_full_cache[layer_idx] = torch.cat([self.key_states_full_cache[layer_idx], key_states], dim=2)
        else:
            self.key_states_full_cache[layer_idx] = key_states
        
        self.value_states_full_cache[layer_idx] = torch.cat([self.value_states_full_cache[layer_idx], value_states], dim=2)

    def lazy_update(self, layer_idx: int):
        # expected shape: (B, nh, M, K)

        if self.key_states_full_cache[layer_idx].shape[-2] == self.kivi_config.residual_length:
            assert self.kivi_config.residual_length % self.kivi_config.k_group_size[layer_idx] == 0
            key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(self.key_states_full_cache[layer_idx].transpose(2, 3).contiguous(), 
                                                                                                                        self.kivi_config.k_group_size[layer_idx], 
                                                                                                                        self.kivi_config.k_bits[layer_idx])
            self.key_states_full_cache[layer_idx] = None
            if self.key_states_quant_trans_cache[layer_idx] is not None:
                self.key_states_quant_trans_cache[layer_idx] = torch.cat([self.key_states_quant_trans_cache[layer_idx], key_states_quant_trans_new], dim=3)
                self.key_scale_trans_cache[layer_idx] = torch.cat([self.key_scale_trans_cache[layer_idx], key_scale_trans_new], dim=3)
                self.key_mn_trans_cache[layer_idx] = torch.cat([self.key_mn_trans_cache[layer_idx], key_mn_trans_new], dim=3)
            else:
                self.key_states_quant_trans_cache[layer_idx] = key_states_quant_trans_new
                self.key_scale_trans_cache[layer_idx] = key_scale_trans_new
                self.key_mn_trans_cache[layer_idx] = key_mn_trans_new
        
        value_full_length = self.value_states_full_cache[layer_idx].shape[-2]
        if value_full_length > self.kivi_config.residual_length:
            assert value_full_length == self.kivi_config.residual_length + 1
            value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(self.value_states_full_cache[layer_idx][:, :, :1, :].contiguous(), 
                                                                                            self.kivi_config.v_group_size[layer_idx], 
                                                                                            self.kivi_config.v_bits[layer_idx])
            self.value_states_full_cache[layer_idx] = self.value_states_full_cache[layer_idx][:, :, 1:, :].contiguous()
            if self.value_states_quant_cache[layer_idx] is not None:
                self.value_states_quant_cache[layer_idx] = torch.cat([self.value_states_quant_cache[layer_idx], value_states_quant_new], dim=2)
                self.value_scale_cache[layer_idx] = torch.cat([self.value_scale_cache[layer_idx], scale], dim=2)
                self.value_mn_cache[layer_idx] = torch.cat([self.value_mn_cache[layer_idx], mn], dim=2)
            else:
                self.value_states_quant_cache[layer_idx] = value_states_quant_new
                self.value_scale_cache[layer_idx] = scale
                self.value_mn_cache[layer_idx] = mn


class ThinkKIVIDynamicCache(KIVIDynamicCache):
    """
    KIVI + ThinK cache: applies channel pruning (ThinK) before quantization (KIVI).
    Supports both K and V pruning per Appendix D of ThinK.
    """

    def __init__(self, cache_config: "KIVICacheConfig", s_obs: int = None) -> None:
        super().__init__(cache_config)
        self.s_obs = s_obs if s_obs is not None else DEFAULT_S_OBS
        self.k_keep_indices: List[Optional[torch.Tensor]] = []
        self.v_keep_indices: List[Optional[torch.Tensor]] = []

    def _apply_pruning_prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute channel scores and prune K, V. Returns pruned key_states, value_states, k_indices, v_indices."""
        if not _THINK_AVAILABLE:
            return key_states, value_states, None, None

        k_ratio = self.kivi_config.k_prune[layer_idx]
        v_ratio = self.kivi_config.v_prune[layer_idx]
        if k_ratio >= 1.0 and v_ratio >= 1.0:
            return key_states, value_states, None, None

        head_dim = key_states.shape[-1]
        k_indices = None
        v_indices = None

        if k_ratio < 1.0:
            k_scores = compute_k_channel_scores_efficient(
                query_states, key_states, self.s_obs
            )
            k_indices = get_top_k_indices(k_scores, k_ratio)
            if k_indices.shape[-1] < head_dim:
                gs = self.kivi_config.k_group_size[layer_idx]
                t_k = k_indices.shape[-1]
                eff_gs = min(gs, t_k) if gs > 0 else t_k
                if eff_gs > 0:
                    t_k_round = (t_k // eff_gs) * eff_gs
                    if t_k_round > 0 and t_k_round < t_k:
                        k_indices = k_indices[..., :t_k_round]

        if v_ratio < 1.0:
            v_scores = compute_v_channel_scores(
                query_states, key_states, value_states,
                self.s_obs, head_dim
            )
            v_indices = get_top_k_indices(v_scores, v_ratio)
            if v_indices.shape[-1] < head_dim:
                gs = self.kivi_config.v_group_size[layer_idx]
                t_v = v_indices.shape[-1]
                eff_gs = min(gs, t_v) if gs > 0 else t_v
                if eff_gs > 0:
                    t_v_round = (t_v // eff_gs) * eff_gs
                    if t_v_round > 0 and t_v_round < t_v:
                        v_indices = v_indices[..., :t_v_round]

        k_indices_stored = None
        v_indices_stored = None
        if k_indices is not None or v_indices is not None:
            if k_indices is None:
                k_indices = torch.arange(head_dim, device=key_states.device).unsqueeze(0).expand(
                    key_states.shape[1], -1
                )
            if v_indices is None:
                v_indices = torch.arange(head_dim, device=value_states.device).unsqueeze(0).expand(
                    value_states.shape[1], -1
                )
            key_states, value_states = prune_kv_by_indices(
                key_states, value_states, k_indices, v_indices
            )
            k_indices_stored = k_indices[0].cpu().clone() if k_indices.shape[0] > 0 else None
            v_indices_stored = v_indices[0].cpu().clone() if v_indices.shape[0] > 0 else None

        return key_states, value_states, k_indices_stored, v_indices_stored

    def _prune_full_for_lazy_update(
        self, full_tensor: torch.Tensor, indices: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Prune full cache by channel indices (for lazy_update)."""
        if indices is None or not _THINK_AVAILABLE:
            return full_tensor
        device = full_tensor.device
        indices = indices.to(device)
        if indices.dim() == 2:
            indices = indices.unsqueeze(0).expand(full_tensor.shape[0], -1, -1)
        return torch.gather(
            full_tensor, dim=3,
            index=indices.unsqueeze(2).expand(-1, -1, full_tensor.shape[2], -1),
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_kwargs = cache_kwargs or {}
        query_states = cache_kwargs.get("query_states")

        if self.is_prefill and query_states is not None and _THINK_AVAILABLE:
            while len(self.k_keep_indices) <= layer_idx:
                self.k_keep_indices.append(None)
                self.v_keep_indices.append(None)

            k_ratio = self.kivi_config.k_prune[layer_idx]
            v_ratio = self.kivi_config.v_prune[layer_idx]
            if k_ratio < 1.0 or v_ratio < 1.0:
                key_states, value_states, k_idx, v_idx = self._apply_pruning_prefill(
                    query_states, key_states, value_states, layer_idx
                )
                self.k_keep_indices[layer_idx] = k_idx
                self.v_keep_indices[layer_idx] = v_idx

        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def _update_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        # key_states, value_states may already be pruned by update() when query was passed
        super()._update_prefill(key_states, value_states, layer_idx)

    def _update_generation(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        super()._update_generation(key_states, value_states, layer_idx)

    def lazy_update(self, layer_idx: int):
        if layer_idx >= len(self.k_keep_indices):
            self.k_keep_indices.append(None)
            self.v_keep_indices.append(None)

        k_idx = self.k_keep_indices[layer_idx] if layer_idx < len(self.k_keep_indices) else None
        v_idx = self.v_keep_indices[layer_idx] if layer_idx < len(self.v_keep_indices) else None

        full_k = self.key_states_full_cache[layer_idx]
        full_v = self.value_states_full_cache[layer_idx]
        if full_k is not None and full_k.shape[-2] == self.kivi_config.residual_length:
            if k_idx is not None and _THINK_AVAILABLE:
                full_k = self._prune_full_for_lazy_update(full_k, k_idx, layer_idx)
            gs = self.kivi_config.k_group_size[layer_idx]
            t_k = full_k.shape[-1]
            eff_gs = min(gs, t_k) if gs > 0 else t_k
            if eff_gs > 0 and t_k % eff_gs != 0:
                t_k = (t_k // eff_gs) * eff_gs
                full_k = full_k[..., :t_k]
            eff_gs = min(gs, full_k.shape[-1]) if gs > 0 else full_k.shape[-1]
            key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(
                full_k.transpose(2, 3).contiguous(), eff_gs, self.kivi_config.k_bits[layer_idx]
            )
            self.key_states_full_cache[layer_idx] = None
            if self.key_states_quant_trans_cache[layer_idx] is not None:
                self.key_states_quant_trans_cache[layer_idx] = torch.cat(
                    [self.key_states_quant_trans_cache[layer_idx], key_states_quant_trans_new], dim=3
                )
                self.key_scale_trans_cache[layer_idx] = torch.cat(
                    [self.key_scale_trans_cache[layer_idx], key_scale_trans_new], dim=3
                )
                self.key_mn_trans_cache[layer_idx] = torch.cat(
                    [self.key_mn_trans_cache[layer_idx], key_mn_trans_new], dim=3
                )
            else:
                self.key_states_quant_trans_cache[layer_idx] = key_states_quant_trans_new
                self.key_scale_trans_cache[layer_idx] = key_scale_trans_new
                self.key_mn_trans_cache[layer_idx] = key_mn_trans_new

        value_full_length = self.value_states_full_cache[layer_idx].shape[-2]
        if value_full_length > self.kivi_config.residual_length:
            assert value_full_length == self.kivi_config.residual_length + 1
            chunk = self.value_states_full_cache[layer_idx][:, :, :1, :].contiguous()
            if v_idx is not None and _THINK_AVAILABLE:
                chunk = self._prune_full_for_lazy_update(chunk, v_idx, layer_idx)
            v_gs = self.kivi_config.v_group_size[layer_idx]
            t_v = chunk.shape[-1]
            v_eff_gs = min(v_gs, t_v) if v_gs > 0 else t_v
            if v_eff_gs > 0 and t_v % v_eff_gs != 0:
                t_v = (t_v // v_eff_gs) * v_eff_gs
                chunk = chunk[..., :t_v]
            v_eff_gs = min(v_gs, chunk.shape[-1]) if v_gs > 0 else chunk.shape[-1]
            value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(
                chunk, v_eff_gs, self.kivi_config.v_bits[layer_idx]
            )
            self.value_states_full_cache[layer_idx] = self.value_states_full_cache[layer_idx][:, :, 1:, :].contiguous()
            if self.value_states_quant_cache[layer_idx] is not None:
                self.value_states_quant_cache[layer_idx] = torch.cat(
                    [self.value_states_quant_cache[layer_idx], value_states_quant_new], dim=2
                )
                self.value_scale_cache[layer_idx] = torch.cat(
                    [self.value_scale_cache[layer_idx], scale], dim=2
                )
                self.value_mn_cache[layer_idx] = torch.cat(
                    [self.value_mn_cache[layer_idx], mn], dim=2
                )
            else:
                self.value_states_quant_cache[layer_idx] = value_states_quant_new
                self.value_scale_cache[layer_idx] = scale
                self.value_mn_cache[layer_idx] = mn


class KIVIFakeCache(DynamicCache):
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

    def __init__(self, cache_config: "KIVICacheConfig") -> None:
        super().__init__()

        self.is_prefill = True
        self.kivi_config = cache_config

        self.key_states_quant_trans_cache: List[torch.Tensor] = []
        self.key_states_full_cache: List[torch.Tensor] = []
        self.value_states_quant_cache: List[torch.Tensor] = []
        self.value_states_full_cache: List[torch.Tensor] = []

        # for sageattention
        self.km: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_states_quant_trans_cache[layer_idx], self.key_states_full_cache[layer_idx], self.value_states_quant_cache[layer_idx], self.value_states_full_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_states_quant_trans_cache[layer_idx], self.key_states_full_cache[layer_idx], self.value_states_quant_cache[layer_idx], self.value_states_full_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_states_quant_trans_cache)

    def update(
        self,
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
        if layer_idx == 0:
            self.is_prefill = (len(self) <= layer_idx)
            self._seen_tokens += key_states.shape[-2]

        # prefill
        if self.is_prefill:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self), layer_idx):
                self.key_states_quant_trans_cache.append([])
                self.key_states_full_cache.append([])
                self.value_states_quant_cache.append([])
                self.value_states_full_cache.append([])

            # for sageattention
            # self.km.append(key_states.mean(dim=2).unsqueeze(2))
            # key_states -= self.km[layer_idx]
            
            self._update_prefill(key_states, value_states, layer_idx)

            keys_to_return = key_states
            values_to_return = value_states
        # generation
        else:
            # for sageattention
            # key_states -= self.km[layer_idx]
            
            self._update_generation(key_states, value_states, layer_idx)

            keys_to_return = torch.cat([self.key_states_quant_trans_cache[layer_idx], self.key_states_full_cache[layer_idx]], dim=-2)
            values_to_return = torch.cat([self.value_states_quant_cache[layer_idx], self.value_states_full_cache[layer_idx]], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_states_quant_trans_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _update_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        # import pdb; pdb.set_trace()
        if key_states.shape[-2] % self.kivi_config.residual_length != 0:
            if key_states.shape[-2] < self.kivi_config.residual_length:
                key_states_quant = None
                key_states_full = key_states
            else:
                key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.kivi_config.residual_length), :].contiguous()
                key_states_full = key_states[:, :, -(key_states.shape[-2] % self.kivi_config.residual_length):, :].contiguous()
        else:
            key_states_quant = key_states
            key_states_full = None

        if key_states_quant is not None:
            key_states_quant = fake_quant(key_states_quant, self.kivi_config.k_group_size[layer_idx], self.kivi_config.k_bits[layer_idx], self.kivi_config.k_quant_scheme)

        if value_states.shape[-2] <= self.kivi_config.residual_length:
            value_states_quant = None
            value_states_full = value_states
        else:
            value_states_quant = value_states[:, :, :-self.kivi_config.residual_length, :].contiguous()
            value_states_full = value_states[:, :, -self.kivi_config.residual_length:, :].contiguous()

            value_states_quant = fake_quant(value_states_quant, self.kivi_config.v_group_size[layer_idx], self.kivi_config.v_bits[layer_idx], self.kivi_config.v_quant_scheme)

        self.key_states_quant_trans_cache.append(key_states_quant)
        self.key_states_full_cache.append(key_states_full)
        self.value_states_quant_cache.append(value_states_quant)
        self.value_states_full_cache.append(value_states_full)

    def _update_generation(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        if self.key_states_full_cache[layer_idx] is not None:
            self.key_states_full_cache[layer_idx] = torch.cat([self.key_states_full_cache[layer_idx], key_states], dim=2)
        else:
            self.key_states_full_cache[layer_idx] = key_states
        
        self.value_states_full_cache[layer_idx] = torch.cat([self.value_states_full_cache[layer_idx], value_states], dim=2)

    def lazy_update(self, layer_idx: int):
        if self.key_states_full_cache[layer_idx].shape[-2] == self.kivi_config.residual_length:
            assert self.kivi_config.residual_length % self.kivi_config.k_group_size[layer_idx] == 0
            key_states_quant_trans_new = fake_quant(self.key_states_full_cache[layer_idx], self.kivi_config.k_group_size[layer_idx], self.kivi_config.k_bits[layer_idx], self.kivi_config.k_quant_scheme)

            self.key_states_full_cache[layer_idx] = None
            if self.key_states_quant_trans_cache[layer_idx] is not None:
                self.key_states_quant_trans_cache[layer_idx] = torch.cat([self.key_states_quant_trans_cache[layer_idx], key_states_quant_trans_new], dim=2)
            else:
                self.key_states_quant_trans_cache[layer_idx] = key_states_quant_trans_new
        
        value_full_length = self.value_states_full_cache[layer_idx].shape[-2]
        if value_full_length > self.kivi_config.residual_length:
            assert value_full_length == self.kivi_config.residual_length + 1
            value_states_quant_new = fake_quant(self.value_states_full_cache[layer_idx][:, :, :1, :].contiguous(), 
                                                                                            self.kivi_config.v_group_size[layer_idx], 
                                                                                            self.kivi_config.v_bits[layer_idx],
                                                                                            self.kivi_config.v_quant_scheme)
            self.value_states_full_cache[layer_idx] = self.value_states_full_cache[layer_idx][:, :, 1:, :].contiguous()
            if self.value_states_quant_cache[layer_idx] is not None:
                self.value_states_quant_cache[layer_idx] = torch.cat([self.value_states_quant_cache[layer_idx], value_states_quant_new], dim=2)
            else:
                self.value_states_quant_cache[layer_idx] = value_states_quant_new