import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.cache_utils import DynamicCache, CacheConfig
from transformers.configuration_utils import PretrainedConfig

from quant.kivi_utils.new_pack import triton_quantize_and_pack_along_last_dim, fake_quant, unpack_and_dequant_kcache, unpack_and_dequant_vcache
import math

def _think_key_pruner_query_driven(kv_states: torch.Tensor, q_states: torch.Tensor, pruning_dim: int):
    """
    Query-driven key pruning (ThinK). Prunes along head_dim.

    kv_states:   (B, n_kv, L, D)
    q_states:    (B, n_h,  L, D)
    pruning_dim: number of head_dim channels to prune (integer)
    Returns:
      pruned_kv_states: (B, n_kv, L, D-pruning_dim)
      keep_mask: (B, n_kv, D) boolean mask of kept dims
    """
    if pruning_dim is None or pruning_dim <= 0:
        keep_mask = torch.ones(kv_states.shape[0], kv_states.shape[1], kv_states.shape[3], dtype=torch.bool, device=kv_states.device)
        return kv_states, keep_mask

    bz, n_kv, seqlen, head_dim = kv_states.shape
    n_h = q_states.shape[1]
    k = int(pruning_dim)
    if k <= 0 or k >= head_dim:
        keep_mask = torch.ones(bz, n_kv, head_dim, dtype=torch.bool, device=kv_states.device)
        return kv_states, keep_mask

    q_tail = q_states[..., -32:, :] if seqlen >= 32 else q_states
    queries_norm = torch.pow(q_tail, 2).mean(dim=2)  # (B, n_h, D)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)  # (B, n_kv, D)

    if n_h != n_kv:
        assert n_h % n_kv == 0, f"n_h={n_h}, n_kv={n_kv} not divisible"
        groups = n_h // n_kv
        queries_norm = queries_norm.view(bz, n_kv, groups, head_dim).mean(dim=2)

    score = queries_norm * keys_norm

    # ThinK: prune the *least important* k dimensions (smallest scores),
    # then keep the remaining (head_dim - k) dimensions.
    _, prune_indices = torch.topk(score, k, dim=-1, largest=False)
    prune_idx = prune_indices.sort().values
    prune_mask = torch.zeros(score.shape, dtype=torch.bool, device=kv_states.device).scatter_(-1, prune_idx, 1)

    # keep_mask: True for kept dimensions (shape: B, n_kv, D)
    keep_mask = ~prune_mask
    keep_mask_k = keep_mask.unsqueeze(2).expand(-1, -1, seqlen, -1)
    kept = kv_states[keep_mask_k].reshape(bz, n_kv, seqlen, head_dim - k)
    return kept, keep_mask

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

        # packing config
        self.packing = packing

        # ThinK options (optional)
        self.enable_think = False
        self.k_pruning_dim: List[int] = [0] * len(k_bits)
        self.v_pruning_dim: List[int] = [0] * len(v_bits)


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
        self._is_stride: bool = False
        self.kivi_config = cache_config

        self.key_states_quant_trans_cache: List[torch.Tensor] = []
        self.key_states_full_cache: List[torch.Tensor] = []
        self.key_scale_trans_cache: List[torch.Tensor] = []
        self.key_mn_trans_cache: List[torch.Tensor] = []
        self.value_states_quant_cache: List[torch.Tensor] = []
        self.value_states_full_cache: List[torch.Tensor] = []
        self.value_scale_cache: List[torch.Tensor] = []
        self.value_mn_cache: List[torch.Tensor] = []
        # ThinK: store last keep-mask per layer for generation path
        self.key_keep_mask_cache: List[torch.Tensor] = []

        # for sageattention
        self.km: List[torch.Tensor] = []

    @property
    def is_stride(self) -> bool:
        return self._is_stride

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
            self._is_stride = (not self.is_prefill) and (key_states.shape[-2] > 1)
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
                self.key_keep_mask_cache.append([])

                # for sageattention
                self.km.append([])

            # for sageattention
            self.km.append(key_states.mean(dim=2).unsqueeze(2))
            key_states -= self.km[layer_idx]
            # key_states.sub_(self.km[layer_idx])

            self._update_prefill(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)
        # stride: non-prefill multi-token chunk
        elif self._is_stride:
            key_states -= self.km[layer_idx]
            return self._update_stride(key_states, value_states, layer_idx)
        # generation: single-token decoding
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

    def _update_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None):
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
        key_keep_mask = None
        if key_states_quant is not None:
            # ThinK: prune key dimensions using query-driven mask (needs query_states from attention forward)
            if getattr(self.kivi_config, "enable_think", False):
                q_states = None if cache_kwargs is None else cache_kwargs.get("query_states", None)
                if q_states is not None:
                    pruning_dim = 0
                    try:
                        pruning_dim = int(self.kivi_config.k_pruning_dim[layer_idx])
                    except Exception:
                        pruning_dim = 0
                    key_states_quant, key_keep_mask = _think_key_pruner_query_driven(key_states_quant, q_states, pruning_dim)

            key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(
                key_states_quant.transpose(2, 3).contiguous(), self.kivi_config.k_group_size[layer_idx], self.kivi_config.k_bits[layer_idx]
                )
        else:
            key_states_quant_trans = None
            key_scale_trans = None
            key_mn_trans = None
            key_keep_mask = None

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
        self.key_keep_mask_cache.append(key_keep_mask)

    def _update_generation(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        if self.key_states_full_cache[layer_idx] is not None:
            self.key_states_full_cache[layer_idx] = torch.cat([self.key_states_full_cache[layer_idx], key_states], dim=2)
        else:
            self.key_states_full_cache[layer_idx] = key_states

        self.value_states_full_cache[layer_idx] = torch.cat([self.value_states_full_cache[layer_idx], value_states], dim=2)

    def _update_stride(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle a multi-token stride chunk (non-prefill).

        Merges the existing full-precision residual with the new stride tokens,
        re-partitions into quantized and residual according to the residual_length
        invariant, and returns the fully reconstructed K/V for standard attention.
        """
        cfg = self.kivi_config
        residual_length = cfg.residual_length

        # ── KEYS ─────────────────────────────────────────────────────────
        if self.key_states_full_cache[layer_idx] is not None:
            combined_key = torch.cat([self.key_states_full_cache[layer_idx], key_states], dim=2)
        else:
            combined_key = key_states

        key_len = combined_key.shape[2]
        rem = key_len % residual_length
        if rem == 0:
            key_to_quant = combined_key
            new_key_full = None
        elif key_len < residual_length:
            key_to_quant = None
            new_key_full = combined_key
        else:
            key_to_quant = combined_key[:, :, :-rem, :].contiguous()
            new_key_full = combined_key[:, :, -rem:, :].contiguous()

        if key_to_quant is not None:
            assert key_to_quant.shape[2] % cfg.k_group_size[layer_idx] == 0, (
                f"Stride key tokens ({key_to_quant.shape[2]}) must be divisible by "
                f"k_group_size ({cfg.k_group_size[layer_idx]}). Adjust stride accordingly."
            )
            key_to_pack = key_to_quant
            # ThinK: apply keep_mask to prune head dimensions before packing
            if getattr(cfg, 'enable_think', False):
                keep_mask = self.key_keep_mask_cache[layer_idx] if layer_idx < len(self.key_keep_mask_cache) else None
                if keep_mask is not None and isinstance(keep_mask, torch.Tensor):
                    b, n_kv, slen, d = key_to_pack.shape
                    keep_mask_k = keep_mask.unsqueeze(2).expand(-1, -1, slen, -1)
                    key_to_pack = key_to_pack[keep_mask_k].reshape(b, n_kv, slen, -1)

            kq_new, ks_new, km_new = triton_quantize_and_pack_along_last_dim(
                key_to_pack.transpose(2, 3).contiguous(),
                cfg.k_group_size[layer_idx],
                cfg.k_bits[layer_idx],
            )
            if self.key_states_quant_trans_cache[layer_idx] is not None:
                self.key_states_quant_trans_cache[layer_idx] = torch.cat(
                    [self.key_states_quant_trans_cache[layer_idx], kq_new], dim=3)
                self.key_scale_trans_cache[layer_idx] = torch.cat(
                    [self.key_scale_trans_cache[layer_idx], ks_new], dim=3)
                self.key_mn_trans_cache[layer_idx] = torch.cat(
                    [self.key_mn_trans_cache[layer_idx], km_new], dim=3)
            else:
                self.key_states_quant_trans_cache[layer_idx] = kq_new
                self.key_scale_trans_cache[layer_idx] = ks_new
                self.key_mn_trans_cache[layer_idx] = km_new

        self.key_states_full_cache[layer_idx] = new_key_full

        # ── VALUES ───────────────────────────────────────────────────────
        combined_val = torch.cat([self.value_states_full_cache[layer_idx], value_states], dim=2)
        val_len = combined_val.shape[2]
        if val_len <= residual_length:
            val_to_quant = None
            new_val_full = combined_val
        else:
            val_to_quant = combined_val[:, :, :-residual_length, :].contiguous()
            new_val_full = combined_val[:, :, -residual_length:, :].contiguous()

        if val_to_quant is not None:
            assert val_to_quant.shape[2] % cfg.v_group_size[layer_idx] == 0, (
                f"Stride value tokens ({val_to_quant.shape[2]}) must be divisible by "
                f"v_group_size ({cfg.v_group_size[layer_idx]}). Adjust stride accordingly."
            )
            vq_new, vs_new, vm_new = triton_quantize_and_pack_along_last_dim(
                val_to_quant,
                cfg.v_group_size[layer_idx],
                cfg.v_bits[layer_idx],
            )
            if self.value_states_quant_cache[layer_idx] is not None:
                self.value_states_quant_cache[layer_idx] = torch.cat(
                    [self.value_states_quant_cache[layer_idx], vq_new], dim=2)
                self.value_scale_cache[layer_idx] = torch.cat(
                    [self.value_scale_cache[layer_idx], vs_new], dim=2)
                self.value_mn_cache[layer_idx] = torch.cat(
                    [self.value_mn_cache[layer_idx], vm_new], dim=2)
            else:
                self.value_states_quant_cache[layer_idx] = vq_new
                self.value_scale_cache[layer_idx] = vs_new
                self.value_mn_cache[layer_idx] = vm_new

        self.value_states_full_cache[layer_idx] = new_val_full

        # ── Reconstruct full K/V for standard attention ───────────────────
        # Keys are stored as (B, n_kv, D, T_q//feat_per_int) with scale/mn (B, n_kv, D, num_groups).
        # This layout is identical to value layout, so unpack_and_dequant_vcache (pack_dim=3)
        # dequantizes it to (B, n_kv, D, T_q); we then transpose to (B, n_kv, T_q, D).
        if self.key_states_quant_trans_cache[layer_idx] is not None:
            kq = self.key_states_quant_trans_cache[layer_idx]
            ks = self.key_scale_trans_cache[layer_idx]
            km_s = self.key_mn_trans_cache[layer_idx]
            key_dequant_DT = unpack_and_dequant_vcache(
                kq, ks, km_s, cfg.k_group_size[layer_idx], cfg.k_bits[layer_idx]
            )  # (B, n_kv, D, T_q)
            key_dequant = key_dequant_DT.transpose(2, 3).contiguous()  # (B, n_kv, T_q, D)

            # ThinK: zero-fill pruned dimensions back to full head_dim
            if getattr(cfg, 'enable_think', False):
                keep_mask = self.key_keep_mask_cache[layer_idx] if layer_idx < len(self.key_keep_mask_cache) else None
                if keep_mask is not None and isinstance(keep_mask, torch.Tensor):
                    b_, n_kv_, T_q, D_kept = key_dequant.shape
                    D_full = keep_mask.shape[2]
                    key_full_dim = torch.zeros(
                        b_, n_kv_, T_q, D_full,
                        device=key_dequant.device, dtype=key_dequant.dtype,
                    )
                    km_expand = keep_mask.unsqueeze(2).expand(-1, -1, T_q, -1)
                    key_full_dim[km_expand] = key_dequant.reshape(-1)
                    key_dequant = key_full_dim

            key_for_attn = torch.cat([key_dequant, new_key_full], dim=2) if new_key_full is not None else key_dequant
        else:
            key_for_attn = new_key_full  # only residual, no quantized part yet

        if self.value_states_quant_cache[layer_idx] is not None:
            val_dequant = unpack_and_dequant_vcache(
                self.value_states_quant_cache[layer_idx],
                self.value_scale_cache[layer_idx],
                self.value_mn_cache[layer_idx],
                cfg.v_group_size[layer_idx],
                cfg.v_bits[layer_idx],
            )  # (B, n_kv, T_v, D)
            val_for_attn = torch.cat([val_dequant, new_val_full], dim=2)
        else:
            val_for_attn = new_val_full  # only residual, no quantized part yet

        return key_for_attn, val_for_attn

    def lazy_update(self, layer_idx: int):
        # expected shape: (B, nh, M, K)
        # Stride phase invariants are maintained by _update_stride; skip here.
        if self._is_stride:
            return

        if self.key_states_full_cache[layer_idx].shape[-2] == self.kivi_config.residual_length:
            assert self.kivi_config.residual_length % self.kivi_config.k_group_size[layer_idx] == 0

            key_to_pack = self.key_states_full_cache[layer_idx]
            # ThinK: if we pruned key dimensions during prefill, apply the same keep-mask
            # before packing the residual window, so packed K dims stay consistent.
            if getattr(self.kivi_config, "enable_think", False):
                keep_mask = None
                try:
                    keep_mask = self.key_keep_mask_cache[layer_idx]
                except Exception:
                    keep_mask = None
                if keep_mask is not None and isinstance(keep_mask, torch.Tensor):
                    b, n_kv, seqlen, d = key_to_pack.shape
                    keep_mask_k = keep_mask.unsqueeze(2).expand(-1, -1, seqlen, -1)
                    key_to_pack = key_to_pack[keep_mask_k].reshape(b, n_kv, seqlen, -1)

            key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(
                key_to_pack.transpose(2, 3).contiguous(),
                self.kivi_config.k_group_size[layer_idx],
                self.kivi_config.k_bits[layer_idx],
            )
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
        self._is_stride: bool = False
        self.kivi_config = cache_config

        self.key_states_quant_trans_cache: List[torch.Tensor] = []
        self.key_states_full_cache: List[torch.Tensor] = []
        self.value_states_quant_cache: List[torch.Tensor] = []
        self.value_states_full_cache: List[torch.Tensor] = []
        # ThinK: keep_mask per layer for lazy_update (apply same channel mask to residual window)
        self.key_keep_mask_cache: List[Optional[torch.Tensor]] = []

        # for sageattention
        self.km: List[torch.Tensor] = []

    @property
    def is_stride(self) -> bool:
        return self._is_stride

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
            self._is_stride = (not self.is_prefill) and (key_states.shape[-2] > 1)
            self._seen_tokens += key_states.shape[-2]

        # prefill
        if self.is_prefill:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self), layer_idx):
                self.key_states_quant_trans_cache.append([])
                self.key_states_full_cache.append([])
                self.value_states_quant_cache.append([])
                self.value_states_full_cache.append([])
                self.key_keep_mask_cache.append(None)

            # for sageattention
            # self.km.append(key_states.mean(dim=2).unsqueeze(2))
            # key_states -= self.km[layer_idx]

            self._update_prefill(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)

            keys_to_return = key_states
            values_to_return = value_states
        # stride: non-prefill multi-token chunk
        elif self._is_stride:
            return self._update_stride(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)
        # generation: single-token decoding
        else:
            # for sageattention
            # key_states -= self.km[layer_idx]

            self._update_generation(key_states, value_states, layer_idx)

            # Guard against None quant cache (prefill length was < residual_length)
            if self.key_states_quant_trans_cache[layer_idx] is not None:
                keys_to_return = torch.cat(
                    [self.key_states_quant_trans_cache[layer_idx], self.key_states_full_cache[layer_idx]], dim=-2)
            else:
                keys_to_return = self.key_states_full_cache[layer_idx]
            if self.value_states_quant_cache[layer_idx] is not None:
                values_to_return = torch.cat(
                    [self.value_states_quant_cache[layer_idx], self.value_states_full_cache[layer_idx]], dim=-2)
            else:
                values_to_return = self.value_states_full_cache[layer_idx]

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_states_quant_trans_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _update_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # import pdb; pdb.set_trace()
        key_keep_mask = None
        if cache_kwargs is not None and getattr(self.kivi_config, "enable_think", False):
            key_keep_mask = cache_kwargs.get("key_keep_mask", None)

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
            # ThinK: apply keep_mask before and after fake_quant so stored cache has exact zeros in pruned dims (decode matches ThinK_kivi)
            if key_keep_mask is not None and isinstance(key_keep_mask, torch.Tensor):
                key_states_quant = key_states_quant * key_keep_mask.unsqueeze(2).to(key_states_quant.dtype)
            key_states_quant = fake_quant(key_states_quant, self.kivi_config.k_group_size[layer_idx], self.kivi_config.k_bits[layer_idx], self.kivi_config.k_quant_scheme)
            if key_keep_mask is not None and isinstance(key_keep_mask, torch.Tensor):
                key_states_quant = key_states_quant * key_keep_mask.unsqueeze(2).to(key_states_quant.dtype)

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
        self.key_keep_mask_cache.append(key_keep_mask)

    def _update_generation(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        if self.key_states_full_cache[layer_idx] is not None:
            self.key_states_full_cache[layer_idx] = torch.cat([self.key_states_full_cache[layer_idx], key_states], dim=2)
        else:
            self.key_states_full_cache[layer_idx] = key_states

        self.value_states_full_cache[layer_idx] = torch.cat([self.value_states_full_cache[layer_idx], value_states], dim=2)

    def _update_stride(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle a multi-token stride chunk (non-prefill).

        Merges the existing full-precision residual with the new stride tokens,
        re-partitions into fake-quantized and residual according to the residual_length
        invariant, and returns the full accumulated K/V for standard attention.
        """
        cfg = self.kivi_config
        residual_length = cfg.residual_length

        # ── KEYS ─────────────────────────────────────────────────────────
        if self.key_states_full_cache[layer_idx] is not None:
            combined_key = torch.cat([self.key_states_full_cache[layer_idx], key_states], dim=2)
        else:
            combined_key = key_states

        key_len = combined_key.shape[2]
        rem = key_len % residual_length
        if rem == 0:
            key_to_quant = combined_key
            new_key_full = None
        elif key_len < residual_length:
            key_to_quant = None
            new_key_full = combined_key
        else:
            key_to_quant = combined_key[:, :, :-rem, :].contiguous()
            new_key_full = combined_key[:, :, -rem:, :].contiguous()

        if key_to_quant is not None:
            assert key_to_quant.shape[2] % cfg.k_group_size[layer_idx] == 0, (
                f"Stride key tokens ({key_to_quant.shape[2]}) must be divisible by "
                f"k_group_size ({cfg.k_group_size[layer_idx]}). Adjust stride accordingly."
            )
            keep_mask = None
            if getattr(cfg, 'enable_think', False) and layer_idx < len(self.key_keep_mask_cache):
                keep_mask = self.key_keep_mask_cache[layer_idx]
            key_to_quant_m = key_to_quant
            if keep_mask is not None and isinstance(keep_mask, torch.Tensor):
                key_to_quant_m = key_to_quant * keep_mask.unsqueeze(2).to(key_to_quant.dtype)
            key_quant_new = fake_quant(
                key_to_quant_m, cfg.k_group_size[layer_idx], cfg.k_bits[layer_idx], cfg.k_quant_scheme
            )
            if keep_mask is not None and isinstance(keep_mask, torch.Tensor):
                key_quant_new = key_quant_new * keep_mask.unsqueeze(2).to(key_quant_new.dtype)
            if self.key_states_quant_trans_cache[layer_idx] is not None:
                self.key_states_quant_trans_cache[layer_idx] = torch.cat(
                    [self.key_states_quant_trans_cache[layer_idx], key_quant_new], dim=2)
            else:
                self.key_states_quant_trans_cache[layer_idx] = key_quant_new

        self.key_states_full_cache[layer_idx] = new_key_full

        # ── VALUES ───────────────────────────────────────────────────────
        combined_val = torch.cat([self.value_states_full_cache[layer_idx], value_states], dim=2)
        val_len = combined_val.shape[2]
        if val_len <= residual_length:
            val_to_quant = None
            new_val_full = combined_val
        else:
            val_to_quant = combined_val[:, :, :-residual_length, :].contiguous()
            new_val_full = combined_val[:, :, -residual_length:, :].contiguous()

        if val_to_quant is not None:
            assert val_to_quant.shape[2] % cfg.v_group_size[layer_idx] == 0, (
                f"Stride value tokens ({val_to_quant.shape[2]}) must be divisible by "
                f"v_group_size ({cfg.v_group_size[layer_idx]}). Adjust stride accordingly."
            )
            val_quant_new = fake_quant(
                val_to_quant, cfg.v_group_size[layer_idx], cfg.v_bits[layer_idx], cfg.v_quant_scheme
            )
            if self.value_states_quant_cache[layer_idx] is not None:
                self.value_states_quant_cache[layer_idx] = torch.cat(
                    [self.value_states_quant_cache[layer_idx], val_quant_new], dim=2)
            else:
                self.value_states_quant_cache[layer_idx] = val_quant_new

        self.value_states_full_cache[layer_idx] = new_val_full

        # ── Return full K/V for standard attention (no dequant needed for FakeCache) ─
        if self.key_states_quant_trans_cache[layer_idx] is not None:
            if new_key_full is not None:
                keys_to_return = torch.cat(
                    [self.key_states_quant_trans_cache[layer_idx], new_key_full], dim=2)
            else:
                keys_to_return = self.key_states_quant_trans_cache[layer_idx]
        else:
            keys_to_return = new_key_full

        if self.value_states_quant_cache[layer_idx] is not None:
            values_to_return = torch.cat(
                [self.value_states_quant_cache[layer_idx], new_val_full], dim=2)
        else:
            values_to_return = new_val_full

        return keys_to_return, values_to_return

    def lazy_update(self, layer_idx: int):
        # Stride phase invariants are maintained by _update_stride; skip here.
        if self._is_stride:
            return
        if self.key_states_full_cache[layer_idx].shape[-2] == self.kivi_config.residual_length:
            assert self.kivi_config.residual_length % self.kivi_config.k_group_size[layer_idx] == 0
            key_to_pack = self.key_states_full_cache[layer_idx]
            # ThinK: apply same channel mask (zero out pruned dims) so all quant keys are consistent with prefill
            keep_mask = None
            if layer_idx < len(self.key_keep_mask_cache):
                keep_mask = self.key_keep_mask_cache[layer_idx]
            if keep_mask is not None and isinstance(keep_mask, torch.Tensor):
                # keep shape (B, n_kv, seqlen, D); zero pruned dims to match prefill masked keys
                key_to_pack = key_to_pack * keep_mask.unsqueeze(2).to(key_to_pack.dtype)
            key_states_quant_trans_new = fake_quant(key_to_pack, self.kivi_config.k_group_size[layer_idx], self.kivi_config.k_bits[layer_idx], self.kivi_config.k_quant_scheme)

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

# Backwards-compatible aliases used by generation.py.
# ThinK is enabled via `KIVICacheConfig.enable_think` and the keep-mask stored in `KIVIDynamicCache`.
ThinkKIVIDynamicCache = KIVIDynamicCache
ThinkKIVIFakeCache = KIVIFakeCache