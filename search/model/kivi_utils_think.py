"""
KIVI utils for Think_kivi (KIVI + ThinK pruning).
Uses ThinkKIVIDynamicCache and prunes query for matmul with pruned keys/values.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.KIVICache import KIVICacheConfig, KIVIDynamicCache, KIVIFakeCache, ThinkKIVIDynamicCache
from quant.kivi_utils.matmul import cuda_bmm_fA_qB_outer
from model.kivi_utils import repeat_kv

try:
    from model.think_utils import (
        compute_k_channel_scores_efficient,
        compute_v_channel_scores,
        get_top_k_indices,
        prune_kv_by_indices,
        expand_pruned_to_full,
        DEFAULT_S_OBS,
    )
    _THINK_AVAILABLE = True
except ImportError:
    _THINK_AVAILABLE = False
    DEFAULT_S_OBS = 256


def apply_think_pruning_kv_output(module, query_states, key_states, value_states):
    """
    Apply ThinK channel pruning to key_states and value_states when quant_kv_output=True
    and past_key_value is None (e.g., PPL evaluation with use_cache=False).
    Pruning is applied in-place of fake_quant: both affect the K/V used for attention.
    Returns full head_dim tensors with pruned channels zeroed (compatible with standard attention).

    Handles GQA: when num_attention_heads > num_key_value_heads, key/value are repeated
    to match query for score computation, then scores are aggregated per KV head.
    """
    if not _THINK_AVAILABLE:
        return key_states, value_states
    if not (hasattr(module.config, "quant_kv_output") and module.config.quant_kv_output):
        return key_states, value_states
    kivi_config = getattr(module.config, "kivi_config", None)
    if kivi_config is None:
        return key_states, value_states
    layer_idx = getattr(module, "layer_idx", 0)
    k_ratio = kivi_config.k_prune[layer_idx]
    v_ratio = kivi_config.v_prune[layer_idx]
    if k_ratio >= 1.0 and v_ratio >= 1.0:
        return key_states, value_states

    head_dim = key_states.shape[-1]
    s_obs = DEFAULT_S_OBS
    k_indices = None
    v_indices = None

    # GQA: query has num_attention_heads, key/value have num_key_value_heads.
    # Repeat KV to match Q for score computation, then aggregate scores per KV head.
    n_rep = getattr(module, "num_key_value_groups", 1)
    if query_states.shape[1] != key_states.shape[1]:
        key_for_scores = repeat_kv(key_states, n_rep)
        value_for_scores = repeat_kv(value_states, n_rep)
    else:
        key_for_scores = key_states
        value_for_scores = value_states

    if k_ratio < 1.0:
        k_scores = compute_k_channel_scores_efficient(query_states, key_for_scores, s_obs)
        # Aggregate scores per KV head when GQA: (bsz, num_heads, head_dim) -> (bsz, num_kv_heads, head_dim)
        if key_for_scores.shape[1] != key_states.shape[1]:
            bsz, nq, d = k_scores.shape
            nkv = key_states.shape[1]
            k_scores = k_scores.view(bsz, nkv, n_rep, d).mean(dim=2)
        k_indices = get_top_k_indices(k_scores, k_ratio)
        if k_indices.shape[-1] < head_dim:
            gs = kivi_config.k_group_size[layer_idx]
            t_k = k_indices.shape[-1]
            eff_gs = min(gs, t_k) if gs > 0 else t_k
            if eff_gs > 0:
                t_k_round = (t_k // eff_gs) * eff_gs
                if t_k_round > 0 and t_k_round < t_k:
                    k_indices = k_indices[..., :t_k_round]

    if v_ratio < 1.0:
        v_scores = compute_v_channel_scores(
            query_states, key_for_scores, value_for_scores, s_obs, head_dim
        )
        # Aggregate scores per KV head when GQA
        if value_for_scores.shape[1] != value_states.shape[1]:
            bsz, nq, d = v_scores.shape
            nkv = value_states.shape[1]
            v_scores = v_scores.view(bsz, nkv, n_rep, d).mean(dim=2)
        v_indices = get_top_k_indices(v_scores, v_ratio)
        if v_indices.shape[-1] < head_dim:
            gs = kivi_config.v_group_size[layer_idx]
            t_v = v_indices.shape[-1]
            eff_gs = min(gs, t_v) if gs > 0 else t_v
            if eff_gs > 0:
                t_v_round = (t_v // eff_gs) * eff_gs
                if t_v_round > 0 and t_v_round < t_v:
                    v_indices = v_indices[..., :t_v_round]

    if k_indices is None and v_indices is None:
        return key_states, value_states

    if k_indices is None:
        k_indices = torch.arange(head_dim, device=key_states.device).unsqueeze(0).expand(
            key_states.shape[1], -1
        )
    if v_indices is None:
        v_indices = torch.arange(head_dim, device=value_states.device).unsqueeze(0).expand(
            value_states.shape[1], -1
        )
    k_pruned, v_pruned = prune_kv_by_indices(key_states, value_states, k_indices, v_indices)
    key_states = expand_pruned_to_full(k_pruned, k_indices, head_dim, fill_value=0.0)
    value_states = expand_pruned_to_full(v_pruned, v_indices, head_dim, fill_value=0.0)
    return key_states, value_states


def get_past_key_values_think(module):
    """Return ThinkKIVIDynamicCache when method is 'think'."""
    if getattr(module.config, 'kv_method', 'kivi') == 'think':
        cfg = module.config.kivi_config
        return ThinkKIVIDynamicCache(cfg) if cfg.packing else KIVIFakeCache(cfg)
    cfg = module.config.kivi_config
    return KIVIDynamicCache(cfg) if cfg.packing else KIVIFakeCache(cfg)


def forward_for_think_kivi_gemv(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: float = 1.0,
    past_key_value: Optional[ThinkKIVIDynamicCache] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generation forward for Think_kivi: pruned K/V have fewer channels.
    Query must be pruned for the quantized part; full query for the residual part.
    """
    assert past_key_value is not None, "past_key_value is None"

    bsz, nh, t, hd = query_states.shape
    kivi_config = module.config.kivi_config
    layer_idx = module.layer_idx

    key_states_quant_trans = past_key_value.key_states_quant_trans_cache[layer_idx]
    key_states_full = past_key_value.key_states_full_cache[layer_idx]
    key_scale_trans = past_key_value.key_scale_trans_cache[layer_idx]
    key_mn_trans = past_key_value.key_mn_trans_cache[layer_idx]
    value_states_quant = past_key_value.value_states_quant_cache[layer_idx]
    value_states_full = past_key_value.value_states_full_cache[layer_idx]
    value_scale = past_key_value.value_scale_cache[layer_idx]
    value_mn = past_key_value.value_mn_cache[layer_idx]

    k_keep = past_key_value.k_keep_indices[layer_idx] if layer_idx < len(past_key_value.k_keep_indices) else None
    v_keep = past_key_value.v_keep_indices[layer_idx] if layer_idx < len(past_key_value.v_keep_indices) else None

    device = query_states.device
    k_group = kivi_config.k_group_size[layer_idx]
    v_group = kivi_config.v_group_size[layer_idx]

    # Query pruned for quantized K part (has T_k channels)
    if key_states_quant_trans is not None:
        if k_keep is not None and k_keep.numel() > 0:
            k_keep = k_keep.to(device)
            if k_keep.dim() == 2:
                k_keep = k_keep.unsqueeze(0).expand(bsz, -1, -1)
            # query_pruned: (B, nh, t, T_k)
            query_pruned = torch.gather(
                query_states.unsqueeze(-1).expand(-1, -1, -1, k_keep.shape[-1]),
                dim=3,
                index=k_keep.unsqueeze(2).expand(-1, -1, t, -1),
            )
        else:
            query_pruned = query_states
        att_qkquant = cuda_bmm_fA_qB_outer(
            k_group, query_pruned, key_states_quant_trans,
            key_scale_trans, key_mn_trans, kivi_config.k_bits[layer_idx],
        )
    else:
        att_qkquant = None

    att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, module.num_key_value_groups).transpose(2, 3))

    if att_qkquant is not None:
        attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) * scaling
    else:
        attn_weights = att_qkfull * scaling

    if attention_mask is not None and attention_mask.dim() == 4:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    value_full_length = value_states_full.shape[-2]
    if value_states_quant is None:
        attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, module.num_key_value_groups))
    else:
        attn_quant_part = attn_weights[:, :, :, :-value_full_length]
        attn_full_part = attn_weights[:, :, :, -value_full_length:]

        attn_output_quant = cuda_bmm_fA_qB_outer(
            v_group, attn_quant_part, value_states_quant,
            value_scale, value_mn, kivi_config.v_bits[layer_idx],
        )
        attn_output_full = torch.matmul(attn_full_part, repeat_kv(value_states_full, module.num_key_value_groups))

        if v_keep is not None and v_keep.numel() > 0 and attn_output_quant.shape[-1] < hd:
            v_keep_dev = v_keep.to(device)
            if v_keep_dev.dim() == 2:
                v_keep_dev = v_keep_dev.unsqueeze(0).unsqueeze(2).expand(bsz, nh, t, -1).long()
            attn_output = torch.zeros(bsz, nh, t, hd, device=device, dtype=attn_output_quant.dtype)
            attn_output.scatter_(3, v_keep_dev, attn_output_quant)
            attn_output += attn_output_full
        else:
            attn_output = attn_output_quant + attn_output_full

    attn_output = attn_output.transpose(1, 2).contiguous()

    past_key_value.lazy_update(layer_idx)

    return attn_output, attn_weights


def attention_forward_think(module, past_key_value):
    """Route to Think GEMV when using ThinkKIVIDynamicCache."""
    is_prefill = past_key_value is not None and past_key_value.is_prefill

    if is_prefill or past_key_value is None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        return is_prefill, ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
    elif past_key_value is not None and isinstance(past_key_value, ThinkKIVIDynamicCache) and module.config.kivi_config.packing:
        return is_prefill, forward_for_think_kivi_gemv
    else:
        from model.kivi_utils import attention_forward
        return attention_forward(module, past_key_value)
