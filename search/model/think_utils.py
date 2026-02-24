"""
ThinK pruning utilities for KV cache (KIVI + ThinK / Think_kivi).

Key pruning (Section 3.2): Score_i[j] = ||Q_i[-S_obs:, j] K_i[:, j]^T||_F
Value pruning (Appendix D Eq 3-4): Scorev,i[j] = ||softmax(Q[-S_obs:] K^T / sqrt(D)) V[:, j]||_F
"""

import torch
from typing import List, Optional, Tuple


# Default observation window size (following SnapKV)
DEFAULT_S_OBS = 256


def compute_k_channel_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    s_obs: int = DEFAULT_S_OBS,
) -> torch.Tensor:
    """
    Compute query-driven importance scores for each channel of the key cache (ThinK Section 3.2).

    Score_i[j] = ||Q_i[-S_obs:, j] K_i[:, j]^T||_F

    Args:
        query_states: (batch, num_heads, seq_len, head_dim)
        key_states: (batch, num_heads, seq_len, head_dim)
        s_obs: observation window - use last s_obs query tokens

    Returns:
        scores: (batch, num_heads, head_dim) - per-channel importance
    """
    bsz, nh, seq_len, head_dim = key_states.shape
    s_obs = min(s_obs, seq_len, query_states.shape[2])

    # Q_obs: (bsz, nh, s_obs, head_dim), K: (bsz, nh, seq_len, head_dim)
    q_obs = query_states[:, :, -s_obs:, :]  # (bsz, nh, s_obs, D)
    # For each channel j: Q[:,:,j] (bsz, nh, s_obs), K[:,:,j] (bsz, nh, seq_len)
    # Score[j] = ||Q[:,:,j] @ K[:,:,j].T||_F = (bsz, nh, s_obs, 1) * (bsz, nh, 1, seq_len) -> (bsz, nh, s_obs, seq_len)
    # Frobenius norm over last two dims per batch/head
    # Equivalently: (Q_obs * K) sum over seq then norm, or we compute per channel
    # Q[:,:,:,j] @ K[:,:,:,j].T -> (bsz, nh, s_obs, seq_len), norm = sqrt(sum of squares)
    q_obs_t = q_obs.transpose(2, 3)  # (bsz, nh, D, s_obs)
    k_t = key_states  # (bsz, nh, seq_len, D)
    # For channel j: (bsz, nh, s_obs, 1) * (bsz, nh, 1, seq_len) via matmul
    # matmul: (bsz, nh, s_obs, D) @ (bsz, nh, D, seq_len) -> (bsz, nh, s_obs, seq_len) gives full QK^T
    # We need per-channel: Q[:,:,:,j:j+1] @ K[:,:,:,j:j+1].T
    # Simpler: compute full QK^T and extract... no that doesn't give per-channel.
    # Score[j] = ||Q[:,:,:,j] K[:,:,:,j].T||_F
    # Q[:,:,:,j] -> (bsz, nh, s_obs), K[:,:,:,j].T -> (bsz, nh, seq_len)
    # (bsz, nh, s_obs) @ (bsz, nh, seq_len) = batched matmul: (bsz, nh, s_obs, seq_len)
    scores = torch.zeros(bsz, nh, head_dim, device=query_states.device, dtype=query_states.dtype)
    for j in range(head_dim):
        qj = query_states[:, :, -s_obs:, j]  # (bsz, nh, s_obs)
        kj = key_states[:, :, :, j]  # (bsz, nh, seq_len)
        # (bsz, nh, s_obs) @ (bsz, nh, seq_len) -> need (bsz, nh, s_obs, seq_len)
        out = torch.einsum("bhs,bht->bhst", qj, kj)
        scores[:, :, j] = out.norm(p="fro", dim=(-2, -1))

    return scores


def compute_k_channel_scores_efficient(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    s_obs: int = DEFAULT_S_OBS,
) -> torch.Tensor:
    """
    Efficient batched computation of K channel scores.
    Score_i[j] = ||Q_i[-S_obs:, j] K_i[:, j]^T||_F
    = sqrt(sum_over_tokens( (Q[:,j] K[:,j])^2 ))
    The (s_obs, seq_len) matrix for channel j has elements Q[n,j]*K[m,j].
    Frobenius = sqrt(sum_n sum_m (Q[n,j]^2 * K[m,j]^2)) = sqrt( (sum_n Q[n,j]^2) * (sum_m K[m,j]^2) )
    Actually no: (Q K^T)_{nm} = Q[n,j] K[m,j], so ||A||_F^2 = sum_n sum_m (Q[n,j] K[m,j])^2
    = (sum_n Q[n,j]^2) * (sum_m K[m,j]^2)  when we expand. So ||A||_F = sqrt(sum_n Q^2) * sqrt(sum_m K^2)
    = ||Q[:,j]||_2 * ||K[:,j]||_2
    So for each channel j, Score[j] = ||Q_obs[:,j]||_2 * ||K[:,j]||_2
    """
    bsz, nh, seq_len, head_dim = key_states.shape
    s_obs = min(s_obs, seq_len, query_states.shape[2])
    q_obs = query_states[:, :, -s_obs:, :]  # (bsz, nh, s_obs, D)
    # (bsz, nh, D): norm over s_obs for Q, norm over seq_len for K
    q_norm = q_obs.norm(dim=2)  # (bsz, nh, D)
    k_norm = key_states.norm(dim=2)  # (bsz, nh, D)
    scores = q_norm * k_norm
    return scores


def compute_v_channel_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    s_obs: int = DEFAULT_S_OBS,
    head_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute query-driven importance scores for each channel of the value cache (Appendix D Eq 3).

    Scorev,i[j] = ||softmax(Qi[-S_obs:] K^T_i / sqrt(D)) Vi[:, j]||_F

    Args:
        query_states: (batch, num_heads, seq_len, head_dim)
        key_states: (batch, num_heads, seq_len, head_dim)
        value_states: (batch, num_heads, seq_len, head_dim)
        s_obs: observation window
        head_dim: head dimension (for scaling), default from query shape

    Returns:
        scores: (batch, num_heads, head_dim)
    """
    return compute_v_channel_scores_efficient(
        query_states, key_states, value_states, s_obs, head_dim
    )


def compute_v_channel_scores_efficient(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    s_obs: int = DEFAULT_S_OBS,
    head_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Efficient batched computation of V channel scores (Appendix D Eq 3).

    Scorev,i[j] = ||softmax(Qi[-S_obs:] K^T_i / sqrt(D)) Vi[:, j]||_F

    Uses F.scaled_dot_product_attention when available to fuse Q@K^T, softmax,
    and @V into a single optimized kernel (FlashAttention / memory-efficient attention).
    Falls back to explicit matmul+softmax+matmul when SDPA is not supported.

    Args:
        query_states: (batch, num_heads, seq_len, head_dim)
        key_states: (batch, num_heads, seq_len, head_dim)
        value_states: (batch, num_heads, seq_len, head_dim)
        s_obs: observation window
        head_dim: head dimension (for scaling), default from query shape

    Returns:
        scores: (batch, num_heads, head_dim)
    """
    bsz, nh, seq_len, d = value_states.shape
    if head_dim is None:
        head_dim = d
    s_obs = min(s_obs, seq_len, query_states.shape[2])

    q_obs = query_states[:, :, -s_obs:, :]  # (bsz, nh, s_obs, D)

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        # Fused path: SDPA = softmax(QK^T/sqrt(D)) @ V in one kernel
        # O(1) memory for attention weights (FlashAttention), faster on GPU
        scale = 1.0 / (head_dim ** 0.5)
        attn_v = torch.nn.functional.scaled_dot_product_attention(
            q_obs,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )  # (bsz, nh, s_obs, D)
    else:
        # Fallback: explicit matmul + softmax + matmul
        scale = 1.0 / (head_dim ** 0.5)
        attn_weights = torch.matmul(q_obs, key_states.transpose(2, 3)) * scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_v = torch.matmul(attn_weights, value_states)  # (bsz, nh, s_obs, D)

    # Score[j] = ||(attn @ V)[:,:,:,j]||_2 = L2 norm over s_obs for each channel
    scores = attn_v.norm(p=2, dim=2)  # (bsz, nh, D)
    return scores


def get_top_k_indices(
    scores: torch.Tensor,
    retention_ratio: float,
) -> torch.Tensor:
    """
    Get indices of top T channels by score.
    T = floor(retention_ratio * head_dim). retention_ratio=1.0 means no pruning.
    scores: (batch, num_heads, head_dim)
    Returns: (batch, num_heads, T) integer indices
    """
    bsz, nh, head_dim = scores.shape
    t = max(1, int(retention_ratio * head_dim))
    t = min(t, head_dim)
    if t >= head_dim:
        return torch.arange(head_dim, device=scores.device).unsqueeze(0).unsqueeze(0).expand(bsz, nh, -1)
    _, indices = torch.topk(scores, t, dim=-1)  # (bsz, nh, t)
    return indices


def prune_kv_by_indices(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    k_indices: torch.Tensor,
    v_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prune key and value states to keep only the selected channels.
    key_states: (B, nh, S, D), value_states: (B, nh, S, D)
    k_indices: (B, nh, T_k) or (nh, T_k) - indices to keep
    v_indices: (B, nh, T_v) or (nh, T_v)

    Returns pruned key_states (B, nh, S, T_k), value_states (B, nh, S, T_v)
    """
    if k_indices.dim() == 2:
        k_indices = k_indices.unsqueeze(0).expand(key_states.shape[0], -1, -1)
    if v_indices.dim() == 2:
        v_indices = v_indices.unsqueeze(0).expand(value_states.shape[0], -1, -1)

    bsz, nh, seq_len, _ = key_states.shape
    t_k = k_indices.shape[-1]
    t_v = v_indices.shape[-1]

    # Gather: output[b,h,s,t] = key_states[b,h,s,k_indices[b,h,t]]
    # index (B, nh, S, T_k) with index[b,h,s,t] = k_indices[b,h,t]
    k_idx = k_indices.unsqueeze(2).expand(-1, -1, seq_len, -1)
    v_idx = v_indices.unsqueeze(2).expand(-1, -1, seq_len, -1)
    k_pruned = torch.gather(key_states, dim=3, index=k_idx)
    v_pruned = torch.gather(value_states, dim=3, index=v_idx)
    return k_pruned, v_pruned


def expand_pruned_to_full(
    pruned_tensor: torch.Tensor,
    keep_indices: torch.Tensor,
    head_dim: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    Expand pruned tensor (fewer channels) back to full head_dim by scattering.
    pruned_tensor: (B, nh, S, T)
    keep_indices: (B, nh, T) or (nh, T)
    Returns: (B, nh, S, D) with pruned channels at keep_indices, rest = fill_value
    """
    bsz, nh, seq_len, t = pruned_tensor.shape
    if keep_indices.dim() == 2:
        keep_indices = keep_indices.unsqueeze(0).expand(bsz, -1, -1)
    out = torch.full(
        (bsz, nh, seq_len, head_dim),
        fill_value,
        device=pruned_tensor.device,
        dtype=pruned_tensor.dtype,
    )
    out.scatter_(3, keep_indices.unsqueeze(2).expand(-1, -1, seq_len, -1), pruned_tensor)
    return out
