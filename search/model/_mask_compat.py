"""transformers 4.57.6 compat: LlamaModel lost `_update_causal_mask` /
`_prepare_4d_causal_attention_mask_with_cache_position` (causal masking moved to
masking_utils). The KIVI-patched forward (llama_kivi.py) still calls
`self._update_causal_mask(...)`. We bind these verbatim-from-transformers-4.50
implementations onto the model in convert_model_kivi so behavior is identical.
Deps (Cache/StaticCache/AttentionMaskConverter) all exist in 4.57.6; the
flex-attention branch is guarded (never hit on the sdpa eval path)."""
import torch
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
try:
    from transformers.integrations.flex_attention import make_flex_block_causal_mask, BlockMask
except Exception:  # flex attention not used on the search/eval path
    make_flex_block_causal_mask = None
    BlockMask = None


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask, sequence_length, target_length, dtype, device, cache_position, batch_size, **kwargs,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype)
    return causal_mask


def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values,
                        output_attentions=False):
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and (attention_mask == 0.0).any():
            return attention_mask
        return None
    if self.config._attn_implementation == "flex_attention":
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = make_flex_block_causal_mask(attention_mask)
        if BlockMask is not None and isinstance(attention_mask, BlockMask):
            return attention_mask

    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask, inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens, is_training=self.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_cache_shape()
    else:
        target_length = (
            attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1)

    causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask, sequence_length=sequence_length, target_length=target_length,
        dtype=dtype, device=device, cache_position=cache_position, batch_size=input_tensor.shape[0])

    if (self.config._attn_implementation == "sdpa" and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"] and not output_attentions):
        min_dtype = torch.finfo(dtype).min
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    return causal_mask


def bind_mask_compat(model):
    """Attach _update_causal_mask onto the LlamaModel instance if 4.57.6 removed it."""
    import types
    target = getattr(model, "model", model)  # LlamaForCausalLM.model is the LlamaModel
    if not hasattr(target, "_update_causal_mask"):
        target._update_causal_mask = types.MethodType(_update_causal_mask, target)
        target._prepare_4d_causal_attention_mask_with_cache_position = (
            _prepare_4d_causal_attention_mask_with_cache_position)
