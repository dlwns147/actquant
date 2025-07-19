import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from quant.kivi_utils.new_pack import triton_quantize_and_pack_along_last_dim, fake_quant
from quant.kivi_utils.matmul import cuda_bmm_fA_qB_outer

from transformers.models.mistral.configuration_mistral import *
from transformers.models.mistral.modeling_mistral import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from .cache import KIVIDynamicCache
from transformers.utils import is_flash_attn_greater_or_equal_2_10, logging

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"
class MistralKIVIAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, orig_attn, config: MistralConfig):
        super().__init__()
        self.config = config
        self.layer_idx = orig_attn.layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = orig_attn.is_causal

        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj

        self.rotary_emb = orig_attn.rotary_emb
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += cache_position[0]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.config.quant_kv_output:
            key_states = fake_quant(key_states, self.config.k_group_size[self.layer_idx], self.config.k_bits[self.layer_idx], self.config.k_quant_per)
            value_states = fake_quant(value_states, self.config.v_group_size[self.layer_idx], self.config.v_bits[self.layer_idx], self.config.v_quant_per)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        assert isinstance(past_key_value, KIVIDynamicCache) or past_key_value == None, f"past_key_value : {type(past_key_value)}"
        if past_key_value is not None and len(past_key_value) > self.layer_idx:
            key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn = past_key_value[self.layer_idx]
            
            attn_qkquant = cuda_bmm_fA_qB_outer(self.config.k_group_size[self.layer_idx], query_states, key_states_quant_trans, key_scale_trans, key_mn_trans, self.config.k_bits[self.layer_idx]) if key_states_quant_trans is not None else None
            key_states_full = torch.cat([key_states_full, key_states], dim=2) if key_states_full is not None else key_states
            attn_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))
            attn_weights = torch.cat([attn_qkquant, attn_qkfull], dim=-1) / math.sqrt(self.head_dim) if attn_qkquant is not None else attn_qkfull / math.sqrt(self.head_dim)

            # update key cache
            if key_states_full.shape[-2] == self.config.residual_length:
                assert self.config.residual_length % self.config.k_group_size[self.layer_idx] == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), self.config.k_group_size[self.layer_idx], self.config.k_bits[self.layer_idx])
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
           
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))
            else:
                attn_output = cuda_bmm_fA_qB_outer(self.config.v_group_size[self.layer_idx], attn_weights[:, :, :, :-value_full_length], value_states_quant, value_scale, value_mn, self.config.v_bits[self.layer_idx])
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], repeat_kv(value_states_full, self.num_key_value_groups))
            attn_output = attn_output.transpose(1, 2).contiguous()

            # update value cache
            if value_full_length > self.config.residual_length:
                assert value_full_length == self.config.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), self.config.v_group_size[self.layer_idx], self.config.v_bits[self.layer_idx])
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn

        else:
            # print(f"kivi with flash! {self.config.k_bits}")
            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                # Handle the case where the model is quantized
                if hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            if self.config.use_flash:
                attn_output = _flash_attention_forward(
                    query_states=query_states.transpose(1, 2),
                    key_states=key_states.transpose(1, 2),
                    value_states=value_states.transpose(1, 2),
                    # attention_mask=attention_mask,
                    attention_mask=None,
                    query_length=q_len,
                    position_ids=position_ids,
                    dropout=0.0,
                    sliding_window=getattr(self, "sliding_window", None),
                    use_top_left_mask=self._flash_attn_uses_top_left_mask,
                    is_causal=self.is_causal,
                ) # flash attention

            else:
                causal_mask = attention_mask
                if attention_mask is not None:
                    causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states.contiguous() if query_states.device.type == "cuda"  and causal_mask is not None else query_states,
                    repeat_kv(key_states, self.num_key_value_groups).contiguous() if query_states.device.type == "cuda" and causal_mask is not None else repeat_kv(key_states, self.num_key_value_groups),
                    repeat_kv(value_states, self.num_key_value_groups).contiguous() if query_states.device.type == "cuda" and causal_mask is not None else repeat_kv(value_states, self.num_key_value_groups),
                    attn_mask=causal_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=True if causal_mask is None and q_len > 1 else False,
                )

            # quantize
            if self.config.residual_length > 0 and key_states.shape[-2] % self.config.residual_length != 0:
                if key_states.shape[-2] < self.config.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.config.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.config.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None

            if key_states_quant is not None and use_cache:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.config.k_group_size[self.layer_idx], self.config.k_bits[self.layer_idx])
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.config.residual_length or self.config.residual_length == 0:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.config.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.config.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, self.config.v_group_size[self.layer_idx], self.config.v_bits[self.layer_idx])

        # past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, 
        #                   value_states_quant, value_states_full, value_scale, value_mn) if use_cache else None
        
        if use_cache:
            past_key_value.update(key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, self.layer_idx)
        if self.config.use_flash:
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size) # flash attention 2
        else:
            attn_output = attn_output.transpose(1, 2).contiguous() # sdpa
            attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralKIVIModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, orig_model, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = orig_model.embed_tokens
        self.layers = orig_model.layers
        self._attn_implementation = config._attn_implementation
        self.norm = orig_model.norm

        self.gradient_checkpointing = orig_model.gradient_checkpointing

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and (past_key_values is None or (isinstance(past_key_values, Cache) and len(past_key_values) == 0)):
            past_key_values = KIVIDynamicCache()

        # if use_cache and not isinstance(past_key_values, Cache):
        #     return_legacy_cache = True
        #     if past_key_values is None:
        #         past_key_values = DynamicCache()
        #     else:
        #         past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        #         logger.warning_once(
        #             "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
        #             "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
        #             "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
        #         )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        use_cache: bool,
        output_attentions: bool,
    ):
        if self._attn_implementation == "flash_attention_2":
            if attention_mask is not None and use_cache:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.

        # cache_position must be valid here no matter which cache we use
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache
        if using_sliding_window_cache:
            target_length = max(sequence_length, self.config.sliding_window)
        # StaticCache
        elif using_static_cache:
            target_length = past_key_values.get_max_length()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            exclude_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if self.config.sliding_window is not None:
                if not using_sliding_window_cache or sequence_length > self.config.sliding_window:
                    exclude_mask.bitwise_or_(
                        torch.arange(target_length, device=device)
                        <= (cache_position.reshape(-1, 1) - self.config.sliding_window)
                    )
            causal_mask *= exclude_mask
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.dim() == 2:
                    mask_length = attention_mask.shape[-1]
                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class MistralKIVIForCausalLM(MistralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, orig_lm, config):
        super().__init__(config)
        self.model = orig_lm.model
        self.vocab_size = config.vocab_size
        self.lm_head = orig_lm.lm_head

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        # logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        
        if isinstance(past_key_values, DynamicCache) and past_key_values.get_seq_length() == 0:
            past_key_values = KIVIDynamicCache()

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
