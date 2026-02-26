"""
Llama KIVI + ThinK (Think_kivi): KV cache with quantization and channel pruning.
"""

import types
from typing import Optional, Tuple, Union

import torch

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from transformers.models.llama.modeling_llama import (
    add_start_docstrings_to_model_forward,
    LLAMA_INPUTS_DOCSTRING,
)

from model.KIVICache import KIVICacheConfig, KIVIDynamicCache, KIVIFakeCache, ThinkKIVIDynamicCache, ThinkKIVIFakeCache
from model.kivi_utils import quant_kv_output, update_causal_mask, lazy_update
from model.kivi_utils_think import get_past_key_values_think, attention_forward_think, apply_think_pruning_kv_output

logger = logging.get_logger(__name__)


def replace_attention_forward_think(self):
    """Attention forward that passes query_states for ThinK pruning and uses Think routing."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states, value_states = quant_kv_output(self, key_states, value_states, attention_mask)

        if hasattr(self.config, "quant_kv_output") and self.config.quant_kv_output:
            key_states, value_states = apply_think_pruning_kv_output(
                self, query_states, key_states, value_states
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if isinstance(past_key_value, (ThinkKIVIDynamicCache, ThinkKIVIFakeCache)):
                cache_kwargs["query_states"] = query_states
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        is_prefill, attention_interface = attention_forward_think(self, past_key_value)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            past_key_value=past_key_value,
            **kwargs,
        )

        lazy_update(self, is_prefill, past_key_value)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    self.forward = types.MethodType(forward, self)


def replace_model_forward_think(model_inner):
    """Model forward using ThinkKIVIDynamicCache when appropriate. model_inner is the LlamaModel."""
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            from transformers.utils import logging
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = get_past_key_values_think(self)

        if past_key_values is not None:
            assert isinstance(
                past_key_values, (KIVIDynamicCache, KIVIFakeCache, ThinkKIVIDynamicCache, ThinkKIVIFakeCache)
            ), "past_key_values must be KIVIDynamicCache, KIVIFakeCache, ThinkKIVIDynamicCache, or ThinkKIVIFakeCache"
            assert self.config.quant_kv_output is False, "quant_kv_output must be False when packing is False"

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        causal_mask = update_causal_mask(self, causal_mask, inputs_embeds, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
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
                    position_embeddings,
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
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    model_inner.forward = types.MethodType(forward, model_inner)


def convert_model_think_kivi(model):
    """Convert model to use KIVI + ThinK (Think_kivi)."""
    model.config.kv_method = "think"
    replace_model_forward_think(model.model)
    for layer in model.model.layers:
        replace_attention_forward_think(layer.self_attn)
