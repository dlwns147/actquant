import math
import warnings
import types
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from quant.kivi_utils.new_pack import triton_quantize_and_pack_along_last_dim, fake_quant
from quant.kivi_utils.matmul import cuda_bmm_fA_qB_outer

from transformers.models.qwen2.configuration_qwen2 import *
from transformers.models.qwen2.modeling_qwen2 import *
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils import logging

from model.KIVICache import KIVICacheConfig, KIVIDynamicCache, KIVIFakeCache
from model.kivi_utils import (
    is_prefill,
    get_past_key_values,
    quant_kv_output,
    update_causal_mask,
    forward_for_kivi_gemv,
    attention_forward,
    lazy_update,
)

logger = logging.get_logger(__name__)


def replace_attention_forward(self):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states, value_states = quant_kv_output(self, key_states, value_states, attention_mask)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        is_prefill, attention_interface = attention_forward(self, past_key_value)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            past_key_value=past_key_value,
            **kwargs,
        )

        lazy_update(self, is_prefill, past_key_value)

# torch.save(attn_output, '/SSD/Woo/qwen2.5/100_token/true/attn_output.pt')
# torch.save(attn_weights, '/SSD/Woo/qwen2.5/100_token/true/attn_weights.pt')
# torch.save(query_states, '/SSD/Woo/qwen2.5/100_token/true/query_states.pt')
# torch.save(key_states, '/SSD/Woo/qwen2.5/100_token/true/key_states.pt')
# torch.save(value_states, '/SSD/Woo/qwen2.5/100_token/true/value_states.pt')

        # if self.layer_idx == 0:
        #     self.seen_tokens += 1

        # if not is_prefill and past_key_value is not None:
        # #     for i in range(28):
        # #         print(i, attn_output[0, 0, i].mean())
        #     if self.seen_tokens == 0 or self.seen_tokens == 10 or self.seen_tokens == 100:
        #         import code; code.interact('qwen2_kivi.py line 95', local=dict(globals(), **locals()))

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    self.forward = types.MethodType(forward, self)


def replace_model_forward(self):
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
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
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = get_past_key_values(self)
        
        if past_key_values is not None:
            assert isinstance(past_key_values, (KIVIDynamicCache, KIVIFakeCache)), "past_key_values must be KIVIDynamicCache or KIVIFakeCache"
            assert self.config.quant_kv_output == False, "quant_kv_output must be False when packing is False"

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

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
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

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    self.forward = types.MethodType(forward, self)

def convert_model_kivi(self):
    replace_model_forward(self.model)
    for layer in self.model.layers:
        replace_attention_forward(layer.self_attn)