import math
import types
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer, triton_bmm_fA_qB_outer

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    _CONFIG_FOR_DOC,
    LLAMA_START_DOCSTRING,
    LLAMA_INPUTS_DOCSTRING,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils import logging

from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache
from models.KIVICache import KIVICacheConfig, KIVIDynamicCache

logger = logging.get_logger(__name__)


def init_kivi_cache(self, config: KIVICacheConfig):
    self.nbits_key = config.nbits_key if config.nbits_key else config.nbits
    self.nbits_value = config.nbits_value if config.nbits_value else config.nbits
    self.q_group_size = config.q_group_size
    self.per_layer_quant = config.per_layer_quant
    if self.per_layer_quant:
        self.per_layer_config = config.per_layer_config
    else:
        self.per_layer_config = None

    if self.per_layer_quant:
        self.nbits_key = self.per_layer_config[self.layer_idx]["nbits_key"]
        self.nbits_value = self.per_layer_config[self.layer_idx]["nbits_value"]


def is_prefill(past_key_value: Optional[KIVIDynamicCache], layer_idx: int):
    if len(past_key_value) == layer_idx:
        return True
    return False


def forward_for_kivi_gemv(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: float = 1.0,
    past_key_value: Optional[KIVIDynamicCache] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    assert past_key_value is not None, "past_key_value is None"

    # [bsz, nh, t, hd]
    key_states_quant_trans = past_key_value.key_states_quant_trans_cache[module.layer_idx]
    key_states_full = past_key_value.key_states_full_cache[module.layer_idx]
    key_scale_trans = past_key_value.key_scale_trans_cache[module.layer_idx]
    key_mn_trans = past_key_value.key_mn_trans_cache[module.layer_idx]
    value_states_quant = past_key_value.value_states_quant_cache[module.layer_idx]
    value_states_full = past_key_value.value_states_full_cache[module.layer_idx]
    value_scale = past_key_value.value_scale_cache[module.layer_idx]
    value_mn = past_key_value.value_mn_cache[module.layer_idx]

    # calculate quantized query-key attention
    if key_states_quant_trans is not None:
        att_qkquant = cuda_bmm_fA_qB_outer(module.q_group_size, query_states, key_states_quant_trans, 
                        key_scale_trans, key_mn_trans, module.nbits_key)
        # att_qkquant = triton_bmm_fA_qB_outer(module.q_group_size, query_states, key_states_quant_trans, 
        #                 key_scale_trans, key_mn_trans, module.nbits_key)
    else:
        att_qkquant = None

    # concatenate full key states
    if key_states_full is not None:
        key_states_full = torch.cat([key_states_full, key_states], dim=2)
    else:
        key_states_full = key_states

    # calculate full query-key attention
    att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, module.num_key_value_groups).transpose(2, 3))

    # concatenate quantized and full query-key attention
    if att_qkquant is not None:
        attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) * scaling
    else:
        attn_weights = att_qkfull * scaling

    # add attention mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)


    value_states_full = torch.cat([value_states_full, value_states], dim=2)
    value_full_length = value_states_full.shape[-2]
    if value_states_quant is None:
        # This seems to be a bug in original code. It should be repeat_kv as well.
        # if module.layer_idx == 0:
            # import code; code.interact('forward_for_kivi_gemv line 126', local=dict(globals(), **locals()))
        attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, module.num_key_value_groups))
    else:
        attn_output = cuda_bmm_fA_qB_outer(module.q_group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant, 
                                        value_scale, value_mn, module.nbits_value)
        # attn_output = triton_bmm_fA_qB_outer(module.q_group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant, 
        #                                 value_scale, value_mn, module.nbits_value)
        attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], repeat_kv(value_states_full, module.num_key_value_groups))

    attn_output = attn_output.transpose(1, 2).contiguous()

    if past_key_value is not None:
        past_key_value.update(key_states_full, value_states_full, module.layer_idx)
                
    return attn_output, attn_weights


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

        if past_key_value is not None and is_prefill(past_key_value, self.layer_idx):
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        elif past_key_value is None:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        else:
            attention_interface = forward_for_kivi_gemv

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    self.forward = types.MethodType(forward, self)


def replace_model_forward(self, cache_config: KIVICacheConfig):
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
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = KIVIDynamicCache(cache_config)

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


def convert_model_kivi(self, cache_config: KIVICacheConfig):
    replace_model_forward(self.model, cache_config)
    for layer in self.model.layers:
        layer.self_attn.init_kivi_cache = types.MethodType(init_kivi_cache, layer.self_attn)
        layer.self_attn.init_kivi_cache(cache_config)
        replace_attention_forward(layer.self_attn)