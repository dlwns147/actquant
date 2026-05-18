import math
import types
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from quant.kivi_utils.new_pack import fake_quant, unpack_and_dequant_kcache, unpack_and_dequant_vcache
from quant.kivi_utils.matmul import cuda_bmm_fA_qB_outer

from transformers.models.gemma3.configuration_gemma3 import *
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3RMSNorm,
    Gemma3MLP,
    Gemma3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
    GEMMA3_INPUTS_DOCSTRING,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache

from model.KIVICache import KIVICacheConfig, KIVIDynamicCache, KIVIFakeCache, _think_key_pruner_query_driven, _think_value_pruner_attention_driven
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

        # Gemma3-specific: per-head RMSNorm on Q/K BEFORE RoPE (not present in
        # Llama/Qwen/Mistral). Must run on the unquantised states so the cached
        # K (and the query that re-attends to it) are consistent.
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ThinK pruning for packing=False (mirror llama_kivi.py):
        # - Prefill: keep full Q/K for attention; only compute keep_mask and pass
        #   to cache, so cache stores masked K (and V) consistent with decode.
        keep_mask_for_cache = None
        v_keep_mask_for_cache = None
        kivi_config = getattr(self.config, "kivi_config", None)
        if (
            past_key_value is not None  # skip when use_cache=False
            and kivi_config is not None
            and getattr(kivi_config, "enable_think", False)
            and not getattr(kivi_config, "packing", False)
        ):
            residual_length = getattr(kivi_config, "residual_length", 0) or 0
            bsz, n_kv, seqlen, dim = key_states.shape
            old_len = max(0, seqlen - residual_length) if residual_length > 0 else seqlen
            if old_len > 0:
                key_old = key_states[:, :, :old_len, :]
                try:
                    pruning_dim = int(kivi_config.k_pruning_dim[self.layer_idx])
                except Exception:
                    pruning_dim = 0
                if pruning_dim > 0:
                    with torch.no_grad():
                        _, keep_mask = _think_key_pruner_query_driven(
                            key_old, query_states, pruning_dim=pruning_dim
                        )
                    keep_mask_for_cache = keep_mask
                try:
                    v_pruning_dim = int(kivi_config.v_pruning_dim[self.layer_idx])
                except Exception:
                    v_pruning_dim = 0
                if v_pruning_dim > 0:
                    value_old = value_states[:, :, :old_len, :]
                    with torch.no_grad():
                        _, v_keep_mask = _think_value_pruner_attention_driven(
                            value_old, query_states, key_old, pruning_dim=v_pruning_dim
                        )
                    v_keep_mask_for_cache = v_keep_mask

        key_states, value_states = quant_kv_output(self, key_states, value_states, attention_mask, query_states=query_states)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # ThinK: allow cache to use query_states for query-driven pruning during prefill packing
            if hasattr(self.config, "kivi_config") and getattr(self.config.kivi_config, "enable_think", False):
                cache_kwargs["query_states"] = query_states
                if keep_mask_for_cache is not None:
                    cache_kwargs["key_keep_mask"] = keep_mask_for_cache
                if v_keep_mask_for_cache is not None:
                    cache_kwargs["value_keep_mask"] = v_keep_mask_for_cache
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        is_prefill, attention_interface = attention_forward(self, past_key_value)

        if attention_mask is not None:
            # Gemma3 casts the (already sliding-window-sliced by the decoder
            # layer) mask to the query dtype before attention.
            attention_mask = attention_mask.to(query_states)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,  # Gemma3: query_pre_attn_scalar**-0.5
            sliding_window=self.sliding_window,  # None for global layers
            past_key_value=past_key_value,
            **kwargs,
        )

        lazy_update(self, is_prefill, past_key_value)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    self.forward = types.MethodType(forward, self)


def replace_model_forward(self):
    @add_start_docstrings_to_model_forward(GEMMA3_INPUTS_DOCSTRING)
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
        last_cache_position: Optional[int] = None,
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

        # KIVI/ThinK: replace Gemma3's default HybridCache with the KIVI cache.
        if use_cache and past_key_values is None:
            past_key_values = get_past_key_values(self)

        if past_key_values is not None:
            assert isinstance(past_key_values, (KIVIDynamicCache, KIVIFakeCache)), "past_key_values must be KIVIDynamicCache or KIVIFakeCache"
            assert self.config.quant_kv_output == False, "quant_kv_output must be False when use_cache is True"

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Needed for the sliding-window mask slicing inside Gemma3DecoderLayer.
        if last_cache_position is None:
            last_cache_position = 0
            if attention_mask is not None:
                last_cache_position = (
                    attention_mask.shape[-1] if attention_mask.dim() == 2 else cache_position[-1].item()
                )

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        causal_mask = update_causal_mask(self, causal_mask, inputs_embeds, past_key_values)

        hidden_states = inputs_embeds

        # Gemma3 keeps two RoPE tables: global (full-attention layers) and local
        # (sliding-window layers). The decoder layer picks one per layer.
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings_global,
                    position_embeddings_local,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    last_cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    last_cache_position=last_cache_position,
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

    self.forward = types.MethodType(forward, self)


def convert_model_kivi(self):
    replace_model_forward(self.model)
    for layer in self.model.layers:
        replace_attention_forward(layer.self_attn)
