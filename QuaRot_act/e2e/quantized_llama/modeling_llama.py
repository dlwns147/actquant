import functools
import quarot
import quarot.transformers
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaSdpaAttention, \
LlamaFlashAttention2, LlamaForCausalLM, apply_rotary_pos_emb, LlamaMLP
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing import Optional, Tuple
from transformers import Cache
from quarot import PackedQuantizedTensor


ALL_LAYERNORM_LAYERS.append(quarot.nn.RMSNorm)

class QuarotLlamaConfig(LlamaConfig):
    model_type = "llama_quarot"

class QuarotFP16LlamaAttention(LlamaFlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = torch.nn.Identity()
        self.o_proj_hadamard = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        hidden_states = self.quantizer(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # kv_seq_len = key_states.shape[1]
        # kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states, position_ids)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, unsqueeze_dim=2)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        assert past_key_value is not None
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "attention_mask": attention_mask}
        cache_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        

        dropout_rate = self.attention_dropout if self.training else 0.0
        assert self.is_causal

        if isinstance(cache_out, tuple):
            key_states, value_states = cache_out
            # attn_output = self._flash_attention_forward(
            #     query_states, 
            #     key_states, 
            #     value_states, 
            #     query_length=q_len, 
            #     attention_mask=attention_mask
            # )
            attn_output = _flash_attention_forward(
                query_states, 
                key_states, 
                value_states, 
                query_length=q_len, 
                position_ids=position_ids,
                attention_mask=attention_mask,
                dropout=dropout_rate,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                is_causal=self.is_causal,
            )
        else:
            attn_output = cache_out(query_states)

        attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class QuarotLlamaAttention(QuarotFP16LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quarot.nn.Quantizer()
        self.q_proj = quarot.nn.Linear4bit.from_float(self.q_proj)
        self.k_proj = quarot.nn.Linear4bit.from_float(self.k_proj)
        self.v_proj = quarot.nn.Linear4bit.from_float(self.v_proj)
        self.o_proj_hadamard = quarot.nn.OnlineHadamard(self.num_heads)
        self.o_proj = torch.nn.Sequential(
            quarot.nn.Quantizer(),
            quarot.nn.Linear4bit.from_float(self.o_proj)
        )

class QuarotLlamaMLP(LlamaMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quarot.nn.Quantizer()
        self.up_proj = quarot.nn.Linear4bit.from_float(self.up_proj)
        self.gate_proj = quarot.nn.Linear4bit.from_float(self.gate_proj)
        self.down_proj = torch.nn.Sequential(
            quarot.nn.OnlineHadamard(self.intermediate_size),
            quarot.nn.Quantizer(),
            quarot.nn.Linear4bit.from_float(self.down_proj)
        )

    def forward(self, x):
        x = self.quantizer(x)
        return super().forward(x)


class QuarotFP16LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # assert config._attn_implementation == "flash_attention_2"
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = (QuarotFP16LlamaAttention if config._attn_implementation == "flash_attention_2" else QuarotFP16LlamaSdpaAttention)(config=config, layer_idx=layer_idx)
        self.cache_dtype = "float16"
        self._expected_max_length = None

        
    def build_cache(self, batch_size, page_size, max_length):
        device = self.model.layers[0].self_attn.v_proj.weight.device
        dtype = self.cache_dtype or self.model.layers[0].self_attn.v_proj.weight.dtype
        
        num_heads = self.config.num_key_value_heads
        model_dim = self.config.hidden_size
        head_dim = model_dim // num_heads
        disable_quant = self.cache_dtype == "float16" 
        return quarot.transformers.MultiLayerPagedKVCache4Bit(
            batch_size=batch_size,
            page_size=page_size, 
            max_seq_len=max_length, 
            device=device, 
            n_layers=len(self.model.layers),
            num_heads=num_heads,
            head_dim=head_dim,
            disable_quant=disable_quant,
            hadamard_dtype=None if disable_quant else torch.float16
        )

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        # This is a hack to get the max length from generation_config.
        # Doing it here because max_length might not be set before this 
        # method is called.
        self._expected_max_length = generation_config.max_length # This value will be reset at the next forward call
        return super()._get_logits_processor(generation_config, *args, **kwargs)


    def forward(self, input_ids, *args, past_key_values=None, **kwargs):
        if past_key_values is None:
            max_length = self._expected_max_length or input_ids.shape[1]
            self._expected_max_length = None # Reset this value.
            past_key_values = self.build_cache(
                input_ids.shape[0], 
                page_size=max_length,  # For now working with single page per batch.
                max_length=max_length)
        out = super().forward(input_ids, *args, past_key_values=past_key_values, **kwargs)
        return out
    


class QuarotLlamaForCausalLM(QuarotFP16LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        self.norm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = QuarotLlamaAttention(config=config, layer_idx=layer_idx)
            layer.input_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = QuarotLlamaMLP(config=config)
        self.cache_dtype = "int4"


class QuarotFP16LlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = torch.nn.Identity()
        self.o_proj_hadamard = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        hidden_states = self.quantizer(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        assert past_key_value is not None
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "attention_mask": attention_mask}
        cache_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.

        if isinstance(cache_out, tuple):
            key_states, value_states = cache_out
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True if causal_mask is None and q_len > 1 else False,
            )
        else:
            attn_output = cache_out(query_states)

        attn_output = self.o_proj_hadamard(attn_output.transpose(1, 2)).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)
        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        # attn_output = self.o_proj(attn_output)


        return attn_output, None, past_key_value
