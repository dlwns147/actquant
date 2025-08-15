from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from model.KIVICache import KIVICacheConfig, KIVIDynamicCache, KIVIFakeCache
from quant.kivi_utils.matmul import cuda_bmm_fA_qB_outer
from quant.kivi_utils.new_pack import fake_quant, unpack_and_dequant_kcache, unpack_and_dequant_vcache
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def is_prefill(past_key_value: Optional[KIVIDynamicCache], layer_idx: int):
    if past_key_value is not None and len(past_key_value) == layer_idx:
        return True
    return False

def get_past_key_values(module):
    past_key_values = KIVIDynamicCache(module.config.kivi_config) if module.config.kivi_config.packing else KIVIFakeCache(module.config.kivi_config)
    return past_key_values

def quant_kv_output(module, key_states, value_states, attention_mask):
    if hasattr(module.config, 'quant_kv_output') and module.config.quant_kv_output:
        kivi_config = module.config.kivi_config
        key_states = fake_quant(key_states, kivi_config.k_group_size[module.layer_idx], kivi_config.k_bits[module.layer_idx], kivi_config.k_quant_scheme, attention_mask=attention_mask)
        value_states = fake_quant(value_states, kivi_config.v_group_size[module.layer_idx], kivi_config.v_bits[module.layer_idx], kivi_config.v_quant_scheme, attention_mask=attention_mask)

    return key_states, value_states
    
    # # If metric is zeroshot tasks, self.config.quant_kv_output must be False.
    # # It only valid when metric is ppl or loss. (need only one forward pass)
    # if hasattr(module.config, 'quant_kv_output') and module.config.quant_kv_output:
    #     kivi_config = module.config.kivi_config
    #     # quantize key states
    #     if key_states.shape[-2] % kivi_config.residual_length != 0:
    #         if key_states.shape[-2] > kivi_config.residual_length:
    #             key_states_quant = key_states[:, :, :-(key_states.shape[-2] % kivi_config.residual_length), :].contiguous()
    #             key_states_full = key_states[:, :, -(key_states.shape[-2] % kivi_config.residual_length):, :].contiguous()
                
    #             key_states_quant = fake_quant(key_states_quant, kivi_config.k_group_size[module.layer_idx], kivi_config.k_bits[module.layer_idx], kivi_config.k_quant_scheme)
    #             key_states = torch.cat([key_states_quant, key_states_full], dim=-2)
    #     else:
    #         key_states = fake_quant(key_states, kivi_config.k_group_size[module.layer_idx], kivi_config.k_bits[module.layer_idx], kivi_config.k_quant_scheme)

    #     # quantize value states
    #     if value_states.shape[-2] > kivi_config.residual_length:
    #         value_states_quant = value_states[:, :, :-kivi_config.residual_length, :].contiguous()
    #         value_states_full = value_states[:, :, -kivi_config.residual_length:, :].contiguous()

    #         value_states_quant = fake_quant(value_states_quant, kivi_config.v_group_size[module.layer_idx], kivi_config.v_bits[module.layer_idx], kivi_config.v_quant_scheme)
    #         value_states = torch.cat([value_states_quant, value_states_full], dim=-2)

    # return key_states, value_states

def update_causal_mask(module, causal_mask, inputs_embeds, past_key_values):
    # 함수명 변경
    # For KIVI-GEMV, we need a 4D mask during generation. Prepare it here once.
    is_gemv_generation = past_key_values is not None and past_key_values.get_seq_length(0) > 0
    if module.config._attn_implementation == 'flash_attention_2' and is_gemv_generation \
        and causal_mask is not None and causal_mask.dim() == 2 and module.config.kivi_config.packing:

        bsz, seq_len = causal_mask.shape
        query_length = inputs_embeds.shape[1]
        causal_mask = causal_mask[:, None, None, :].expand(bsz, 1, query_length, seq_len)

    return causal_mask

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

    bsz, nh, t, hd = query_states.shape
    kivi_config = module.config.kivi_config

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
        att_qkquant = cuda_bmm_fA_qB_outer(kivi_config.k_group_size[module.layer_idx], query_states, key_states_quant_trans, 
                        key_scale_trans, key_mn_trans, kivi_config.k_bits[module.layer_idx])
        # att_qkquant = triton_bmm_fA_qB_outer(module.k_group_size, query_states, key_states_quant_trans, 
        #                 key_scale_trans, key_mn_trans, module.k_bits)
    else:
        att_qkquant = None

    # calculate full query-key attention
    att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, module.num_key_value_groups).transpose(2, 3))

    # concatenate quantized and full query-key attention
    if att_qkquant is not None:
        attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) * scaling
    else:
        attn_weights = att_qkfull * scaling

    # add attention mask
    if attention_mask is not None and attention_mask.dim() == 4:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    value_full_length = value_states_full.shape[-2]
    if value_states_quant is None:
        attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, module.num_key_value_groups))
    else:
        attn_output = cuda_bmm_fA_qB_outer(kivi_config.v_group_size[module.layer_idx], attn_weights[:, :, :, :-value_full_length], value_states_quant, 
                                        value_scale, value_mn, kivi_config.v_bits[module.layer_idx])
        # attn_output = triton_bmm_fA_qB_outer(module.v_group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant, 
        #                                 value_scale, value_mn, module.v_bits)
        attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], repeat_kv(value_states_full, module.num_key_value_groups))

    attn_output = attn_output.transpose(1, 2).contiguous()

    past_key_value.lazy_update(module.layer_idx)

    return attn_output, attn_weights

def attention_forward(module, past_key_value):
    is_prefill = past_key_value is not None and past_key_value.is_prefill

    if is_prefill or past_key_value is None:
        # prefill stage or no cache
        attention_interface = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
    elif past_key_value is not None and module.config.kivi_config.packing:
        # generation stage and real quantization
        attention_interface = forward_for_kivi_gemv
        # key_states = unpack_and_dequant_kcache(past_key_value.key_states_quant_trans_cache[module.layer_idx].transpose(2, 3).contiguous(), past_key_value.key_scale_trans_cache[module.layer_idx].transpose(2, 3).contiguous(), past_key_value.key_mn_trans_cache[module.layer_idx].transpose(2, 3).contiguous(), module.config.kivi_config.k_group_size[module.layer_idx], module.config.kivi_config.k_bits[module.layer_idx])
        # value_states = unpack_and_dequant_vcache(past_key_value.value_states_quant_cache[module.layer_idx], past_key_value.value_scale_cache[module.layer_idx], past_key_value.value_mn_cache[module.layer_idx], module.config.kivi_config.v_group_size[module.layer_idx], module.config.kivi_config.v_bits[module.layer_idx])
        # key_states_full = past_key_value.key_states_full_cache[module.layer_idx]
        # value_states_full = past_key_value.value_states_full_cache[module.layer_idx]

        # key_states = torch.cat([key_states, key_states_full], dim=-2)
        # value_states = torch.cat([value_states, value_states_full], dim=-2)
        # attention_interface = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
    else:
        # generation stage and fake quantization
        attention_interface = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]

    return is_prefill, attention_interface

def lazy_update(module, is_prefill, past_key_value):
    if not is_prefill and past_key_value is not None and not module.config.kivi_config.packing:
        # import code; code.interact('kiti_utils.py line 167', local=dict(globals(), **locals()))
        past_key_value.lazy_update(module.layer_idx)