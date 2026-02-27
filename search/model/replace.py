from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2DecoderLayer
from .KIVICache import KIVICacheConfig
from .generation import convert_generation
# from utils.data import get_tokenizer


def replace_kv_cache(model,
                    tokenizer,
                    method=['kivi'],
                    n_block=-1,
                    k_quant_scheme='channel',
                    v_quant_scheme='token',
                    residual_length=128,
                    packing=False,
                    quant_kv_output=False):
                    
    if 'hqq' in method:
        model.config.cache_implementation = 'HQQ'
        model.generation_config.cache_implementation = "quantized"
        model.generation_config.cache_config = {
            'backend': 'HQQ',
            'k_bits': [16] * n_block,
            'v_bits': [16] * n_block,
            'k_group_size': [0] * n_block,
            'v_group_size': [0] * n_block,  

            'k_quant_scheme': k_quant_scheme,
            'v_quant_scheme': v_quant_scheme,
            'residual_length': residual_length,
        }
    elif 'kivi' in method:
        model.config.kivi_config = KIVICacheConfig(
            k_bits=[16] * n_block,
            v_bits=[16] * n_block,
            k_group_size=[0] * n_block,
            v_group_size=[0] * n_block,
            k_quant_scheme=k_quant_scheme,
            v_quant_scheme=v_quant_scheme,
            residual_length=residual_length,
            packing=packing,
        )
        # Store kv_method as list for downstream use (e.g. generation)
        model.config.kv_method = method
        if "think" in method:
            print(f'ThinK KiVi model')
            if isinstance(model, LlamaForCausalLM):
                from .llama_kivi_think import convert_model_think_kivi
                convert_model_think_kivi(model, method)
            else:
                raise NotImplementedError(f"Think_kivi not implemented for {model.__class__}")
        else:
            print(f'KiVi model')
            if isinstance(model, Qwen2ForCausalLM):
                from .qwen2_kivi import convert_model_kivi
                convert_model_kivi(model)
            elif isinstance(model, LlamaForCausalLM):
                from .llama_kivi import convert_model_kivi
                convert_model_kivi(model)
            elif isinstance(model, MistralForCausalLM):
                from .mistral_kivi import convert_model_kivi
                convert_model_kivi(model)
            else:
                raise NotImplementedError(f"Unsupported model: {model.__class__}")
        # model.config.use_cache = use_cache
        model.config.quant_kv_output = quant_kv_output
        convert_generation(model.config)

        # tokenizer = get_tokenizer(self.model_id)
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    return model

def set_cache_config(model,
                    method='kivi',
                    arch=None,
                    n_block=-1,
                    k_quant_scheme='channel',
                    v_quant_scheme='token',
                    residual_length=128,
                    packing=False,
                    quant_kv_output=False):
    
    if 'hqq' in method:
        model.config.cache_implementation = 'HQQ'
        model.generation_config.cache_implementation = "quantized"
        model.generation_config.cache_config = {
            'backend': 'HQQ',
            'k_bits': [16] * n_block,
            'v_bits': [16] * n_block,
            'k_group_size': [0] * n_block,
            'v_group_size': [0] * n_block,  

            'k_quant_scheme': k_quant_scheme,
            'v_quant_scheme': v_quant_scheme,
            'residual_length': residual_length,
        }
    elif 'kivi' in method:
        model.config.kivi_config = KIVICacheConfig(
            k_bits=[16] * n_block,
            v_bits=[16] * n_block,
            k_group_size=[0] * n_block,
            v_group_size=[0] * n_block,
            k_quant_scheme=k_quant_scheme,
            v_quant_scheme=v_quant_scheme,
            residual_length=residual_length,
            packing=packing,
        )
    else:
        raise NotImplementedError(f"Unsupported kv cache method: {method}")
    model.config.quant_kv_output = quant_kv_output
    return model