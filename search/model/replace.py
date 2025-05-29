from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
# from transformers.models.llama.modeling_qwen import LlamaAttention
from .llama_kivi import LlamaAttentionKIVI, LlamaForCausalLMKIVI, LlamaModelKIVI

def replace_model(model, config):
    if isinstance(model, LlamaForCausalLM):
        # model = LlamaForCausalLMKIVI(model, config)
        model.model = LlamaModelKIVI(model.model, config)
        layers = model.model.layers
        for i in range(len(layers)):
            if type(layers[i]) == LlamaDecoderLayer:
                layers[i].self_attn = LlamaAttentionKIVI(layers[i].self_attn, config)
    return model
