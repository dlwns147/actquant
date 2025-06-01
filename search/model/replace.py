from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2DecoderLayer
# from transformers.models.llama.modeling_qwen import LlamaAttention
from .llama_kivi import LlamaKIVIAttention, LlamaKIVIForCausalLM, LlamaKIVIModel
from .mistral_kivi import MistralKIVIAttention, MistralKIVIForCausalLM, MistralKIVIModel
from .qwen2_kivi import Qwen2KIVIAttention, Qwen2KIVIForCausalLM, Qwen2KIVIModel


def replace_model(model, config):
    if isinstance(model, LlamaForCausalLM):
        # model = LlamaKIVIForCausalLM(model, config)
        model.model = LlamaKIVIModel(model.model, config)
        layers = model.model.layers
        for i in range(len(layers)):
            # if type(layers[i]) == LlamaDecoderLayer:
            layers[i].self_attn = LlamaKIVIAttention(layers[i].self_attn, config)

    elif isinstance(model, MistralForCausalLM):
        # model = MistralKIVIForCausalLM(model, config)
        model.model = MistralKIVIModel(model.model, config)
        layers = model.model.layers
        for i in range(len(layers)):
            # if type(layers[i]) == LlamaDecoderLayer:
            layers[i].self_attn = MistralKIVIAttention(layers[i].self_attn, config)
    
    elif isinstance(model, Qwen2ForCausalLM):
        # model = Qwen2KIVIForCausalLM(model, config)
        model.model = Qwen2KIVIModel(model.model, config)
        layers = model.model.layers
        for i in range(len(layers)):
            # if type(layers[i]) == LlamaDecoderLayer:
            layers[i].self_attn = Qwen2KIVIAttention(layers[i].self_attn, config)
    else:
        raise NotImplementedError
    
    return model
