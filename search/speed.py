import os
import json
import time
import contextlib
import random
import logging
import gc
import argparse
import math

from pathlib import Path
from typing import Optional, Tuple
from glob import glob
from copy import deepcopy
from tqdm import tqdm

import torch
# import torchao
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaForCausalLM
# from fastchat.llm_judge.common import load_questions
# from fastchat.model import get_conversation_template
# from torchao._models.llama.model import Transformer, prepare_inputs_for_model, ModelArgs

# from hqq.core.quantize import BaseQuantizeConfig
from hqq.utils.patching_woo import prepare_for_inference
from hqq.models.hf.base import AutoHQQHFModel
from hqq.backends.autogptq import GPTQLinear
from hqq.backends.bitblas import HQQLinearBitBlas
from hqq.core.quantize import HQQLinear
from monkeypatch.ftllama_modeling import convert_model_to_ft
from monkeypatch.ftllama_generate import replace_generate_functions

# from get_eval import get_eval
from data_utils import get_loader
from benchmark_inference_latency import profile_model
# from manage_json import init_json, write_json

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['HF_HOME']='/SSD/Woo'

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = None

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_proxy(model, int2_layers, int3_layers, int4_layers, base_speed = None):
    # if model.device == default_device:
    model = model.to('cpu')     ## We expect the default_devide to be 'cuda'
    
    model_layers = model.model.layers

    proxy_model = deepcopy(model)
    proxy_model = proxy_model.to(default_device)

    proxy_result = {}

    cleanup()
    
    if base_speed is not None:
        device_sync(default_device)
        start = time.perf_counter()
        token_per_second = profile_model(proxy_model, generation = True)
        device_sync(default_device)
        end = time.perf_counter()

        proxy_result['base'] = token_per_second
    else:
        proxy_result['base'] = base_speed

    print("Get Proxy...")
    for name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']:
        module, linear = name.split('.')

        proxy_result[name] = {}
        
        ori_linear = []
        for i in range(len(model_layers)):
            ori_linear.append(getattr(getattr(proxy_model.model.layers[i], module), linear))

        for bit in [2, 3, 4]:
            layers = (None, None, int2_layers, int3_layers, int4_layers)[bit]

            for i in range(len(model_layers)):
                source = getattr(getattr(layers[i], module), linear)

                setattr(getattr(proxy_model.model.layers[i], module), linear, source.to(default_device))

            device_sync(default_device)
            start = time.perf_counter()
            token_per_second = profile_model(proxy_model, generation = True)
            device_sync(default_device)
            end = time.perf_counter()

            proxy_result[name][bit] = token_per_second

        for i in range(len(model_layers)):
            setattr(getattr(proxy_model.model.layers[i], module), linear, ori_linear[i])

        cleanup()

    proxy_model = proxy_model.to('cpu')
    del proxy_model
    cleanup()

    model = model.to(default_device)

    return proxy_result


def device_sync(default_device):
    if "cuda" in default_device:
        torch.cuda.synchronize(default_device)
    elif ("cpu" in default_device) or ("mps" in default_device):
        pass
    else:
        print(f"device={default_device} is not yet suppported")


def get_named_linears(module, specific_cls=None):
    if specific_cls is None:
        return {name: m for name, m in module.named_modules() if (isinstance(m, nn.Linear) 
                                                              or isinstance(m, GPTQLinear)
                                                              or isinstance(m, HQQLinearBitBlas))
                                                              or isinstance(m, HQQLinear)}
    else:
        return {name: m for name, m in module.named_modules() if isinstance(m, specific_cls)}


def get_conv_prompt(text, tokenizer):
    ## TODO : tokenizer.apply_chat_template()는 무엇?
    conv = get_conversation_template(model_name)
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return tokenizer(prompt, return_tensors='pt').to(default_device)


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def warm_up_model(model, tokenizer, generation_config):
    warm_up_text = "Hello, how are you?"
    inputs = get_conv_prompt(warm_up_text, tokenizer)

    start = time.perf_counter()
    
    # 2-3회 워밍업 실행
    while time.perf_counter() - start < 5:      ## 5초 동안 워밍업
        with torch.no_grad():
            _ = model.generate(**inputs, generation_config=generation_config)
            
    # GPU 캐시 동기화
    device_sync(default_device)
    cleanup()


def get_hfmodel(model_name_or_path: str,
                dtype='auto',
                device_map='cpu',
                trust_remote_code=False,
                **kwargs
                ):

    assert kwargs.get('attn_implementation') in ['hf', 'ft']        ## hf : huggingface, ft : faster transformer
    
    # for fast model loading
    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    ft = False
    if kwargs.get('attn_implementation') == 'ft':
        assert 'llama' in model_name_or_path.lower() or 'vicuna' in model_name_or_path.lower()
        ft = True
    
    print('attention implementaion is :', kwargs.pop('attn_implementation'))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=dtype,
        device_map=device_map, 
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    if ft:
        convert_model_to_ft(model)
        model = model.to(torch.half).to(device) # half
        replace_generate_functions()

    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal
    
    return model


def get_memory_footprint(module: torch.nn.Module, return_buffers: bool = True) -> int:
    if not isinstance(module, torch.nn.Module):
        raise TypeError("Input must be a PyTorch Module")
    mem = sum([param.nelement() * param.element_size() for param in module.parameters()])
    if return_buffers:
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in module.buffers()])
        mem = mem + mem_bufs
    return mem


@torch.no_grad()
def hf_generation_speed(model, tokenizer, examples, generation_config):
    output_tokens_list = []
    generation_time_list = []
    num_generated_tokens_list = []

    num_samples = len(examples)
    start = 0
    progress_bar = tqdm(range(start, num_samples))

    for i in progress_bar:
        random.seed(42)
        torch.manual_seed(42)
        inputs = get_conv_prompt(examples[i]["turns"][0], tokenizer)
        
        device_sync(default_device)
        start = time.perf_counter()

        output_ids = model.generate(**inputs, generation_config = generation_config)

        device_sync(default_device)
        end = time.perf_counter()

        output_tokens_list.append(output_ids[0])
        generation_time_list.append(end - start)
        num_generated_tokens = 0
        num_generated_tokens += len(
            [token_id for token_id in output_ids[0][inputs['input_ids'].numel() :] if token_id != tokenizer.pad_token_id]
            )
        num_generated_tokens_list.append(num_generated_tokens)

        output_text = tokenizer.decode(output_ids[0])
        progress_bar.set_postfix(
            num_tokens=num_generated_tokens_list[-1],
            time=generation_time_list[-1],
            speed=f"{num_generated_tokens_list[-1] / generation_time_list[-1]:.3f} tokens/s",
        )

    total_tokens = sum(num_generated_tokens_list)
    total_seconds = sum(generation_time_list)
    print(
        f"generated {total_tokens} tokens using {total_seconds:.3f} seconds, "
        f"generation speed: {total_tokens / total_seconds:.3f} tokens/s"
    )

    return total_tokens, total_seconds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name_or_path', type=str, help='model path')
    parser.add_argument('--use_ft', action='store_true', help='use faster transformer')
    parser.add_argument('--use_owq', action='store_true', help='use owq')

    parser.add_argument('--backend_2bit', type=str, choices=['gptq', 'bitblas'], help='backend for 2bit', default = 'gptq')
    parser.add_argument('--backend_3bit', type=str, choices=['gptq', 'bitblas'], help='backend for 3bit', default = 'gptq')
    parser.add_argument('--backend_4bit', type=str, choices=['gptq', 'bitblas'], help='backend for 4bit', default = 'gptq')

    parser.add_argument('--arch_path', type=str, help='arch path', default = None)

    parser.add_argument('--num_samples', type=int, help='number of samples', default = 3)
    parser.add_argument('--num_beams', type=int, help='number of beams', default = 1)
    parser.add_argument('--do_sample', action='store_true', help='do sample')
    parser.add_argument('--max_new_tokens', type=int, help='max new tokens', default = 512)
    parser.add_argument('--temperature', type=float, help='temperature', default = 1.0)

    parser.add_argument('--ppl', action='store_true', help='evaluation ppl')
    parser.add_argument('--ppl_dataset', type=str, help='ppl dataset', default = 'wikitext2')

    parser.add_argument('--latency', action='store_true', help='latency')
    parser.add_argument('--token_per_second', action='store_true', help='token per second')

    parser.add_argument('--use_proxy', action='store_true', help='use proxy')

    parser.add_argument('--file_name', type=str, help='save path', default = None)

    args = parser.parse_args()

    global model_name
    model_name = args.model_name_or_path
    model_path, model_id = model_name.split('/')

    owq_path = f"/NAS/Woo/Automation/autoopt/kernel/outlier/{model_id}/w16_r32/outlier.pth"
    owq = torch.load(owq_path, weights_only = True) if os.path.exists(owq_path) else None

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        use_fast=False,
        trust_remote_code=True
        )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # examples = load_questions(glob('**/mtbench_question.jsonl', recursive=True)[0], 0, args.num_samples) # single question

    int2_model = AutoHQQHFModel.from_quantized(f'/SSD/Woo/hqq/{model_id}_2bit_128gs_1axis')
    int3_model = AutoHQQHFModel.from_quantized(f'/SSD/Woo/hqq/{model_id}_3bit_128gs_1axis')
    int4_model = AutoHQQHFModel.from_quantized(f'/SSD/Woo/hqq/{model_id}_4bit_128gs_1axis')

    assert args.backend_3bit == 'gptq', '3bit only support gptq backend'
    prepare_for_inference(int2_model, backend = args.backend_2bit, load_path = f"/SSD/Woo/hqq/{model_id}_2bit_128gs_1axis_GPTQLinear.pt")
    prepare_for_inference(int3_model, backend = args.backend_3bit, load_path = f"/SSD/Woo/hqq/{model_id}_3bit_128gs_1axis_GPTQLinear.pt")
    prepare_for_inference(int4_model, backend = args.backend_4bit, load_path = f"/SSD/Woo/hqq/{model_id}_4bit_128gs_1axis_GPTQLinear.pt")

    int2_layers = int2_model.model.layers
    int3_layers = int3_model.model.layers
    int4_layers = int4_model.model.layers

    # import code; code.interact("quantized model test", local = dict(globals(), **locals()))

    int2_model = int2_model.to('cpu')
    int3_model = int3_model.to('cpu')
    int4_model = int4_model.to('cpu')

    # del int2_model
    # del int3_model
    # del int4_model

    if args.use_owq:
        int2_owq_model = deepcopy(int2_model)
        int3_owq_model = deepcopy(int3_model)
        int4_owq_model = deepcopy(int4_model)

        intN_models = [int2_owq_model, int3_owq_model, int4_owq_model]
        for intN_model in intN_models:
            linears = get_named_linears(intN_model, specific_cls=HQQLinear)
            for name, module in linears.items():
                module.name = name

                if 'o_proj' in name:
                    continue

                owq_layer = owq[name]
                module.do_owq = False
                module.register_buffer('oweight', torch.zeros((len(owq_layer), module.out_features), dtype=torch.float))
                module.register_buffer('outlieridx', torch.Tensor(owq_layer).to(torch.int))

        prepare_for_inference(int2_owq_model, backend = args.backend_2bit, load_path = f"/SSD/Woo/hqq/{model_id}_2bit_64gs_1axis_GPTQLinear_owq.pt", owq = True)
        prepare_for_inference(int3_owq_model, backend = args.backend_3bit, load_path = f"/SSD/Woo/hqq/{model_id}_3bit_128gs_1axis_GPTQLinear_owq.pt", owq = True)
        prepare_for_inference(int4_owq_model, backend = args.backend_4bit, load_path = f"/SSD/Woo/hqq/{model_id}_4bit_128gs_1axis_GPTQLinear_owq.pt", owq = True)

        int2_owq_layers = int2_owq_model.model.layers
        int3_owq_layers = int3_owq_model.model.layers
        int4_owq_layers = int4_owq_model.model.layers

        del int2_owq_model
        del int3_owq_model
        del int4_owq_model

    cleanup()

    base_model = get_hfmodel(model_name, dtype='float16', attn_implementation='ft' if args.use_ft else 'hf')
    base_model_memory = get_memory_footprint(base_model) / 1024 / 1024 / 1024
    base_layers = base_model.model.layers

    generation_config = GenerationConfig(
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            do_sample=args.do_sample,
            min_new_tokens=args.max_new_tokens,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            temperature=args.temperature,
        )

    base_model.eval()
    base_model = base_model.to('cuda')

    if args.ppl:
        print(f"Download {args.ppl_dataset}..., if you want to evaluate other dataset, please change the dataset name")
        test_loaders = {args.ppl_dataset : get_loader(args.ppl_dataset, tokenizer = tokenizer)}

        print(f"Evaluate {args.ppl_dataset} PPL...")
        metric_ppl = get_eval_wData(base_model, None, test_loaders=test_loaders, metrices = ['ppl'])

        print(f'{args.ppl_dataset} ppl : {metric_ppl["ppl"][args.ppl_dataset]}')

    if args.latency:
        print(f"Get Latency...")
        latency = profile_model(base_model, False)
        print('latency : ', *latency, sep='\n')

    if args.token_per_second:
        print(f"Get Token per second...")
        token_per_second = profile_model(base_model, True)
        print('Token per second : ', token_per_second)

    if args.use_proxy:
        print(f"Get Proxy...")
        proxy = get_proxy(base_model, int2_layers, int3_layers, int4_layers, base_speed = token_per_second)

    if args.file_name:
        result_path = init_json(args, '/NAS/Woo/Automation/autoopt/kernel', args.file_name, model_id, algorithm = 'replace', dataset = 'dummy', metric = 'token_per_second')

        if args.use_proxy:
            write_json(result_path, {
                'base' : token_per_second,
                'proxy' : proxy,
            })
        else:
            write_json(result_path, {
                'base' : token_per_second,
            })

    base_model = base_model.to('cpu')

    arch_path = args.arch_path if args.arch_path else "/NAS/Woo/Automation/autoopt/archs/HQQ_woPrior_random_linear_wINT3_meta-llama_Llama-2-7b-hf.json"
    with open(arch_path, 'r') as f:
        data = json.load(f)
        archive = data['archive']

    for i, archs in enumerate(archive):
        print("Arch Index : ", i)

        arch = archs['arch']
        bit = archs['bit']

        model = deepcopy(base_model)

        print("Replacing...")
        for layer_idx, layer in enumerate(base_layers):
            named_linears = get_named_linears(layer)
            for name in named_linears:
                module, linear = name.split('.')

                if math.isclose(arch[name][layer_idx], 2):
                    source = getattr(getattr(int2_layers[layer_idx], module), linear)
                elif math.isclose(arch[name][layer_idx], 3):
                    source = getattr(getattr(int3_layers[layer_idx], module), linear)
                elif math.isclose(arch[name][layer_idx], 4):
                    source = getattr(getattr(int4_layers[layer_idx], module), linear)
                elif args.use_owq and math.isclose(int(arch[name][layer_idx]), 2):
                    source = getattr(getattr(int2_owq_layers[layer_idx], module), linear)
                elif args.use_owq and math.isclose(int(arch[name][layer_idx]), 3):
                    source = getattr(getattr(int3_owq_layers[layer_idx], module), linear)
                elif args.use_owq and math.isclose(int(arch[name][layer_idx]), 4):
                    source = getattr(getattr(int4_owq_layers[layer_idx], module), linear)
                else:
                    raise ValueError(f'bit should be 2, 3, 4, but got {arch[name][layer_idx]}')

                setattr(getattr(model.model.layers[layer_idx], module), linear, source.to('cuda'))

                quantized_model_memory = get_memory_footprint(model) / 1024 / 1024 / 1024

        print(f"Base Model Memory : {base_model_memory} GB\nQuantized Model Memory : {quantized_model_memory} GB")
        print(f"Memory Saving : {1 - quantized_model_memory / base_model_memory} %")

        model.eval()
        model = model.to('cuda')

        if args.ppl:
            print(f"Evaluate {args.ppl_dataset} PPL...")
            metric_ppl = get_eval_wData(model, None, test_loaders=test_loaders, metrices = ['ppl'])

            print(f'{args.ppl_dataset} ppl : {metric_ppl["ppl"][args.ppl_dataset]}')

        if args.latency:
            print(f"Get Latency...")
            latency = profile_model(model, False)
            print('latency : ', *latency, sep='\n')

        if args.token_per_second:
            print(f"Get Token per second...")
            token_per_second = profile_model(model, True)
            print('Token per second : ', token_per_second)

        if args.use_proxy:
            base_proxy_speed_predict = {}
            for batch_size, seq_len in [(1, 64), (1, 128)]:
                base_proxy_speed = proxy['base'][f'{batch_size}.{seq_len}']['median']
                proxy_sum = 0

                for name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']:
                    module, linear = name.split('.')
                    for idx in range(len(base_layers)):
                        bit = int(arch[name][idx])
                        proxy_sum += (proxy[name][bit][f'{batch_size}.{seq_len}']['median'] - base_proxy_speed)

                proxy_sum = proxy_sum / len(base_layers)

                base_proxy_speed_predict[f'{batch_size}.{seq_len}'] = base_proxy_speed + proxy_sum
        
            print("Base Proxy Speed : ", base_proxy_speed_predict)


        if args.file_name:
            if args.use_proxy:
                write_json(result_path, {
                f'arch_{i}_actual' : token_per_second,
                f'arch_{i}_proxy' : base_proxy_speed_predict,
                })
            else:
                write_json(result_path, {
                f'arch_{i}_actual' : token_per_second,
                })

        # print(f"Base Model Speed : {base_total_token / base_total_time} tokens/s\nQuantized Model Speed : {quantized_total_token / quantized_total_time} tokens/s")
        # print(f"Speedup : {(quantized_total_token / quantized_total_time) / (base_total_token / base_total_time)}")

        model = model.cpu()
        del model
        cleanup()


if __name__ == '__main__':
    main()
