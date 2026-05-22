import logging
import random
from argparse import ArgumentParser

import torch

logger = logging.getLogger(__name__)

random.seed(0)

import time
import torch
from hqq.utils.patching_woo import prepare_for_inference
from hqq.models.hf.base import AutoHQQHFModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import gc
import json
import math

from utils.func import get_net_info, getsubattr, setsubattr, getblock, delsubattr
from utils.data import get_loader
from utils.eval import eval_ppl
from accelerate import Accelerator
from tqdm import tqdm
from copy import deepcopy


default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(
    model_name_or_path: str,
    backend: str,
    from_pretrained: bool = False,
    device_map: str = 'cpu',
    use_ft: bool = False,
    **kwargs
):
    start_time = time.time()
    if from_pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype='float16',
            device_map=device_map, 
            low_cpu_mem_usage=True,
            **kwargs
        )

    else:
        model = AutoHQQHFModel.from_quantized(
            model_name_or_path,
            device_map='auto', 
            low_cpu_mem_usage=True
            # torch_dtype='float16'
        )
        prepare_for_inference(model, backend=backend)
        model.to('cpu')
        
    if use_ft:
        from kernel.monkeypatch.ftllama_modeling import convert_model_to_ft
        convert_model_to_ft(model) 
        
    load_time = time.time() - start_time
    print(f'{model_name_or_path} load_time : {int(load_time)} ')
    return model
    
def device_warmup(device: str):
    warm_up = torch.randn((4096, 4096)).to(device)
    for i in range(100):
        torch.mm(warm_up, warm_up)

def gen_inputs(batch_size, prompt_length, device):
    inputs = torch.randint(0, 31999, (batch_size, prompt_length), dtype=torch.long)
    inputs = inputs.to(device).contiguous()
    return inputs

def measure_latency(model, args, device='cuda', prompt_length=64, gen_length=256, batch_size=1, iter=10):
    time_lis = []
    # input_ids = [1 for _ in range(prompt_length)]
    inputs = gen_inputs(batch_size, prompt_length, device)
    # inputs = torch.as_tensor([input_ids], device=device) 
    
    with torch.inference_mode():
        # device_warmup(device)
        if args.use_ft:
            start_pos = 0
            for i in range(128):
                out = model(inputs, start_pos=start_pos, use_cache=False)
                start_pos += out.logits.shape[1]
                token = out.logits[:, -1].max(1)[1].unsqueeze(1)
                inputs = torch.as_tensor([[token]], device=device)
        else:
            last_key_values = None
            for i in range(128):
                out = model(inputs, past_key_values=last_key_values)
                out, last_key_values = out.logits, out.past_key_values                
                token = out[:, -1].max(1)[1].unsqueeze(1)
                inputs = torch.as_tensor([[token]], device=device)
                
        inputs = gen_inputs(batch_size, prompt_length, device)
        torch.cuda.reset_peak_memory_stats()

        if args.only_gemv:
            if args.use_ft:
                start_pos = 0
                for i in tqdm(range(gen_length)):
                    torch.cuda.synchronize()
                    t_st = time.perf_counter()
                    out = model(inputs, start_pos=start_pos, use_cache=False)
                    torch.cuda.synchronize()
                    t_ed = time.perf_counter()
                    start_pos += out.logits.shape[1]

                    time_lis.append(t_ed - t_st)
                    
                    token = out.logits[:, -1].max(1)[1].unsqueeze(1)
                    inputs = torch.as_tensor([[token]], device=device)
                    if args.verbose:
                        print(i, token, np.median(time_lis))

            else:
                last_key_values = None
                for i in tqdm(range(gen_length)):
                    torch.cuda.synchronize()
                    t_st = time.perf_counter()
                    out = model(inputs, past_key_values=last_key_values)
                    torch.cuda.synchronize()
                    t_ed = time.perf_counter()
                    out, last_key_values = out.logits, out.past_key_values

                    time_lis.append(t_ed - t_st)
                    
                    token = out[:, -1].max(1)[1].unsqueeze(1)
                    inputs = torch.as_tensor([[token]], device=device)
                    if args.verbose:
                        print(i, token, np.median(time_lis))

        else:
            for _ in tqdm(range(iter)):
                inputs = gen_inputs(batch_size, prompt_length, device)
                if args.use_ft:
                    start_pos = 0
                    t_st = time.perf_counter()
                    for i in range(gen_length):
                        out = model(inputs, start_pos=start_pos, use_cache=False)
                        start_pos += out.logits.shape[1]
                        token = out.logits[:, -1].max(1)[1].unsqueeze(1)
                        inputs = torch.as_tensor([[token]], device=device)
                    torch.cuda.synchronize()
                    t_ed = time.perf_counter()
                    time_lis.append((t_ed - t_st) / gen_length)

                else:
                    # last_key_values = None
                    t_st = time.perf_counter()
                    # for i in range(gen_length):
                        # if i == 0:
                        #     inputs = torch.as_tensor([input_ids], device=device)
                        # else:
                        #     inputs = torch.as_tensor([[token]], device=device)
                        # out = model(inputs, past_key_values=last_key_values)
                        # out, last_key_values = out.logits, out.past_key_values
                        # token = out[:, -1].max(1)[1].unsqueeze(1)
                        
                    model.generate(inputs, min_new_tokens=gen_length, max_new_tokens=gen_length)
                    torch.cuda.synchronize()
                    t_ed = time.perf_counter()
                    time_lis.append((t_ed - t_st) / gen_length)

    if args.output_file:
        with open(args.output_file, 'a') as f:
            f.write(f"Method : {args.method} | {args.owq} {add_str} Speed: {1 / np.median(time_lis):.2f} tokens per second. ({np.median(time_lis) * 1000:.2f}ms per token)\n")
    else:
        print(f"Max memory usage : {torch.cuda.max_memory_reserved() / 1024 / 1024}MB")
        print(f"Speed: {1 / np.median(time_lis):.2f} tokens per second. ({np.median(time_lis) * 1000:.2f}ms per token)")


def remove_linears(model, config):
    for blk in getblock(model, config):
        for linear_group in config['linear']:
            for linear in linear_group.split(','):
                delsubattr(blk, linear)
    gc.collect()
    torch.cuda.empty_cache()


def sample(model, arch, config, quant_model_bits, quant_models):
    for linear_group, linear_group_bits in arch['linear'].items():
        for blk_idx, bits in enumerate(linear_group_bits):
            flag = False
            for q_bits, q_model in zip(quant_model_bits, quant_models):
                # if math.isclose(bits, q_bits):
                if math.isclose(int(bits), q_bits) and q_bits > 0:
                    for linear in linear_group.split(','):
                        setsubattr(getblock(model, config)[blk_idx], linear, getsubattr(getblock(q_model, config)[blk_idx], linear))
                    flag = True

            # if not math.isclose(bits - int(bits), 0):
            #     for linear in linear_group.split(','):
            #         # insert_fp16_channel_hqq(getsubattr(getblock(self.model, self.config)[blk_idx], linear), self.outlier[f'{self.config["layers"]}.{blk_idx}.{linear}'])
            #         insert_fp16_channel_hqq(getsubattr(getblock(self.model, self.config)[blk_idx], linear), self.outlier[f'{blk_idx}.{linear}'])
            # else:
            #     for linear in linear_group.split(','):
            #         remove_fp16_channel_hqq(getsubattr(getblock(self.model, self.config)[blk_idx], linear))

            if not flag:
                raise NotImplementedError(f'{linear_group}: {linear_group_bits} is not available')
    return model
                

def main(accelerator):
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument('--quant_model_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--backend', type=str, nargs='+', default=[], help='')
    parser.add_argument('--config', type=str, default='config/llama.json', help='')
    parser.add_argument("--prompt_length", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--use_ft", action="store_true")
    parser.add_argument("--output_file", type=str, default='')
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only_gemv", action="store_true")
    parser.add_argument('--dataset', type=str, nargs='+', default=[],
                        help='')
    args = parser.parse_args()
    print(args)

    assert len(args.quant_model_bits) == len(args.quant_model_paths)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    arch_list = [
        {'linear': {k: [4] * config['n_block'] for k in config['linear']}},
        {'linear': {k: [3] * config['n_block'] for k in config['linear']}},
        {'linear': {k: [2] * config['n_block'] for k in config['linear']}},
    ]
    
    model_id = f'{args.model_path}/{args.model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    if args.dataset:
        loaders = {dataset: accelerator.prepare(get_loader(dataset, train=False, tokenizer=tokenizer)) for dataset in args.dataset}
    
    model = load_model(
        model_name_or_path=model_id,
        from_pretrained=True,
        use_ft=args.use_ft,
        device_map='auto',
        backend=''
    )
    
    # measure_latency(model, args, device, prompt_length=args.prompt_length, gen_length=args.gen_length, iter=args.iter, batch_size=args.batch_size)

    # if args.dataset:
    #     for dataset in args.dataset:
    #         ppl = eval_ppl(model, accelerator, loaders[dataset])
    #         print(f'{dataset} ppl : {ppl}')

    remove_linears(model, config)
    gc.collect()
    torch.cuda.empty_cache()

    quant_models = [load_model(model_name_or_path=q_model, backend=args.backend[i]) for i, q_model in enumerate(args.quant_model_paths)]

    for arch in arch_list:
        complexity = get_net_info(arch, config)
        print(f'bits : {complexity["bits"]}')
        model.to('cpu')
        for m in quant_models:
            m.to('cpu')
        model = sample(model, arch, config, args.quant_model_bits, quant_models)
        model.to(device)
        
        if args.dataset:
            for dataset in args.dataset:
                ppl = eval_ppl(model, accelerator, loaders[dataset])
                print(f'{dataset} ppl : {ppl}')
        gc.collect()
        torch.cuda.empty_cache()

        measure_latency(model, args, device, prompt_length=args.prompt_length, gen_length=args.gen_length, iter=args.iter, batch_size=args.batch_size)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    accelerator = Accelerator()
    main(accelerator)
