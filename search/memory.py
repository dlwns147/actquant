import argparse
import gc
import pprint
import numpy as np
import torch
import time
import os
import json

import torch
from transformers import AutoConfig
from utils import init_accelerator, clean_up
from quant.model import get_quantized_model
from evaluator import LlamaEvaluator


benchmark_dtypes = ["int4", torch.float16]


def main(args):


    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    
    accelerator, device_map = init_accelerator(args.gpu_id, config)

    model_id = f'{args.model_path}/{args.model_name}'
    group_size = {'w': args.w_group_size, 'k': args.k_group_size, 'v': args.v_group_size}

    evaluator = LlamaEvaluator(
        config,
        accelerator=accelerator,
        model_id=model_id,
        method=args.method,
        quant_model_paths=args.quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=args.datasets,
        device_map=device_map,
        bits={'w': args.w_bits, 'k': args.k_bits, 'v': args.v_bits},
        group_size=group_size,
        residual_length=args.residual_length,
        use_flash=args.use_flash,
        k_quant_per=args.k_quant_per,
        v_quant_per=args.v_quant_per,
    )

    arch = {
        'w': {
            l: [4] * config['n_layer'] for l in config['linear']
        },
        'k': [4] * config['n_layer'],
        'v': [4] * config['n_layer']
    }

    awq_gptq_owq = 'awq' in args.method or 'gptq' in args.method or 'owq' in args.method
    weight_bits = np.concatenate(list(arch['w'].values()))
    do_owq = ((weight_bits - weight_bits.astype(int)).sum() != 0)
    print(f'do_owq : {do_owq}, awq_gptq_owq : {awq_gptq_owq}')
    if awq_gptq_owq:
        method = 'awq' if 'awq' in args.method else 'gptq' if 'gptq' in args.method else 'owq' if 'owq' in args.method else None
        model_config = AutoConfig.from_pretrained(model_id)
        model = get_quantized_model(method, arch, model_id, device_map, config=config, prune='layer_prune' in args.method, do_owq=do_owq, owq_path=args.outlier_path)
    else:
        model = evaluator.sample(arch)
    del evaluator
    clean_up()
    device = model.device

    torch.cuda.memory._record_memory_history()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
    ) as prof, torch.inference_mode() :
           

        inp = torch.randint(100, 200, (args.batch_size, args.prefill_seq_len), dtype=torch.int32, device=device)
    
        if args.decode_steps > 0:
            with torch.autograd.profiler.record_function("generate"):
                model.generate(inp, min_new_tokens=args.decode_steps, max_new_tokens=args.decode_steps)
        else:
            model(inp)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
    os.makedirs(args.output_path, exist_ok=True)
    file_name = f'{args.model_config.split("/")[-1]}_{args.model_type}_bs{args.batch_size}_pl{args.prefill_seq_len}_gl{args.decode_steps}_{args.attn_implementation}.pkl'
    torch.cuda.memory._dump_snapshot(os.path.join(args.output_path, file_name))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='')
    parser.add_argument('--k_bits', type=int, default=16, 
                        help='')
    parser.add_argument('--v_bits', type=int, default=16, 
                        help='')

    parser.add_argument('--w_group_size', type=int, default=128, 
                        help='')
    parser.add_argument('--k_group_size', type=int, default=128, 
                        help='')
    parser.add_argument('--v_group_size', type=int, default=128, 
                        help='')
    
    parser.add_argument('--residual_length', type=int, default=128, 
                        help='')
    parser.add_argument('--use_flash', action='store_true', help='')

    parser.add_argument('--quant_kv_output', action='store_true', help='')
    parser.add_argument('--k_quant_per', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_per', type=str, choices=['channel', 'token'], 
                        help='')
    
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
    
    parser.add_argument(
        '--model_config', type=str,
        help='',
        required=True,
        default='meta-llama/Llama-2-7b-hf',
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--decode_steps', type=int,
        help='Decode steps',
        required=False,
        default=0,
    )
    parser.add_argument(
        '--num_warmup_steps', type=int,
        help='',
        default=1,
    )
    parser.add_argument(
        '--num_bench_steps', type=int,
        help='',
        default=1,
    )
    # parser.add_argument(
    #     '--attn_implementation', type=str,
    #     help='',
    #     default='flash_attention_2',
    # )
    # parser.add_argument(
    #     '--model_type', type=str, choices=['hf', 'fp16', 'int4'],
    #     help='',
    # )
    parser.add_argument(
        '--output_path', type=str, default='',
        help='',
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)
