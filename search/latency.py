import json
import torch
import argparse
import numpy as np
import gc
import random

from evaluator import LlamaEvaluator

from utils.eval import measure_latency, eval_metric
from utils.func import init_accelerator, get_net_info

import csv

def main(args):
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    accelerator, device_map = init_accelerator(args.gpu_id, config)

    model_id = f'{args.model_path}/{args.model_name}'
    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        device_map=device_map,
        model_id=model_id,
        method=args.method,
        quant_model_bits=args.quant_model_bits,
        quant_model_paths=args.quant_model_paths,
    )

    n_block = config['n_block']
    arch = {'layer': {'self_attn': [1] * n_block, 'mlp': [1] * n_block}}

    latency_table = None
    if args.latency_table_file:
        with open(args.latency_table_file, 'r') as f:
            latency_table = json.load(f)

    if args.data:
        with open(args.data, 'r') as f:
            archive = json.load(f)['archive']
            archs = [x[0] for x in archive]

    elif args.greedy_result_path and args.last_layer:
        # latency_list = []
        with open(args.greedy_result_path, 'r') as f:
            selected_layers = list(csv.reader(f))[0]

        selected_layers = selected_layers[: selected_layers.index(args.last_layer) + 1]

        for selected_layer in selected_layers:
            blk_idx, layer = selected_layer.split('.')
            blk_idx = int(blk_idx)
            arch['layer'][layer][blk_idx] = 0
            # latency = get_net_info(arch, config, latency_table)['latency']
            # latency_list.append(latency)


    print(f'arch : {arch}, attn : {sum(arch["layer"]["self_attn"])}, mlp : {sum(arch["layer"]["mlp"])}')
    model = evaluator.sample(arch)

    latency = measure_latency(model, generation=True, batch_size=args.batch_size, device=model.device, iteration=args.iteration, max_time=args.max_time)
    accelerator.print(f'latency : {latency}, arch : {arch}')
    print(f"latency table : {get_net_info(arch, config, latency_table)['latency']}")
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--arch_file', type=str, default='',
                        help='file path to measure latency')
    parser.add_argument('--method', type=str, default='',
                        help='')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--quant_model_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--sec_obj', type=str, default='bits',
                        help='second objective to optimize simultaneously')
    parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[2, 4], 
                        help='')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='')
    parser.add_argument('--iteration', type=int, default=10,
                        help='')
    parser.add_argument('--metric', type=str, nargs='+', default=[], 
                        help="'latency', 'ppl'")
    parser.add_argument('--latency_table_file', type=str, default='',
                        help='')
    parser.add_argument('--data', type=str, default='',
                        help='')
    parser.add_argument('--max_time', type=int, default=1e9,
                        help='')
    parser.add_argument('--greedy_result_path', type=str, default='',
                        help='')
    parser.add_argument('--last_layer', type=str, default='',
                        help='')
    
    cfgs = parser.parse_args()
    main(cfgs)

