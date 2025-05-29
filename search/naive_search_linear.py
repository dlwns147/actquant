import argparse
from time import time
import csv
import torch
import numpy as np

import gc
import json

from evaluator import LlamaEvaluator
from utils.func_utils import get_net_info

def naive_search_linear(args):

    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # device = torch.device("cuda:0")
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    evaluator = LlamaEvaluator(
        config=config,
        quant_method=args.quant_method,
        model_name=args.model_name,
        large_model_path=args.large_model_path,
        large_model_bits=args.large_model_bits,
        small_model_path=args.small_model_path,
        small_model_bits=args.small_model_bits,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=[args.dataset]
    )
    
    # replaced_linear_list = []
    n_block = config['n_block']
    
    arch = {l: [] for l in config['linear']}
    
    for blk_idx in range(n_block):
        for linear in config['linear']:
            key = f'{blk_idx}.{linear}'
            arch[linear].append(args.large_model_bits)


    with open(args.linear_sensitivity, 'r') as f:
        data = list(csv.reader(f))
        linear_list = data[0]
        linear_sensitivity = data[1]
        idx_list = np.argsort(linear_sensitivity)
    linear_list = [linear_list[i] for i in idx_list]

    cur_linear_list = list()
    ppl_list = list()
    loss_list = list()
    bits_list = list()
    total_time_list = list()
    ppl = 0

    total_time_start = time()
    for i, blk_linear in enumerate(linear_list):
        linear_time_start = time()

        blk_idx, linear = blk_linear.split('.', 1)
        arch[linear][int(blk_idx)] = args.small_model_bits

        ppl, _ = evaluator.eval(arch, metric='ppl')
        ppl = ppl[args.dataset]
        # loss, _ = evaluator.eval(arch, metric='loss')
        # loss = loss[args.dataset]
        cur_bit = get_net_info(arch, config)['bits']

        linear_time = time() - linear_time_start
        total_time = time() - total_time_start

        bits_list.append(cur_bit)
        ppl_list.append(ppl)
        # loss_list.append(loss)
        total_time_list.append(total_time)
        cur_linear_list.append(blk_linear)
        print(f"Phase {i}, current bit: {cur_bit:.2f}, ppl : {ppl:.2f}, [{blk_idx}.{linear} replaced] iter time: {linear_time:.2f}s, total time : {total_time:.2f}s") 

        if args.ppl_csv_file:
            with open(args.ppl_csv_file, 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(cur_linear_list)
                write.writerow(bits_list)
                write.writerow(ppl_list)
                # write.writerow(loss_list)
                write.writerow(total_time_list)

    finish_point = time()
    time_elapsed = finish_point - total_time_start

    print(f"Time_Elapsed: {time_elapsed}")
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--quant_method', type=str, default='',
                        help='')
    parser.add_argument('--large_model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--large_model_bits', type=float, default=4,
                        help='test batch size for inference')
    parser.add_argument('--small_model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--small_model_bits', type=float, default=2,
                        help='test batch size for inference')
    parser.add_argument('--target_bit', type=float, default=3,
                        help='')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='test batch size for inference')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--ppl_csv_file', type=str, default='',
                        help='')
    parser.add_argument('--linear_sensitivity', type=str, default='',
                        help='')

    cfgs = parser.parse_args()
    naive_search_linear(cfgs)

