import argparse
import time
# from tqdm import tqdm
import csv
import os

# import numpy as np
import torch
# import gc
import json
import csv
import time
# from accelerate import Accelerator

from evaluator import LlamaEvaluator
from utils.func import init_accelerator, set_seed
# from utils.eval import load_and_eval_ppl, eval_zeroshot
# from transformers import AutoModelForCausalLM
import warnings
warnings.simplefilter("ignore")

def sensitivity(args):
    set_seed(args.seed)
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)

    evaluator = LlamaEvaluator(
        config,
        accelerator=accelerator,
        model_id=f'{args.model_path}/{args.model_name}',
        method={'w': args.w_method, 'kv': args.kv_method},
        # quant_model_bits=self.quant_model_bits,
        quant_model_paths=args.quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=args.seqlen,
        min_seqlen=args.min_seqlen,
        n_sample=args.n_sample,
        data_batch_size=args.data_batch_size,
        datasets=[args.dataset],
        loss_func=args.loss_func,
        device_map=device_map,
        bits={'w': args.w_bits, 'k': args.k_bits, 'v': args.v_bits},
        group_size={'w': args.w_group_size, 'k': args.k_group_size, 'v': args.v_group_size},
        residual_length=args.residual_length,
        # use_flash=args.use_flash,
        quant_kv_output=args.quant_kv_output,
        k_quant_scheme=args.k_quant_scheme,
        v_quant_scheme=args.v_quant_scheme,
        use_key_token=args.use_key_token,
        trunc_len=args.trunc_len,
        sliding_window=args.sliding_window,
        alpha=args.alpha,
        beta=args.beta,
        packing=args.packing
    )
    
    n_block = config['n_block']
    ppl = 0
    loss_list = dict()
    ppl_list = dict()
    # arch = {'linear': {l: [max(args.quant_model_bits)] * n_block for lg in config['linear'] for l in lg.split(',')}, 'layer': {l: [1]* n_block for l in config['layer']}}
    arch = {
        'w': {l: [max(args.w_bits)] * n_block for lg in config['linear'] for l in lg.split(',')},
        'k': [[max(args.k_bits), min(args.k_group_size[-1])]] * n_block,
        'v': [[max(args.v_bits), min(args.v_group_size[-1])]] * n_block,
    }
    
    for target in args.target:
        if target == 'w':
            ppl_list[target] = dict()
            for linear in config['linear']:
                for block_idx in range(n_block):
                    ppl_list[target][f'{block_idx}.{linear}'] = 0
        else:
            ppl_list[target] = dict()
            for i in range(n_block):
                ppl_list[target][str(i)] = 0
    # accelerator.print(f'arch : {arch}')

    start_point = time.time()
    for target in args.target:
        loss_list[target] = dict()
        for block_idx in range(n_block):
            if target == 'w':
                for linear in config['linear']:
                    iter_start = time.time()
                    
                    # for linear in linear_group.split(','):
                    arch[target][linear][block_idx] = min(args.w_bits)

                    key = f'{block_idx}.{linear}'
                    loss, _ = evaluator.eval(accelerator=accelerator, arch=arch, metric='loss', loss_func=args.loss_func)
                    loss_list[target][key] = loss[args.dataset]
                    if args.eval_ppl:
                        ppl, _ = evaluator.eval(accelerator=accelerator, arch=arch, metric='ppl', loss_func=args.loss_func)
                        ppl_list[target][key] = ppl[args.dataset]
                    iter_time = time.time() - iter_start
                    accelerator.print(f"[{target} {key} replaced] Loss={loss_list[target][key]:.4f}, PPL={ppl_list[target][key]:.2f}, time: {iter_time:.2f}")
                    
                    # for linear in linear_group.split(','):
                    arch[target][linear][block_idx] = max(args.w_bits)
            else:
                iter_start = time.time()
                arch[target][block_idx] = [min(getattr(args, f'{target}_bits')), max(getattr(args, f'{target}_group_size')[0])]
                
                key = str(block_idx)
                loss, _ = evaluator.eval(accelerator=accelerator, arch=arch, metric='loss', loss_func=args.loss_func)
                loss_list[target][key] = loss[args.dataset]
                if args.eval_ppl:
                    ppl, _ = evaluator.eval(accelerator=accelerator, arch=arch, metric='ppl', loss_func=args.loss_func)
                    ppl_list[target][key] = ppl[args.dataset]
                iter_time = time.time() - iter_start
                accelerator.print(f"[{target} {key} replaced] Loss={loss_list[target][key]:.4f}, PPL={ppl_list[target][key]:.2f}, time: {iter_time:.2f}")
                    
                arch[target][block_idx] = [max(getattr(args, f'{target}_bits')), min(getattr(args, f'{target}_group_size')[-1])]
            
        if accelerator.is_main_process:
            if args.result_path:
                loss_result_path = os.path.join(args.result_path, 'loss')
                os.makedirs(loss_result_path, exist_ok=True)
                save_path = os.path.join(loss_result_path, f'{target}.csv')
                with open(save_path, 'w', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(list(loss_list[target].keys()))
                    write.writerow(list(loss_list[target].values()))
                    
                if args.eval_ppl:
                    ppl_result_path = os.path.join(args.result_path, 'ppl')
                    os.makedirs(ppl_result_path, exist_ok=True)
                    save_path = os.path.join(ppl_result_path, f'{target}.csv')
                    with open(save_path, 'w', newline='') as f:
                        write = csv.writer(f)
                        write.writerow(list(ppl_list[target].keys()))
                        write.writerow(list(ppl_list[target].values()))
        accelerator.wait_for_everyone()

    finish_point = time.time()
    time_elapsed = finish_point - start_point
    
    accelerator.print(f"Time_Elapsed: {time_elapsed}")
    accelerator.print(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--target', type=str, nargs='+', default=['w', 'k', 'v'], choices=['w', 'k', 'v'], 
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--w_bits', type=int, nargs='+', default=[], 
                        help='')
    parser.add_argument('--k_bits', type=int, nargs='+', default=[2, 4], 
                        help='')
    parser.add_argument('--v_bits', type=int, nargs='+', default=[2, 4], 
                        help='')
    # parser.add_argument('--w_group_size', type=int, default=128, 
    #                     help='')
    # parser.add_argument('--k_group_size', type=int, default=128, 
    #                     help='')
    # parser.add_argument('--v_group_size', type=int, default=128, 
    #                     help='')  
    parser.add_argument('--w_group_size', type=int, default=128,
                        help='')
    parser.add_argument('--k_group_size', type=int, nargs='+', action='append', default=[],
                        help='')
    parser.add_argument('--v_group_size', type=int, nargs='+', action='append', default=[],
                        help='')
    parser.add_argument('--residual_length', type=int, default=128, 
                        help='')
    # parser.add_argument('--use_flash', action='store_true', help='')
    parser.add_argument('--quant_kv_output', action='store_true', help='')
    parser.add_argument('--k_quant_scheme', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_scheme', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--data_batch_size', type=int, default=1,
                        help='sequential length of the calibaration (train) set')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='test batch size for inference')
    parser.add_argument('--min_seqlen', type=int, default=0,
                        help='minimum sequential length of the calibaration gsm8k set')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--result_path', type=str, default='',
                        help='')
    parser.add_argument('--w_method', type=str, nargs='+', default=[], choices=['fp16', 'awq', 'gptq', 'qeft', 'hqq'],
                        help='')
    parser.add_argument('--kv_method', type=str, default='kivi', choices=['hqq', 'kivi'],
                        help='')
    parser.add_argument('--eval_ppl', action='store_true')
    parser.add_argument('--loss_func', type=str, default='cross_entropy', help='')
    parser.add_argument('--outlier_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
        
    parser.add_argument('--use_key_token', action='store_true', help='Only use key tokens for loss calculation (Long PPL/JSD)')
    parser.add_argument('--trunc_len', type=int, default=512, 
                        help='truncation length for long PPL/JSD calculation')
    parser.add_argument('--sliding_window', type=int, default=128, 
                        help='sliding_window length for long PPL/JSD calculation')
    parser.add_argument('--alpha', type=int, default=2, 
                        help='Long-short distance (LSD) threshold for long PPL/JSD calculation')
    parser.add_argument('--beta', type=int, default=-2, 
                        help='Long context likelihood (LCL) threshold for long PPL/JSD calculation')
    
    parser.add_argument('--packing', action='store_true', help='Packing the quantized kv cache')


    cfgs = parser.parse_args()
    sensitivity(cfgs)

