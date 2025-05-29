import torch
import argparse
from tqdm import tqdm
import numpy as np
import json

from time import time

from search_space.llama import LlamaQuantSearchSpace
from evaluator import LlamaEvaluator
from utils.func import init_accelerator

def gen_data_linear(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)

    model_id=f'{args.model_path}/{args.model_name}'
    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        model_id=model_id,
        method=args.method,
        quant_model_paths=args.quant_model_paths,
        quant_model_bits=args.quant_model_bits,
        outlier=None,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=[args.dataset],
        loss_func=args.loss_func,
        device_map=device_map,
    )

    outlier_bits = {l: [] for l in config['linear']}
                
    search_space = LlamaQuantSearchSpace(
        n_block=config['n_block'],
        quant_model_bits=args.quant_model_bits,
        pass_linear_list=args.pass_linear_list,
        sec_obj=args.sec_obj,
        sec_obj_range=args.sec_obj_range,
        config=config,
        # layer_prune_range=self.layer_prune_range,
        outlier_bits=outlier_bits,
        only_outlier_bits=False
    )

    # ppl_archive = list()
    loss_archive = list()

    # complexity_list = list()
    if accelerator.is_main_process:
        if not args.pool:
            archs = search_space.initialize(args.n_data)
        else:
            with open(args.pool, 'r') as f:
                pool = [x[0] for x in json.load(f)['archive']]

            archs = search_space.sample(args.n_data, pool=pool)
            # import pdb; pdb.set_trace()
    else:
        archs = list()
    archs = accelerator.gather_for_metrics(archs, use_gather_object=True)
    
    for arch in tqdm(archs):
        iter_start = time()

        # ppl, complexity = evaluator.eval(arch=arch, accelerator=accelerator, metric='ppl')
        # ppl_archive.append([arch, ppl[args.dataset], complexity[args.sec_obj]])
        loss, complexity = evaluator.eval(arch=arch, accelerator=accelerator, metric='loss', loss_func=args.loss_func)
        loss = min(np.nan_to_num(loss[args.dataset], nan=args.max_value), args.max_value)
        loss_archive.append([arch, loss, complexity[args.sec_obj]])

        iter_time = time() - iter_start
        # print(f'complexity: {complexity:.3f}, loss : {loss:2f}, time : {iter_time:.2f}')
        accelerator.print(f'{args.sec_obj}: {complexity[args.sec_obj]:.3f}, loss : {loss:2f}, time : {iter_time:.2f}s')
        # print(f'{args.sec_obj}: {complexity[args.sec_obj]:.3f}, ppl : {ppl[args.dataset]:.2f}, loss : {loss[args.dataset]:.2f}, time : {iter_time:.2f}s')
        # complexity_list.append(complexity)

        if accelerator.is_main_process:
            # if args.ppl_json_file:
            #     with open(args.ppl_json_file, 'w') as f:
            #         json.dump({'archive': ppl_archive}, f, ensure_ascii=False, indent=4)

            if args.loss_json_file:
                with open(args.loss_json_file, 'w') as f:
                    json.dump({'archive': loss_archive, 'iteration': 0}, f, ensure_ascii=False, indent=4)
        accelerator.wait_for_everyone()
    print(args)
    # from matplotlib import pyplot as plt
    # plt.hist(complexity_list)
    # plt.show()
    # plt.savefig('test.png', dpi=300)


def main(args):
    gen_data_linear(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--quant_model_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='test batch size for inference')
    parser.add_argument('--metric', type=str, default='ppl',
                        help='which accuracy predictor model to fit (ppl/loss)')
    parser.add_argument('--pass_linear_list', type=str, nargs='+', default=[], 
                        help='which accuracy predictor model to fit (ppl/loss)')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--n_data', type=int, default=1000,
                        help='test batch size for inference')
    parser.add_argument('--loss_json_file', type=str, default='',
                        help='')
    parser.add_argument('--ppl_json_file', type=str, default='',
                        help='')
    parser.add_argument('--sec_obj', type=str, default='bits',
                        help='second objective to optimize simultaneously')
    parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[2, 4], 
                        help='')
    parser.add_argument('--loss_func', type=str, default='cross_entropy',
                        help='')
    parser.add_argument('--max_value', type=float, default=50,
                        help='')
    parser.add_argument('--layer_prune_range', type=float, nargs='+', default=[1., 1.], 
                        help='')
    parser.add_argument('--use_linear_group', action='store_true',
                        help='')
    parser.add_argument('--pool', type=str, default='',
                        help='')
    
    cfgs = parser.parse_args()
    main(cfgs)

