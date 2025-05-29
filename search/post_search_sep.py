import os
import json
import argparse
import torch
import numpy as np
from pymoo.decomposition.asf import ASF
from pymoo.visualization.scatter import Scatter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# from pymoo.model.decision_making import DecisionMaking, normalize, find_outliers_upper_tail, NeighborFinder
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder
from pymoo.util.normalization import normalize

from evaluator import LlamaEvaluator
from tqdm import tqdm
import csv
from matplotlib import pyplot as plt
from utils.func import init_accelerator, get_net_info, clean_up
from utils.eval import measure_latency, eval_zeroshot
from utils.data import get_tokenizer
from quant.model import get_quantized_model
import gc
import warnings
warnings.simplefilter("ignore")

import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True 

class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, normalize=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected
        self.normalize = normalize

    def _do(self, F, **kwargs):
        n, m = F.shape

        if self.normalize:
            F = normalize(F, estimate_bounds_if_none=True)
            # F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

        neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            # np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            return find_outliers_upper_tail(mu)  # return points with trade-off > 2*sigma


def main(args):
    print(args)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    
    accelerator, device_map = init_accelerator(args.gpu_id, config)

    assert len(args.expr) == len(args.expr_comp_obj)
    n_comp_obj, n_comp_obj_min, n_comp_obj_max = len(args.comp_obj), len(args.comp_obj_min), len(args.comp_obj_max)
    assert n_comp_obj == n_comp_obj_min and n_comp_obj_min == n_comp_obj_max

    group_size = {'w': args.w_group_size, 'k': args.k_group_size, 'v': args.v_group_size}
    archive_list = []
    subnets_list = []
    metric_list = []
    F_list = []
    sort_idx_list = []
    for expr, expr_comp_obj in zip(args.expr, args.expr_comp_obj):
        with open(expr, 'r') as f:
            result_json = json.load(open(expr))
            archive = result_json['archive'] + result_json['candidates']
            archive_list.append(archive)

    # assert n_comp_obj == len(archive[0][2:])
        subnets, metric = [v[0] for v in archive], [v[1] for v in archive]
        subnets_list.append(subnets)
        metric_list.append(metric)
        comp_obj = [get_net_info(n, config, group_size)[expr_comp_obj] for n in subnets]

        sort_idx = np.argsort(metric)
        sort_idx_list.append(sort_idx)
        F = np.column_stack((metric, comp_obj))[sort_idx, :]
        F_list.append(F)
        
    import pdb; pdb.set_trace()

    pf_list, ps_list = [], []
    if n_comp_obj_min > 0:
        # range_idx = np.argwhere(np.logical_and(F[:, 1] > args.sec_obj_range[0], F[:, 1] < args.sec_obj_range[1])).flatten()
        # flag = np.ones((F.shape[0]), dtype=bool)
        # for i in range(n_comp_obj):
        #     flag = np.logical_and(flag, np.logical_and(F[:, i + 1] >= args.comp_obj_min[i], F[:, i + 1] <= args.comp_obj_max[i]))
        # range_idx = np.argwhere(flag).flatten()
        for i, comp_obj in enumerate(args.comp_obj):
            range_idx = np.argwhere(np.logical_and(F_list[i][:, 1] >= args.comp_obj_min[i], F_list[i][:, 1]  <= args.comp_obj_max[i])).flatten()

            print(f'range_idx : {len(range_idx)}')
            
            pf = F_list[i][range_idx, :]
            ps = np.array(subnets_list[i])[sort_idx_list[i]][range_idx]
            pf_list.append(pf)
            ps_list.append(ps)

    elif args.only_front:
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        pf = F[front, :]
        ps = np.array(subnets)[sort_idx][front]
        
    else:
        pf = F
        ps = np.array(subnets)[sort_idx]
        
    I_list = []
    if args.high_tradeoff:        
        I = NonDominatedSorting().do(pf, only_non_dominated_front=True)

    elif args.prefer:
        # preferences
        preferences = {}
        # for p in args.prefer.split("+"):
        for p in args.prefer:
            k, v = p.split("#")
            preferences[k] = float(v)
        weights = np.fromiter(preferences.values(), dtype=float)

        for i, comp_obj in args.comp_obj:
        # choose the architectures thats closest to the preferences
            I = ASF().do(pf, weights[0, i + 1]).argsort()[:args.n].reshape(args.n)
            I_list.append(I)
    else:
        I = range(len(pf))

    # always add most accurate architectures
    # I = np.append(I, 0)

    for idx in I:
        # print(f'Selected arch[{idx}] {args.sec_obj}: {pf[idx, 1]:.4f}, metric: {pf[idx, 0]:.4f}, arch: {ps[idx]}')
        # print(f'arch : {ps[idx]}')
        print(f'Selected arch[{idx}] {args.comp_obj}: {pf[idx, 1:].tolist()}, metric: {pf[idx, 0].item():.4f}')

    model_id = f'{args.model_path}/{args.model_name}'
    awq_gptq_owq = 'awq' in args.method or 'gptq' in args.method or 'owq' in args.method
    
    if awq_gptq_owq:
        args.quant_model_bits = []
        args.quant_model_paths = []

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

    for idx in tqdm(I):

        arch = ps[idx]
        # arch = dict()
        # arch['linear'] = {linear: [4] * config['n_block'] for linear in config['linear']}
        accelerator.print(arch)
        
        weight_bits = np.concatenate(list(arch['w'].values()))
        do_owq = ((weight_bits - weight_bits.astype(int)).sum() != 0)
        print(f'do_owq : {do_owq}, awq_gptq_owq : {awq_gptq_owq}')
        if awq_gptq_owq:
            method = 'awq' if 'awq' in args.method else 'gptq' if 'gptq' in args.method else 'owq' if 'owq' in args.method else None
            model = get_quantized_model(method, arch, model_id, device_map, config=config, prune='layer_prune' in args.method, do_owq=do_owq, owq_path=args.outlier_path)
        else:
            model = evaluator.sample(arch)
            
        model.config.residual_length = 0
        model.config.quant_kv_output = True

        metric = evaluator.eval(arch=arch, metric='ppl', model=model, accelerator=accelerator)[0] if args.datasets else 0
        complexity = get_net_info(arch, config, group_size)
        latency = measure_latency(model, generation=True, device=model.device) if args.latency else 0
        print(f'Selected arch[{idx}] {args.comp_obj}: {pf[idx, 1:]}, ppl: {[p for p in metric.values()]}, metric: {pf[idx, 0]:.4f} complexity: {complexity}, latency: {latency}\n')
        
        if args.zeroshot:
            clean_up()
            # model.use_cache = False
            model.config.residual_length = args.residual_length
            model.config.quant_kv_output = False
            
            results = eval_zeroshot(model, tokenizer=get_tokenizer(model_id), task_list=args.tasks)
            # acc_norm = [task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] for task_result in results.values()]
            # acc = [task_result['acc,none'] for task_result in results.values()]
            
            task = list(results.keys())            
            print(f'task : {task}')
            for task, result in results.items():
                print(f'task: {task}, result: {result}')
            # avg_acc_norm = np.mean(acc_norm)
            # avg_acc = np.mean(acc)
            # print(f'avg_acc_norm : {avg_acc_norm}, avg_acc : {avg_acc}')
            # print(f'task : {task}')
            # print(f'acc_norm : {acc_norm}')
            # print(f'acc : {acc}')
        if awq_gptq_owq:
            del model
            clean_up()

    print(args)
    return

    # if args.debug:
    #     # print(ps[I])
    #     # plot = Scatter()
    #     # plot.add(pf, alpha=0.2)
    #     # plot.add(pf[I, :], color="blue", s=10)
    #     # plot.add(gs_data, color="red", s=10)
    #     # plot.show()
    #     # plot.save(os.path.join(args.save, "best_trade_off_line.png"))
    #     os.makedirs(args.save, exist_ok=True)
        
    #     plt.scatter(complexity_list, [p[args.datasets[0]] for p in ppl_list], color='b', s=5, label='NSGA2')
    #     if args.greedy_search_result_path:
    #         with open(args.greedy_search_result_path, 'r') as f:
    #             gs_data = list(csv.reader(f))
    #             gs_bits = list(map(float, gs_data[1]))[:-3]
    #             gs_metric = list(map(float, gs_data[2]))[:-3]
    #             plt.scatter(gs_bits, gs_metric, color='r', s=5, label='Greedy Search')
        
    #     plt.xlabel(f'{args.sec_obj}')
    #     plt.ylabel('PPL')
    #     plt.legend()
    #     plt.show()
    #     plt.savefig(os.path.join(args.save, "best_trade_off_line.png"), dpi=300)

    sentences = []
    for k, v in vars(args).items():
        sentences.append(f"{k}: {v}\n")
    sentences.append("\n")
    for a, c, p in zip(arch_list, complexity_list, ppl_list):
        sentences.append(f"arch: {a}, bits: {c:.4f}, ppl: {p}\n")

    with open(os.path.join(args.save, args.results_file), 'w') as f:
        for sentence in sentences:
            f.write(sentence)

    with open(os.path.join(args.save, args.results_csv_file), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['arch', 'bits', 'params', 'sparsity', 'metric', 'latency'] + args.datasets)
        for a, b, p, s, m, l, ppl in zip(arch_list, bits_list, param_list, sparsity_list, metric_list, latency_list, ppl_list):
            writer.writerow([a, b, p, s, m, l] + list(ppl.values()))

    with open(os.path.join(args.save, args.results_arch_file), 'w') as f:
        json.dump({'archive': [[a, c, p] for a, c, p in zip(arch_list, complexity_list, ppl_list)]}, f, ensure_ascii=False, indent=4)

    return





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--comp_obj', type=str, nargs='+', default=['bits'], 
                        help='second objective to optimize simultaneously')
    parser.add_argument('--comp_obj_min', type=float, nargs='+', default=[],
                        help='')
    parser.add_argument('--comp_obj_max', type=float, nargs='+', default=[],
                        help='')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    
    parser.add_argument('--w_bits', type=int, nargs='+', default=[], 
                        help='')
    parser.add_argument('--k_bits', type=int, nargs='+', default=[2, 4], 
                        help='')
    parser.add_argument('--v_bits', type=int, nargs='+', default=[2, 4], 
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

    parser.add_argument('--k_quant_per', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_per', type=str, choices=['channel', 'token'], 
                        help='')

    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--expr', type=str, nargs='+', default=[''],
                        help='')
    parser.add_argument('--expr_comp_obj', type=str, nargs='+', default=[''],
                        help='')
    parser.add_argument('--prefer', type=str, nargs='+', default=[], 
                        help='preferences in choosing architectures (metric#10 bits#150)')
    # parser.add_argument('--high_tradeoff', action='store_true', help='')
    parser.add_argument('--high_tradeoff', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('-n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='')
    parser.add_argument('--debug', action='store_true', help='')
    parser.add_argument('--datasets', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--only_front', action='store_true', help='')
    parser.add_argument('--results_file', type=str, default='results.txt',
                        help='')
    parser.add_argument('--results_csv_file', type=str, default='results.csv',
                        help='')
    parser.add_argument('--results_arch_file', type=str, default='results_arch.json',
                        help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
    parser.add_argument('--latency', action='store_true', help='')
    parser.add_argument('--zeroshot', action='store_true', help='')
    parser.add_argument('--tasks', type=str, nargs='+', default=['coqa', 'gsm8k', 'truthfulqa'])


    cfgs = parser.parse_args()
    main(cfgs)
