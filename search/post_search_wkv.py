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
from time import time
import csv
from matplotlib import pyplot as plt
from utils.func import init_accelerator, get_net_info, clean_up, process_dtype
from utils.eval import measure_latency, eval_zeroshot
from utils.eval_long_bench import pred_long_bench, eval_long_bench
from utils.data import get_tokenizer
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
    dtype = process_dtype(args.dtype)

    assert len(args.expr) == len(args.expr_comp_obj)
    n_comp_obj, n_comp_obj_min, n_comp_obj_max = len(args.comp_obj), len(args.comp_obj_min), len(args.comp_obj_max)
    assert n_comp_obj == n_comp_obj_min and n_comp_obj_min == n_comp_obj_max

    group_size = {'w': args.w_group_size, 'k': args.k_group_size, 'v': args.v_group_size}
    archive_list = []
    subnets_list = []
    metric_list = []
    prev_metric_list = []
    F_list = []
    sort_idx_list = []
    for expr, comp_obj in zip(args.expr, args.expr_comp_obj):
        with open(expr, 'r') as f:
            result_json = json.load(open(expr))
            archive = result_json['archive'] + result_json['candidates']
            archive_list.append(archive)

    # assert n_comp_obj == len(archive[0][2:])
        subnets, metric = [v[0] for v in archive], [v[1] for v in archive]
        subnets_list.append(subnets)
        metric_list.append(metric)
        prev_metric_list.append(metric)
        comp_obj = [get_net_info(n, config, group_size)[comp_obj] for n in subnets]

        sort_idx = np.argsort(metric)
        sort_idx_list.append(sort_idx)
        F = np.column_stack((metric, comp_obj))[sort_idx, :]
        F_list.append(F)
    
    if n_comp_obj == 1 and len(args.expr_comp_obj) == 2:
        subnets = []
        metric = []
        prev_metric = []
        
        front0 = NonDominatedSorting().do(F_list[0], only_non_dominated_front=True)
        f0_list = F_list[0][front0]
        subnets0_list = np.array(subnets_list[0])[sort_idx_list[0]][front0]
        
        front1 = NonDominatedSorting().do(F_list[1], only_non_dominated_front=True)
        f1_list = F_list[1][front1]
        subnets1_list = np.array(subnets_list[1])[sort_idx_list[1]][front1]
    
        # subnets0_bits_list = [get_net_info(n, config, group_size)['wbits'] for n in subnets0_list]
        # subnets1_bits_list = [get_net_info(n, config, group_size)['kvbits'] for n in subnets1_list]
        
        # print(f'subnets0_bits_list: {subnets0_bits_list}')
        # print(f'subnets1_bits_list: {subnets1_bits_list}')
        
        for f0, subnet0 in zip(f0_list, subnets0_list):
                
            for f1, subnet1 in zip(f1_list, subnets1_list):
                arch = dict()
                # f0 = f0[NonDominatedSorting().do(f0[:, 0], only_non_dominated_front=True)]
                if args.expr_comp_obj[0] == 'wbits':
                    arch['w'] = subnet0['w']
                elif args.expr_comp_obj[0] == 'kvbits':
                    arch['k'] = subnet0['k']
                    arch['v'] = subnet0['v']
                elif args.expr_comp_obj[0] == 'kbits':
                    arch['k'] = subnet0['k']
                elif args.expr_comp_obj[0] == 'vbits':
                    arch['v'] = subnet0['v']
                
                # f1 = f1[NonDominatedSorting().do(f1[:, 0], only_non_dominated_front=True)]
                if args.expr_comp_obj[1] == 'wbits':
                    arch['w'] = subnet1['w']
                elif args.expr_comp_obj[1] == 'kvbits':
                    arch['k'] = subnet1['k']
                    arch['v'] = subnet1['v']
                elif args.expr_comp_obj[1] == 'kbits':
                    arch['k'] = subnet1['k']
                elif args.expr_comp_obj[1] == 'vbits':
                    arch['v'] = subnet1['v']
                    
                cur_metric = f0[0] + f1[0] - (f0[0] * f1[0])
                metric.append(cur_metric)
                prev_metric.append([f0[0], f1[0]])
                subnets.append(arch)
        
        comp_obj = [get_net_info(a, config, group_size, n_token=args.n_token)[args.comp_obj[0]] for a in subnets] 
        sort_idx = np.argsort(metric)
        F = np.column_stack((metric, comp_obj))[sort_idx, :]
        # F_list.append(F)
        F_list, sort_idx_list, subnets_list, metric_list, prev_metric_list = [F], [sort_idx], [subnets], [metric], [prev_metric]
        
        # wbits = [get_net_info(a, config, group_size, n_token=args.n_token)['wbits'] for a in subnets] 
        # kvbits = [get_net_info(a, config, group_size, n_token=args.n_token)['kvbits'] for a in subnets] 
        # print(f'wbits: {wbits}')
        # print(f'kvbits: {kvbits}')
    
    pf_list, ps_list, pm_list, ppm_list = [], [], [], []
    if n_comp_obj_min > 0:
        for i, comp_obj in enumerate(args.comp_obj):
            range_idx = np.argwhere(np.logical_and(F_list[i][:, 1] >= args.comp_obj_min[i], F_list[i][:, 1]  <= args.comp_obj_max[i])).flatten()
            print(f'range_idx : {len(range_idx)}')
            
            pf = F_list[i][range_idx, :]
            ps = np.array(subnets_list[i])[sort_idx_list[i]][range_idx]
            pm = np.array(metric_list[i])[sort_idx_list[i]][range_idx]
            ppm = np.array(prev_metric_list[i])[sort_idx_list[i]][range_idx]
            # wbits = [get_net_info(a, config, group_size, n_token=args.n_token)['wbits'] for a in ps] 
            # kvbits = [get_net_info(a, config, group_size, n_token=args.n_token)['kvbits'] for a in ps] 
            # print(f'wbits: {wbits}')
            # print(f'kvbits: {kvbits}')
            pf_list.append(pf)
            ps_list.append(ps)
            pm_list.append(pm)
            ppm_list.append(ppm)

    elif args.only_front:
        for i, comp_obj in enumerate(args.comp_obj):
            front = NonDominatedSorting().do(F_list[i], only_non_dominated_front=True)
            pf = F_list[i][front, :]
            ps = np.array(subnets_list[i])[sort_idx_list[i]][front]
            pm = np.array(metric_list[i])[sort_idx_list[i]][front]
            ppm = np.array(prev_metric_list[i])[sort_idx_list[i]][front]
            pf_list.append(pf)
            ps_list.append(ps)
            pm_list.append(pm)
            ppm_list.append(ppm)

    else:
        pf_list = F_list
        for i, comp_obj in enumerate(args.comp_obj):
            ps = np.array(subnets_list[i])[sort_idx_list[i]]
            pm = np.array(metric_list[i])[sort_idx_list[i]]
            ppm = np.array(prev_metric_list[i])[sort_idx_list[i]]
            ps_list.append(ps)
            pm_list.append(pm)
            ppm_list.append(ppm)
        
    I_list = []
    if args.high_tradeoff:   
        temp = []     
        for i, comp_obj in enumerate(args.comp_obj):
            I = NonDominatedSorting().do(pf_list[i], only_non_dominated_front=True)
            temp.append(I)
        I_list.append(temp)

    elif args.prefer:
        # preferences
        preferences = {}
        # for p in args.prefer.split("+"):
        for p in args.prefer:
            k, v = p.split("#")
            preferences[k] = float(v)
        weights = np.fromiter(preferences.values(), dtype=float)

        temp = []
        for i, comp_obj in enumerate(args.comp_obj):
        # choose the architectures thats closest to the preferences
            I = ASF().do(pf_list[i], [weights[prefer_idx] for prefer_idx in [0, i + 1]]).argsort()[0]
            temp.append(I)
        I_list.append(temp)
        
    else:
        # comp_list = {c: [] for c in args.comp_obj}
        comp_save_list = {c: [] for c in get_net_info({}, None, group_size=-1, n_token=0).keys()}
        metric_save_list = {d: [] for d in args.datasets}
        new_metric_save_list = {c: [] for c in args.comp_obj}
        prev_metric_save_list = {c: {e_c: [] for e_c in args.expr_comp_obj} for c in args.comp_obj}
        for i, comp_obj in enumerate(args.comp_obj):
            I = list(range(len(pf_list[i])))
            if args.random_sample is not None and args.random_sample < len(pf_list[i]):
                front = np.random.choice(I, size=args.random_sample, replace=False)
                front.sort()
            I_list.append(I)
        

    # always add most accurate architectures
    # I = np.append(I, 0)

    # for idx_list in I_list:
    #     # print(f'Selected arch[{idx}] {args.sec_obj}: {pf[idx, 1]:.4f}, metric: {pf[idx, 0]:.4f}, arch: {ps[idx]}')
    #     # print(f'arch : {ps[idx]}')
    #     for i, comp_obj in enumerate(args.comp_obj):
    #         accelerator.print(f'Selected arch[{idx_list[i]}] {comp_obj}: {pf_list[i][idx_list[i], 1:].tolist()}, metric: {pf_list[i][idx_list[i], 0].tolist()}')

    model_id = f'{args.model_path}/{args.model_name}'
    # use_awq_gptq_owq = 'awq' in args.w_method or 'gptq' in args.w_method or 'owq' in args.w_method
    
    if 'hqq' not in args.w_method:
        args.quant_model_paths = []

    evaluator = LlamaEvaluator(
        config,
        accelerator=accelerator,
        model_id=model_id,
        method={'w': args.w_method, 'kv': args.kv_method},
        quant_model_paths=args.quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=args.seqlen,
        min_seqlen=args.min_seqlen,
        n_sample=args.n_sample,        
        datasets=args.datasets,
        device_map=device_map,
        dtype=dtype,
        bits={'w': args.w_bits, 'k': args.k_bits, 'v': args.v_bits},
        group_size=group_size,
        residual_length=args.residual_length,
        # use_flash=args.use_flash,
        k_quant_scheme=args.k_quant_scheme,
        v_quant_scheme=args.v_quant_scheme,
        loss_func=args.loss_func
    )
    
    for idx_list in tqdm(I_list):
        for idx in idx_list:
            if args.comp_obj == ['wbits', 'kvbits']:
                raise NotImplementedError
                # arch = {}
                # for i, comp_obj in enumerate(args.comp_obj):
                #     ps, idx = ps_list[i], idx_list[i]
                #     if comp_obj == 'wbits':
                #         arch['w'] = ps[idx]['w']
                #     elif comp_obj == 'kvbits':
                #         arch['k'] = ps[idx]['k']
                #         arch['v'] = ps[idx]['v']
                #     elif comp_obj == 'kbits':
                #         arch['k'] = ps[idx]['k']
                #     elif comp_obj == 'vbits':
                #         arch['v'] = ps[idx]['v']
                # accelerator.print(arch)
                
            elif args.comp_obj == ['memory']: 
                arch = ps_list[0][idx]
                
            else:
                raise NotImplementedError
            
            # complexity = get_net_info(arch, config, group_size, n_token=args.n_token)
            # latency = measure_latency(model, generation=True, device=model.device) if args.latency else 0
            # print(f'complexity: {complexity}')
            # print(f'arch: {arch}')
            model = evaluator.sample(arch)
            
            # for i, comp_obj in enumerate(args.comp_obj):
            #     # for idx in 
            #     # accelerator.print(f'Selected arch[{idx}] {comp_obj}: {pf_list[i][idx_list[i], 1:].tolist()}, metric: {pf_list[i][idx_list[i], 0].tolist()}')   
            #     accelerator.print(f'Selected arch[{idx}] {comp_obj}: {pf_list[i][idx, 1:].tolist()}, metric: {pf_list[i][idx_list[i], 0].tolist()}')            
                
            if args.datasets:
                if args.kv_method == 'kivi':
                    model.config.kivi_config.residual_length = 0
                elif args.kv_method == 'hqq':
                    model.generation_config.cache_config = 0
                model.config.quant_kv_output = True
                model.config.use_cache = False

                metric = evaluator.eval(arch=arch, metric=args.metric, model=model, accelerator=accelerator, loss_func=args.loss_func)[0] if args.datasets else 0
                complexity = get_net_info(arch, config, group_size, n_token=args.n_token)
                # latency = measure_latency(model, generation=True, device=model.device) if args.latency else 0
                print(f'[{i}] complexity: {complexity}, {args.metric}: {[p for p in metric.values()]}, metric: {[m[idx].item() for m in metric_list]}, prev_metric: {[m[idx] for m in prev_metric_list]}')
                if args.random_sample is not None and args.save and args.results_csv_file:
                    for c in comp_save_list:
                        comp_save_list[c].append(complexity[c])
                    for d in args.datasets:
                        metric_save_list[d].append(metric[d])
                    for c_i, c in enumerate(args.comp_obj):
                        new_metric_save_list[c].append(metric_list[c_i][idx].item())
                    for c_i, c in enumerate(args.comp_obj):
                        for p_i, expr_c in enumerate(args.expr_comp_obj):
                            prev_metric_save_list[c][expr_c].append(prev_metric_list[c_i][idx][p_i])

                    os.makedirs(args.save, exist_ok=True)
                    with open(os.path.join(args.save, args.results_csv_file), 'w') as f:
                        writer = csv.writer(f)
                        for c in comp_save_list:
                            writer.writerow(comp_save_list[c])
                        for d in args.datasets:
                            writer.writerow(metric_save_list[d])
                        for c in args.comp_obj:
                            writer.writerow(new_metric_save_list[c])
                        for c in args.comp_obj:
                            for e_c in args.expr_comp_obj:
                                writer.writerow(prev_metric_save_list[c][e_c])

            if args.pass_key_file:
                clean_up()
                # model.config.residual_length = args.residual_length
                if args.kv_method == 'kivi':
                    model.config.kivi_config.residual_length = args.residual_length
                elif args.kv_method == 'hqq':
                    model.generation_config.cache_config = args.residual_length
                model.config.quant_kv_output = False
                model.config.use_cache = True
                
                # method_name = f"K{config.k_bits}V{config.v_bits} KiVi"
                print( "-----------------------------------" )
                enc = get_tokenizer(model_id)
                for line in open(args.pass_key_file, "r"):
                    clean_up()
                    torch.cuda.reset_max_memory_allocated()
                    example = json.loads(line)
                    prompt_postfix = "What is the pass key? The pass key is "
                    prompt = example["input"] + prompt_postfix
                    input_ids = enc(prompt, return_tensors="pt").input_ids.cuda()
                    print( "-----------------------------------" )
                    print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
                    print( "Passkey target:", example["target"] )

                    tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
                    answer = prompt_postfix + enc.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
                    answer = answer.replace("\n", "\\n")
                    # answer= f"{method_name}:\n     [ {answer} ]"
                    answer= f"[ {answer} ]"
                    
                    peak_memory = torch.cuda.max_memory_allocated()
                    print( answer )
                    print(f"Mem: {peak_memory / 1024 / 1024 / 1024:.3f} GB")
                    # print(f"Mem: {peak_memory / 1024 / 1024:.3f} MB")
                    print( "-----------------------------------\n" )
            
            if args.zeroshot:
                clean_up()
                # model.config.residual_length = args.residual_length
                if args.kv_method == 'kivi':
                    model.config.kivi_config.residual_length = args.residual_length
                elif args.kv_method == 'hqq':
                    model.generation_config.cache_config = args.residual_length
                model.config.quant_kv_output = False
                model.config.use_cache = True
                
                results = eval_zeroshot(model, tokenizer=get_tokenizer(model_id), task_list=args.tasks, batch_size=args.lm_eval_batch_size)
                
                task = list(results.keys())
                total_result = []
                print(f'task : {task}')
                for task, result in results.items():
                    # print(f'task: {task}, result: {result}')
                    new_result = {}
                    for k, v in result.items():
                        if k in ['em,none', 'exact_match,strict-match', 'exact_match,flexible-extract', 'bleu_max,none', 'bleu_acc,none', 'acc,none']:
                            new_result[k] = float(v)
                    print(f'task: {task}, result: {list(new_result.keys())}, {list(new_result.values())}')
                    total_result += list(new_result.values())
                print(f'total_result: {total_result}')
            
            if args.long_bench:
                clean_up()
                # model.config.residual_length = args.residual_length
                if args.kv_method == 'kivi':
                    model.config.kivi_config.residual_length = args.residual_length
                elif args.kv_method == 'hqq':
                    model.generation_config.cache_config = args.residual_length
                model.config.quant_kv_output = False
                model.config.use_cache = True
                
                # if len(args.long_bench_task) == 0 and not args.long_bench_task_e:
                #     args.long_bench_task = []
                long_bench_start = time()
                pred_long_bench(model, tokenizer=get_tokenizer(model_id), save_path=args.long_bench_result_path, long_bench_config=args.long_bench_config, e=args.long_bench_e)
                eval_long_bench(args.long_bench_result_path, args.long_bench_e)
                long_bench_time = time() - long_bench_start
                
                sentences = []
                for k, v in vars(args).items():
                    sentences.append(f"{k}: {v}\n")
                sentences.append(f'Longbench Time: {long_bench_time:.2f}s')
                sentences.append("\n")

                with open(os.path.join(args.long_bench_result_path, "pred_e" if args.long_bench_e else "pred", 'result.txt'), 'w') as f:
                    for sentence in sentences:
                        f.write(sentence)
            
            if 'awq' in args.w_method or 'gptq' in args.w_method or 'qeft' in args.w_method:
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
    parser.add_argument('--dtype', type=str, default='auto', choices=['float16', 'float', 'fp16', 'bfloat16', 'bfloat', 'bf16', 'auto'],
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
    
    parser.add_argument('--w_method', type=str, nargs='+', default=[], choices=['fp16', 'awq', 'gptq', 'qeft', 'hqq'],
                        help='')
    parser.add_argument('--kv_method', type=str, default='kivi', choices=['hqq', 'kivi'],
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

    parser.add_argument('--k_quant_scheme', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_scheme', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--score', type=str, default='kivi', choices=['hqq', 'kivi'],
                        help='')

    parser.add_argument('--metric', type=str, default='ppl',
                        help='which metric predictor model to fit (ppl/loss)')
    parser.add_argument('--loss_func', type=str, default='cross_entropy',
                        help='')
    
    parser.add_argument('--save', type=str, default='',
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
    # parser.add_argument('-n', type=int, default=1,
    #                     help='number of architectures desired')
    parser.add_argument('--n_token', type=int, default=0, 
                        help='target sequence length for memory calculation')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='sample number of the calibration set')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='sequential length of the calibaration (train) set')
    parser.add_argument('--min_seqlen', type=int, default=0,
                        help='minimum sequential length of the calibaration gsm8k set')
    
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
    parser.add_argument('--lm_eval_batch_size', type=int, default=None,
                        help='')
    parser.add_argument('--long_bench', action='store_true', help='')
    parser.add_argument('--long_bench_e', action='store_true',
                        help='number of architectures desired')
    parser.add_argument('--long_bench_result_path', type=str, default='',
                        help='')
    parser.add_argument('--long_bench_config', type=str, default='',
                        help='')
    parser.add_argument('--long_bench_task', type=str, nargs='+', default=[])
    parser.add_argument('--pass_key_file', type=str, default='',
                        help='')
    
    parser.add_argument('--random_sample', type=int, default=None, 
                        help='')


    cfgs = parser.parse_args()
    main(cfgs)
