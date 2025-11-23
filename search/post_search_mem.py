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
import math
import scipy.stats as stats
from matplotlib import pyplot as plt
from utils.func import init_accelerator, get_net_info, clean_up, process_dtype
from utils.eval import measure_latency, eval_zeroshot
from utils.eval_long_bench import pred_long_bench, eval_long_bench
from utils.data import get_tokenizer
import warnings
warnings.simplefilter("ignore")

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

    # assert len(args.expr) == len(args.expr_comp_obj)
    n_comp_obj, n_comp_obj_min, n_comp_obj_max = len(args.comp_obj), len(args.comp_obj_min), len(args.comp_obj_max)
    assert n_comp_obj == n_comp_obj_min and n_comp_obj_min == n_comp_obj_max

    group_size = {'w': args.w_group_size, 'k': args.k_group_size, 'v': args.v_group_size}
    metric_list = []
    
    # for expr, comp_obj in zip(args.expr, args.expr_comp_obj):
    with open(args.expr, 'r') as f:
        result_json = json.load(f)
        archive = result_json['archive'] + result_json['candidates']

    subnets, metric = [v[0] for v in archive], [v[1] for v in archive]
    comp_obj = [[get_net_info(a, config, group_size, n_token=args.n_token)[comp_obj] for comp_obj in args.comp_obj] for a in subnets]
    
    sort_idx = np.argsort(metric)
    F = np.column_stack((metric, comp_obj))[sort_idx]
    subnets = np.array(subnets)[sort_idx]
    
    # pf_list, ps_list, pm_list, ppm_list = [], [], [], []
    if n_comp_obj_min > 0:
        flag = np.ones(len(F), dtype=bool)
        for i in range(n_comp_obj):
            flag = np.logical_and(flag, np.logical_and(F[:, -n_comp_obj + i] >= args.comp_obj_min[i], F[:, -n_comp_obj + i] <= args.comp_obj_max[i]))
        range_idx = np.argwhere(flag).flatten()
        print(f'range_idx : {len(range_idx)}')
                
        pf = F[range_idx]
        ps = subnets[range_idx]
        # wbits = [get_net_info(a, config, group_size, n_token=args.n_token)['wbits'] for a in ps] 
        # kvbits = [get_net_info(a, config, group_size, n_token=args.n_token)['kvbits'] for a in ps] 
        # print(f'wbits: {wbits}')
        # print(f'kvbits: {kvbits}')

    # elif args.only_front:
    #     for i, comp_obj in enumerate(args.comp_obj):
    #         front = NonDominatedSorting().do(F[i], only_non_dominated_front=True)
    #         pf = F[front]
    #         ps = subnets[front]

    else:
        pf = F
        ps = subnets
        
    if args.high_tradeoff:
        I = NonDominatedSorting().do(pf[:, [0, *[i for i in range(-n_comp_obj, 0)]]], only_non_dominated_front=True)

    if args.prefer:
        # preferences
        preferences = {}
        # for p in args.prefer.split("+"):
        for p in args.prefer:
            k, v = p.split("#")
            preferences[k] = float(v)
        weights = np.fromiter(preferences.values(), dtype=float)        
        # I = ASF().do(pf[:, [0, *[i for i in range(-n_comp_obj, 0)]]], weights).argsort()[0]
        I = ASF().do(pf[:, [0, *[i for i in range(-n_comp_obj, 0)]]], weights).argsort()[:args.n].reshape(args.n)
        
    else:
        I = list(range(len(pf)))
        if args.random_sample is not None and args.random_sample < len(pf):
            I = np.random.choice(I, size=args.random_sample, replace=False)
            I.sort()


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
        loss_func=args.loss_func,
        use_key_token=args.use_key_token,
        trunc_len=args.trunc_len,
        sliding_window=args.sliding_window,
        alpha=args.alpha,
        beta=args.beta
    )
    
    comp_save_list = [list() for _ in get_net_info({}, None, group_size=-1, n_token=0).keys()]
    metric_save_list = [list() for _ in range((len(args.datasets) + 3))] 
    for idx in tqdm(I):
        arch = ps[idx]
            
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
            print(f'[{idx}] complexity: {complexity}, {args.metric}: {[p for p in metric.values()]}, metric: {[pf[idx, 0]]}, prev_metric: {pf[idx, 1: -n_comp_obj]}')
            if args.random_sample is not None and args.save and args.results_csv_file:
                for c_i, c in enumerate(complexity.values()):
                    comp_save_list[c_i].append(c)
                for m_i, m in enumerate(metric.values()):
                    metric_save_list[m_i].append(m)
                for m_i, m in enumerate(pf[idx, 0: -n_comp_obj]):
                    metric_save_list[m_i + len(args.datasets)].append(m)
                    
                os.makedirs(args.save, exist_ok=True)
                with open(os.path.join(args.save, args.results_csv_file), 'w') as f:
                    writer = csv.writer(f)
                    for c in comp_save_list:
                        writer.writerow(c)                        
                    for m in metric_save_list:
                        writer.writerow(m)
                        

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
                    
        if args.ruler:
            pass
        
        if 'awq' in args.w_method or 'gptq' in args.w_method or 'qeft' in args.w_method:
            del model, evaluator.model
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


def lsq(jsd_w, jsd_kv, jsd_actual, add_intercept=False):
    """
    Least-squares fit for: jsd_pred = alpha * jsd_w + beta * jsd_kv (+ gamma if add_intercept).
    """
    jsd_w = np.asarray(jsd_w, dtype=float).reshape(-1)
    jsd_kv = np.asarray(jsd_kv, dtype=float).reshape(-1)
    y = np.asarray(jsd_actual, dtype=float).reshape(-1)
    assert jsd_w.shape == jsd_kv.shape == y.shape, "All inputs must have same length"
    
    # Design matrix X
    if add_intercept:
        X = np.column_stack([jsd_w, jsd_kv, np.ones_like(jsd_w)])
    else:
        X = np.column_stack([jsd_w, jsd_kv])

    theta, *_ = np.linalg.lstsq(X, y, rcond=None)

    gamma = 0
    if add_intercept:
        alpha, beta, gamma = theta
    else:
        alpha, beta = theta    
    return {"alpha": float(alpha), "beta": float(beta), "gamma": float(gamma)}



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
    parser.add_argument('--expr', type=str, default='',
                        help='')
    # parser.add_argument('--w_expr', type=str, default='',
    #                     help='')
    # parser.add_argument('--kv_expr', type=str, default='',
    #                     help='')
    # parser.add_argument('--expr_front', action='store_true', help='')
    parser.add_argument('--expr_comp_obj', type=str, nargs='+', default=[''],
                        help='')
    parser.add_argument('--prefer', type=str, nargs='+', default=[], 
                        help='preferences in choosing architectures (metric#10 bits#150)')
    # parser.add_argument('--high_tradeoff', action='store_true', help='')
    parser.add_argument('--high_tradeoff', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('-n', type=int, default=1,
                        help='number of architectures desired')
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
    parser.add_argument('--random_sample_path', type=str, default='', 
                        help='')
    parser.add_argument('--grid_search', type=float, nargs='+', default=[])
    
    parser.add_argument('--sqrt', action='store_true', help='')
    parser.add_argument('--kv_scale', type=float, default=1.,
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
    

    cfgs = parser.parse_args()
    main(cfgs)
