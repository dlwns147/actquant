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
from utils.func import init_accelerator, clean_up, process_dtype, get_net_info, set_seed
from utils.eval import measure_latency, eval_zeroshot
from utils.longbench import pred_longbench, eval_longbench
from utils.data import get_tokenizer
from utils.ruler import eval_ruler
from utils.longeval import eval_longeval_lines
import warnings
from time import time
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

    model_id = f'{args.model_path}/{args.model_name}'
    dtype = process_dtype(args.dtype)

    # use_awq_or_gptq = 'awq' in args.w_method or 'gptq' in args.w_method
    # method = 'awq' if 'awq' in args.method else 'gptq' if 'gptq' in args.method else None
    bits = {'w': [args.w_bits]}
    if args.k_bits is not None:
        bits['k'] = [args.k_bits]
    if args.v_bits is not None:
        bits['v'] = [args.v_bits]
        
    group_size = {'w': args.w_group_size, 'k': [[args.k_group_size]], 'v': [[args.v_group_size]]}
    
    if 'hqq' not in args.w_method:
        args.quant_model_paths = []

    evaluator = LlamaEvaluator(
        config,
        accelerator=accelerator,
        model_id=model_id,
        method={'w': args.w_method, 'kv': args.kv_method},
        quant_model_paths=args.quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        metric=args.metric,
        loss_func=args.loss_func,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=args.datasets,
        min_seqlen=args.min_seqlen,
        data_batch_size=args.data_batch_size,
        device_map=device_map,
        bits=bits,
        pruning_ratio=args.pruning_ratio,
        dtype=dtype,
        group_size=group_size,
        residual_length=args.residual_length,
        # use_flash=args.use_flash,
        k_quant_scheme=args.k_quant_scheme,
        v_quant_scheme=args.v_quant_scheme,
        use_key_token=args.use_key_token,
        trunc_len=args.trunc_len,
        sliding_window=args.sliding_window,
        alpha=args.alpha,
        beta=args.beta,
        key_token_path=args.key_token_path,
        last_tokens=args.last_tokens
    )

    arch = dict()
    # arch['w'] = 
    arch = {'w': {linear: [args.w_bits] * config['n_block'] for linear in config['linear']}}
    if args.k_bits is not None:
        arch['k'] = [[args.k_bits, args.k_group_size]] * config['n_block']
    if args.v_bits is not None:
        arch['v'] = [[args.v_bits, args.v_group_size]] * config['n_block']
    if args.pruning_ratio is not None:
        arch['k_prune'] = [args.pruning_ratio] * config['n_block']
        arch['v_prune'] = [args.pruning_ratio] * config['n_block']
    accelerator.print(arch)
    
    model = evaluator.sample(arch)

    if args.datasets:
        if args.stride is not None:
            if 'kivi' in args.kv_method:
                model.config.kivi_config.residual_length = args.residual_length
            elif 'hqq' in args.kv_method:
                model.generation_config.cache_config = args.residual_length
            model.config.quant_kv_output = False
            model.config.use_cache = True
            
        else:
            if 'kivi' in args.kv_method:
                model.config.kivi_config.residual_length = 0
            elif 'hqq' in args.kv_method:
                model.generation_config.cache_config = 0
            model.config.quant_kv_output = True
            model.config.use_cache = False

        metric_start = time()
        metric, complexity = evaluator.eval(arch=arch, metric=args.metric, model=model, accelerator=accelerator, loss_func=args.loss_func, stride=args.stride)
        metric_time = time() - metric_start
        print(f'[0] {args.metric}: {[p for p in metric.values()]}, metric: {list(metric.values())}, prev_metric: [0]')
        print(f'complexity: {list(complexity.keys())}')
        print(f'complexity: {list(complexity.values())}')
        print(f'Metric Time: {metric_time:.2f}s')
        # accelerator.print(arch)
        # print(f'complexity: {complexity}, ppl: {[p for p in metric.values()]}')

    if args.pass_key_file:
        clean_up()
        # model.config.residual_length = args.residual_length
        if 'kivi' in args.kv_method:
            model.config.kivi_config.residual_length = args.residual_length
        elif 'hqq' in args.kv_method:
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
        if 'kivi' in args.kv_method:
            model.config.kivi_config.residual_length = args.residual_length
        elif 'hqq' in args.kv_method:
            model.generation_config.cache_config = args.residual_length
        model.config.quant_kv_output = False
        model.config.use_cache = True
        
        results = eval_zeroshot(model, tokenizer=get_tokenizer(model_id), task_list=args.tasks, batch_size=args.lm_eval_batch_size, num_fewshot=args.num_fewshot)
        
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
            
        # acc_norm = [task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] for task_result in results.values()]
        # acc = [task_result['acc,none'] for task_result in results.values()]
        
        # task = list(results.keys())
        # avg_acc_norm = np.mean(acc_norm)
        # avg_acc = np.mean(acc)
        # print(f'avg_acc_norm : {avg_acc_norm}, avg_acc : {avg_acc}')
        # print(f'task : {task}')
        # print(f'acc_norm : {acc_norm}')
        # print(f'acc : {acc}')

    if args.longbench:
        clean_up()
        # model.config.residual_length = args.residual_length
        if 'kivi' in args.kv_method:
            model.config.kivi_config.residual_length = args.residual_length
        elif 'hqq' in args.kv_method:
            model.generation_config.cache_config = args.residual_length
        model.config.quant_kv_output = False
        model.config.use_cache = True
        
        longbench_start = time()
        pred_longbench(model, tokenizer=get_tokenizer(model_id), save_path=args.longbench_result_path, longbench_config=args.longbench_config, e=args.longbench_e)
        eval_longbench(args.longbench_result_path, args.longbench_e)
        longbench_time = time() - longbench_start
        
        sentences = []
        for k, v in vars(args).items():
            sentences.append(f"{k}: {v}\n")
        sentences.append(f'Longbench Time: {longbench_time:.2f}s')
        sentences.append("\n")

        with open(os.path.join(args.longbench_result_path, "pred_e" if args.longbench_e else "pred", 'result.txt'), 'w') as f:
            for sentence in sentences:
                f.write(sentence)
        
    if args.ruler:
        clean_up()
        set_seed(args.seed, deterministic=True)
        if 'kivi' in args.kv_method:
            model.config.kivi_config.residual_length = args.residual_length
        elif 'hqq' in args.kv_method:
            model.generation_config.cache_config = args.residual_length
        model.config.quant_kv_output = False
        model.config.use_cache = True
        # tokenizer=get_tokenizer(model_id)
        # tokenizer.pad_token = tokenizer.eos_token
        
        ruler_start = time()
        eval_ruler(model, tokenizer=get_tokenizer(model_id), model_id=model_id, tasks=args.ruler_task, yaml_path=args.ruler_yaml_path, batch_size=args.ruler_batch_size, length=args.ruler_length, nsample=args.ruler_sample, gen_toks=args.ruler_gen_toks, result_path=args.ruler_result_path, seed=args.seed)
        ruler_time = time() - ruler_start
        print(f'RULER Time: {ruler_time:.2f}s')


    if args.longeval:
        clean_up()
        if 'kivi' in args.kv_method:
            model.config.kivi_config.residual_length = args.residual_length
        elif 'hqq' in args.kv_method:
            model.generation_config.cache_config = args.residual_length
        model.config.quant_kv_output = False
        model.config.use_cache = True
        
        tokenizer = get_tokenizer(model_id)
        longeval_start = time()
        
        # Prepare result path
        if args.longeval_result_path:
            os.makedirs(os.path.dirname(args.longeval_result_path) if os.path.dirname(args.longeval_result_path) else '.', exist_ok=True)
            result_path = args.longeval_result_path
        else:
            result_path = ''
        
        # Evaluate longeval lines task
        results = eval_longeval_lines(
            model=model,
            tokenizer=tokenizer,
            test_dir=args.longeval_test_dir,
            model_name_or_path=model_id,
            num_lines_list=args.longeval_num_lines,
            eval_shortest_only=args.longeval_shortest_only,
            result_path=result_path,
            use_cache=True
        )
        
        longeval_time = time() - longeval_start
        print(f'LongEval Lines Task Time: {longeval_time:.2f}s')
        
        # Save results summary
        if args.longeval_result_path:
            sentences = []
            for k, v in vars(args).items():
                sentences.append(f"{k}: {v}\n")
            sentences.append(f'LongEval Time: {longeval_time:.2f}s')
            sentences.append("\n")
            
            summary_path = args.longeval_result_path.replace('.json', '_summary.txt')
            with open(summary_path, 'w') as f:
                for sentence in sentences:
                    f.write(sentence)
        

    print(args)
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
    # parser.add_argument('--bits', type=int, default=2,
    #                     help='')
    # parser.add_argument('--group_size', type=int, default=128,
    #                     help='')
    
    parser.add_argument('--w_method', type=str, nargs='+', default=[], choices=['fp16', 'awq', 'gptq', 'qeft', 'hqq'],
                        help='')
    parser.add_argument('--kv_method', type=str, nargs='+', default=['kivi'], choices=['fp16', 'hqq', 'kivi', 'think'],
                        help='')
    
    parser.add_argument('--w_bits', type=int, default=4, 
                        help='')
    parser.add_argument('--k_bits', type=int, default=None, 
                        help='')
    parser.add_argument('--v_bits', type=int, default=None, 
                        help='')
    
    parser.add_argument('--w_group_size', type=int, default=128, 
                        help='')
    parser.add_argument('--k_group_size', type=int, default=128, 
                        help='')
    parser.add_argument('--v_group_size', type=int, default=128, 
                        help='')
    
    parser.add_argument('--quant_kv_output', action='store_true', help='')
    parser.add_argument('--k_quant_scheme', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_scheme', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--residual_length', type=int, default=128, 
                        help='')
    # parser.add_argument('--use_flash', action='store_true', help='')

    parser.add_argument('--pruning_ratio', type=float, default=1.0, 
                        help='pruning ratio for ThinK')

    parser.add_argument('--clip_asym', action='store_true', help='')
    parser.add_argument('--comp_obj', type=str, nargs='+', default=['bits'], 
                        help='second objective to optimize simultaneously')
    parser.add_argument('--comp_obj_min', type=float, nargs='+', default=[],
                        help='')
    parser.add_argument('--comp_obj_max', type=float, nargs='+', default=[],
                        help='')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    # parser.add_argument('--method', type=str, nargs='+', default=[],
    #                     help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    # parser.add_argument('--expr', type=str, default='',
    #                     help='location of search experiment dir')
    parser.add_argument('--prefer', type=str, nargs='+', default=[], 
                        help='preferences in choosing architectures (metric#10 bits#150)')
    # parser.add_argument('--high_tradeoff', action='store_true', help='')
    parser.add_argument('--high_tradeoff', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('-n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='')
    parser.add_argument('--min_seqlen', type=int, default=0,
                        help='')
    parser.add_argument('--data_batch_size', type=int, default=1,
                        help='')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='')
    parser.add_argument('--n_token', type=int, default=0, 
                        help='target sequence length for memory calculation')
    
    parser.add_argument('--metric', type=str, default='ppl',
                        help='which metric predictor model to fit (ppl/loss)')
    parser.add_argument('--loss_func', type=str, default='cross_entropy',
                        help='')
    parser.add_argument('--stride', type=int, default=None, 
                        help='')
    parser.add_argument('--last_tokens', type=int, default=None, 
                        help='')
                        
    parser.add_argument('--seed', type=int, default=0,
                        help='')
    # parser.add_argument('--debug', action='store_true', help='')
    # parser.add_argument('--sec_obj', type=str, default='bits',
    #                     help='second objective to optimize simultaneously')
    parser.add_argument('--datasets', type=str, nargs='+', default=[], 
                        help='')
    # parser.add_argument('--greedy_search_result_path', type=str, default='',
    #                     help='')
    # parser.add_argument('--last_layer', type=str, default='',
    #                     help='')
    # parser.add_argument('--only_front', action='store_true', help='')
    parser.add_argument('--results_file', type=str, default='results.txt',
                        help='')
    parser.add_argument('--results_csv_file', type=str, default='results.csv',
                        help='')
    parser.add_argument('--results_arch_file', type=str, default='results_arch.json',
                        help='')
    # parser.add_argument('--sec_obj_range', type=float, nargs='+', default=[],
    #                     help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
    # parser.add_argument('--latency_table_file', type=str, default=None,
    #                     help='')
    parser.add_argument('--latency', action='store_true', help='')
    parser.add_argument('--zeroshot', action='store_true', help='')
    parser.add_argument('--tasks', type=str, nargs='+', default=['piqa','winogrande','hellaswag','arc_challenge','arc_easy', 'lambada_openai', 'boolq'])
    parser.add_argument('--zeroshot_csv_file', type=str, default=None,
                        help='')
    parser.add_argument('--lm_eval_batch_size', type=int, default=None,
                        help='')
    parser.add_argument('--num_fewshot', type=int, default=None,
                        help='')
    
    parser.add_argument('--use_key_token', action='store_true',
                        help='Only use key tokens for loss calculation (Long PPL/JSD)')
    parser.add_argument('--trunc_len', type=int, default=512,
                        help='truncation length for long PPL/JSD calculation')
    parser.add_argument('--sliding_window', type=int, default=128,
                        help='sliding_window length for long PPL/JSD calculation')
    parser.add_argument('--alpha', type=int, default=2,
                        help='Long-short distance (LSD) threshold for long PPL/JSD calculation')
    parser.add_argument('--beta', type=int, default=-2,
                        help='Long context likelihood (LCL) threshold for long PPL/JSD calculation')
    parser.add_argument('--key_token_path', type=str, default='',
                        help='')

    parser.add_argument('--longbench', action='store_true', help='')
    parser.add_argument('--longbench_e', action='store_true',
                        help='number of architectures desired')
    parser.add_argument('--longbench_result_path', type=str, default='',
                        help='')
    parser.add_argument('--longbench_config', type=str, default='',
                        help='')
    parser.add_argument('--pass_key_file', type=str, default='',
                        help='')

    parser.add_argument('--ruler', action='store_true', help='')
    parser.add_argument("--ruler_task", type=str, default=None, help="Task name", nargs="+",
                        choices=["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery", "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot"])
    # parser.add_argument("--max_seq_length", type=int, default=4096, 
                        # choices=[4096,8192,16384,32768,65536,131072,262144,524288,1048576], help="Maximum sequence length")
    parser.add_argument("--ruler_length", type=int, nargs='+', default=[4096])
    parser.add_argument('--ruler_yaml_path', type=str, default='',
                        help='')
    parser.add_argument("--ruler_sample", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--ruler_gen_toks", type=int, default=None, help="Number of tokens to generate")
    parser.add_argument("--ruler_batch_size", type=int, default=1, help="Batch size")
    parser.add_argument('--ruler_result_path', type=str, default='',
                        help='')

    
    # LongEval arguments
    parser.add_argument('--longeval', action='store_true', help='Enable LongEval lines task evaluation')
    parser.add_argument('--longeval_test_dir', type=str, default='',
                        help='Directory containing longeval test cases (should have lines/testcases/ subdirectory)')
    parser.add_argument('--longeval_num_lines', type=int, nargs='+', default=[200, 300, 400, 500, 600, 680, 700, 800, 900, 1000, 1100, 1200, 1350],
                        help='List of number of lines to test')
    parser.add_argument('--longeval_shortest_only', action='store_true', help='Only evaluate the shortest case')
    parser.add_argument('--longeval_result_path', type=str, default='',
                        help='Path to save LongEval results JSON file')
    


    cfgs = parser.parse_args()
    main(cfgs)
