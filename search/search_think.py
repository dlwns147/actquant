import os
import json
import torch
import argparse
import numpy as np
from evaluator import LlamaEvaluator
from tqdm import tqdm
from time import time
from copy import deepcopy
import csv

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.indicators.hv import Hypervolume
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PolynomialMutation

from search_space.llama_think import LlamaGroupSizeSearchSpaceThink
from predictor.factory import get_predictor
from utils.func import get_net_info, init_accelerator, set_seed, get_correlation
from utils.ga import MySampling, BinaryCrossover, MyMutation, IntPolynomialMutation, MyTwoPointCrossover, MyUniformCrossover, IntegerFromFloatMutation, IntMutation
from lm_eval.tasks import TaskManager, get_task_dict

import warnings
warnings.simplefilter("ignore")

class SearchThink:
    def __init__(self, config, accelerator, device_map, kwargs):
        self.args = deepcopy(kwargs)
        self.config = config
        self.device_map = device_map

        self.save_path = kwargs.pop('save', 'save')  # path to save results
        self.result_file = kwargs.pop('result_file', 'results.txt')  # path to save results
        self.resume = kwargs.pop('resume', None)  # resume search from a checkpoint
        self.iterations = kwargs.pop('iterations', 30)  # number of iterations to run search
        self.n_doe = kwargs.pop('n_doe', 100)  # number of architectures to train before fit surrogate model
        self.n_iter = kwargs.pop('n_iter', 8)  # number of architectures to train in each iteration
        self.predictor = kwargs.pop('predictor', 'rbf')  # which surrogate model to fit
        self.dataset = kwargs.pop('dataset', 'wikitext2')  # which dataset to run search on
        self.loss_func = kwargs.pop('loss_func', 'jsd')

        self.method = {'w': kwargs.pop('w_method', ['fp16']), 'kv': kwargs.pop('kv_method', 'think')}
        self.quant_model_paths = kwargs.pop('quant_model_paths', [])

        model_path = kwargs.pop('model_path', 'meta-llama')
        model_name = kwargs.pop('model_name', 'Llama-2-7b-hf')
        model_id=f'{model_path}/{model_name}'
        self.metric = kwargs.pop('metric', 'loss')
        self.limit = kwargs.pop('limit', 20)
        self.lm_eval_batch_size = kwargs.pop('lm_eval_batch_size', 1)
        self.num_fewshot = kwargs.pop('num_fewshot', None)
        outlier_path = kwargs.pop('outlier_path' , '')
        base_outlier_bits = sorted(kwargs.pop('base_outlier_bits', []))
        n_outlier = kwargs.pop('n_outlier' , 0)
        
        assert (outlier_path and base_outlier_bits and n_outlier > 0) or (not outlier_path and not base_outlier_bits and n_outlier == 0), "must use outlier_path, outlier_bits and n_outlier together when using outlier channel"
        
        outlier_bits = {l: [] for l in config['linear']}
        if outlier_path and base_outlier_bits and n_outlier > 0:
            for linear in config['linear']:
                for base_bits in base_outlier_bits:
                    _, in_dim = config['linear_shape'][linear]
                    avg_linear_bits = ((in_dim - n_outlier) * base_bits + n_outlier * 16) / (in_dim)
                    outlier_bits[linear].append(avg_linear_bits)

        w_bits = kwargs.pop('w_bits', [])
        assert len(w_bits) == len(self.quant_model_paths)
        k_bits = kwargs.pop('k_bits', [])
        v_bits = kwargs.pop('v_bits', [])
        bits = {'w': w_bits, 'k': k_bits, 'v': v_bits}
        self.bits = bits

        w_group_size = kwargs.pop('w_group_size', 128)
        k_group_size = kwargs.pop('k_group_size', [[128]])
        v_group_size = kwargs.pop('v_group_size', [[128]])
        self.group_size = {'w': w_group_size, 'k': k_group_size, 'v': v_group_size}

        # Pruning ratio options: 25%, 50%, 75%, 100% retention rate
        pruning_ratio_options = kwargs.pop('pruning_ratio_options', [0.25, 0.5, 0.75, 1.0])
        self.pruning_ratio_options = pruning_ratio_options

        self.residual_length = kwargs.pop('residual_length', 128)
        self.verbosity = kwargs.pop('verbosity', 'FATAL')
        self.task_manager = TaskManager(self.verbosity) if self.metric not in ['ppl', 'loss'] else None
        self.task_dict = get_task_dict([self.metric], self.task_manager) if self.metric not in ['ppl', 'loss'] else None
        
        self.comp_obj = kwargs.pop('comp_obj', ['wbits', 'kvbits'])  # second objective to optimize simultaneously
        self.comp_obj_min = kwargs.pop('comp_obj_min', [min(w_bits), min(k_bits)])
        self.comp_obj_max = kwargs.pop('comp_obj_max', [max(w_bits), max(k_bits)])        
        assert len(self.comp_obj) == len(self.comp_obj_min) and len(self.comp_obj_min) == len(self.comp_obj_max)
                
        self.n_token = kwargs.pop('n_token', 0)
        if 'memory' in self.comp_obj:
            assert self.n_token > 0, "n_token should be bigger than 0 when using memory objective."
            
        self.use_key_token = kwargs.pop('use_key_token', False)
        self.trunc_len = kwargs.pop('trunc_len', 512)
        self.sliding_window = kwargs.pop('sliding_window', 128)
        self.alpha = kwargs.pop('alpha', 2)
        self.beta = kwargs.pop('beta', -2)
        self.key_token_path = kwargs.pop('key_token_path', '')

        self.sensitivity_result_path = kwargs.pop('sensitivity_result_path', '')
        total_module = dict()
        total_sensitivity = dict()

        pass_module = {'w': [], 'k': [], 'v': []}        
        
        if self.sensitivity_result_path:
            for target in pass_module.keys():
                with open(os.path.join(self.sensitivity_result_path, f'{target}.csv'), 'r') as f:
                    module_list, sensitivity = list(csv.reader(f))
                    sensitivity = list(map(float, sensitivity))
                    total_module[target] = list(map(int, module_list)) if target in ['k', 'v'] else module_list
                    total_sensitivity[target] = sensitivity
            total_sensitivity_list = np.nan_to_num(np.concatenate(list(total_sensitivity.values())), nan=float('inf'))
            upper_bound = np.median(total_sensitivity_list) * kwargs.pop('sensitivity_threshold', 2)
            print(f'upper_bound: {upper_bound}')
            pass_idx_list = np.where(total_sensitivity_list > upper_bound)[0].tolist()

            start = 0
            for target in pass_module.keys():
                end = start + len(total_module[target])
                for idx in pass_idx_list:
                    if start <= idx and idx < end:
                        pass_module[target].append(total_module[target][idx - start])
                start = end

        self.pass_module = pass_module
        self.args['pass_module'] = pass_module
        print(f'pass_module: {pass_module}')

        self.evaluator = LlamaEvaluator(
            self.config,
            accelerator=accelerator,
            model_id=model_id,
            method=self.method,
            quant_model_paths=self.quant_model_paths,
            outlier=torch.load(outlier_path) if outlier_path else None,
            seqlen=kwargs.pop('seqlen', 2048),
            min_seqlen=kwargs.pop('min_seqlen', 0),
            n_sample=kwargs.pop('n_sample', 128),
            datasets=[self.dataset],
            data_batch_size=kwargs.pop('data_batch_size', 1),
            loss_func=self.loss_func,
            device_map=device_map,
            bits=bits,
            group_size=self.group_size,
            residual_length=self.residual_length,
            quant_kv_output=kwargs.pop('quant_kv_output', True),
            k_quant_scheme=kwargs.pop('k_quant_scheme', 'channel'),
            v_quant_scheme=kwargs.pop('v_quant_scheme', 'token'),
            n_token=self.n_token,
            limit=self.limit,
            lm_eval_batch_size=self.lm_eval_batch_size,
            num_fewshot=self.num_fewshot,
            task_manager=self.task_manager,
            task_dict=self.task_dict,
            verbosity=self.verbosity,
            use_key_token=self.use_key_token,
            trunc_len=self.trunc_len,
            sliding_window=self.sliding_window,
            alpha=self.alpha,
            beta=self.beta,
            key_token_path=self.key_token_path,
            pruning_ratio_options=self.pruning_ratio_options
        )
        self.search_space = LlamaGroupSizeSearchSpaceThink(
            bits=self.bits,
            group_size=self.group_size,
            pass_module=self.pass_module,
            comp_obj=self.comp_obj,
            comp_obj_min=self.comp_obj_min,
            comp_obj_max=self.comp_obj_max,
            config=self.config,
            n_token=self.n_token,
            outlier_bits=outlier_bits,
            only_outlier_bits=kwargs.pop('only_outlier_bits', False),
            pruning_ratio_options=self.pruning_ratio_options,
        )
        self.ga_pop_size = kwargs.pop('ga_pop_size', 40)
        self.subset_pop_size = kwargs.pop('subset_pop_size', 100)
        self.debug = kwargs.pop('debug', False)
        self.ga_algorithm = kwargs.pop('ga_algorithm', 'nsga2')
        self.max_value = kwargs.pop('max_value', 50)
        self.mut_prob = kwargs.pop('mut_prob', 0.1)
        self.crossover_prob = kwargs.pop('crossover_prob', 0.9)
        self.save_iter = kwargs.pop('save_iter', 1)
        accelerator.wait_for_everyone()
        
    def search(self, accelerator):
        total_start = time()
        start_it = 1
        
        if self.resume:
            archive, start_it = self._resume_from_dir()

        else:
            archive = []

            # Design Of Experiment
            if accelerator.is_main_process:
                if self.iterations < 1:
                    arch_doe = self.search_space.sample(
                        n_samples=self.n_doe,
                        pool=[x[0] for x in archive])
                else:
                    arch_doe = self.search_space.initialize(self.n_doe, pool=[x[0] for x in archive])
            else:
                arch_doe = list()
            arch_doe = accelerator.gather_for_metrics(arch_doe, use_gather_object=True)
            accelerator.wait_for_everyone()

            # parallel evaluation of arch_doe
            metric, complexity = self._evaluate(archs=arch_doe, accelerator=accelerator)

            if accelerator.is_main_process:
                # store evaluated / trained architectures
                for a, m, c in zip(arch_doe, metric, complexity):
                    archive.append([a, m, *c])

        if accelerator.is_main_process:
            # reference point (nadir point) for calculating hypervolume
            ref_pt = np.array([np.max([x[i] for x in archive]) for i in range(1, len(self.comp_obj) + 2)])
            accelerator.print(f'data preparation time : {time() - total_start:.2f}s')
        accelerator.wait_for_everyone()

        # main loop of the search
        for it in range(start_it, self.iterations + 1):
            if accelerator.is_main_process:
                accelerator.print(self.args)
                iter_start = time()

                # construct accuracy predictor surrogate model from archive
                predictor_start = time()
                metric_predictor, a_metric_pred = self._fit_predictor(archive, device=accelerator.device)
                predictor_time = time() - predictor_start

                # search for the next set of candidates for high-fidelity evaluation (lower level)
                next_start = time()
                candidates, c_metric_pred = self._next(archive, metric_predictor, self.n_iter)
                next_time = time() - next_start
            else:
                candidates = list()
            accelerator.wait_for_everyone()
            candidates = accelerator.gather_for_metrics(candidates, use_gather_object=True)

            # high-fidelity evaluation (lower level)
            c_metric, complexity = self._evaluate(archs=candidates, accelerator=accelerator)

            if accelerator.is_main_process:
                # check for accuracy predictor's performance
                rmse, rho, tau = get_correlation(
                    np.vstack((a_metric_pred, c_metric_pred)), np.array([x[1] for x in archive] + c_metric))

                # add to archive
                for a, m, c in zip(candidates, c_metric, complexity):
                    archive.append([a, m, *c])

                # calculate hypervolume
                hv = self._calc_hv(
                    ref_pt, np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)]))

                iter_time = time() - iter_start
                # print iteration-wise statistics
                accelerator.print(f"Iter {it}: hv = {hv:.2f}, iter time : {(time() - iter_start):.2f}s, predictor_time : {predictor_time:.2f}, next_time : {next_time:.2f}")
                accelerator.print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall's Tau = {tau:.4f}")
                accelerator.print(f'iteration time : {iter_time:.2f}s')

                # dump the statistics
                if it % self.save_iter == 0:
                    os.makedirs(self.save_path, exist_ok=True)
                    with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                        json.dump({'archive': archive, 'candidates': archive[-self.n_iter:], 'hv': hv,
                                'surrogate': {
                                    'model': self.predictor, 'name': metric_predictor.name,
                                    'winner': metric_predictor.winner if self.predictor == 'as' else metric_predictor.name,
                                    'rmse': rmse, 'rho': rho, 'tau': tau, 'total_time': iter_time}, 'iteration' : it}, handle)
                    if self.debug:
                        import matplotlib.pyplot as plt
                        n_obj = len(self.comp_obj)
                        comp_np = np.array(complexity)
                        fig, axes = plt.subplots(nrows=1, ncols=n_obj, figsize=(5 * n_obj, 5))
                        if not isinstance(axes, np.ndarray):
                            axes = [axes]
                        for i in range(n_obj):
                            comp = np.array([x[i+2] for x in archive])  # comp obj
                            perf = np.array([x[1] for x in archive])  # performance
                            axes[i].scatter(comp, perf, s=5, facecolors='none', edgecolors='b', label='archive')
                            comp = comp_np[:, i]
                            perf = np.array(c_metric)
                            axes[i].scatter(comp, perf, s=10, color='r', label='candidates evaluated')
                            perf = c_metric_pred[:, 0]
                            axes[i].scatter(comp, perf, s=10, facecolors='none', edgecolors='g', label='candidates predicted')
                            axes[i].legend(loc="upper right")
                            axes[i].set_xlabel(f'f{i+2}')
                            axes[i].grid(c='0.8') 
                        axes[0].set_ylabel('f1')
                        fig.tight_layout() 
                        plt.savefig(os.path.join(self.save_path, 'iter_{}.png'.format(it)))
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            total_time_elapsed = time() - total_start
            accelerator.print(f'total time elapsed : {total_time_elapsed:.2f}s')

            sentences = []
            for k, v in self.args.items():
                sentences.append(f"{k}: {v}\n")
            sentences.append(f'Total time: {total_time_elapsed:.2f}s')

            with open(os.path.join(self.save_path, self.result_file), 'w') as f:
                for sentence in sentences:
                    f.write(sentence)

            accelerator.print(self.args)
        return

    def _resume_from_dir(self):
        """ resume search from a previous iteration """

        with open(self.resume, 'r') as f:
            resume_file = json.load(f)
            archive = resume_file['archive'] + resume_file['candidates']
            it = resume_file['iteration']

        return archive, it + 1

    def _evaluate(self, archs, accelerator):
        metric_list, complexity_list = [], []
        for arch in tqdm(archs, desc='Eval Arch'):
            metric, complexity = self.evaluator.eval(accelerator=accelerator, arch=arch, metric=self.metric, loss_func=self.loss_func)
            metric_list.append(min(self.max_value, np.nan_to_num(list(metric.values())[0], nan=self.max_value)))
            complexity_list.append([complexity[obj] for obj in self.comp_obj])

        return metric_list, complexity_list

    def _fit_predictor(self, archive, device='cpu'):
        inputs = np.array([self.search_space.encode_predictor(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])

        kwargs = {}
        if self.predictor == 'rbf':
            n_block = self.config['n_block']
            n_linear = self.config['n_linear']
            lb = np.zeros((n_linear + 3, n_block))  # +3 for k, v, and pruning_ratio
            ub = np.ones((n_linear + 3, n_block))
            
            for linear_idx, linear in enumerate(self.config['linear']):
                ub[linear_idx] = len(getattr(self.search_space, f"{linear.split('.')[-1]}_option")) - 1            
            ub[n_linear] = len(self.search_space.k_option) - 1
            ub[n_linear + 1] = len(self.search_space.v_option) - 1
            ub[n_linear + 2] = len(self.search_space.pruning_ratio_option) - 1  # pruning_ratio
            
            lb = np.delete(lb.flatten(), self.search_space.pass_idx_list, axis=-1)
            ub = np.delete(ub.flatten(), self.search_space.pass_idx_list, axis=-1)

            kwargs = {'lb': lb, 'ub': ub}

        metric_predictor = get_predictor(self.predictor, inputs, targets, device=device, **kwargs)

        return metric_predictor, metric_predictor.predict(inputs)
    
    def _next(self, archive, predictor, K):
        """ searching for next K candidate for high-fidelity evaluation (lower level) """

        # get non-dominated architectures from archive
        F = np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)])
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # non-dominated arch bit-strings
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # initiate a multi-objective solver to optimize the problem
        method = NSGA2(pop_size=self.ga_pop_size, sampling=nd_X,
            crossover=BinomialCrossover(prob=self.crossover_prob, n_offsprings=1),
            mutation=IntMutation(prob=self.mut_prob),
            eliminate_duplicates=True)
        
        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblemThink(self.search_space, predictor, self.config, self.comp_obj, self.comp_obj_max, self.comp_obj_min, self.group_size, self.n_token)
        
        # kick-off the search
        res = minimize(problem, method, termination=('n_gen', 20), save_history=True, verbose=True)
        
        # check for duplicates
        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])
        print(f'not_duplicate : {sum(not_duplicate)}')

        pop = res.pop[not_duplicate]
        if sum(not_duplicate) >= K:
            indices = self._subset_selection(pop, F[front, 1:], K, self.subset_pop_size)
            pop = pop[indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        # decode integer bit-string to config and also return predicted top1_err
        return candidates, predictor.predict(self.search_space.decode_encode_predictor(pop.get("X")))

    def _subset_selection(self, pop, nd_F, K, pop_size):
        problem = SubsetProblem(pop.get("F")[:, 1:], nd_F, K, len(self.comp_obj))
        algorithm = GA(
            pop_size=pop_size, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)

        res = minimize(
            problem, algorithm, ('n_gen', 60), verbose=False)

        return res.X

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        # calculate hypervolume on the non-dominated set of F
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv = Hypervolume(ref_point=ref_point).do(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv


class AuxiliarySingleLevelProblemThink(Problem):
    """ The optimization problem for finding the next N candidate architectures with pruning ratio """

    def __init__(self, search_space, predictor, config, comp_obj, comp_obj_max, comp_obj_min, group_size, n_token):
        n_block, n_linear = search_space.n_block, search_space.n_linear
        n_comp_obj = len(search_space.comp_obj)
        super().__init__(n_var=n_block * (n_linear + 3), n_obj=n_comp_obj + 1, n_constr=2 * n_comp_obj, type_var=int)  # +3 for k, v, pruning_ratio

        self.ss = search_space
        self.predictor = predictor
        self.comp_obj = comp_obj
        self.comp_obj_max = comp_obj_max
        self.comp_obj_min = comp_obj_min
        self.config = config
        self.group_size = group_size
        self.n_token = n_token
        self.xl = np.zeros((n_linear + 3, n_block))
        self.xu = np.ones((n_linear + 3, n_block))
        
        for linear_idx, linear in enumerate(config['linear']):
            self.xu[linear_idx] = len(getattr(search_space, f"{linear.split('.')[-1]}_option")) - 1
        self.xu[n_linear] = len(search_space.k_option) - 1
        self.xu[n_linear + 1] = len(search_space.v_option) - 1
        self.xu[n_linear + 2] = len(search_space.pruning_ratio_option) - 1  # pruning_ratio
        
        for pass_w_linear in search_space.pass_module['w']:
            blk, linear = pass_w_linear.split('.', 1)
            linear_idx =  config['linear'].index(linear)
            self.xl[linear_idx, int(blk)] = len(getattr(search_space, f"{linear.split('.')[-1]}_option")) - 1
        for pass_k_layer in search_space.pass_module['k']:
            self.xl[n_linear, pass_k_layer] = len(search_space.k_option) - 1
        for pass_v_layer in search_space.pass_module['v']:
            self.xl[n_linear + 1, pass_v_layer] = len(search_space.v_option) - 1
        # pruning_ratio doesn't have pass_module, so no need to set xl

        self.xl = self.xl.flatten()
        self.xu = self.xu.flatten()

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)
        g = np.full((x.shape[0], self.n_constr), np.nan)

        metrics = self.predictor.predict(self.ss.decode_encode_predictor(x))[:, 0]

        for i, (_x, metric) in enumerate(zip(x, metrics)):
            arch = self.ss.decode(_x)
            info = get_net_info(arch, self.config, self.group_size, n_token=self.n_token)
            f[i, 0] = metric
            for j in range(len(self.comp_obj)):
                f[i, 1 + j] = info[self.comp_obj[j]]
                g[i, 2 * j] = 1 - info[self.comp_obj[j]] / self.comp_obj_min[j]
                g[i, 2 * j + 1] = info[self.comp_obj[j]] / self.comp_obj_max[j] - 1

        out["F"] = f
        out["G"] = g

class SubsetProblem(Problem):
    """ select a subset to diversify the pareto front """
    def __init__(self, candidates, archive, K, n_obj):
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1, type_var=bool)
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            tmp = np.sort(np.concatenate((self.archive, self.candidates[_x])), axis=0)
            f[i, 0] = np.std(np.diff(tmp, axis=0))
            g[i, 0] = (self.n_max - np.sum(_x)) ** 2

        out["F"] = f
        out["G"] = g

def main(args):
    set_seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)
    engine = SearchThink(config=config, accelerator=accelerator, device_map=device_map, kwargs=vars(args))
    engine.search(accelerator)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='save',
                        help='location of dir to save')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume search from a checkpoint')
    parser.add_argument('--iterations', type=int, default=50,
                        help='number of search iterations')
    parser.add_argument('--n_doe', type=int, default=100,
                        help='initial sample size for DOE')
    parser.add_argument('--n_iter', type=int, default=8,
                        help='number of architectures to high-fidelity eval (low level) in each iteration')
    parser.add_argument('--predictor', type=str, default='rbf',
                        help='which accuracy predictor model to fit (rbf/gp/cart/mlp/as)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
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
    parser.add_argument('--k_group_size', type=int, nargs='+', action='append', default=[],
                        help='')
    parser.add_argument('--v_group_size', type=int, nargs='+', action='append', default=[],
                        help='')
    
    parser.add_argument('--residual_length', type=int, default=128, 
                        help='')

    parser.add_argument('--quant_kv_output', action='store_true', help='')
    parser.add_argument('--k_quant_scheme', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_scheme', type=str, choices=['channel', 'token'], 
                        help='')
    
    parser.add_argument('--comp_obj', type=str, nargs='+', default=['wbits', 'kvbits'], choices=['wbits', 'kvbits', 'memory'], 
                        help='complexity objectives to optimize simultaneously')
    parser.add_argument('--comp_obj_min', type=float, nargs='+', default=[2, 2], 
                        help='')
    parser.add_argument('--comp_obj_max', type=float, nargs='+', default=[4, 4], 
                        help='')
    
    parser.add_argument('--pruning_ratio_options', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0],
                        help='Pruning ratio options (retention rate): 25%%, 50%%, 75%%, 100%%')
    
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset name')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for selecting calibration set, etc.')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='sample number of the calibration set')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='sequential length of the calibaration (train) set')
    parser.add_argument('--min_seqlen', type=int, default=0,
                        help='minimum sequential length of the calibaration gsm8k set')
    parser.add_argument('--metric', type=str, default='ppl',
                        help='which metric predictor model to fit (ppl/loss/gsm8k)')
    parser.add_argument('--data_batch_size', type=int, default=1,
                        help='sequential length of the calibaration (train) set')
    parser.add_argument('--limit', type=int, default=None,
                        help='')
    parser.add_argument('--lm_eval_batch_size', type=int, default=None,
                        help='batch size for measuring lm_eval tasks.')
    parser.add_argument('--num_fewshot', type=int, default=None,
                        help='# fewshot sample for measuring lm_eval tasks.')
    parser.add_argument('--verbosity', type=str, default='INFO',
                        help='verbosity for measuring lm_eval tasks.')
    
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='config file to read the model meta data')
    parser.add_argument('--ga_pop_size', type=int, default=40,
                        help='population size of the NSGA stage')
    parser.add_argument('--subset_pop_size', type=int, default=100,
                        help='population size of the subset selection stage')
    parser.add_argument('--debug', action='store_true', help='visualization of each iteration results')
    parser.add_argument('--result_file', type=str, default='results.txt',
                        help='')
    parser.add_argument('--ga_algorithm', type=str, default='nsga2',
                        help='')

    parser.add_argument('--max_value', type=float, default=50,
                        help='')
    parser.add_argument('--crossover_prob', type=float, default=0.9,
                        help='')
    parser.add_argument('--mut_prob', type=float, default=0.1,
                        help='')
    parser.add_argument('--loss_func', type=str, default='cross_entropy',
                        help='')
    parser.add_argument('--base_outlier_bits', type=int, nargs='+', default=[], 
                        help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
    parser.add_argument('--n_outlier', type=int, default=0, 
                        help='')
    parser.add_argument('--only_outlier_bits', action='store_true', help='')
    parser.add_argument('--sensitivity_result_path', type=str, default='',
                        help='')
    parser.add_argument('--save_iter', type=int, default=1, 
                        help='')
    parser.add_argument('--sensitivity_threshold', type=int, default=2,
                        help='')
    
    parser.add_argument('--n_token', type=int, default=0, 
                        help='target sequence length for memory calculation')

    
    parser.add_argument('--use_key_token', action='store_true', help='Only use key tokens for loss calculation (Long PPL/JSD)')
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
    
    parser.add_argument('--packing', action='store_true', help='Only use key tokens for loss calculation (Long PPL/JSD)')
    
    cfgs = parser.parse_args()
    main(cfgs)
