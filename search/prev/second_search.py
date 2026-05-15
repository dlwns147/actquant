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
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import Hypervolume
from pymoo.algorithms.soo.nonconvex.ga import GA

from search_space.llama import LlamaGroupSizeSearchSpace
from predictor.factory import get_predictor
from utils.func import get_net_info, init_accelerator, set_seed, get_correlation
from utils.ga import MySampling, BinaryCrossover, MyMutation

import warnings
warnings.simplefilter("ignore")

"""
SecondSearch: Two-stage search for W/KV quantization configurations

This script performs a two-stage search:
1. First, use search.py to find pareto frontiers for:
   - [performance, weights bits] 
   - [performance, kv caches vbits]
2. Then combine W/KV pareto frontiers and use a predictor to:
   - Predict longppl/longjsd for all combinations
   - Select K solutions from [performance/average weights bits/average kv bits] pareto frontier
   - Actually measure longppl/longjsd for selected solutions
   - Retrain predictor and repeat

Usage:
    First, run search.py twice to generate pareto frontiers:
        python search.py --comp_obj wbits --save w_search_results ...
        python search.py --comp_obj kvbits --save kv_search_results ...
    
    Then run this script:
        python second_search.py --w_pareto_path w_search_results/iter_N.stats --kv_pareto_path kv_search_results/iter_N.stats ...
"""


class SecondSearch:
    def __init__(self, config, accelerator, device_map, kwargs):
        self.args = deepcopy(kwargs)
        self.config = config
        self.device_map = device_map

        self.save_path = kwargs.pop('save', 'save')
        self.result_file = kwargs.pop('result_file', 'results.txt')
        self.resume = kwargs.pop('resume', None)
        self.iterations = kwargs.pop('iterations', 10)
        self.n_doe = kwargs.pop('n_doe', 50)  # initial random samples for predictor
        assert self.n_doe > 4, "n_doe must be greater than 4 to include 4 extreme combinations"
        self.n_iter = kwargs.pop('n_iter', 5)  # number of solutions to measure per iteration
        self.predictor = kwargs.pop('predictor', 'rbf')
        # initial sampling mode: 'random' (default) or 'pareto_random'
        # - 'random': original behavior (random + 4 extreme (wbits/kvbits) combinations)
        # - 'pareto_random': sample only from pareto-frontier of [metric, w_bits, kv_bits]
        self.init_mode = kwargs.pop('init_mode', 'random')
        self.metric = kwargs.pop('metric', 'loss')
        self.dataset = kwargs.pop('dataset', 'wikitext2')
        self.loss_func = kwargs.pop('loss_func', 'cross_entropy')
        self.method = {'w': kwargs.pop('w_method', ['fp16']), 'kv': kwargs.pop('kv_method', 'kivi')}
        self.quant_model_paths = kwargs.pop('quant_model_paths', [])

        model_path = kwargs.pop('model_path', 'meta-llama')
        model_name = kwargs.pop('model_name', 'Llama-2-7b-hf')
        model_id = f'{model_path}/{model_name}'
        outlier_path = kwargs.pop('outlier_path', '')
        base_outlier_bits = sorted(kwargs.pop('base_outlier_bits', []))
        n_outlier = kwargs.pop('n_outlier', 0)

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
        self.residual_length = kwargs.pop('residual_length', 128)
        self.comp_obj = ['wbits', 'kvbits']  # For compatibility with _evaluate
        
        self.n_token = kwargs.pop('n_token', 0)
        self.use_key_token = kwargs.pop('use_key_token', False)
        self.trunc_len = kwargs.pop('trunc_len', 4096)
        self.sliding_window = kwargs.pop('sliding_window', 1024)
        self.alpha = kwargs.pop('alpha', 2)
        self.beta = kwargs.pop('beta', -2)
        self.key_token_path = kwargs.pop('key_token_path', '')

        # self.sensitivity_result_path = kwargs.pop('sensitivity_result_path', '')
        # total_module = dict()
        # total_sensitivity = dict()

        # self.pass_module = {'w': [], 'k': [], 'v': []}        
        
        # if self.sensitivity_result_path:
        #     for target in self.pass_module.keys():
        #         with open(os.path.join(self.sensitivity_result_path, f'{target}.csv'), 'r') as f:
        #             module_list, sensitivity = list(csv.reader(f))
        #             sensitivity = list(map(float, sensitivity))
        #             total_module[target] = list(map(int, module_list)) if target in ['k', 'v'] else module_list
        #             total_sensitivity[target] = sensitivity
        #     total_sensitivity_list = np.nan_to_num(np.concatenate(list(total_sensitivity.values())), nan=float('inf'))
        #     upper_bound = np.median(total_sensitivity_list) * kwargs.pop('sensitivity_threshold', 2)
        #     print(f'upper_bound: {upper_bound}')
        #     pass_idx_list = np.where(total_sensitivity_list > upper_bound)[0].tolist()

        #     start = 0
        #     for target in self.pass_module.keys():
        #         end = start + len(total_module[target])
        #         for idx in pass_idx_list:
        #             if start <= idx < end:
        #                 self.pass_module[target].append(total_module[target][idx - start])
        #         start = end

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
            use_key_token=self.use_key_token,
            trunc_len=self.trunc_len,
            sliding_window=self.sliding_window,
            alpha=self.alpha,
            beta=self.beta,
            key_token_path=self.key_token_path
        )

        self.w_pareto_path = kwargs.pop('w_pareto_path', '')
        self.kv_pareto_path = kwargs.pop('kv_pareto_path', '')
        self.max_value = kwargs.pop('max_value', 50)
        self.save_iter = kwargs.pop('save_iter', 1)
        self.debug = kwargs.pop('debug', False)
        # population size for GA-based subset selection (diversifying w_bits / kv_bits)
        self.subset_pop_size = kwargs.pop('subset_pop_size', 100)
        accelerator.wait_for_everyone()

    def load_pareto_frontiers(self):
        """Load W and KV pareto frontiers from search.py results"""
        assert self.w_pareto_path and self.kv_pareto_path, "w_pareto_path and kv_pareto_path must be provided"
        
        with open(self.w_pareto_path, 'r') as f:
            w_result = json.load(f)
            w_archive = w_result.get('archive', []) + w_result.get('candidates', [])
        
        with open(self.kv_pareto_path, 'r') as f:
            kv_result = json.load(f)
            kv_archive = kv_result.get('archive', []) + kv_result.get('candidates', [])
        
        # Extract pareto frontiers
        # Optimize: extract all data in single pass
        w_subnets = []
        w_metrics = []
        w_complexity = []
        for v in w_archive:
            arch = v[0]
            net_info = get_net_info(arch, self.config, self.group_size, n_token=self.n_token)
            w_subnets.append(arch)
            w_metrics.append(v[1])
            w_complexity.append(net_info['wbits'])
        
        w_F = np.column_stack((w_metrics, w_complexity))
        w_front = NonDominatedSorting().do(w_F, only_non_dominated_front=True)
        # (arch, metric, complexity) and sort by complexity (ascending)
        w_pareto = [(w_subnets[i], w_metrics[i], w_complexity[i]) for i in w_front]
        w_pareto.sort(key=lambda x: x[-1])

        kv_subnets = []
        kv_metrics = []
        kv_complexity = []
        for v in kv_archive:
            arch = v[0]
            net_info = get_net_info(arch, self.config, self.group_size, n_token=self.n_token)
            kv_subnets.append(arch)
            kv_metrics.append(v[1])
            kv_complexity.append(net_info['kvbits'])
        
        kv_F = np.column_stack((kv_metrics, kv_complexity))
        kv_front = NonDominatedSorting().do(kv_F, only_non_dominated_front=True)
        kv_pareto = [(kv_subnets[i], kv_metrics[i], kv_complexity[i]) for i in kv_front]
        kv_pareto.sort(key=lambda x: x[-1])

        return w_pareto, kv_pareto

    def combine_pareto_frontiers(self, w_pareto, kv_pareto):
        """Combine W and KV pareto frontiers to create all combinations"""
        # Optimize: use list comprehension for better performance
        linear_keys = self.config['linear']  # Cache to avoid repeated dict access
        combinations = [
            [
                {'w': {l: w_arch['w'][l] for l in linear_keys}, 'k': kv_arch['k'], 'v': kv_arch['v']},
                w_perf + kv_perf,
                w_bits,
                kv_bits,
                w_idx,
                kv_idx
            ]
            for w_idx, (w_arch, w_perf, w_bits) in enumerate(w_pareto)
            for kv_idx, (kv_arch, kv_perf, kv_bits) in enumerate(kv_pareto)
        ]
        return combinations        

    def _find_arch_in_archive(self, arch, archive):
        """Find if arch exists in archive and return its metric"""
        # Direct structural comparison to avoid expensive JSON serialization
        for archived_arch, metric, _, _ in archive:
            if archived_arch == arch:
                return metric
        return None

    def _encode_combination(self, combination):
        """Encode combination using [w_pareto_index, kv_pareto_index] instead of llamasearchspace
        
        Note: This method is kept for backward compatibility but is no longer
        used in optimized code paths. Direct array indexing is preferred.
        """
        # combination format: [arch_dict, metric, w_bits, kv_bits, w_idx, kv_idx]
        w_idx = combination[4]
        kv_idx = combination[5]
        return np.array([w_idx, kv_idx], dtype=float)

    def _evaluate(self, archs, accelerator):
        """Evaluate architectures and return metrics and complexity"""
        metric_list, complexity_list = [], []
        for arch in tqdm(archs, desc='Eval Arch'):
            metric, complexity = self.evaluator.eval(accelerator=accelerator, arch=arch, metric=self.metric, loss_func=self.loss_func)
            metric_list.append(min(self.max_value, np.nan_to_num(list(metric.values())[0], nan=self.max_value)))
            complexity_list.append([complexity[obj] for obj in self.comp_obj])

        return metric_list, complexity_list

    def _fit_predictor(self, archive, device='cpu'):
        """Fit predictor on archive data"""
        # Archive entries: [arch_dict, metric, w_bits, kv_bits, w_idx, kv_idx]
        # Use [w_idx, kv_idx] for encoding instead of llamasearchspace.encode_predictor
        # Optimize: directly extract indices instead of calling _encode_combination
        inputs = np.array([[x[4], x[5]] for x in archive], dtype=float)
        targets = np.array([x[1] for x in archive], dtype=float)

        kwargs = {}
        if self.predictor == 'rbf':
            # For 2D encoding [w_idx, kv_idx], bounds are:
            # w_idx: [0, len(w_pareto)-1]
            # kv_idx: [0, len(kv_pareto)-1]
            lb = np.array([0.0, 0.0])
            ub = np.array([len(self.w_pareto) - 1, len(self.kv_pareto) - 1])

            kwargs = {'lb': lb, 'ub': ub}

        metric_predictor = get_predictor(self.predictor, inputs, targets, device=device, **kwargs)
        return metric_predictor, metric_predictor.predict(inputs)

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        """Calculate hypervolume on the non-dominated set of F"""
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv = Hypervolume(ref_point=ref_point).do(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv

    def _build_index_to_metric(self, combinations, archive):
        """
        Build mapping from combination index to measured metric
        for *legacy* archives that store full architectures.

        This is only used when resuming from old JSON files where
        each archive entry has the form:
            [arch_dict, metric, w_bits, kv_bits]
        """
        index_to_metric = {}
        for i, c in enumerate(combinations):
            arch = c[0]
            for archived_arch, metric, _, _ in archive:
                if archived_arch == arch:
                    index_to_metric[i] = metric
                    break
        return index_to_metric
 
    def search(self, accelerator):
        """Main search loop"""
        total_start = time()
        
        # Step 1: Load W and KV pareto frontiers        
        w_pareto, kv_pareto = self.load_pareto_frontiers()
        # Store as instance variables for use in _fit_predictor
        self.w_pareto = w_pareto
        self.kv_pareto = kv_pareto
        
        # Step 2: Combine pareto frontiers
        combinations = self.combine_pareto_frontiers(w_pareto, kv_pareto)
        # Cache complexity (w_bits, kv_bits) for all combinations as numpy arrays.
        # These never change during search, so we avoid repeatedly rebuilding
        # python lists and stacks in every iteration.
        w_bits_arr = np.array([c[2] for c in combinations], dtype=float)
        kv_bits_arr = np.array([c[3] for c in combinations], dtype=float)
        # Metrics can be updated by the predictor and by real evaluations.
        # Keep them in a separate array for fast vectorized operations.
        metric_arr = np.array([c[1] for c in combinations], dtype=float)
        # Pre-compute encoded representations for all combinations once.
        # This avoids calling encode_predictor on ~200K combinations at
        # every iteration, which is very expensive.
        # Use [w_idx, kv_idx] encoding instead of llamasearchspace.encode_predictor
        # Vectorized encoding: directly extract indices from combinations
        encoded_combinations = np.array([[c[4], c[5]] for c in combinations], dtype=float)
        # Track which combination indices have been actually evaluated
        # (i.e., real metric measured by evaluator, not surrogate).
        evaluated_mask = np.zeros(len(combinations), dtype=bool)
        index_to_metric = {}
        
        if accelerator.is_main_process:
            accelerator.print(f"W pareto: {len(w_pareto)}, KV pareto: {len(kv_pareto)}, Total combinations: {len(combinations)}")
        accelerator.wait_for_everyone()
 
        # Step 3: Initial random sampling and predictor training
        
        if self.resume:
            # When resuming, rebuild both the in-memory archive
            # (storing full architectures) and the index_to_metric
            # mapping from the JSON file, which now stores indices
            # instead of full architectures.
            archive, index_to_metric, start_it = self._resume_from_dir(combinations)
            if accelerator.is_main_process:
                for idx in index_to_metric.keys():
                    evaluated_mask[idx] = True
                    metric_arr[idx] = index_to_metric[idx]

        else:
            archive = []
            start_it = 1
            if accelerator.is_main_process:
                # Initial sampling for predictor training
                # Branch by init_mode: 'random' (original) vs 'pareto_random'
                n_total = len(combinations)
                assert n_total > 4, "Total number of W/KV combinations must be greater than 4."

                if self.init_mode == 'pareto_random':
                    # 1) Build objective matrix: [metric, w_bits, kv_bits]
                    # Use cached arrays instead of list comprehensions
                    F_init = np.column_stack([
                        metric_arr,
                        w_bits_arr,
                        kv_bits_arr,
                    ])

                    # 2) Get pareto-frontier indices
                    pareto_indices = NonDominatedSorting().do(F_init, only_non_dominated_front=True)
                    pareto_indices = np.array(pareto_indices, dtype=int)

                    # Target DOE size (cannot exceed total combinations)
                    target_n_doe = min(self.n_doe, n_total)

                    selected_indices = []

                    n_pareto = len(pareto_indices)
                    if n_pareto > 0:
                        # 3) Select up to 4 extreme points in terms of w_bits / kv_bits within pareto-frontier
                        # Use cached arrays for faster access
                        w_bits_list = w_bits_arr[pareto_indices]
                        kv_bits_list = kv_bits_arr[pareto_indices]

                        w_min_idx = pareto_indices[np.argmin(w_bits_list)]
                        w_max_idx = pareto_indices[np.argmax(w_bits_list)]
                        kv_min_idx = pareto_indices[np.argmin(kv_bits_list)]
                        kv_max_idx = pareto_indices[np.argmax(kv_bits_list)]

                        extreme_indices = []
                        for idx_val in (w_min_idx, w_max_idx, kv_min_idx, kv_max_idx):
                            if idx_val not in extreme_indices:
                                extreme_indices.append(idx_val)

                        # 4) Fill the rest (within pareto-frontier) randomly up to min(target_n_doe, #pareto)
                        max_from_pareto = min(target_n_doe, n_pareto)
                        remaining_needed_pareto = max(0, max_from_pareto - len(extreme_indices))

                        remaining_pareto_pool = [i for i in pareto_indices if i not in extreme_indices]
                        random_pareto_indices = []
                        if remaining_needed_pareto > 0 and len(remaining_pareto_pool) > 0:
                            random_pareto_indices = np.random.choice(
                                remaining_pareto_pool,
                                size=min(remaining_needed_pareto, len(remaining_pareto_pool)),
                                replace=False,
                            ).tolist()

                        selected_indices = extreme_indices + random_pareto_indices

                    # 5) If Pareto points are fewer than target_n_doe, sample remaining from non-Pareto combinations
                    remaining_needed_global = max(0, target_n_doe - len(selected_indices))
                    if remaining_needed_global > 0:
                        all_indices = np.arange(n_total, dtype=int)
                        if selected_indices:
                            mask = np.ones(n_total, dtype=bool)
                            mask[selected_indices] = False
                            global_pool = all_indices[mask]
                        else:
                            global_pool = all_indices

                        if len(global_pool) > 0:
                            random_global_indices = np.random.choice(
                                global_pool,
                                size=min(remaining_needed_global, len(global_pool)),
                                replace=False,
                            ).tolist()
                            selected_indices.extend(random_global_indices)

                    idx = np.array(sorted(set(selected_indices)), dtype=int)
                else:
                    # Original behavior:
                    # Random sampling for initial predictor training
                    # Always include 4 extreme combinations based on complexity (2 extremes from each of W / KV)
                    n_doe = min(self.n_doe, n_total)

                    # w_pareto / kv_pareto are already sorted by complexity (ascending)
                    extreme_indices = [
                        wi * len(kv_pareto) + ki
                        for wi in [0, len(w_pareto) - 1]
                        for ki in [0, len(kv_pareto) - 1]
                    ]
                    # Filter invalid indices just in case w_pareto/kv_pareto are smaller
                    extreme_indices = [
                        idx for idx in extreme_indices
                        if 0 <= idx < n_total
                    ]

                    # We want up to 4 extreme points if possible
                    remaining_needed = max(0, n_doe - len(extreme_indices))
                    remaining_pool = list(set(range(n_total)) - set(extreme_indices))
                    random_indices = (
                        np.random.choice(
                            remaining_pool,
                            size=min(remaining_needed, len(remaining_pool)),
                            replace=False,
                        ).tolist()
                        if remaining_needed > 0 and len(remaining_pool) > 0
                        else []
                    )

                    idx = np.array(sorted(set(extreme_indices + random_indices)), dtype=int)

                archive_doe = [combinations[i] for i in idx]
            else:
                archive_doe = []
        
            accelerator.wait_for_everyone()
            archive_doe = accelerator.gather_for_metrics(archive_doe, use_gather_object=True)
        
            # Evaluate random samples
            metrics, _ = self._evaluate([c[0] for c in archive_doe], accelerator)
            
            if accelerator.is_main_process:
                for comb_idx, (c, m) in zip(idx, zip(archive_doe, metrics)):
                    archive.append([c[0], m, c[2], c[3], c[4], c[5]])  # [arch, metric, w_bits, kv_bits, w_idx, kv_idx]
                    # Update mapping, mask and metric_arr with true measured metric
                    comb_idx = int(comb_idx)
                    index_to_metric[comb_idx] = m
                    evaluated_mask[comb_idx] = True
                    metric_arr[comb_idx] = float(m)

        accelerator.wait_for_everyone()
 
        # Set reference point (nadir point) for hypervolume once, based on initial archive.
        # Assume that after DOE/resume the archive always contains at least one
        # evaluated solution (len(archive) >= 1), so we do not handle the empty case.
        if accelerator.is_main_process:
            archive_array = np.array([[x[1], x[2], x[3]] for x in archive])
            ref_pt = np.max(archive_array, axis=0)

        # Main iteration loop
        for it in range(start_it, self.iterations + 1):
            if accelerator.is_main_process:
                accelerator.print(f"\n=== Iteration {it} ===")
                iter_start = time()

                # Fit predictor
                predictor_start = time()
                predictor, a_metrics_pred = self._fit_predictor(archive, device=accelerator.device)
                
                # Predict for all combinations that have not been
                # actually evaluated yet. We reuse the pre-computed
                # encoded_combinations to avoid calling
                # encode_predictor on every architecture each
                # iteration.
                indices_to_predict = np.where(~evaluated_mask)[0]

                # accelerator.print(f"Predicting {len(indices_to_predict)} architectures...")
                encoded_archs = encoded_combinations[indices_to_predict]
                predictions = predictor.predict(encoded_archs)[:, 0]
                predictor_time = time() - predictor_start
                # Vectorized update: update both combinations list and cached metric array
                pred_vals = predictions.astype(float)
                metric_arr[indices_to_predict] = pred_vals
                # Update combinations list (for compatibility)
                for i, comb_idx in enumerate(indices_to_predict):
                    combinations[comb_idx][1] = float(pred_vals[i])
                
                # Find pareto solutions
                # Create objective matrix: [performance, w_bits, kv_bits]
                # Use cached numpy arrays to avoid repeatedly constructing
                # large python lists on every iteration.
                F = np.column_stack([metric_arr, w_bits_arr, kv_bits_arr])
                
                # Get pareto frontier
                front = NonDominatedSorting().do(F, only_non_dominated_front=True)
                pareto_indices = np.array(front, dtype=int)
                
                # Filter out archs that are already in archive
                # candidates: pareto solutions not in archive
                # Optimize: use boolean indexing for vectorized filtering
                pareto_candidates = pareto_indices[~evaluated_mask[pareto_indices]].tolist()
                
                # import pdb; pdb.set_trace()
                # Select K solutions from pareto candidates
                if len(pareto_candidates) <= self.n_iter:
                    # If we have fewer (or equal) candidates than K, take all.
                    selected_indices = pareto_candidates
                else:
                    # Use GA-based subset selection (adapted from `search.py`)
                    # to select K candidates that are spread along the
                    # (w_bits, kv_bits) axes.

                    # Build objective matrices.
                    # archive_F_full: [metric, w_bits, kv_bits] for archive pareto
                    # cand_F: [w_bits, kv_bits] for candidate solutions
                    # Optimize: extract arrays once and stack.
                    # Assume archive is always non-empty here (len(archive) >= 1).
                    archive_array = np.array([[x[1], x[2], x[3]] for x in archive])
                    archive_F_full = archive_array
                    archive_front = NonDominatedSorting().do(
                        archive_F_full, only_non_dominated_front=True
                    )
                    # SubsetProblem sees only [w_bits, kv_bits] columns.
                    nd_F = archive_F_full[archive_front, 1:]

                    # Candidate complexity matrix from cached arrays
                    cand_F = np.stack(
                        [w_bits_arr[pareto_candidates], kv_bits_arr[pareto_candidates]],
                        axis=1,
                    )

                    problem = SubsetProblem(
                        cand_F,
                        nd_F,
                        self.n_iter,
                        n_obj=len(self.comp_obj),
                    )
                    algorithm = GA(
                        pop_size=self.subset_pop_size,
                        sampling=MySampling(),
                        crossover=BinaryCrossover(),
                        mutation=MyMutation(),
                        eliminate_duplicates=True,
                    )

                    res = minimize(
                        problem, algorithm, ('n_gen', 60), verbose=False
                    )

                    mask = res.X
                    # `res.X` can be shape (n_var,) or (1, n_var)
                    if mask.ndim > 1:
                        mask = mask[0]
                    chosen_local = np.where(mask)[0].tolist()

                    # Map local indices within `pareto_candidates` back to
                    # global combination indices.
                    selected_indices = [pareto_candidates[j] for j in chosen_local]
                
                candidates = [combinations[i] for i in selected_indices]
            else:
                candidates = []
            
            accelerator.wait_for_everyone()
            candidates = accelerator.gather_for_metrics(candidates, use_gather_object=True)

            # If no candidates found, terminate search
            if len(candidates) == 0:
                if accelerator.is_main_process:
                    accelerator.print(f"No more candidates found. Terminating search at iteration {it}.")
                break

            # Evaluate selected candidates
            # Note: candidates are already filtered to exclude archive duplicates
            archs_to_eval = [c[0] for c in candidates]
            c_metrics, complexity = self._evaluate(archs_to_eval, accelerator)

            if accelerator.is_main_process:
                # Check predictor performance
                # Get predictions for candidates
                # Use cached encoded_combinations instead of re-encoding
                c_encoded = encoded_combinations[selected_indices]
                c_metrics_pred = predictor.predict(c_encoded)
                # Optimize: build arrays more efficiently
                archive_metrics = np.array([x[1] for x in archive], dtype=float)
                combined_metrics = np.concatenate([archive_metrics, np.array(c_metrics, dtype=float)])
                combined_preds = np.vstack((a_metrics_pred, c_metrics_pred))
                rmse, rho, tau = get_correlation(combined_preds, combined_metrics)

                # Add to archive
                for comb_idx, (c, metric) in zip(selected_indices, zip(candidates, c_metrics)):
                    archive.append([c[0], metric, c[2], c[3], c[4], c[5]])  # [arch, metric, w_bits, kv_bits, w_idx, kv_idx]
                    comb_idx = int(comb_idx)
                    index_to_metric[comb_idx] = metric
                    # Overwrite cached metric with true evaluation result
                    metric_arr[comb_idx] = float(metric)
                    evaluated_mask[comb_idx] = True

                # Calculate hypervolume using fixed reference point
                # Optimize: extract columns directly from archive array
                archive_array = np.array([[x[1], x[2], x[3]] for x in archive])
                F = archive_array[:, :1 + len(self.comp_obj)]
                hv = self._calc_hv(ref_pt, F)

                iter_time = time() - iter_start
                accelerator.print(f"Iter {it}: hv = {hv:.2f}, iter time: {iter_time:.2f}s, predictor_time: {predictor_time:.2f}s")
                accelerator.print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall's Tau = {tau:.4f}")
                accelerator.print(f'iteration time: {iter_time:.2f}s')

                # Dump the statistics
                if it % self.save_iter == 0:
                    # Prepare remains: indices for archs that are not in archive (only predictions)
                    # Vectorized: use boolean indexing for faster filtering
                    remains_indices = np.where(~evaluated_mask)[0]
                    remains = [
                        [int(idx_c), float(metric_arr[idx_c]), float(w_bits_arr[idx_c]), float(kv_bits_arr[idx_c])]
                        for idx_c in remains_indices
                    ]

                    # Prepare archive for saving: use indices instead of full architectures
                    # Vectorized: build list directly from index_to_metric
                    archive_to_save = [
                        [int(idx_c), float(metric_val), float(w_bits_arr[idx_c]), float(kv_bits_arr[idx_c])]
                        for idx_c, metric_val in index_to_metric.items()
                    ]

                    os.makedirs(self.save_path, exist_ok=True)
                    with open(os.path.join(self.save_path, f"iter_{it}.stats"), "w") as handle:
                        json.dump({
                            # Actual measured values (index-based)
                            'archive': archive_to_save,
                            # Predicted values for archs not in archive (index-based)
                            'remains': remains,
                            'hv': hv,
                            'surrogate': {
                                'model': self.predictor,
                                'name': predictor.name,
                                'winner': predictor.winner if self.predictor == 'as' else predictor.name,
                                'rmse': rmse, 'rho': rho, 'tau': tau, 'total_time': iter_time
                            },
                            'iteration': it
                        }, handle)
                    
                    if self.debug:
                        self._save_debug_plot(it, archive, complexity, c_metrics, c_metrics_pred)

            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            total_time = time() - total_start
            accelerator.print(f"Total time: {total_time:.2f}s")
            
            # Save results
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, self.result_file), 'w') as f:
                for k, v in self.args.items():
                    f.write(f"{k}: {v}\n")
                f.write(f"Total time: {total_time:.2f}s\n")
            accelerator.print(self.args)
        return

    def _save_debug_plot(self, it, archive, complexity, c_metrics, c_metrics_pred):
        """Save debug visualization plots"""
        import matplotlib.pyplot as plt
        
        n_obj = len(self.comp_obj)
        comp_np = np.array(complexity)
        fig, axes = plt.subplots(nrows=1, ncols=n_obj + 1, figsize=(5 * (n_obj + 1), 5))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for i in range(n_obj):
            comp = np.array([x[i+2] for x in archive])  # comp obj
            perf = np.array([x[1] for x in archive])  # performance
            axes[i].scatter(comp, perf, s=5, facecolors='none', edgecolors='b', label='archive')
            
            comp = comp_np[:, i]
            perf = np.array(c_metrics)
            axes[i].scatter(comp, perf, s=10, color='r', label='candidates evaluated')
            axes[i].scatter(comp[:len(c_metrics_pred)], c_metrics_pred, s=10, facecolors='none', edgecolors='g', label='candidates predicted')
            # axes[i].legend(loc="upper right")
            axes[i].legend()
            axes[i].set_xlabel(f'f{i+2}')
            axes[i].grid(c='0.8')
        
        axes[0].set_ylabel('f1')
        axes[-1].scatter(np.array([x[2] for x in archive]), np.array([x[3] for x in archive]), s=5, facecolors='none', edgecolors='b', label='archive')
        axes[-1].scatter(np.array(complexity)[:, 0], np.array(complexity)[:, 1], s=10, color='r', label='candidates')
        # axes[-1].legend(loc="upper right")
        axes[-1].legend()
        axes[-1].set_xlabel('f2')
        axes[-1].set_ylabel('f3')
        axes[-1].grid(c='0.8')
        
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'iter_{it}.png'))
        plt.close()

    def _resume_from_dir(self, combinations):
        """Resume search from a previous iteration.

        Supports two formats of the saved JSON:
        - Legacy format:
            'archive': [[arch_dict, metric, w_bits, kv_bits], ...]
        - New index-based format (to reduce JSON size):
            'archive': [[index, metric, w_bits, kv_bits], ...]

        This method reconstructs the in-memory archive in the legacy
        format (with full architectures) and also returns an
        index_to_metric mapping for fast lookup.
        """
        with open(self.resume, 'r') as f:
            resume_file = json.load(f)
            raw_archive = resume_file.get('archive', [])
            it = resume_file.get('iteration', 0)

        archive = []
        index_to_metric = {}

        if len(raw_archive) == 0:
            # Nothing to resume
            return archive, index_to_metric, it + 1

        first_entry = raw_archive[0]

        # New format: [index, metric, w_bits, kv_bits]
        if isinstance(first_entry[0], int):
            for idx_c, metric, w_bits, kv_bits in raw_archive:
                arch = combinations[idx_c][0]
                w_idx = combinations[idx_c][4]
                kv_idx = combinations[idx_c][5]
                archive.append([arch, metric, w_bits, kv_bits, w_idx, kv_idx])
                index_to_metric[idx_c] = metric
        else:
            # Legacy format: [arch_dict, metric, w_bits, kv_bits]
            # Need to find w_idx and kv_idx from combinations
            # Optimize: build lookup dict for faster matching
            arch_to_idx = {id(c[0]): (idx_c, c) for idx_c, c in enumerate(combinations)}
            for entry in raw_archive:
                arch_dict = entry[0]
                metric = entry[1]
                w_bits = entry[2]
                kv_bits = entry[3]
                # Find matching combination using lookup dict
                found = False
                arch_id = id(arch_dict)
                if arch_id in arch_to_idx:
                    idx_c, c = arch_to_idx[arch_id]
                    # Verify it's the same architecture (not just same id)
                    if c[0] == arch_dict:
                        w_idx = c[4]
                        kv_idx = c[5]
                        archive.append([arch_dict, metric, w_bits, kv_bits, w_idx, kv_idx])
                        index_to_metric[idx_c] = metric
                        found = True
                if not found:
                    # Fallback: linear search if lookup fails (should be rare)
                    for idx_c, c in enumerate(combinations):
                        if c[0] == arch_dict:
                            w_idx = c[4]
                            kv_idx = c[5]
                            archive.append([arch_dict, metric, w_bits, kv_bits, w_idx, kv_idx])
                            index_to_metric[idx_c] = metric
                            found = True
                            break
                    if not found:
                        # Final fallback: try to infer from w_bits and kv_bits (less reliable)
                        archive.append([arch_dict, metric, w_bits, kv_bits, 0, 0])

        return archive, index_to_metric, it + 1

class SubsetProblem(Problem):
    """Select a subset of candidates to diversify the (w_bits, kv_bits) Pareto front.
 
    This is adapted from `search.py`'s `SubsetProblem` so that we can
    spread candidates more evenly along the w_bits / kv_bits axes.
    """
 
    def __init__(self, candidates, archive, K, n_obj):
        # `n_var` is the number of candidates; each variable is a bool
        # indicating whether the corresponding candidate is selected.
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1, type_var=bool)
        # We assume archive always has at least one entry when this problem
        # is constructed (len(archive) >= 1), so we do not handle the empty case.
        self.archive = archive
        self.candidates = candidates
        self.n_max = K
 
    def _evaluate(self, x, out, *args, **kwargs):
        batch_size = x.shape[0]
        f = np.full((batch_size, 1), np.nan)
        g = np.full((batch_size, 1), np.nan)
        
        for i, _x in enumerate(x):
            # Append selected candidates to archive and measure how evenly
            # they are spread along the (w_bits, kv_bits) axes.
            #
            # Following `search.py`'s SubsetProblem idea, we use the
            # standard deviation of pairwise differences along each axis
            # and then combine (sum) the per-axis std values.
            selected_candidates = self.candidates[_x]
            # points: existing archive + newly selected candidates
            points = np.vstack((self.archive, selected_candidates))
            
            if len(points) > 1:
                sorted_points = np.sort(points, axis=0)
                diffs = np.diff(sorted_points, axis=0)
                axis_std = np.std(diffs, axis=0)
                # f is a scalar objective: combine wbits/kvbits stds by summation
                f[i, 0] = axis_std.sum()
            else:
                f[i, 0] = 0.0

            # Penalize if the number of selected candidates is not exactly K
            n_selected = np.sum(_x)
            g[i, 0] = (self.n_max - n_selected) ** 2

        out["F"] = f
        out["G"] = g

def main(args):
    set_seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)
    engine = SecondSearch(config=config, accelerator=accelerator, device_map=device_map, kwargs=vars(args))
    engine.search(accelerator)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='save',
                        help='location of dir to save')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume search from a checkpoint')
    parser.add_argument('--iterations', type=int, default=10,
                        help='number of search iterations')
    parser.add_argument('--n_doe', type=int, default=50,
                        help='initial sample size for DOE')
    parser.add_argument('--n_iter', type=int, default=5,
                        help='number of solutions to measure per iteration')
    parser.add_argument('--predictor', type=str, default='rbf',
                        help='which predictor model to fit (rbf/gp/cart/mlp/as)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--w_method', type=str, nargs='+', default=[], 
                        choices=['fp16', 'awq', 'gptq', 'qeft', 'hqq'],
                        help='')
    parser.add_argument('--kv_method', type=str, default='kivi', 
                        choices=['hqq', 'kivi'],
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
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset name')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for selecting calibration set, etc.')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='sample number of the calibration set')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='sequential length of the calibration (train) set')
    parser.add_argument('--min_seqlen', type=int, default=0,
                        help='minimum sequential length of the calibration set')
    parser.add_argument('--data_batch_size', type=int, default=1,
                        help='batch size for data loading')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='config file to read the model meta data')
    parser.add_argument('--result_file', type=str, default='results.txt',
                        help='file to save final results')
    parser.add_argument('--max_value', type=float, default=50,
                        help='maximum value for metric clipping')
    parser.add_argument('--loss_func', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'jsd'],
                        help='loss function for evaluation')
    parser.add_argument('--save_iter', type=int, default=1,
                        help='save iteration results every N iterations')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode with visualization')
    parser.add_argument('--metric', type=str, default='loss',
                        choices=['loss', 'ppl'],
                        help='which metric to use (loss/ppl)')
    parser.add_argument('--base_outlier_bits', type=int, nargs='+', default=[],
                        help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')
    parser.add_argument('--n_outlier', type=int, default=0,
                        help='')
    parser.add_argument('--only_outlier_bits', action='store_true', help='')
    parser.add_argument('--sensitivity_result_path', type=str, default='',
                        help='')
    parser.add_argument('--sensitivity_threshold', type=int, default=2,
                        help='')
    parser.add_argument('--n_token', type=int, default=0,
                        help='target sequence length for memory calculation')
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
    parser.add_argument('--w_pareto_path', type=str, required=True,
                        help='path to W pareto frontier results from search.py')
    parser.add_argument('--kv_pareto_path', type=str, required=True,
                        help='path to KV pareto frontier results from search.py')
    parser.add_argument('--selection_mode', type=str, default='random',
                        choices=['even', 'random'],
                        help='mode for selecting K solutions from pareto frontier')
    parser.add_argument('--init_mode', type=str, default='random',
                        choices=['random', 'pareto_random'],
                        help='initial sampling mode: random or pareto_random')
    parser.add_argument('--subset_pop_size', type=int, default=100,
                        help='population size of the subset selection stage')

    cfgs = parser.parse_args()
    main(cfgs)
