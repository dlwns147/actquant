import os
import json
import math
import torch
import argparse
import numpy as np
import csv
from copy import deepcopy
from time import time
from tqdm import tqdm

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.indicators.hv import Hypervolume
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2

from search import SubsetProblem
from search_space.llama import LlamaGroupSizeSearchSpace
from predictor.factory import get_predictor
from utils.func import get_net_info, init_accelerator, set_seed, get_correlation
from utils.ga import MySampling, BinaryCrossover, MyMutation, IntMutation
from evaluator import LlamaEvaluator
from lm_eval.tasks import TaskManager, get_task_dict

import warnings
warnings.simplefilter("ignore")

PHASE_W  = 'w'    # fix KV, search W bits
PHASE_KV = 'kv'   # fix W, search KV bits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_threshold(cycle, n_cycles, val_min, val_max, schedule='cosine'):
    """
    Threshold that decreases from val_max → val_min over n_cycles.
    cycle: current cycle index (0-based)
    """
    if n_cycles <= 1:
        return val_min
    alpha = min(cycle, n_cycles - 1) / (n_cycles - 1)   # 0 → 1
    if schedule == 'cosine':
        # Slow decrease at first, faster later
        return val_min + 0.5 * (val_max - val_min) * (1.0 + math.cos(math.pi * alpha))
    else:  # linear
        return val_max - alpha * (val_max - val_min)


def arch_to_key(arch):
    """Canonical JSON key for deduplication."""
    a = arch['q'] if 'q' in arch else arch
    return json.dumps({'w': a['w'], 'k': a['k'], 'v': a['v']}, sort_keys=True)


def to_eval_format(arch):
    """
    Wrap search-space arch into the evaluator/get_net_info format.
    LlamaGroupSizeSearchSpace.decode() returns {'w':..,'k':..,'v':..}.
    evaluator.sample() and get_net_info() expect {'q': {'w':..,'k':..,'v':..}, 'p': {}}.
    """
    if 'q' in arch:
        return arch
    return {'q': arch}


# ---------------------------------------------------------------------------
# NSGA2 problem for one phase
# ---------------------------------------------------------------------------

class PhaseSearchProblem(Problem):
    """
    2-objective problem: (relative_metric, free_complexity).

    One half of the bit-string is fixed to ctx_arch via xl==xu
    (IntMutation uses randint(xl, xu+1), so xl==xu → always returns that value).
    relative_metric = surrogate(arch) - surrogate(baseline_arch)
      where baseline_arch = ctx_arch + max-precision other-objective.
    """

    def __init__(self, search_space, predictor, config,
                 free_comp_obj, free_comp_obj_min, free_comp_obj_max,
                 group_size, n_token,
                 phase, ctx_arch, baseline_arch):
        n_block  = search_space.n_block
        n_linear = search_space.n_linear
        # 2 objectives: (relative_metric, free_complexity)
        # 2 constraints: min/max on free_complexity
        super().__init__(n_var=n_block * (n_linear + 2),
                         n_obj=2, n_constr=2, type_var=int)

        self.ss               = search_space
        self.predictor        = predictor
        self.free_comp_obj    = free_comp_obj
        self.free_comp_obj_min = free_comp_obj_min
        self.free_comp_obj_max = free_comp_obj_max
        self.config    = config
        self.group_size = group_size
        self.n_token   = n_token
        self.phase     = phase

        # ---- xl / xu (same logic as AuxiliarySingleLevelProblem) ----
        xl = np.zeros((n_linear + 2, n_block))
        xu = np.ones( (n_linear + 2, n_block))
        for i, linear in enumerate(config['linear']):
            xu[i] = len(getattr(search_space,
                                f"{linear.split('.')[-1]}_option")) - 1
        xu[n_linear]     = len(search_space.k_option) - 1
        xu[n_linear + 1] = len(search_space.v_option) - 1

        for pass_w in search_space.pass_module['w']:
            blk, linear = pass_w.split('.', 1)
            li = config['linear'].index(linear)
            xl[li, int(blk)] = len(
                getattr(search_space, f"{linear.split('.')[-1]}_option")) - 1
        for lyr in search_space.pass_module['k']:
            xl[n_linear,     lyr] = len(search_space.k_option) - 1
        for lyr in search_space.pass_module['v']:
            xl[n_linear + 1, lyr] = len(search_space.v_option) - 1

        xl = xl.flatten()
        xu = xu.flatten()

        # ---- Fix one half via xl == xu ----
        ctx_enc = search_space.encode(ctx_arch)
        w_end   = n_linear * n_block

        if phase == PHASE_KV:
            # Searching KV → fix W portion
            xl[:w_end] = xu[:w_end] = ctx_enc[:w_end]
            self._fixed_slice  = slice(None, w_end)
            self._fixed_values = ctx_enc[:w_end].copy()
        else:  # PHASE_W
            # Searching W → fix KV portion
            xl[w_end:] = xu[w_end:] = ctx_enc[w_end:]
            self._fixed_slice  = slice(w_end, None)
            self._fixed_values = ctx_enc[w_end:].copy()

        self.xl = xl
        self.xu = xu

        # ---- Baseline prediction (surrogate, no GPU eval) ----
        baseline_enc = search_space.decode_encode_predictor(
            search_space.encode(baseline_arch)[None, :])   # (1, pred_dim)
        self.baseline_pred = float(predictor.predict(baseline_enc)[0, 0])

    def _evaluate(self, x, out, *args, **kwargs):
        # Guard: enforce fixed portion (handles possible crossover drift)
        x[:, self._fixed_slice] = self._fixed_values

        f = np.full((x.shape[0], 2), np.nan)
        g = np.full((x.shape[0], 2), np.nan)

        # Relative metric: positive = quality worse than baseline
        metrics    = self.predictor.predict(
            self.ss.decode_encode_predictor(x))[:, 0]
        rel_metrics = metrics - self.baseline_pred

        for i, (_x, rel_m) in enumerate(zip(x, rel_metrics)):
            arch     = to_eval_format(self.ss.decode(_x))
            info     = get_net_info(arch, self.config, self.group_size,
                                    n_token=self.n_token)
            obj_val  = info[self.free_comp_obj[0]]

            f[i, 0] = rel_m
            f[i, 1] = obj_val

            # g ≤ 0  → feasible
            g[i, 0] = (1 - obj_val / self.free_comp_obj_min[0]
                       if self.free_comp_obj_min[0] != 0 else 0.0)
            g[i, 1] = (obj_val / self.free_comp_obj_max[0] - 1
                       if self.free_comp_obj_max[0] != 0 else 0.0)

        out["F"] = f
        out["G"] = g


# ---------------------------------------------------------------------------
# Main search class
# ---------------------------------------------------------------------------

class CyclicSearch:
    """
    Cyclic Pareto Frontier Search (CPFS).

    Alternates between Phase-W (fix KV, search W) and Phase-KV (fix W, search KV).
    Each phase uses a 2-objective NSGA2 with relative metric to avoid signal
    saturation in low-bit contexts.  A progressive threshold ensures the search
    starts from high-W-bits (stable) contexts before expanding to low-W-bits
    (high-sensitivity) contexts.

    Archive format: [arch_dict, metric_abs, comp_obj_0, comp_obj_1, ...]
    requires comp_obj = ['wbits', 'kvbits'].
    """

    def __init__(self, config, accelerator, device_map, kwargs):
        self.args        = deepcopy(kwargs)
        self.config      = config
        self.device_map  = device_map

        self.save_path   = kwargs.pop('save', 'save')
        self.result_file = kwargs.pop('result_file', 'results.txt')
        self.resume      = kwargs.pop('resume', None)
        self.iterations  = kwargs.pop('iterations', 30)
        self.n_doe       = kwargs.pop('n_doe', 100)
        self.n_iter      = kwargs.pop('n_iter', 8)
        self.predictor   = kwargs.pop('predictor', 'rbf')
        self.dataset     = kwargs.pop('dataset', 'wikitext2')
        self.loss_func   = kwargs.pop('loss_func', 'jsd')

        self.method            = {'w': kwargs.pop('w_method', ['fp16']),
                                  'kv': kwargs.pop('kv_method', 'kivi')}
        self.quant_model_paths = kwargs.pop('quant_model_paths', [])

        model_path  = kwargs.pop('model_path', 'meta-llama')
        model_name  = kwargs.pop('model_name', 'Llama-2-7b-hf')
        model_id    = f'{model_path}/{model_name}'
        self.metric = kwargs.pop('metric', 'loss')
        self.limit  = kwargs.pop('limit', 20)
        self.lm_eval_batch_size = kwargs.pop('lm_eval_batch_size', 1)
        self.num_fewshot        = kwargs.pop('num_fewshot', None)

        outlier_path       = kwargs.pop('outlier_path', '')
        base_outlier_bits  = sorted(kwargs.pop('base_outlier_bits', []))
        n_outlier          = kwargs.pop('n_outlier', 0)

        assert (outlier_path and base_outlier_bits and n_outlier > 0) or \
               (not outlier_path and not base_outlier_bits and n_outlier == 0)

        outlier_bits = {l: [] for l in config['linear']}
        if outlier_path and base_outlier_bits and n_outlier > 0:
            for linear in config['linear']:
                for base_bits in base_outlier_bits:
                    _, in_dim = config['linear_shape'][linear]
                    avg = ((in_dim - n_outlier) * base_bits + n_outlier * 16) / in_dim
                    outlier_bits[linear].append(avg)

        w_bits = kwargs.pop('w_bits', [])
        assert len(w_bits) == len(self.quant_model_paths)
        k_bits = kwargs.pop('k_bits', [])
        v_bits = kwargs.pop('v_bits', [])
        self.bits = {'w': w_bits, 'k': k_bits, 'v': v_bits}

        w_group_size = kwargs.pop('w_group_size', 128)
        k_group_size = kwargs.pop('k_group_size', [[128]])
        v_group_size = kwargs.pop('v_group_size', [[128]])
        self.group_size = {'w': w_group_size, 'k': k_group_size, 'v': v_group_size}

        self.residual_length = kwargs.pop('residual_length', 128)
        self.verbosity       = kwargs.pop('verbosity', 'FATAL')
        self.task_manager    = TaskManager(self.verbosity) \
                               if self.metric not in ['ppl', 'loss'] else None
        self.task_dict       = get_task_dict([self.metric], self.task_manager) \
                               if self.metric not in ['ppl', 'loss'] else None

        self.comp_obj     = kwargs.pop('comp_obj', ['wbits', 'kvbits'])
        self.comp_obj_min = kwargs.pop('comp_obj_min', [min(w_bits), min(k_bits)])
        self.comp_obj_max = kwargs.pop('comp_obj_max', [max(w_bits), max(k_bits)])
        assert len(self.comp_obj) == len(self.comp_obj_min) == len(self.comp_obj_max)
        assert 'wbits' in self.comp_obj and 'kvbits' in self.comp_obj, \
            "CyclicSearch requires both 'wbits' and 'kvbits' in --comp_obj"

        self.w_obj_idx  = self.comp_obj.index('wbits')
        self.kv_obj_idx = self.comp_obj.index('kvbits')

        self.n_token = kwargs.pop('n_token', 0)
        if 'memory' in self.comp_obj:
            assert self.n_token > 0

        self.use_key_token  = kwargs.pop('use_key_token', False)
        self.trunc_len      = kwargs.pop('trunc_len', 512)
        self.sliding_window = kwargs.pop('sliding_window', 128)
        self.alpha          = kwargs.pop('alpha', 2)
        self.beta           = kwargs.pop('beta', -2)
        self.key_token_path = kwargs.pop('key_token_path', '')

        # Sensitivity / pass_module (identical to Search)
        self.sensitivity_result_path = kwargs.pop('sensitivity_result_path', '')
        total_module      = {}
        total_sensitivity = {}
        pass_module       = {'w': [], 'k': [], 'v': []}

        if self.sensitivity_result_path:
            for target in pass_module:
                with open(os.path.join(self.sensitivity_result_path,
                                       f'{target}.csv'), 'r') as f:
                    module_list, sensitivity = list(csv.reader(f))
                    sensitivity = list(map(float, sensitivity))
                    total_module[target] = (list(map(int, module_list))
                                            if target in ['k', 'v']
                                            else module_list)
                    total_sensitivity[target] = sensitivity
            total_sens_list = np.nan_to_num(
                np.concatenate(list(total_sensitivity.values())), nan=float('inf'))
            upper_bound = np.median(total_sens_list) * kwargs.pop('sensitivity_threshold', 2)
            pass_idx_list = np.where(total_sens_list > upper_bound)[0].tolist()
            start = 0
            for target in pass_module:
                end = start + len(total_module[target])
                for idx in pass_idx_list:
                    if start <= idx < end:
                        pass_module[target].append(total_module[target][idx - start])
                start = end
        else:
            kwargs.pop('sensitivity_threshold', 2)   # discard even if unused

        self.pass_module        = pass_module
        self.args['pass_module'] = pass_module

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
            bits=self.bits,
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
        )

        self.search_space = LlamaGroupSizeSearchSpace(
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
        )

        self.ga_pop_size    = kwargs.pop('ga_pop_size', 40)
        self.subset_pop_size = kwargs.pop('subset_pop_size', 100)
        self.debug          = kwargs.pop('debug', False)
        self.max_value      = kwargs.pop('max_value', 50)
        self.mut_prob       = kwargs.pop('mut_prob', 0.1)
        self.crossover_prob = kwargs.pop('crossover_prob', 0.9)
        self.save_iter      = kwargs.pop('save_iter', 1)

        # CPFS-specific
        self.n_cycles            = kwargs.pop('n_cycles', 3)
        self.max_contexts        = kwargs.pop('max_contexts', 5)
        self.phases              = kwargs.pop('phases', [PHASE_KV, PHASE_W])
        self.threshold_schedule  = kwargs.pop('threshold_schedule', 'cosine')

        accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # Main search loop
    # -----------------------------------------------------------------------

    def search(self, accelerator):
        total_start = time()
        start_it    = 1
        n_phases    = len(self.phases)

        if self.resume:
            archive, start_it = self._resume_from_dir()
        else:
            archive = []
            if accelerator.is_main_process:
                arch_doe = self.search_space.initialize(self.n_doe, pool=[])
            else:
                arch_doe = list()
            arch_doe = accelerator.gather_for_metrics(arch_doe,
                                                      use_gather_object=True)
            accelerator.wait_for_everyone()

            metric, complexity = self._evaluate(archs=arch_doe,
                                                accelerator=accelerator)
            if accelerator.is_main_process:
                for a, m, c in zip(arch_doe, metric, complexity):
                    archive.append([a, m, *c])

        if accelerator.is_main_process:
            ref_pt = np.array([np.max([x[i] for x in archive])
                               for i in range(1, len(self.comp_obj) + 2)])
            accelerator.print(f'DOE done. archive size: {len(archive)}, ref_pt: {ref_pt}')
        accelerator.wait_for_everyone()

        for it in range(start_it, self.iterations + 1):
            phase     = self.phases[(it - 1) % n_phases]
            cycle     = (it - 1) // n_phases   # which cycle we are in

            if accelerator.is_main_process:
                accelerator.print(self.args)
                iter_start = time()
                accelerator.print(
                    f"\n=== Iter {it}/{self.iterations} | phase={phase} | cycle={cycle} ===")

                threshold = self._get_threshold(cycle, phase)
                accelerator.print(f"  threshold={threshold:.4f}")

                # --- Fit surrogate on entire joint archive ---
                pred_t0 = time()
                metric_predictor, a_metric_pred = self._fit_predictor(
                    archive, device=accelerator.device)
                pred_time = time() - pred_t0

                # --- Extract Pareto contexts with progressive threshold ---
                contexts_with_c = self._extract_pareto_contexts_with_threshold(
                    archive, phase, threshold)
                selected_ctxs = self._select_diverse_contexts(contexts_with_c)
                accelerator.print(
                    f"  n_eligible_contexts={len(contexts_with_c)}, "
                    f"n_selected={len(selected_ctxs)}")

                # --- Collect baselines + NSGA2 candidates ---
                archive_keys       = {arch_to_key(e[0]) for e in archive}
                baselines_to_eval  = []
                nsga2_candidates   = []

                for ctx_arch in selected_ctxs:
                    # Baseline = ctx + max-precision other objective
                    bl = self._get_baseline_arch(ctx_arch, phase)
                    if arch_to_key(bl) not in archive_keys:
                        baselines_to_eval.append(bl)
                        archive_keys.add(arch_to_key(bl))  # avoid dups across ctxs

                    # NSGA2 on the free objective
                    ctx_cands = self._run_nsga2_for_context(
                        ctx_arch, phase, metric_predictor, archive)
                    nsga2_candidates.extend(ctx_cands)

                # Deduplicate NSGA2 candidates against archive
                nsga2_candidates = [
                    c for c in nsga2_candidates
                    if arch_to_key(c) not in archive_keys
                ]
                # Rebuild key set to include newly filtered candidates too
                # (no need to update archive_keys for subset selection purposes)

                # --- Subset selection on NSGA2 candidates ---
                if len(nsga2_candidates) > self.n_iter:
                    F_arch = np.array([[x[1], x[2], x[3]] for x in archive])
                    front  = NonDominatedSorting().do(
                        F_arch, only_non_dominated_front=True)
                    pareto_nd_F = F_arch[front, 1:]   # (wbits, kvbits) columns
                    sel_mask = self._subset_selection_from_archs(
                        nsga2_candidates, pareto_nd_F, self.n_iter)
                    nsga2_candidates = [
                        nsga2_candidates[i]
                        for i, sel in enumerate(sel_mask) if sel
                    ]

                candidates = baselines_to_eval + nsga2_candidates
                accelerator.print(
                    f"  baselines={len(baselines_to_eval)}, "
                    f"nsga2_selected={len(nsga2_candidates)}, "
                    f"total_to_eval={len(candidates)}")

                # predicted metrics for correlation check (use surrogate on candidates)
                if len(candidates) > 0:
                    cand_enc = np.array([self.search_space.encode(c)
                                         for c in candidates])
                    c_metric_pred = metric_predictor.predict(
                        self.search_space.decode_encode_predictor(cand_enc))
                else:
                    c_metric_pred = np.empty((0, 1))
            else:
                candidates = list()

            accelerator.wait_for_everyone()
            candidates = accelerator.gather_for_metrics(candidates,
                                                        use_gather_object=True)

            # --- Joint GPU evaluation ---
            c_metric, complexity = self._evaluate(archs=candidates,
                                                  accelerator=accelerator)

            if accelerator.is_main_process:
                # Predictor correlation
                if len(c_metric) > 0:
                    rmse, rho, tau = get_correlation(
                        np.vstack((a_metric_pred, c_metric_pred)),
                        np.array([x[1] for x in archive] + c_metric))
                else:
                    rmse, rho, tau = float('nan'), float('nan'), float('nan')

                # Add to archive
                for a, m, c in zip(candidates, c_metric, complexity):
                    archive.append([a, m, *c])

                # Update reference point (monotone max)
                for j in range(len(self.comp_obj)):
                    ref_pt[j] = max(ref_pt[j],
                                    max(x[j + 2] for x in archive[-len(candidates):]))
                ref_pt[len(self.comp_obj)] = max(
                    ref_pt[len(self.comp_obj)],
                    max(x[1] for x in archive[-len(candidates):]))

                # Hypervolume
                F_all = np.column_stack(
                    [[x[i] for x in archive]
                     for i in range(1, len(self.comp_obj) + 2)])
                hv = self._calc_hv(ref_pt, F_all)

                iter_time = time() - iter_start
                accelerator.print(
                    f"  hv={hv:.4f} | RMSE={rmse:.4f} Rho={rho:.4f} Tau={tau:.4f}")
                accelerator.print(
                    f"  iter_time={iter_time:.2f}s pred_time={pred_time:.2f}s")

                if it % self.save_iter == 0:
                    os.makedirs(self.save_path, exist_ok=True)
                    with open(os.path.join(self.save_path,
                                           f'iter_{it}.stats'), 'w') as fh:
                        json.dump({
                            'archive'   : archive,
                            'candidates': archive[-len(candidates):],
                            'hv'        : hv,
                            'iteration' : it,
                            'cycle'     : cycle,
                            'phase'     : phase,
                            'threshold' : threshold,
                            'surrogate' : {
                                'model': self.predictor,
                                'rmse' : rmse, 'rho': rho, 'tau': tau,
                                'total_time': iter_time,
                            },
                        }, fh)

                    if self.debug:
                        self._save_debug_plot(it, archive, candidates,
                                              c_metric, c_metric_pred)

            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            total_elapsed = time() - total_start
            accelerator.print(f'Total time: {total_elapsed:.2f}s')
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, self.result_file), 'w') as f:
                for k, v in self.args.items():
                    f.write(f"{k}: {v}\n")
                f.write(f'Total time: {total_elapsed:.2f}s\n')

    # -----------------------------------------------------------------------
    # Phase helpers
    # -----------------------------------------------------------------------

    def _get_free_comp_obj(self, phase):
        """Return (free_comp_obj, free_min, free_max) for this phase."""
        if phase == PHASE_KV:
            idx = self.kv_obj_idx
        else:
            idx = self.w_obj_idx
        return ([self.comp_obj[idx]],
                [self.comp_obj_min[idx]],
                [self.comp_obj_max[idx]])

    def _get_baseline_arch(self, ctx_arch, phase):
        """
        Full arch where the FREE objective is set to maximum precision
        (highest bits) and the FIXED objective comes from ctx_arch.
        This serves as the reference for relative metric calculation.
        """
        n_block  = self.config['n_block']
        baseline = deepcopy(ctx_arch)
        if phase == PHASE_KV:
            # W is fixed to ctx; set KV to max precision
            baseline['k'] = [self.search_space.k_option[-1]] * n_block
            baseline['v'] = [self.search_space.v_option[-1]] * n_block
        else:  # PHASE_W
            # KV is fixed to ctx; set W to max precision
            default_w = max(self.bits['w'])
            baseline['w'] = {l: [default_w] * n_block
                             for l in self.config['linear']}
        return baseline

    def _get_threshold(self, cycle, phase):
        """
        Progressive threshold on the FIXED objective:
        - Phase KV → threshold on wbits (start high → include low-W contexts later)
        - Phase W  → threshold on kvbits (start high → include low-KV contexts later)
        """
        if phase == PHASE_KV:
            idx = self.w_obj_idx
        else:
            idx = self.kv_obj_idx
        return get_threshold(cycle, self.n_cycles,
                             self.comp_obj_min[idx],
                             self.comp_obj_max[idx],
                             self.threshold_schedule)

    def _extract_pareto_contexts_with_threshold(self, archive, phase, threshold):
        """
        Extract unique context archs from current Pareto frontier,
        keeping only those where fixed-objective complexity >= threshold.
        Returns list of (arch, fixed_complexity) tuples.
        """
        F = np.array([[x[1], x[2], x[3]] for x in archive])
        pareto_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)

        contexts  = []
        seen_keys = set()

        if phase == PHASE_KV:
            fixed_col = 2 + self.w_obj_idx   # wbits column in archive
        else:
            fixed_col = 2 + self.kv_obj_idx  # kvbits column in archive

        for i in pareto_idx:
            entry          = archive[i]
            arch           = entry[0]
            fixed_complexity = entry[fixed_col]

            if phase == PHASE_KV:
                ctx_key = json.dumps({'w': arch['w']}, sort_keys=True)
            else:
                ctx_key = json.dumps({'k': arch['k'], 'v': arch['v']}, sort_keys=True)

            if fixed_complexity >= threshold and ctx_key not in seen_keys:
                seen_keys.add(ctx_key)
                contexts.append((deepcopy(arch), fixed_complexity))

        # Fallback: guarantee at least one context even when no point meets threshold
        if not contexts:
            best_i = min(pareto_idx,
                         key=lambda i: abs(archive[i][fixed_col] - threshold))
            a  = archive[best_i][0]
            fc = archive[best_i][fixed_col]
            if phase == PHASE_KV:
                ck = json.dumps({'w': a['w']}, sort_keys=True)
            else:
                ck = json.dumps({'k': a['k'], 'v': a['v']}, sort_keys=True)
            if ck not in seen_keys:
                contexts.append((deepcopy(a), fc))

        return contexts

    def _select_diverse_contexts(self, contexts_with_complexity):
        """
        Greedy diversity selection of at most max_contexts contexts,
        spread over the complexity range.
        Returns list of arch dicts (without complexity value).
        """
        if len(contexts_with_complexity) <= self.max_contexts:
            return [c[0] for c in contexts_with_complexity]

        vals     = np.array([c[1] for c in contexts_with_complexity])
        selected = [int(np.argmin(vals))]
        if len(contexts_with_complexity) > 1:
            selected.append(int(np.argmax(vals)))

        while len(selected) < self.max_contexts:
            remaining = [i for i in range(len(contexts_with_complexity))
                         if i not in selected]
            if not remaining:
                break
            sel_vals = vals[selected]
            dists    = [min(abs(vals[r] - s) for s in sel_vals) for r in remaining]
            selected.append(remaining[int(np.argmax(dists))])

        return [contexts_with_complexity[i][0] for i in selected]

    def _is_in_archive(self, arch, archive):
        key = arch_to_key(arch)
        return any(arch_to_key(e[0]) == key for e in archive)

    def _run_nsga2_for_context(self, ctx_arch, phase, predictor, archive):
        """
        Run NSGA2 on the free objective for one fixed context.
        Objectives: (relative_metric, free_complexity)
        Population seeded from the current Pareto frontier.
        Returns list of decoded arch dicts.
        """
        free_comp_obj, free_min, free_max = self._get_free_comp_obj(phase)
        baseline_arch = self._get_baseline_arch(ctx_arch, phase)

        problem = PhaseSearchProblem(
            self.search_space, predictor, self.config,
            free_comp_obj, free_min, free_max,
            self.group_size, self.n_token,
            phase, ctx_arch, baseline_arch,
        )

        # Seed NSGA2 with current Pareto front
        F_arch = np.array([[x[1], x[2], x[3]] for x in archive])
        front  = NonDominatedSorting().do(F_arch, only_non_dominated_front=True)
        nd_X   = np.array([self.search_space.encode(archive[i][0]) for i in front])

        method = NSGA2(
            pop_size=self.ga_pop_size,
            sampling=nd_X,
            crossover=BinomialCrossover(prob=self.crossover_prob, n_offsprings=1),
            mutation=IntMutation(prob=self.mut_prob),
            eliminate_duplicates=True,
        )
        res = minimize(problem, method,
                       termination=('n_gen', 20),
                       save_history=False, verbose=False)

        return [self.search_space.decode(x) for x in res.pop.get("X")]

    def _subset_selection_from_archs(self, candidates, pareto_nd_F, K):
        """
        Wrapper for SubsetProblem that accepts a plain list of arch dicts.
        cand_F shape: (n_candidates, n_comp_obj)
        pareto_nd_F shape: (n_pareto, n_comp_obj)
        Returns boolean mask of length len(candidates).
        """
        cand_F = np.array([
            [get_net_info(to_eval_format(a), self.config, self.group_size,
                          n_token=self.n_token)[obj]
             for obj in self.comp_obj]
            for a in candidates
        ])
        problem   = SubsetProblem(cand_F, pareto_nd_F, K, len(self.comp_obj))
        algorithm = GA(
            pop_size=self.subset_pop_size,
            sampling=MySampling(),
            crossover=BinaryCrossover(),
            mutation=MyMutation(),
            eliminate_duplicates=True,
        )
        res = minimize(problem, algorithm, ('n_gen', 60), verbose=False)
        return res.X

    # -----------------------------------------------------------------------
    # Methods reused verbatim from Search
    # -----------------------------------------------------------------------

    def _resume_from_dir(self):
        with open(self.resume, 'r') as f:
            data = json.load(f)
        archive = data['archive'] + data['candidates']
        it      = data['iteration']
        return archive, it + 1

    def _evaluate(self, archs, accelerator):
        metric_list, complexity_list = [], []
        for arch in tqdm(archs, desc='Eval Arch'):
            metric, complexity = self.evaluator.eval(
                accelerator=accelerator, arch=to_eval_format(arch),
                metric=self.metric, loss_func=self.loss_func)
            metric_list.append(
                min(self.max_value,
                    np.nan_to_num(list(metric.values())[0], nan=self.max_value)))
            complexity_list.append([complexity[obj] for obj in self.comp_obj])
        return metric_list, complexity_list

    def _fit_predictor(self, archive, device='cpu'):
        inputs  = np.array([self.search_space.encode_predictor(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])

        kwargs = {}
        if self.predictor == 'rbf':
            n_block  = self.config['n_block']
            n_linear = self.config['n_linear']
            lb = np.zeros((n_linear + 2, n_block))
            ub = np.ones( (n_linear + 2, n_block))
            for i, linear in enumerate(self.config['linear']):
                ub[i] = len(getattr(self.search_space,
                                    f"{linear.split('.')[-1]}_option")) - 1
            ub[n_linear]     = len(self.search_space.k_option) - 1
            ub[n_linear + 1] = len(self.search_space.v_option) - 1
            lb = np.delete(lb.flatten(), self.search_space.pass_idx_list, axis=-1)
            ub = np.delete(ub.flatten(), self.search_space.pass_idx_list, axis=-1)
            kwargs = {'lb': lb, 'ub': ub}
            # CubicKernel + LinearTail requires n_points > n_var+1.
            # Fall back to LinearKernel + ConstantTail (ntail=1) when not enough points.
            if len(inputs) <= inputs.shape[1] + 1:
                kwargs['kernel'] = 'linear'
                kwargs['tail']   = 'constant'

        predictor = get_predictor(self.predictor, inputs, targets,
                                  device=device, **kwargs)
        return predictor, predictor.predict(inputs)

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        front  = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F   = F[front, :]
        ref_pt = 1.01 * ref_pt
        hv     = Hypervolume(ref_point=ref_pt).do(nd_F)
        if normalized:
            hv /= np.prod(ref_pt)
        return hv

    def _save_debug_plot(self, it, archive, candidates, c_metric, c_metric_pred):
        try:
            import matplotlib.pyplot as plt
            n_obj   = len(self.comp_obj)
            fig, axes = plt.subplots(1, n_obj, figsize=(5 * n_obj, 5))
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            for j in range(n_obj):
                comp_arch = np.array([x[j + 2] for x in archive])
                perf_arch = np.array([x[1]     for x in archive])
                axes[j].scatter(comp_arch, perf_arch, s=5,
                                facecolors='none', edgecolors='b', label='archive')
                comp_cand = np.array(
                    [get_net_info(c, self.config, self.group_size,
                                  n_token=self.n_token)[self.comp_obj[j]]
                     for c in candidates])
                axes[j].scatter(comp_cand, c_metric, s=10,
                                color='r', label='evaluated')
                axes[j].scatter(comp_cand, c_metric_pred[:, 0], s=10,
                                facecolors='none', edgecolors='g', label='predicted')
                axes[j].legend()
                axes[j].set_xlabel(self.comp_obj[j])
            axes[0].set_ylabel('metric')
            fig.tight_layout()
            plt.savefig(os.path.join(self.save_path, f'iter_{it}.png'))
            plt.close(fig)
        except Exception as e:
            print(f"[debug plot failed] {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)
    engine = CyclicSearch(config=config, accelerator=accelerator,
                          device_map=device_map, kwargs=vars(args))
    engine.search(accelerator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ---- Search params (identical to search.py) ----
    parser.add_argument('--save', type=str, default='save')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--n_doe', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=8)
    parser.add_argument('--predictor', type=str, default='rbf')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[])
    parser.add_argument('--w_method', type=str, nargs='+', default=[],
                        choices=['fp16', 'awq', 'gptq', 'qeft', 'hqq'])
    parser.add_argument('--kv_method', type=str, default='kivi',
                        choices=['hqq', 'kivi'])
    parser.add_argument('--w_bits', type=int, nargs='+', default=[])
    parser.add_argument('--k_bits', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--v_bits', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--w_group_size', type=int, default=128)
    parser.add_argument('--k_group_size', type=int, nargs='+',
                        action='append', default=[])
    parser.add_argument('--v_group_size', type=int, nargs='+',
                        action='append', default=[])
    parser.add_argument('--residual_length', type=int, default=128)
    parser.add_argument('--quant_kv_output', action='store_true')
    parser.add_argument('--k_quant_scheme', type=str,
                        choices=['channel', 'token'])
    parser.add_argument('--v_quant_scheme', type=str,
                        choices=['channel', 'token'])
    parser.add_argument('--comp_obj', type=str, nargs='+',
                        default=['wbits', 'kvbits'],
                        choices=['wbits', 'kvbits', 'memory'])
    parser.add_argument('--comp_obj_min', type=float, nargs='+', default=[2, 2])
    parser.add_argument('--comp_obj_max', type=float, nargs='+', default=[4, 4])
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_sample', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--min_seqlen', type=int, default=0)
    parser.add_argument('--metric', type=str, default='loss')
    parser.add_argument('--data_batch_size', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--lm_eval_batch_size', type=int, default=None)
    parser.add_argument('--num_fewshot', type=int, default=None)
    parser.add_argument('--verbosity', type=str, default='INFO')
    parser.add_argument('--config', type=str, default='config/llama.json')
    parser.add_argument('--ga_pop_size', type=int, default=40)
    parser.add_argument('--subset_pop_size', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--result_file', type=str, default='results.txt')
    parser.add_argument('--max_value', type=float, default=50)
    parser.add_argument('--crossover_prob', type=float, default=0.9)
    parser.add_argument('--mut_prob', type=float, default=0.1)
    parser.add_argument('--loss_func', type=str, default='cross_entropy')
    parser.add_argument('--base_outlier_bits', type=int, nargs='+', default=[])
    parser.add_argument('--outlier_path', type=str, default='')
    parser.add_argument('--n_outlier', type=int, default=0)
    parser.add_argument('--only_outlier_bits', action='store_true')
    parser.add_argument('--sensitivity_result_path', type=str, default='')
    parser.add_argument('--save_iter', type=int, default=1)
    parser.add_argument('--sensitivity_threshold', type=int, default=2)
    parser.add_argument('--n_token', type=int, default=0)
    parser.add_argument('--use_key_token', action='store_true')
    parser.add_argument('--trunc_len', type=int, default=512)
    parser.add_argument('--sliding_window', type=int, default=128)
    parser.add_argument('--alpha', type=int, default=2)
    parser.add_argument('--beta', type=int, default=-2)
    parser.add_argument('--key_token_path', type=str, default='')
    parser.add_argument('--packing', action='store_true')

    # ---- CPFS-specific args ----
    parser.add_argument('--n_cycles', type=int, default=3,
                        help='Number of cycles for progressive threshold decay')
    parser.add_argument('--max_contexts', type=int, default=5,
                        help='Max unique Pareto contexts used per phase iteration')
    parser.add_argument('--phases', type=str, nargs='+',
                        default=[PHASE_KV, PHASE_W],
                        choices=[PHASE_W, PHASE_KV],
                        help='Phase order to cycle through')
    parser.add_argument('--threshold_schedule', type=str,
                        default='cosine', choices=['cosine', 'linear'],
                        help='Schedule for progressive threshold decay')

    cfgs = parser.parse_args()
    main(cfgs)
