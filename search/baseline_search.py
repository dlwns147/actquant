"""baseline_search.py — single-stage joint (loss × wbits × eff_kvbits) NSGA-III NAS.

Baseline for the two-stage method (search.py per-axis → second_search.py joint): one
NSGA-III search over the FULL joint space (W bits + KV bits/gs + ThinK pruning) with
(loss, wbits, eff_kvbits) as objectives. Subclasses search.py::Search and overrides only
_next; evaluator / RBF surrogate / DOE / stats output are inherited, so a run is
post_search-consumable exactly like an ours run.

_next = NSGA-III candidate generation, then candidate down-select. DEFAULT down-select is
'subset' = search.py's legacy SubsetProblem: single-objective GA minimizing the pooled
std-of-gaps over the UNION of the archive front and the picked K, so an isolated edge
candidate SPLITS the front's largest gap and is REWARDED → comp-space holes (esp. the
eff_kvbits right end) get filled. The second_search.py-style selectors (maximin/grid/
hybrid/moo) remain as --cand_even options, but they score the K-subset's OWN geometry
only, which penalises isolated edge candidates (2607100451/0638 post-mortem: same pool +
30 injected right-end archs → 'subset' kept 29/30 vs 'moo axis_gap' 13/30, maximin 16/30).
"""
import os
import json
import argparse
import numpy as np
from time import time

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.binx import BinomialCrossover

from search import Search, AuxiliarySingleLevelProblemThink, SubsetProblem
from utils.ga import IntMutation, MySampling, BinaryCrossover, MyMutation
from utils.func import get_net_info, get_correlation, set_seed, init_accelerator
from utils.select import maximin_extras, even_select, moo_subset_select
from utils.second_stage import save_viz


def subset_select(comp, nd_F, K, pop_size=100, endpoints=None, n_gen=60, seed=None):
    """search.py::SubsetProblem down-select on plain arrays: pick K rows of `comp` by a
    single-objective GA minimizing the pooled std-of-gaps over nd_F(front comps) ∪ picks.
    The union with the front is what makes it hole-filling: an isolated edge candidate
    splits the front's largest gap and LOWERS the objective, where subset-only scores
    (moo axis_gap / gap-std) raise it and drop the candidate. `endpoints` = (2, n_comp)
    [lo, hi] rows (see Search._subset_endpoints) to anchor the achievable range."""
    problem = SubsetProblem(np.asarray(comp, float), np.asarray(nd_F, float), K,
                            comp.shape[1], endpoints=endpoints)
    algo = GA(pop_size=pop_size, sampling=MySampling(), crossover=BinaryCrossover(),
              mutation=MyMutation(), eliminate_duplicates=True)
    res = minimize(problem, algo, ('n_gen', n_gen), verbose=False,
                   seed=None if seed is None else int(seed))
    return np.where(np.asarray(res.X).ravel())[0]


class BaselineJointSearch(Search):
    def __init__(self, config, accelerator, device_map, kwargs):
        super().__init__(config, accelerator, device_map, kwargs)
        self.accelerator = accelerator            # for accelerator.print (main-process-only output)
        a = self.args
        self.ref_partitions = a.get('ref_partitions', 12)  # 3-obj/12-part = 91 ref dirs
        self.seed = a.get('seed', 0)
        # post-NSGA-III down-select knobs ('subset' = legacy SubsetProblem; rest mirror second_search.py)
        self.cand_even = a.get('cand_even', 'subset')
        self.cand_grid = a.get('cand_grid', 0)
        self.even_frac = a.get('even_frac', 0.5)
        self.moo_algo = a.get('moo_algo', 'nsga3')
        self.moo_pop = a.get('moo_pop', 80)
        self.moo_gen = a.get('moo_gen', 80)
        self.moo_coverage = a.get('moo_coverage', 'rad')
        self.moo_gap_std = a.get('moo_gap_std', False)
        self.moo_objs = a.get('moo_objs', 'loss_cov')
        self.al_frac = a.get('al_frac', 0.0)
        self.xu = self._encoding_xu()

    def _encoding_xu(self):
        """Per-position upper bound of the full ss.encode() genome (len n_var)."""
        ss = self.search_space
        nb, nl = ss.n_block, ss.n_linear
        ub = np.ones((nl + 4, nb))
        for i, linear in enumerate(self.config['linear']):
            ub[i] = len(getattr(ss, f"{linear.split('.')[-1]}_option")) - 1
        ub[nl] = len(ss.k_option) - 1
        ub[nl + 1] = len(ss.v_option) - 1
        ub[nl + 2] = len(ss.k_pruning_dim_option) - 1
        ub[nl + 3] = len(ss.v_pruning_dim_option) - 1
        return ub.flatten().astype(int)

    def _next(self, archive, predictor, K):
        # ── NSGA-III over (predicted loss, wbits, eff_kvbits) ──
        F = np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)])
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        Ga = self._encode_archive(archive, self.search_space.encode, '_enc_cache')
        nd_X = Ga[front]

        endpoint_archs = self._get_endpoint_archs() if self.anchor_endpoints else []
        if endpoint_archs:
            nd_X = self._inject_endpoint_rows(nd_X, endpoint_archs)

        ref_dirs = get_reference_directions("das-dennis", len(self.comp_obj) + 1, n_partitions=self.ref_partitions)
        if self.ga_pop_size < len(ref_dirs):
            self.accelerator.print(f'[NSGA3] warning: ga_pop_size ({self.ga_pop_size}) < #ref_dirs ({len(ref_dirs)})')
        method = NSGA3(pop_size=self.ga_pop_size, ref_dirs=ref_dirs, sampling=nd_X,
                       crossover=BinomialCrossover(prob=self.crossover_prob, n_offsprings=1),
                       mutation=IntMutation(prob=self.mut_prob),
                       eliminate_duplicates=True)
        problem = AuxiliarySingleLevelProblemThink(
            self.search_space, predictor, self.config, self.comp_obj,
            self.comp_obj_max, self.comp_obj_min, self.group_size, self.n_token, self.attn_sink)
        res = minimize(problem, method, termination=('n_gen', 20), save_history=True, verbose=True)

        res_pop = res.pop
        if endpoint_archs:
            res_pop = self._merge_endpoint_pop(res_pop, endpoint_archs, problem)

        # ── post-NSGA-III candidate down-select (mirrors second_search.py::_next) ──
        seen = {tuple(g) for g in Ga}
        Xc = np.unique(np.clip(np.round(res_pop.get('X')), 0, self.xu).astype(int), axis=0)
        Xc = np.array([g for g in Xc if tuple(g) not in seen])
        self.accelerator.print(f'not_duplicate : {len(Xc)}')
        if len(Xc) == 0:
            return [], np.zeros((0, 1))

        pred = np.asarray(predictor.predict(self.search_space.decode_encode_predictor(Xc))).reshape(-1, 1)
        predr = pred.ravel()
        archs = [self.search_space.decode(g) for g in Xc]
        comp = np.array([[get_net_info(a, self.config, self.group_size, n_token=self.n_token,
                                       attn_sink=self.attn_sink)[o] for o in self.comp_obj] for a in archs])

        # optional ACTIVE-LEARNING quota (--al_frac): farthest-from-archive in predictor-input space
        n_al = int(round(self.al_frac * K)) if len(Xc) > K else 0
        K_sel = K - n_al
        if len(Xc) > K:
            g = self.cand_grid if self.cand_grid > 0 else int(np.ceil(np.sqrt(max(K, 1))))
            if self.cand_even == 'subset':                  # legacy union std-gap (hole-filling; default)
                ep = self._subset_endpoints() if self.anchor_endpoints else None
                idx = subset_select(comp, F[front][:, 1:], K_sel, self.subset_pop_size,
                                    endpoints=ep, seed=self.seed)
            elif self.cand_even == 'maximin':               # extent coverage
                z = (comp - comp.mean(0)) / (comp.std(0) + 1e-9)
                idx = np.asarray(maximin_extras(z, anchor_idx=[], K=K_sel, seed=self.seed), int)
            elif self.cand_even == 'grid':                  # per-axis-even quota over the box
                idx = np.asarray(even_select(comp, predr, K_sel, g, self.comp_obj_min, self.comp_obj_max), int)
            elif self.cand_even == 'moo':                   # loss×coverage knee, OR geometry-only axis_gap+cov_rad
                idx = np.asarray(moo_subset_select(comp, predr, K_sel, self.comp_obj_min, self.comp_obj_max, g,
                                                   algo=self.moo_algo, pop=self.moo_pop, n_gen=self.moo_gen,
                                                   seed=self.seed, gap_std=self.moo_gap_std,
                                                   coverage=self.moo_coverage, objs=self.moo_objs), int)
            else:                                           # hybrid: front pressure + grid-even coverage
                k_even = int(round(self.even_frac * K_sel)); k_front = K_sel - k_even
                fr = np.argsort(predr)[:k_front]
                rest = np.setdiff1d(np.arange(len(Xc)), fr)
                ev = (rest[even_select(comp[rest], predr[rest], k_even, g, self.comp_obj_min, self.comp_obj_max)]
                      if len(rest) else np.array([], int))
                idx = np.concatenate([fr, ev]).astype(int)
            if n_al > 0:
                rest = np.setdiff1d(np.arange(len(Xc)), idx)
                if len(rest):
                    from scipy.spatial.distance import cdist
                    A = self.search_space.decode_encode_predictor(Ga).astype(float)
                    R = self.search_space.decode_encode_predictor(Xc[rest]).astype(float)
                    mu, sd = A.mean(0), A.std(0); sd[sd < 1e-9] = 1.0
                    dmin = cdist((R - mu) / sd, (A - mu) / sd).min(1)
                    idx = np.concatenate([idx, rest[np.argsort(-dmin)[:n_al]]]).astype(int)
            archs = [archs[i] for i in idx]
            pred = pred[idx]
        return archs, pred

    # ── main loop — MULTI-PROCESS SAFE (mirrors search.py's accelerator structure) ──
    # With `accelerate launch --num_processes N>1`, candidate GENERATION, surrogate FIT,
    # printing and iter_<it>.stats dumps run on the MAIN rank only; the arch/candidate
    # lists are broadcast with gather_for_metrics so every rank EVALUATES the same set
    # (the evaluator shards the calibration batches across ranks and gathers → ~N× eval
    # speedup). Without these guards every rank reruns the whole loop → the duplicated
    # prints / redundant search / iter_<it>.stats file-races seen at np>1. Reduces to the
    # plain single-process loop at num_processes=1 (is_main always True, gather/barrier
    # no-ops). NOTE: the DP speedup needs #calibration batches >= num_processes.
    def search(self, accelerator):
        main = accelerator.is_main_process
        t0 = time(); start_it = 1
        if self.resume:                                       # resume from an iter_<it>.stats
            archive, start_it = self._resume_from_dir()
            if main:
                print(f"[resume] {len(archive)} archs → start iter {start_it}")
        else:
            arch_doe = self.search_space.initialize(self.n_doe, anchor_levels=self.anchor_levels) if main else []
            arch_doe = accelerator.gather_for_metrics(arch_doe, use_gather_object=True)   # → all ranks
            accelerator.wait_for_everyone()
            metric, comp = self._evaluate(arch_doe, accelerator)          # data-parallel across ranks
            archive = [[a, m, *c] for a, m, c in zip(arch_doe, metric, comp)] if main else []
            if main:
                print(f"[DOE] {len(archive)} archs  loss {min(metric):.4f}-{max(metric):.4f}  ({time()-t0:.1f}s)")
        if main:
            ref_pt = np.array([np.max([x[i] for x in archive]) for i in range(1, len(self.comp_obj) + 2)])
            print(f'data preparation time : {time()-t0:.2f}s')
        accelerator.wait_for_everyone()

        for it in range(start_it, self.iterations + 1):
            iter_start = time()
            if main:
                predictor_start = time()
                predictor, a_pred = self._fit_predictor(archive)
                predictor_time = time() - predictor_start
                next_start = time()
                cands, c_pred = self._next(archive, predictor, self.n_iter)
                next_time = time() - next_start
            else:
                cands = []
            accelerator.wait_for_everyone()
            cands = accelerator.gather_for_metrics(cands, use_gather_object=True)   # broadcast to all ranks
            if not cands:
                if main:
                    print(f"Iter {it}: no new candidates; stop")
                break
            c_metric, c_comp = self._evaluate(cands, accelerator)         # data-parallel across ranks
            if main:
                # check accuracy predictor's performance (ravel so RMSE is exact, like second_search.py)
                rmse, rho, tau = get_correlation(
                    np.concatenate([np.asarray(a_pred).ravel(), np.asarray(c_pred).ravel()]),
                    np.array([x[1] for x in archive] + c_metric))
                for a, m, c in zip(cands, c_metric, c_comp):
                    archive.append([a, m, *c])
                F = np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)])
                hv = self._calc_hv(ref_pt, F); cov = self._front_coverage(archive)
                iter_time = time() - iter_start
                print(f"Iter {it}: hv = {hv:.2f}, iter time : {iter_time:.2f}s, "
                      f"predictor_time : {predictor_time:.2f}, next_time : {next_time:.2f}")
                print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall's Tau = {tau:.4f}")
                for obj in self.comp_obj:
                    c = cov[obj]
                    print(f"  {obj} front-coverage : {c['coverage']*100:.1f}%  "
                          f"front=[{c['front_min']:.3f}, {c['front_max']:.3f}] / "
                          f"full=[{c['full_min']:.3f}, {c['full_max']:.3f}]")
                print(f'iteration time : {iter_time:.2f}s')
                # dump stats (+ per-save_iter viz), also always on the final iter
                if it % self.save_iter == 0 or it == self.iterations:
                    os.makedirs(self.save_path, exist_ok=True)
                    with open(os.path.join(self.save_path, f"iter_{it}.stats"), 'w') as f:
                        json.dump({'archive': archive, 'candidates': archive[-self.n_iter:], 'hv': hv,
                                   'surrogate': {'model': self.predictor, 'name': self.predictor,
                                                 'winner': self.predictor, 'rmse': rmse, 'rho': rho,
                                                 'tau': tau, 'total_time': iter_time},
                                   'coverage': cov, 'iteration': it}, f)
                    if self.debug:
                        save_viz(self.save_path, it, archive, c_metric, c_pred, c_comp, cov,
                                 self.comp_obj, self.comp_obj_min, self.comp_obj_max)
            accelerator.wait_for_everyone()

        if main:
            print(f"[done] {len(archive)} archs, {time()-t0:.1f}s → {self.save_path}")
            self._write_results(archive, time() - t0)
        return archive

    def _write_results(self, archive, total_time):
        """Final run summary → <save>/<result_file> (mirrors second_search.py): all args +
        total time, plus archive/front size, best loss, and per-obj front coverage."""
        os.makedirs(self.save_path, exist_ok=True)
        losses = [x[1] for x in archive]
        F = np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)])
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        cov = self._front_coverage(archive)
        lines = [f"{k}: {v}\n" for k, v in self.args.items()]
        lines.append(f"\nTotal time: {total_time:.2f}s\n")
        lines.append(f"archive: {len(archive)} archs | front: {len(nd)} | "
                     f"loss {min(losses):.4f}-{max(losses):.4f} (best {min(losses):.4f})\n")
        for obj in self.comp_obj:
            c = cov[obj]
            lines.append(f"  {obj}: front=[{c['front_min']:.3f}, {c['front_max']:.3f}] "
                         f"full=[{c['full_min']:.3f}, {c['full_max']:.3f}] coverage {c['coverage']*100:.1f}%\n")
        with open(os.path.join(self.save_path, self.result_file), 'w') as f:
            f.writelines(lines)
        print(f"[results] {os.path.join(self.save_path, self.result_file)}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Single-stage joint (loss × wbits × eff_kvbits) NSGA-III NAS baseline")

    # search loop
    p.add_argument('--save', type=str, default='save/baseline_search/run')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--result_file', type=str, default='results.txt')
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--n_doe', type=int, default=500)
    p.add_argument('--n_iter', type=int, default=50)
    p.add_argument('--anchor_levels', type=int, default=3,
                   help='thin each DOE anchor axis to N evenly spaced options (3=min/mid/max); 0=full grid. '
                        'Keep >0 for the full joint space (raw product explodes past n_doe).')
    p.add_argument('--predictor', type=str, default='rbf')
    p.add_argument('--save_iter', type=int, default=10)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--seed', type=int, default=0)

    # NSGA-III
    p.add_argument('--ga_pop_size', type=int, default=200, help='NSGA-III population (>= #ref_dirs)')
    p.add_argument('--ref_partitions', type=int, default=12, help='das-dennis partitions (3-obj/12 → 91 dirs)')
    p.add_argument('--mut_prob', type=float, default=0.1)
    p.add_argument('--crossover_prob', type=float, default=0.9)
    p.add_argument('--anchor_endpoints', action='store_true')
    p.add_argument('--ga_algorithm', type=str, default='nsga3')
    p.add_argument('--max_value', type=float, default=0.7, help='loss cap for failed/nan evals')

    # post-NSGA-III candidate down-select. 'subset' (default) = search.py's legacy
    # SubsetProblem: union(front, picks) std-of-gaps GA → fills comp-space holes, keeps
    # edge candidates (29/30 injected right-end kept vs moo 13/30). Others mirror
    # second_search.py and score the picked subset's own geometry only.
    p.add_argument('--cand_even', default='subset', choices=['subset', 'maximin', 'grid', 'hybrid', 'moo'])
    p.add_argument('--subset_pop_size', type=int, default=100, help='subset: GA population size')
    p.add_argument('--cand_grid', type=int, default=0, help='grid side for grid/hybrid/moo (0=auto=ceil(sqrt(n_iter)))')
    p.add_argument('--even_frac', type=float, default=0.5, help='hybrid: fraction of K on grid-even coverage')
    p.add_argument('--moo_algo', default='nsga3', choices=['nsga3', 'nsga2'])
    p.add_argument('--moo_pop', type=int, default=80)
    p.add_argument('--moo_gen', type=int, default=80)
    p.add_argument('--moo_coverage', default='rad', choices=['rad', 'gap'])
    p.add_argument('--moo_gap_std', action='store_true', help='moo: add gap-std as a 3rd objective')
    p.add_argument('--moo_objs', default='loss_cov', choices=['loss_cov', 'axis_gap'],
                   help="moo objectives: 'loss_cov' (default) = (mean pred-loss × coverage[+gap_std]) knee; "
                        "'axis_gap' = GEOMETRY-ONLY, NO loss = one std-of-gaps objective per comp axis "
                        "(wbits, eff_kvbits) + 2D cov_rad → spreads candidates across the whole bit box "
                        "instead of clustering where the loss knee pulls")
    p.add_argument('--al_frac', type=float, default=0.0, help='fraction of K reserved for farthest-from-archive AL picks')

    # objectives (the 3-D joint front). Bounds must be generous enough that DOE boundary
    # anchors stay feasible (wbits max > uniform-4-bit ~4.06; eff_kvbits min < pruned corner).
    p.add_argument('--comp_obj', type=str, nargs='+', default=['wbits', 'eff_kvbits'],
                   choices=['wbits', 'kvbits', 'kbits', 'vbits', 'memory', 'kvdim', 'kdim', 'vdim',
                            'eff_kvbits', 'eff_kbits', 'eff_vbits'])
    p.add_argument('--comp_obj_min', type=float, nargs='+', default=[2, 0.1])
    p.add_argument('--comp_obj_max', type=float, nargs='+', default=[5, 5])
    p.add_argument('--n_token', type=int, default=0)
    p.add_argument('--attn_sink', type=int, default=8)

    # model / quantization
    p.add_argument('--config', type=str, default='config/llama.json')
    p.add_argument('--gpu_id', type=str, default='0')
    p.add_argument('--model_path', type=str, default='/SSD/huggingface/meta-llama')
    p.add_argument('--model_name', type=str, default='Llama-3.1-8B-Instruct')
    p.add_argument('--dtype', type=str, default='bfloat16',
                   choices=['float16', 'float', 'fp16', 'bfloat16', 'bfloat', 'bf16', 'auto'])
    p.add_argument('--w_method', type=str, nargs='+', default=['hqq'], choices=['fp16', 'awq', 'gptq', 'qeft', 'hqq'])
    p.add_argument('--kv_method', type=str, nargs='+', default=['kivi', 'think'], choices=['fp16', 'hqq', 'kivi', 'think'])
    p.add_argument('--quant_model_paths', type=str, nargs='+', default=[], help='HQQ dirs, one per --w_bits')
    p.add_argument('--w_bits', type=int, nargs='+', default=[2, 3, 4])
    p.add_argument('--k_bits', type=int, nargs='+', default=[2, 3, 4])
    p.add_argument('--v_bits', type=int, nargs='+', default=[2, 3, 4])
    p.add_argument('--w_group_size', type=int, default=128)
    p.add_argument('--k_group_size', type=int, nargs='+', action='append', default=[],
                   help='per-bit-width K group sizes, repeated once per --k_bits; empty → [[128]]')
    p.add_argument('--v_group_size', type=int, nargs='+', action='append', default=[])
    p.add_argument('--k_pruning_dim', type=int, nargs='+', default=[0, 16, 32, 48, 64])
    p.add_argument('--v_pruning_dim', type=int, nargs='+', default=[0, 16, 32, 48, 64])
    p.add_argument('--residual_length', type=int, default=128)
    p.add_argument('--k_quant_scheme', type=str, default='channel', choices=['channel', 'token'])
    p.add_argument('--v_quant_scheme', type=str, default='token', choices=['channel', 'token'])
    p.add_argument('--quant_kv_output', action='store_true')

    # QEFT outlier-column axis (off by default)
    p.add_argument('--base_outlier_bits', type=int, nargs='+', default=[])
    p.add_argument('--outlier_path', type=str, default='')
    p.add_argument('--n_outlier', type=int, default=0)
    p.add_argument('--n_qeft_column', type=int, nargs='+', default=[0])
    p.add_argument('--only_outlier_bits', action='store_true')

    # data / measurement protocol
    p.add_argument('--dataset', type=str, default='wikitext2')
    p.add_argument('--n_sample', type=int, default=128)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--min_seqlen', type=int, default=0)
    p.add_argument('--data_batch_size', type=int, default=1)
    p.add_argument('--metric', type=str, default='loss')
    p.add_argument('--loss_func', type=str, default='jsd')
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--prefill_prompt', action='store_true')
    p.add_argument('--last_tokens', type=int, default=512)

    # long-PPL / key-token knobs (kept for parity, unused by default)
    p.add_argument('--use_key_token', action='store_true')
    p.add_argument('--trunc_len', type=int, default=512)
    p.add_argument('--sliding_window', type=int, default=128)
    p.add_argument('--alpha', type=int, default=2)
    p.add_argument('--beta', type=int, default=-2)
    p.add_argument('--key_token_path', type=str, default='')

    # lm_eval / sensitivity (kept for parity, unused for loss search)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--lm_eval_batch_size', type=int, default=None)
    p.add_argument('--num_fewshot', type=int, default=None)
    p.add_argument('--verbosity', type=str, default='FATAL')
    p.add_argument('--sensitivity_result_path', type=str, default='')
    p.add_argument('--sensitivity_threshold', type=int, default=2)
    return p


def main(args):
    if not args.k_group_size:   # action='append' leaves an empty list when omitted
        args.k_group_size = [[128]]
    if not args.v_group_size:
        args.v_group_size = [[128]]

    n_comp = len(args.comp_obj)
    assert len(args.comp_obj_min) == n_comp and len(args.comp_obj_max) == n_comp, \
        "comp_obj / comp_obj_min / comp_obj_max lengths must match"
    assert len(args.w_bits) == len(args.quant_model_paths), \
        "one --quant_model_paths entry is required per --w_bits value"

    set_seed(args.seed)
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)
    engine = BaselineJointSearch(config=config, accelerator=accelerator, device_map=device_map, kwargs=vars(args))
    engine.search(accelerator)


if __name__ == '__main__':
    main(build_parser().parse_args())
