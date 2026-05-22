import os
import re
import json
import atexit
import multiprocessing as mp

import numpy as np
from time import time
from matplotlib import pyplot as plt

from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2

from search.space import SearchSpace
from search.problem import AuxiliarySingleLevelProblem, SubsetProblem
from predictor.factory import get_predictor
from utils.ga import MySampling, BinaryCrossover, MyMutation, IntMutation
from utils.func import get_correlation, set_seed

# =========================================================
# Worker
# =========================================================

def _parse_n_sample_arg(raw, datasets):
    """Parse --n_sample list into int or {dataset: int}.

    Accepted forms:
      - ["16"]                         -> 16
      - ["gsm8k=5", "livebench=4"]     -> {"gsm8k": 5, "livebench": 4}
      - ["5", "4"] with datasets       -> {"gsm8k": 5, "livebench": 4}
        (matched positionally to ``datasets`` order; lengths must match)

    wikitext2 entries in ``datasets`` are skipped — wikitext2 ignores
    n_sample (entire fixed pool used per iter), so positional matching is
    done against the non-wikitext2 subset only.
    """
    from evaluation.data_io import is_wikitext2 as _is_wikitext2

    if raw is None or len(raw) == 0:
        raise ValueError("--n_sample is empty")
    if len(raw) == 1 and "=" not in raw[0]:
        return int(raw[0])
    if all("=" in s for s in raw):
        out = {}
        for kv in raw:
            k, v = kv.split("=", 1)
            out[k.strip()] = int(v)
        return out
    if all("=" not in s for s in raw):
        targets = [ds for ds in datasets if not _is_wikitext2(ds)]
        if len(raw) != len(targets):
            raise ValueError(
                f"--n_sample positional form needs one value per "
                f"non-wikitext2 dataset (got {len(raw)} for "
                f"datasets={list(datasets)}, wikitext2-excluded "
                f"targets={targets})"
            )
        return {ds: int(v) for ds, v in zip(targets, raw)}
    raise ValueError("--n_sample mixes 'k=v' and bare values; pick one form")


def worker_evaluate_loop(
    worker_id,                   # index within the jsd pool
    tqdm_position,               # globally unique row for tqdm
    visible_gpu_ids,             # e.g. ["0"], ["1"], ["0","1"]
    config,
    model_id,
    quantization_proxy_paths,
    quantization_proxy_fake_paths,
    group_size,
    datasets,
    n_sample,
    batch_size,
    sample_seed,
    max_value,
    task_queue,
    result_queue,
    jsd_stratified=False,
    loader="plain",
):
    """
    Persistent jsd worker.
    """

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_gpu_ids))

        import torch
        if jsd_stratified:
            from evaluation.evaluator_jsd_stratified import BenchmarkProxyEvaluator
            print("jsd-bin stratified sample")
        else:
            from evaluation.evaluator import BenchmarkProxyEvaluator
            print("all random sample")

        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        local_gpu_ids = list(range(len(visible_gpu_ids)))

        evaluator = BenchmarkProxyEvaluator(
            config=config,
            gpu_id=local_gpu_ids,
            model_id=model_id,
            quantization_proxy_paths=quantization_proxy_paths,
            quantization_proxy_fake_paths=quantization_proxy_fake_paths,
            bits_range=[2, 3, 4],
            group_size=group_size,
            datasets=datasets,
            n_sample=n_sample,
            batch_size=batch_size,
            sample_seed=sample_seed,
            mode="jsd",
            tqdm_position=tqdm_position,
            loader=loader,
        )

        result_queue.put({
            "worker_id": worker_id,
            "kind": "init",
            "ok": True,
        })

        while True:
            item = task_queue.get()
            if item is None:
                break
            job_id = item["job_id"]
            architectures = item["architectures"]

            try:
                if len(architectures) == 0:
                    metric_list = {}
                    for ds in datasets:
                        metric_list[f"{ds}_jsd"]  = []
                    bits_usage_list = []
                else:
                    metric_list, bits_usage_list = evaluator.eval(
                        architectures=architectures,
                        max_value=max_value,
                    )

                result_queue.put({
                    "worker_id": worker_id,
                    "kind": "result",
                    "job_id": job_id,
                    "ok": True,
                    "result": {
                        "metric_list": metric_list,
                        "bits_usage_list": bits_usage_list,
                    },
                })

            except Exception as e:
                result_queue.put({
                    "worker_id": worker_id,
                    "kind": "result",
                    "job_id": job_id,
                    "ok": False,
                    "error": repr(e),
                })

    except Exception as e:
        result_queue.put({
            "worker_id": worker_id,
            "kind": "init",
            "ok": False,
            "error": repr(e),
        })


class Search:
    def __init__(self, args, config):
        self.args = args
        self.quantization_proxy_paths = args.quantization_proxy_paths
        self.quantization_proxy_fake_paths = args.quantization_proxy_fake_paths
        self.resume_path = args.resume_path
        self.iterations = args.iterations
        self.n_doe = args.n_doe
        self.n_iter = args.n_iter
        self.save_iter = args.save_iter
        self.crossover_prob = args.crossover_prob
        self.mut_prob = args.mut_prob
        self.ga_pop_size = args.ga_pop_size
        self.subset_pop_size = args.subset_pop_size
        self.predictor = args.predictor
        self.save_path = args.save_path
        self.result_file = args.result_file
        self.max_value = args.max_value
        self.datasets = args.datasets
        self.n_sample = _parse_n_sample_arg(args.n_sample, self.datasets)
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.jsd_stratified = getattr(args, "jsd_stratified", False)
        self.loader = getattr(args, "loader", "plain")

        self.group_size = 128
        self.config = config

        self.gpu_id = args.gpu_id

        self.model_id = f'{args.model_path}/{args.model_name}'

        self.sensitivity_json = args.sensitivity_json
        self.sensitivity_threshold = args.sensitivity_threshold

        set_seed(self.seed)

        linear_list = list(self.sensitivity_json['loss'].keys())
        medium = np.median(list(map(float, self.sensitivity_json['loss'].values())))
        pass_linear_list = [linear for linear in linear_list if self.sensitivity_json['loss'][linear] > medium * self.sensitivity_threshold]

        self.gpu_id = args.gpu_id
        self.gpu_id_list = [x.strip() for x in args.gpu_id.split(",")]

        self.jsd_tp = args.jsd_tp

        self.search_space = SearchSpace(
            config=self.config,
            n_block=self.config['n_block'],
            n_linear=len(self.config['linear']),
            group_size=self.group_size,
            pass_linear_list=pass_linear_list,
            bits_range=[2, 3, 4],
            seed=self.seed,
        )

        self.ctx = mp.get_context("spawn")

        n_gpu = len(self.gpu_id_list)
        if len(self.gpu_id_list) % self.jsd_tp != 0:
            raise ValueError(
                f"jsd gpus ({n_gpu}) not divisible by jsd_tp={self.jsd_tp}"
            )

        n_jsd_workers = n_gpu // self.jsd_tp
        if n_jsd_workers == 0:
            raise ValueError(
                f"need at least one jsd worker. "
                f"jsd_gpus={self.gpu_id_list}, jsd_tp={self.jsd_tp}"
            )

        self.jsd_worker_gpus = [
            self.gpu_id_list[i * self.jsd_tp:(i + 1) * self.jsd_tp]
            for i in range(n_jsd_workers)
        ]

        self.result_queue = self.ctx.Queue()
        self.jsd_task_queues = [self.ctx.Queue() for _ in range(n_jsd_workers)]

        self.jsd_processes = []

        orig = os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        for wid, visible in enumerate(self.jsd_worker_gpus):
            p = self.ctx.Process(
                target=worker_evaluate_loop,
                args=(
                    wid,
                    wid,  # tqdm_position
                    visible,
                    self.config,
                    self.model_id,
                    self.quantization_proxy_paths,
                    self.quantization_proxy_fake_paths,
                    self.group_size,
                    self.datasets,
                    self.n_sample,
                    self.batch_size,
                    self.seed,
                    self.max_value,
                    self.jsd_task_queues[wid],
                    self.result_queue,
                    self.jsd_stratified,
                    self.loader,
                ),
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
            p.start()
            self.jsd_processes.append(p)

        if orig is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self.job_counter = 0
        self._pending_results = {}
        self._closed = False

        self._wait_for_init()
        atexit.register(self.close_workers)

    def search(self):
        total_start = time()
        start_it = 1

        # Per-iter HV trace, kept for traceability / resume.
        self._hv_trace = []

        if self.resume_path:
            archive, start_it = self._resume_from_dir()
        else:
            archive = []

            if self.iterations < 1:
                architectures_doe = self.search_space.sample(
                    n_samples=self.n_doe,
                    pool=[x[0] for x in archive])
            else:
                architectures_doe = self.search_space.initialize(self.n_doe, pool=[x[0] for x in archive])

            results = self.evaluate_parallel(architectures=architectures_doe)
            bits_usage_list = results["bits_usage_list"]
            n_ds = len(self.datasets)

            for idx in range(len(architectures_doe)):
                jsds  = [results["metric_list"][f"{ds}_jsd"][idx]  for ds in self.datasets]
                archive.append((
                    architectures_doe[idx],
                    *jsds,
                    sum(jsds) / n_ds,    # jsd_avg
                    bits_usage_list[idx],
                ))

        # archive entry layout:
        #   (arch, *jsd_per_ds, jsd_avg, bits)
        # indices: jsd_avg at 1+n_ds, bits at -1.
        n_ds = len(self.datasets)
        _JSD_AVG = 1 + n_ds

        if archive:
            ref_pt = np.array([np.max([x[_JSD_AVG] for x in archive]),
                               np.max([x[-1]      for x in archive])])
        else:
            raise NotImplementedError

        print(f'data preparation time : {time() - total_start:.2f}s')

        for it in range(start_it, self.iterations + 1):
            # Make `it` visible to inner GAs so each iter uses an independent
            # NSGA2 / SubsetGA random stream (seed = self.seed + self._it
            # rather than the same self.seed every iter).
            self._it = it
            iter_start = time()

            predictor_start = time()
            jsd_predictors, archive_preds = self._fit_predictor(
                archive, device='cuda')
            predictor_time = time() - predictor_start

            next_start = time()
            candidates, candidate_preds = self._next(
                archive, jsd_predictors, self.n_iter)
            next_time = time() - next_start

            results = self.evaluate_parallel(architectures=candidates)

            n_ds = len(self.datasets)
            # Per-dataset surrogate correlation for jsd.
            rmse = {"jsd": {}}
            rho  = {"jsd": {}}
            tau  = {"jsd": {}}
            for ds_idx, ds in enumerate(self.datasets):
                ds_archive_jsd = np.array([x[1 + ds_idx] for x in archive])
                ds_pred_jsd    = np.concatenate(
                    (archive_preds['jsd'][ds], candidate_preds['jsd'][ds]))
                ds_target_jsd  = np.concatenate(
                    (ds_archive_jsd, results["metric_list"][f"{ds}_jsd"]))
                rmse['jsd'][ds], rho['jsd'][ds], tau['jsd'][ds] = \
                    get_correlation(ds_pred_jsd, ds_target_jsd)

            cand_bits_usage_list = results["bits_usage_list"]
            for idx in range(len(candidates)):
                jsds  = [results["metric_list"][f"{ds}_jsd"][idx]  for ds in self.datasets]
                J_avg = sum(jsds) / n_ds
                archive.append((
                    candidates[idx],
                    *jsds,
                    J_avg,
                    cand_bits_usage_list[idx],
                ))

            hv = self._calc_hv(
                ref_pt, np.column_stack(([x[_JSD_AVG] for x in archive],
                                         [x[-1]      for x in archive])))
            self._hv_trace.append(float(hv))

            iter_time = time() - iter_start
            print(f"Iter {it}: hv = {hv:.2f}, "
                  f"iter time : {(time() - iter_start):.2f}s, "
                  f"predictor_time : {predictor_time:.2f}, next_time : {next_time:.2f}")
            for ds in self.datasets:
                print(f"jsd  fitting {self.predictor} [{ds}]: RMSE = {rmse['jsd'][ds]:.4f}, "
                      f"Spearman's Rho = {rho['jsd'][ds]:.4f}, Kendall's Tau = {tau['jsd'][ds]:.4f}")
            print(f'iteration time : {iter_time:.2f}s')

            if it % self.save_iter == 0:
                os.makedirs(self.save_path, exist_ok=True)
                with open(os.path.join(self.save_path, f"iter_{it}.stats"), "w") as handle:
                    json.dump({
                        'iteration': it,
                        'archive': archive,
                        'candidates': candidates,
                        'hv': hv,
                        'history': {
                            'hv': self._hv_trace,
                        },
                        'surrogate': {
                            'jsd': {
                                'model': self.predictor,
                                'rmse': rmse['jsd'],
                                'rho':  rho['jsd'],
                                'tau':  tau['jsd'],
                            },
                            'total_time': iter_time,
                        }}, handle)

                self._plot_iter(it, archive, candidates, candidate_preds)

        total_time_elapsed = time() - total_start
        print(f'total time elapsed : {total_time_elapsed:.2f}s')

        sentences = []
        for k, v in self.args.__dict__.items():
            sentences.append(f"{k}: {v}\n")
        sentences.append(f'Total time: {total_time_elapsed:.2f}s')

        with open(os.path.join(self.save_path, self.result_file), 'w') as f:
            for sentence in sentences:
                f.write(sentence)

        print(self.args)

        return

    def _resume_from_dir(self):
        with open(self.resume_path, 'r') as f:
            resume_file = json.load(f)
            archive = resume_file['archive']
            it = resume_file['iteration']

        try:
            hist = resume_file.get("history") or {}
            if "hv" in hist:
                self._hv_trace = [float(v) for v in hist.get("hv", [])]
                print(f"Restored history from resume file: "
                      f"hv_trace={len(self._hv_trace)} points")
            else:
                resume_dir = os.path.dirname(self.resume_path)
                per_iter = []
                for fn in os.listdir(resume_dir):
                    m = re.match(r"iter_(\d+)\.stats$", fn)
                    if not m:
                        continue
                    k = int(m.group(1))
                    if k > it:
                        continue
                    with open(os.path.join(resume_dir, fn), "r") as g:
                        s = json.load(g)
                    per_iter.append((k, s))
                per_iter.sort(key=lambda x: x[0])
                self._hv_trace = [float(s["hv"]) for _, s in per_iter if "hv" in s]
                print(f"Restored hv_trace by scanning stats: "
                      f"{len(self._hv_trace)} points")
        except Exception as e:
            print(f"Warning: failed to restore hv state ({e!r}); starting cold.")

        return archive, it + 1

    def _fit_predictor(self, archive, device='cpu'):
        """Fit one JSD predictor per dataset.

        archive entry layout:
            (arch, *jsd_per_ds, jsd_avg, bits)
        so for dataset i: jsd target at x[1+i].

        Returns
        -------
        jsd_predictors : {ds: predictor}
        archive_preds  : {'jsd': {ds: np.ndarray}}
        """
        inputs = np.array([self.search_space.encode_predictor(x[0]) for x in archive])
        jsd_targets = {ds: np.array([x[1 + i] for x in archive], dtype=np.float32)
                       for i, ds in enumerate(self.datasets)}

        kwargs = {}
        if self.predictor == 'rbf' or self.predictor == 'rbf_gpu':
            n_block = self.config['n_block']
            n_linear = self.config['n_linear']
            lb = np.zeros((n_linear, n_block))
            ub = np.ones((n_linear, n_block))

            for linear_idx, linear in enumerate(self.config['linear']):
                ub[linear_idx] = len(self.search_space.bits_range) - 1

            lb = np.delete(lb.flatten(), self.search_space.pass_linear_idx_list, axis=-1)
            ub = np.delete(ub.flatten(), self.search_space.pass_linear_idx_list, axis=-1)

            kwargs = {'lb': lb, 'ub': ub}

        jsd_predictors = {
            ds: get_predictor(self.predictor, inputs, jsd_targets[ds],
                              device=device, **kwargs)
            for ds in self.datasets
        }
        archive_preds = {
            'jsd': {ds: np.asarray(jsd_predictors[ds].predict(inputs)).reshape(-1)
                    for ds in self.datasets},
        }

        return jsd_predictors, archive_preds

    def _next(self, archive, jsd_predictors, K):
        # archive entry: (arch, *jsd_per_ds, jsd_avg, bits)
        n_ds = len(self.datasets)
        # Per-task normalization on J so the front objective matches the
        # NSGA2 objective in AuxiliarySingleLevelProblem.
        from evaluation.data_io import task_jsd_mean
        mu_norm = np.array([task_jsd_mean(ds) for ds in self.datasets])
        jsd_cols = np.array([[x[1 + i] for x in archive] for i in range(n_ds)])  # (n_ds, N)
        jsd_mean = (jsd_cols / mu_norm[:, None]).mean(axis=0)   # cross-task mean of normalized J
        bits     = np.array([x[-1] for x in archive])
        F = np.column_stack((jsd_mean, bits))
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        method = NSGA2(pop_size=self.ga_pop_size, sampling=nd_X,
            crossover=BinomialCrossover(prob=self.crossover_prob, n_offsprings=1),
            mutation=IntMutation(prob=self.mut_prob),
            eliminate_duplicates=True)

        problem = AuxiliarySingleLevelProblem(
            self.search_space, jsd_predictors,
            self.config, self.group_size)

        res = minimize(problem, method, termination=('n_gen', 20),
                       seed=self.seed + self._it,
                       save_history=False, verbose=True)

        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])
        print(f'not_duplicate : {sum(not_duplicate)}')

        pop = res.pop[not_duplicate]
        if sum(not_duplicate) >= K:
            indices = self._subset_selection(pop, F[front, 1], K, self.subset_pop_size)
            pop = pop[indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        x_pred = self.search_space.decode_encode_predictor(pop.get("X"))
        candidate_preds = {
            'jsd': {ds: np.asarray(jsd_predictors[ds].predict(x_pred)).reshape(-1)
                    for ds in self.datasets},
        }

        return candidates, candidate_preds

    def _subset_selection(self, pop, nd_F, K, pop_size):
        problem = SubsetProblem(pop.get("F")[:, 1], nd_F, K)
        algorithm = GA(
            pop_size=pop_size, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)

        res = minimize(
            problem, algorithm, ('n_gen', 60),
            seed=self.seed + self._it + 42,
            verbose=False)

        return res.X

    def _plot_iter(self, it, archive, candidates, candidate_preds):
        """Per-dataset (and average) bits-vs-jsd scatter.

        archive entry layout:
            (arch, *jsd_per_ds, jsd_avg, bits)
        so for col in 0..n_ds-1: jsd_col at index 1+col.
        jsd_avg lives at 1+n_ds (used for col == n_ds).

        candidate_preds layout (from _next):
            {'jsd': {ds: np.ndarray}}
        """
        n_ds = len(self.datasets)
        jsd_avg_idx = 1 + n_ds

        n_cand = len(candidates)
        prev_jsd = archive[:-n_cand] if n_cand > 0 else archive
        new_jsd  = archive[-n_cand:] if n_cand > 0 else []

        prev_jsd_bits = np.array([x[-1] for x in prev_jsd])
        new_jsd_bits  = np.array([x[-1] for x in new_jsd])

        cand_jsd_pred_mean = np.mean(
            np.stack([candidate_preds['jsd'][ds] for ds in self.datasets], axis=0), axis=0)

        series = []
        for i, ds in enumerate(self.datasets):
            series.append((1 + i, ds, candidate_preds['jsd'][ds]))
        series.append((jsd_avg_idx, 'average', cand_jsd_pred_mean))

        for jsd_col_idx, name, cand_jsd_pred in series:
            fig, ax_jsd = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

            if len(prev_jsd) > 0:
                prev_jsd_actual = np.array([x[jsd_col_idx] for x in prev_jsd])
                ax_jsd.scatter(prev_jsd_bits, prev_jsd_actual, s=10,
                               color='b', label='archive')
            if len(new_jsd) > 0:
                new_jsd_actual = np.array([x[jsd_col_idx] for x in new_jsd])
                ax_jsd.scatter(new_jsd_bits, new_jsd_actual, s=10,
                               color='r', label='new candidate (actual)')
                ax_jsd.scatter(new_jsd_bits, cand_jsd_pred, s=10,
                               color='g', label='new candidate (predicted)')
            ax_jsd.set_title(f'{name} jsd')
            ax_jsd.set_xlabel('bits'); ax_jsd.set_ylabel('score')
            ax_jsd.legend(); ax_jsd.grid(c='0.8')

            fig.tight_layout()
            plt.savefig(os.path.join(self.save_path, f'iter_{it}_{name}.png'))
            plt.close(fig)

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv = Hypervolume(ref_point=ref_point).do(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv

    # Multi Process Function
    def _terminate_all(self):
        for p in self.jsd_processes:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass

    def _wait_for_init(self):
        n_jsd = len(self.jsd_processes)
        inits = {}
        remaining = n_jsd

        while remaining > 0:
            msg = self.result_queue.get()
            if msg.get("kind") != "init":
                key = (msg["job_id"], msg["worker_id"])
                self._pending_results[key] = msg
                continue
            inits[msg["worker_id"]] = msg
            remaining -= 1

        for wid in range(n_jsd):
            m = inits.get(wid)
            if m is None or not m["ok"]:
                err = m["error"] if m else "no init message"
                self._terminate_all()
                raise RuntimeError(f"jsd worker {wid} init failed: {err}")

    def _chunk(self, items, n):
        if n <= 0:
            return []
        chunks = [[] for _ in range(n)]
        if len(items) == 0:
            return chunks
        for i, sub in enumerate(np.array_split(np.arange(len(items)), n)):
            chunks[i] = [items[j] for j in sub.tolist()]
        return chunks

    def _gather_pool(self, job_id, n_workers):
        collected = {}
        for wid in range(n_workers):
            key = (job_id, wid)
            if key in self._pending_results:
                collected[wid] = self._pending_results.pop(key)

        while len(collected) < n_workers:
            msg = self.result_queue.get()
            if msg.get("kind") != "result":
                continue
            mid = msg["job_id"]
            mwid = msg["worker_id"]
            if not msg["ok"]:
                self._terminate_all()
                raise RuntimeError(f"jsd[{mwid}] failed: {msg['error']}")
            if mid == job_id:
                collected[mwid] = msg
            else:
                self._pending_results[(mid, mwid)] = msg

        return [collected[wid] for wid in range(n_workers)]

    def _merge_pool(self, ordered_msgs):
        metric_list = {}
        for ds in self.datasets:
            metric_list[f"{ds}_jsd"] = []
        bits_usage_list = []
        for msg in ordered_msgs:
            r = msg["result"]
            for ds in self.datasets:
                metric_list[f"{ds}_jsd"].extend(r["metric_list"][f"{ds}_jsd"])
            bits_usage_list.extend(r["bits_usage_list"])
        return {"metric_list": metric_list, "bits_usage_list": bits_usage_list}

    def evaluate_parallel(self, architectures):
        job_id = self.job_counter
        self.job_counter += 1

        n_jsd = len(self.jsd_processes)
        jsd_chunks = self._chunk(list(architectures), n_jsd)

        for wid, chunk in enumerate(jsd_chunks):
            self.jsd_task_queues[wid].put({
                "job_id": job_id,
                "architectures": chunk,
            })

        jsd_msgs = self._gather_pool(job_id, n_jsd)
        return self._merge_pool(jsd_msgs)

    def close_workers(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        for q in getattr(self, "jsd_task_queues", []):
            try:
                q.put(None)
            except Exception:
                pass

        for p in self.jsd_processes:
            try:
                p.join(timeout=5)
            except Exception:
                pass

        for p in self.jsd_processes:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
