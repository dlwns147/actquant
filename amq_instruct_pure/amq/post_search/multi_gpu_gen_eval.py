"""Multi-GPU HQQ-proxy generation evaluation.

Spawns one persistent worker per GPU group (modeled on
``search/optimizer.py``'s worker pool), each holding its own long-lived
``BenchmarkProxyEvaluatorVLLM(mode='generate')``. Candidate
architectures are split across workers, evaluated in parallel, and the
per-dataset metric lists + bits_usage are concatenated in worker order.

Use as a context manager so the workers (and their vLLM engines) are
torn down before AWQ runs or before ``lm_eval`` spawns a separate vLLM
process.
"""

from __future__ import annotations

import os
import atexit
import multiprocessing as mp

import numpy as np


def _gen_worker_loop(
    worker_id,
    visible_gpu_ids,
    config,
    model_id,
    quantization_proxy_paths,
    quantization_proxy_fake_paths,
    bits_range,
    group_size,
    datasets,
    n_sample,
    batch_size,
    sample_seed,
    gpu_memory_utilization,
    max_model_len,
    tensor_parallel_size,
    task_queue,
    result_queue,
):
    """Persistent worker — one BenchmarkProxyEvaluatorVLLM per GPU group."""

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_gpu_ids))

        import torch
        from evaluation.evaluator_vllm import BenchmarkProxyEvaluatorVLLM

        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        local_gpu_ids = list(range(len(visible_gpu_ids)))

        evaluator = BenchmarkProxyEvaluatorVLLM(
            config=config,
            gpu_id=local_gpu_ids,
            model_id=model_id,
            quantization_proxy_paths=quantization_proxy_paths,
            quantization_proxy_fake_paths=quantization_proxy_fake_paths,
            bits_range=bits_range,
            group_size=group_size,
            datasets=datasets,
            n_sample=n_sample,
            batch_size=batch_size,
            seed=sample_seed,
            mode='generate',
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            tqdm_position=worker_id,
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
                    metric_list = {f"{ds}_gen": [] for ds in datasets}
                    bits_usage_list = []
                else:
                    metric_list, bits_usage_list = evaluator.eval_generate(
                        architectures=architectures,
                        max_value=None,
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


def _parse_gpu_id_list(gpu_id):
    """Accept "0,1", ["0","1"], or [0,1] — return list of strings."""
    if isinstance(gpu_id, str):
        return [x.strip() for x in gpu_id.split(",") if x.strip() != ""]
    return [str(x) for x in gpu_id]


class MultiGPUGenPerfEvaluator:
    """Multi-GPU pool of HQQ gen-eval workers.

    Splits ``args.gpu_id`` (comma-separated) into worker groups of size
    ``args.tensor_parallel_size`` and starts one worker per group. The
    candidate pool is distributed across workers via
    ``np.array_split``; results are concatenated in worker order so the
    output shape matches the single-worker path.
    """

    def __init__(self, args, config):
        self.args = args
        self.config = config

        gpu_ids = _parse_gpu_id_list(args.gpu_id)
        if len(gpu_ids) == 0:
            raise ValueError(f"--gpu_id is empty: {args.gpu_id!r}")

        tp = int(args.tensor_parallel_size)
        if tp <= 0:
            raise ValueError(f"tensor_parallel_size must be > 0, got {tp}")
        if len(gpu_ids) % tp != 0:
            raise ValueError(
                f"len(gpu_id)={len(gpu_ids)} not divisible by "
                f"tensor_parallel_size={tp}"
            )

        n_workers = len(gpu_ids) // tp
        self.worker_gpus = [
            gpu_ids[i * tp:(i + 1) * tp] for i in range(n_workers)
        ]

        self.ctx = mp.get_context("spawn")
        self.result_queue = self.ctx.Queue()
        self.task_queues = [self.ctx.Queue() for _ in range(n_workers)]
        self.processes = []

        self.job_counter = 0
        self._pending_results = {}
        self._closed = False
        self._started = False

    def __enter__(self):
        a = self.args
        model_id = f'{a.model_path}/{a.model_name}'

        orig = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            for wid, visible in enumerate(self.worker_gpus):
                p = self.ctx.Process(
                    target=_gen_worker_loop,
                    args=(
                        wid,
                        visible,
                        self.config,
                        model_id,
                        list(a.quantization_proxy_paths),
                        list(a.quantization_proxy_fake_paths),
                        list(a.bits_range),
                        a.group_size,
                        list(a.datasets),
                        a.n_sample,
                        a.batch_size,
                        a.seed,
                        a.gpu_memory_utilization,
                        a.max_model_len,
                        a.tensor_parallel_size,
                        self.task_queues[wid],
                        self.result_queue,
                    ),
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
                p.start()
                self.processes.append(p)
        finally:
            if orig is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = orig
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self._started = True
        atexit.register(self.close)
        self._wait_for_init()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ------------------------------------------------------------------ ipc

    def _wait_for_init(self):
        n = len(self.processes)
        seen = {}
        while len(seen) < n:
            msg = self.result_queue.get()
            if msg.get("kind") != "init":
                key = (msg["job_id"], msg["worker_id"])
                self._pending_results[key] = msg
                continue
            seen[msg["worker_id"]] = msg

        for wid in range(n):
            m = seen.get(wid)
            if m is None or not m["ok"]:
                err = m["error"] if m else "no init message"
                self._terminate_all()
                raise RuntimeError(f"gen worker {wid} init failed: {err}")

    @staticmethod
    def _chunk(items, n):
        chunks = [[] for _ in range(n)]
        if len(items) == 0:
            return chunks
        for i, sub in enumerate(np.array_split(np.arange(len(items)), n)):
            chunks[i] = [items[j] for j in sub.tolist()]
        return chunks

    def _gather(self, job_id, n_workers):
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
                raise RuntimeError(f"gen worker[{mwid}] failed: {msg['error']}")
            if mid == job_id:
                collected[mwid] = msg
            else:
                self._pending_results[(mid, mwid)] = msg

        return [collected[wid] for wid in range(n_workers)]

    def _merge(self, ordered_msgs):
        datasets = list(self.args.datasets)
        metric_list = {f"{ds}_gen": [] for ds in datasets}
        bits_usage_list = []
        for msg in ordered_msgs:
            r = msg["result"]
            for ds in datasets:
                metric_list[f"{ds}_gen"].extend(r["metric_list"][f"{ds}_gen"])
            bits_usage_list.extend(r["bits_usage_list"])
        return metric_list, bits_usage_list

    # ------------------------------------------------------------------ api

    def evaluate(self, archs):
        if not self._started:
            raise RuntimeError("MultiGPUGenPerfEvaluator must be entered as a context manager")

        archs = list(archs)
        n_workers = len(self.processes)
        chunks = self._chunk(archs, n_workers)

        job_id = self.job_counter
        self.job_counter += 1

        for wid, chunk in enumerate(chunks):
            self.task_queues[wid].put({
                "job_id": job_id,
                "architectures": chunk,
            })

        ordered = self._gather(job_id, n_workers)
        return self._merge(ordered)

    # ------------------------------------------------------------------ teardown

    def _terminate_all(self):
        for p in self.processes:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass

    def close(self):
        if self._closed:
            return
        self._closed = True

        for q in self.task_queues:
            try:
                q.put(None)
            except Exception:
                pass

        for p in self.processes:
            try:
                p.join(timeout=10)
            except Exception:
                pass

        for p in self.processes:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
