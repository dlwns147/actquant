"""Arch-parallel multi-GPU evaluation pool for expensive weight backends (AWQ).

Why not `accelerate launch --num_processes N` (the existing DP path)? DP shards the
CALIBRATION BATCHES of one arch across ranks — but with --w_method awq every rank
still runs the per-arch run_awq quantization (~7.7 min build, the bottleneck) on its
own copy, so the build is DUPLICATED N times and only the ~1.5 min JSD pass speeds
up. Arch-level parallelism (each worker owns whole archs) gives the full ~N x.

Design (mirrors tests/awq_alloc_flip/pilot.py per-arch loop):
  * N persistent spawn-context workers, each pinned to one GPU via
    CUDA_VISIBLE_DEVICES *before* any CUDA init, each holding its own
    LlamaEvaluator (loaders + dense teacher logits stay resident; only the
    per-arch AWQ model is built/dropped).
  * restart-safe: a worker that dies mid-task (run_awq's known inter-build
    memory leak) is respawned and the task requeued (<=3 tries -> nan);
    workers also self-recycle every `recycle_after` archs as leak defense.
  * map(archs) preserves input order; hard failures return nan (caller drops).
"""
import os
import multiprocessing as mp
from queue import Empty
from time import time

MAX_TRIES = 3


def _worker_main(gpu_id, wid, task_q, result_q, cfg, recycle_after):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)   # before any CUDA init
    # QUIET WORKERS: redirect this worker's OS-level stdout+stderr (fd 1/2) to a per-worker
    # log file BEFORE any import/CUDA init, so run_awq / tqdm / C-extension prints don't flood
    # the terminal — only the main process's [awq_pool] aggregate lines show. dup2 (not just
    # sys.stdout) is required to catch C-level and tqdm(stderr) output. Progress still reaches
    # main via result_q (unaffected). worker_log_dir=None keeps the old inline spew.
    log_dir = cfg.get('worker_log_dir')
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        _f = open(os.path.join(log_dir, f'worker{wid}_gpu{gpu_id}.log'), 'a', buffering=1)
        os.dup2(_f.fileno(), 1); os.dup2(_f.fileno(), 2)
    import json
    from evaluator import LlamaEvaluator
    from utils.func import init_accelerator, set_seed, process_dtype, clean_up
    set_seed(cfg['seed'])
    config = json.load(open(cfg['config']))[cfg['model_name']]
    accelerator, device_map = init_accelerator('0', config)
    evaluator = LlamaEvaluator(
        config, accelerator=accelerator, device_map=device_map,
        model_id=f"{cfg['model_path']}/{cfg['model_name']}",
        method=cfg['method'], quant_model_paths=cfg['quant_model_paths'],
        outlier=None, seqlen=cfg['seqlen'], n_sample=cfg['n_sample'],
        datasets=[cfg['dataset']], dtype=process_dtype(cfg['dtype']),
        bits=cfg['bits'], group_size=cfg['group_size'],
        residual_length=cfg['residual_length'], attn_sink=cfg['attn_sink'],
        k_quant_scheme=cfg['k_quant_scheme'], v_quant_scheme=cfg['v_quant_scheme'],
        loss_func=cfg['loss_func'], last_tokens=cfg['last_tokens'])
    result_q.put(('ready', wid, None, None))
    n_done = 0
    while True:
        task = task_q.get()
        if task is None:
            break
        i, arch = task
        result_q.put(('claim', wid, i, None))
        t0 = time()
        try:
            m, _ = evaluator.eval(accelerator=accelerator, arch=arch, metric='loss',
                                  loss_func=cfg['loss_func'], stride=cfg['stride'],
                                  prefill_prompt=cfg['prefill_prompt'])
            result_q.put(('ok', wid, i, (float(list(m.values())[0]), round(time() - t0, 1))))
        except Exception as e:                          # noqa: BLE001 — report, main decides
            result_q.put(('err', wid, i, repr(e)[:500]))
        finally:
            if evaluator.model is not None:
                del evaluator.model
                evaluator.model = None
            clean_up()
        n_done += 1
        if recycle_after and n_done >= recycle_after:
            result_q.put(('recycle', wid, None, None))
            return


class AWQEvalPool:
    def __init__(self, gpu_ids, cfg, recycle_after=32, log=print):
        self.cfg, self.recycle_after, self.log = cfg, recycle_after, log
        self.ctx = mp.get_context('spawn')
        self.task_q, self.result_q = self.ctx.Queue(), self.ctx.Queue()
        self.procs = {}        # wid -> (process, gpu_id)
        self.claims = {}       # wid -> task index currently being evaluated
        self._next_wid = 0
        for g in gpu_ids:
            self._spawn(g)
        self.log(f"[awq_pool] spawning {len(self.procs)} workers on GPUs {list(gpu_ids)} "
                 f"(each loads model + dense teacher logits once)")

    def _spawn(self, gpu):
        wid = self._next_wid; self._next_wid += 1
        p = self.ctx.Process(target=_worker_main, daemon=True,
                             args=(gpu, wid, self.task_q, self.result_q, self.cfg, self.recycle_after))
        p.start()
        self.procs[wid] = (p, gpu)

    def _fail_or_retry(self, i, archs, tries, out):
        """Returns 1 if the task terminally failed (pending decreases), else 0."""
        tries[i] += 1
        if tries[i] >= MAX_TRIES:
            out[i] = float('nan')
            self.log(f"[awq_pool] task {i} failed {tries[i]}x -> nan (dropped)")
            return 1
        self.task_q.put((i, archs[i]))
        return 0

    def map(self, archs):
        """Evaluate archs on the pool; returns losses in input order (nan = failed)."""
        n = len(archs)
        out = [None] * n
        tries = [0] * n
        for i, a in enumerate(archs):
            self.task_q.put((i, a))
        pending, t0 = n, time()
        while pending > 0:
            try:
                kind, wid, i, payload = self.result_q.get(timeout=30)
            except Empty:
                # liveness sweep: requeue claims of dead workers, respawn on same GPU
                for wid, (p, g) in list(self.procs.items()):
                    if not p.is_alive():
                        del self.procs[wid]
                        j = self.claims.pop(wid, None)
                        self.log(f"[awq_pool] worker {wid} (gpu {g}) died"
                                 + (f" on task {j}" if j is not None else "") + " -> respawn")
                        if j is not None and out[j] is None:
                            pending -= self._fail_or_retry(j, archs, tries, out)
                        self._spawn(g)
                continue
            if kind == 'ready':
                self.log(f"[awq_pool] worker {wid} ready")
            elif kind == 'claim':
                self.claims[wid] = i
            elif kind == 'ok':
                y, sec = payload
                out[i] = y; pending -= 1
                self.claims.pop(wid, None)
                self.log(f"[awq_pool] {n - pending}/{n} loss={y:.4f} ({sec:.0f}s, worker {wid}, "
                         f"elapsed {(time() - t0) / 60:.0f}m)")
            elif kind == 'err':
                self.claims.pop(wid, None)
                self.log(f"[awq_pool] task {i} error (try {tries[i] + 1}): {payload}")
                pending -= self._fail_or_retry(i, archs, tries, out)
            elif kind == 'recycle':
                p, g = self.procs.pop(wid)
                p.join(timeout=60)
                self._spawn(g)
        return out

    def close(self):
        for _ in self.procs:
            self.task_q.put(None)
        for p, _ in self.procs.values():
            p.join(timeout=120)
        self.log("[awq_pool] closed")
