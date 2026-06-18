"""GPU-pool AWQ benchmark. One worker per GPU; each worker pulls candidates from a
shared queue and runs build(subprocess, venv) -> eval(subprocess, sys-vLLM) -> rm
sequentially on its own GPU. Fresh build subprocess => no inter-build memory leak;
per-GPU serial => no contention. ~ceil(N/ngpu) * (build+eval) wall-clock.

  /opt/conda/bin/python awq_pool_bench.py --specs cands.json --pool NAME --gpus 0,1,2,3,4,5,6,7
"""
from __future__ import annotations
import argparse, glob, json, os, queue, shutil, subprocess, threading, time

ANALYSE = "/NAS/SJ/actquant/poc/benchmark_proxy/analyse_metric"
LM_DIR = f"{ANALYSE}/results/lm_eval"
LM_EVAL = f"{ANALYSE}/lm_eval_vllm.py"
SEARCH = "/NAS/SJ/actquant/search"
VENV_PY = "/tmp/sv45/bin/python"
SYS_PY = "/opt/conda/bin/python"
BASE_ENV = dict(os.environ)
BASE_ENV.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_CACHE="/SSD/JSY",
                HF_ALLOW_CODE_EVAL="1", PYTHONUNBUFFERED="1",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
_print = threading.Lock()


def log(m):
    with _print:
        print(m, flush=True)


def done(pool, name):
    return bool(glob.glob(f"{LM_DIR}/{pool}/{name}/*/results_*.json"))


def worker(gpu, q, specs_path, pool, out_root, tasks, log_dir):
    env = dict(BASE_ENV); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    while True:
        try:
            idx, name = q.get_nowait()
        except queue.Empty:
            return
        d = f"{out_root}/{name}"
        try:
            t0 = time.time()
            # 1) build (fresh venv subprocess => memory reclaimed on exit)
            blog = open(f"{log_dir}/{name}.build.log", "w")
            rc = subprocess.call([VENV_PY, f"{SEARCH}/awq_build_one.py",
                                  "--specs", specs_path, "--idx", str(idx), "--out", d],
                                 stdout=blog, stderr=subprocess.STDOUT, env=env, cwd=SEARCH)
            blog.close()
            if rc != 0 or not os.path.exists(f"{d}/model.safetensors.index.json"):
                log(f"[GPU{gpu}] BUILD FAILED {name} rc={rc}"); q.task_done(); continue
            # 2) eval (system vLLM subprocess, same GPU)
            elog = open(f"{log_dir}/{name}.eval.log", "w")
            rc = subprocess.call([SYS_PY, LM_EVAL, "--model", d, "--task", tasks,
                                  "--output_path", f"{LM_DIR}/{pool}/{name}",
                                  "--log_samples", "--device", "cuda:0"],
                                 stdout=elog, stderr=subprocess.STDOUT, env=env)
            elog.close()
            ok = done(pool, name)
            log(f"[GPU{gpu}] {'OK' if ok else 'EVAL FAILED'} {name} rc={rc} ({time.time()-t0:.0f}s)")
        finally:
            shutil.rmtree(d, ignore_errors=True)
            q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out-root", default="/root/awq_models")
    ap.add_argument("--tasks", default="gsm8k_cot,ifeval,mbpp")
    args = ap.parse_args()
    gpus = args.gpus.split(",")
    specs = json.load(open(args.specs))
    names = [f"cand_{i:02d}_bits{s['bits']:.6f}_jsd{s['jsd']:.6f}" for i, s in enumerate(specs)]
    todo = [(i, n) for i, n in enumerate(names) if not done(args.pool, n)]
    log(f"[{args.pool}] {len(todo)}/{len(specs)} picks to AWQ-benchmark on GPUs {gpus}")
    out_pool = f"{args.out_root}/{args.pool}"; os.makedirs(out_pool, exist_ok=True)
    log_dir = f"{LM_DIR}/{args.pool}/_logs"; os.makedirs(log_dir, exist_ok=True)
    q = queue.Queue()
    for item in todo:
        q.put(item)
    ths = [threading.Thread(target=worker, args=(g, q, args.specs, args.pool, out_pool,
                                                 args.tasks, log_dir)) for g in gpus]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    ndone = sum(done(args.pool, n) for n in names)
    log(f"[{args.pool}] AWQ benchmark complete. {ndone}/{len(specs)} have results.")


if __name__ == "__main__":
    main()
