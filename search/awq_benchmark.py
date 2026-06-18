"""Final benchmark with AWQ (NOT layer-swap — AWQ folds smoothing into layernorms,
so MP-AWQ must be quantized per-arch). For each HQQ-search pick arch, build the
AWQ-quantized model via get_quantized_model(method='awq', arch) (run_awq pileval
calib + apply_awq at the arch's per-layer bits), save, lm_eval (system vLLM), rm.

Run with the search venv python (uses quant.model); lm_eval runs as a system-python
subprocess. Build on --build-gpu, eval on --eval-gpus.
  /tmp/sv45/bin/python awq_benchmark.py --specs <candidates.json> --pool <name> \
      --build-gpu 2 --eval-gpus 3,4,5,6,7
"""
from __future__ import annotations
import argparse, json, os, glob, shutil, subprocess, sys, time
import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, "/NAS/SJ/actquant/search")
from quant.model import get_quantized_model

ANALYSE = "/NAS/SJ/actquant/poc/benchmark_proxy/analyse_metric"
LM_DIR = f"{ANALYSE}/results/lm_eval"
LM_EVAL = f"{ANALYSE}/lm_eval_vllm.py"
FP16 = "/SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct"
SYS_PY = "/opt/conda/bin/python"
ENV = dict(os.environ); ENV.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                                   HF_HUB_CACHE="/SSD/JSY", HF_ALLOW_CODE_EVAL="1", PYTHONUNBUFFERED="1")


def done(pool, name):
    return bool(glob.glob(f"{LM_DIR}/{pool}/{name}/*/results_*.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--build-gpu", default="2")
    ap.add_argument("--eval-gpus", default="3,4,5,6,7")
    ap.add_argument("--out-root", default="/root/awq_models")
    ap.add_argument("--tasks", default="gsm8k_cot,ifeval,mbpp")
    args = ap.parse_args()
    egpus = args.eval_gpus.split(",")
    specs = json.load(open(args.specs))
    names = [f"cand_{i:02d}_bits{s['bits']:.6f}_jsd{s['jsd']:.6f}" for i, s in enumerate(specs)]
    todo = [(n, s) for n, s in zip(names, specs) if not done(args.pool, n)]
    print(f"[{args.pool}] {len(todo)}/{len(specs)} picks to AWQ-benchmark", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.build_gpu  # build on this GPU
    cfg = AutoConfig.from_pretrained(FP16)
    tok = AutoTokenizer.from_pretrained(FP16)
    out_pool = f"{args.out_root}/{args.pool}"
    log_dir = f"{LM_DIR}/{args.pool}/_logs"; os.makedirs(log_dir, exist_ok=True)

    for bs in range(0, len(todo), len(egpus)):
        batch = todo[bs:bs + len(egpus)]
        built = []
        for (name, sp) in batch:
            d = f"{out_pool}/{name}"
            if not os.path.exists(f"{d}/model.safetensors.index.json"):
                t0 = time.time()
                model = get_quantized_model(method="awq", arch=sp["arch"]["linear"],
                                            model_name=FP16, device_map={"": 0},
                                            group_size=128, dtype=torch.bfloat16, config=cfg)
                model.save_pretrained(d, safe_serialization=True)
                tok.save_pretrained(d)
                del model; torch.cuda.empty_cache()
                print(f"  AWQ-built {name} ({time.time()-t0:.0f}s)", flush=True)
            built.append((name, d))
        procs = []
        for i, (name, d) in enumerate(built):
            g = egpus[i % len(egpus)]
            cmd = [SYS_PY, LM_EVAL, "--model", d, "--task", args.tasks,
                   "--output_path", f"{LM_DIR}/{args.pool}/{name}", "--log_samples", "--device", "cuda:0"]
            e = dict(ENV); e["CUDA_VISIBLE_DEVICES"] = g
            lf = open(f"{log_dir}/{name}.log", "w")
            print(f"  [GPU {g}] eval {name}", flush=True)
            procs.append((name, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=e), lf))
        for (name, p, lf) in procs:
            rc = p.wait(); lf.close()
            print(f"  done {name} rc={rc}" + ("" if rc == 0 else " *** FAILED"), flush=True)
        for (name, d) in built:
            shutil.rmtree(d, ignore_errors=True)
        print(f"  batch {bs//len(egpus)} done", flush=True)
    print(f"[{args.pool}] AWQ benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
