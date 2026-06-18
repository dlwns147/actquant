"""Build ONE per-arch AWQ fake-quant model in a fresh process (so all GPU memory
is reclaimed on exit -- avoids the run_awq inter-build leak that OOM'd the
in-process loop). Pin the GPU via CUDA_VISIBLE_DEVICES before launching.

  CUDA_VISIBLE_DEVICES=2 /tmp/sv45/bin/python awq_build_one.py \
      --specs cands.json --idx 3 --out /root/awq_models/pool/cand_03
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, "/NAS/SJ/actquant/search")
from quant.model import get_quantized_model

FP16 = "/SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(f"{args.out}/model.safetensors.index.json"):
        print(f"[build] {args.out} exists, skip", flush=True); return
    sp = json.load(open(args.specs))[args.idx]
    cfg = AutoConfig.from_pretrained(FP16)
    tok = AutoTokenizer.from_pretrained(FP16)
    t0 = time.time()
    model = get_quantized_model(method="awq", arch=sp["arch"]["linear"],
                                model_name=FP16, device_map={"": 0},
                                group_size=128, dtype=torch.bfloat16, config=cfg)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"[build] idx={args.idx} bits={sp['bits']:.4f} -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
