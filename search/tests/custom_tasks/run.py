#!/usr/bin/env python
"""
Standalone runner for the `ifeval_pp` / `apps` custom tasks.

Self-contained: depends only on `lm_eval` + `transformers` + `datasets`.
No actquant / search code is imported. The custom task yamls in this directory
are discovered via `TaskManager(include_path=...)` and evaluated with
`lm_eval.evaluator.simple_evaluate` on a HF model.

Usage (from anywhere):

    cd /NAS/SJ/actquant/search/tests/custom_tasks

    # quick smoke: 2 GPUs, 10 docs/task
    CUDA_VISIBLE_DEVICES=2,3 python run.py --limit 10

    # full run
    CUDA_VISIBLE_DEVICES=2,3 python run.py

    # single task / explicit batch size / no chat template
    python run.py --tasks ifeval_pp --batch-size 8
    python run.py --no-chat-template

`--chat-template` is ON by default: Llama-3.1-8B-Instruct is instruction-tuned
and both tasks are instruction-following / code-generation, so the chat
template must be applied or the scores are meaningless.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-path",
        default="/SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct",
    )
    p.add_argument("--tasks", default="ifeval_pp,apps",
                   help="comma-separated task names")
    p.add_argument("--include-path", default=THIS_DIR,
                   help="dir scanned by lm-eval TaskManager for the custom yamls")
    p.add_argument("--limit", type=int, default=None,
                   help="cap docs per task (smoke testing)")
    p.add_argument("--batch-size", default="auto",
                   help='int or "auto" (default: auto)')
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--chat-template", dest="chat_template",
                   action="store_true", default=True,
                   help="apply the model chat template (default: on)")
    p.add_argument("--no-chat-template", dest="chat_template",
                   action="store_false")
    p.add_argument(
        "--output",
        default=os.path.join(THIS_DIR, "results", "run_results.json"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    batch_size = (
        args.batch_size if args.batch_size == "auto" else int(args.batch_size)
    )

    task_manager = TaskManager(include_path=args.include_path)
    missing = [t for t in tasks if t not in task_manager.all_tasks]
    if missing:
        raise SystemExit(
            f"tasks not found under {args.include_path}: {missing}"
        )

    print(f"[run] visible GPUs : {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"[run] model        : {args.model_path} ({args.dtype})")
    print(f"[run] tasks        : {tasks}")
    print(f"[run] limit={args.limit} batch_size={batch_size} "
          f"chat_template={args.chat_template}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",  # spreads across the visible GPUs
        trust_remote_code=True,
    )
    model.eval()

    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size if batch_size is not None else 1,
    )

    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=batch_size,
        limit=args.limit,
        task_manager=task_manager,
        apply_chat_template=args.chat_template,
        fewshot_as_multiturn=args.chat_template,
    )

    res = results["results"]
    print("\n========== RESULTS ==========")
    print(json.dumps(res, indent=2, default=str))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\n[run] saved -> {args.output}")


if __name__ == "__main__":
    main()
