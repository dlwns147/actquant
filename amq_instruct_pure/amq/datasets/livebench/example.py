"""Minimal usage example for ``evaluator.generate`` / ``evaluator.evaluate``."""

from evaluator import _load_gt_and_kwargs, evaluate, generate
import json
from evaluator import generate, evaluate

def test1():
    # Load one ground-truth row (includes prompt, instruction_id_list, kwargs)
    with open("/NAS/SJ/actquant/poc/benchmark_datasets/livebench/gt.jsonl") as f:
        row = json.loads(f.readline())

    # kwargs live on the HF dataset, not in gt.jsonl — pull them in once:
    from datasets import load_dataset
    ds = load_dataset("livebench/instruction_following", split="test")
    kwargs_by_id = {q["question_id"]: q["kwargs"] for q in ds}
    row["kwargs"] = kwargs_by_id[row["question_id"]]

    # 1) generate
    response = generate(row["prompt"])
    print(response[:200])

    # 2) evaluate
    score = evaluate(response, reference=row)
    print(f"IF score: {score}")


def test2():
    rows = _load_gt_and_kwargs()
    row = rows[0]

    response = generate(row["prompt"])
    score = evaluate(response, row)

    print(f"question_id:  {row['question_id']}")
    print(f"instructions: {row['instruction_id_list']}")
    print(f"--- response (first 300 chars) ---")
    print(response[:300])
    print(f"--- IF score ---")
    print(score)
