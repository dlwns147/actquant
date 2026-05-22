"""LiveBench instruction-following evaluator (HF Transformers backend).

Mirrors the interface of ``gsm8k_train/evaluator.py``:

- Deterministic generation (``do_sample=False``) via Hugging Face Transformers.
- Evaluation reuses LiveBench's ``instruction_following_eval`` strict scorer
  (see ``LiveBench/livebench/if_runner/instruction_following_eval``) so the
  per-instruction follow flags match the original pipeline exactly.
- Per-example score follows ``process_results/instruction_following/utils.py``
  (``(follow_all + mean(follow_list)) / 2``).
"""

from __future__ import annotations

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_LIVEBENCH_IF_PATH = "/NAS/SJ/actquant/poc/benchmark_proxy/LiveBench/livebench/if_runner"
if _LIVEBENCH_IF_PATH not in sys.path:
    sys.path.insert(0, _LIVEBENCH_IF_PATH)

from instruction_following_eval import instructions_registry  # noqa: E402

_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
_MAX_NEW_TOKENS = 4096
# CUDA_VISIBLE_DEVICES="1" remaps physical GPU 1 to logical cuda:0.
_DEVICE = "cuda:0"

_GT_PATH = "/NAS/SJ/actquant/poc/benchmark_datasets/livebench/gt.jsonl"
_RESULTS_PATH = "/NAS/SJ/actquant/poc/benchmark_datasets/livebench/results.jsonl"
_HF_DATASET = "livebench/instruction_following"

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForCausalLM.from_pretrained(
    _MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
_model.eval()


def generate(prompt: str) -> str:
    """Greedy completion for ``prompt`` on GPU 1 via HF Transformers."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_DEVICE)
    with torch.no_grad():
        output_ids = _model.generate(
            input_ids=input_ids,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    return _tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    )


def _score_results(follow_all_instructions: bool, follow_instruction_list: list) -> float:
    score_1 = 1.0 if follow_all_instructions else 0.0
    score_2 = sum(1 for f in follow_instruction_list if f) / len(follow_instruction_list)
    return (score_1 + score_2) / 2


def _check_instructions(prompt: str, instruction_id_list, kwargs_list, response: str):
    follow_list = []
    for idx, instruction_id in enumerate(instruction_id_list):
        cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = cls(instruction_id)
        kw = {k: v for k, v in kwargs_list[idx].items() if v is not None}
        instruction.build_description(**kw)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        follow_list.append(bool(response.strip()) and instruction.check_following(response))
    return all(follow_list), follow_list


def evaluate(response: str, reference: dict) -> float:
    """Score ``response`` against a LiveBench ground-truth record.

    ``reference`` must carry ``prompt``, ``instruction_id_list`` and ``kwargs``
    (as supplied by ``livebench/instruction_following`` on HF). Returns the
    LiveBench IF score in ``[0, 1]``.
    """
    follow_all, follow_list = _check_instructions(
        reference["prompt"],
        reference["instruction_id_list"],
        reference["kwargs"],
        response,
    )
    return _score_results(follow_all, follow_list)


def _load_gt_and_kwargs():
    with open(_GT_PATH) as f:
        gt = [json.loads(line) for line in f]
    ds = load_dataset(_HF_DATASET, split="test")
    kwargs_by_id = {q["question_id"]: q["kwargs"] for q in ds}
    for row in gt:
        row["kwargs"] = kwargs_by_id[row["question_id"]]
    return gt


def _run_test():
    gt = _load_gt_and_kwargs()
    total = len(gt)
    mismatches = []
    with open(_RESULTS_PATH, "w") as out:
        for i, row in enumerate(gt):
            response = generate(row["prompt"])
            follow_all, follow_list = _check_instructions(
                row["prompt"], row["instruction_id_list"], row["kwargs"], response
            )
            record = {
                "follow_all_instructions": follow_all,
                "follow_instruction_list": follow_list,
                "instruction_id_list": row["instruction_id_list"],
                "prompt": row["prompt"],
                "question_id": row["question_id"],
                "response": response,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            ok = (
                follow_all == row["follow_all_instructions"]
                and follow_list == row["follow_instruction_list"]
            )
            if not ok:
                mismatches.append(i)
            print(
                f"[{i + 1}/{total}] qid={row['question_id'][:12]} "
                f"follow_all={follow_all} match={ok}"
            )

    matches = total - len(mismatches)
    print()
    print(f"results written to: {_RESULTS_PATH}")
    print(f"total examples: {total}")
    print(f"exact matches:  {matches}")
    print(f"mismatches:     {len(mismatches)}")
    if mismatches:
        print(f"mismatched indices: {mismatches}")
    print(f"IDENTICAL = {len(mismatches) == 0}")


if __name__ == "__main__":
    _run_test()
