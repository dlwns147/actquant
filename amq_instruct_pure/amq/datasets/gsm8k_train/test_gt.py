"""Run ``evaluator.py`` over every entry of ``gt.jsonl`` and save a per-sample
comparison against the reference run.

For each row we record:
- our generation and the extracted final answer
- the gt extracted answer and gt exact_match score
- score_match: do we and gt agree on right/wrong?
- answer_identical: are the extracted final answers byte-identical?
- perfect_match: both score_match and answer_identical (strongest agreement)

At the end, print aggregate counts.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, "/NAS/SJ/actquant/poc/benchmark_datasets/gsm8k_train")

from evaluator import _strict_match, evaluate, generate  # noqa: E402

GT_PATH = "/NAS/SJ/actquant/poc/benchmark_datasets/gsm8k_train/gt.jsonl"
OUT_PATH = "/NAS/SJ/actquant/poc/benchmark_datasets/gsm8k_train/test_gt_results.jsonl"
SUMMARY_PATH = "/NAS/SJ/actquant/poc/benchmark_datasets/gsm8k_train/test_gt_summary.json"


def main() -> int:
    with open(GT_PATH) as f:
        entries = [json.loads(line) for line in f]
    total = len(entries)

    n_score_match = 0
    n_answer_identical = 0
    n_perfect = 0
    our_correct = 0
    gt_correct = 0

    t0 = time.time()
    with open(OUT_PATH, "w") as fout:
        for i, entry in enumerate(entries):
            prompt = entry["arguments"]["gen_args_0"]["arg_0"]
            response = generate(prompt)
            our_extracted = _strict_match(response)
            our_score = evaluate(response, entry["target"])

            gt_extracted = entry["filtered_resps"][0]
            gt_score = entry["exact_match"]

            score_match = our_score == gt_score
            answer_identical = our_extracted == gt_extracted
            perfect = score_match and answer_identical

            n_score_match += int(score_match)
            n_answer_identical += int(answer_identical)
            n_perfect += int(perfect)
            our_correct += int(our_score == 1.0)
            gt_correct += int(gt_score == 1.0)

            fout.write(json.dumps({
                "idx": i,
                "doc_id": entry["doc_id"],
                "target": entry["target"],
                "our_response": response,
                "our_extracted": our_extracted,
                "our_score": our_score,
                "gt_extracted": gt_extracted,
                "gt_score": gt_score,
                "score_match": score_match,
                "answer_identical": answer_identical,
                "perfect_match": perfect,
            }) + "\n")
            fout.flush()

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(
                    f"[{i + 1}/{total}] elapsed={elapsed:.0f}s "
                    f"rate={rate:.2f}/s eta={eta:.0f}s "
                    f"score_match={n_score_match} identical={n_answer_identical} perfect={n_perfect}",
                    flush=True,
                )

    summary = {
        "total": total,
        "score_match": n_score_match,
        "answer_identical": n_answer_identical,
        "perfect_match": n_perfect,
        "our_correct": our_correct,
        "gt_correct": gt_correct,
        "elapsed_sec": time.time() - t0,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"total: {total}")
    print(f"score_match (right/wrong agreement): {n_score_match}/{total}")
    print(f"answer_identical (byte-identical extracted answer): {n_answer_identical}/{total}")
    print(f"perfect_match (both): {n_perfect}/{total}")
    print(f"our_correct: {our_correct}  gt_correct: {gt_correct}")
    print(f"results -> {OUT_PATH}")
    print(f"summary -> {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
