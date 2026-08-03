import logging
import re
from functools import cache
from typing import TYPE_CHECKING, Union

# from transformers import AutoTokenizer

if TYPE_CHECKING:
    import transformers

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data import get_tokenizer

eval_logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENGTHS = [
    4096,
]


# @cache
# def get_tokenizer(
#     tokenizer=None, pretrained=None, **kwargs
# ) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
#     pretrained = tokenizer or pretrained
#     assert pretrained, "No tokenizer or pretrained provided."
#     eval_logger.info(f"Using tokenizer {pretrained} for synthetic tasks.")
#     return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [
            (sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
             if ref else 0.0)                       # guard empty ref -> no div-by-zero
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    # Official RULER semantics: per (pred, ref-list) score 1.0 if ANY reference
    # alias is a substring of pred (max over aliases), averaged over preds.
    # The previous form had max/sum swapped (max over preds, mean over aliases),
    # which under-scored multi-alias answers (e.g. SQuAD) — it computed the
    # fraction of aliases matched rather than "any alias matched".
    score = sum(
        [
            (max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
             if ref else 0.0)                       # guard empty ref -> no max([]) ValueError
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_all(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_part(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return -1
    return sum(res) / len(res)
