"""APPS (codeparrot/apps) task helpers: prompt construction, code extraction
and functional-correctness scoring via testing_util.run_test."""

import json
import os
import re
import sys

# lm-eval's `!function` loader execs this file via spec_from_file_location
# WITHOUT putting its directory on sys.path, so the sibling `testing_util`
# is not importable by default. Add this file's own dir to sys.path so the
# import works both there and when imported as the `custom_tasks.apps` package.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from testing_util import run_test, DEFAULT_TIMEOUT  # noqa: E402


# -----------------------------------------------------------------------------
# Prompt construction (official APPS format, see hendrycks/apps generate_*.py)
# -----------------------------------------------------------------------------
def doc_to_text(doc) -> str:
    starter = doc.get("starter_code") or ""
    try:
        fn_name = bool(json.loads(doc.get("input_output") or "{}").get("fn_name"))
    except Exception:  # noqa: BLE001
        fn_name = False
    call_format = "\nUse Call-Based format" if fn_name else "\nUse Standard Input format"

    prompt = (
        "Solve the following programming problem. Respond with a single, "
        "complete Python solution inside one ```python code block and nothing "
        "else.\n\n"
        "QUESTION:\n"
        f"{doc['question']}\n"
    )
    if starter:
        prompt += f"{starter}\n"
    prompt += call_format + "\nANSWER:\n"
    return prompt


def doc_to_target(doc) -> str:
    # Reference solution is not used for scoring (we run test cases); kept for
    # logging / few-shot compatibility.
    try:
        sols = json.loads(doc.get("solutions") or "[]")
        return sols[0] if sols else ""
    except Exception:  # noqa: BLE001
        return ""


# -----------------------------------------------------------------------------
# Code extraction from the model's free-form generation
# -----------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    blocks = _FENCE.findall(text)
    if blocks:
        # last fenced block is usually the final answer
        return blocks[-1].strip()
    # fall back to text after the last "ANSWER:" marker
    if "ANSWER:" in text:
        text = text.split("ANSWER:")[-1]
    return text.strip()


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def process_results(doc, results):
    generation = results[0]
    code = extract_code(generation)

    sample = {"input_output": doc.get("input_output", "")}
    try:
        rc = run_test(sample, code, timeout=DEFAULT_TIMEOUT)
    except Exception:  # noqa: BLE001 - never let a single problem crash the run
        rc = []

    if not rc:
        passed_all = 0.0
        case_rate = 0.0
    else:
        passed_all = 1.0 if all(r == 1 for r in rc) else 0.0
        case_rate = sum(1 for r in rc if r == 1) / len(rc)

    return {"strict_acc": passed_all, "test_case_acc": case_rate}


# -----------------------------------------------------------------------------
# Difficulty filters (used by the apps_<difficulty> sub-tasks)
# -----------------------------------------------------------------------------
def _filter(dataset, level):
    return dataset.filter(lambda d: d.get("difficulty") == level)


def filter_introductory(dataset):
    return _filter(dataset, "introductory")


def filter_interview(dataset):
    return _filter(dataset, "interview")


def filter_competition(dataset):
    return _filter(dataset, "competition")
