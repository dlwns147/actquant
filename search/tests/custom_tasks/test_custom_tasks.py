"""
Tests for the ifeval_pp and apps custom tasks.

Run everything (network needed for the dataset-backed tests):

    cd /NAS/SJ/actquant/search/tests/custom_tasks
    python test_custom_tasks.py
    # or:  pytest -q test_custom_tasks.py

Offline-only subset (no HF downloads, no model):

    python test_custom_tasks.py --offline

The model-backed end-to-end check is opt-in (slow, loads Llama 3.1 8B):

    CUDA_VISIBLE_DEVICES=2,3 RUN_MODEL=1 python test_custom_tasks.py
"""

import os
import sys

# custom_tasks/ lives under /NAS/SJ/actquant/search/tests ; put that parent on
# the path so `custom_tasks` is importable as a package.
PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_PARENT not in sys.path:
    sys.path.insert(0, PKG_PARENT)

from custom_tasks.apps import testing_util
from custom_tasks.apps import utils as apps_utils


INCLUDE_PATH = os.path.join(PKG_PARENT, "custom_tasks")


# --------------------------------------------------------------------------- #
# 1. Offline unit tests: APPS execution harness                               #
# --------------------------------------------------------------------------- #
def test_apps_stdin_correct():
    sample = {
        "input_output": '{"inputs": ["2 3\\n", "10 5\\n"], '
        '"outputs": ["5\\n", "15\\n"]}'
    }
    code = "a, b = map(int, input().split())\nprint(a + b)"
    rc = testing_util.run_test(sample, code, timeout=10)
    assert rc == [1, 1], rc


def test_apps_stdin_wrong_answer():
    sample = {'input_output': '{"inputs": ["2 3\\n"], "outputs": ["5\\n"]}'}
    code = "a, b = map(int, input().split())\nprint(a - b)"
    rc = testing_util.run_test(sample, code, timeout=10)
    assert rc == [0], rc


def test_apps_syntax_error():
    sample = {'input_output': '{"inputs": ["1\\n"], "outputs": ["1\\n"]}'}
    rc = testing_util.run_test(sample, "def f(:\n  pass", timeout=10)
    assert rc == [-2], rc


def test_apps_runtime_error():
    sample = {'input_output': '{"inputs": ["1\\n"], "outputs": ["1\\n"]}'}
    rc = testing_util.run_test(sample, "raise ValueError('boom')", timeout=10)
    assert rc == [-1], rc


def test_apps_timeout():
    sample = {'input_output': '{"inputs": ["1\\n"], "outputs": ["1\\n"]}'}
    rc = testing_util.run_test(sample, "while True:\n    pass", timeout=3)
    assert rc == [-3], rc


def test_apps_call_based():
    sample = {
        "input_output": '{"fn_name": "add", "inputs": [[2, 3], [10, 5]], '
        '"outputs": [5, 15]}'
    }
    code = "def add(a, b):\n    return a + b"
    rc = testing_util.run_test(sample, code, timeout=10)
    assert rc == [1, 1], rc


def test_extract_code_fenced():
    text = "Sure!\n```python\nprint(1)\n```\nDone."
    assert apps_utils.extract_code(text) == "print(1)"


def test_extract_code_answer_marker():
    text = "ANSWER:\nprint(2)"
    assert apps_utils.extract_code(text).strip() == "print(2)"


def test_apps_process_results_keys():
    doc = {"input_output": '{"inputs": ["1\\n"], "outputs": ["1\\n"]}'}
    out = apps_utils.process_results(doc, ["```python\nprint(int(input()))\n```"])
    assert set(out) == {"strict_acc", "test_case_acc"}
    assert out["strict_acc"] == 1.0 and out["test_case_acc"] == 1.0


# --------------------------------------------------------------------------- #
# 2. Task-registration tests (no network)                                     #
# --------------------------------------------------------------------------- #
def test_tasks_registered():
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=INCLUDE_PATH)
    for name in ("ifeval_pp", "apps", "apps_introductory",
                 "apps_interview", "apps_competition"):
        assert name in tm.all_tasks, f"{name} not registered"


# --------------------------------------------------------------------------- #
# 3. Dataset-backed tests (need network / HF cache)                           #
# --------------------------------------------------------------------------- #
def test_ifeval_pp_end_to_end():
    from lm_eval.tasks import TaskManager, get_task_dict

    tm = TaskManager(include_path=INCLUDE_PATH)
    task = get_task_dict(["ifeval_pp"], tm)["ifeval_pp"]
    docs = list(task.test_docs())
    assert len(docs) > 5000
    doc = docs[0]
    # prompt is fed verbatim
    assert task.doc_to_text(doc) == doc["prompt"]
    # a response that follows zero instructions still scores cleanly
    res = task.process_results(doc, ["I will not comply with anything."])
    assert set(res) == {
        "prompt_level_strict_acc",
        "inst_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_loose_acc",
    }
    assert isinstance(res["inst_level_strict_acc"], list)


def test_apps_dataset_with_reference_solution():
    """A canonical APPS solution must pass all of its own test cases."""
    from datasets import load_dataset
    import json

    ds = load_dataset(
        "codeparrot/apps", "all", split="test",
        trust_remote_code=True, streaming=True,
    )
    checked = 0
    for d in ds:
        if not d.get("input_output") or not d.get("solutions"):
            continue
        sols = json.loads(d["solutions"])
        if not sols:
            continue
        rc = testing_util.run_test({"input_output": d["input_output"]},
                                   sols[0], timeout=15)
        if rc:  # found a problem with usable test cases
            assert all(r == 1 for r in rc), (
                f"reference solution failed problem {d['problem_id']}: {rc}"
            )
            checked += 1
        if checked >= 2:
            break
    assert checked >= 1, "no checkable APPS problem found"


# --------------------------------------------------------------------------- #
# 4. Optional model-backed end-to-end (slow, opt-in)                           #
# --------------------------------------------------------------------------- #
def test_model_smoke():
    if os.environ.get("RUN_MODEL") != "1":
        print("  [skip] test_model_smoke (set RUN_MODEL=1 to enable)")
        return
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=INCLUDE_PATH)
    model_path = os.environ.get(
        "MODEL_PATH", "/SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct"
    )
    res = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=["ifeval_pp", "apps"],
        limit=2,
        apply_chat_template=True,
        task_manager=tm,
    )
    assert "ifeval_pp" in res["results"]
    assert "apps" in res["results"]
    print("  model smoke results:", res["results"])


# --------------------------------------------------------------------------- #
# Standalone runner                                                           #
# --------------------------------------------------------------------------- #
def _run(offline: bool):
    offline_tests = [
        test_apps_stdin_correct,
        test_apps_stdin_wrong_answer,
        test_apps_syntax_error,
        test_apps_runtime_error,
        test_apps_timeout,
        test_apps_call_based,
        test_extract_code_fenced,
        test_extract_code_answer_marker,
        test_apps_process_results_keys,
        test_tasks_registered,
    ]
    online_tests = [
        test_ifeval_pp_end_to_end,
        test_apps_dataset_with_reference_solution,
        test_model_smoke,
    ]
    tests = offline_tests + ([] if offline else online_tests)
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback

            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run("--offline" in sys.argv) else 0)
