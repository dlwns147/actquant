import json
import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.loss import JSD


# Location of split_meta.json files created by lm_eval_vllm/split_dataset.py.
# Used to translate task names like "ifeval_test" / "mbpp_train" / etc into
# a real lm_eval task name + a doc-id filter passed via --samples.
_SPLIT_DATASETS_ROOT = Path(
    "/NAS/SJ/actquant/poc/benchmark_proxy/lm_eval_vllm/datasets"
)
# Only ``_test`` is treated as a split suffix here. ``_train`` is reserved
# for tasks that have their own lm_eval task name (e.g. ``gsm8k_cot_train``)
# and should be passed through unchanged.
_SPLIT_SUFFIXES = ("_test",)

# Tasks that execute model-generated code at scoring time. These need
# ``HF_ALLOW_CODE_EVAL=1`` in the env and ``--confirm_run_unsafe_code`` on the
# command line. Match by the base name (after stripping any ``_test`` suffix).
_CODE_EVAL_TASKS = frozenset({"mbpp", "humaneval"})


def _resolve_split_task(name: str):
    """If ``name`` ends with ``_test`` and a matching
    ``datasets/<base>/split_meta.json`` exists, return
    ``(base_task_name, [test_doc_ids])``. Otherwise return ``(name, None)``.

    mmlu_pro_<subject>_test falls back to
    ``datasets/mmlu_pro/<subject>/split_meta.json``.
    """
    for suffix in _SPLIT_SUFFIXES:
        if not name.endswith(suffix):
            continue
        base = name[: -len(suffix)]
        split_key = f"{suffix.lstrip('_')}_ids"  # "test_ids"

        candidates = [_SPLIT_DATASETS_ROOT / base / "split_meta.json"]
        if base.startswith("mmlu_pro_"):
            candidates.append(_SPLIT_DATASETS_ROOT / "mmlu_pro" / base / "split_meta.json")

        for meta_path in candidates:
            if meta_path.exists():
                with meta_path.open() as f:
                    meta = json.load(f)
                if split_key not in meta:
                    raise KeyError(
                        f"{meta_path} has no {split_key!r} (keys: {list(meta.keys())})"
                    )
                ids = [int(x) for x in meta[split_key]]
                if not ids:
                    raise ValueError(f"{meta_path}: empty {split_key!r}")
                return base, ids

        # Suffix matched but no split_meta.json — leave name as-is and let
        # lm_eval raise its own error if the task is unknown.
        return name, None

    return name, None

@torch.inference_mode()
def get_logits(model, loader):
    logits = []
    for inputs in tqdm(loader, desc='Get Logits'):
        outputs = model(inputs)
        lm_logits = outputs.logits
        logits.append(lm_logits)

    dense_logits_list = torch.cat(logits, dim=0).detach()

    return dense_logits_list


@torch.inference_mode()
def eval_loss(model, accelerator, loader, dense_logits_list, seqlen=2048):
    losses = []
    
    for i, inputs in enumerate(loader):
        # Forward pass through the model
        outputs = model(inputs)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].reshape(-1, lm_logits.size(-1)).contiguous()
        
        # Compute loss
        dense_logits = dense_logits_list[i]
        dense_logits = dense_logits[:-1, :].reshape(-1, lm_logits.size(-1)).contiguous()
        loss_fct = JSD()
        loss = loss_fct(shift_logits, dense_logits)

        # Calculate negative log likelihood
        loss = loss.float() * seqlen * lm_logits.shape[0]
        losses.append(loss)

    losses = torch.stack(accelerator.gather_for_metrics(losses)).flatten()
    loss_sum = losses.sum() / (len(losses) * seqlen)

    return loss_sum.item()


@torch.inference_mode()
def eval_ppl(model, accelerator, loader, seqlen=2048):
    ppls = []

    for i, inputs in enumerate(loader):
        outputs = model(inputs)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        shift_labels = inputs[:, 1:].reshape(-1)

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * lm_logits.shape[0]

        # Append to list of negative log likelihoods
        ppls.append(neg_log_likelihood)

    ppls = torch.stack(accelerator.gather_for_metrics(ppls)).flatten()
    ppl = torch.exp(ppls.sum() / (len(ppls) * seqlen))
    
    return ppl.item()

@torch.inference_mode()
def evel_lm_eval(model, task=['gsm8k_cot', 'if_eval'], batch_size='auto', device='cuda',
                output_path=None, log_samples=False):
    """Run lm_eval. Task names with a ``_train`` or ``_test`` suffix are
    translated into the base task plus a ``--samples '{base: [doc_ids]}'``
    filter loaded from ``lm_eval_vllm/datasets/<base>/split_meta.json``.

    Example: ``task=["ifeval_test", "mbpp_test"]`` becomes
    ``--tasks ifeval,mbpp --samples '{"ifeval":[<test_ids>], "mbpp":[<test_ids>]}'``.
    Plain task names (no suffix or unmatched split_meta) are passed through
    unchanged.
    """
    real_tasks = []
    samples_dict = {}
    for t in task:
        base, ids = _resolve_split_task(t)
        real_tasks.append(base)
        if ids is not None:
            samples_dict[base] = ids

    needs_code_eval = any(t in _CODE_EVAL_TASKS for t in real_tasks)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={model}",
        "--tasks", ",".join(real_tasks),
        "--batch_size", batch_size,
        "--device", device,
    ]
    # "--apply_chat_template",

    if samples_dict:
        cmd.extend(["--samples", json.dumps(samples_dict)])
        print(f"[lm_eval] split filter applied: " +
              ", ".join(f"{k}={len(v)} docs" for k, v in samples_dict.items()))

    if output_path:
        cmd.extend(["--output_path", output_path])
    if log_samples:
        cmd.append("--log_samples")

    # Always append --confirm_run_unsafe_code (harmless for tasks that don't
    # execute code, required for mbpp / humaneval).
    cmd.append("--confirm_run_unsafe_code")

    env = os.environ.copy()
    if needs_code_eval:
        env["HF_ALLOW_CODE_EVAL"] = "1"
        code_tasks = [t for t in real_tasks if t in _CODE_EVAL_TASKS]
        print(f"[lm_eval] code-eval tasks present ({code_tasks}); "
              f"HF_ALLOW_CODE_EVAL=1")

    subprocess.run(cmd, check=True, env=env)