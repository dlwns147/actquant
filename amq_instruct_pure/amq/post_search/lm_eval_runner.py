"""Wrapper around the typo'd ``utils.eval.evel_lm_eval``.

Exposes a ``run_lm_eval`` alias so callers don't have to import the
misspelled name.
"""

from __future__ import annotations

from utils.eval import evel_lm_eval


def run_lm_eval(model_path, task='gsm8k_cot', batch_size='auto',
                device='cuda', output_path=None, log_samples=False):
    return evel_lm_eval(
        model=model_path,
        task=task,
        batch_size=batch_size,
        device=device,
        output_path=output_path,
        log_samples=log_samples,
    )