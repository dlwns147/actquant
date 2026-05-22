"""HQQ-proxy generation evaluation via vLLM.

Wraps ``BenchmarkProxyEvaluatorVLLM(mode='generate')`` as a context
manager so the long-lived vLLM engine and the on-CPU fp16 proxies are
torn down before AWQ runs or before ``lm_eval`` spawns its own vLLM
subprocess.
"""

from __future__ import annotations

import numpy as np
import torch

from utils.func import clean_up
from evaluation.evaluator_vllm import BenchmarkProxyEvaluatorVLLM


class GenPerfEvaluator:
    """Context manager around the vLLM gen-eval engine."""

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.evaluator = None

    def __enter__(self):
        a = self.args
        self.evaluator = BenchmarkProxyEvaluatorVLLM(
            config=self.config,
            gpu_id=a.gpu_id,
            model_id=f'{a.model_path}/{a.model_name}',
            quantization_proxy_paths=a.quantization_proxy_paths,
            quantization_proxy_fake_paths=a.quantization_proxy_fake_paths,
            bits_range=a.bits_range,
            group_size=a.group_size,
            datasets=a.datasets,
            n_sample=a.n_sample,
            batch_size=a.batch_size,
            seed=a.seed,
            mode='generate',
            gpu_memory_utilization=a.gpu_memory_utilization,
            max_model_len=a.max_model_len,
            tensor_parallel_size=a.tensor_parallel_size,
        )
        return self

    def evaluate(self, archs):
        metric_list, bits_usage_list = self.evaluator.eval_generate(
            list(archs), max_value=None,
        )
        return metric_list, bits_usage_list

    def __exit__(self, exc_type, exc, tb):
        ev = self.evaluator
        if ev is not None:
            for attr in ('llm', 'model', 'quantization_proxies', 'fp16_model'):
                if hasattr(ev, attr):
                    try:
                        delattr(ev, attr)
                    except Exception:
                        pass
            self.evaluator = None
        clean_up()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def mean_perf(metric_list, datasets):
    """Average per-dataset gen-perf into a single mean per arch."""
    keys = [f"{ds}_gen" for ds in datasets]
    arrays = [np.asarray(metric_list[k], dtype=float) for k in keys]
    return np.mean(np.stack(arrays, axis=0), axis=0)
