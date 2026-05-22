"""vLLM-backed benchmark-proxy evaluator.

Same public shape as ``BenchmarkProxyEvaluator`` (``__init__`` / ``sample`` /
``eval``), but generation runs through vLLM with continuous batching and
JSD scoring batches the HF forward pass by ``batch_size``.

HF remains the source of truth for the candidate model: ``self.model`` is
the fake-quantized HF module that ``sample(arch)`` mutates in place.
After each ``sample(...)`` call, the new weights are pushed into a
long-lived ``vllm.LLM`` engine via ``load_weights`` and used purely as a
fast generation runtime. JSD mode is unchanged from the HF evaluator
except for the batched forward pass.
"""

from __future__ import annotations

import re
import sys
import math
from copy import deepcopy
from statistics import fmean
from tqdm import tqdm

import warnings
warnings.simplefilter("ignore")

import torch

from utils.func import (
    init_accelerator,
    get_hfmodel,
    get_quantization_proxy,
    get_bits_usage,
    clean_up,
    getsubattr,
    setsubattr,
    getblock,
)
from utils.data import get_tokenizer
from utils.loss import JSD

from evaluation.evaluator_stratified import (
    _SampleLoader,
)

_LIVEBENCH_IF_SRC = "/NAS/SJ/actquant/poc/benchmark_proxy/LiveBench/livebench/if_runner"

_GSM8K_STOP_SEQUENCES = ("Q:", "</s>", "<|im_end|>")
_GSM8K_MAX_NEW_TOKENS = 256
# _LIVEBENCH_MAX_NEW_TOKENS = 4096
_LIVEBENCH_MAX_NEW_TOKENS = 1024

_ANSWER_REGEX = re.compile(r"The answer is (\-?[0-9\.\,]+).")
_REGEXES_TO_IGNORE = (
    re.compile(r","),
    re.compile(r"\$"),
    re.compile(r"(?s).*#### "),
    re.compile(r"\.$"),
)

_MONOTONIC       = ("000", "100", "110", "111")
_NON_MONOTONIC   = ("001", "010", "011", "101")
_BALANCED_GROUPS = ("000", "100", "110", "111", "non_monotonic")


# ---------------------------------------------------------------------------
# vLLM helpers

def _stop_sequences(dataset):
    return list(_GSM8K_STOP_SEQUENCES) if dataset == "gsm8k" else []


def _max_new_tokens(dataset):
    return _GSM8K_MAX_NEW_TOKENS if dataset == "gsm8k" else _LIVEBENCH_MAX_NEW_TOKENS


def _truncate_at_stops(text, stops):
    for stop in stops:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text


def _vllm_inner_model(llm):
    return llm.llm_engine.model_executor.driver_worker.model_runner.model


def _load_weights_into_vllm(llm, hf_model):
    inner = _vllm_inner_model(llm)
    inner.load_weights(hf_model.state_dict().items())


# ---------------------------------------------------------------------------
# scoring (ported from dataset-side evaluator.py files)

def _strict_match(text):
    match = _ANSWER_REGEX.search(text)
    return match.group(1) if match else "[invalid]"


def _normalize(text):
    text = text.strip()
    for pattern in _REGEXES_TO_IGNORE:
        text = pattern.sub("", text)
    return text.lower()


def _score_gsm8k(response, reference):
    return float(_normalize(_strict_match(response)) == _normalize(reference))


_INSTRUCTIONS_REGISTRY = None

def _instructions_registry():
    global _INSTRUCTIONS_REGISTRY
    if _INSTRUCTIONS_REGISTRY is None:
        if _LIVEBENCH_IF_SRC not in sys.path:
            sys.path.insert(0, _LIVEBENCH_IF_SRC)
        from instruction_following_eval import instructions_registry
        _INSTRUCTIONS_REGISTRY = instructions_registry
    return _INSTRUCTIONS_REGISTRY


def _score_livebench(response, reference):
    registry = _instructions_registry()
    prompt = reference["prompt"]
    instruction_id_list = reference["instruction_id_list"]
    kwargs_list = reference["kwargs"]
    follow_list = []
    for idx, instruction_id in enumerate(instruction_id_list):
        cls = registry.INSTRUCTION_DICT[instruction_id]
        instruction = cls(instruction_id)
        kw = {k: v for k, v in kwargs_list[idx].items() if v is not None}
        instruction.build_description(**kw)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        follow_list.append(
            bool(response.strip()) and instruction.check_following(response)
        )
    follow_all = all(follow_list) if follow_list else False
    s1 = 1.0 if follow_all else 0.0
    s2 = (sum(1 for f in follow_list if f) / len(follow_list)) if follow_list else 0.0
    return (s1 + s2) / 2


# ---------------------------------------------------------------------------
# answer-token span

def _answer_span_ids(dataset, tokenizer, row, answer_text):
    if dataset == "gsm8k":
        prompt = row["arguments"]["gen_args_0"]["arg_0"]
    else:
        prompt = _build_prompt("livebench", row, tokenizer)

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids   = tokenizer(prompt + answer_text, return_tensors="pt",
                           add_special_tokens=False).input_ids[0]
    answer_ids = full_ids[prompt_ids.numel():]
    return prompt_ids, answer_ids


# ---------------------------------------------------------------------------
# prompts + references

def _build_prompt(dataset, row, tokenizer=None):
    if dataset == "gsm8k":
        return row["arguments"]["gen_args_0"]["arg_0"]
    messages = [{"role": "user", "content": row["prompt"]}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )


def _build_reference(dataset, row):
    if dataset == "gsm8k":
        return row["target"]
    return {
        "prompt":              row["prompt"],
        "instruction_id_list": row["instruction_id_list"],
        "kwargs":              row["kwargs"],
    }

# ---------------------------------------------------------------------------
# evaluator

class BenchmarkProxyEvaluatorVLLM:
    def __init__(
        self,
        config,
        gpu_id,
        model_id="",
        quantization_proxy_paths=(),
        quantization_proxy_fake_paths=(),
        bits_range=(),
        group_size=128,
        datasets=("gsm8k", "livebench"),
        n_sample=16,
        batch_size=4,
        seed=0,
        dev="cuda",
        dtype="auto",
        mode='generate',
        gpu_memory_utilization=0.8,
        # max_model_len=4096,
        max_model_len=8096,
        tensor_parallel_size=1,
        tqdm_position=0,
        **kwargs,
    ):
        self.config = config
        self.gpu_id = gpu_id
        self.model_id = model_id
        self.group_size = group_size
        self.dev = dev
        self.dtype = dtype
        self.datasets = tuple(datasets)
        self.mode = mode
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.tqdm_position = tqdm_position

        self.accelerator, self.device_map = init_accelerator(self.gpu_id, config)
        if self.mode == 'generate':
            self.eval = self.eval_generate
            self.batch_size = batch_size
        elif self.mode == 'jsd':
            self.eval = self.eval_jsd
            self.batch_size = 1

        self.tokenizer = get_tokenizer(model_id)

        if mode == 'generate':
            self.quantization_proxies = [
                get_hfmodel(
                    quantization_proxy_fake_path, device_map='cpu',
                    dtype='float16', use_cache=True,
                ) for quantization_proxy_fake_path in quantization_proxy_fake_paths
            ]
        elif self.mode == 'jsd':
            self.quantization_proxies = get_quantization_proxy(
                quantization_proxy_paths, self.device_map, False,
            )

        self.bits_range = list(bits_range)
        assert len(self.bits_range) == len(self.quantization_proxies), (
            "Number of bits range and quantization proxies must be the same"
        )

        self.sample_loader = _SampleLoader(
            datasets=self.datasets,
            n_sample=n_sample,
            seed=seed,
            load_fp16=(self.mode == 'jsd'),
        )

        if self.mode == 'jsd':
            print("Loading FP16 model")
            self.fp16_model = get_hfmodel(self.model_id, dtype=self.dtype, device_map=self.device_map)
            self.fp16_answers = self.sample_loader.fp16_answers

        self.model = deepcopy(self.quantization_proxies[-1])
        if self.mode == 'jsd':
            self.model = self.model.to(dev)

        if self.mode == 'generate':
            import os as _os
            _os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            from vllm import LLM, SamplingParams
            self._SamplingParams = SamplingParams
            self.llm = LLM(
                model=self.model_id,
                dtype='float16',
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=True,
                distributed_executor_backend="uni",
                disable_log_stats=True,
            )

        self.accelerator.wait_for_everyone()

    # --- sample: swap in per-block quantized linears for this architecture ---

    def sample(self, arch):
        for linear, linear_bits in arch["linear"].items():
            for blk_idx, bits in enumerate(linear_bits):
                flag = False
                for q_bits, q_model in zip(self.bits_range, self.quantization_proxies):
                    if math.isclose(int(bits), q_bits) and q_bits > 0:
                        setsubattr(
                            getblock(self.model, self.config)[blk_idx],
                            linear,
                            # deepcopy(getsubattr(getblock(q_model, self.config)[blk_idx], linear)).to(self.dev),
                            getsubattr(getblock(q_model, self.config)[blk_idx], linear),
                        )
                        flag = True
                if not flag:
                    raise NotImplementedError(f"{linear}: {bits} is not available")
        return self.model

    # --- generate scoring (vLLM batched) ---
    @torch.inference_mode()
    def _score_generate(self, dataset, samples):
        score_fn = _score_gsm8k if dataset == "gsm8k" else _score_livebench
        stops = _stop_sequences(dataset)

        prompts = [_build_prompt(dataset, row, self.tokenizer) for row in samples]
        refs    = [_build_reference(dataset, row) for row in samples]

        sampling_params = self._SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=_max_new_tokens(dataset),
            stop=stops,
        )

        scores = []
        for start in range(0, len(prompts), self.batch_size):
            chunk_prompts = prompts[start:start + self.batch_size]
            chunk_refs    = refs[start:start + self.batch_size]
            outs = self.llm.generate(chunk_prompts, sampling_params, use_tqdm=False)
            for ref, out in zip(chunk_refs, outs):
                resp = _truncate_at_stops(out.outputs[0].text, stops)
                scores.append(score_fn(resp, ref))
        return fmean(scores) if scores else 0.0

    # --- jsd scoring (HF batched by self.batch_size) ---

    @torch.inference_mode()
    def _score_jsd(self, dataset, samples):
        jsd = JSD()
        answers_by_id = self.fp16_answers[dataset]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        spans = []
        for row in samples:
            answer_text = answers_by_id.get(row["doc_id"], "")
            if not answer_text:
                continue
            prompt_ids, answer_ids = _answer_span_ids(
                dataset, self.tokenizer, row, answer_text,
            )
            if answer_ids.numel() == 0:
                continue
            spans.append((prompt_ids, answer_ids))

        losses = []
        for start in range(0, len(spans), self.batch_size):
            chunk = spans[start:start + self.batch_size]
            seqs  = [torch.cat([p, a]) for p, a in chunk]
            max_len = max(s.numel() for s in seqs)

            ids  = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, :s.numel()] = s
                attn[i, :s.numel()] = 1
            ids  = ids.to(self.dev)
            attn = attn.to(self.dev)

            fp16_logits = self.fp16_model(ids, attention_mask=attn).logits
            cand_logits = self.model(ids, attention_mask=attn).logits

            for i, (prompt_ids, answer_ids) in enumerate(chunk):
                start_pos = prompt_ids.numel() - 1
                end_pos   = start_pos + answer_ids.numel()
                losses.append(
                    jsd(cand_logits[i, start_pos:end_pos], fp16_logits[i, start_pos:end_pos]).item()
                )
        return fmean(losses) if losses else 0.0

    # --- eval loops ---

    def eval_generate(self, architectures, max_value):
        samples = self.sample_loader.extract_sample(groups=_BALANCED_GROUPS)
        metric_list = {f"{ds}_gen": [] for ds in self.datasets}
        bits_usage_list = []

        for architecture in tqdm(
            architectures,
            desc=f"Evaluating Generate vLLM [w{self.tqdm_position}]",
            position=self.tqdm_position,
            leave=True,
            dynamic_ncols=True,
        ):
            self.sample(architecture)
            _load_weights_into_vllm(self.llm, self.model)

            for ds in self.datasets:
                metric_list[f"{ds}_gen"].append(float(self._score_generate(ds, samples[ds])))

            bits_usage_list.append(get_bits_usage(architecture, self.config, self.group_size))
            clean_up()

        return metric_list, bits_usage_list

    def eval_jsd(self, architectures, max_value):
        samples = self.sample_loader.extract_sample(groups=_MONOTONIC)
        metric_list = {f"{ds}_jsd": [] for ds in self.datasets}
        bits_usage_list = []

        for architecture in tqdm(
            architectures,
            desc=f"Evaluating JSD [w{self.tqdm_position}]",
            position=self.tqdm_position,
            leave=True,
            dynamic_ncols=True,
        ):
            self.sample(architecture)

            for ds in self.datasets:
                metric_list[f"{ds}_jsd"].append(self._score_jsd(ds, samples[ds]))

            bits_usage_list.append(get_bits_usage(architecture, self.config, self.group_size))
            clean_up()

        return metric_list, bits_usage_list
