"""GSM8K (chain-of-thought) evaluator.

Mirrors the evaluation methodology of lm-evaluation-harness
`lm_eval/tasks/gsm8k/gsm8k-cot.yaml`:

- Deterministic generation (`do_sample=False`) with stop sequences
  ``["Q:", "</s>", "<|im_end|>"]``.
- `strict-match` filter: extract the first match of
  ``The answer is (\\-?[0-9\\.\\,]+).`` from the generation.
- `exact_match` metric with ``ignore_case=True``, ``ignore_punctuation=False``,
  and ``regexes_to_ignore=[",", "\\$", "(?s).*#### ", "\\.$"]`` applied to
  both prediction and reference before comparison.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
_STOP_SEQUENCES = ("Q:", "</s>", "<|im_end|>")
_MAX_NEW_TOKENS = 256

_ANSWER_REGEX = re.compile(r"The answer is (\-?[0-9\.\,]+).")
_REGEXES_TO_IGNORE = (
    re.compile(r","),
    re.compile(r"\$"),
    re.compile(r"(?s).*#### "),
    re.compile(r"\.$"),
)

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForCausalLM.from_pretrained(
    _MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
_model.eval()


class _StopOnSequences(StoppingCriteria):
    def __init__(self, stop_sequences, tokenizer, prompt_len):
        self._stops = stop_sequences
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        generated = self._tokenizer.decode(
            input_ids[0, self._prompt_len:], skip_special_tokens=True
        )
        return any(stop in generated for stop in self._stops)


def generate(prompt: str) -> str:
    """Generate a deterministic completion for ``prompt``.

    Stops generation once any of the GSM8K stop sequences appears in the newly
    generated text, then strips that trailing stop sequence.
    """
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    stopping = StoppingCriteriaList(
        [_StopOnSequences(_STOP_SEQUENCES, _tokenizer, inputs.input_ids.shape[1])]
    )
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    generated = _tokenizer.decode(
        output_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    for stop in _STOP_SEQUENCES:
        idx = generated.find(stop)
        if idx != -1:
            generated = generated[:idx]
    return generated


def _strict_match(text: str) -> str:
    match = _ANSWER_REGEX.search(text)
    return match.group(1) if match else "[invalid]"


def _normalize(text: str) -> str:
    text = text.strip()
    for pattern in _REGEXES_TO_IGNORE:
        text = pattern.sub("", text)
    return text.lower()


def evaluate(response: str, reference: str) -> float:
    """Return 1.0 if the extracted answer matches ``reference``, else 0.0."""
    prediction = _strict_match(response)
    return float(_normalize(prediction) == _normalize(reference))
