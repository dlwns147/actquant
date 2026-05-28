"""AWQ-quantize a fresh HF model for a given arch and save to disk.

The AWQ pipeline in this repo runs pseudo-quantization (weights stay in
fp16 dtype, values rounded to the chosen bit grid). The resulting model
is a vanilla HF model; ``save_pretrained`` writes a standard checkpoint
that downstream tooling (vLLM, lm_eval) can load.
"""

from __future__ import annotations

import os

from transformers import AutoTokenizer

from utils.func import get_hfmodel, get_bits_usage, clean_up
from quantization.model import get_quantized_model


def awq_quantize_and_save(args, config, arch, save_dir, device_map):
    os.makedirs(save_dir, exist_ok=True)

    model_id = f'{args.model_path}/{args.model_name}'
    bits_usage = get_bits_usage(arch, config, args.group_size)

    # Gemma-3 activations exceed the fp16 range (>65504) and cascade to NaN
    # through AWQ's scale search. Use the model's native dtype (bf16 for
    # Gemma-3) instead of forcing fp16.
    load_dtype = 'auto' if 'gemma-3' in args.model_name.lower() else 'float16'
    base = get_hfmodel(model_id, dtype=load_dtype, device_map='cpu')
    quantized = get_quantized_model(
        base, AutoTokenizer.from_pretrained(model_id),
        'awq', arch, bits_usage,
        args.group_size, config, 'cuda',
        device_map=device_map,
    )
    quantized = quantized.to('cuda').eval()

    quantized.save_pretrained(save_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(save_dir)

    del quantized
    clean_up()
    return save_dir
