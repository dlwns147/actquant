import argparse
import gc
import pprint
import numpy as np
import torch
import time
import os

# from e2e.quantized_llama import modeling_llama
from quantized_llama import modeling_llama
from quarot.nn import Linear4bit
import torch
import transformers
from tqdm import tqdm
from quarot.transformers import MultiLayerPagedKVCache4Bit


model_configs = [
    "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Llama-2-13b-hf", 
    # "meta-llama/Llama-2-70b-hf", 
]

benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 1
num_bench_steps = 1

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def get_model_quantized(config_name, attn_implementation='flash_attention_2'):
    config = transformers.AutoConfig.from_pretrained(
        config_name,
        attn_implementation=attn_implementation
    )
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights(): 
        model = modeling_llama.QuarotLlamaForCausalLM(config=config)
    torch.set_default_dtype(dtype_old)
    return model


def get_model_hf(config_name, attn_implementation='flash_attention_2'):
    return transformers.LlamaForCausalLM.from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation=attn_implementation
    )

def get_model_fp16(config_name, attn_implementation='flash_attention_2'):
    return modeling_llama.QuarotFP16LlamaForCausalLM.from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation=attn_implementation
    )


def benchmark(args):
    torch.cuda.memory._record_memory_history()
    config_name = args.model_config
    # model = get_model_quantized(config_name, attn_implementation=args.attn_implementation)
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
    ) as prof, torch.inference_mode() :
    
        with torch.autograd.profiler.record_function("load model"):
            model = (get_model_quantized if args.model_type == 'int4' else \
                    get_model_fp16 if args.model_type == 'fp16' else \
                    get_model_hf)(config_name, attn_implementation=args.attn_implementation)
            model.eval()
            model = model.cuda()
            device = model.device

        inp = torch.randint(100, 200, (args.batch_size, args.prefill_seq_len), dtype=torch.int32, device=device)
    
        if args.decode_steps > 0:
            if args.model_type == 'hf':
                with torch.autograd.profiler.record_function("generate"):
                    model.generate(inp, min_new_tokens=args.decode_steps, max_new_tokens=args.decode_steps)

            else:
                # next_input = torch.tensor([[100] for _ in range (args.batch_size)], dtype=torch.int32, device=device)
                model._expected_max_length = args.prefill_seq_len + args.decode_steps
                out = model(inp)              
                
                for _ in tqdm(range(args.decode_steps)):
                    logits, past_key_values = out.logits, out.past_key_values
                    # import pdb; pdb.set_trace()
                    # inp = torch.tensor([[logits[:, -1].max(1)[1].unsqueeze(1)] for _ in range (args.batch_size)], dtype=torch.int32, device=device)
                    inp = logits[:, -1].max(1)[1].unsqueeze(1)
                    out = model(inp, past_key_values=past_key_values)
                    print(f'past_key_values: {past_key_values.length}')

        else:
            model(inp)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
    os.makedirs(args.output_path, exist_ok=True)
    file_name = f'{args.model_config.split("/")[-1]}_{args.model_type}_bs{args.batch_size}_pl{args.prefill_seq_len}_gl{args.decode_steps}_{args.attn_implementation}.pkl'
    torch.cuda.memory._dump_snapshot(os.path.join(args.output_path, file_name))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_config', type=str,
        help='',
        required=True,
        default='meta-llama/Llama-2-7b-hf',
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--prefill_seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--decode_steps', type=int,
        help='Decode steps',
        required=False,
        default=0,
    )
    parser.add_argument(
        '--num_warmup_steps', type=int,
        help='',
        default=1,
    )
    parser.add_argument(
        '--num_bench_steps', type=int,
        help='',
        default=1,
    )
    parser.add_argument(
        '--attn_implementation', type=str,
        help='',
        default='flash_attention_2',
    )
    parser.add_argument(
        '--model_type', type=str, choices=['hf', 'fp16', 'int4'],
        help='',
    )
    parser.add_argument(
        '--output_path', type=str, default='',
        help='',
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    num_warmup_steps = args.num_warmup_steps
    num_bench_steps = args.num_bench_steps
    benchmark(args)
