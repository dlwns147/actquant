import argparse
import gc
import pprint
import numpy as np
import torch
import time

from e2e.quantized_llama import modeling_llama
from quarot.nn import Linear4bit
import torch
import transformers
from tqdm import tqdm
from quarot.transformers import MultiLayerPagedKVCache4Bit

# model_configs = [
#     "meta-llama/Llama-2-7b-hf",
#     # "meta-llama/Llama-2-13b-hf", 
#     # "meta-llama/Llama-2-70b-hf", 
# ]

benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 1
num_bench_steps = 1

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()

@torch.inference_mode()
def device_warmup(device: str):
    warm_up = torch.randn((4096, 4096)).to(device)
    for i in range(100):
        torch.mm(warm_up, warm_up)
    _cleanup()
    torch.cuda.reset_peak_memory_stats()

def repeated_run(num_repeats=10):
    def func(module):
        def _f(*args, **kwargs):
            times = []
            for i in range(num_repeats):
                times.append(module(*args, **kwargs))
            return tuple(zip(*times))
        return _f
    return func


@torch.inference_mode()
@repeated_run()
def module_benchmark(module):
    # warmup
    # for i in range(num_warmup_steps):
    #     out = module()
    # torch.cuda.synchronize()
    
    _cleanup()
    torch.cuda.reset_max_memory_allocated()
    start_time = time.perf_counter()
    
    
    for i in range(num_bench_steps):
        out = module()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    end_time = time.perf_counter()
    _cleanup()

    return (end_time - start_time) * 1000 / num_bench_steps, peak_memory

def get_graph_wrapper(cls, device=0):

    class GraphWrapper(cls):

        def __init__(self, *args, **kwargs):
            super(GraphWrapper, self).__init__(*args, **kwargs)
            self.built_graph = False
            self.graph_device = device

        def forward(self, *args, **kwargs):
            with torch.cuda.device(self.graph_device):
                if not self.built_graph:
                    self.static_args = args
                    self.static_kwargs = kwargs

                    s = torch.cuda.Stream(device=self.graph_device)
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        super(GraphWrapper,
                              self).forward(*self.static_args,
                                            **self.static_kwargs)
                    torch.cuda.current_stream().wait_stream(s)

                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=s):
                        self.static_output = super(GraphWrapper, self).forward(
                            *self.static_args, **self.static_kwargs)

                    self.built_graph = True
                    print("Built CUDA graph of model.")

                # these two loops take < 1e-4 seconds for llama2
                for i in range(len(args)):
                    if isinstance(args[i], torch.Tensor):
                        self.static_args[i].copy_(args[i])
                for kw in kwargs:
                    if isinstance(kwargs[kw], torch.Tensor):
                        self.static_kwargs[kw].copy_(kwargs[kw])

                self.graph.replay()
                return self.static_output

        def reset(self):
            if self.built_graph:
                del self.static_args, self.static_kwargs
                del self.graph
                del self.static_output
                self.built_graph = False

    return GraphWrapper

def maybe_wrap(use_cuda_graph):
    return (lambda x: get_graph_wrapper(x)
            ) if use_cuda_graph else (lambda x: x)

def get_model_quantized(config_name, attn_implementation='flash_attention_2', use_cuda_graph=False):
    config = transformers.AutoConfig.from_pretrained(
        config_name,
        attn_implementation=attn_implementation
    )
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights(): 
        model = maybe_wrap(use_cuda_graph)(modeling_llama.QuarotLlamaForCausalLM)(config=config)
    torch.set_default_dtype(dtype_old)
    return model


def get_model_hf(config_name, attn_implementation='flash_attention_2', use_cuda_graph=False):
    return maybe_wrap(use_cuda_graph)(transformers.LlamaForCausalLM).from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation=attn_implementation
    )

def get_model_fp16(config_name, attn_implementation='flash_attention_2', use_cuda_graph=False):
    return maybe_wrap(use_cuda_graph)(modeling_llama.QuarotFP16LlamaForCausalLM).from_pretrained(
        config_name, 
        torch_dtype=torch.float16, 
        attn_implementation=attn_implementation
    )


def run_prefill(model, bsz, prefill_length):
    device = model.device
    device_warmup(device)
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    return module_benchmark(lambda: model(test_input))


def run_decode(model, bsz, prefill_length, decode_steps):
    device = model.device
    device_warmup(device)
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    model._expected_max_length = prefill_length + decode_steps
    prefill = model(test_input)
    # past_key_values = out.past_key_values
    del prefill.logits
    _cleanup()
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _decode_for_multiple_steps():
        past_key_values = prefill.past_key_values
        if type(past_key_values) == MultiLayerPagedKVCache4Bit:
            past_key_values.length = prefill_length
        for _ in tqdm(range(decode_steps)):
            # past_key_values = model(next_input, past_key_values=past_key_values).past_key_values
            out = model(next_input, past_key_values=past_key_values)
            # print(f'logits: {out.logits.shape}, past_key_values: {out.past_key_values[0][0].shape if type(out.past_key_values) == tuple else out.past_key_values.length}')
            past_key_values = out.past_key_values
    return module_benchmark(_decode_for_multiple_steps)
    

def run_e2e(model, bsz, prefill_length, decode_steps):
    device = model.device
    device_warmup(device)
    test_input = torch.randint(100, 200, (bsz, prefill_length), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range (bsz)], dtype=torch.int32, device=device)
    def _prefill_and_decode_for_multiple_steps():
        model._expected_max_length = prefill_length + decode_steps
        out = model(test_input)
        for _ in tqdm(range(decode_steps)):
            out = model(next_input, past_key_values=out.past_key_values)
            # print(f'logits: {out.logits.shape}, past_key_values: {out.past_key_values[0][0].shape if type(out.past_key_values) == tuple else out.past_key_values.length}')
    return module_benchmark(_prefill_and_decode_for_multiple_steps)


def _wait_for_input():
    print("Press enter")
    input()

@torch.no_grad
def run_all_for_model(model, bsz, prefill, decode):
    model.eval()
    model = model.cuda()
    print(f'run prefill')
    time_prefill, memory_prefil = run_prefill(model, bsz, prefill)
    _cleanup()
    if decode > 0:
        print(f'run decode')
        time_decode, memory_decode = run_decode(model, bsz, prefill, decode)
        _cleanup()
        print(f'run e2e')
        time_e2e, memory_e2e = run_e2e(model, bsz, prefill, decode)
        _cleanup()
    else:
        time_decode = time_e2e = None
        memory_e2e = memory_prefil
    return time_prefill, time_decode, time_e2e, memory_e2e


def benchmark(args):   
    # for config_name in model_configs:
    config_name = args.model_config

    model = get_model_quantized(config_name, attn_implementation=args.attn_implementation, use_cuda_graph=args.use_cuda_graph)
    print(f'Benchmark Int4 model')
    time_prefill_i4, time_decode_i4, time_e2e_i4, mem_i4 = run_all_for_model(
        model, args.batch_size, args.prefill_seq_len, args.decode_steps)
    del model
    _cleanup()

    # model = get_model_fp16(config_name)
    model = get_model_fp16(config_name, attn_implementation=args.attn_implementation, use_cuda_graph=args.use_cuda_graph)
    print(f'Benchmark FP16 model')
    time_prefill_f16, time_decode_f16, time_e2e_f16, mem_f16 = run_all_for_model(
        model, args.batch_size, args.prefill_seq_len, args.decode_steps)
    del model
    _cleanup()

    time_prefill_hf_f16 = time_decode_hf_f16 = time_e2e_hf_f16 = mem_hf_f16 = 0
    # model = get_model_hf(config_name, attn_implementation=args.attn_implementation, use_cuda_graph=args.use_cuda_graph)
    # print(f'Benchmark HF FP16 model')
    # time_prefill_hf_f16, time_decode_hf_f16, time_e2e_hf_f16, mem_hf_f16 = run_all_for_model(
    #     model, args.batch_size, args.prefill_seq_len, args.decode_steps)
    # del model
    # _cleanup()

    print(f"Prefill Int4 time: {np.median(time_prefill_i4):.3f} +- {1.96 * np.std(time_prefill_i4):.3f} ms")
    print(f"Prefill FP16 time: {np.median(time_prefill_f16):.3f} +- {1.96 * np.std(time_prefill_f16):.3f} ms")
    print(f"Prefill HF FP16 time: {np.median(time_prefill_hf_f16):.3f} +- {1.96 * np.std(time_prefill_hf_f16):.3f} ms")
    print(f"FP16/Int4 Speedup: {np.median(time_prefill_f16) / np.median(time_prefill_i4):.3f}x")
    print(f"HF FP16/Int4Speedup: {np.median(time_prefill_hf_f16) / np.median(time_prefill_i4):.3f}x")
    print(f'Prefill & {config_name} & {args.batch_size} & {args.prefill_seq_len} & HF FP16: {np.median(time_prefill_hf_f16):.3f} & FP16: {np.median(time_prefill_f16):.3f} & Int4: {np.median(time_prefill_i4):.3f}\\\\')

    if args.decode_steps is not None:
        print(f"Decode Int4 time: {np.median(time_decode_i4):.3f} +- {1.96 * np.std(time_decode_i4):.3f} ms")
        print(f"Decode FP16 time: {np.median(time_decode_f16):.3f} +- {1.96 * np.std(time_decode_f16):.3f} ms")
        print(f"Decode HF FP16 time: {np.median(time_decode_hf_f16):.3f} +- {1.96 * np.std(time_decode_hf_f16):.3f} ms")
        print(f"FP16/Int4 Speedup: {np.median(time_decode_f16) / np.median(time_decode_i4):.3f}x")
        print(f"HF FP16/Int4 Speedup: {np.median(time_decode_hf_f16) / np.median(time_decode_i4):.3f}x")
        print(f'Decode & {config_name} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & HF FP16: {np.median(time_decode_hf_f16):.3f} & FP16: {np.median(time_decode_f16):.3f} & Int4: {np.median(time_decode_i4):.3f}\\\\')

        print(f"E2E Int4 time: {np.median(time_e2e_i4):.3f} +- {1.96 * np.std(time_e2e_i4):.3f} ms")
        print(f"E2E FP16 time: {np.median(time_e2e_f16):.3f} +- {1.96 * np.std(time_e2e_f16):.3f} ms")
        print(f"E2E HF FP16 time: {np.median(time_e2e_hf_f16):.3f} +- {1.96 * np.std(time_e2e_hf_f16):.3f} ms")
        print(f"FP16/Int4 Speedup: {np.median(time_e2e_f16) / np.median(time_e2e_i4):.3f}x")
        print(f"HF FP16/Int4 Speedup: {np.median(time_e2e_hf_f16) / np.median(time_e2e_i4):.3f}x")
        print(f'E2E & {config_name} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & HF FP16: {np.median(time_e2e_hf_f16):.3f} & FP16: {np.median(time_e2e_f16):.3f} & Int4: {np.median(time_e2e_i4):.3f}\\\\')
    
    # table-style output

    print(f"Int4 memory: {np.median(mem_i4) / (1024 * 1024 * 1024):.3f} GB +- {1.96 * np.std(mem_i4):.3f}")
    print(f"FP16 memory: {np.median(mem_f16) / (1024 * 1024 * 1024):.3f} GB +- {1.96 * np.std(mem_f16):.3f}")
    print(f"HF FP16 memory: {np.median(mem_hf_f16) / (1024 * 1024 * 1024):.3f} GB +- {1.96 * np.std(mem_hf_f16):.3f}")
    print(f"FP16/Int4 Memory saving: {np.median(mem_f16) / np.median(mem_i4):.3f}x")
    print(f"HF FP16/Int4 Memory saving: {np.median(mem_hf_f16) / np.median(mem_i4):.3f}x")
    print(f'Memory saving & {config_name} & {args.batch_size} & {args.prefill_seq_len} & {args.decode_steps} & HF FP16: {np.median(mem_hf_f16) / (1024 * 1024 * 1024):.3f} GB & FP16: {np.median(mem_f16) / (1024 * 1024 * 1024):.3f} GB & Int4: {np.median(mem_i4) / (1024 * 1024 * 1024):.3f} GB & \\\\')
    
    print('--------------')

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
        default=None,
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
        '--use_hf', action='store_true',
        help='',
    )
    parser.add_argument(
        '--use_cuda_graph', action='store_true',
        help='',
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    num_warmup_steps = args.num_warmup_steps
    num_bench_steps = args.num_bench_steps
    benchmark(args)
