import os
import json
import torch
import argparse
from time import time
from copy import deepcopy
import gc

import lm_eval
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantizedCacheConfig, HQQQuantizedCache, QuantoQuantizedCache
from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleHQQQuantizedCache, FlexibleVanillaQuantizedCache

from common_code.func import clean_up
from common_code.eval import eval_zeroshot
from common_code.eval_long_bench import pred_long_bench, eval_long_bench

from models.llama_kivi_4_50_3_generation import convert_generation_kivi
from models.llama_kivi_4_50_3 import convert_model_kivi
from models.KIVICache import KIVICacheConfig, KIVIDynamicCache
from transformers.cache_utils import DynamicCache


def pass_key(model, tokenizer, past_key_values, pass_key_file='/NAS/SJ/actquant/common_code/passkey_examples.jsonl'):
    print( "-----------------------------------" )

    original_past_key_values = deepcopy(past_key_values)

    for line in open(pass_key_file, "r"):
        torch.cuda.reset_max_memory_allocated() 

        past_key_values = deepcopy(original_past_key_values)

        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"] + prompt_postfix
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        print( "-----------------------------------" )
        print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
        print( "Passkey target:", example["target"] )

        tokens = model.generate(input_ids, past_key_values=past_key_values, use_cache=True, max_new_tokens=len(example["target"]))
        answer = prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
        answer = answer.replace("\n", "\\n")
        answer= f"[ {answer} ]"
        
        peak_memory = torch.cuda.max_memory_allocated()
        print( answer )
        print(f"Mem: {peak_memory / 1024 / 1024 / 1024:.3f} GB")
        print( "-----------------------------------\n" )

        del past_key_values
        gc.collect()
        torch.cuda.empty_cache()


def zeroshot(model, tokenizer, task_list=["coqa", "gsm8k", "truthfulqa_gen"], num_fewshots=0, batch_size='auto', limit=None, device='cuda'):
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator

    clean_up()

    if batch_size.isdigit():
        batch_size = int(batch_size)
    else:
        batch_size = 'auto'

    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = None

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)


    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=task_list,
        num_fewshot=num_fewshots, # 0 for no fewshot, 1 for one shot, 2 for two shot, etc.
        batch_size=batch_size,
        device=device
    )

    return results['results']


def long_bench(model, tokenizer, save_path=None, long_bench_config=None, e=False, past_key_values=None):
    long_bench_start = time()
    pred_long_bench(model, tokenizer=tokenizer, save_path=save_path, long_bench_config=long_bench_config, e=e, past_key_values=past_key_values)
    scores = eval_long_bench(save_path, e)
    long_bench_time = time() - long_bench_start
    
    sentences = []
    for k, v in vars(args).items():
        sentences.append(f"{k}: {v}\n")
    sentences.append(f'Longbench Time: {long_bench_time:.2f}s')
    sentences.append("\n")

    if save_path is not None:
        with open(os.path.join(save_path, "pred_e" if e else "pred", 'result.txt'), 'w') as f:
            for sentence in sentences:
                f.write(sentence)
                
    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    # parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # parser.add_argument("--per_layer_config_path", type=str, default="/NAS/SJ/actquant/KVTuner/search_results/post_search/llama_31_8b_instruct_channel.json")
    parser.add_argument("--use_kivi", action="store_true")
    parser.add_argument("--k_bits", type=int, default=4)
    parser.add_argument("--v_bits", type=int, default=4)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--q_group_size", type=int, default=128)
    # parser.add_argument("--task_list", type=list, default=["coqa", "gsm8k", "truthfulqa_gen"])
    parser.add_argument("--task_list", type=str, nargs='+', default=["coqa", "gsm8k", "truthfulqa_gen"])
    parser.add_argument("--num_fewshots", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to evaluate. If None, evaluate all examples.")
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--pass_key", action="store_true")
    parser.add_argument("--zeroshot", action="store_true")
    parser.add_argument("--long_bench", action="store_true")
    parser.add_argument("--long_bench_e", action="store_true")
    parser.add_argument("--long_bench_config", type=str, default="/NAS/SJ/actquant/common_code/long_bench_config")
    # parser.add_argument("--long_bench_result_path", type=str, default=None)

    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    return args

def main(args):
    results = {}
    model_name = args.model_name

    if args.save_path is not None:
        save_path = f'./eval_results/g{args.q_group_size}l{args.residual_length}/' + args.save_path
    else:
        save_path = None

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'args.txt'), 'w') as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")

    kivi_cache_config = KIVICacheConfig(nbits_key=args.k_bits, 
                                        nbits_value=args.v_bits, 
                                        q_group_size=args.q_group_size, 
                                        residual_length=args.residual_length, 
                                        per_layer_quant=False)
    if args.use_kivi:
        past_key_values = KIVIDynamicCache(kivi_cache_config)
    else:
        past_key_values = DynamicCache()

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, 
                                                 _attn_implementation="flash_attention_2", device_map="auto").cuda().eval()
    model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if args.use_kivi:
        convert_model_kivi(model, kivi_cache_config)
        convert_generation_kivi(kivi_cache_config)

    if args.pass_key:
        pass_key(model, tokenizer, past_key_values)

    if args.zeroshot:
        # task_list가 없으면 기본값 사용
        zeroshot_results = zeroshot(model, tokenizer, task_list=args.task_list, num_fewshots=args.num_fewshots, batch_size=args.batch_size, limit=args.limit, device=args.device)
        print(zeroshot_results)
        results['zeroshot'] = zeroshot_results

    if args.long_bench:
        cache_config = FlexibleQuantizedCacheConfig(axis_key=0, axis_value=0, asym=True, q_group_size=args.q_group_size, residual_length=args.residual_length, device='cuda', 
                                                    per_layer_quant=True, per_layer_config=per_layer_config)
        # past_key_values = FlexibleHQQQuantizedCache(cache_config=cache_config) # it seems in HQQ, 0 for per-token and 1 for per-channel
        past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config)

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = None
        model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        long_bench_results = long_bench(model, tokenizer, save_path=save_path, long_bench_config=args.long_bench_config, e=args.long_bench_e, past_key_values=past_key_values)
        print(long_bench_results)
        results['long_bench'] = long_bench_results

    if args.save_path is not None:
        try:
            with open(save_path + '.json', 'w') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            import code; code.interact('KIVI_cache_zeroshot line 185', local=dict(globals(), **locals()))

    import code; code.interact('KIVI_cache_zeroshot line 206', local=dict(globals(), **locals()))


if __name__ == "__main__":
    args = parse_args()
    main(args)