import os
import json
import torch
import argparse
from time import time

import lm_eval
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantizedCacheConfig, HQQQuantizedCache, QuantoQuantizedCache
from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleHQQQuantizedCache, FlexibleVanillaQuantizedCache

from common_code.func import clean_up
from common_code.eval import eval_zeroshot
from common_code.eval_long_bench import pred_long_bench, eval_long_bench


def pass_key(model, tokenizer, per_layer_config, residual_length=128, q_group_size=32, pass_key_file='/NAS/SJ/actquant/common_code/passkey_examples.jsonl'):
    print( "-----------------------------------" )
    for line in open(pass_key_file, "r"):
        cache_config = FlexibleQuantizedCacheConfig(axis_key=0, axis_value=0, asym=True, q_group_size=q_group_size, residual_length=residual_length, device='cuda', 
                                            per_layer_quant=True,per_layer_config=per_layer_config)
        past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config)

        clean_up()
        torch.cuda.reset_max_memory_allocated() 

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


def zeroshot(model_name, quant_scheme, per_layer_config, residual_length=128, q_group_size=32, 
            task_list=["coqa", "gsm8k", "truthfulqa_gen"], num_fewshots=0, batch_size='auto', limit=None, 
            quantilizer='vanilla', device='cuda'):

    from lm_eval.models.huggingface_quant import HFLM_Quant
    from lm_eval import evaluator

    clean_up()

    if batch_size.isdigit():
        batch_size = int(batch_size)
    else:
        batch_size = 'auto'

    model = HFLM_Quant(pretrained=model_name,
                        nbits_key=1,
                        nbits_value=1,
                        residual_length=residual_length if quant_scheme == 'per-channel-asym' else 0,
                        q_group_size=q_group_size if quant_scheme == 'per-channel-asym' else -1,
                        asym=True,
                        axis_key=1 if quant_scheme == 'per-channel-asym' else 0,
                        axis_value=0,
                        dtype=torch.bfloat16,
                        force_quant=False,
                        per_layer_quant=True,
                        per_layer_config=per_layer_config,
                        quantilizer=quantilizer,
                        device_map='auto',
                        parallelize=True, # True for multi-gpu, False for single-gpu 
                    )

    model._model.generation_config.temperature = 1.0
    model._model.generation_config.top_p = 1.0
    model._model.generation_config.top_k = None

    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=num_fewshots, # 0 for no fewshot, 1 for one shot, 2 for two shot, etc.
        batch_size=batch_size,
        device=device
    )

    # import code; code.interact('flexible_quant_zeroshot line 70', local=dict(globals(), **locals()))
    # print(results['results']['gsm8k']['exact_match,flexible-extract'])
    # return float(results['results']['gsm8k']['exact_match,flexible-extract'])

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
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--q_group_size", type=int, default=32)
    parser.add_argument("--quant_scheme", type=str, default="per-channel-asym", choices=["per-channel-asym", "per-token-asym"])
    parser.add_argument("--quantilizer", type=str, default="vanilla", choices=["vanilla", "hqq"])
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

    parser.add_argument("--target_bit", type=float, default=3.0)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    return args

def main(args):
    results = {}
    model_name = args.model_name

    if 'llama' in model_name.lower():
        per_layer_config_path = f'/NAS/SJ/actquant/KVTuner/search_results/g{args.q_group_size}l{args.residual_length}/post_search/llama_31_8b_instruct_channel.json'
    elif 'qwen' in model_name.lower():
        per_layer_config_path = f'/NAS/SJ/actquant/KVTuner/search_results/g{args.q_group_size}l{args.residual_length}/post_search/qwen_25_7b_instruct_channel.json'
    else:
        raise ValueError(f"Model {model_name} is not supported")

    save_path = f'./eval_results/g{args.q_group_size}l{args.residual_length}/' + args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    with open(per_layer_config_path, 'r') as f:
        data = json.load(f)

    target_bit = str(args.target_bit)
    tot_scale, per_layer_config = data[target_bit]['tot_scale'] if target_bit in data else None, data[target_bit]['per_layer_config'] if target_bit in data else None
    if per_layer_config is None:
        items = [item for item in data.values() if abs(item['tot_scale'] - args.target_bit) < 0.1]
        items = sorted(items, key=lambda x: x['accuracy'], reverse=True)

        if len(items) > 0:
            tot_scale, per_layer_config = items[0]['tot_scale'], items[0]['per_layer_config']
        else:
            print(f"No per_layer_config found for {model_name} with target_bit {args.target_bit}")
            return

    results['tot_scale'] = tot_scale
    results['per_layer_config'] = per_layer_config

    if args.pass_key:
        pass_key(model, tokenizer, per_layer_config, residual_length=args.residual_length, q_group_size=args.q_group_size)

    if args.zeroshot:
        # task_list가 없으면 기본값 사용
        zeroshot_results = zeroshot(model_name, args.quant_scheme, per_layer_config, residual_length=args.residual_length, q_group_size=args.q_group_size, 
                 task_list=args.task_list, num_fewshots=args.num_fewshots, batch_size=args.batch_size, limit=args.limit, 
                 quantilizer=args.quantilizer, device=args.device)
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
            import code; code.interact('flexible_quant_zeroshot line 185', local=dict(globals(), **locals()))


if __name__ == "__main__":
    args = parse_args()
    main(args)