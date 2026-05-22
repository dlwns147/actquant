# LLaMA model with KIVI
import warnings
import os
warnings.filterwarnings("ignore")
import torch
import random
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from utils.eval import *
from utils.data import *
from models.replace import replace_model

# For reproducibility
random.seed(0)
torch.manual_seed(0)


def main(args):
    print(args)
    model_id = os.path.join(args.model_path, args.model_name)

    # config = LlamaConfig.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    config.k_bits = args.k_bits # KiVi currently support 2/4 K/V bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length # corresponding to the number of recent fp16 tokens
    config.use_flash = args.use_flash
    config.quant_kv_output = args.quant_kv_output
    config.k_quant_per = args.k_quant_per
    config.v_quant_per = args.v_quant_per

    device = 'cuda'

    # model = LlamaForCausalLM_KIVI.from_pretrained(
    #     pretrained_model_name_or_path=model_id,
    #     config=config,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     # torch_dtype='auto',
    # )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # torch_dtype='auto',
        attn_implementation='flash_attention_2' if config.use_flash else 'sdpa'
    )
    model = replace_model(model, config)

    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

    # batch_size = args.zeroshot_batch_size
    # prompt_len = 64
    # gen_len = 192
    # inp = torch.randint(100, 200, (batch_size, prompt_len), dtype=torch.int32, device=device)
    # out = model.generate(inp, min_new_tokens=gen_len, max_new_tokens=gen_len)
    # print(f'out: {out.shape}')
    # exit()

    if args.eval_ppl:
        ppl_list = []
        testloaders = {dataset: get_loaders(dataset, tokenizer=tokenizer, model=model_id, seqlen=args.seqlen)[1] for dataset in args.datasets}
        print('Loading datasets')
        use_cache = model.config.use_cache
        model.config.use_cache = False
        for dataset in args.datasets:
            ppl = eval_ppl(model, testloaders[dataset], device=device, seqlen=args.seqlen)
            ppl_list.append(ppl)
            print(f'{dataset} : {ppl}')
        print(f'ppl: {ppl_list}')
        model.config.use_cache = use_cache
    
    if args.zeroshot:
        from lm_eval.models.huggingface import HFLM
        from lm_eval import tasks, evaluator, utils
        import datasets
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        
        # model.tie_weights = lambda: None
        hflm = HFLM(pretrained=model, tokenizer=tokenizer)# batch_size=args.zeroshot_batch_size)# , batch_size='auto')
        
        results = evaluator.simple_evaluate(
            model=hflm,
            tasks=args.tasks,
            # batch_size=args.zeroshot_batch_size,
        )['results']
        
        # acc_norm = [task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] for task_result in results.values()]
        # acc = [task_result['acc,none'] for task_result in results.values()]
        
        task = list(results.keys())
        # avg_acc_norm = np.mean(acc_norm)
        # avg_acc = np.mean(acc)
        # print(f'avg_acc_norm : {avg_acc_norm}, avg_acc : {avg_acc}')
        print(f'task : {task}')
        for task, result in results.items():
            print(f'task: {task}, result: {result}')
        # print(f'results {results}')
        # print(f'acc_norm : {acc_norm}')
        # print(f'acc : {acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--k_bits', type=int, default=2,
                        help='')
    parser.add_argument('--v_bits', type=int, default=2,
                        help='')
    parser.add_argument('--group_size', type=int, default=128,
                        help='')
    parser.add_argument('--residual_length', type=int, default=128,
                        help='')
    parser.add_argument('--eval_ppl', action='store_true', help='')
    parser.add_argument('--datasets', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='')
    parser.add_argument('--quant_kv_output', action='store_true', help='')
    parser.add_argument('--k_quant_per', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_per', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--zeroshot', action='store_true', help='')
    parser.add_argument('--tasks', type=str, nargs='+', default=['piqa','winogrande','hellaswag','arc_challenge','arc_easy', 'lambada_openai', 'boolq'])
    parser.add_argument('--zeroshot_batch_size', type=int, default=64,
                        help='')
    parser.add_argument('--use_flash', action='store_true', help='')
    parser.add_argument('--kivi_implementation', action='store_true', help='')

    cfgs = parser.parse_args()
    main(cfgs)
