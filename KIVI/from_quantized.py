# LLaMA model with KIVI
import warnings
import os
warnings.filterwarnings("ignore")
import torch
import random
from models.llama_autogptq import LlamaForCausalLM_KIVI, LlamaAttention_KIVI
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from transformers import LlamaConfig, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from utils.eval import *
from utils.data import *

# For reproducibility
random.seed(0)
torch.manual_seed(0)
device = 'cuda'

def main(args):
    model = LlamaForCausalLM_KIVI.from_quantized(
        save_dir_or_hub = args.load_qmodel_path
    )
    model.eval()
    model.seqlen = args.seqlen
    # model = replace_model(model, config)
    model = model.to(device)
    
    model_id = model.config._name_or_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

    testloaders = {dataset: get_loaders(dataset, tokenizer=tokenizer, model=model_id, seqlen=args.seqlen)[1] for dataset in args.datasets}
    print('Loading datasets')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    for dataset in args.datasets:
        ppl = eval_ppl(model, testloaders[dataset], device=device, seqlen=args.seqlen)
        print(f'{dataset} : {ppl}')
    model.config.use_cache = use_cache
    
    from lm_eval.models.huggingface import HFLM
    from lm_eval import tasks, evaluator, utils
    import datasets
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    
    
    # model.tie_weights = lambda: None
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.zeroshot_batch_size)# , batch_size='auto')
    
    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=args.tasks,
        num_fewshot=0,
        batch_size=args.zeroshot_batch_size,
    )
    
    acc_norm = [task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] for task_result in results.values()]
    acc = [task_result['acc,none'] for task_result in results.values()]
    
    task = list(results.keys())
    avg_acc_norm = np.mean(acc_norm)
    avg_acc = np.mean(acc)
    print(f'avg_acc_norm : {avg_acc_norm}, avg_acc : {avg_acc}')
    print(f'task : {task}')
    print(f'acc_norm : {acc_norm}')
    print(f'acc : {acc}')
        
        
supported_datasets = ['wikitext2', 'ptb', 'c4']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_qmodel_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--k_bits', type=int, default=2,
                        help='')
    parser.add_argument('--v_bits', type=int, default=2,
                        help='')
    parser.add_argument('--group_size', type=int, default=128,
                        help='')
    parser.add_argument('--residual_length', type=int, default=32,
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


    cfgs = parser.parse_args()
    main(cfgs)