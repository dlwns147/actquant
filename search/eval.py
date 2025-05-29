
from hqq.models.hf.base import AutoHQQHFModel
from transformers import AutoModelForCausalLM

from utils.data import get_loader
from utils.eval import eval_metric
from utils.func import init_accelerator
from utils.dispatch import simple_dispatch_model
import gc
import torch
import json

def main():
    model_path = '/SSD/huggingface/meta-llama'
    # model_name = 'Llama-2-7b-hf'
    model_name = 'Llama-2-13b-hf'
    model_id = f'{model_path}/{model_name}'

    config_path = 'config/llama.json'
    with open(config_path, 'r') as f:
        config = json.load(f)[model_name]

    accelerator, device_map = init_accelerator('1', config)

    # awq_path = f'/SSD/awq/${model_name}_w4_g128_fake_4bit_awq.pt'
    # print(f'awq_path : {awq_path}')
    # model = AutoModelForCausalLM.from_pretrained(awq_path, torch_dtype='auto', low_cpu_mem_usage=True, device_map='auto')

    hqq_path = f'/SSD/hqq/{model_name}_3bit_128gs_1axis_qscale_false_qzero_false'
    model = AutoHQQHFModel.from_quantized(hqq_path, device_map='cpu')
    model = simple_dispatch_model(model, device_map)
    print(f'hqq_path : {hqq_path}')

    print(f'model.device : {model.device}, accelerator.device : {accelerator.device}, device_map : {device_map}')

    gc.collect()
    torch.cuda.empty_cache()
    print(f'memory : {torch.cuda.memory_allocated()}')

    loader = get_loader('wikitext2', train=False, model=model_id)
    w2_ppl = eval_metric(model, accelerator=accelerator, metric='ppl', loader=loader, seqlen=2048)
    print(f'wikitext2 : {w2_ppl}')

    loader = get_loader('c4', train=False, model=model_id)
    c4_ppl = eval_metric(model, accelerator=accelerator, metric='ppl', loader=loader, seqlen=2048)
    print(f'c4 : {c4_ppl}')


    from lm_eval.models.huggingface import HFLM
    from lm_eval import tasks, evaluator, utils
    import os

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=64)

    task_manager = tasks.TaskManager(include_path='lm-evaluation-harness/lm_eval/tasks')

    zeroshot_tasks = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"]
    task_names = task_manager.match_tasks(zeroshot_tasks)
    for task in [task for task in zeroshot_tasks if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)

    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=zeroshot_tasks,
        num_fewshot=0,
        batch_size=64,
        max_batch_size=None,
        device='cuda:0',
        use_cache=None,
        limit=None,
        check_integrity=False,
        write_out=False,
        gen_kwargs=None,
        task_manager=task_manager,
        # decontamination_ngrams_path=None,
    )['results']
    print(f'results : {results}')


if __name__ == '__main__':
    main()
