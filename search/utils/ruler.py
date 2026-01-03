import os
import torch
import json
from time import time
from copy import deepcopy
from tqdm import tqdm
# from transformers import StopStringCriteria
from lm_eval.models.utils import stop_sequences_criteria
from lm_eval.tasks import utils
from .ruler_utils import niah_utils, vt_utils, cwe_utils, fwe_utils, qa_utils, common_utils
from torch.utils.data import DataLoader

def prepare_generation_kwargs(task_config_map, task_name: str, yaml_path:str, gen_toks=None) -> tuple[dict, int]:
    """태스크별 generation_kwargs와 max_gen_toks를 준비"""
    config_path = task_config_map.get(task_name)
    if config_path is None:
        # 기본값 사용 (niah_single_1과 동일)
        config_path = os.path.join(yaml_path, "niah_single_1.yaml")
    config = utils.load_yaml_config(config_path)
    generation_kwargs = deepcopy(config["generation_kwargs"])
    generation_kwargs.pop("until", None)
    max_gen_toks = generation_kwargs.pop("max_gen_toks")
    
    if gen_toks is not None:
        max_gen_toks = gen_toks
    
    return generation_kwargs, max_gen_toks


def eval_ruler(model, 
               tokenizer, 
               model_id,
               yaml_path='',
               tasks=[],
               length=[],
               batch_size=1,
               nsample=50, 
               seed=0, 
               gen_toks=128,
               result_path=''):
    
    task_function = {
        # NIAH tasks
        "niah_single_1": niah_utils.niah_single_1,
        "niah_single_2": niah_utils.niah_single_2,
        "niah_single_3": niah_utils.niah_single_3,
        "niah_multikey_1": niah_utils.niah_multikey_1,
        "niah_multikey_2": niah_utils.niah_multikey_2,
        "niah_multikey_3": niah_utils.niah_multikey_3,
        "niah_multivalue": niah_utils.niah_multivalue,
        "niah_multiquery": niah_utils.niah_multiquery,

        # Ruler tasks
        "ruler_vt": vt_utils.get_vt_dataset,
        "ruler_cwe": cwe_utils.get_cw_dataset,
        "ruler_fwe": fwe_utils.fwe_download,
        "ruler_qa_squad": qa_utils.get_squad,
        "ruler_qa_hotpot": qa_utils.get_hotpotqa
    }

    # 태스크별 config 파일 경로 매핑
    task_config_map = {
        "niah_single_1": os.path.join(yaml_path, 'niah_single_1.yaml'),
        "niah_single_2": os.path.join(yaml_path, 'niah_single_2.yaml'),
        "niah_single_3": os.path.join(yaml_path, 'niah_single_3.yaml'),
        "niah_multikey_1": os.path.join(yaml_path, 'niah_multikey_1.yaml'),
        "niah_multikey_2": os.path.join(yaml_path, 'niah_multikey_2.yaml'),
        "niah_multikey_3": os.path.join(yaml_path, 'niah_multikey_3.yaml'),
        "niah_multivalue": os.path.join(yaml_path, 'niah_multivalue.yaml'),
        "niah_multiquery": os.path.join(yaml_path, 'niah_multiquery.yaml'),
        "ruler_vt": os.path.join(yaml_path, 'vt.yaml'),
        "ruler_cwe": os.path.join(yaml_path, 'cwe.yaml'),
        "ruler_fwe": os.path.join(yaml_path, 'fwe.yaml'),
        "ruler_qa_squad": os.path.join(yaml_path, 'qa_squad.yaml'),
        "ruler_qa_hotpot": os.path.join(yaml_path, 'qa_hotpot.yaml'),
    }
    
    task_until = {
        "niah_single_1": ['.'],
        "niah_single_2": ['.'],
        "niah_single_3": ['.'],
        "niah_multikey_1": ['.'],
        "niah_multikey_2": ['.'],
        "niah_multikey_3": ['.'],
        "niah_multivalue": ['.'],
        "niah_multiquery": ['.'],
        "ruler_vt": ['.', '\n\n'],
        "ruler_cwe": [],
        "ruler_fwe": [],
        "ruler_qa_squad": ['.'],
        "ruler_qa_hotpot": ['.'],
    }
    
    task_function = {task: task_function[task] for task in tasks}

    # 태스크별 generation 설정 저장
    task_generation_configs = {task: prepare_generation_kwargs(task_config_map, task, yaml_path, gen_toks) for task in task_function.keys()}

    datasets = dict()
    for task in tqdm(task_function.keys(), desc=f'Creating datasets'):
        # print(f'Preparing {task} dataset')
        dataset = task_function[task](model=model_id, max_seq_lengths=length, num_samples=nsample)['test']
        
        dataset.shuffle(seed)
        dataset = dataset.select(range(nsample))
        dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # dataset = dataset.batch(batch_size)
        
        datasets[task] = dataset

    tot_scores = dict()
    start_time = time()
    device = model.device

    for task in task_function.keys():
        kwargs, max_gen_toks = task_generation_configs[task]
        task_scores = []
        
        for docs in tqdm(datasets[task], desc=f"Evaluating {task}"):
            for i in range(len(docs['input'])):
                docs['input'][i] = docs['input'][i] + ' ' + docs['gen_prefix'][i]

            # tokenized_sample = tokenizer(docs["input"], return_tensors="pt", padding=True)
            tokenized_sample = tokenizer(docs["input"], return_tensors="pt")
            print(f'tokenized_sample: {tokenized_sample.input_ids.shape}')
            context_enc = tokenized_sample.input_ids.to(device)
            attn_masks = tokenized_sample.attention_mask.to(device)

            # stopping_criteria = stop_sequences_criteria(
            #     tokenizer, until, context_enc.shape[1], context_enc.shape[0])
            
            stopping_criteria = stop_sequences_criteria(tokenizer, [tokenizer.eos_token] + task_until[task], context_enc.shape[1], context_enc.shape[0])

            kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            with torch.inference_mode():
                output = model.generate(
                    input_ids=context_enc,
                    attention_mask=attn_masks,
                    # pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=stopping_criteria,
                    use_cache=True,
                    **kwargs
                )

            # output이 올바른 device에 있는지 확인하고 필요시 이동
            # device_map="auto"를 사용할 때는 output이 모델의 마지막 레이어 device에 있을 수 있음
            # if output.device != context_enc.device:
            #     output = output.to(context_enc.device)
            
            output = output[:, context_enc.shape[1]:]
            output = tokenizer.batch_decode(output, skip_special_tokens=True)

            for i in range(len(docs['input'])):
                doc = {key: docs[key][i] for key in docs.keys()}
                score = common_utils.process_results(doc, [output[i]])[str(doc["max_length"])]
                task_scores.append(score)
        
        if len(task_scores) > 0:
            avg_score = sum(task_scores) / len(task_scores)
            tot_scores[task] = avg_score
            print(f"Average score for {task}: {avg_score}")

    elapsed_time = (time() - start_time)
  
    print(f"RULER Time: {elapsed_time:.2f}")
    print(list(tot_scores.keys()))
    print(list(tot_scores.values()))
            
    if result_path:
        tot_scores["time"] = elapsed_time
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                existing_scores = json.load(f)
                existing_scores.update(tot_scores)
                tot_scores = existing_scores
                
        save_dir = os.path.dirname(result_path)
        os.makedirs(save_dir, exist_ok=True)

        with open(result_path, "w") as f:
            json.dump(tot_scores, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {result_path}")