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

from .func import set_seed

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
               result_path='',
               use_chat_template=True):
    
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
    
    # NOTE: no per-task early-stop STRING. The old task_until={'niah...':['.']}
    # truncated answers at the first '.', which killed chat-formatted list answers
    # ("1." -> stop) and contradicted the yaml `until: []`. Token-level analysis
    # showed the model's own turn-end token (eos, e.g. <|eot_id|>/<|im_end|>/
    # <end_of_turn>) is the only reliable answer terminator. With use_chat_template
    # the Instruct model emits it right after the answer -> generation self-limits
    # (correct AND ~2.5-4.5x fewer tokens than raw, which never emits eos and runs
    # to max_gen_toks). So we stop on eos only; max_gen_toks stays as a safety cap.

    task_function = {task: task_function[task] for task in tasks}

    # yaml_path doubles as the JSON data cache (hotpot/squad dev sets).
    # Pre-place hotpot_dev_distractor_v1.json / dev-v2.0.json there to skip network.
    if yaml_path:
        qa_utils.CACHE_DIR = yaml_path

    # Reproducibility: set seed before dataset creation and generation
    set_seed(seed)

    # Batched generation needs LEFT padding (generated tokens are sliced off with
    # the shared prompt length below); this tokenizer instance is dedicated to the
    # RULER call so mutating pad side/token is safe.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Chat template: for Instruct models, wrap `input` as a user turn and continue
    # the assistant turn with the RULER answer prefix (gen_prefix). The model then
    # emits its turn-end token right after the answer -> eos-only stop self-limits
    # generation (correct + fast). Falls back to raw prompting when the tokenizer
    # has no chat template (base models) or when explicitly disabled.
    use_chat = bool(use_chat_template and getattr(tokenizer, 'chat_template', None))
    if use_chat_template and not use_chat:
        print("[eval_ruler] tokenizer has no chat_template -> raw prompting")
    print(f"[eval_ruler] prompting={'chat' if use_chat else 'raw'}, "
          f"stop=eos-only, max_gen_toks=safety-cap")

    # 태스크별 generation 설정 저장
    task_generation_configs = {task: prepare_generation_kwargs(task_config_map, task, yaml_path, gen_toks) for task in task_function.keys()}

    datasets = dict()
    for task in tqdm(task_function.keys(), desc=f'Creating datasets'):
        # print(f'Preparing {task} dataset')
        dataset = task_function[task](model=model_id, max_seq_lengths=length, num_samples=nsample)['test']

        # NOTE: with multiple lengths in `length`, the builder yields nsample
        # samples PER length (chained), so select(range(nsample)) keeps a shuffled
        # nsample MIXED across lengths (not per-length). Fine for the single-length
        # usage; for a proper per-length breakdown keep all samples instead.
        dataset = dataset.shuffle(seed).select(range(nsample))
        # collate_fn=lambda b: b keeps the batch as a LIST OF SAMPLE DICTS. The
        # default DataLoader collate transposes list-valued fields — for RULER's
        # multi-answer 'outputs' (cwe/fwe/vt/niah_multivalue/niah_multiquery/
        # qa_squad) it turns ['a','b','c','d'] into [('a',),('b',),('c',),('d',)]
        # so downstream doc['outputs'][0] kept only the FIRST reference answer,
        # scoring a partial hit as a full match. Passthrough collate preserves the
        # full answer list.
        dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: b)

        datasets[task] = dataset

    tot_scores = dict()
    start_time = time()
    device = model.device

    # Per-task scorer: qa_squad/qa_hotpot use string_match_PART (score 1.0 if ANY
    # reference alias substring-matches), matching their yaml (qa_squad.yaml sets
    # process_results_part; qa_hotpot includes it). All other tasks use
    # string_match_ALL (fraction of references found). eval_ruler previously
    # hardcoded process_results (ALL) for EVERY task, under-scoring SQuAD's
    # multi-alias answers (a model matching one alias got fraction<1 instead of 1).
    PART_TASKS = {"ruler_qa_squad", "ruler_qa_hotpot"}

    for task in task_function.keys():
        kwargs, max_gen_toks = task_generation_configs[task]
        scorer = (common_utils.process_results_part if task in PART_TASKS
                  else common_utils.process_results)
        task_scores = []
        
        for docs in tqdm(datasets[task], desc=f"Evaluating {task}"):
            # docs is a list of sample dicts (see collate_fn above).
            if use_chat:
                # user turn = context+question; assistant turn is STARTED (via
                # add_generation_prompt) and continued with the answer prefix so
                # the model resumes from `gen_prefix` (RULER priming). The template
                # already carries the model's special/BOS tokens -> no extra BOS.
                inputs = [tokenizer.apply_chat_template(
                              [{"role": "user", "content": d['input']}],
                              tokenize=False, add_generation_prompt=True) + d['gen_prefix']
                          for d in docs]
                tokenized_sample = tokenizer(inputs, return_tensors="pt",
                                             padding=True, add_special_tokens=False)
            else:
                # raw prompting: input + ' ' + answer-prefix (BOS added by tokenizer)
                inputs = [d['input'] + ' ' + d['gen_prefix'] for d in docs]
                tokenized_sample = tokenizer(inputs, return_tensors="pt", padding=True)
            context_enc = tokenized_sample.input_ids.to(device)
            attn_masks = tokenized_sample.attention_mask.to(device)

            # eos-only stop; generate() also honors generation_config.eos_token_id
            # (the full turn-end id list), so termination is model-driven.
            stopping_criteria = stop_sequences_criteria(tokenizer, [tokenizer.eos_token], context_enc.shape[1], context_enc.shape[0])

            kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            with torch.inference_mode():
                output = model.generate(
                    input_ids=context_enc,
                    attention_mask=attn_masks,
                    stopping_criteria=stopping_criteria,
                    use_cache=True,
                    **kwargs
                )

            output = output[:, context_enc.shape[1]:]
            output = tokenizer.batch_decode(output, skip_special_tokens=True)

            for i, doc in enumerate(docs):
                score = scorer(doc, [output[i]])[str(doc["max_length"])]
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