import os
import torch
import json
import hashlib
from time import time
from copy import deepcopy
from tqdm import tqdm
# from transformers import StopStringCriteria
from lm_eval.models.utils import stop_sequences_criteria
from lm_eval.tasks import utils
from .ruler_utils import niah_utils, vt_utils, cwe_utils, fwe_utils, qa_utils, common_utils
from torch.utils.data import DataLoader

from .func import set_seed, topk_records

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


def default_per_example_path(result_path, seed=0):
    """`<ruler scores>.json` → `<ruler scores>_per_example_s<seed>.jsonl`.

    The dump is keyed by SEED because the seed decides the samples: eval_ruler
    re-seeds before dataset construction, so needle placement / shuffling — and
    therefore every prompt and every generation — is a function of it. Two seeds
    are two different sample sets and must not overwrite each other.

    A few hundred rows, so callers write it by default; an explicit
    --ruler_per_example_path overrides. '' in → '' out (no result path means
    nowhere to put it)."""
    if not result_path:
        return ''
    return f'{os.path.splitext(result_path)[0]}_per_example_s{int(seed)}.jsonl'


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
               per_example_path='',
               stamp=None,
               append_scores=False,
               topk_logits=5,
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

    _eos = getattr(model.generation_config, 'eos_token_id', None)
    _eos_ids = set(_eos if isinstance(_eos, (list, tuple)) else
                   ([_eos] if _eos is not None else []))
    if tokenizer.eos_token_id is not None:
        _eos_ids.add(tokenizer.eos_token_id)

    tot_scores = dict()
    per_example = []
    prompt_hashes = []      # every sample's input hash, dump or no dump
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
        sample_index = 0
        
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

            # output_logits (RAW, pre-processor) is what makes the per-example
            # dump diagnostic: for every generated token we keep the top-k
            # candidates the model weighed. Costs one (batch, vocab) tensor per
            # step while generate runs (~33 MB at max_gen 128); .scores would be
            # post-processor and this yaml carries temperature 0.0.
            _tk = dict(return_dict_in_generate=True, output_logits=True) \
                if topk_logits else {}
            with torch.inference_mode():
                output = model.generate(
                    input_ids=context_enc,
                    attention_mask=attn_masks,
                    stopping_criteria=stopping_criteria,
                    use_cache=True,
                    **kwargs, **_tk
                )

            step_logits = output.logits if topk_logits else None
            new_ids = (output.sequences if topk_logits else output)[:, context_enc.shape[1]:]
            output = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

            for i, doc in enumerate(docs):
                score = scorer(doc, [output[i]])[str(doc["max_length"])]
                task_scores.append(score)
                input_sha = hashlib.sha256(
                    doc["input"].encode("utf-8")).hexdigest()
                prompt_hashes.append(input_sha)
                if per_example_path:
                    refs = doc.get("outputs", [])
                    if isinstance(refs, str):
                        refs = [refs]
                    # The GENERATION plus what identifies it. The prompt itself
                    # is not stored: (seed, task, sample_index, max_length)
                    # regenerates it exactly — eval_ruler re-seeds before
                    # dataset construction — and input_sha256 proves the
                    # regenerated prompt is the one the model saw.
                    row = {
                        **(stamp or {}),
                        "task": task,
                        "sample_index": sample_index + i,
                        "seed": int(seed),
                        "requested_length": [int(x) for x in length],
                        "sample_length": int(doc["max_length"]),
                        "input_sha256": input_sha,
                        "context_tokens": int(attn_masks[i].sum().item()),
                        "generated_tokens": len(tokenizer.encode(
                            output[i], add_special_tokens=False)),
                        "prediction": output[i],
                        "references": [str(x) for x in refs],
                        "score": float(score),
                    }
                    if topk_logits:
                        row["topk"] = topk_records(
                            step_logits, new_ids[i], i, k=topk_logits,
                            eos_ids=_eos_ids)
                    per_example.append(row)
            sample_index += len(docs)

        if len(task_scores) > 0:
            avg_score = sum(task_scores) / len(task_scores)
            tot_scores[task] = avg_score
            print(f"Average score for {task}: {avg_score}")

    # The RULER samples are GENERATED at runtime from (seed, task, length), not
    # loaded from a fixed file: a change in niah_utils / datasets / the tokenizer
    # can silently redefine what "the same metric" measures. Fingerprint the
    # prompt SET so two runs can be proven comparable (and a drift is visible
    # even without the per-example dump).
    prompt_set_sha = hashlib.sha256(
        ''.join(sorted(prompt_hashes)).encode('utf-8')
    ).hexdigest()[:16] if prompt_hashes else ''

    elapsed_time = (time() - start_time)
  
    print(f"RULER Time: {elapsed_time:.2f}")
    print(list(tot_scores.keys()))
    print(list(tot_scores.values()))
            
    if result_path:
        tot_scores["time"] = elapsed_time
        # '_'-prefixed → correlation.py's aggregate skips it as a score column
        if prompt_set_sha:
            tot_scores["_prompt_set_sha"] = prompt_set_sha
        # One file = ONE run. Merging with whatever a previous call left here
        # (the old default) silently mixed two configurations' task scores in a
        # single scores.json — the per-example dump is replaced wholesale, so
        # the two artefacts then disagreed. append_scores=True restores the old
        # accumulate-across-calls behaviour if a caller really wants it.
        if append_scores and os.path.exists(result_path):
            with open(result_path, "r") as f:
                existing_scores = json.load(f)
                existing_scores.update(tot_scores)
                tot_scores = existing_scores
                
        save_dir = os.path.dirname(result_path)
        os.makedirs(save_dir, exist_ok=True)

        with open(result_path, "w") as f:
            json.dump(tot_scores, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {result_path}")

    if per_example_path:
        save_dir = os.path.dirname(per_example_path)
        os.makedirs(save_dir if save_dir else '.', exist_ok=True)
        tmp_path = per_example_path + '.tmp'
        with open(tmp_path, 'w') as f:
            for row in per_example:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        os.replace(tmp_path, per_example_path)
        print(f"Per-example RULER results saved to {per_example_path} "
              f"({len(per_example)} rows)")

    # THIS call's scores (plus 'time' when result_path was written). Callers that
    # loop over context lengths need the per-call dict: the on-disk scores.json is
    # merged with whatever a previous run left there, so reading it back cannot
    # tell this run's tasks from stale ones.
    return tot_scores
