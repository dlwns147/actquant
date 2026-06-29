import os
import torch
import json
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from .metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

MINILONGBENCH_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "minilongbench_data", "data")

MINILONGBENCH_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader",
    "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
    "lcc", "repobench-p",
]

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def build_chat(tokenizer, prompt, model_name):
    if "instruct" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
    elif "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def _prepare_prompt(tokenizer, json_obj, max_length, prompt_format, dataset,
                    model_name):
    """Final prompt for one sample (truncate-in-middle + chat template)."""
    prompt = prompt_format.format(**json_obj)
    tp = tokenizer(prompt, truncation=False,
                   return_tensors="pt").input_ids[0]
    if len(tp) > max_length:
        half = int(max_length / 2)
        prompt = (tokenizer.decode(tp[:half], skip_special_tokens=True)
                  + tokenizer.decode(tp[-half:], skip_special_tokens=True))
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
                       "repobench-p"]:
        prompt = build_chat(tokenizer, prompt, model_name)
    return prompt


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format,
             dataset, device, model_name):
    """Per-sample (batch_size=1) generation.

    Length-bucket batching was investigated and removed —
    see analysis/batching_investigation/."""
    preds = []
    for json_obj in tqdm(data, desc=dataset):
        prompt = _prepare_prompt(tokenizer, json_obj, max_length,
                                 prompt_format, dataset, model_name)
        input = tokenizer(prompt, truncation=False,
                          return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id,
                              tokenizer.encode("\n",
                                               add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:],
                                skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        })
    return preds


def pred_minilongbench(model, tokenizer, save_path, longbench_config,
                       data_dir=None, model_name=None):
    if data_dir is None:
        data_dir = MINILONGBENCH_DATA_DIR

    model2maxlen = json.load(
        open(os.path.join(longbench_config, "model2maxlen.json"), "r"))
    dataset2prompt = json.load(
        open(os.path.join(longbench_config, "dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(
        open(os.path.join(longbench_config, "dataset2maxlen.json"), "r"))

    if model_name is None:
        name_or_path = model.config._name_or_path
        if name_or_path.endswith('.json'):
            name_or_path = os.path.dirname(name_or_path)
        model_name = name_or_path.rstrip('/').split('/')[-1]
    if model_name not in model2maxlen:
        matched = next((k for k in model2maxlen
                        if model_name.startswith(k) or k in model_name), None)
        if matched:
            model_name = matched
        else:
            raise KeyError(
                f"model_name '{model_name}' not found in model2maxlen. "
                f"Available keys: {list(model2maxlen.keys())}")
    max_length = model2maxlen[model_name]
    device = model.device

    pred_dir = os.path.join(save_path, "pred")
    os.makedirs(pred_dir, exist_ok=True)

    all_preds = {}
    for dataset in MINILONGBENCH_DATASETS:
        jsonl_path = os.path.join(data_dir, f"{dataset}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"[MiniLongBench] Skipping {dataset}: file not found at "
                  f"{jsonl_path}")
            continue

        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        out_path = os.path.join(pred_dir, f"{dataset}.jsonl")

        preds = get_pred(model, tokenizer, data, max_length, max_gen,
                         prompt_format, dataset, device, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
        all_preds[dataset] = preds
    # Return the in-memory predictions keyed by dataset so the caller can score
    # exactly THIS run's outputs (eval_minilongbench_preds) instead of re-reading
    # the dir, which would pick up stale/other-config .jsonl files via listdir.
    return all_preds


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](
                prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def _score_preds(dataset, preds):
    """Score one dataset's prediction list (in-memory dicts from get_pred).
    Shared by eval_minilongbench_preds (in-memory) and eval_minilongbench
    (file-based) so both compute identical scores."""
    predictions = [d["pred"] for d in preds]
    answers = [d["answers"] for d in preds]
    all_classes = preds[0]["all_classes"] if preds else None
    return scorer(dataset, predictions, answers, all_classes)


def _finalize_scores(scores, save_path):
    """Append the avg, print, and write result.json (shared tail)."""
    print(f'task: {list(scores.keys())}')
    print(f'score: {list(scores.values())}')
    if scores:
        avg = round(np.mean(list(scores.values())), 2)
        print(f'avg score: {avg}')
        scores["avg"] = avg
    if save_path:
        pred_dir = os.path.join(save_path, "pred")
        os.makedirs(pred_dir, exist_ok=True)
        with open(os.path.join(pred_dir, "result.json"), "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
    return scores


def eval_minilongbench_preds(preds_by_dataset, save_path=None):
    """Score the predictions returned by pred_minilongbench DIRECTLY (in
    memory), with no re-read of the prediction dir. Corruption-safe: scores
    exactly the datasets produced in THIS run, never stale or other-config
    .jsonl files lying in the directory.

    When save_path is given, the score summary is still written to
    <save_path>/pred/result.json (predictions themselves were already written
    by pred_minilongbench), so the on-disk artifacts are unchanged."""
    scores = {dataset: _score_preds(dataset, preds)
              for dataset, preds in preds_by_dataset.items()}
    return _finalize_scores(scores, save_path)


def eval_minilongbench(save_path):
    """File-based scoring: re-read every .jsonl in <save_path>/pred and score.
    Kept for backward compatibility / offline rescoring. NOTE: this reads ALL
    .jsonl files in the dir via listdir, so it can pick up stale predictions
    from a previous run — prefer eval_minilongbench_preds(pred_minilongbench(
    ...)) for the live eval path."""
    pred_dir = os.path.join(save_path, "pred")
    all_files = os.listdir(pred_dir)
    print("Evaluating on:", all_files)

    preds_by_dataset = {}
    for filename in all_files:
        if not filename.endswith(".jsonl"):
            continue
        dataset = filename.split('.')[0]
        with open(os.path.join(pred_dir, filename), "r",
                  encoding="utf-8") as f:
            preds_by_dataset[dataset] = [json.loads(line) for line in f]
    scores = {dataset: _score_preds(dataset, preds)
              for dataset, preds in preds_by_dataset.items()}
    return _finalize_scores(scores, save_path)
