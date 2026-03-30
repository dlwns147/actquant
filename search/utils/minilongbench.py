import os
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

MINILONGBENCH_DATA_DIR = os.path.join(os.path.dirname(__file__), "minilongbench_data", "data")

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
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm(data, desc=dataset):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                     tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        })
    return preds


def pred_minilongbench(model, tokenizer, save_path, longbench_config, data_dir=None):
    if data_dir is None:
        data_dir = MINILONGBENCH_DATA_DIR

    model2maxlen = json.load(open(os.path.join(longbench_config, "model2maxlen.json"), "r"))
    dataset2prompt = json.load(open(os.path.join(longbench_config, "dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(open(os.path.join(longbench_config, "dataset2maxlen.json"), "r"))

    model_name = model.config._name_or_path.split('/')[-1]
    max_length = model2maxlen[model_name]
    device = model.device

    pred_dir = os.path.join(save_path, "pred")
    os.makedirs(pred_dir, exist_ok=True)

    for dataset in MINILONGBENCH_DATASETS:
        jsonl_path = os.path.join(data_dir, f"{dataset}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"[MiniLongBench] Skipping {dataset}: file not found at {jsonl_path}")
            continue

        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        out_path = os.path.join(pred_dir, f"{dataset}.jsonl")

        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def eval_minilongbench(save_path):
    pred_dir = os.path.join(save_path, "pred")
    scores = {}
    all_files = os.listdir(pred_dir)
    print("Evaluating on:", all_files)

    for filename in all_files:
        if not filename.endswith(".jsonl"):
            continue
        dataset = filename.split('.')[0]
        predictions, answers = [], []
        all_classes = None
        with open(os.path.join(pred_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
        score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score

    out_path = os.path.join(pred_dir, "result.json")
    print(f'task: {list(scores.keys())}')
    print(f'score: {list(scores.values())}')
    if scores:
        avg = round(np.mean(list(scores.values())), 2)
        print(f'avg score: {avg}')
        scores["avg"] = avg

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
