import os
from datasets import load_dataset
import json
from tqdm import tqdm
import numpy as np
os.environ["WANDB_DISABLED"] = "true"
import warnings
warnings.filterwarnings("ignore")

# from utils.process_args import process_args
# from transformers import LlamaConfig, MistralConfig, AutoTokenizer, AutoConfig, AutoModelForCausalLM
# from models.replace import replace_model

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


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # For results in KIVI paper (Llama, Llama-Chat, Mistral-7B-v0.1), we do not apply any special treatment to the prompt.
    # For lmsys/longchat-7b-v1.5-32k and mistralai/Mistral-7B-Instruct-v0.2, we need to rewrite the prompt a little bit.
    # Update: we add the template for the new llama-3-instruct model
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
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
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
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
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def pred_long_bench(model, tokenizer, save_path, long_bench_config, e):
    # print(args)
    # seed_everything(args.seed)
    # args = parse_args()
    # model2path = json.load(open("config/model2path.json", "r"))
    # model2maxlen = json.load(open("long_bench_config/model2maxlen.json", "r"))
    model2maxlen = json.load(open(os.path.join(long_bench_config, "model2maxlen.json"), "r"))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_name = args.model

    model_name = model.config._name_or_path.split('/')[-1]
    max_length = model2maxlen[model_name]
    device = model.device
    
    if e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    # dataset2prompt = json.load(open("long_bench_config/dataset2prompt.json", "r"))
    # dataset2maxlen = json.load(open("long_bench_config/dataset2maxlen.json", "r"))
    
    dataset2prompt = json.load(open(os.path.join(long_bench_config, "dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(open(os.path.join(long_bench_config, "dataset2maxlen.json"), "r"))
    # predict on each dataset

    if e:
        os.makedirs(os.path.join(save_path, "pred_e"), exist_ok=True)
    else:
        os.makedirs(os.path.join(save_path, "pred"), exist_ok=True)
    # if not os.path.exists("pred"):
    #     os.makedirs("pred")
    # if not os.path.exists("pred_e"):
    #     os.makedirs("pred_e")
    
    for dataset in datasets:
        if e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            # path = f"pred_e/{args.model_name}_{max_length}_k{args.k_bits}v${args.v_bits}bits_group{args.group_size}_residual{args.residual_length}"
            path = os.path.join(save_path, "pred_e")
            # if not os.path.exists(path):
            #     os.makedirs(path)
            out_path = os.path.join(path, f'{dataset}.jsonl')
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            # path = f"pred/{args.model_name}_{max_length}_k{args.k_bits}v${args.v_bits}bits_group{args.group_size}_residual{args.residual_length}"
            path = os.path.join(save_path, "pred")
            # if not os.path.exists(path):
            #     os.makedirs(path)
            out_path = os.path.join(path, f'{dataset}.jsonl')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
                
def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

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

def eval_long_bench(path, e):
    # args = parse_args()
    if e:
        path = os.path.join(path, "pred_e")
    else:
        path = os.path.join(path, "pred")
    scores = dict()
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    # if e:
    #     # out_path = f"pred_e/{args.model}/result.json"
    #     out_path = os.path.join(path, 'pred_e', 'result.json')
    # else:
    #     # out_path = f"pred/{args.model}/result.json"
    #     out_path = os.path.join(path, 'pred', 'result.json')
    out_path = os.path.join(path, 'result.json')
    print(f'task: {list(scores.keys())}')
    print(f'score: {list(scores.values())}')

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)