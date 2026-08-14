import os
import torch
import hashlib
from datasets import load_dataset
import json
from tqdm import tqdm
import numpy as np
os.environ["WANDB_DISABLED"] = "true"
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

# The dataset lists pred_longbench actually scores. Module-level (not inlined in
# pred_longbench) so other code can ask "is this corpus also a benchmark target?"
# without duplicating the list — see correlation.py's contamination check: a
# calibration corpus that the benchmark ALSO scores makes their correlation
# circular.
LONGBENCH_DATASETS = ["triviaqa", "qasper", "trec", "samsum", "lcc",
                      "repobench-p", "qmsum", "multi_news"]
LONGBENCH_E_DATASETS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
                        "gov_report", "multi_news", "trec", "triviaqa", "samsum",
                        "passage_count", "passage_retrieval_en", "lcc",
                        "repobench-p"]

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
    name = model_name.lower()
    # Instruct/chat models -> apply the model's chat template. Covers "*-Instruct"
    # (Llama/Qwen), Gemma "*-it", and "*-chat". The old check was only
    # `"instruct" in name`, which MISSED Gemma ("gemma-3-12b-it") -> Gemma silently
    # ran raw prompting on LongBench. Gated by chat_template presence so base models
    # (some of which still ship a template) stay raw as before.
    if "longchat" in name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif (any(k in name for k in ("instruct", "-it", "chat"))
          and getattr(tokenizer, "chat_template", None)):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def _prepare_prompt(tokenizer, json_obj, max_length, prompt_format, dataset,
                    model_name):
    """Build the final prompt string for one sample (truncate-in-middle +
    optional chat template)."""
    prompt = prompt_format.format(**json_obj)
    tokenized_prompt = tokenizer(prompt, truncation=False,
                                 return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = (tokenizer.decode(tokenized_prompt[:half],
                                   skip_special_tokens=True)
                  + tokenizer.decode(tokenized_prompt[-half:],
                                     skip_special_tokens=True))
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
                       "repobench-p"]:
        prompt = build_chat(tokenizer, prompt, model_name)
    return prompt


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format,
             dataset, device, model_name, stamp=None):
    """Per-sample (batch_size=1) generation, legacy KIVI/ThinK path.

    Length-bucket batched generation was investigated and removed —
    see analysis/batching_investigation/ for the full investigation
    (probes, decode-length sweeps, equal-length vs mod-R, the
    flash-varlen-vs-dense Gemma3 cause). For our workloads the speedup
    was small (RULER ≈×1.05–1.2) or impossible (LongBench-E with the
    shipped max_length=131072 is ≈×1.0 under the only Δ0-safe mode for
    Gemma3, since natural document lengths almost never collide
    exactly)."""
    preds = []
    for json_obj in tqdm(data, desc=dataset):
        prompt = _prepare_prompt(tokenizer, json_obj, max_length,
                                 prompt_format, dataset, model_name)
        # The chat-templated prompt (build_chat -> apply_chat_template) already
        # carries the model's BOS; tokenizing with the default add_special_tokens
        # would prepend a SECOND BOS (Llama-3 / Gemma -> 2x, degrading generation).
        # Only add specials when the prompt isn't already BOS-prefixed.
        _bos = getattr(tokenizer, "bos_token", None)
        _add_special = not (_bos and prompt.startswith(_bos))
        input = tokenizer(prompt, truncation=False, return_tensors="pt",
                          add_special_tokens=_add_special).to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            # samsum stops on newline (upstream LongBench idiom, tuned for the
            # Llama-2 SentencePiece "\n" token). Guard the encode result: on some
            # tokenizers "\n" encodes to an empty list -> [-1] would IndexError.
            _nl = tokenizer.encode("\n", add_special_tokens=False)
            _eos = [tokenizer.eos_token_id] + ([_nl[-1]] if _nl else [])
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=_eos,
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
        # The first four keys are the upstream LongBench record (_score_preds
        # and every offline reader depend on them). The rest identifies the
        # generation: `_id` + `dataset` point back at the source example (the
        # split is fixed, so the prompt is regenerable from them — no reason to
        # store ~80 MB of prompt text per pass), and input_sha256 over the FINAL
        # prompt (post truncate-in-middle, post chat template) proves a
        # regenerated prompt is the one the model actually saw.
        # `score` is filled in later by eval_longbench_preds.
        preds.append({**(stamp or {}),
                      "pred": pred,
                      "answers": json_obj["answers"],
                      "all_classes": json_obj["all_classes"],
                      "length": json_obj["length"],
                      "dataset": dataset,
                      "_id": json_obj.get("_id"),
                      "context_tokens": int(context_length),
                      "generated_tokens": int(output.shape[-1] - context_length),
                      "input_sha256": hashlib.sha256(
                          prompt.encode("utf-8")).hexdigest()})
    return preds


def pred_longbench(model, tokenizer, save_path, longbench_config, e,
                   model_name=None, stamp=None):
    """LongBench / LongBench-E predictor (per-sample, batch_size=1).

    Writes <save_path>/pred[_e]/<dataset>.jsonl — one JSON per example with the
    GENERATION, the references, and the fields that identify it (dataset / _id /
    context_tokens / generated_tokens / input_sha256). Prompts are not stored:
    they are regenerable from (dataset, _id)."""
    model2maxlen = json.load(
        open(os.path.join(longbench_config, "model2maxlen.json"), "r"))

    # HQQ's AutoHQQHFModel.from_quantized() loads via a config.json path so
    # model.config._name_or_path can end in 'config.json'. Prefer the
    # explicit model_name; fall back to the config field otherwise.
    if not model_name:
        model_name = model.config._name_or_path.split('/')[-1]
    if model_name not in model2maxlen:
        raise KeyError(
            f"model_name '{model_name}' not in {longbench_config}/"
            f"model2maxlen.json (keys: {sorted(model2maxlen)}). "
            f"Pass the correct --model_name.")
    max_length = model2maxlen[model_name]
    device = model.device

    datasets = LONGBENCH_E_DATASETS if e else LONGBENCH_DATASETS

    dataset2prompt = json.load(
        open(os.path.join(longbench_config, "dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(
        open(os.path.join(longbench_config, "dataset2maxlen.json"), "r"))

    if e:
        os.makedirs(os.path.join(save_path, "pred_e"), exist_ok=True)
    else:
        os.makedirs(os.path.join(save_path, "pred"), exist_ok=True)

    all_preds = {}
    for dataset in datasets:
        if e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e",
                                split='test')
            out_path = os.path.join(save_path, "pred_e", f'{dataset}.jsonl')
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            out_path = os.path.join(save_path, "pred", f'{dataset}.jsonl')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen,
                         prompt_format, dataset, device, model_name,
                         stamp=stamp)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
        all_preds[dataset] = preds
    # Return the in-memory predictions keyed by dataset so the caller can score
    # exactly THIS run's outputs (eval_longbench_preds) instead of re-reading
    # the dir, which would pick up stale/other-config .jsonl files via listdir.
    return all_preds


def score_one(dataset, prediction, ground_truths, all_classes):
    """Score ONE prediction against its reference list — the single definition
    of a LongBench per-sample score. scorer / scorer_e aggregate it, and
    _score_preds writes it back per example, so a per-example score can never
    drift from the reported average."""
    score = 0.
    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        prediction = prediction.lstrip('\n').split('\n')[0]
    for ground_truth in ground_truths:
        score = max(score, dataset2metric[dataset](
            prediction, ground_truth, all_classes=all_classes))
    return score


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers,
                                                   lengths):
        score = score_one(dataset, prediction, ground_truths, all_classes)
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        # guard empty length bucket: np.mean([]) -> nan + RuntimeWarning (and NaN
        # in result.json). A length bin with no samples scores 0.0.
        scores[key] = round(100 * np.mean(scores[key]), 2) if scores[key] else 0.0
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        total_score += score_one(dataset, prediction, ground_truths, all_classes)
    return round(100 * total_score / len(predictions), 2)


def _score_preds(dataset, preds, e, annotate=False):
    """Score one dataset's prediction list (in-memory, the dicts produced by
    get_pred). Shared by eval_longbench_preds (in-memory) and eval_longbench
    (file-based) so both compute identical scores.

    annotate=True also stamps each pred dict with its own `score` (same
    score_one call the aggregate uses)."""
    predictions = [d["pred"] for d in preds]
    answers = [d["answers"] for d in preds]
    all_classes = preds[0]["all_classes"] if preds else None
    if annotate:
        for d in preds:
            d["score"] = float(score_one(dataset, d["pred"], d["answers"],
                                         all_classes))
    if e:
        lengths = [d["length"] for d in preds]
        return scorer_e(dataset, predictions, answers, lengths, all_classes)
    return scorer(dataset, predictions, answers, all_classes)


def eval_longbench_preds(preds_by_dataset, e, save_path=None,
                         rewrite_preds=True):
    """Score the predictions returned by pred_longbench DIRECTLY (in memory),
    with no re-read of the prediction dir. This is the corruption-safe path:
    it scores exactly the datasets produced in THIS run, never stale or
    other-config .jsonl files lying in the directory.

    When save_path is given, the score summary is written to
    <save_path>/pred[_e]/result.json AND each <dataset>.jsonl is rewritten with
    the per-example `score` stamped in (pred_longbench had to write the
    predictions before scoring, so this is the point where the two meet). Set
    rewrite_preds=False to leave the prediction files byte-identical."""
    scores = {dataset: _score_preds(dataset, preds, e, annotate=bool(save_path))
              for dataset, preds in preds_by_dataset.items()}
    print(f'task: {list(scores.keys())}')
    print(f'score: {list(scores.values())}')
    if save_path:
        out_dir = os.path.join(save_path, "pred_e" if e else "pred")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'result.json'), "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
        if rewrite_preds:
            for dataset, preds in preds_by_dataset.items():
                path = os.path.join(out_dir, f'{dataset}.jsonl')
                tmp = path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    for p in preds:
                        json.dump(p, f, ensure_ascii=False)
                        f.write('\n')
                os.replace(tmp, path)
    return scores


def eval_longbench(path, e):
    """File-based scoring: re-read every .jsonl in <path>/pred[_e] and score.
    Kept for backward compatibility / offline rescoring. NOTE: this reads ALL
    .jsonl files in the dir via listdir, so it can pick up stale predictions
    from a previous run — prefer eval_longbench_preds(pred_longbench(...)) for
    the live eval path."""
    if e:
        path = os.path.join(path, "pred_e")
    else:
        path = os.path.join(path, "pred")
    preds_by_dataset = dict()
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        dataset = filename.split('.')[0]
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            preds_by_dataset[dataset] = [json.loads(line) for line in f]
    scores = {dataset: _score_preds(dataset, preds, e)
              for dataset, preds in preds_by_dataset.items()}
    out_path = os.path.join(path, 'result.json')
    print(f'task: {list(scores.keys())}')
    print(f'score: {list(scores.values())}')

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
