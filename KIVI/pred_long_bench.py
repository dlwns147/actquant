import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
os.environ["WANDB_DISABLED"] = "true"
import warnings
warnings.filterwarnings("ignore")

# from utils.process_args import process_args
from transformers import LlamaConfig, MistralConfig, AutoTokenizer, AutoConfig, AutoModelForCausalLM
# from models.replace import replace_model


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
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm(data):
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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main(args):
    print(args)
    seed_everything(args.seed)
    # args = parse_args()
    # model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_name = args.model

    # define your model
    # model_args, data_args, training_args = process_args()
    # print(model_args, data_args, training_args)
    # model_name = model_args.model_name_or_path.split("/")[-1]
    # dtype = torch.bfloat16 if training_args.bf16 else torch.float
    # dtype = torch.float16

    
    model_id = os.path.join(args.model_path, args.model_name)
    dtype = torch.float16
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    
    # if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
    #     config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
    #                                         use_fast=False, 
    #                                         trust_remote_code=True, 
    #                                         tokenizer_type='llama')
    #                                         # model_max_length=training_args.model_max_length)
    # elif 'mistral' in model_args.model_name_or_path.lower():
    #     config = MistralConfig.from_pretrained(model_args.model_name_or_path)
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
    #                                         use_fast=False, 
    #                                         trust_remote_code=True)
    # else:
    #     raise NotImplementedError
    
    if 'llama' in model_id.lower() or 'longchat' in model_id.lower():
        if args.k_bits < 16 and args.v_bits < 16:
            from models.llama_kivi import LlamaForCausalLM_KIVI
            from models.replace import replace_model

            config.k_bits = args.k_bits # KiVi currently support 2/4 K/V bits
            config.v_bits = args.v_bits
            config.group_size = args.group_size
            config.residual_length = args.residual_length # corresponding to the number of recent fp16 tokens
            config.use_flash = args.use_flash
            config.quant_kv_output = args.quant_kv_output
            config.k_quant_per = args.k_quant_per
            config.v_quant_per = args.v_quant_per

            # model = LlamaForCausalLM_KIVI.from_pretrained(
            #     pretrained_model_name_or_path=model_id,
            #     config=config,
            #     low_cpu_mem_usage=True,
            #     torch_dtype=torch.float16,
            #     # torch_dtype='auto',
                # device_map="auto",
            # )

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
                attn_implementation='flash_attention_2' if config.use_flash else 'sdpa',
                device_map="auto"
            )
            model = replace_model(model, config)

        else:
            from transformers import LlamaForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id,
                config=config,
                # cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_flash_attention_2=True,
                device_map="auto",
            )

    elif 'mistral' in model_id.model_name_or_path.lower():
        if args.k_bits < 16 and args.v_bits < 16:
            from models.mistral_kivi import MistralForCausalLM_KIVI
            config.k_bits = args.k_bits
            config.v_bits = args.v_bits
            config.group_size = args.group_size
            config.residual_length = args.residual_length
            config.use_flash = True
            model = MistralForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=model_id,
                config=config,
                # cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            from transformers import MistralForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id,
                config=config,
                # cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_flash_attention_2=True,
                device_map="auto",
            )

    else:
        raise NotImplementedError

    #
    # Load model directly
    # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    # model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K")

    model.eval()
    max_length = model2maxlen[args.model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            path = f"pred_e/{args.model_name}_{max_length}_k{args.k_bits}v${args.v_bits}bits_group{args.group_size}_residual{args.residual_length}"
            if not os.path.exists(path):
                os.makedirs(path)
            out_path = os.path.join(path, f'{dataset}.jsonl')
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            path = f"pred/{args.model_name}_{max_length}_k{args.k_bits}v${args.v_bits}bits_group{args.group_size}_residual{args.residual_length}"
            if not os.path.exists(path):
                os.makedirs(path)
            out_path = os.path.join(path, f'{dataset}.jsonl')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, args.model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
                
                
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
    parser.add_argument('--residual_length', type=int, default=32,
                        help='')
    parser.add_argument('--seed', type=int, default=42,
                        help='')
    parser.add_argument('--use_flash', action='store_true', help='')
    parser.add_argument('--quant_kv_output', action='store_true', help='')
    parser.add_argument('--k_quant_per', type=str, default='channel', choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_per', type=str, default='token', choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--e', action='store_true', help='')

    cfgs = parser.parse_args()
    main(cfgs)
