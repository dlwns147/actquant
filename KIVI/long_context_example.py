# LLaMA model with KIVI
# LLaMA model with KIVI
import warnings
import os
warnings.filterwarnings("ignore")
import torch
import random
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from models.replace import replace_model
import json
import gc

# For reproducibility

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def main(args):

    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_id = os.path.join(args.model_path, args.model_name)

    # config = LlamaConfig.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    config.k_bits = args.k_bits # KiVi currently support 2/4 K/V bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length # corresponding to the number of recent fp16 tokens
    config.use_flash = args.use_flash
    config.quant_kv_output = args.quant_kv_output
    config.k_quant_per = args.k_quant_per
    config.v_quant_per = args.v_quant_per

    device = 'cuda'

    # model = LlamaForCausalLM_KIVI.from_pretrained(
    #     pretrained_model_name_or_path=model_id,
    #     config=config,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     # torch_dtype='auto',
    # )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # torch_dtype='auto',
        attn_implementation='flash_attention_2' if config.use_flash else 'sdpa'
    )
    model = replace_model(model, config)

    model.to(device)
    model.eval()
    enc = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

    file_name = "passkey_examples.jsonl"
    method_name = f"K{config.k_bits}V{config.v_bits} KiVi"
    print("=========="*2 + f"**{method_name}**" + "=========="*2)
    for line in open(file_name, "r"):
        _cleanup()
        torch.cuda.reset_max_memory_allocated()
        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"] + prompt_postfix
        input_ids = enc(prompt, return_tensors="pt").input_ids.cuda()
        print( "-----------------------------------" )
        print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
        print( "Passkey target:", example["target"] )

        tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
        answer = prompt_postfix + enc.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
        answer = answer.replace("\n", "\\n")
        answer= f"{method_name}:\n     [ {answer} ]"
        
        peak_memory = torch.cuda.max_memory_allocated()
        print( answer )
        print(f"Mem: {peak_memory / 1024 / 1024 / 1024:.3f} GB")
        # print(f"Mem: {peak_memory / 1024 / 1024:.3f} MB")
        print( "-----------------------------------\n" )



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
    parser.add_argument('--eval_ppl', action='store_true', help='')
    parser.add_argument('--datasets', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='')
    parser.add_argument('--seed', type=int, default=0,
                        help='')
    parser.add_argument('--use_flash', action='store_true', help='')
    parser.add_argument('--quant_kv_output', action='store_true', help='')
    parser.add_argument('--k_quant_per', type=str, default='channel', choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_per', type=str, default='token', choices=['channel', 'token'], 
                        help='')

    cfgs = parser.parse_args()
    main(cfgs)
