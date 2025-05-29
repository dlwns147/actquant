# LLaMA model with KIVI
import warnings
import os
warnings.filterwarnings("ignore")
import torch
import random
from models.llama_autogptq import LlamaForCausalLM_KIVI, LlamaAttention_KIVI
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from transformers import LlamaConfig, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from utils.eval import *
from utils.data import *
from models.quantization import gptqv2_utils

class KIVILlamaConfig(AutoConfig):    
    @classmethod
    def from_pretrained(cls, model_name_or_path, args=None):
        config = super(KIVILlamaConfig, cls).from_pretrained(model_name_or_path)

        if args is not None:
            for key, value in vars(args).items():
                setattr(config, key, value)
                    
        return config

# For reproducibility
random.seed(0)
torch.manual_seed(0)

def replace_model(model, config):
    if isinstance(model, LlamaForCausalLM):
        model = LlamaForCausalLM_KIVI(model, config)
        layers = model.model.layers
        for i in range(len(layers)):
            if type(layers[i]) == LlamaDecoderLayer:
                layers[i].self_attn = LlamaAttention_KIVI(layers[i].self_attn, config)
    return model


def main(args):
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.load_qmodel_path is None:
        model_id = os.path.join(args.model_path, args.model_name)

        config = KIVILlamaConfig.from_pretrained(model_id, args=args)
        config.use_flash = True

        model = LlamaForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=model_id,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype='auto',
        )
        model.eval()
        model.seqlen = args.seqlen
        model = model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
        
        trainloader = get_loaders(name=args.cal_dataset, nsamples=args.nsamples,
                                seed=0, seqlen=args.seqlen,
                                tokenizer=tokenizer, model=model_id)[0]
        
        quantizers = gptqv2_utils.gptqv2_fwrd(model, trainloader, 'cuda', args)
        
        if args.save_qmodel_path is not None:
            model.save_quantized(args.save_qmodel_path)
    
    else:
        model = LlamaForCausalLM_KIVI.from_quantized(
            save_dir_or_hub = args.load_qmodel_path
        )
        model.eval()
        model.seqlen = args.seqlen
        
        model_id = model.config._name_or_path
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    
    model = model.to(device)
    
    if args.eval_ppl:
        testloaders = {dataset: get_loaders(dataset, tokenizer=tokenizer, model=model_id, seqlen=args.seqlen)[1] for dataset in args.datasets}
        print('Loading datasets')
        use_cache = model.config.use_cache
        model.config.use_cache = False
        for dataset in args.datasets:
            ppl = eval_ppl(model, testloaders[dataset], device=device, seqlen=args.seqlen)
            print(f'{dataset} : {ppl}')
        model.config.use_cache = use_cache
    
    if args.zeroshot:
        from common_code.eval import eval_zeroshot
        
        results = eval_zeroshot(model, tokenizer, task_list = args.tasks, batch_size='auto')

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        print(metric_vals)
        
        
supported_datasets = ['wikitext2', 'ptb', 'c4']


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
    parser.add_argument('--quant_kv_output', action='store_true', help='')
    parser.add_argument('--k_quant_per', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--v_quant_per', type=str, choices=['channel', 'token'], 
                        help='')
    parser.add_argument('--zeroshot', action='store_true', help='')
    parser.add_argument('--tasks', type=str, nargs='+', default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai", "boolq", "openbookqa", "social_iqa"])
    parser.add_argument('--zeroshot_batch_size', type=int, default=64,
                        help='')
    
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ(v2)')
    parser.add_argument('--static_groups', action=argparse.BooleanOptionalAction, default=False,
                        help='static groups in GPTQ(v2)')
    
    parser.add_argument('--use_arch', action=argparse.BooleanOptionalAction, default=False,
                        help='use architecture')
    parser.add_argument('--fake_kv_cache', action='store_true',
                        help='Fake KV cache for the model.')
    parser.add_argument('--save_qmodel_path', type=str, default=None,
                        help='Save the quantized model to the specified path!')
    parser.add_argument('--load_qmodel_path', type=str, default=None,
                        help='Load the quantized model from the specified path!')

    cfgs = parser.parse_args()
    main(cfgs)