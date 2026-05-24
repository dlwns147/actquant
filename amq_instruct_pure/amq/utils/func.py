import gc
import random
from datetime import timedelta

import torch
import numpy as np
import scipy.stats as stats

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import set_seed as set_seed_transformers
from accelerate import Accelerator, InitProcessGroupKwargs

from utils.dispatch import simple_dispatch_model
from hqq.models.hf.base import AutoHQQHFModel

def clean_up():
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    set_seed_transformers(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def getsubattr(obj, attr):
    attrs = attr.split('.')
    if len(attrs) > 1:
        return getsubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]))
    else:
        return getattr(obj, attr)
    
def setsubattr(obj, attr, value):
    attrs = attr.split('.')
    if len(attrs) > 1:
        setsubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]), value)
    else :
        setattr(obj, attr, value)

def delsubattr(obj, attr):
    attrs = attr.split('.')
    if len(attrs) > 1:
        return delsubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]))
    else:
        return delattr(obj, attr)
    
def hassubattr(obj, attr):
    attrs = attr.split('.')
    if len(attrs) > 1:
        return hassubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]))
    else:
        return hasattr(obj, attr)

def getblock(model, config):
    return getsubattr(model, config['layers'])

def get_correlation(prediction, target):
    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau

def build_device_map(gpu_id, config):
    """Single-process device_map across local GPUs ``0..len(gpu_id)-1``.

    Mirrors what ``init_accelerator`` produces when ``num_processes=1``,
    without instantiating an ``Accelerator`` (which the post-search
    pipelines don't otherwise use).
    """
    n_gpu = len(gpu_id)
    n_block = int(config['n_block'])
    if n_block % n_gpu != 0:
        raise AssertionError(f'n_block {n_block} is not divisible by {n_gpu}')

    blk_per_gpu = n_block // n_gpu
    cur_gpu_id = list(range(n_gpu))

    device_map = dict()
    for pre_layer in config['pre_layer']:
        device_map[pre_layer] = cur_gpu_id[0]
    for layer_idx in range(n_block):
        device_map[f"{config['layers']}.{layer_idx}"] = cur_gpu_id[layer_idx // blk_per_gpu]
    for post_layer in config['post_layer']:
        device_map[post_layer] = cur_gpu_id[-1]

    return device_map


def init_accelerator(gpu_id, config):
    ipg_handler = InitProcessGroupKwargs(
            timeout=timedelta(seconds=5400)
            )

    accelerator = Accelerator(kwargs_handlers=[ipg_handler])
    n_proc = accelerator.num_processes
    assert len(gpu_id) % n_proc == 0, 'Total number of gpus (args.gpu_id) should be divisible by num_processes'

    gpu_start_idx = accelerator.device.index if accelerator.device.index is not None else 0
    
    gpu_per_proc = len(gpu_id) // n_proc
    n_block = int(config['n_block'])
    assert n_block % gpu_per_proc == 0, f'n_block {n_block} is not divisible by {gpu_per_proc}'

    blk_per_gpu = n_block // gpu_per_proc
    cur_gpu_id = list(range(gpu_start_idx, len(gpu_id), n_proc))

    device_map = dict()
    for pre_layer in config['pre_layer']:
        device_map[pre_layer] = cur_gpu_id[0]

    for layer_idx in range(n_block):
        device_map[f"{config['layers']}.{layer_idx}"] = cur_gpu_id[layer_idx // blk_per_gpu]
            
    for post_layer in config['post_layer']:
        device_map[post_layer] = cur_gpu_id[-1]

    return accelerator, device_map


def get_bits_usage(architecture, config, group_size=128):
    bits_usage = 0
    memory_usage = 0

    for linear_group, bits in architecture['linear'].items():
        for block_idx, bit in enumerate(bits):
            out_dim, in_dim = config['linear_shape'][linear_group]
            c_group_size = in_dim if group_size == -1 else group_size
            bit += 32 / c_group_size if bit < 16 else 0
            memory_usage += int(out_dim) * int(in_dim) * bit
                
    bits_usage = memory_usage / config['model_numel']
    
    return bits_usage
    

def get_hfmodel(model_name_or_path: str,
                device_map='auto',
                dtype='auto',
                trust_remote_code=False,
                use_cache=False,
                **kwargs
                ):

    clean_up()

    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if getattr(cfg, 'model_type', None) == 'gemma3' and hasattr(cfg, 'text_config'):
        from transformers import Gemma3ForCausalLM
        model_cls = Gemma3ForCausalLM
    else:
        model_cls = AutoModelForCausalLM
    model = model_cls.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        **kwargs
    )
    model.config.use_cache = use_cache
    
    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal

    model.eval()
    
    return model

def get_quantization_proxy(quant_model_paths, device_map, use_cache):
    clean_up()

    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    quantization_proxies = []
    for quant_model_path in quant_model_paths:
        # HQQ defaults compute_dtype=float16. Gemma-3 activations overflow fp16
        # (>65504) → inf → NaN cascade. Derive dtype from the path suffix
        # (e.g. `..._bfloat16` / `..._float16`) which encodes the compute dtype
        # used at quantization time.
        if quant_model_path.endswith('_bfloat16') or '_bfloat16' in quant_model_path.rstrip('/').split('/')[-1]:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        model = AutoHQQHFModel.from_quantized(quant_model_path, device='cuda:0',
                                              compute_dtype=compute_dtype)
        model = simple_dispatch_model(model, device_map)
        
        quantization_proxies.append(model)
        model.config.use_cache = use_cache

        clean_up()

        print(f'{quant_model_path} :  {torch.cuda.max_memory_reserved() / 1024 / 1024}MB')

    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal

    return quantization_proxies

def get_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id)