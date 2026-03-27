import numpy as np
from accelerate import Accelerator, InitProcessGroupKwargs
from hqq.models.hf.base import AutoHQQHFModel
from .dispatch import simple_dispatch_model
import scipy.stats as stats
import torch
from transformers import AutoModelForCausalLM
from datetime import timedelta
import gc
from copy import deepcopy
import random
# from hqq.utils.patching_woo import prepare_for_inference

def clean_up():
    torch.cuda.empty_cache()
    gc.collect()
    
def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def get_correlation(prediction, target):
    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau


def compute_bits(arch, config, group_size, target='w', include_pruning=True):
    # arch schema: {'q': {'w': {linear: [bits,...],...}, 'k': [[bits,gs],...], 'v': [...]}, 'p': {'k': [dim,...], 'v': [...]}}
    memory_usage = 0
    if target == 'w':
        n_param = 0
        w_group_size = group_size[target]
        for linear in config['linear']:
            out_dim, in_dim = map(int, config['linear_shape'][linear])
            linear_bits = arch['q']['w'][linear]
            n_param += out_dim * in_dim * len(linear_bits)
            linear_group_size = in_dim if w_group_size == -1 else w_group_size
            assert in_dim % linear_group_size == 0
            if type(linear_bits[0]) == int:
                for bits in linear_bits:
                    memory_usage += out_dim * in_dim * bits
                    if bits < 16:
                        memory_usage += (in_dim // linear_group_size) * out_dim * 32 # scale + zero point
            elif type(linear_bits[0]) == list and len(linear_bits[0]) == 2:
                for bits, n_outlier_column in linear_bits:
                    memory_usage += out_dim * in_dim * bits
                    if bits < 16:
                        memory_usage += (in_dim // linear_group_size) * out_dim * 32 # scale + zero point
                    memory_usage += out_dim * n_outlier_column * 16
            else:
                raise NotImplementedError(type(linear_bits[0]))
        return memory_usage / n_param

    elif target == 'k' or target == 'v':
        kv_list = arch['q'][target]
        prune_list = arch.get('p', {}).get(target, [0] * len(kv_list))
        head_dim = int(config['head_dim'])
        bits_list = []
        for (bits, gs), prune_dim in zip(kv_list, prune_list):
            scale_overhead = (32 / gs) if gs != 0 else 0
            b = bits + scale_overhead
            if include_pruning:
                b *= (1 - prune_dim / head_dim)
            bits_list.append(b)
        return np.mean(bits_list).item()

    elif target == 'kv':
        k_list = arch['q']['k']
        v_list = arch['q']['v']
        p_arch = arch.get('p', {})
        k_prune_list = p_arch.get('k', [0] * len(k_list))
        v_prune_list = p_arch.get('v', [0] * len(v_list))
        head_dim = int(config['head_dim'])
        bits_list = []
        for (bits, gs), prune_dim in zip(k_list + v_list, k_prune_list + v_prune_list):
            scale_overhead = (32 / gs) if gs != 0 else 0
            b = bits + scale_overhead
            if include_pruning:
                b *= (1 - prune_dim / head_dim)
            bits_list.append(b)
        return np.mean(bits_list).item()

    else:
        raise NotImplementedError

def compute_memory(arch, config, group_size, n_token=0, residual_length=0):
    # arch schema: {'q': {'w': {linear: [bits,...],...}, 'k': [[bits,gs],...], 'v': [...]}, 'p': {'k': [dim,...], 'v': [...]}}
    w_group_size = group_size['w']
    weight_memory = 0
    for linear in config['linear']:
        out_dim, in_dim = map(int, config['linear_shape'][linear])
        linear_group_size = in_dim if w_group_size == -1 else w_group_size
        assert in_dim % linear_group_size == 0
        linear_bits = arch['q']['w'][linear]
        if type(linear_bits[0]) == int:
            for bits in linear_bits:
                weight_memory += out_dim * in_dim * bits // 8
                if bits < 16:
                    weight_memory += (in_dim // linear_group_size) * out_dim * 4 # scale + zero point
        elif type(linear_bits[0]) == list and len(linear_bits[0]) == 2:
            for bits, n_outlier_column in linear_bits:
                weight_memory += out_dim * in_dim * bits // 8
                if bits < 16:
                    weight_memory += (in_dim // linear_group_size) * out_dim * 4 # scale + zero point
                weight_memory += out_dim * n_outlier_column * 2
        else:
            raise NotImplementedError(type(linear_bits[0]))

    weight_memory += int(config['vocab_size']) * int(config['hidden_size']) * 4 # lm_head + embed_token
    weight_memory += int(config['n_norm']) * int(config['hidden_size']) * 2 # norms
    weight_memory += int(config['max_position_embeddings']) * int(config['head_dim']) * 2 # positional embedding

    head_dim = int(config['head_dim'])
    p_arch = arch.get('p', {})
    cache_memory = 0
    for target in ['k', 'v']:
        kv_dim = int(config['linear_shape'][config[f'{target}_linear']][0])
        n_kv_heads = kv_dim // head_dim
        prune_list = p_arch.get(target, [0] * len(arch['q'][target]))
        for (bits, gs), prune_dim in zip(arch['q'][target], prune_list):
            effective_kv_dim = n_kv_heads * (head_dim - prune_dim)
            layer_mem = effective_kv_dim * bits / 8
            if bits < 16:
                layer_mem += (effective_kv_dim / gs) * 4 # scale + zero point
            cache_memory += layer_mem * n_token

    return weight_memory + cache_memory


def compute_sparsity(arch):
    return np.concatenate([v for v in arch['layer'].values()]).mean()

def compute_params(arch, config):
    params = 0
    total_params = 0
    for layer, layer_arch in arch['layer'].items():
        for layer_mask in layer_arch:
            total_params += config['layer_numel'][layer]
            params += config['layer_numel'][layer] * layer_mask
            
    return params / total_params

def get_net_info(arch, config, group_size, n_token=0, residual_length=0):
    # arch schema: {'q': {'w': {linear: [bits,...],...}, 'k': [[bits,gs],...], 'v': [...]}, 'p': {'k': [dim,...], 'v': [...]}}
    q_arch = arch.get('q', {})
    p_arch = arch.get('p', {})
    net_info = {}
    net_info['wbits']      = compute_bits(arch=arch, config=config, group_size=group_size, target='w')  if 'w' in q_arch else 0
    net_info['kvbits']     = compute_bits(arch=arch, config=config, group_size=group_size, target='kv', include_pruning=False) if 'k' in q_arch and 'v' in q_arch else 0
    net_info['kbits']      = compute_bits(arch=arch, config=config, group_size=group_size, target='k',  include_pruning=False) if 'k' in q_arch else 0
    net_info['vbits']      = compute_bits(arch=arch, config=config, group_size=group_size, target='v',  include_pruning=False) if 'v' in q_arch else 0
    head_dim = int(config['head_dim'])
    kv_dims = [head_dim - d for d in p_arch.get('k', [])] + [head_dim - d for d in p_arch.get('v', [])]
    net_info['kvdim'] = float(np.mean(kv_dims)) if kv_dims else float(head_dim)
    net_info['kdim']  = float(np.mean([head_dim - d for d in p_arch['k']])) if 'k' in p_arch else float(head_dim)
    net_info['vdim']  = float(np.mean([head_dim - d for d in p_arch['v']])) if 'v' in p_arch else float(head_dim)
    net_info['eff_kvbits'] = compute_bits(arch=arch, config=config, group_size=group_size, target='kv', include_pruning=True)  if 'k' in q_arch and 'v' in q_arch else 0
    net_info['eff_kbits']  = compute_bits(arch=arch, config=config, group_size=group_size, target='k',  include_pruning=True)  if 'k' in q_arch else 0
    net_info['eff_vbits']  = compute_bits(arch=arch, config=config, group_size=group_size, target='v',  include_pruning=True)  if 'v' in q_arch else 0
    net_info['memory']     = compute_memory(arch=arch, config=config, group_size=group_size, n_token=n_token, residual_length=residual_length) if 'w' in q_arch and 'k' in q_arch and 'v' in q_arch else 0
    net_info['n_token'] = n_token
    return net_info

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


def init_accelerator(gpu_id, config):
    gpu_id = gpu_id.split(',')

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

    # print(f'cur_gpu_ids : {cur_gpu_id}, blk_per_gpu : {blk_per_gpu}, device : {accelerator.device}, device_map : {device_map}')
    # print(f'device_map : {device_map}')

    return accelerator, device_map

def load_hqq_model(model_id,
                   device_map,
                   use_cache=False,
                   attn_implementation='flash_attention_2',
                   inference=False):

    # # for fast model loading
    # org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    # org_uniform = torch.nn.init.uniform_
    # org_normal = torch.nn.init.normal_
    # def skip(*args, **kwargs):
    #     pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip

    model = None
    if model_id is not None:
        model = AutoHQQHFModel.from_quantized(model_id, device_map='cpu', attn_implementation=attn_implementation)
        model = simple_dispatch_model(model, device_map)
        model.config.use_cache = use_cache
        clean_up()
        print(f'{model_id} : {torch.cuda.max_memory_reserved() / 1024 / 1024}MB')
        # if inference:
        #     prepare_for_inference(model, backend='gptq')

    # torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    # torch.nn.init.uniform_ = org_uniform
    # torch.nn.init.normal_ = org_normal

    return model

def insert_fp16_channel_hqq(linear, outlier):
    # import pdb; pdb.set_trace()
    linear.meta['outlier'] = outlier

def remove_fp16_channel_hqq(linear):
    if 'outlier' in linear.meta:
        del linear.meta['outlier']

def get_fp16_channel(linear, idx):
    # print(f'linear.weight : {linear.weight.data.device}, idx : {idx}')
    return deepcopy(linear.weight.data[:, idx])
    # return linear.weight.data[:, idx]

def get_outlier_bits(config):
    pass


def get_hfmodel(model_name_or_path: str,
                device_map='auto',
                dtype='auto',
                trust_remote_code=False,
                use_cache=False,
                attn_implementation='flash_attention_2',
                **kwargs):

    # assert kwargs.get('attn_implementation') in ['hf', 'ft']        ## hf : huggingface, ft : faster transformer
    
    # # for fast model loading
    # org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    # org_uniform = torch.nn.init.uniform_
    # org_normal = torch.nn.init.normal_

    # def skip(*args, **kwargs):
    #     pass

    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    
    # ft = False
    # if kwargs.get('attn_implementation') == 'ft':
    #     assert 'llama' in model_name_or_path.lower() or 'vicuna' in model_name_or_path.lower()
    #     ft = True
    
    # print('attention implementaion is :', kwargs.pop('attn_implementation'))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=dtype,
        device_map=device_map, 
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        use_cache=use_cache,
        attn_implementation=attn_implementation,
        **kwargs
    )
    model.config.use_cache = use_cache
    
    # if ft:
    #     convert_model_to_ft(model)
    #     replace_generate_functions()

    # torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    # torch.nn.init.uniform_ = org_uniform
    # torch.nn.init.normal_ = org_normal
    
    return model

def load_outlier(model, outlier, config):
    outlier = dict()
    for blk_idx in range(int(config['n_block'])):
        # for linear_group in config['linear']:
        #     for linear in linear_group.split(','):
        for linear in config['linear']:
            key = f'{config["layers"]}.{blk_idx}.{linear}'
            if key in outlier:
                outlier[f'{blk_idx}.{linear}'] = [outlier[key], get_fp16_channel(getsubattr(getblock(model, config)[blk_idx], linear), outlier[key])]
    return outlier

def process_dtype(dtype):
    if dtype in ['float16', 'float', 'fp16']:
        return torch.float16
    elif dtype in ['bfloat16', 'bfloat', 'bf16']:
        return torch.bfloat16
    elif dtype == 'auto':
        return 'auto'
    else:
        raise NotImplementedError(dtype)