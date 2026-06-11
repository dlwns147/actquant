import os
import json
import numpy as np
import scipy.stats as stats
import torch
from datetime import timedelta
# NOTE: accelerate / transformers / hqq are imported lazily inside the few
# functions that need them (init_accelerator / load_hqq_model / get_hfmodel).
# func.py is imported by nearly every entry point, and eagerly importing
# accelerate→transformers→hqq cost ~77s on this filesystem (transformers
# version-metadata scan alone ~59s); the numpy-only combo/sampling paths
# (build_nd, load_expr, ...) must not pay that.
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

# ThinK residual: the original ThinK_kivi keeps the newest 32 tokens at full
# head_dim and only channel-prunes everything older (ThinK_orig/ThinK_kivi
# example.py uses residual_length=32). This is independent of the KV-quant
# residual_length and is fixed at 32 ("원본대로").
THINK_RESIDUAL = 32


def compute_memory(arch, config, group_size, n_token=0, residual_length=0,
                   think_residual=THINK_RESIDUAL, sink=0):
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

    # Three token partitions over the n_token positions (oldest=0 … newest):
    #   * KV-quant residual: the newest `residual_length` tokens stay fp16
    #     (16-bit, no scale); older tokens are quantized to `bits` (+scale).
    #   * ThinK residual: when `prune_dim > 0`, only the newest
    #     `think_residual` (=32, 원본대로) tokens keep full head_dim; older
    #     tokens drop `prune_dim` channels.
    #   * Attention-sink (KVSink/KIVI-K2V2*): the OLDEST `sink` tokens are kept
    #     fp16 at FULL head_dim — peeled before quant AND before ThinK pruning,
    #     so they are never quantized and never channel-pruned. (Deployment
    #     rounds `sink` up to a group boundary; the raw value is used here — a
    #     ≤1-group, ~0.2%-mem difference added equally to every arch.)
    # Per position i: fp16 iff i<sink (sink) or i>=n-R (residual); full head_dim
    # iff i<sink (sink) or i>=n-think_residual (ThinK newest) or prune_dim==0.
    # A segment sweep over the boundaries handles every R/think/sink ordering.
    R = max(int(residual_length), 0) if residual_length else 0
    S = max(int(sink), 0)
    cache_memory = 0
    for target in ['k', 'v']:
        kv_dim = int(config['linear_shape'][config[f'{target}_linear']][0])
        n_kv_heads = kv_dim // head_dim
        prune_list = p_arch.get(target, [0] * len(arch['q'][target]))
        for (bits, gs), prune_dim in zip(arch['q'][target], prune_list):
            n = int(n_token)
            r = min(R, n)
            s = min(S, n)
            t = min(int(think_residual), n) if prune_dim > 0 else n  # newest full
            c = {('fp', 'full'): 0, ('fp', 'pru'): 0,
                 ('q', 'full'): 0, ('q', 'pru'): 0}
            bps = sorted({0, s, n - r, n - t, n})
            for a, b in zip(bps, bps[1:]):
                if b <= a:
                    continue
                is_fp = (a < s) or (a >= n - r)
                is_full = (a < s) or (a >= n - t)
                c[('fp' if is_fp else 'q', 'full' if is_full else 'pru')] += b - a

            full_dim   = n_kv_heads * head_dim
            pruned_dim = n_kv_heads * (head_dim - prune_dim)

            def _q_mem(dim):
                m = dim * bits / 8
                if bits < 16:
                    m += (dim / gs) * 4              # scale + zero point
                return m

            cache_memory += c[('fp', 'full')] * full_dim   * 16 / 8
            cache_memory += c[('fp', 'pru')]  * pruned_dim * 16 / 8
            cache_memory += c[('q', 'full')]  * _q_mem(full_dim)
            cache_memory += c[('q', 'pru')]   * _q_mem(pruned_dim)

    return weight_memory + cache_memory


def compute_weight_memory(arch, config, group_size):
    """Return weight (non-KV) memory in bytes. Mirrors the weight_memory part of compute_memory."""
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
                    weight_memory += (in_dim // linear_group_size) * out_dim * 4
        elif type(linear_bits[0]) == list and len(linear_bits[0]) == 2:
            for bits, n_outlier_column in linear_bits:
                weight_memory += out_dim * in_dim * bits // 8
                if bits < 16:
                    weight_memory += (in_dim // linear_group_size) * out_dim * 4
                weight_memory += out_dim * n_outlier_column * 2
        else:
            raise NotImplementedError(type(linear_bits[0]))
    weight_memory += int(config['vocab_size']) * int(config['hidden_size']) * 4
    weight_memory += int(config['n_norm']) * int(config['hidden_size']) * 2
    weight_memory += int(config['max_position_embeddings']) * int(config['head_dim']) * 2
    return weight_memory


def compute_cache_memory_single(arch, config, n_token, sink=0):
    """Return KV cache memory in bytes for a single arch. Mirrors the cache_memory
    part of compute_memory (this simplified model omits the fp16 back-residual).
    Attention-sink: the oldest `sink` tokens are fp16 at FULL head_dim (never
    quantized/pruned); the remaining (n_token - sink) follow the per-layer quant."""
    head_dim = int(config['head_dim'])
    p_arch = arch.get('p', {})
    S = min(max(int(sink), 0), int(n_token))
    n_q = int(n_token) - S
    cache_memory = 0
    for target in ['k', 'v']:
        kv_dim = int(config['linear_shape'][config[f'{target}_linear']][0])
        n_kv_heads = kv_dim // head_dim
        prune_list = p_arch.get(target, [0] * len(arch['q'][target]))
        for (bits, gs), prune_dim in zip(arch['q'][target], prune_list):
            effective_kv_dim = n_kv_heads * (head_dim - prune_dim)
            layer_mem = effective_kv_dim * bits / 8
            if bits < 16:
                layer_mem += (effective_kv_dim / gs) * 4
            cache_memory += layer_mem * n_q
            cache_memory += (n_kv_heads * head_dim) * 16 / 8 * S   # sink: fp16 full-dim
    return cache_memory


def compute_cache_memory_batch(kv_subnets, kvdim_subnets, config, n_token, sink=0):
    """
    Vectorized KV cache memory for all (kv_arch, kvdim_arch) pairs.

    kv_subnets    : list of arch dicts with arch['q']['k/v'] holding (bits, gs) per layer.
    kvdim_subnets : list of arch dicts with arch['p']['k/v'] holding prune_dim per layer.
                    Pass None to assume zero pruning (no kvdim_expr).
    config        : model config dict
    n_token       : number of KV-cache tokens
    sink          : attention-sink window — the oldest `sink` tokens are fp16 at
                    FULL head_dim (never quantized/pruned). The remaining
                    (n_token - sink) follow the per-layer quant.

    Returns np.ndarray
        shape (N_kv, N_kvdim)  when kvdim_subnets is not None
        shape (N_kv,)           when kvdim_subnets is None
    """
    head_dim = int(config['head_dim'])
    n_kv_h = {t: int(config['linear_shape'][config[f'{t}_linear']][0]) // head_dim
              for t in ('k', 'v')}

    bits_k = np.array([[e[0] for e in sv['q']['k']] for sv in kv_subnets], dtype=float)  # (N_kv, L)
    gs_k   = np.array([[e[1] for e in sv['q']['k']] for sv in kv_subnets], dtype=float)
    bits_v = np.array([[e[0] for e in sv['q']['v']] for sv in kv_subnets], dtype=float)
    gs_v   = np.array([[e[1] for e in sv['q']['v']] for sv in kv_subnets], dtype=float)

    S = min(max(int(sink), 0), int(n_token))
    n_q = int(n_token) - S                       # quantized token count
    L = bits_k.shape[1]
    # sink: oldest S tokens fp16 at full head_dim, summed over all L layers (k+v).
    # Constant across archs (global sink) → added to every output entry.
    sink_total = S * 2.0 * L * (n_kv_h['k'] * head_dim + n_kv_h['v'] * head_dim)

    if kvdim_subnets is not None:
        prune_k = np.array([sv['p']['k'] for sv in kvdim_subnets], dtype=float)  # (N_kvdim, L)
        prune_v = np.array([sv['p']['v'] for sv in kvdim_subnets], dtype=float)

        eff_k = n_kv_h['k'] * (head_dim - prune_k)  # (N_kvdim, L)
        eff_v = n_kv_h['v'] * (head_dim - prune_v)

        # broadcast to (N_kv, N_kvdim, L)
        k_mem   = bits_k[:, None, :] * eff_k[None, :, :] / 8.0 * n_q
        k_scale = np.where(bits_k[:, None, :] < 16,
                           eff_k[None, :, :] / gs_k[:, None, :] * 4.0 * n_q, 0.0)
        v_mem   = bits_v[:, None, :] * eff_v[None, :, :] / 8.0 * n_q
        v_scale = np.where(bits_v[:, None, :] < 16,
                           eff_v[None, :, :] / gs_v[:, None, :] * 4.0 * n_q, 0.0)
        return (k_mem + k_scale + v_mem + v_scale).sum(axis=-1) + sink_total  # (N_kv, N_kvdim)

    else:
        # zero pruning → effective dim is constant per target
        eff_k = float(n_kv_h['k'] * head_dim)
        eff_v = float(n_kv_h['v'] * head_dim)

        k_mem   = bits_k * eff_k / 8.0 * n_q                                    # (N_kv, L)
        k_scale = np.where(bits_k < 16, eff_k / gs_k * 4.0 * n_q, 0.0)
        v_mem   = bits_v * eff_v / 8.0 * n_q
        v_scale = np.where(bits_v < 16, eff_v / gs_v * 4.0 * n_q, 0.0)
        return (k_mem + k_scale + v_mem + v_scale).sum(axis=-1) + sink_total  # (N_kv,)


def compute_eff_kvbits_batch(kv_subnets, kvdim_subnets, config, target='kv'):
    """
    Vectorized effective KV bits for all (kv_arch, kvdim_arch) pairs.
    Mirrors compute_bits(..., include_pruning=True).

    target : 'kv' → eff_kvbits (mean over k+v layers)
             'k'  → eff_kbits
             'v'  → eff_vbits

    Returns np.ndarray shape (N_kv, N_kvdim)
    """
    head_dim = float(config['head_dim'])
    targets = ['k', 'v'] if target == 'kv' else [target]
    per_target = []
    for t in targets:
        bits  = np.array([[e[0] for e in sv['q'][t]] for sv in kv_subnets], dtype=float)  # (N_kv, L)
        gs    = np.array([[e[1] for e in sv['q'][t]] for sv in kv_subnets], dtype=float)
        prune = np.array([sv['p'][t] for sv in kvdim_subnets], dtype=float)               # (N_kvdim, L)

        scale       = np.where(gs != 0, 32.0 / gs, 0.0)        # (N_kv, L)
        bits_scaled = bits + scale
        eff_factor  = 1.0 - prune / head_dim                    # (N_kvdim, L)

        b = bits_scaled[:, None, :] * eff_factor[None, :, :]   # (N_kv, N_kvdim, L)
        per_target.append(b)

    # mean over all layers (k+v concatenated), matching compute_bits behaviour
    return np.concatenate(per_target, axis=-1).mean(axis=-1)    # (N_kv, N_kvdim)


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

def get_net_info(arch, config, group_size, n_token=0, residual_length=0, attn_sink=0):
    # arch schema: {'q': {'w': {linear: [bits,...],...}, 'k': [[bits,gs],...], 'v': [...]}, 'p': {'k': [dim,...], 'v': [...]}}
    # Also accepts legacy format: {'w': {linear: [bits,...],...}, 'k': [...], 'v': [...]}
    if 'q' in arch:
        q_arch = arch['q']
        p_arch = arch.get('p', {})
    elif 'w' in arch or 'k' in arch or 'v' in arch:
        # Legacy flat format — treat arch itself as q_arch
        q_arch = arch
        p_arch = {}
        arch = {'q': arch}   # rebuild for compute_bits/compute_memory compatibility
    else:
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
    net_info['memory']     = compute_memory(arch=arch, config=config, group_size=group_size, n_token=n_token, residual_length=residual_length, sink=attn_sink) if 'w' in q_arch and 'k' in q_arch and 'v' in q_arch else 0
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
    from accelerate import Accelerator, InitProcessGroupKwargs
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
                   attn_implementation='sdpa',
                   inference=False,
                   compute_dtype=None):
    """Load a pre-quantised HQQ model.

    HQQ's `AutoHQQHFModel.from_quantized` hard-defaults to
    `compute_dtype=torch.float16` regardless of what the saved
    `config.json` says — passing a bf16-baked dir without an explicit
    override silently downcasts every fp param to fp16, which then
    cascades into KIVI (cache scale/min follow model dtype). To match
    the path's stored `torch_dtype` (e.g. `_bfloat16` dirs), this
    wrapper:
      • when `compute_dtype` is given (a torch.dtype), uses it directly,
      • otherwise reads `config.json` and uses its `torch_dtype`,
      • falling back to fp16 if neither is available.
    """
    from hqq.models.hf.base import AutoHQQHFModel
    from .dispatch import simple_dispatch_model

    model = None
    if model_id is not None:
        if not isinstance(compute_dtype, torch.dtype):
            cfg_path = os.path.join(model_id, "config.json")
            stored = None
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    stored = json.load(f).get("torch_dtype")
            compute_dtype = {
                "float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }.get(stored, torch.float16)
        model = AutoHQQHFModel.from_quantized(
            model_id, device_map='cpu',
            attn_implementation=attn_implementation,
            compute_dtype=compute_dtype,
        )
        model = simple_dispatch_model(model, device_map)
        model.config.use_cache = use_cache
        clean_up()
        print(f'{model_id} (compute_dtype={compute_dtype}) : '
              f'{torch.cuda.max_memory_reserved() / 1024 / 1024}MB')
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
                attn_implementation='sdpa',
                **kwargs):
    from transformers import AutoModelForCausalLM

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


# ══════════════════════════════════════════════════════════════════════════
# Two-stage post-search shared core (sample_surrogate.py + post_search.py)
#
# Only logic both entry points genuinely need lives here:
#   init_run        seed + config + accelerator + default arch
#   build_expr_map  load the per-axis search archives
#   build_nd        vectorised combo space (metric ND + comp_obj ND)
#   comp_key_order  the get_net_info key order (results.csv complexity rows)
#   evaluate_metric one calibration-set metric eval (+ KV-cache toggle)
#
# results.csv row layout (sample_surrogate writes, post_search reads), matching
# analysis/v5/_common.py:
#   rows 0 .. n_comp-1       complexity (comp_key_order; n_comp == 12)
#   rows n_comp .. +n_ds-1   measured metric, one row per --datasets entry
#   row  n_comp+n_ds         combined predicted metric (pf column 0)
#   rows n_comp+n_ds+1 ..    per-axis search metric, order == expr_keys
# ══════════════════════════════════════════════════════════════════════════
class RunCtx:
    """Shared per-run setup: config + accelerator + the default arch."""
    def __init__(self, config, accelerator, device_map, dtype, group_size,
                 default_arch, n_block):
        self.config = config
        self.accelerator = accelerator
        self.device_map = device_map
        self.dtype = dtype
        self.group_size = group_size
        self.default_arch = default_arch
        self.n_block = n_block


def init_run(args):
    set_seed(args.seed)
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    dtype = process_dtype(args.dtype)

    group_size = {'w': args.w_group_size, 'k': args.k_group_size,
                  'v': args.v_group_size}
    n_block = config['n_block']
    w_linears = config['linear']
    default_w_bits = max(args.w_bits) if args.w_bits else 16
    default_k_bits = max(args.k_bits) if args.k_bits else 4
    default_v_bits = max(args.v_bits) if args.v_bits else 4
    k_gs = args.k_group_size if isinstance(args.k_group_size, int) else min(args.k_group_size)
    v_gs = args.v_group_size if isinstance(args.v_group_size, int) else min(args.v_group_size)
    default_arch = {
        'q': {
            'w': {linear: [default_w_bits] * n_block for linear in w_linears},
            'k': [[default_k_bits, k_gs]] * n_block,
            'v': [[default_v_bits, v_gs]] * n_block,
        },
        'p': {'k': [0] * n_block, 'v': [0] * n_block},
    }
    return RunCtx(config, accelerator, device_map, dtype, group_size,
                  default_arch, n_block)


def _adapt_to_n_block(value, n):
    """Truncate/pad per-layer lists so a source-model subnet fits target
    n_block. Structural workaround only — choices were tuned for the source
    model's sensitivity, not the target's (load_expr warns)."""
    if isinstance(value, dict):
        return {k: _adapt_to_n_block(v, n) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) == 0 or not isinstance(value[0], (int, float, list)):
            return value
        if len(value) == n:
            return value
        if len(value) > n:
            return value[:n]
        return list(value) + [value[-1]] * (n - len(value))
    return value


def _first_list_len(s):
    if isinstance(s, dict):
        for vv in s.values():
            r = _first_list_len(vv)
            if r is not None:
                return r
    elif isinstance(s, list) and s and not isinstance(s[0], dict):
        return len(s)
    return None


def load_expr(expr_path, comp_obj_key, config, group_size, n_block, expr_front):
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    with open(expr_path, 'r') as f:
        result_json = json.load(f)
    archive = result_json['archive'] + result_json['candidates']
    raw_subnets = [v[0] for v in archive]
    if raw_subnets:
        src_n = _first_list_len(raw_subnets[0])
        if src_n is not None and src_n != n_block:
            print(f"[load_expr] WARNING: {os.path.basename(expr_path)} subnets have "
                  f"{src_n} layers but target config has {n_block}; auto-truncating/"
                  f"padding. Per-layer choices were tuned for the source model's "
                  f"sensitivity, not the target's — NOT comparable to a native search.")
        raw_subnets = [_adapt_to_n_block(s, n_block) for s in raw_subnets]
    subnets_arr = np.array(raw_subnets)
    metric_vals = [v[1] for v in archive]
    # The comp column F[:,1] is ONLY consumed by the expr_front Pareto sort
    # below. Both build_nd paths recompute comp themselves (dense: comp_nd_list
    # from esm; lazy: _build_lazy_comp), and nothing reads efm[:,1] when
    # --expr_front is off. So skip the per-subnet get_net_info (≈17k calls,
    # each computing every net_info key incl. compute_memory — the dominant
    # load_expr cost) unless it's actually needed.
    if expr_front:
        comp_vals = [get_net_info(n, config, group_size)[comp_obj_key]
                     for n in subnets_arr]
    else:
        comp_vals = np.zeros(len(metric_vals))   # unused placeholder
    sort_idx = np.argsort(metric_vals)
    F = np.column_stack((metric_vals, comp_vals))[sort_idx]
    subnets_arr = subnets_arr[sort_idx]
    if expr_front:
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F = F[front]
        subnets_arr = subnets_arr[front]
    return subnets_arr, F


def build_expr_map(args, ctx):
    """Load each provided --*_expr archive. Returns ordered {key: (sv, fv)}."""
    expr_map = {}
    spec = [('w', args.w_expr, 'wbits'),
            ('kv', args.kv_expr, 'kvbits'),
            ('kvdim', args.kvdim_expr, 'kvdim'),
            ('eff_kv', args.eff_kv_expr, 'eff_kvbits')]
    for key, path, comp_key in spec:
        if path:
            expr_map[key] = load_expr(path, comp_key, ctx.config, ctx.group_size,
                                      ctx.n_block, args.expr_front)
    assert len(expr_map) >= 1, ("At least one of --w_expr, --kv_expr, "
                                "--kvdim_expr, --eff_kv_expr must be provided")
    return expr_map


class NDCombo:
    """The vectorised combo space: per-axis subnet/metric arrays + comp_obj ND
    arrays + the additive combined-metric ND array (args scales)."""
    def __init__(self, expr_keys, esm, efm, nd_shape, new_metric_nd,
                 comp_nd_list, metric_1d, lazy=False):
        self.expr_keys = expr_keys
        self.esm = esm          # {key: subnet array}
        self.efm = efm          # {key: (metric, comp) F array}
        self.nd_shape = nd_shape
        self.new_metric_nd = new_metric_nd      # additive, args scales
        self.comp_nd_list = comp_nd_list
        self.metric_1d = metric_1d              # [per-axis raw metric], order==expr_keys
        # lazy=True (no --expr_front, huge product): new_metric_nd is None and
        # comp_nd_list is a _LazyComp; comp_obj filtering happens BEFORE any
        # nd_shape array is materialised (select_valid_nd_idx / assemble_F
        # branch on the _LazyComp). F[:,0] is the additive combined metric
        # (== dense new_metric_nd) so it ranks with OR without a surrogate.
        self.lazy = lazy

    @property
    def n_total(self):
        return int(np.prod(self.nd_shape))


# Single-axis comp_obj → the axis name it depends on (eff_kv collapses kv+kvdim).
_SINGLE_AXIS_COMP = {
    'wbits': 'w',
    'kvbits': 'kv', 'kbits': 'kv', 'vbits': 'kv',
    'kvdim': 'kvdim', 'kdim': 'kvdim', 'vdim': 'kvdim',
}


class _LazyComp:
    """Low-rank comp_obj descriptors for the no--expr_front (huge product)
    path. Holds, per comp_obj (order == args.comp_obj), the *pre-broadcast*
    source array build_nd would otherwise broadcast to nd_shape:

        {'kind':'1d','axis':a,'vals':(n_a,)}                  single-axis obj
        {'kind':'2d','axes':(a,b),'vals':(n_a,n_b)}           eff_kvbits etc.
        {'kind':'memory','w_axis','w_mem','w_const',          additive memory
                 'kv':{'kind':'scalar'|'1d'|'2d',...}}

    No nd_shape array is ever built. select_valid_nd_idx / assemble_F detect
    this type and filter/gather sparsely. __getitem__(i) returns the
    achievable [lo, hi] for comp_obj i so post_search's 0-candidate
    diagnostic (`np.asarray(nd.comp_nd_list[i]).min()/.max()`) works
    unchanged."""

    def __init__(self, expr_keys, efm, comp_specs, comp_obj, metric_1d=None):
        self.expr_keys = list(expr_keys)
        self.efm = efm
        self.comp_specs = comp_specs
        self.comp_obj = list(comp_obj)
        self.nd_shape = tuple(len(efm[k]) for k in self.expr_keys)
        # per-axis scale·(sqrt?) JSD, order==expr_keys; the additive combined
        # metric (== dense new_metric_nd) so the no-surrogate fallback ranks
        # correctly. None → treated as zeros (back-compat for unit tests).
        self.metric_1d = metric_1d

    def combined_metric(self, nd_idx):
        if not self.metric_1d:
            return np.zeros(len(nd_idx), dtype=np.float64)
        out = np.zeros(len(nd_idx), dtype=np.float64)
        for ax in range(len(self.expr_keys)):
            out += np.asarray(self.metric_1d[ax], np.float64)[nd_idx[:, ax]]
        return out

    def spec_range(self, spec):
        if spec['kind'] in ('1d', '2d'):
            v = np.asarray(spec['vals'], dtype=np.float64)
            return float(v.min()), float(v.max())
        kv = np.asarray(spec['kv']['vals'], dtype=np.float64)
        if spec['w_axis'] is None:
            return spec['w_const'] + float(kv.min()), \
                   spec['w_const'] + float(kv.max())
        w = np.asarray(spec['w_mem'], dtype=np.float64)
        return float(w.min() + kv.min()), float(w.max() + kv.max())

    def __getitem__(self, i):
        lo, hi = self.spec_range(self.comp_specs[i])
        return np.array([lo, hi], dtype=np.float64)

    def comp_values(self, spec, nd_idx):
        """comp_obj value for each row of nd_idx (M, n_dims)."""
        if spec['kind'] == '1d':
            return np.asarray(spec['vals'])[nd_idx[:, spec['axis']]]
        if spec['kind'] == '2d':
            a0, a1 = spec['axes']
            return np.asarray(spec['vals'])[nd_idx[:, a0], nd_idx[:, a1]]
        # memory: w-part (1D or const) + kv-part (scalar/1d/2d), additive
        v = np.full(len(nd_idx), spec['w_const'], dtype=np.float64)
        if spec['w_axis'] is not None:
            v = np.asarray(spec['w_mem'], np.float64)[nd_idx[:, spec['w_axis']]]
        kv = spec['kv']
        if kv['kind'] == 'scalar':
            return v + kv['vals']
        if kv['kind'] == '1d':
            return v + np.asarray(kv['vals'], np.float64)[nd_idx[:, kv['axis']]]
        a0, a1 = kv['axes']
        return v + np.asarray(kv['vals'], np.float64)[nd_idx[:, a0],
                                                      nd_idx[:, a1]]


def _build_lazy_comp(args, expr_keys, esm, efm, config, group_size,
                     default_arch):
    """Build _LazyComp — mirrors build_nd's comp_nd_list dispatch (the obj
    loop) but keeps the raw pre-broadcast arrays instead of nd_shape views."""
    def ax(name):
        return expr_keys.index(name) if name in expr_keys else None

    specs = []
    for obj in args.comp_obj:
        if obj in _SINGLE_AXIS_COMP:
            if obj == 'wbits':
                axn = 'w'
            elif obj in ('kvbits', 'kbits', 'vbits'):
                axn = 'eff_kv' if 'eff_kv' in expr_keys else 'kv'
            else:
                axn = 'eff_kv' if 'eff_kv' in expr_keys else 'kvdim'
            a = ax(axn)
            if a is None:
                raise SystemExit(f"[build_nd/lazy] comp_obj '{obj}' needs axis "
                                 f"'{axn}' but expr_keys={expr_keys}")
            vals = np.array([get_net_info(sv, config, group_size,
                                          n_token=args.n_token)[obj]
                             for sv in esm[axn]], dtype=np.float64)
            specs.append({'kind': '1d', 'axis': a, 'vals': vals})
        elif obj in ('eff_kvbits', 'eff_kbits', 'eff_vbits'):
            t = {'eff_kvbits': 'kv', 'eff_kbits': 'k', 'eff_vbits': 'v'}[obj]
            if 'eff_kv' in expr_keys:
                a = ax('eff_kv')
                vals = np.array([get_net_info(sv, config, group_size,
                                              n_token=args.n_token)[obj]
                                 for sv in esm['eff_kv']], dtype=np.float64)
                specs.append({'kind': '1d', 'axis': a, 'vals': vals})
            else:
                if 'kv' not in expr_keys or 'kvdim' not in expr_keys:
                    raise SystemExit(f"[build_nd/lazy] '{obj}' needs eff_kv_expr "
                                     f"or both kv_expr and kvdim_expr")
                vals = compute_eff_kvbits_batch(esm['kv'], esm['kvdim'],
                                                config, target=t)
                specs.append({'kind': '2d',
                              'axes': (ax('kv'), ax('kvdim')), 'vals': vals})
        elif obj == 'memory':
            w_ax = ax('w')
            if w_ax is not None:
                w_mem = np.array([compute_weight_memory(sv, config, group_size)
                                  for sv in esm['w']], dtype=np.float64)
                w_const = 0.0
            else:
                w_mem = None
                w_const = float(compute_weight_memory(default_arch, config,
                                                      group_size))
            if 'eff_kv' in expr_keys:
                kvv = np.array([compute_cache_memory_single(sv, config,
                                                            args.n_token,
                                                            sink=getattr(args, 'attn_sink', 0))
                                for sv in esm['eff_kv']], dtype=np.float64)
                kv = {'kind': '1d', 'axis': ax('eff_kv'), 'vals': kvv}
            elif 'kv' in expr_keys and 'kvdim' in expr_keys:
                kvv = compute_cache_memory_batch(esm['kv'], esm['kvdim'],
                                                 config, args.n_token,
                                                 sink=getattr(args, 'attn_sink', 0))
                kv = {'kind': '2d', 'axes': (ax('kv'), ax('kvdim')),
                      'vals': np.asarray(kvv, np.float64)}
            elif 'kv' in expr_keys:
                kvv = compute_cache_memory_batch(esm['kv'], None, config,
                                                 args.n_token,
                                                 sink=getattr(args, 'attn_sink', 0))
                kv = {'kind': '1d', 'axis': ax('kv'),
                      'vals': np.asarray(kvv, np.float64)}
            elif 'kvdim' in expr_keys:
                kvv = compute_cache_memory_batch([default_arch], esm['kvdim'],
                                                 config, args.n_token,
                                                 sink=getattr(args, 'attn_sink', 0))[0]
                kv = {'kind': '1d', 'axis': ax('kvdim'),
                      'vals': np.asarray(kvv, np.float64)}
            else:
                kv = {'kind': 'scalar', 'vals': 0.0}
            specs.append({'kind': 'memory', 'w_axis': w_ax, 'w_mem': w_mem,
                          'w_const': w_const, 'kv': kv})
        else:
            raise SystemExit(
                f"[build_nd/lazy] comp_obj='{obj}' unsupported. Supported: "
                f"{sorted(_SINGLE_AXIS_COMP) + ['eff_kvbits','eff_kbits','eff_vbits','memory']}")
    # additive combined metric pieces == dense build_nd (scale applied AFTER
    # sqrt): used as F[:,0] so post_search's no-surrogate fallback ranks by
    # Σ scale·JSD, and as a pre-surrogate order otherwise.
    _sc = {'w': args.w_scale, 'kv': args.kv_scale,
           'kvdim': args.kvdim_scale, 'eff_kv': args.eff_kv_scale}
    metric_1d = []
    for k in expr_keys:
        m = np.asarray(efm[k][:, 0], np.float64)
        if args.sqrt:
            m = np.sqrt(m)
        metric_1d.append(_sc[k] * m)
    return _LazyComp(expr_keys, efm, specs, args.comp_obj, metric_1d)


def build_nd(args, ctx, expr_map):
    """Vectorised combo builder (port of post_search_split.py).

    new_metric_nd uses the args scales (additive); post_search overrides the
    ranking with a fitted surrogate, sample_surrogate uses it for candidate
    ordering / quantile axes.
    """
    expr_keys = list(expr_map.keys())
    scales = {'w': args.w_scale, 'kv': args.kv_scale,
              'kvdim': args.kvdim_scale, 'eff_kv': args.eff_kv_scale}
    _esm = {k: sv for k, (sv, fv) in expr_map.items()}
    _efm = {k: fv for k, (sv, fv) in expr_map.items()}
    nd_shape = tuple(len(_esm[k]) for k in expr_keys)
    n_dims = len(nd_shape)
    # Guard: build_nd materialises full nd_shape arrays (new_metric_nd + comp).
    # An archive loaded without --expr_front keeps every archive+candidate point
    # per axis, so the Cartesian product explodes (e.g. 10450×2828×4173 ≈ 1.2e11
    # → 919 GiB float64). Fail fast with an actionable message instead.
    _n_total = 1
    for _s in nd_shape:
        _n_total *= int(_s)
    _MAX_COMBO = 5e8
    config, group_size, default_arch = ctx.config, ctx.group_size, ctx.default_arch
    if _n_total > _MAX_COMBO:
        # Huge product (typically --expr_front omitted → full archives, no NDS).
        if not args.comp_obj:
            raise SystemExit(
                f"[build_nd] combo space too large: nd_shape={nd_shape} "
                f"(n_total={_n_total:.3e} > {_MAX_COMBO:.0e}) and no comp_obj "
                f"given — the product is unbounded. Pass --comp_obj/"
                f"--comp_obj_min/--comp_obj_max (comp_obj-first pruning), or "
                f"--expr_front to Pareto-filter the per-axis archives.")
        # comp_obj present → lazy comp-pruned mode: NO nd_shape array is built;
        # comp_obj filtering runs first (select_valid_nd_idx), then the
        # surrogate predicts on the filtered set. Dense code below is skipped.
        return NDCombo(
            expr_keys, _esm, _efm, nd_shape, None,
            _build_lazy_comp(args, expr_keys, _esm, _efm, config, group_size,
                             default_arch),
            [], lazy=True)

    def _bcast_1d(arr, ax):
        shape = [1] * n_dims
        shape[ax] = len(arr)
        return np.broadcast_to(arr.reshape(shape), nd_shape)

    def _bcast_2d(arr, ax0, ax1):
        shape = [1] * n_dims
        shape[ax0] = arr.shape[0]
        shape[ax1] = arr.shape[1]
        return np.broadcast_to(arr.reshape(shape), nd_shape)

    metric_1d = [_efm[k][:, 0] for k in expr_keys]
    metric_1d_used = [np.sqrt(m) if args.sqrt else m for m in metric_1d]
    scale_vals = [scales[k] for k in expr_keys]
    new_metric_nd = sum(s * a for s, a in
                        zip(scale_vals, np.ix_(*metric_1d_used)))

    comp_nd_list = []
    for obj in args.comp_obj:
        if obj == 'wbits':
            ax = expr_keys.index('w')
            v = np.array([get_net_info(sv, config, group_size)['wbits']
                          for sv in _esm['w']])
            comp_nd_list.append(_bcast_1d(v, ax))
        elif obj in ('kvbits', 'kbits', 'vbits'):
            kv_key = 'eff_kv' if 'eff_kv' in expr_keys else 'kv'
            ax = expr_keys.index(kv_key)
            v = np.array([get_net_info(sv, config, group_size)[obj]
                          for sv in _esm[kv_key]])
            comp_nd_list.append(_bcast_1d(v, ax))
        elif obj in ('kvdim', 'kdim', 'vdim'):
            kv_key = 'eff_kv' if 'eff_kv' in expr_keys else 'kvdim'
            ax = expr_keys.index(kv_key)
            v = np.array([get_net_info(sv, config, group_size)[obj]
                          for sv in _esm[kv_key]])
            comp_nd_list.append(_bcast_1d(v, ax))
        elif obj in ('eff_kvbits', 'eff_kbits', 'eff_vbits'):
            t_map = {'eff_kvbits': 'kv', 'eff_kbits': 'k', 'eff_vbits': 'v'}
            t = t_map[obj]
            if 'eff_kv' in expr_keys:
                ax = expr_keys.index('eff_kv')
                v = np.array([get_net_info(sv, config, group_size)[obj]
                              for sv in _esm['eff_kv']])
                comp_nd_list.append(_bcast_1d(v, ax))
            else:
                assert 'kv' in expr_keys and 'kvdim' in expr_keys, \
                    f"'{obj}' requires eff_kv_expr or both kv_expr and kvdim_expr"
                vals_2d = compute_eff_kvbits_batch(_esm['kv'], _esm['kvdim'],
                                                   config, target=t)
                comp_nd_list.append(_bcast_2d(vals_2d, expr_keys.index('kv'),
                                              expr_keys.index('kvdim')))
        elif obj == 'memory':
            if 'w' in expr_keys:
                w_mem = np.array([compute_weight_memory(sv, config, group_size)
                                  for sv in _esm['w']])
                mem_nd = _bcast_1d(w_mem, expr_keys.index('w')).astype(np.float64)
            else:
                mem_nd = np.full(nd_shape,
                                 compute_weight_memory(default_arch, config,
                                                       group_size),
                                 dtype=np.float64)
            if 'eff_kv' in expr_keys:
                kv_cache = np.array([compute_cache_memory_single(sv, config,
                                                                 args.n_token,
                                                                 sink=getattr(args, 'attn_sink', 0))
                                     for sv in _esm['eff_kv']])
                mem_nd = mem_nd + _bcast_1d(kv_cache, expr_keys.index('eff_kv'))
            elif 'kv' in expr_keys and 'kvdim' in expr_keys:
                kv_2d = compute_cache_memory_batch(_esm['kv'], _esm['kvdim'],
                                                   config, args.n_token,
                                                   sink=getattr(args, 'attn_sink', 0))
                mem_nd = mem_nd + _bcast_2d(kv_2d, expr_keys.index('kv'),
                                            expr_keys.index('kvdim'))
            elif 'kv' in expr_keys:
                kv_1d = compute_cache_memory_batch(_esm['kv'], None, config,
                                                   args.n_token,
                                                   sink=getattr(args, 'attn_sink', 0))
                mem_nd = mem_nd + _bcast_1d(kv_1d, expr_keys.index('kv'))
            elif 'kvdim' in expr_keys:
                kv_1d = compute_cache_memory_batch([default_arch], _esm['kvdim'],
                                                   config, args.n_token,
                                                   sink=getattr(args, 'attn_sink', 0))[0]
                mem_nd = mem_nd + _bcast_1d(kv_1d, expr_keys.index('kvdim'))
            comp_nd_list.append(mem_nd)
        else:
            raise ValueError(
                f"comp_obj='{obj}' not supported for vectorized computation. "
                f"Supported: wbits, kvbits, kbits, vbits, kvdim, kdim, vdim, "
                f"eff_kvbits, eff_kbits, eff_vbits, memory")

    return NDCombo(expr_keys, _esm, _efm, nd_shape, new_metric_nd,
                   comp_nd_list, metric_1d)


def comp_key_order(config, group_size):
    """get_net_info key order — the complexity rows of results.csv."""
    return list(get_net_info({}, config, group_size=-1, n_token=0).keys())


def configure_model_cache(args, model, *, use_cache):
    """Toggle KV-cache residual length / quant_kv_output on the sampled model.

    use_cache=True  → benchmarks / strided / prefill_prompt (needs past_kv)
    use_cache=False → single-shot loss with quantised KV output
    """
    res_len = args.residual_length if use_cache else 0
    # 'think' is a KIVI variant — model attaches the same kivi_config (see
    # evaluator.py active_kv = 'kivi' if 'kivi' in m or 'think' in m). Match
    # that here so residual_length is set when kv_method='think'.
    if 'kivi' in args.kv_method or 'think' in args.kv_method:
        model.config.kivi_config.residual_length = res_len
    elif 'hqq' in args.kv_method:
        model.generation_config.cache_config = res_len
    model.config.quant_kv_output = not use_cache
    model.config.use_cache = use_cache


def evaluate_metric(args, arch, model, evaluator, accelerator):
    """Run the calibration-set metric (loss/JSD/ppl) for one architecture."""
    use_cache = args.stride is not None or args.prefill_prompt
    configure_model_cache(args, model, use_cache=use_cache)
    return evaluator.eval(arch=arch, metric=args.metric, model=model,
                          accelerator=accelerator, loss_func=args.loss_func,
                          stride=args.stride,
                          prefill_prompt=args.prefill_prompt)[0]