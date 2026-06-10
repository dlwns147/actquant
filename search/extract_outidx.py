"""Extract OWQ/QEFT outlier-column indices for one or more outlier counts.

Unified replacement for the former extract_outidx.py / extract_outidx_multi.py —
one script handles a single rank or a list of ranks.

Selection (faithful to QEFT):
  * qkv / gate / up  → a GLOBAL outlier set over hidden channels, shared across
    layers (top-k of the summed, mean-normalized Hessian diagonal).
  * down_proj        → per-layer outlier set (per-layer Hessian sorting).

Output (`<output_dir>/w{wbits}_r{R1_R2_..}_{dataset}/outlier.pth`):
  {key: {n_out: [col idx]}}  + 'n_outlier': [ranks]
  (consumed by the QEFT (bits, n_outlier) search axis; see search_space/llama.py
   + quant/qeft.py tuple_arch path, and quant/awq_qeft.py resolve_outliers).

Modes:
  --target_rank R [R ...]   explicit outlier counts (shared candidate ranks)
  --target_bit  B           OWQ effective-bit budget → per-layer count by ratio
"""

import time
import os
import argparse

import torch
import torch.nn as nn

from quant.qeft_utils.recon import GPTQ_OWQ
from quant.qeft_utils.quant import *
from quant.qeft_utils.misc import *
from quant.qeft_utils.reorder import *
from quant.base import get_owq_calib_dataset
from transformers import AutoModelForCausalLM
from utils.data import get_tokenizer


@torch.no_grad()
def extract_outlieridx(model, dataloader, dev, args):
    if args.perhead is not None:
        args.target_rank = [args.perhead * model.config.num_attention_heads]

    meta = args.meta
    # transformers>=4.45 passes position_embeddings (cos,sin) to each decoder
    # layer; capture it so the Catcher replay below doesn't pass None (the
    # original extract scripts predate this and crash on `cos, sin = None`).
    if 'position_embeddings' not in meta['inp_kwargs']:
        meta['inp_kwargs'] = list(meta['inp_kwargs']) + ['position_embeddings']
    owq_layers = meta['owq_layers']
    ratios = meta['ratios']

    # ---- resolve per-layer candidate outlier counts -----------------------
    # n_out_dict[name] is ALWAYS a list of counts (one entry per requested rank;
    # a single entry for --target_bit). bit_mode counts differ per layer
    # (ratio-based) → each layer's dict just has its own single-count key.
    bit_mode = args.target_bit is not None
    n_out_dict = {l: [0] for l in owq_layers.keys()}
    if bit_mode:
        # per-layer ratio derived once layer 0 is on device (see below); the
        # rate `r` only needs wbits / target_bit / #owq-layers.
        n_owq_layers = sum(owq_layers.values())
        r = (12 / (16 - args.wbits)) * (args.target_bit - args.wbits) / n_owq_layers
    ranks = sorted(set(int(x) for x in (args.target_rank or [])))
    if not bit_mode:
        assert ranks, "provide --target_rank (one or more) or --target_bit"
        for l in owq_layers:
            n_out_dict[l] = list(ranks)

    dirname_rank = '_'.join(map(str, ranks)) if not bit_mode else f'bit{args.target_bit}'
    dirname = os.path.join(args.output_dir, f"w{args.wbits}_r{dirname_rank}_{args.dataset}")
    os.makedirs(dirname, exist_ok=True)

    d = model.get_input_embeddings().weight.shape[1]
    sensitivity_sum = torch.zeros([d], dtype=torch.float, device=dev)

    print('Starting ...')
    use_cache = model.config.use_cache
    layers, pre_layers, _ = parsing_layers(model, meta)
    model.config.use_cache = False

    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(dev)
    layers[0] = layers[0].to(dev)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb.to(dev)

    # bit_mode: now that layer 0 is on device, derive per-layer ratio counts
    if bit_mode:
        layer0 = find_layers(layers[0], layers=[nn.Linear])
        for l in owq_layers:
            n_out = round(layer0[l].weight.data.shape[1] * r * ratios[l])
            if n_out % 2 == 1:
                n_out += 1
            n_out_dict[l] = [int(n_out)]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, args.seqlen, model.config.hidden_size),
                       dtype=dtype, device=dev)
    cache = {kw: None for kw in meta['inp_kwargs']}
    cache['i'] = 0

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            for key in cache:
                if key == 'i':
                    cache['i'] += 1
                else:
                    cache[key] = kwargs.get(key)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module.cpu()
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb.to('cpu')
    for pre_layer in pre_layers:
        pre_layer = pre_layer.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    del cache['i']
    inp_kwargs = cache
    print('Ready.')

    n_outlier_meta = list(ranks) if not bit_mode else \
        sorted({c for v in n_out_dict.values() for c in v})
    outlier = {'n_outlier': n_outlier_meta}

    # ---- pass 1: per-layer reconstruction; down outliers + global sensitivity
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        block_layers = find_layers(layer, layers=[nn.Linear])
        sequential = meta['sequential'] if args.true_sequential else [list(block_layers.keys())]

        for names in sequential:
            subset = {n: block_layers[n] for n in names}

            gptq_owq = {}
            for name in subset:
                gptq_owq[name] = GPTQ_OWQ(subset[name], n_out=max(n_out_dict[name]))
                gptq_owq[name].quantizer = Quantizer(
                    args.wbits, perchannel=True, sym=args.sym,
                    mse=(args.tuning == 'mse'), group_size=args.groupsize)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq_owq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
            for j in range(args.nsamples):
                layer(inps[j].unsqueeze(0), **inp_kwargs)
            for h in handles:
                h.remove()

            # per-(name, rank) outlier ids via Hessian sorting
            out_ids_list = {name: {} for name in subset}
            for name in meta['sequential'][0] + meta['sequential'][2] + meta['sequential'][3]:
                for n_out in n_out_dict[name]:
                    gptq_owq[name].n_out = n_out
                    gptq_owq[name].quantizer.n_out = n_out
                    if not args.no_frob_norm:
                        W = subset[name].weight.data.clone().to(torch.float)
                        tq = Quantizer(args.wbits, perchannel=True, sym=args.sym,
                                       mse=(args.tuning == 'mse'), group_size=args.groupsize)
                        tq.find_params(W, weight=True, num=30)
                        frob_norm_error = (W - tq.quantize(W)).pow(2).sum(dim=0)
                    else:
                        frob_norm_error = None
                    out_ids_list[name][n_out] = gptq_owq[name].hessian_sorting(
                        actorder=args.act_order, frob_norm=frob_norm_error)

            # qkv + gate/up: accumulate GLOBAL sensitivity (shared across layers)
            for name in meta['sequential'][0] + meta['sequential'][2]:
                sensitivity = gptq_owq[name].H_diag
                sensitivity_sum += sensitivity / sensitivity.mean()
                gptq_owq[name].free()

            # down_proj: PER-LAYER outlier set
            for name in meta['sequential'][3]:
                key = f"{meta['prefix']}.{i}.{name}"
                outlier[key] = {n_out: out_ids_list[name][n_out].tolist()
                                for n_out in n_out_dict[name]}
                gptq_owq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]
        layers[i] = layer.cpu()
        del gptq_owq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # ---- pass 2: GLOBAL qkv/gate/up outlier set (top-k of summed sensitivity)
    for i in range(len(layers)):
        block_layers = find_layers(layers[i], layers=[nn.Linear])
        sequential = meta['sequential'] if args.true_sequential else [list(block_layers.keys())]
        for names in sequential:
            for name in [n for n in names if n in meta['sequential'][0] + meta['sequential'][2]]:
                key = f"{meta['prefix']}.{i}.{name}"
                outlier[key] = {
                    n_out: sorted(torch.topk(sensitivity_sum, n_out).indices.cpu().tolist())
                    for n_out in n_out_dict[name]}

    output_path = os.path.join(dirname, 'outlier.pth')
    n_keys = len([k for k in outlier if k != 'n_outlier'])
    print(f"ranks={ranks if not bit_mode else f'target_bit={args.target_bit}'} "
          f"nsamples={args.nsamples} | {n_keys} layer keys")
    print(f"outlieridx saved to {output_path}")
    torch.save(outlier, output_path)
    model.config.use_cache = use_cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='hugging face model to load')
    parser.add_argument('dataset', type=str,
                        help='calibration data source. [wikitext2, ptb, c4, custom_path]')
    parser.add_argument('--seqlen', type=int, default=0,
                        help='calibration seqlen (0 = min(model ctx, 2048); set to cap huge-ctx models)')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 16],
                        help='weight bits (16 = base model, outlier-only extraction).')
    parser.add_argument('--target_bit', type=float, default=None,
                        help='OWQ effective-bit budget → per-layer count by ratio.')
    parser.add_argument('--target_rank', type=int, nargs='+', default=[],
                        help='one or more outlier-column counts, e.g. 32  or  4 16 64 256.')
    parser.add_argument('--tuning', type=str, default='mse', choices=['mse', 'minmax'])
    parser.add_argument('--no_frob_norm', action='store_true',
                        help='skip the Frobenius-norm term in outlier ranking.')
    parser.add_argument('--percdamp', type=float, default=.01)
    parser.add_argument('--dtype', type=str, default=None,
                        help='model dtype (bfloat16 for falcon / llama-65B).')
    parser.add_argument('--layers', nargs='+', type=str, default=None,
                        help='Layers to apply OWQ.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sym', action='store_true')
    parser.add_argument('--nearest', action='store_true')
    parser.add_argument('--groupsize', type=int, default=-1)
    parser.add_argument('--nearest_owq', action='store_true')
    parser.add_argument('--act-order', action='store_true')
    parser.add_argument('--true-sequential', action='store_true')
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--output_dir', type=str, default='', help='outlier save dir')
    parser.add_argument('--perhead', type=int, default=None, help='per-head rank (× n_heads).')
    # consumed by quant.qeft_utils.misc.processing_arguments (kept for parity).
    parser.add_argument('--save', type=str, default='', help='(unused here) packed-ckpt save path.')
    parser.add_argument('--fake', action='store_true', help='(unused here) fake-quant ckpt.')
    parser.add_argument('--packing', action='store_true', help='(unused here) 3/4-bit packed ckpt.')

    args = parser.parse_args()
    assert args.target_rank or args.target_bit is not None or args.perhead is not None, \
        "provide --target_rank, --target_bit, or --perhead"
    meta = processing_arguments(args)
    print(f'args: {args}')
    args.meta = meta
    device = torch.device('cuda:0')
    seed_all(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=args.dtype, device_map='cpu', trust_remote_code=True)
    tokenizer = get_tokenizer(args.model)

    if args.seqlen and args.seqlen > 0:
        pass  # explicit override (avoids sizing the calib buffer from huge ctx, e.g. Llama-3.1 131072)
    elif getattr(model.config, 'max_position_embeddings', None):
        args.seqlen = min(model.config.max_position_embeddings, 2048)
    elif getattr(model.config, 'max_sequence_length', None):
        args.seqlen = min(model.config.max_sequence_length, 2048)
    else:
        args.seqlen = 2048

    dataloader = get_owq_calib_dataset(args.dataset, tokenizer=tokenizer,
                                       n_samples=args.nsamples, seed=args.seed, seqlen=args.seqlen)
    tick = time.time()
    extract_outlieridx(model, dataloader, device, args)
    print(f"Extract Outlieridx Time : {round(time.time() - tick, 1)}")
