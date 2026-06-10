import time

import torch
import torch.nn as nn

import argparse
from tqdm import tqdm

from .qeft_utils.recon import GPTQ_OWQ
from .qeft_utils.quant import *
from .qeft_utils.misc import *
from .qeft_utils.reorder import *

import torch
import torch.nn as nn
import math
import json
import transformers

from .base import BASE, get_owq_calib_dataset
    
class QEFT(BASE):
    def __init__(self, model_name, config, arch, device_map, group_size=128, dtype='auto', dev='cuda', prune=False, do_owq=False, owq=None, **kwargs):
        super().__init__(model_name, config, arch, device_map=device_map, group_size=group_size, dtype=dtype, dev=dev, prune=prune, do_owq=do_owq, owq=owq)
        self.method = 'owq'
        # BASE.__init__ keys outlier loading off `outlier_path`, but subclasses
        # forward `owq=` (lands in BASE **kwargs) -> self.owq stays None. Set it.
        if do_owq and self.owq is None and owq is not None:
            self.owq = torch.load(owq, weights_only=False) if isinstance(owq, str) else owq


    @torch.no_grad()
    def run(
        self,
        samples=None,
        n_samples=128,
        seqlen=2048,
        dataset='c4',
        # dataset='wikitext2',
        reorder=True,
        # reorder=False,
        target_rank=32,
        seed=42,
        sym=False,
        true_sequential=False,
        percdamp=.01,
        act_order=False,
        no_frob_norm=False,
        tuning='mse',
        layers=None,
        nearest_owq=False
    ):
        
        if samples is None:
            dataloader = get_owq_calib_dataset(dataset, tokenizer=self.tokenizer, n_samples=n_samples, seqlen=seqlen, seed=seed)
            
        # assert args.no_frob_norm == True
        meta = get_meta(self.model_name, layers)
        # transformers>=4.45 passes position_embeddings (cos,sin) to each decoder
        # layer; capture it so the Catcher replay below doesn't pass None.
        if 'position_embeddings' not in meta['inp_kwargs']:
            meta['inp_kwargs'] = list(meta['inp_kwargs']) + ['position_embeddings']
        # arch may be {'linear': {name: [...]}} (standalone) or {name: [...]}
        # (evaluator passes q_arch['w'] directly); accept both.
        larch = self.arch['linear'] if isinstance(self.arch, dict) and 'linear' in self.arch else self.arch
        print('Starting ...')

        use_cache = self.model.config.use_cache
        layers, pre_layers, post_layers = parsing_layers(self.model, meta)
        
        self.model.config.use_cache = False
        
        for pre_layer in pre_layers:
            pre_layer = pre_layer.to(self.dev)
        
        layers[0] = layers[0].to(self.dev)
        if hasattr(self.model.model, 'rotary_emb'):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.dev)

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (n_samples, seqlen, self.model.config.hidden_size), dtype=dtype, device=self.dev
        )

        cache = {kw:None for kw in meta['inp_kwargs']}
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
                self.model(batch[0].to(self.dev))
            except ValueError:
                pass
        
        layers[0] = layers[0].module.cpu()
        if hasattr(self.model.model, 'rotary_emb'):
            self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        
        for pre_layer in pre_layers:
            pre_layer = pre_layer.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        del cache['i']
        inp_kwargs = cache

        # print('Ready.')
        owq_layers = meta['owq_layers']
        ratios = meta['ratios']
        # New (search) scheme: arch entries are (w_bits, n_outlier) tuples carrying a
        # searchable per-layer outlier-column count; outlier indices come from a
        # MULTI-rank dict {key: {n_out: [cols]}} (extract_outidx.py). The legacy
        # scheme uses fractional bits + a single target_rank + a flat {key: [cols]} dict.
        tuple_arch = any(isinstance(larch[l][0], (list, tuple)) for l in larch)
        # When the arch carries searchable per-layer outlier counts, the indices
        # must come from a multi-rank dict {key: {n_out: [cols]}}. Fail fast with an
        # actionable message instead of crashing on `None[key]` / a flat list deep in
        # the per-layer loop (e.g. post_search/sample_surrogate run with --w_method
        # qeft but a missing/old-format --outlier_path).
        if tuple_arch and any(e[1] > 0 for l in larch for e in larch[l]):
            # skip the 'n_outlier' metadata key (a list of ranks, saved first by
            # extract_outidx.py) when sampling a representative per-layer entry.
            sample_key = next((k for k in self.owq if k != 'n_outlier'), None) \
                if isinstance(self.owq, dict) and self.owq else None
            if not isinstance(self.owq, dict) or sample_key is None or not isinstance(self.owq[sample_key], dict):
                raise ValueError(
                    "QEFT (bits, n_outlier) arch needs a multi-rank outlier dict "
                    "{key: {n_out: [cols]}} from extract_outidx.py, but got "
                    f"{type(self.owq).__name__}. Pass --outlier_path to that dict.")
        n_out_dict = {l: [0] * len(layers) for l in owq_layers.keys()}
        if tuple_arch:
            for l, is_owq in owq_layers.items():
                if is_owq:
                    n_out_dict[l] = [int(e[1]) for e in larch[l]]
            # Per-layer outlier counts vary across layers, which is incompatible with
            # QEFT's single global channel reorder (one shared out_id set for all
            # qkv/up/gate). Reorder is a kernel-packing optimization only and does not
            # change fake-quant outputs, so disable it for the searchable scheme.
            reorder = False
        elif target_rank is not None:
            for l, is_owq in owq_layers.items():
                if is_owq:
                    n_out_dict[l] = [target_rank if bits != math.ceil(bits) else 0 for bits in larch[l]]
                    # n_out_dict[l] = target_rank
        print(f'tuple_arch : {tuple_arch}, reorder : {reorder}')
        print(f'n_out_dict : {n_out_dict}')
        print(f'self.arch : {self.arch}')

        def _wbits(name, i):
            """Integer weight bits for (name, block i), handling both the (bits,
            n_outlier)-tuple search arch and the legacy scalar/fractional arch."""
            e = larch[name][i]
            return int(e[0]) if tuple_arch else e

        def _outidx(name, i, key):
            """FP16 outlier column indices for (name, block i), or None.
            tuple_arch -> multi-rank dict lookup self.owq[key][n_out];
            legacy -> flat self.owq[key] gated on fractional bits."""
            if key not in self.owq:
                return None
            if tuple_arch:
                n_out = n_out_dict[name][i]
                return torch.tensor(self.owq[key][n_out]) if n_out > 0 else None
            wb = larch[name][i]
            return torch.tensor(self.owq[key]) if wb != math.floor(wb) else None
        
        quantizers = {}
        for i in tqdm(range(len(layers)), "Reconstruction Blocks..."):
            layer = layers[i].to(self.dev)
            block_layers = find_layers(layer, layers=[nn.Linear])

            if true_sequential:
                sequential = meta['sequential']
            else:
                sequential = [list(block_layers.keys())]
                
            for names in sequential:
                subset = {n: block_layers[n] for n in names}

                gptq_owq = {}
                for name in subset:
                    wbits = _wbits(name, i)
                    gptq_owq[name] = GPTQ_OWQ(subset[name], n_out=n_out_dict[name][i])
                    gptq_owq[name].quantizer = Quantizer(
                        math.floor(wbits), perchannel=True, sym=sym, mse=(tuning == 'mse'), group_size=self.group_size
                    )
                    gptq_owq[name].quantizer.n_out = n_out_dict[name][i]
                    
                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq_owq[name].add_batch(inp[0].data, out.data)
                    return tmp
                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(n_samples):
                    layer(inps[j].unsqueeze(0), **inp_kwargs)
                for h in handles:
                    h.remove()
                
                for name in subset:
                    wbits = _wbits(name, i)
                    # tuple_arch supplies outlier indices from the dict, so the
                    # frob-norm outlier-ranking pass is unused — skip it (big speedup).
                    if not no_frob_norm and not tuple_arch and (not reorder or (reorder and name in meta['sequential'][1] + meta['sequential'][3])):
                        W = subset[name].weight.data.clone().to(torch.float)
                        temp_quantizer = Quantizer(
                            math.floor(wbits), perchannel=True, sym=sym, mse=(tuning == 'mse'), group_size=self.group_size
                        )
                        temp_quantizer.find_params(W, weight=True, num=40)
                        W_quant = temp_quantizer.quantize(W)
                        frob_norm_error = (W - W_quant).pow(2).sum(dim=0)
                    else:
                        frob_norm_error = None

                    key = f"{meta['prefix']}.{i}.{name}"
                    outidx = _outidx(name, i, key)
                    # print(f'key : {key}, outidx : {outidx}, wbits : {wbits}, key in self.owq : {key in self.owq}')
                    out_ids = gptq_owq[name].hessian_sorting(
                        actorder=act_order,
                        frob_norm=frob_norm_error,
                        outidx=outidx,
                        )
                    gptq_owq[name].quantizer.out_ids = out_ids
                    gptq_owq[name].quantizer.n_out = out_ids.numel()
                    gptq_owq[name].quantizer.reorder = reorder # if name not in meta['sequential'][1] else False
                    # print(f'n_out_dict[name][i] : {n_out_dict[name][i]}, n_out : {out_ids.numel()}, self.owq[key] : {self.owq[key] if key in self.owq else 0}')

                if not no_frob_norm and not tuple_arch:
                    del W
                    del W_quant
                    del temp_quantizer
                    torch.cuda.empty_cache()
                
                for name in subset:
                    key = f"{meta['prefix']}.{i}.{name}"
                    # print(f"Quantizing {key}")
                    # global_ids feeds the single global make_reorder (legacy scheme
                    # only). Skip when reorder is off — and self.owq[key] is a per-rank
                    # dict in the tuple/multi scheme, not a flat index list.
                    if reorder and name not in meta['sequential'][1] and name not in meta['sequential'][3]:
                        global_ids = torch.tensor(self.owq[key])
                    # print(f'out_ids : {gptq_owq[name].quantizer.out_ids.tolist()}')
                    # print(f'self.owq[key] : {self.owq[key] if key in self.owq else 0}') 
                    # print(f'global_ids : {global_ids.tolist()}')
                    # if key in self.owq:
                    #     print(f'(gptq_owq[name].quantizer.out_ids == self.owq[key]).sum() : {(gptq_owq[name].quantizer.out_ids == self.owq[key]).sum()}')
                    # print('=' * 20)
                    if nearest_owq:
                        if gptq_owq[name].quantizer.reorder:
                            gptq_owq[name].fasterquant_nearest_owq_reorder(groupsize=self.group_size, actorder=act_order)
                        else:
                            gptq_owq[name].fasterquant_nearest_owq(groupsize=self.group_size, actorder=act_order)
                    else:
                        if gptq_owq[name].quantizer.reorder:
                            gptq_owq[name].fasterquant_reorder(percdamp=percdamp, groupsize=self.group_size, actorder=act_order)
                        else:
                            gptq_owq[name].fasterquant(percdamp=percdamp, groupsize=self.group_size, actorder=act_order)
                    quantizers[f"{meta['prefix']}.{i}.{name}"] = gptq_owq[name].quantizer
                    gptq_owq[name].free()
            
            for j in range(n_samples):
                outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]
                
            for name in list(block_layers.keys()):
                quantizers[f"{meta['prefix']}.{i}.{name}"] = quantizers[f"{meta['prefix']}.{i}.{name}"].cpu()

            layers[i] = layer.cpu()
            del layer
            del gptq_owq 
            torch.cuda.empty_cache()

            inps, outs = outs, inps
                
        if reorder:
            global_ids = global_ids.cpu()
            make_reorder(self.model, quantizers, global_ids, meta)
        self.model.config.use_cache = use_cache
