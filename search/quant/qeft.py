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
        reorder=True,
        target_rank=32,
        seed=42,
        sym=False,
        true_sequential=False,
        percdamp=.01,
        act_order=False,
        htop_protect=False,
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

        owq_layers = meta['owq_layers']
        ratios = meta['ratios']
        # tuple_arch = (w_bits, n_outlier) search scheme (indices from a multi-rank
        # dict {key: {n_out: [cols]}}); legacy = fractional bits + flat {key: [cols]}.
        tuple_arch = any(isinstance(larch[l][0], (list, tuple)) for l in larch)
        # tuple_arch with outliers needs a multi-rank dict — fail fast if missing/old-format
        # (else it crashes deep in the loop on None[key] / a flat list).
        if tuple_arch and any(e[1] > 0 for l in larch for e in larch[l]):
            # skip the 'n_outlier' metadata key when sampling a representative entry.
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
            # per-layer outlier counts are incompatible with QEFT's single global
            # reorder (kernel-packing only, no effect on fake-quant), so disable it.
            reorder = False
        elif target_rank is not None:
            for l, is_owq in owq_layers.items():
                if is_owq:
                    n_out_dict[l] = [target_rank if bits != math.ceil(bits) else 0 for bits in larch[l]]
        # DEFAULT act_order=False for ALL models (project decision 2026-07-03:
        # QEFT archs always carry FP16 outlier columns, which remove the
        # Llama-3 original-order chaos source — measured n_out=32 wiki PPL
        # 7.533 vs n_out=0's 11.1 spike — while act_order=True costs up to
        # -17 lcc via calib overfitting). act_order='auto' or None opts into
        # the arch-aware gate below; an EXPLICIT True/False is always honored.
        #
        # act_order is a two-sided knob (Llama-3.1-8B-Instruct, measured):
        #  * act_order=True is a CALIB-OVERFIT AMPLIFIER at high bits: per-layer
        #    c4-calib error improves 20-30% but heldout-c4 flips +11-16% and CODE
        #    inputs +15-25% WORSE (tests/test_actorder_probe2.py, 4-bit n_out=0
        #    layers). Arch-level: REAL w4 arch lcc 47.4 (True) vs 64.5 (False),
        #    w3.5 56.6 vs 60.5 — PPL is blind to it (same manifold as calib).
        #    Ordering is the dominant channel (-13.1 of the 17-pt gap; protected-
        #    column identity only -4.0, htop_protect arm).
        #  * act_order=False is NUMERICALLY CHAOTIC where GPTQ original-order
        #    breaks: 2-bit-heavy archs (REAL w3 arch, 30% 2-bit: 53.9 True vs
        #    46.9 False) and 0-outlier uniform archs on Llama-3.x (W4 wiki PPL
        #    swings 7.8-13.5 vs stable 7.2 with True).
        # Gate: True where original-order chaos dominates (many 2-bit layers, or
        # no FP16 outlier columns to absorb massive activations); False otherwise.
        # Boundary is uncalibrated between 1.8% (w3.5, False wins) and 30% (w3,
        # True wins) 2-bit layers; 10% splits the gap. Measured on Llama-3.1 only;
        # per-linear mixing gates measured WORSE than both globals — keep global.
        if act_order is None or act_order == 'auto':
            if not ('llama-3' in self.model_name.lower()):
                act_order = False   # Qwen/Mistral: original order is stable
            else:
                ents = [e for l in larch for e in larch[l]]
                if tuple_arch:
                    frac2 = sum(int(e[0]) <= 2 for e in ents) / max(len(ents), 1)
                    has_out = any(int(e[1]) > 0 for e in ents)
                else:
                    frac2 = sum(int(e) <= 2 for e in ents) / max(len(ents), 1)
                    has_out = any(b != math.floor(b) for b in ents) and target_rank
                act_order = (not has_out) or frac2 >= 0.10
                print(f'[qeft] auto act_order={act_order} '
                      f'(2bit_frac={frac2:.3f}, outliers={bool(has_out)})')
        elif act_order is False and 'llama-3' in self.model_name.lower():
            ents = [e for l in larch for e in larch[l]]
            has_out = (any(int(e[1]) > 0 for e in ents) if tuple_arch
                       else any(b != math.floor(b) for b in ents) and target_rank)
            if not has_out:
                print('[qeft] WARNING: Llama-3 + 0-outlier + act_order=False is '
                      'the known chaotic config (wiki PPL 7.5<->12.4 by calib '
                      'seed) — check PPL or pass act_order=True/auto.')
        print(f'tuple_arch : {tuple_arch}, reorder : {reorder}, act_order : {act_order}')
        print(f'n_out_dict : {n_out_dict}')
        print(f'self.arch : {self.arch}')

        def _wbits(name, i):
            """Integer weight bits for (name, block i), handling both the (bits,
            n_outlier)-tuple search arch and the legacy scalar/fractional arch."""
            e = larch[name][i]
            return int(e[0]) if tuple_arch else e

        def _ao(name, i):
            """Per-call act_order. act_order='per_linear' gates each linear by
            its OWN bits: True at <=3 bits (order/compensation stability
            dominates), False at 4 bits (act_order's group-misalignment side
            effects dominate). Measured on the real searched archs (lcc, KIVI
            sk8): w4 arch aoTrue 47.4 vs aoFalse 64.5; w3 arch aoTrue 53.9 vs
            aoFalse 46.9 — opposite winners, so a single global flag cannot fit
            mixed archs."""
            if act_order == 'per_linear':
                return _wbits(name, i) <= 3
            if act_order == 'per_linear2':
                # gate on 2-bit only: aoTrue where GPTQ original-order is
                # numerically chaotic, aoFalse everywhere else. Discriminates
                # "2-bit chaos" vs "avg-bits" as the arch-level gate driver.
                return _wbits(name, i) <= 2
            return act_order

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
                    out_ids = gptq_owq[name].hessian_sorting(
                        actorder=_ao(name, i),
                        frob_norm=frob_norm_error,
                        outidx=outidx,
                        htop_protect=htop_protect,
                        )
                    gptq_owq[name].quantizer.out_ids = out_ids
                    gptq_owq[name].quantizer.n_out = out_ids.numel()
                    gptq_owq[name].quantizer.reorder = reorder

                if not no_frob_norm and not tuple_arch:
                    del W
                    del W_quant
                    del temp_quantizer
                    torch.cuda.empty_cache()
                
                for name in subset:
                    key = f"{meta['prefix']}.{i}.{name}"
                    # global_ids must be the quantizer's ACTUAL out_ids (empty for a
                    # 0-outlier arch → make_reorder is a no-op), NOT the raw dict indices
                    # (would permute the residual w/o matching down_proj → PPL spike; and
                    # the multi-rank dict isn't a flat index list). Legacy: out_ids == dict.
                    if reorder and name not in meta['sequential'][1] and name not in meta['sequential'][3]:
                        global_ids = gptq_owq[name].quantizer.out_ids
                    if nearest_owq:
                        if gptq_owq[name].quantizer.reorder:
                            gptq_owq[name].fasterquant_nearest_owq_reorder(groupsize=self.group_size, actorder=_ao(name, i))
                        else:
                            gptq_owq[name].fasterquant_nearest_owq(groupsize=self.group_size, actorder=_ao(name, i))
                    else:
                        if gptq_owq[name].quantizer.reorder:
                            gptq_owq[name].fasterquant_reorder(percdamp=percdamp, groupsize=self.group_size, actorder=_ao(name, i))
                        else:
                            gptq_owq[name].fasterquant(percdamp=percdamp, groupsize=self.group_size, actorder=_ao(name, i))
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
