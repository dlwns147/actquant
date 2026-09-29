"""Exact per-layer AWQ lookup table, so a search can score archs with the DEPLOYED quantizer
at close to HQQ cost (bench_selection_regret findings 69/70).

Why it can be exact. In run_awq (awq_utils/pre_quant.py) each block is calibrated on the
FULL-PRECISION output of the block before it (`inps` is propagated before the block is
scaled or clipped), and within a block
  * each scale group's search sees only its own linears quantized:
        qkv    <- input_layernorm       depends on (b_q, b_k, b_v)     27 combos
        gateup <- post_attention_ln     depends on (b_gate, b_up)       9
        down   <- up_proj               depends on (b_down)             3
    (the v->o group is skipped on GQA models, where v_proj/o_proj shapes differ)
  * each clip is searched on the SCALED weight / SCALED input features:
        v_proj    context (b_q, b_k, b_v)            27    (q_/k_ are never clipped)
        o_proj    context (b_o)                       3
        gate_proj context (b_gate, b_up)              9
        up_proj   context (b_gate, b_up, b_down)     27    (scale_fc_fc divides up's rows)
        down_proj context (b_down)                    3
So the AWQ result of any arch is a lookup: per layer, pick the entries for its bits and
hand the assembled awq_results to the unchanged apply_awq.

Cost: 39 scale + 69 clip searches per layer, against 3 + 5 in one run_awq build.

    build:    python -m quant.awq_table build  --model_path ... --out DIR [--layers a:b]
    verify:   python -m quant.awq_table verify --model_path ... --out DIR --arch_json A.json
"""
import argparse
import functools
import gc
import itertools
import json
import os
import time
from collections import defaultdict

import torch

from .awq_utils.auto_clip import auto_clip_layer_asym, auto_clip_layer_sym
from .awq_utils.auto_scale import apply_scale, auto_scale_block
from .awq_utils.module import append_str_prefix, get_op_name
from .awq_utils.pre_quant import get_blocks, get_named_linears, move_embed, run_awq
from .base import get_awq_calib_dataset

QKV = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj')
GU = ('mlp.gate_proj', 'mlp.up_proj')
DOWN = ('mlp.down_proj',)
O = ('self_attn.o_proj',)
LINEARS = QKV + O + GU + DOWN
# clip order == named_linears order minus q_/k_ (auto_clip_block_* skips them)
CLIP_ORDER = ('self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj',
              'mlp.down_proj')


def _combos(bits, n):
    return list(itertools.product(bits, repeat=n))


def _mb(fill, **kw):
    mb = {n: fill for n in LINEARS}
    mb.update(kw)
    return mb


# ─────────────────────────────── build ───────────────────────────────
@torch.no_grad()
def _calib_inputs(model, tok, n_samples, seqlen, calib_data):
    """The exact prologue of run_awq: calibration samples -> layer-0 inputs + kwargs."""
    samples = get_awq_calib_dataset(data=calib_data, tokenizer=tok, n_samples=n_samples,
                                    block_size=seqlen)
    samples = torch.cat(samples, dim=0)
    layers = get_blocks(model)
    move_embed(model, 'cuda')
    inps, layer_kwargs = [], {}
    layers[0] = layers[0].to('cuda')

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:
        pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    move_embed(model, 'cpu')
    gc.collect(); torch.cuda.empty_cache()
    return inps[0], layer_kwargs


@torch.no_grad()
def _layer_features(layer, inps, layer_kwargs):
    """input features of every linear + the next layer's input, exactly as run_awq."""
    named = get_named_linears(layer)
    feat = defaultdict(list)

    def hook(m, x, y, name):
        feat[name].append(x[0].detach().cpu())

    hs = [named[n].register_forward_hook(functools.partial(hook, name=n)) for n in named]
    inps = inps.to(next(layer.parameters()).device)
    nxt = layer(inps, **layer_kwargs)[0]
    for h in hs:
        h.remove()
    return {k: torch.cat(v, dim=0) for k, v in feat.items()}, nxt


@torch.no_grad()
def _clip(layer, name, feat, bit, q_config, clip_asym):
    lin = layer.get_submodule(name)
    if clip_asym:
        return auto_clip_layer_asym(lin.weight, feat, n_bit=bit, q_config=q_config,
                                    bias=getattr(lin, 'bias', None))
    return (auto_clip_layer_sym(lin.weight, feat, n_bit=bit, q_config=q_config),)


@torch.no_grad()
def build_layer(layer, feat, layer_kwargs, q_config, bits, clip_asym):
    """All scale + clip entries of one block, keyed by the bits they depend on."""
    orig = {k: v.detach().clone() for k, v in layer.state_dict().items()}
    kw = dict(layer_kwargs)
    out = {'scale': {'qkv': {}, 'gateup': {}, 'down': {}},
           'clip': {n: {} for n in CLIP_ORDER}}

    def _scale(group, mb):
        sl = auto_scale_block(layer, kw, q_config=q_config, input_feat=feat, do_owq=False,
                              module_bit=mb, groups={group})
        assert len(sl) == 1, (group, len(sl))
        layer.load_state_dict(orig)
        return sl[0]

    probe = auto_scale_block(layer, kw, q_config=q_config, input_feat=feat, do_owq=False,
                             module_bit=_mb(4), groups={'o'})
    layer.load_state_dict(orig)
    if probe:
        raise NotImplementedError('v->o scale group present (non-GQA model): the v clip '
                                  'context would also depend on b_o; not supported here')

    for c in _combos(bits, 3):
        out['scale']['qkv'][c] = _scale('qkv', _mb(4, **dict(zip(QKV, c))))
    for c in _combos(bits, 2):
        out['scale']['gateup'][c] = _scale('gateup', _mb(4, **dict(zip(GU, c))))
    for c in _combos(bits, 1):
        out['scale']['down'][c] = _scale('down', _mb(4, **dict(zip(DOWN, c))))

    def _scaled(entries, names):
        """layer + input features after applying `entries` (as run_awq's apply_scale)."""
        layer.load_state_dict(orig)
        f = {n: feat[n].clone() for n in names}
        apply_scale(layer, entries, input_feat_dict=f)
        return f

    for c in _combos(bits, 3):                                   # v: (b_q, b_k, b_v)
        f = _scaled([out['scale']['qkv'][c]], QKV)
        out['clip']['self_attn.v_proj'][c] = _clip(layer, 'self_attn.v_proj',
                                                   f['self_attn.v_proj'], c[2], q_config, clip_asym)
    for b in bits:                                               # o: unscaled
        layer.load_state_dict(orig)
        out['clip']['self_attn.o_proj'][(b,)] = _clip(layer, 'self_attn.o_proj',
                                                      feat['self_attn.o_proj'], b, q_config, clip_asym)
    for c in _combos(bits, 2):                                   # gate: (b_g, b_u)
        f = _scaled([out['scale']['gateup'][c]], GU)
        out['clip']['mlp.gate_proj'][c] = _clip(layer, 'mlp.gate_proj', f['mlp.gate_proj'],
                                                c[0], q_config, clip_asym)
    for c in _combos(bits, 3):                                   # up: (b_g, b_u, b_d)
        f = _scaled([out['scale']['gateup'][c[:2]], out['scale']['down'][c[2:]]], GU + DOWN)
        out['clip']['mlp.up_proj'][c] = _clip(layer, 'mlp.up_proj', f['mlp.up_proj'],
                                              c[1], q_config, clip_asym)
    for b in bits:                                               # down: (b_d)
        f = _scaled([out['scale']['down'][(b,)]], DOWN)
        out['clip']['mlp.down_proj'][(b,)] = _clip(layer, 'mlp.down_proj', f['mlp.down_proj'],
                                                   b, q_config, clip_asym)
    layer.load_state_dict(orig)
    out['clip'] = {n: {c: tuple(t.cpu() for t in v) for c, v in d.items()}
                   for n, d in out['clip'].items()}
    return out


def _meta(args, q_config, bits):
    return dict(model=args.model_name, dtype=str(args.dtype), q_config=q_config, bits=list(bits),
                n_samples=args.n_samples, seqlen=args.seqlen, calib=args.calib_data,
                clip_asym=args.clip_asym, torch=torch.__version__)


def _load(args):
    from .awq import AWQ
    m = AWQ(model_name=f'{args.model_path}/{args.model_name}', config=None, arch=None,
            device_map={'': 'cpu'}, group_size=args.group_size, dtype=args.dtype,
            clip_asym=args.clip_asym)
    m.model.config.use_cache = False
    return m


def cmd_build(args):
    q_config = {'zero_point': True, 'q_group_size': args.group_size}
    bits = tuple(args.bits)
    os.makedirs(args.out, exist_ok=True)
    meta_p = os.path.join(args.out, 'meta.json')
    meta = _meta(args, q_config, bits)
    if os.path.exists(meta_p):
        assert json.load(open(meta_p)) == meta, f'{meta_p} was built with other settings'
    else:
        json.dump(meta, open(meta_p, 'w'), indent=1)
    m = _load(args)
    model, tok = m.model, m.tokenizer
    layers = get_blocks(model)
    a, b = (int(x) for x in args.layers.split(':')) if args.layers else (0, len(layers))
    inps, kw = _calib_inputs(model, tok, args.n_samples, args.seqlen, args.calib_data)
    for i in range(b):
        layer = layers[i].cuda()
        path = os.path.join(args.out, f'layer_{i:02d}.pt')
        if i < a or os.path.exists(path):          # only propagate the FP16 input
            with torch.no_grad():
                inps = layer(inps.to('cuda'), **kw)[0]
            layers[i] = layer.cpu()
            continue
        feat, nxt = _layer_features(layer, inps, kw)
        if True:
            t0 = time.time()
            ent = build_layer(layer, feat, kw, q_config, bits, args.clip_asym)
            pre = get_op_name(model, layer) + '.'
            ent['prefix'] = pre
            torch.save(ent, path + '.tmp'); os.replace(path + '.tmp', path)
            print(f'[awq_table] layer {i}: {time.time() - t0:.1f}s -> {path}', flush=True)
        del feat
        inps = nxt
        layers[i] = layer.cpu()
        gc.collect(); torch.cuda.empty_cache()


# ─────────────────────────────── assemble ───────────────────────────────
def awq_results_from_table(table_dir, arch_w, n_layers, layers=None):
    """awq_results (same structure/order as run_awq) for per-layer W bits `arch_w`
    ({'self_attn.q_proj': [b_0..b_{L-1}], ...})."""
    res = {'scale': [], 'clip': []}
    for i in (range(n_layers) if layers is None else layers):
        ent = torch.load(os.path.join(table_dir, f'layer_{i:02d}.pt'))
        b = {n: int(arch_w[n][i]) for n in LINEARS}
        qkv, gu, d = tuple(b[n] for n in QKV), tuple(b[n] for n in GU), (b[DOWN[0]],)
        res['scale'] += append_str_prefix([ent['scale']['qkv'][qkv], ent['scale']['gateup'][gu],
                                           ent['scale']['down'][d]], ent['prefix'])
        ctx = {'self_attn.v_proj': qkv, 'self_attn.o_proj': (b['self_attn.o_proj'],),
               'mlp.gate_proj': gu, 'mlp.up_proj': gu + d, 'mlp.down_proj': d}
        res['clip'] += append_str_prefix([(n,) + ent['clip'][n][ctx[n]] for n in CLIP_ORDER],
                                         ent['prefix'])
    return res


class AWQTableModel:
    """A GPU model whose weights can be re-quantized for any arch from the table:
    restore the FP16 block params from a CPU master, apply the looked-up awq_results
    with the unchanged apply_awq. The table (~2.3 GB) is cached in RAM."""

    def __init__(self, model, table_dir, q_config, clip_asym=True):
        self.model, self.table_dir = model, table_dir
        self.q_config, self.clip_asym = q_config, clip_asym
        self.layers = get_blocks(model)
        self.n = len(self.layers)
        # FP16 block master: on the GPU when it fits (restoring it is then a device-local
        # copy; the pinned-CPU master cost ~0.56 s of H2D per build, finding 81), else pinned CPU.
        nbytes = sum(v.numel() * v.element_size() for l in self.layers for v in l.state_dict().values())
        dev = next(model.parameters()).device
        free = torch.cuda.mem_get_info(dev)[0] if dev.type == 'cuda' else 0
        on_gpu = free > nbytes + 24 * 2**30          # keep >= 24 GB headroom for activations
        self.master = [{k: (v.detach().clone() if on_gpu else v.detach().to('cpu', copy=True).pin_memory())
                        for k, v in l.state_dict().items()} for l in self.layers]
        print(f'[awq_table] FP16 master ({nbytes / 2**30:.1f} GB) on {"GPU" if on_gpu else "pinned CPU"}',
              flush=True)
        self._ent = {}

    def _entries(self):
        for i in range(self.n):
            if i not in self._ent:
                self._ent[i] = torch.load(os.path.join(self.table_dir, f'layer_{i:02d}.pt'))
        return self._ent

    @torch.no_grad()
    def build(self, arch_w):
        from .awq_utils.pre_quant import apply_awq
        for l, m in zip(self.layers, self.master):
            sd = l.state_dict()
            for k, v in m.items():
                sd[k].copy_(v, non_blocking=True)
        ent = self._entries()
        res = {'scale': [], 'clip': []}
        for i in range(self.n):
            b = {n: int(arch_w[n][i]) for n in LINEARS}
            qkv, gu, d = tuple(b[n] for n in QKV), tuple(b[n] for n in GU), (b[DOWN[0]],)
            e = ent[i]
            res['scale'] += append_str_prefix([e['scale']['qkv'][qkv], e['scale']['gateup'][gu],
                                               e['scale']['down'][d]], e['prefix'])
            ctx = {'self_attn.v_proj': qkv, 'self_attn.o_proj': (b['self_attn.o_proj'],),
                   'mlp.gate_proj': gu, 'mlp.up_proj': gu + d, 'mlp.down_proj': d}
            res['clip'] += append_str_prefix([(n,) + e['clip'][n][ctx[n]] for n in CLIP_ORDER],
                                             e['prefix'])
        apply_awq(self.model, res, q_config=self.q_config, arch=arch_w,
                  clip_asym=self.clip_asym, do_owq=False, outlier=None)
        return self.model


def cmd_bench(args):
    """wall time of one table-built arch on a resident GPU model (the search's cost)."""
    import glob as _g
    from transformers import AutoModelForCausalLM
    q_config = {'zero_point': True, 'q_group_size': args.group_size}
    model = AutoModelForCausalLM.from_pretrained(f'{args.model_path}/{args.model_name}',
                                                 torch_dtype=args.dtype, device_map='cuda')
    t0 = time.time()
    tm = AWQTableModel(model, args.out, q_config, args.clip_asym)
    tm._entries()
    print(f'[awq_table] bench setup (CPU master + table in RAM): {time.time() - t0:.1f}s', flush=True)
    archs = sorted(_g.glob(os.path.join(os.path.dirname(args.arch_json) or '.', '*.json')))
    for rep in range(2):
        for a in archs:
            w = json.load(open(a)); w = w.get('q', w).get('w', w)
            torch.cuda.synchronize(); t0 = time.time()
            tm.build(w)
            torch.cuda.synchronize()
            print(f'[awq_table] bench build {os.path.basename(a)} (pass {rep}): {time.time() - t0:.1f}s',
                  flush=True)


def cmd_check(args):
    """END-TO-END weight check: the resident table model, rebuilt for arch B right after a
    DIFFERENT arch A (so a restore bug would show), vs the deployed path
    get_quantized_model('awq', B). Every block parameter must be identical."""
    import glob as _g
    from transformers import AutoModelForCausalLM
    from .model import get_quantized_model
    q_config = {'zero_point': True, 'q_group_size': args.group_size}
    archs = sorted(_g.glob(os.path.join(os.path.dirname(args.arch_json), '*.json')))
    load = lambda a: (lambda w: w.get('q', w).get('w', w))(json.load(open(a)))
    B = load(args.arch_json); A = load([a for a in archs if a != args.arch_json][0])
    ref = get_quantized_model(method='awq', arch=B, model_name=f'{args.model_path}/{args.model_name}',
                              device_map={'': 0}, group_size=args.group_size, dtype=args.dtype)
    model = AutoModelForCausalLM.from_pretrained(f'{args.model_path}/{args.model_name}',
                                                 torch_dtype=args.dtype, device_map='cuda')
    tm = AWQTableModel(model, args.out, q_config, args.clip_asym)
    tm.build(A); tm.build(B)
    rsd, tsd = ref.state_dict(), model.state_dict()
    keys = [k for k in rsd if '.layers.' in k]
    d = max((rsd[k].float() - tsd[k].float().to(rsd[k].device)).abs().max().item() for k in keys)
    print(f'[awq_table] check {os.path.basename(args.arch_json)} (after building another arch): '
          f'{len(keys)} block tensors, max|diff| {d:.3g} {"EXACT" if d == 0 else "DIFFERS"}', flush=True)


# ─────────────────────────────── verify ───────────────────────────────
def _flat(res):
    s = {e[0] + '|' + ','.join(e[1]): e[2] for e in res['scale']}
    c = {e[0]: e[1:] for e in res['clip']}
    return s, c


def cmd_verify(args):
    """table vs run_awq on the same arch: every scale / clip tensor, max |diff|."""
    q_config = {'zero_point': True, 'q_group_size': args.group_size}
    arch_w = json.load(open(args.arch_json))
    arch_w = arch_w.get('q', arch_w).get('w', arch_w.get('w', arch_w))
    m = _load(args)
    t0 = time.time()
    ref = run_awq(m.model, m.tokenizer, q_config=q_config, arch=arch_w,
                  clip_asym=args.clip_asym, n_samples=args.n_samples, seqlen=args.seqlen,
                  calib_data=args.calib_data)
    t_ref = time.time() - t0
    t0 = time.time()
    n = len(get_blocks(m.model))
    built = [i for i in range(n) if os.path.exists(os.path.join(args.out, f'layer_{i:02d}.pt'))]
    tab = awq_results_from_table(args.out, arch_w, n, layers=built)
    t_tab = time.time() - t0
    pre = tuple(f'model.layers.{i}.' for i in built)       # compare only the built layers
    ref = {'scale': [e for e in ref['scale'] if e[0].startswith(pre)],
           'clip': [e for e in ref['clip'] if e[0].startswith(pre)]}
    rs, rc = _flat(ref); ts, tc = _flat(tab)
    print(f'[awq_table] comparing {len(built)}/{n} built layers', flush=True)
    assert rs.keys() == ts.keys() and rc.keys() == tc.keys(), 'entry names differ'
    ds = max((rs[k].float().cpu() - ts[k].float().cpu()).abs().max().item() for k in rs)
    dc = max((a.float().cpu() - b.float().cpu()).abs().max().item()
             for k in rc for a, b in zip(rc[k], tc[k]))
    exact = ds == 0 and dc == 0
    print(f'[awq_table] verify {os.path.basename(args.arch_json)}: {len(rs)} scales, '
          f'{len(rc)} clips | max|d scale| {ds:.3g}  max|d clip| {dc:.3g}  '
          f'{"EXACT" if exact else "DIFFERS"} | run_awq {t_ref:.0f}s, lookup {t_tab:.1f}s',
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['build', 'verify', 'bench', 'check'])
    p.add_argument('--model_path', default='/SSD/huggingface/meta-llama')
    p.add_argument('--model_name', default='Llama-3.1-8B-Instruct')
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--group_size', type=int, default=128)
    p.add_argument('--bits', type=int, nargs='+', default=[2, 3, 4])
    p.add_argument('--n_samples', type=int, default=128)     # AWQ.run() defaults
    p.add_argument('--seqlen', type=int, default=512)
    p.add_argument('--calib_data', default='pileval')
    p.add_argument('--clip_asym', type=int, default=1)       # AWQ class default: True
    p.add_argument('--out', required=True)
    p.add_argument('--layers', default='', help='a:b — build only these layers (parallel)')
    p.add_argument('--arch_json', default='')
    args = p.parse_args()
    args.clip_asym = bool(args.clip_asym)
    args.dtype = getattr(torch, args.dtype) if args.dtype != 'auto' else 'auto'
    {'build': cmd_build, 'verify': cmd_verify, 'bench': cmd_bench, 'check': cmd_check}[args.cmd](args)


if __name__ == '__main__':
    main()
