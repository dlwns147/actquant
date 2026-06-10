"""AWQ-based QEFT (outlier-aware weight quantization on top of AWQ).

QEFT / OWQ keeps the most sensitive *input columns* of each weight in FP16 and
quantizes the rest.  The repo's QEFT (``quant/qeft.py``) is **GPTQ-based**: it
reconstructs with the Hessian and reorders columns for a fast kernel.  This file
adds the **AWQ-based** counterpart: run the standard AWQ activation-aware
scale/clip search, then pseudo-quantize every linear while holding its outlier
columns in FP16.

How it differs from the existing AWQ ``do_owq`` path (``awq_utils/pre_quant.py``):

* The legacy path triggers OWQ on **fractional bits** (``is_owq`` →
  ``round(b) != b``) and consumes a **flat** ``{key: [col indices]}`` dict — the
  outlier *count* is fixed at extraction time.
* The new weight-search axis encodes each layer as a ``(w_bits, n_outlier)``
  pair with ``n_outlier ∈ {0, 32, 64, 96, 128}`` (see
  ``search_space/llama_qeft.py``).  The count is **searchable per layer**, so the
  outlier indices are resolved on demand from the *multi* dict produced by
  ``extract_outidx.py`` (``{key: {n_out: [col indices]}}``) at the count
  this arch selected.  The OWQ trigger becomes ``n_outlier > 0`` (bits stay
  integer).

The pseudo-quant numerics are identical to QEFT's fake-quant: outlier columns are
bit-exact FP16, the rest is grouped affine-quantized.  (QEFT's column *reorder*
is a kernel-packing optimization and does not change fake-quant outputs.)
"""

import gc
from copy import deepcopy

import torch
import torch.nn as nn
from accelerate import dispatch_model

from .awq import AWQ
from .awq_utils.pre_quant import (
    run_awq, get_blocks, get_named_linears,
)
from .awq_utils.module import append_str_prefix, get_op_name
from .awq_utils.auto_scale import apply_scale
from .awq_utils.auto_clip import apply_clip_asym, apply_clip_sym
from .awq_utils.quantizer import pseudo_quantize_tensor


# --------------------------------------------------------------------------- #
# arch helpers
# --------------------------------------------------------------------------- #
def split_bits_outlier(w_arch):
    """Split a ``(bits, n_outlier)`` weight arch into two parallel arches.

    ``w_arch`` maps ``linear_name -> [per-block entry]`` where each entry is
    ``(bits, n_outlier)`` (list or tuple) or a bare int ``bits`` (n_outlier=0).
    Returns ``(bits_arch, n_outlier_arch)`` with the same keys, each value a
    per-block list of ints.
    """
    bits_arch, n_out_arch = {}, {}
    for name, entries in w_arch.items():
        bits, nouts = [], []
        for e in entries:
            if isinstance(e, (list, tuple)):
                b, c = int(e[0]), int(e[1])
            else:
                b, c = int(e), 0
            bits.append(b)
            nouts.append(c)
        bits_arch[name] = bits
        n_out_arch[name] = nouts
    return bits_arch, n_out_arch


@torch.no_grad()
def build_outlier_multidict(model, calib_loader, config, ranks,
                            prefix='model.layers', device='cuda:0'):
    """Compute OWQ outlier columns for ANY model via a calibration pass.

    Replicates ``extract_outidx.py``'s selection but model-agnostic and
    robust under transformers>=4.45 (no Catcher / inp_kwargs replay): the OWQ
    sensitivity is the Hessian diagonal ``H_ii = sum_t x_{t,i}^2`` (per input
    column).  qkv/gate/up share a GLOBAL top-k over hidden channels (sensitivity
    summed + mean-normalized across those layers, exactly like the extract
    script); down_proj is per-layer.  Returns the same
    ``{key: {rank: [cols]}}`` (+ ``'n_outlier'``) structure as the on-disk dict.
    """
    linears = config['linear']
    owq_linears = [l for l in linears if l.split('.')[-1] != 'o_proj']
    hidden = int(config['hidden_size'])
    nb = int(config['n_block'])
    layers = model.model.layers

    def _getsub(mod, name):
        for p in name.split('.'):
            mod = getattr(mod, p)
        return mod

    state = {'global': torch.zeros(hidden, device=device), 'down': {}}
    handles = []

    def _mk(key, is_down):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            ss = (x * x).sum(0)
            if is_down:
                state['down'][key] = state['down'].get(key, torch.zeros_like(ss)) + ss
            else:
                state['global'] += ss / ss.mean().clamp_min(1e-9)
        return hook

    for i in range(nb):
        for l in owq_linears:
            key = f'{prefix}.{i}.{l}'
            handles.append(_getsub(layers[i], l).register_forward_hook(
                _mk(key, l.split('.')[-1] == 'down_proj')))
    for batch in calib_loader:
        inp = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(inp.to(device))
    for h in handles:
        h.remove()

    gorder = torch.argsort(state['global'], descending=True).cpu().tolist()
    out = {'n_outlier': list(ranks)}
    for i in range(nb):
        for l in owq_linears:
            key = f'{prefix}.{i}.{l}'
            if l.split('.')[-1] == 'down_proj':
                order = torch.argsort(state['down'][key], descending=True).cpu().tolist()
            else:
                order = gorder
            out[key] = {r: sorted(order[:r]) for r in ranks}
    return out


def resolve_outliers(outlier_multidict, n_outlier_arch, prefix='model.layers'):
    """Resolve per-(block, linear) outlier counts to a flat index dict.

    ``outlier_multidict`` is the ``extract_outidx.py`` output:
    ``{f'{prefix}.{i}.{name}': {n_out: [col indices]}}`` (plus an ``'n_outlier'``
    metadata key listing the available counts).  ``n_outlier_arch`` is
    ``{name: [n_out per block]}``.  Returns ``{full_key: [col indices]}`` holding
    only the entries with ``n_out > 0`` (so non-outlier layers are untouched).
    """
    available = set(outlier_multidict.get('n_outlier', []))
    resolved = {}
    for name, per_block in n_outlier_arch.items():
        for i, n_out in enumerate(per_block):
            if n_out <= 0:
                continue
            key = f'{prefix}.{i}.{name}'
            if key not in outlier_multidict:
                raise KeyError(f'{key} not in outlier dict (have {len(outlier_multidict)} keys)')
            table = outlier_multidict[key]
            if n_out not in table:
                raise KeyError(
                    f'n_outlier={n_out} not extracted for {key}; '
                    f'available counts: {sorted(available)}')
            resolved[key] = list(table[n_out])
    return resolved


# --------------------------------------------------------------------------- #
# apply: AWQ scale/clip + outlier-preserving pseudo-quant
# --------------------------------------------------------------------------- #
@torch.no_grad()
def apply_awq_qeft(model, awq_results, q_config, bits_arch, resolved_outlier, clip_asym):
    """Apply AWQ scale/clip, then pseudo-quantize keeping FP16 outlier columns.

    Mirrors ``awq_utils.pre_quant.apply_awq`` but the OWQ trigger is
    ``key in resolved_outlier`` (explicit ``n_outlier > 0``) rather than
    fractional bits, and the bit-width comes from ``bits_arch`` (integer).
    """
    apply_scale(model, awq_results["scale"])
    if clip_asym:
        # clip skips the outlier columns it is given (they stay FP16-precise)
        apply_clip_asym(model, awq_results["clip"], do_owq=bool(resolved_outlier),
                        outlier=resolved_outlier)
    else:
        apply_clip_sym(model, awq_results["clip"])

    layers = get_blocks(model)
    for i, layer in enumerate(layers):
        named_linears = get_named_linears(layer)
        for n, m in named_linears.items():
            key = append_str_prefix(n, get_op_name(model, layer) + ".")
            bits = int(bits_arch[n][i])
            has_outlier = key in resolved_outlier and bits < 16

            if has_outlier:
                cols = resolved_outlier[key]
                orig = m.weight.data[:, cols].detach().clone()
                m.weight.data[:, cols] = 0

            if bits < 16:
                m.weight.data = pseudo_quantize_tensor(
                    m.weight.data, n_bit=bits, **q_config)

            if has_outlier:
                m.weight.data[:, cols] = orig
                del orig
    torch.cuda.empty_cache()
    gc.collect()


# --------------------------------------------------------------------------- #
# AWQ_QEFT class — drop-in alongside AWQ / GPTQ / QEFT
# --------------------------------------------------------------------------- #
class AWQ_QEFT(AWQ):
    """AWQ + searchable per-layer FP16 outlier columns.

    ``arch`` is the weight arch ``{linear_name: [(bits, n_outlier), ...]}``.
    ``owq`` is the multi-outlier dict from ``extract_outidx.py`` (either a
    path or the loaded dict); it MUST contain every requested count for the
    layers that select ``n_outlier > 0``.
    """

    def __init__(self, model_name, config, arch, device_map, group_size=128,
                 dtype='auto', dev='cuda', prune=False, do_owq=True, owq=None,
                 outlier_prefix='model.layers', **kwargs):
        super().__init__(model_name, config, arch, device_map=device_map,
                         group_size=group_size, dtype=dtype, dev=dev, prune=prune,
                         do_owq=do_owq, owq=owq, **kwargs)
        # NOTE: BASE.__init__ keys outlier loading off `outlier_path`, but the
        # AWQ/GPTQ/QEFT subclasses forward `owq=` (which lands in BASE **kwargs),
        # so BASE leaves self.owq=None. Populate it explicitly here.
        if do_owq and self.owq is None and owq is not None:
            self.owq = torch.load(owq, weights_only=False) if isinstance(owq, str) else owq
        self.method = 'awq_qeft'
        self.outlier_prefix = outlier_prefix
        self.bits_arch, self.n_outlier_arch = split_bits_outlier(arch)

    def run(self, nsamples=128, seqlen=512, no_zero_point=False):
        q_config = {"zero_point": not no_zero_point, "q_group_size": self.group_size}
        print("AWQ-QEFT quantization config:", q_config)
        self.model.config.use_cache = False

        resolved = resolve_outliers(self.owq, self.n_outlier_arch, self.outlier_prefix) \
            if self.do_owq else {}
        print(f"AWQ-QEFT: {len(resolved)} (block,linear) layers carry FP16 outliers")

        # AWQ scale/clip search uses integer bits; outliers are excluded from the
        # clip range search so they are not distorted before being restored.
        awq_results = run_awq(
            self.model, self.tokenizer, q_config=q_config, arch=self.bits_arch,
            clip_asym=self.clip_asym, n_samples=nsamples, seqlen=seqlen,
            do_owq=bool(resolved), outlier=resolved,
        )
        self.load_model(device_map='cpu', dtype=self.dtype)
        self.model = dispatch_model(self.model, self.device_map)
        apply_awq_qeft(self.model, awq_results, q_config, self.bits_arch,
                       resolved, self.clip_asym)
        torch.cuda.empty_cache()
        gc.collect()
