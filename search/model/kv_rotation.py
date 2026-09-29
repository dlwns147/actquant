"""KV Hadamard rotation as a global eval-time primitive (finding 48 measurement).

RotateKV / QuaRot-style: quantise K and V in a Hadamard-rotated head_dim basis,
    fake_quant_rot(x) = unrot(fake_quant(rot(x)))       rot(x) = x @ H,  H orthonormal
Faithful to the deployed form because attention is rotation-invariant per head:
(Q H)(K H)^T == Q K^T, and the V output is un-rotated once. In this FAKE-quant
form the rotation is applied and undone around every quantisation, so it costs
two head_dim x head_dim matmuls per quantised block — an UPPER bound on the
deployed cost, where K/V are rotated once at write and only Q per decode step.

It is injected by re-binding the name `fake_quant` in every module that imported
it from quant.kivi_utils.new_pack (model.kivi_utils, model.KIVICache and the
per-architecture converters), so the prefill path (quant_kv_output) and the
cache path (KIVIFakeCache, incl. _fake_quant_k_excl_sink) both pick it up with no
change to search / evaluator / post_search code. Like attn_sink it is NOT stored
in the arch and must be set identically wherever a number is meant to be
comparable — correlation.py puts it in the measurement config for that reason.
"""
import importlib
import math
import os

import torch

_STATE = {'enabled': False, 'H': {}, 'orig': None, 'patched': [],
          'parts': 'kv', 'k_scheme': 'channel', 'v_scheme': 'token',
          'basis': 'hadamard', 'basis_seed': 0}
_MODULES = ('model.kivi_utils', 'model.KIVICache', 'model.llama_kivi',
            'model.qwen2_kivi', 'model.mistral_kivi', 'model.gemma3_kivi')


def hadamard(n, device, dtype):
    """Normalised Sylvester Hadamard matrix (n a power of two): H @ H.T == I."""
    assert n & (n - 1) == 0, f"head_dim {n} is not a power of two"
    key = (n, str(device), dtype)
    if key not in _STATE['H']:
        h = torch.ones(1, 1, dtype=torch.float32)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _STATE['H'][key] = (h / math.sqrt(n)).to(device=device, dtype=dtype)
    return _STATE['H'][key]


def random_orthogonal(n, device, dtype, seed=0):
    """A fixed random orthogonal matrix (QR of a seeded gaussian). The CONTROL for
    'does the basis matter, or is any energy-spreading rotation as good?' — if this
    matches Hadamard, a data-driven basis (OSCAR-style PCA) is unlikely to pay."""
    key = ('rand', n, str(device), dtype, seed)
    if key not in _STATE['H']:
        g = torch.Generator(device='cpu').manual_seed(seed)
        q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float32))
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)      # unique QR -> reproducible
        _STATE['H'][key] = q.to(device=device, dtype=dtype)
    return _STATE['H'][key]


def basis(n, device, dtype):
    if _STATE['basis'] == 'random':
        return random_orthogonal(n, device, dtype, _STATE['basis_seed'])
    return hadamard(n, device, dtype)


def _rotates(along):
    """Which tensor is this call quantising, and do we rotate it?
    `along` is the scheme the call site passed (k_quant_scheme for K, v_quant_scheme
    for V), so it identifies the tensor as long as the two differ — a partial setting
    asserts that in enable_kv_rotation()."""
    parts = _STATE['parts']
    if parts == 'kv':
        return True
    if along == _STATE['k_scheme']:
        return 'k' in parts
    if along == _STATE['v_scheme']:
        return 'v' in parts
    return True                       # unknown scheme: rotate rather than silently skip


def _rotated_fake_quant(inp, group_size, bits, along='channel', attention_mask=None):
    if not _rotates(along):
        return _STATE['orig'](inp, group_size, bits, along, attention_mask)
    H = basis(inp.shape[-1], inp.device, inp.dtype)
    out = _STATE['orig'](inp @ H, group_size, bits, along, attention_mask)
    return out @ H.T


def enable_kv_rotation(parts=None, k_scheme='channel', v_scheme='token',
                       basis_name=None, basis_seed=0):
    """Re-bind fake_quant to the rotated version everywhere it was imported.

    parts: 'kv' (default) rotates both tensors; 'k' / 'v' rotate only that one — the
    ablation for WHICH tensor the primitive actually fixes (keys carry the outlier
    channels per KVQuant / RotateKV, so rotating V may be dead weight). Taken from
    $KV_ROTATE_PARTS when not passed."""
    parts = (parts or os.environ.get('KV_ROTATE_PARTS') or 'kv').lower()
    assert parts in ('kv', 'k', 'v'), f'KV_ROTATE_PARTS must be kv|k|v, got {parts!r}'
    if parts != 'kv':
        assert k_scheme != v_scheme, (
            f'cannot rotate only {parts!r}: K and V share the scheme {k_scheme!r}')
    basis_name = (basis_name or os.environ.get('KV_ROTATE_BASIS') or 'hadamard').lower()
    assert basis_name in ('hadamard', 'random'), f'unknown basis {basis_name!r}'
    _STATE.update(parts=parts, k_scheme=k_scheme, v_scheme=v_scheme,
                  basis=basis_name, basis_seed=int(basis_seed))
    if _STATE['enabled']:
        return
    from quant.kivi_utils import new_pack
    _STATE['orig'] = new_pack.fake_quant
    for name in _MODULES:
        try:
            m = importlib.import_module(name)
        except Exception:                                   # noqa: BLE001
            continue
        if getattr(m, 'fake_quant', None) is _STATE['orig']:
            setattr(m, 'fake_quant', _rotated_fake_quant)
            _STATE['patched'].append(name)
    new_pack.fake_quant = _rotated_fake_quant
    _STATE['enabled'] = True
    # spawn-context workers (utils/awq_pool) re-import every module and would lose the
    # re-binding; they inherit os.environ and call enable_from_env() on start-up.
    os.environ['ACTQUANT_KV_ROTATE'] = _STATE['parts']
    os.environ['ACTQUANT_KV_ROTATE_BASIS'] = _STATE['basis']
    print(f"[kv_rotation] ENABLED: Hadamard rotation around KV fake_quant in "
          f"{_STATE['patched']} (+ quant.kivi_utils.new_pack); parts={_STATE['parts']} "
          f"(K={_STATE['k_scheme']}, V={_STATE['v_scheme']}); basis={_STATE['basis']}"
          f"{'' if _STATE['basis'] != 'random' else ' seed=%d' % _STATE['basis_seed']}")


def is_enabled():
    return _STATE['enabled']


def self_test(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """rot/unrot round-trip and attention invariance; run before any measurement."""
    from quant.kivi_utils import new_pack
    x = torch.randn(1, 4, 256, 128, device=device, dtype=torch.float16)
    H = hadamard(128, device, torch.float16)
    assert torch.allclose(x @ H @ H.T, x, atol=2e-2), 'H is not orthonormal at fp16'
    q = torch.randn(1, 4, 16, 128, device=device, dtype=torch.float16)
    a = q @ x.transpose(-1, -2); b = (q @ H) @ (x @ H).transpose(-1, -2)
    assert torch.allclose(a, b, atol=0.5, rtol=2e-2), 'attention is not rotation-invariant'
    plain = new_pack.fake_quant if _STATE['orig'] is None else _STATE['orig']
    e0 = (plain(x, 128, 2, 'channel') - x).pow(2).mean().item()
    e1 = ((plain(x @ H, 128, 2, 'channel') @ H.T) - x).pow(2).mean().item()
    print(f"[kv_rotation] self_test ok; 2-bit channel quant MSE on gaussian data: "
          f"plain {e0:.4g}  rotated {e1:.4g} (gaussian data has no outliers, so ~equal is expected)")


# ── pipeline default (2026-09-23): rotation is ON for search and post_search ─────
# Findings 51/60/66: +4..+10 RULER on both Llama and Qwen at zero memory cost, and the
# proxy is nearly blind to it, so it is fixed OUTSIDE the search rather than searched.
# Every search / post_search entry point shares these two helpers so the setting cannot
# drift between the stage that ranks archs and the stage that benchmarks them.
# correlation.py keeps its old opt-in flag: its pools hold UNROTATED labels at the root.
def add_args(parser):
    import argparse
    parser.add_argument('--kv_rotate', action=argparse.BooleanOptionalAction, default=True,
                        help='Hadamard-rotate K/V around every KV fake-quant (default ON; '
                             '--no-kv_rotate for the unrotated pipeline). Must match between '
                             'search and post_search.')
    parser.add_argument('--kv_rotate_parts', choices=['kv', 'k', 'v'], default='kv',
                        help='(with --kv_rotate) which tensor to rotate.')
    return parser


def setup_from_args(args):
    if getattr(args, 'kv_rotate', False):
        self_test()
        enable_kv_rotation(parts=getattr(args, 'kv_rotate_parts', 'kv'))
    else:
        os.environ.pop('ACTQUANT_KV_ROTATE', None)
        print('[kv_rotation] OFF (--no-kv_rotate)')


def enable_from_env():
    parts = os.environ.get('ACTQUANT_KV_ROTATE')
    if parts:
        enable_kv_rotation(parts=parts,
                           basis_name=os.environ.get('ACTQUANT_KV_ROTATE_BASIS', 'hadamard'))
