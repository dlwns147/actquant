"""Vectorised candidate-set / metric / sampling helpers used by post_search_split.py.

Extracted from main() so the heavy nested closures don't obscure the orchestration
flow and can be unit-tested in isolation. None of these helpers touch GPU or load
models — they operate on numpy arrays + arch dicts.

Glossary:
- expr_keys : tuple/list naming each search-axis ('w', 'kv', 'kvdim', 'eff_kv')
- esm       : dict mapping expr_key → 1D ndarray of subnet dicts (one per axis)
- efm       : dict mapping expr_key → (n, 2) F-array [metric, comp_obj] per axis
- nd_idx    : (N, n_dims) int64 array indexing into each axis (one column per dim)
"""

import itertools
import numpy as np

from utils.func import get_net_info


# ───────────────────────── arch construction ─────────────────────────

def axis_of_map(expr_keys):
    """Map a per-arch metric key to the expr axis it depends on.

    Returns None for keys that don't reduce to a single axis under the current
    expr_keys (e.g. 'memory' always spans w + kv + kvdim).

    `metric_*` keys are per-axis search-time JSD/loss values read from each
    axis's own PF archive (column 0 of `efm[axis]`). They map 1-to-1 to the
    expr axis name regardless of whether `eff_kv` is in expr_keys."""
    has_eff = 'eff_kv' in expr_keys
    return {
        'wbits':         'w',
        'kvbits':        'eff_kv' if has_eff else 'kv',
        'kbits':         'eff_kv' if has_eff else 'kv',
        'vbits':         'eff_kv' if has_eff else 'kv',
        'kvdim':         'eff_kv' if has_eff else 'kvdim',
        'kdim':          'eff_kv' if has_eff else 'kvdim',
        'vdim':          'eff_kv' if has_eff else 'kvdim',
        'eff_kvbits':    'eff_kv' if has_eff else None,
        'eff_kbits':     'eff_kv' if has_eff else None,
        'eff_vbits':     'eff_kv' if has_eff else None,
        'metric_w':      'w',
        'metric_kv':     'kv',
        'metric_kvdim':  'kvdim',
        'metric_eff_kv': 'eff_kv',
    }


def build_arch(default_arch, expr_keys, esm, nd_idx_row):
    """Merge per-axis subnets into a full arch dict for one ND-index row."""
    arch = {
        'q': {'w': default_arch['q']['w'],
              'k': default_arch['q']['k'],
              'v': default_arch['q']['v']},
        'p': {'k': default_arch['p']['k'],
              'v': default_arch['p']['v']},
    }
    for dim_i, key in enumerate(expr_keys):
        sv = esm[key][nd_idx_row[dim_i]]
        if key == 'w':
            arch['q']['w'] = sv['q']['w']
        elif key == 'kv':
            arch['q']['k'] = sv['q']['k']; arch['q']['v'] = sv['q']['v']
        elif key == 'kvdim':
            arch['p']['k'] = sv['p']['k']; arch['p']['v'] = sv['p']['v']
        elif key == 'eff_kv':
            arch['q']['k'] = sv['q']['k']; arch['q']['v'] = sv['q']['v']
            arch['p']['k'] = sv['p']['k']; arch['p']['v'] = sv['p']['v']
    return arch


class LazyPs:
    """Build arch dict on access. Avoids materialising up to ~32M dicts when the
    candidate set is large; the I-selection step usually only touches O(50)."""
    def __init__(self, builder, nd_idx):
        self._build = builder
        self._idx = nd_idx
    def __len__(self):
        return len(self._idx)
    def __getitem__(self, k):
        if isinstance(k, (int, np.integer)):
            return self._build(self._idx[int(k)])
        if isinstance(k, slice):
            return [self._build(self._idx[i]) for i in range(*k.indices(len(self)))]
        return [self._build(self._idx[int(i)]) for i in k]


# ───────────────────────── metric extraction ─────────────────────────

def per_axis_metric(key, expr_keys, esm, config, group_size, n_token, cache=None):
    """(axis_index, per-axis array) for single-axis keys; None otherwise.

    `cache` (optional dict) memoises results across calls so the same axis array
    isn't rebuilt for every quantile spec."""
    if cache is not None and key in cache:
        return cache[key]
    axis = axis_of_map(expr_keys).get(key)
    result = None
    if axis is not None and axis in expr_keys:
        arr = np.array([
            get_net_info(s, config, group_size, n_token=n_token)[key]
            for s in esm[axis]
        ])
        result = (expr_keys.index(axis), arr)
    if cache is not None:
        cache[key] = result
    return result


def metric_over(nd_idx, key, expr_keys, esm, default_arch,
                config, group_size, n_token, cache=None):
    """Vectorised metric lookup over nd_idx; falls back to per-arch get_net_info
    only for multi-axis keys (e.g. 'memory') that can't be derived per axis."""
    pa = per_axis_metric(key, expr_keys, esm, config, group_size, n_token, cache)
    if pa is not None:
        ax, arr = pa
        return arr[nd_idx[:, ax]]
    return np.array([
        get_net_info(build_arch(default_arch, expr_keys, esm, nd_idx[i]),
                     config, group_size, n_token=n_token)[key]
        for i in range(len(nd_idx))
    ])


# ───────────────────────── sampling ─────────────────────────

def draw_random(n_draw, n_pool, exclude=(), rng=None):
    """Sorted list of ≤ n_draw distinct ints in [0, n_pool) excluding `exclude`."""
    if n_draw <= 0 or n_pool == 0:
        return []
    if rng is None:
        rng = np.random
    if not exclude:
        n = int(min(n_draw, n_pool))
        return sorted(int(x) for x in rng.choice(n_pool, size=n, replace=False))
    mask = np.ones(n_pool, dtype=bool)
    mask[np.fromiter(exclude, dtype=np.int64)] = False
    avail = np.flatnonzero(mask)
    n = int(min(n_draw, len(avail)))
    if n == 0:
        return []
    return sorted(int(x) for x in rng.choice(avail, size=n, replace=False))


# ───────────────────── candidate-set construction ─────────────────────

def select_valid_nd_idx(nd_shape, new_metric_nd, comp_nd_list,
                        comp_obj_min, comp_obj_max,
                        random_sample, has_quantile, has_prefer,
                        verbose=True, rng=None):
    """Build the candidate set, sorted by combined metric ascending.

    Three mutually exclusive paths:
      1) comp_obj filter set            → mask → optional sub-sample
      2) random_sample only & no filter → fast path, skip n_total sort
      3) otherwise                       → full sort over n_total
    """
    if rng is None:
        rng = np.random
    has_filter = len(comp_obj_min) > 0
    only_random = (random_sample is not None and not has_quantile and not has_prefer)
    n_total = int(np.prod(nd_shape))

    if has_filter:
        mask = np.ones(nd_shape, dtype=bool)
        for comp_nd, lo, hi in zip(comp_nd_list, comp_obj_min, comp_obj_max):
            mask &= (comp_nd >= lo) & (comp_nd <= hi)
        valid_nd_idx = np.argwhere(mask)
        if verbose:
            print(f'range_idx : {len(valid_nd_idx)}')
        if len(valid_nd_idx) == 0 and comp_nd_list and verbose:
            first_nd = np.asarray(comp_nd_list[0])
            print(f'[debug] comp_obj[0] range in results: '
                  f'min={first_nd.min():.3f}, max={first_nd.max():.3f}')
            print(f'[debug] comp_obj_min={comp_obj_min}, comp_obj_max={comp_obj_max}')
        if only_random and len(valid_nd_idx) > random_sample:
            chosen = rng.choice(len(valid_nd_idx), size=random_sample, replace=False)
            valid_nd_idx = valid_nd_idx[chosen]
        return valid_nd_idx[np.argsort(new_metric_nd[tuple(valid_nd_idx.T)])]

    if only_random:
        n_draw = min(random_sample, n_total)
        flat = rng.choice(n_total, size=n_draw, replace=False)
        valid_nd_idx = np.stack(np.unravel_index(flat, nd_shape), axis=1)
        return valid_nd_idx[np.argsort(new_metric_nd[tuple(valid_nd_idx.T)])]

    sort_order = np.argsort(new_metric_nd.ravel())
    return np.stack(np.unravel_index(sort_order, nd_shape), axis=1)


def assemble_F(valid_nd_idx, expr_keys, efm, comp_nd_list, new_metric_nd):
    """Build F = [combined_metric | per-component metrics | comp_obj_vals]."""
    vt = tuple(valid_nd_idx.T)
    parts = [new_metric_nd[vt].reshape(-1, 1),
             np.column_stack([efm[k][valid_nd_idx[:, i]]
                              for i, k in enumerate(expr_keys)])]
    if comp_nd_list:
        parts.append(np.column_stack([np.asarray(nd)[vt] for nd in comp_nd_list]))
    return np.column_stack(parts)


# ───────────────────────── selection ─────────────────────────

def quantile_select(quantile_specs, valid_nd_idx, expr_keys, esm, default_arch,
                    config, group_size, n_token, axis_cache=None, efm=None,
                    verbose=True):
    """Pick architecture indices at the cartesian product of per-metric quantile
    positions. Returns (sorted unique I_quant, dict of metric_vals).

    `efm` (optional dict {axis: (n,2) array}) is required when any spec key
    starts with `metric_` — that key reads the per-axis search-time metric
    (loss / JSD) from `efm[axis][nd_idx, 0]` rather than from `get_net_info`."""
    _axis_map = axis_of_map(expr_keys)
    metric_vals = {}
    for k in quantile_specs:
        if k.startswith('metric_'):
            ax_name = _axis_map.get(k)
            if ax_name is None or ax_name not in expr_keys:
                raise ValueError(f"quantile_sample '{k}' needs expr axis "
                                 f"'{ax_name}' but it is not in expr_keys "
                                 f"{expr_keys}")
            if efm is None or ax_name not in efm:
                raise ValueError(f"quantile_sample '{k}' requires efm dict "
                                 f"containing axis '{ax_name}' (its (metric, "
                                 f"comp) array); pass efm= to quantile_select")
            ax_i = expr_keys.index(ax_name)
            metric_vals[k] = efm[ax_name][valid_nd_idx[:, ax_i], 0]
        else:
            metric_vals[k] = metric_over(valid_nd_idx, k, expr_keys, esm,
                                         default_arch, config, group_size,
                                         n_token, cache=axis_cache)
    target_vals = {k: [np.quantile(v, q) for q in quantile_specs[k]]
                   for k, v in metric_vals.items()}
    if verbose:
        for k, v in metric_vals.items():
            print(f'[quantile_sample] {k}: range=[{v.min():.4f}, {v.max():.4f}]')
            print(f'[quantile_sample] {k}: targets='
                  f'{[f"{t:.4f}" for t in target_vals[k]]}')

    keys = list(quantile_specs.keys())
    I_set = set()
    for combo in itertools.product(*[range(len(quantile_specs[k])) for k in keys]):
        targets = {k: target_vals[k][qi] for k, qi in zip(keys, combo)}
        dists = np.zeros(len(valid_nd_idx))
        for k, t in targets.items():
            v = metric_vals[k]
            rng = v.max() - v.min()
            dists += ((v - t) / rng) ** 2 if rng > 0 else (v - t) ** 2
        I_set.add(int(np.argmin(dists)))
    return sorted(I_set), metric_vals
