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

from utils.func import get_net_info, _LazyComp


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

# ───────────── lazy (no --expr_front) comp_obj-first pruning ─────────────
# Every comp_obj is low-rank (single-axis 1D / eff_kvbits 2D / additive
# memory = w-1D + cache-2D). With a comp_obj window the feasible set is
# enumerated EXACTLY here without any nd_shape array, returning the same
# (valid_nd_idx, F) the dense path would — F[:,0] is the additive combined
# metric (Σ scale·JSD), so post_search ranks correctly WITH or WITHOUT a
# surrogate (surrogate, if given, overwrites it then NaN-checks). Candidates
# are also sorted by the first comp_obj value ascending (deterministic order;
# post_search re-ranks by F[:,0] anyway).

# Cap on the POST-filter feasible set (not the full no-NDS product). Raised
# 5e7 → 1e9 now that the enumeration fill is vectorised (np.repeat/np.tile,
# no per-row python loop): 1e9 rows enumerate in a few minutes and fit RAM
# (~140-200 GB peak through F/M_valid/sort on this 503 GB box; the chunked
# GP predict never materialises the full kernel). NOTE: this is a wall-clock /
# RAM guard only — a feasible set this large means the surrogate ranks far
# OUTSIDE its training band, so widen --comp_obj_min/max deliberately, not by
# accident.
_LAZY_MAX_FEASIBLE = 1e9


def _interval(v, lo, hi):
    return (np.asarray(v) >= lo) & (np.asarray(v) <= hi)


def _memory_block(spec, lo, hi):
    """(group_axes, allowed idx-tuples) for an additive memory window."""
    kv = spec['kv']
    if kv['kind'] == 'scalar':
        kv_vals = np.array([kv['vals']], np.float64)
        kv_idx = np.zeros((1, 0), np.int64)
        kv_axes = []
    elif kv['kind'] == '1d':
        kv_vals = np.asarray(kv['vals'], np.float64)
        kv_idx = np.arange(len(kv_vals), dtype=np.int64)[:, None]
        kv_axes = [kv['axis']]
    else:
        kv_vals = np.asarray(kv['vals'], np.float64).ravel()
        n0, n1 = np.asarray(kv['vals']).shape
        gj, gk = np.meshgrid(np.arange(n0), np.arange(n1), indexing='ij')
        kv_idx = np.stack([gj.ravel(), gk.ravel()], 1).astype(np.int64)
        kv_axes = list(spec['kv']['axes'])
    if spec['w_axis'] is None:
        sel = _interval(kv_vals + spec['w_const'], lo, hi)
        return kv_axes, kv_idx[sel]
    w_mem = np.asarray(spec['w_mem'], np.float64)
    order = np.argsort(kv_vals, kind='stable')
    kv_sorted = kv_vals[order]
    group = [spec['w_axis']] + kv_axes
    # Pass 1: count feasible rows via searchsorted only (O(Nw·logNkv), no
    # allocation) and abort BEFORE materialising — a loose memory window on
    # full archives is billions of rows (~100s of GiB) and must fail fast.
    L = np.searchsorted(kv_sorted, lo - w_mem, side='left')
    R = np.searchsorted(kv_sorted, hi - w_mem, side='right')
    counts = np.maximum(R - L, 0)
    total = int(counts.sum())
    if total > _LAZY_MAX_FEASIBLE:
        raise SystemExit(
            f"[lazy] feasible {total:.3e} > {_LAZY_MAX_FEASIBLE:.0e} (memory window admits "
            f"{total:.3e} (w,kv) pairs of the full no-NDS product). The "
            f"--comp_obj_min/--comp_obj_max window is far too loose for "
            f"unfiltered archives — tighten it (e.g. ±0.1% not ±5%), or pass "
            f"--expr_front to Pareto-filter the per-axis archives first.")
    if total == 0:
        return group, np.empty((0, len(group)), np.int64)
    # Pass 2: now safe to materialise (≤ cap rows)
    pieces = []
    for wi in range(len(w_mem)):
        l, r = int(L[wi]), int(R[wi])
        if r <= l:
            continue
        sel = order[l:r]
        blk = np.empty((len(sel), 1 + kv_idx.shape[1]), np.int64)
        blk[:, 0] = wi
        if kv_idx.shape[1]:
            blk[:, 1:] = kv_idx[sel]
        pieces.append(blk)
    return group, np.concatenate(pieces, 0)


def _join_groups(g1, t1, g2, t2):
    """Inner-join two (axes, idx-tuple) constraint sets on shared axes."""
    shared = [a for a in g1 if a in g2]
    if not shared:
        g = list(g1) + list(g2)
        return g, np.concatenate(
            [np.repeat(t1, len(t2), 0), np.tile(t2, (len(t1), 1))], 1)
    from collections import defaultdict
    c1 = [g1.index(a) for a in shared]
    c2 = [g2.index(a) for a in shared]
    mm = defaultdict(list)
    for i, r in enumerate(t1):
        mm[tuple(r[c1].tolist())].append(i)
    extra2 = [j for j in range(len(g2)) if g2[j] not in shared]
    g = list(g1) + [g2[j] for j in extra2]
    rows = [np.concatenate([t1[i], r2[extra2]])
            for r2 in t2 for i in mm.get(tuple(r2[c2].tolist()), ())]
    if not rows:
        return g, np.empty((0, len(g)), np.int64)
    return g, np.stack(rows, 0).astype(np.int64)


def _lazy_feasible(lc, comp_obj_min, comp_obj_max, random_sample,
                   has_quantile, has_prefer, rng, verbose):
    if not comp_obj_min:
        raise SystemExit("[lazy] no --expr_front but no comp_obj range — the "
                          "full no-NDS product is unbounded. Pass "
                          "--comp_obj/--comp_obj_min/--comp_obj_max.")
    n_dims = len(lc.expr_keys)
    sizes = list(lc.nd_shape)
    allowed = [np.ones(n, bool) for n in sizes]
    blocks = []
    for spec, lo, hi in zip(lc.comp_specs, comp_obj_min, comp_obj_max):
        if spec['kind'] == '1d':
            allowed[spec['axis']] &= _interval(spec['vals'], lo, hi)
        elif spec['kind'] == '2d':
            a0, a1 = spec['axes']
            M = _interval(spec['vals'], lo, hi)
            blocks.append(([a0, a1], np.argwhere(M).astype(np.int64)))
        else:  # memory
            blocks.append(_memory_block(spec, lo, hi))

    surv = [np.where(a)[0] for a in allowed]
    if any(len(s) == 0 for s in surv) or any(len(b[1]) == 0 for b in blocks):
        if verbose:
            print('range_idx : 0')
        return np.empty((0, n_dims), np.int64)

    if not blocks:
        prod = 1
        for s in surv:
            prod *= len(s)
        if prod > _LAZY_MAX_FEASIBLE:
            raise SystemExit(
                f"[lazy] feasible {prod:.3e} > {_LAZY_MAX_FEASIBLE:.0e}; "
                f"tighten --comp_obj_min/--comp_obj_max "
                f"(per-axis survivors {[len(s) for s in surv]}).")
        mesh = np.meshgrid(*surv, indexing='ij')
        nd_idx = np.stack([m.ravel() for m in mesh], 1).astype(np.int64)
    else:
        group, tup = blocks[0]
        for g2, t2 in blocks[1:]:
            group, tup = _join_groups(group, tup, g2, t2)
            if len(tup) == 0:
                if verbose:
                    print('range_idx : 0')
                return np.empty((0, n_dims), np.int64)
        # Intersect block tuples with single-axis survivors — ONLY for axes
        # that actually carry a single-axis constraint. comp_obj=[memory]
        # has no single-axis spec → every `allowed` is all-True → skip
        # entirely (removes a useless O(len(tup)) python membership loop on
        # the multi-million-row memory block). Mixed cases (e.g.
        # comp_obj=[memory,kvbits]) hit only the constrained columns, with a
        # vectorised np.isin instead of a python `in set` loop. Pure
        # single-axis comp_obj (wbits/kvbits/kvdim) never reaches here — it
        # has no blocks and takes the separable Cartesian branch above.
        constrained = [col for col, axx in enumerate(group)
                       if not allowed[axx].all()]
        if constrained:
            keep = np.ones(len(tup), bool)
            for col in constrained:
                keep &= np.isin(tup[:, col], surv[group[col]])
            tup = tup[keep]
        if len(tup) == 0:
            if verbose:
                print('range_idx : 0')
            return np.empty((0, n_dims), np.int64)
        free = [a for a in range(n_dims) if a not in group]
        free_surv = [surv[a] for a in free]
        n_free = 1
        for s in free_surv:
            n_free *= len(s)
        total = len(tup) * n_free
        if total > _LAZY_MAX_FEASIBLE:
            raise SystemExit(
                f"[lazy] feasible {total:.3e} > {_LAZY_MAX_FEASIBLE:.0e}; "
                f"tighten --comp_obj_min/--comp_obj_max "
                f"(block={len(tup)}, free={n_free}).")
        nd_idx = np.empty((total, n_dims), np.int64)
        fmesh = (np.stack([m.ravel() for m in
                           np.meshgrid(*free_surv, indexing='ij')], 1)
                 if free else np.zeros((1, 0), np.int64))
        # Vectorised cartesian product tup × fmesh, outer=tup / inner=fmesh —
        # identical row contents and order to the old per-row python loop, but
        # done in C. Critical for large feasible sets (1e8–1e9): when free=[]
        # (memory comp_obj over every axis → n_free=1) the old loop ran one
        # python iteration per feasible row and dominated wall-clock; here it
        # collapses to two C-level assignments. (np.repeat(t, 1) == t, so the
        # free=[] case is just nd_idx[:, group] = tup.)
        nd_idx[:, group] = np.repeat(tup, n_free, axis=0)
        if free:
            nd_idx[:, free] = np.tile(fmesh, (len(tup), 1))

    if verbose:
        print(f'range_idx : {len(nd_idx)}')
    # deterministic, useful order: by the first comp_obj value ascending
    key = lc.comp_values(lc.comp_specs[0], nd_idx)
    nd_idx = nd_idx[np.argsort(key, kind='stable')]
    only_random = (random_sample is not None and not has_quantile
                   and not has_prefer)
    if only_random and len(nd_idx) > random_sample:
        nd_idx = nd_idx[np.sort(rng.choice(len(nd_idx), size=random_sample,
                                           replace=False))]
    return nd_idx


def _lazy_assemble_F(lc, valid_nd_idx):
    """assemble_F for the lazy path. F[:,0] = additive combined metric
    (Σ scale·JSD == dense new_metric_nd) so post_search ranks correctly even
    WITHOUT a surrogate; when --sample_path is given it is overwritten by the
    surrogate prediction (and then NaN-checked) exactly like the dense path."""
    if len(valid_nd_idx) == 0:
        return np.empty((0, 1 + 2 * len(lc.expr_keys) + len(lc.comp_specs)),
                        np.float64)
    parts = [lc.combined_metric(valid_nd_idx).reshape(-1, 1),
             np.column_stack([lc.efm[k][valid_nd_idx[:, i]]
                              for i, k in enumerate(lc.expr_keys)])]
    if lc.comp_specs:
        parts.append(np.column_stack(
            [lc.comp_values(s, valid_nd_idx) for s in lc.comp_specs]))
    return np.column_stack(parts)


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
    # lazy (no --expr_front): comp_obj-first sparse pruning, no nd_shape array
    if isinstance(comp_nd_list, _LazyComp):
        return _lazy_feasible(comp_nd_list, comp_obj_min, comp_obj_max,
                              random_sample, has_quantile, has_prefer,
                              rng, verbose)
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
    if isinstance(comp_nd_list, _LazyComp):
        return _lazy_assemble_F(comp_nd_list, valid_nd_idx)
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


# ───────────────────── coverage-NSGA2 fill sampler ─────────────────────
# Lifted SubsetProblem (search_think.py:499 style) with multi-D coverage
# fitness — selects K extras that minimize the *covering radius* of the
# (anchor ∪ extras) set over a stratified proxy pool in per-axis z-space.
# Compared with pure random sampling, this distributes extras to cover the
# pool's z-distribution uniformly, including the high-z corner that
# Phase-6 acquisition targets. Empirical comparison: analysis/v5/test_coverage_ga.py.

def _pool_z(valid_nd_idx, efm, expr_keys):
    """Per-axis z (search-time metric) for each entry in valid_nd_idx.

    Returns (n_valid, len(expr_keys)) float64 array.
    """
    cols = [efm[k][valid_nd_idx[:, expr_keys.index(k)], 0] for k in expr_keys]
    return np.column_stack(cols).astype(np.float64, copy=False)


def stratified_proxy_indices(pool_z, n_target=3000, n_bins=None, seed=0):
    """Stratified sub-sample of `pool_z` for fitness-proxy / coverage evaluation.

    Divides the d-D z bounding box into a regular grid (≈ equal number of
    bins per axis) and draws roughly equal candidate count per non-empty bin.
    Reasonable n_target ~ 1000–5000 keeps coverage-fitness fast (per-eval
    KDTree on K+|anchor| samples + query of |proxy| points).

    Returns int64 array of indices into pool_z (≤ n_target unique values).

    Implementation note: uses argsort + diff-based group boundaries to handle
    pool sizes up to O(10^7) candidates in ~1 sec. Naive ``for b in unique:
    np.where(bin_idx == b)`` is O(N × n_bins) and exceeds 100s on 23M pools.
    """
    n_d = pool_z.shape[1]
    if n_bins is None:
        # ~10 bins per axis when d=3 → 1000 cells; scale gently with n_target
        n_per_axis = max(2, int(round(n_target ** (1.0 / n_d) * 1.5)))
        n_bins = [n_per_axis] * n_d
    z_lo = pool_z.min(0); z_hi = pool_z.max(0)
    bin_idx = np.zeros(len(pool_z), dtype=np.int64)
    for d, n_b in enumerate(n_bins):
        edges = np.linspace(z_lo[d], z_hi[d], n_b + 1)
        b = np.minimum(np.searchsorted(edges, pool_z[:, d], side='right') - 1, n_b - 1)
        bin_idx = bin_idx * n_b + b
    rng = np.random.RandomState(int(seed))
    per_bin = max(1, n_target // int(np.prod(n_bins)))

    # Fast group-by: sort bin_idx once, slice contiguous runs.
    order = np.argsort(bin_idx, kind='stable')
    sorted_bins = bin_idx[order]
    # Boundaries between distinct bin values
    diff_pos = np.flatnonzero(np.diff(sorted_bins)) + 1
    boundaries = np.concatenate(([0], diff_pos, [len(sorted_bins)]))
    sel = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        cands_in_pool = order[start:end]
        k = min(per_bin, end - start)
        sel.extend(rng.choice(cands_in_pool, k, replace=False))
    return np.array(sel, dtype=np.int64)


def maximin_extras(M, anchor_idx, K, seed=0):
    """Model-free farthest-point (maximin) coverage extras: return K positions
    into ``M`` (rows = per-axis search-metric vectors) that greedily maximise
    the min-distance to ``anchor_idx`` ∪ already-picked, in per-axis
    standardized space. A drop-in coverage sampler that uses NO surrogate and
    NO measured y — the validated best global-representation acquisition
    (beats random / uncertainty-AL on R²/worst-bin/tail across seeds; works
    with ANY deployed surrogate incl. rbf-tps). Memory-safe for tens-of-millions
    of rows (loops anchors; greedy is O(K·N))."""
    M = np.asarray(M, dtype=float)
    sd = M.std(0); sd[sd < 1e-9] = 1.0
    Ms = (M - M.mean(0)) / sd
    n = len(Ms)
    anchor_idx = [int(a) for a in (anchor_idx if anchor_idx is not None else [])]
    dmin = np.full(n, np.inf)
    for a in anchor_idx:
        if 0 <= a < n:
            dmin = np.minimum(dmin, np.linalg.norm(Ms - Ms[a], axis=1))
    if not anchor_idx:                       # cold start: seed at a fixed point
        s0 = int(np.random.RandomState(seed).randint(n))
        dmin = np.linalg.norm(Ms - Ms[s0], axis=1)
    for a in anchor_idx:
        if 0 <= a < n:
            dmin[a] = -1.0
    sel = []
    n_pick = int(min(K, n - sum(1 for a in anchor_idx if 0 <= a < n)))
    for _ in range(max(0, n_pick)):
        j = int(np.argmax(dmin))
        sel.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(Ms - Ms[j], axis=1))
        dmin[j] = -1.0
    return sel


def _comb(n, k):
    from math import comb
    return comb(n, k)


def even_select(comp, score, K, g, lo, hi):
    """Coordinate-agnostic grid-quota selection: bucket `comp` rows ((N, d), any
    coordinate system — e.g. (wbits, eff_kvbits) box or (memory, split)) into a
    g^d grid over [lo, hi], round-robin the occupied cells, take the best `score`
    (lowest) first WITHIN a cell → per-axis-uniform marginal density + quality.
    Returns selected indices into `comp` (≤ K)."""
    comp = np.asarray(comp, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cell = np.clip(((comp - lo) / (hi - lo + 1e-9) * g).astype(int), 0, g - 1)
    cid = np.zeros(len(comp), int)
    for d in range(comp.shape[1]):
        cid = cid * g + cell[:, d]
    buckets = {}
    for i, c in enumerate(cid):
        buckets.setdefault(int(c), []).append(i)
    for c in buckets:                                # within-cell: best score first
        buckets[c].sort(key=lambda i: score[i])
    cells, sel = list(buckets), []
    while len(sel) < K and cells:                    # round-robin over occupied cells
        for c in list(cells):
            if not buckets[c]:
                cells.remove(c); continue
            sel.append(buckets[c].pop(0))
            if len(sel) >= K:
                break
    return np.array(sel, int)


def moo_subset_select(comp, pred, K, comp_min, comp_max, g, algo='nsga3',
                      pop=80, n_gen=80, seed=0, gap_std=False, coverage='rad'):
    """Pick K candidate indices (into `comp`/`pred`) by a 2/3-objective subset-selection GA
    that balances arch QUALITY against even COVERAGE of the [comp_min, comp_max] box —
    the principled replacement for a hard exploit/explore split (validated in tests/:
    MOO-knee NSGA-III dominates the hybrid hard split on evenness+quality, and
    NSGA-III ≥ NSGA-II). Coordinate-agnostic like even_select. Chromosome = K pool indices:
        obj1 = mean normalised predicted loss   (exploit: pull the low-loss front)
        obj2 = coverage='rad' (default): covering radius over a g^d grid of cell centres
               (2D REACH — minimax "biggest hole"; blind to clumping once cells are reached)
               coverage='gap': max over axes of the std of consecutive sorted-coordinate
               gaps (1D MARGINAL spacing evenness, the coverage_subset_nsga2_extras
               marginal/max fitness; blind to joint 2D structure — a diagonal has perfect
               marginals but covers no off-diagonal area)
        obj3 (gap_std=True) = the OTHER criterion added as a 3rd objective (both at once;
              tested: dilutes the quality objective in the knee — not recommended for the
              closed-loop; per-iter subset evenness does not compound into archive evenness).
    Solved with NSGA-III (reference-direction; default) or NSGA-II, then the KNEE of the
    Pareto front (argmin of the min-max-normalised objective sum) is returned = the balanced
    explore↔exploit subset. Guarantees K picks: any de-dup shortfall is filled by farthest-
    point maximin (keeps coverage). O(pop·n_gen) cheap comp-space evals (no model calls).

    NOTE: operates on the SUPPLIED pool only — pair with supply seeding
    (utils.second_stage.grid_seed) so the high-comp corner NSGA drops is present."""
    from scipy.spatial.distance import cdist
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize

    comp = np.asarray(comp, float); pred = np.asarray(pred, float).ravel()
    N, d = comp.shape
    if N <= K:
        return np.arange(N)
    lo, hi = np.asarray(comp_min, float), np.asarray(comp_max, float)
    cn = (comp - lo) / (hi - lo + 1e-9)                          # comp → unit box
    centers = np.stack(np.meshgrid(*([np.linspace(0, 1, g)] * d), indexing='ij'),
                       -1).reshape(-1, d)                        # g^d box cell centres
    # obj1 scale-free AND outlier-robust: RANK-normalised predictions, not min-max — exact-
    # interpolant surrogates (rbf/tps) extrapolate wildly at mutated/seed candidates (observed
    # candidate RMSE ~120 on a 0.02-0.7 JSD scale at production iter 1); one blown-up value
    # under min-max compresses every other prediction to ~0 and degenerates the quality
    # objective. Ranks keep the surrogate's ORDERING (all the subset pressure needs) with
    # zero outlier leverage.
    _r = np.empty(N); _r[np.argsort(pred, kind='stable')] = np.arange(N)
    Ln = _r / max(N - 1, 1)
    rng = np.random.RandomState(int(seed))
    n_obj = 3 if gap_std else 2

    class SubProb(Problem):
        def __init__(self): super().__init__(n_var=K, n_obj=n_obj, n_constr=0, xl=0, xu=N - 1, vtype=int)
        def _evaluate(self, X, out, *a, **k):
            Xi = X.astype(int)
            f_loss = Ln[Xi].mean(1)
            dup = np.array([K - len(set(r.tolist())) for r in Xi], float) / K   # penalise dup picks
            need_rad = (coverage == 'rad') or gap_std
            need_gap = (coverage == 'gap') or gap_std
            f_cov = (np.array([cdist(centers, cn[row]).min(1).max() for row in Xi])
                     if need_rad else None)
            f_gap = None
            if need_gap:                                         # per-axis spacing evenness
                picks = cn[Xi]                                   # (npop, K, d)
                f_gap = np.zeros(len(Xi))
                for ax in range(d):
                    Sd = np.sort(picks[:, :, ax], axis=1)
                    f_gap = np.maximum(f_gap, np.std(np.diff(Sd, axis=1), axis=1))
            main = f_cov if coverage == 'rad' else f_gap
            cols = [f_loss + dup, main + dup]
            if gap_std:                                          # 3-obj: add the OTHER criterion
                other = f_gap if coverage == 'rad' else f_cov
                cols.append(other + dup)
            out['F'] = np.column_stack(cols)

    class Smp(Sampling):
        def _do(self, p, n, **k): return np.array([rng.choice(N, K, replace=False) for _ in range(n)])

    class Cx(Crossover):
        def __init__(self): super().__init__(2, 2)
        def _do(self, p, X, **k):
            Y = np.empty_like(X)
            for j in range(X.shape[1]):
                m = rng.rand(K) < 0.5
                Y[0, j] = np.where(m, X[0, j], X[1, j]); Y[1, j] = np.where(m, X[1, j], X[0, j])
            return Y

    class Mt(Mutation):
        def _do(self, p, X, **k):
            X = X.copy()
            for i in range(len(X)):
                for j in range(K):
                    if rng.rand() < 1.0 / K: X[i, j] = rng.randint(N)
            return X

    if algo == 'nsga3':
        if n_obj == 2:
            parts = max(pop - 1, 2)
        else:                                      # largest p with C(p+n_obj-1, n_obj-1) <= pop
            parts = 1
            while _comb(parts + n_obj, n_obj - 1) <= pop:
                parts += 1
        ref = get_reference_directions("das-dennis", n_obj, n_partitions=parts)
        a = NSGA3(pop_size=pop, ref_dirs=ref, sampling=Smp(), crossover=Cx(),
                  mutation=Mt(), eliminate_duplicates=True)
    else:
        a = NSGA2(pop_size=pop, sampling=Smp(), crossover=Cx(), mutation=Mt(),
                  eliminate_duplicates=True)
    res = minimize(SubProb(), a, ('n_gen', n_gen), seed=int(seed), verbose=False)
    F, Xr = np.atleast_2d(res.F), np.atleast_2d(res.X).astype(int)
    Fn = (F - F.min(0)) / (F.max(0) - F.min(0)).clip(1e-9)       # knee = min-max-norm sum argmin
    knee = Xr[int(Fn.sum(1).argmin())]
    seen, out = set(), []
    for pk in knee:                                             # de-dup within the knee subset
        if int(pk) not in seen: seen.add(int(pk)); out.append(int(pk))
    if len(out) < K:                                            # fill shortfall by farthest-point coverage
        for e in maximin_extras(comp, anchor_idx=out, K=K - len(out), seed=seed):
            if int(e) not in seen: seen.add(int(e)); out.append(int(e))
    return np.array(out[:K], int)


def coverage_subset_nsga2_extras(valid_nd_idx, efm, expr_keys,
                                  anchor_idx, K,
                                  fitness='joint',
                                  coord='rank',
                                  per_axis_agg='max',
                                  pareto_select='auto',
                                  proxy_size=3000,
                                  pop=80, n_gen=80, seed=0, verbose=False):
    """Generate K extras (indices into valid_nd_idx) that maximise pool
    coverage when union'd with anchor_idx. Drop-in replacement
    for `draw_random(K, len(valid_nd_idx), exclude=anchor_idx)`.

    coord:
      'z'    — fitness in normalised z-value space (per-axis (z - z_min) / z_range).
               Sensitive to per-axis loss-distribution shape (long-tail dominated).
      'rank' — fitness in rank-space (per-axis rank / (n_axis - 1) ∈ [0, 1]).
               Distribution-invariant: each axis is uniform by construction, so
               cov_rad measures rank-coverage rather than z-value-coverage.

    Implementation — **discrete subset-style GA** (mirrors
    `search_think.py:SubsetProblem`):
      - Chromosome = K integer positions into `valid_nd_idx`.
      - Per-axis sorted PFs (efm[axis], column 0 = ascending metric) define
        the implicit pool; the chromosome's K positions reference rows in
        valid_nd_idx whose per-axis ranks already index those sorted PFs.
      - Custom Sampling / Crossover / Mutation preserve both the K-cardinality
        invariant and per-pick validity by construction — no snap step.
      - Strategy-3: union of all Pareto solutions' picks → greedy coverage
        K (in normalised z-space), excluding anchor positions.

    Decision dim = K (e.g. 41), versus K · n_axes for the deprecated
    continuous variant and len(valid_nd_idx) for a naive bitmask. Fitness is
    measured directly on pool z values, so GA-reported and actual fitness
    are consistent (no snap-induced error).

    fitness mode:
      'joint'    — 2-obj NSGA2 minimising (covering radius, mean coverage)
                   over a stratified proxy pool in normalised z-space.
                   Vectorised across the population (cdist on (K, proxy)
                   per chromosome, no per-chrom KDTree build).
      'marginal' — per-axis std-of-consecutive-gaps; aggregator chosen by
                   `per_axis_agg`. Vectorised across population (no Python
                   loop over chromosomes). No KDTree / proxy needed.
      'combined' — 2-obj NSGA2 minimising (covering radius, max-axis
                   std-of-gaps) jointly. cov_rad rewards reaching every
                   region (extent); std_max rewards uniform per-axis spacing.
                   They are orthogonal and conflict under a skewed (long-tail)
                   pool z-distribution, so the Pareto front trades reach vs
                   uniformity. Final K via Strategy-3 greedy union.

    per_axis_agg (only used when fitness='marginal'; default 'max'):
      'max'    — single-obj GA on max_k std_k (Tchebycheff). DEFAULT.
                 "No axis worse than any other"; strongest per-axis-evenness
                 pressure — every axis has the same std-of-gaps.
      'sum'    — single-obj GA on Σ_k std_k. Slightly faster than 'max' but
                 lets a strong axis compensate for a weak axis.
      'pareto' — n_axes-obj NSGA2 over (std_axis_0, …, std_axis_{n-1}).
                 Original multi-objective behaviour; final K via Strategy-3
                 greedy union.

    pareto_select (how to collapse a multi-objective Pareto front → final K):
      'auto'      — DEFAULT. 'knee' when fitness='combined', else 'strategy3'.
      'strategy3' — union of all Pareto solutions' picks → greedy
                    worst-proxy-nearest K. cov_rad-driven (biases the final
                    pick toward the covering-radius end of the front).
      'knee'      — pick the single Pareto solution at the knee of the front
                    (argmin of min-max-normalised objective sum). For
                    'combined' this yields a design that is near-minimal on
                    BOTH cov_rad and std_max simultaneously. No-op for
                    single-objective modes (marginal+sum/max).

    Args:
        valid_nd_idx : (n_valid, n_dims) int array — each row is a tuple
                       (rank_0, rank_1, ...) of per-axis sorted-PF positions.
        efm          : dict axis → (n_archive, 2) array (col 0 = z metric,
                       sorted ascending by `load_expr` upstream).
        expr_keys    : ordered axis names matching valid_nd_idx columns.
        anchor_idx   : indices into valid_nd_idx already picked by
                       quantile_select (may be empty/None for no-anchor mode).
        K            : number of extras to return.
        fitness      : 'joint' or 'marginal'.
        proxy_size   : # of proxy pool points for joint coverage fitness +
                       greedy step (default 3000).
        pop, n_gen   : NSGA2 hyperparameters (default 80, 80).
        seed         : random seed.

    Returns:
        sorted list of K int indices into valid_nd_idx (disjoint from
        anchor_idx).
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.optimize import minimize

    if K <= 0 or len(valid_nd_idx) == 0:
        return []

    if coord not in ('z', 'rank'):
        raise ValueError(f"coord must be 'z' or 'rank', got {coord!r}")
    if fitness not in ('joint', 'marginal', 'combined'):
        raise ValueError(f"fitness must be 'joint'|'marginal'|'combined', got {fitness!r}")
    if fitness == 'marginal' and per_axis_agg not in ('pareto', 'sum', 'max'):
        raise ValueError(f"per_axis_agg must be 'pareto'|'sum'|'max', got {per_axis_agg!r}")
    if pareto_select not in ('auto', 'strategy3', 'knee'):
        raise ValueError(f"pareto_select must be 'auto'|'strategy3'|'knee', "
                         f"got {pareto_select!r}")
    if pareto_select == 'auto':
        pareto_select = 'knee' if fitness == 'combined' else 'strategy3'

    n_axes  = valid_nd_idx.shape[1]
    n_valid = len(valid_nd_idx)
    # Build pool_norm in chosen coordinate space.
    # 'z':   per-axis (z - z_min) / (z_max - z_min); long-tail of loss matters.
    # 'rank': per-axis rank / (n_axis - 1); distribution-invariant per axis.
    if coord == 'z':
        pool_z = _pool_z(valid_nd_idx, efm, expr_keys)
        z_lo = pool_z.min(0); z_hi = pool_z.max(0)
        z_range = (z_hi - z_lo).clip(1e-9)
        pool_norm = (pool_z - z_lo) / z_range
        # Stratified proxy in z-space (joint fitness uses pool_norm distances)
        proxy_local = stratified_proxy_indices(pool_z, n_target=proxy_size, seed=seed + 1)
        proxy_norm  = pool_norm[proxy_local]
    else:  # 'rank'
        n_per_axis = np.array([efm[expr_keys[k]].shape[0] for k in range(n_axes)],
                              dtype=np.float64)
        pool_norm = valid_nd_idx.astype(np.float64) / np.maximum(n_per_axis - 1, 1)
        # Stratified proxy in rank-space (rank-space is uniform by construction,
        # but we still subsample for fitness-eval cost).
        proxy_local = stratified_proxy_indices(pool_norm, n_target=proxy_size, seed=seed + 1)
        proxy_norm  = pool_norm[proxy_local]

    anchor_arr = np.asarray(anchor_idx, dtype=np.int64)
    anchor_norm = pool_norm[anchor_arr] if len(anchor_arr) > 0 \
                  else np.zeros((0, n_axes), dtype=np.float64)
    anchor_set = set(int(i) for i in anchor_arr)

    PEN_DUP = 5.0   # discourage within-chromosome / vs-anchor duplicates

    # ── Pre-compute (constant across generations) ──
    # Anchor → proxy distance (cov_rad baseline) — needed by joint & combined
    if fitness in ('joint', 'combined'):
        # cdist returns (n_anchor, n_proxy) — trivial to broadcast-min later.
        d_anchor_proxy = (cdist(anchor_norm, proxy_norm)
                          if len(anchor_norm) > 0
                          else np.full((1, len(proxy_norm)), np.inf))
        anchor_proxy_min = d_anchor_proxy.min(axis=0)   # (n_proxy,)

    # n_obj: joint→2(cov_rad,mean) | combined→2(cov_rad,std_max)
    #        marginal+pareto→n_axes | marginal+sum/max→1
    if fitness == 'joint':
        n_obj_eff = 2
    elif fitness == 'combined':
        n_obj_eff = 2
    elif per_axis_agg == 'pareto':
        n_obj_eff = n_axes
    else:
        n_obj_eff = 1

    # ─── Problem: K ints in [0, n_valid-1] — fully vectorised _evaluate ───
    class SubsetCoverageProblem(Problem):
        def __init__(self):
            super().__init__(n_var=K, n_obj=n_obj_eff, n_constr=0,
                             xl=0, xu=n_valid - 1, vtype=int)

        def _evaluate(self, X, out, *args, **kwargs):
            X_int = X.astype(int)             # (n_pop, K)
            n_pop = X_int.shape[0]

            # ── Penalties (vectorised) ──
            # within-chrom duplicates: count - len(unique) per row
            n_within = np.array([K - len(set(row.tolist())) for row in X_int])
            # anchor collisions: count picks that hit anchor_set per row
            if len(anchor_set) > 0:
                anchor_hit = np.isin(X_int, list(anchor_set)).sum(axis=1)
            else:
                anchor_hit = np.zeros(n_pop, dtype=int)
            penalty = PEN_DUP * (n_within + anchor_hit) / K     # (n_pop,)

            if fitness in ('joint', 'combined'):
                # picks_norm: (n_pop, K, n_axes) — gather pool_norm[X_int]
                picks_norm = pool_norm[X_int]
                # For each chrom: cdist(picks, proxy) → min over picks-axis
                d_min = np.zeros((n_pop, len(proxy_norm)), dtype=np.float64)
                for i in range(n_pop):
                    d_pp = cdist(picks_norm[i], proxy_norm)     # (K, n_proxy)
                    d_min[i] = np.minimum(anchor_proxy_min, d_pp.min(axis=0))
                cov_rad = d_min.max(axis=1)                      # (n_pop,)
                if fitness == 'joint':
                    F = np.column_stack([cov_rad + penalty,
                                         d_min.mean(axis=1) + penalty])
                else:  # 'combined' — obj2 = max-axis std-of-gaps
                    n_anchor = len(anchor_norm)
                    std_max = np.zeros(n_pop, dtype=np.float64)
                    F_ax = np.zeros((n_pop, n_axes), dtype=np.float64)
                    for k in range(n_axes):
                        pk = pool_norm[X_int, k]
                        if n_anchor > 0:
                            ak = np.broadcast_to(anchor_norm[:, k],
                                                 (n_pop, n_anchor))
                            Sk = np.concatenate([ak, pk], axis=1)
                        else:
                            Sk = pk
                        Sk.sort(axis=1)
                        F_ax[:, k] = np.std(np.diff(Sk, axis=1), axis=1)
                    std_max = F_ax.max(axis=1)
                    F = np.column_stack([cov_rad + penalty,
                                         std_max + penalty])
            else:
                # marginal: per-axis std-of-gaps, vectorised across (n_pop, K)
                # for each axis k: gather pool_norm[picks, k] → (n_pop, K),
                # concat with anchor_sorted, sort, diff, std.
                F_axes = np.zeros((n_pop, n_axes), dtype=np.float64)
                n_anchor = len(anchor_norm)
                for k in range(n_axes):
                    picks_k = pool_norm[X_int, k]               # (n_pop, K)
                    if n_anchor > 0:
                        anc_k = np.broadcast_to(anchor_norm[:, k],
                                                (n_pop, n_anchor))
                        S_k = np.concatenate([anc_k, picks_k], axis=1)
                    else:
                        S_k = picks_k
                    S_k.sort(axis=1)                            # in-place sort per row
                    F_axes[:, k] = np.std(np.diff(S_k, axis=1), axis=1)
                if per_axis_agg == 'pareto':
                    F = F_axes + penalty[:, None]
                elif per_axis_agg == 'sum':
                    F = (F_axes.sum(axis=1) + penalty)[:, None]
                else:  # 'max' / Tchebycheff
                    F = (F_axes.max(axis=1) + penalty)[:, None]

            out["F"] = F

    # ─── Custom ops (validity- and K-preserving by construction) ───
    rng_global = np.random.RandomState(int(seed))

    class SubsetSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            X = np.zeros((n_samples, K), dtype=int)
            for i in range(n_samples):
                X[i] = rng_global.choice(n_valid, size=K, replace=False)
            return X

    class SubsetUniformCrossover(Crossover):
        """Row-wise uniform crossover: each gene from one parent.
        Preserves K cardinality; within-child duplicates handled by penalty."""
        def __init__(self):
            super().__init__(2, 2)
        def _do(self, problem, X, **kwargs):
            n_off, n_mat, _ = self.n_offsprings, X.shape[1], K
            Xp = np.zeros((n_off, n_mat, K), dtype=int)
            for j in range(n_mat):
                p1, p2 = X[0, j], X[1, j]
                mask = rng_global.rand(K) < 0.5
                Xp[0, j] = np.where(mask, p1, p2)
                Xp[1, j] = np.where(mask, p2, p1)
            return Xp

    class SubsetMutation(Mutation):
        """Replace a chromosome gene with a random unused valid index
        (per-gene prob ~1/K). Always lands in valid_nd_idx — no repair."""
        def _do(self, problem, X, **kwargs):
            p = 1.0 / K
            for i in range(X.shape[0]):
                picks = X[i].astype(int)
                cur = set(int(x) for x in picks)
                for j in range(K):
                    if rng_global.rand() < p:
                        for _ in range(20):
                            cand = int(rng_global.randint(n_valid))
                            if cand not in cur:
                                cur.discard(int(picks[j]))
                                picks[j] = cand
                                cur.add(cand)
                                break
                X[i] = picks
            return X

    # Single-obj aggregators (sum/max) use simple GA (no crowding distance);
    # joint and Pareto-marginal use NSGA2.
    if n_obj_eff == 1:
        algo = GA(pop_size=pop,
                  sampling=SubsetSampling(),
                  crossover=SubsetUniformCrossover(),
                  mutation=SubsetMutation(),
                  eliminate_duplicates=True)
    else:
        algo = NSGA2(pop_size=pop,
                     sampling=SubsetSampling(),
                     crossover=SubsetUniformCrossover(),
                     mutation=SubsetMutation(),
                     eliminate_duplicates=True)
    res = minimize(SubsetCoverageProblem(), algo, ('n_gen', n_gen),
                   seed=int(seed), verbose=verbose)

    # ── Knee selection: pick ONE Pareto solution at the front knee ──
    # (argmin of min-max-normalised objective sum). Only meaningful for a
    # multi-objective front with >1 solution; single-obj GA has res.X 1-D.
    if (pareto_select == 'knee' and res.X.ndim > 1 and res.X.shape[0] > 1
            and res.F.ndim == 2 and res.F.shape[1] > 1):
        Frng = (res.F.max(0) - res.F.min(0)).clip(1e-9)
        Fn   = (res.F - res.F.min(0)) / Frng
        i_knee = int(Fn.sum(axis=1).argmin())
        knee_picks = res.X[i_knee].astype(int)
        # Clean: drop within-chrom dups + anchor collisions (GA penalises
        # these so the knee solution is normally already clean).
        seen = set(anchor_set); cleaned = []
        for p in knee_picks:
            p = int(p)
            if p not in seen:
                seen.add(p); cleaned.append(p)
        if len(cleaned) >= K:
            return sorted(cleaned[:K])
        # Rare shortfall → fall through to Strategy-3 (fills remaining).

    # Strategy-3: union of all (Pareto / final-pop) solutions' picks → greedy K.
    # For single-obj GA, res.X is 1-D (best chrom). For NSGA2, res.X is
    # (n_pareto, K). reshape(-1) flattens both into a 1-D index list.
    res_X = res.X if res.X.ndim > 1 else res.X.reshape(1, -1)
    all_picks   = res_X.astype(int).reshape(-1)
    unique_pos  = np.unique(all_picks)
    union_norm  = pool_norm[unique_pos]

    if len(anchor_norm) > 0:
        d_proxy = cKDTree(anchor_norm).query(proxy_norm, k=1)[0]
    else:
        d_proxy = np.full(len(proxy_norm), np.inf)

    avail = np.array([int(p) not in anchor_set for p in unique_pos], dtype=bool)
    selected_local = []
    for _ in range(min(K, int(avail.sum()))):
        if not avail.any(): break
        worst = proxy_norm[int(np.argmax(d_proxy))]
        d_to_worst = np.linalg.norm(union_norm - worst, axis=1)
        d_to_worst[~avail] = np.inf
        idx_local = int(np.argmin(d_to_worst))
        selected_local.append(idx_local); avail[idx_local] = False
        new_d = np.linalg.norm(proxy_norm - union_norm[idx_local], axis=1)
        d_proxy = np.minimum(d_proxy, new_d)
    return sorted(int(unique_pos[i]) for i in selected_local)




