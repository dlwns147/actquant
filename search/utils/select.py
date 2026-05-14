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


def coverage_subset_nsga2_extras(valid_nd_idx, efm, expr_keys,
                                  anchor_idx, K,
                                  fitness='joint',
                                  coord='rank',
                                  per_axis_agg='max',
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

    per_axis_agg (only used when fitness='marginal'; default 'max'):
      'max'    — single-obj GA on max_k std_k (Tchebycheff). DEFAULT.
                 "No axis worse than any other"; strongest per-axis-evenness
                 pressure — every axis has the same std-of-gaps.
      'sum'    — single-obj GA on Σ_k std_k. Slightly faster than 'max' but
                 lets a strong axis compensate for a weak axis.
      'pareto' — n_axes-obj NSGA2 over (std_axis_0, …, std_axis_{n-1}).
                 Original multi-objective behaviour; final K via Strategy-3
                 greedy union.

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
    if fitness not in ('joint', 'marginal'):
        raise ValueError(f"fitness must be 'joint' or 'marginal', got {fitness!r}")
    if fitness == 'marginal' and per_axis_agg not in ('pareto', 'sum', 'max'):
        raise ValueError(f"per_axis_agg must be 'pareto'|'sum'|'max', got {per_axis_agg!r}")

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
    # Anchor → proxy distance (used as baseline for joint mode)
    if fitness == 'joint':
        # cdist returns (n_anchor, n_proxy) — trivial to broadcast-min later.
        d_anchor_proxy = (cdist(anchor_norm, proxy_norm)
                          if len(anchor_norm) > 0
                          else np.full((1, len(proxy_norm)), np.inf))
        anchor_proxy_min = d_anchor_proxy.min(axis=0)   # (n_proxy,)
    # Anchor sorted per axis (constant) — used by marginal vectorisation
    if fitness == 'marginal':
        anchor_sorted_per_axis = np.sort(anchor_norm, axis=0) \
                                 if len(anchor_norm) > 0 \
                                 else np.zeros((0, n_axes), dtype=np.float64)

    # n_obj: 2 (joint) | n_axes (marginal+pareto) | 1 (marginal+sum/max)
    if fitness == 'joint':
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

            if fitness == 'joint':
                # picks_norm: (n_pop, K, n_axes) — gather pool_norm[X_int]
                picks_norm = pool_norm[X_int]
                # For each chrom: cdist(picks, proxy) → min over picks-axis
                # Loop is over n_pop; each cdist call is (K, n_proxy).
                # Faster than per-chrom KDTree build.
                d_min = np.zeros((n_pop, len(proxy_norm)), dtype=np.float64)
                for i in range(n_pop):
                    d_pp = cdist(picks_norm[i], proxy_norm)     # (K, n_proxy)
                    d_min[i] = np.minimum(anchor_proxy_min, d_pp.min(axis=0))
                F = np.column_stack([d_min.max(axis=1) + penalty,
                                     d_min.mean(axis=1) + penalty])
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




