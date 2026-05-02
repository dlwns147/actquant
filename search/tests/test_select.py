"""Unit tests for utils/select.py.

Each helper is exercised in isolation. Synthetic per-axis subnets are constructed
just large enough to cover all branches; the real Llama-3.1-8B-Instruct config
is used because get_net_info needs accurate layer shapes to compute bit metrics.
"""

import json
import numpy as np
import pytest

from utils.func import get_net_info
from utils.select import (
    axis_of_map, build_arch, LazyPs, per_axis_metric, metric_over,
    draw_random, select_valid_nd_idx, assemble_F, quantile_select,
)


# ───────────────────────── fixtures ─────────────────────────

@pytest.fixture
def config():
    with open('/NAS/SJ/actquant/search/config/llama.json') as f:
        return json.load(f)['Llama-3.1-8B-Instruct']


@pytest.fixture
def group_size():
    return {'w': 128, 'k': 128, 'v': 128}


@pytest.fixture
def default_arch(config):
    n_block = config['n_block']
    linears = config['linear']
    return {
        'q': {'w': {ln: [4] * n_block for ln in linears},
              'k': [[4, 128]] * n_block,
              'v': [[4, 128]] * n_block},
        'p': {'k': [0] * n_block, 'v': [0] * n_block},
    }


@pytest.fixture
def synthetic_archives(config):
    """3 weight subnets (2/3/4-bit), 3 KV subnets (2/3/4-bit), 3 kvdim (drop 0/16/32)."""
    n_block = config['n_block']
    linears = config['linear']

    def w(b): return {
        'q': {'w': {ln: [b] * n_block for ln in linears},
              'k': [[4, 128]] * n_block,
              'v': [[4, 128]] * n_block},
        'p': {'k': [0] * n_block, 'v': [0] * n_block},
    }

    def kv(kb, vb): return {
        'q': {'w': {ln: [4] * n_block for ln in linears},
              'k': [[kb, 128]] * n_block,
              'v': [[vb, 128]] * n_block},
        'p': {'k': [0] * n_block, 'v': [0] * n_block},
    }

    def kvd(d): return {
        'q': {'w': {ln: [4] * n_block for ln in linears},
              'k': [[4, 128]] * n_block,
              'v': [[4, 128]] * n_block},
        'p': {'k': [d] * n_block, 'v': [d] * n_block},
    }

    return {
        'w':     np.array([w(2), w(3), w(4)],          dtype=object),
        'kv':    np.array([kv(2, 2), kv(3, 3), kv(4, 4)], dtype=object),
        'kvdim': np.array([kvd(0), kvd(16), kvd(32)],     dtype=object),
    }


# ───────────────────────── axis_of_map ─────────────────────────

def test_axis_of_map_default():
    m = axis_of_map(['w', 'kv', 'kvdim'])
    assert m['wbits'] == 'w'
    assert m['kvbits'] == 'kv'
    assert m['kvdim'] == 'kvdim'
    assert m['eff_kvbits'] is None


def test_axis_of_map_eff_kv_overrides():
    m = axis_of_map(['w', 'eff_kv'])
    assert m['kvbits'] == 'eff_kv'
    assert m['kvdim'] == 'eff_kv'
    assert m['eff_kvbits'] == 'eff_kv'


def test_axis_of_map_unknown_key_missing():
    m = axis_of_map(['w'])
    assert 'memory' not in m  # only single-axis bit metrics are mapped


# ───────────────────────── build_arch ─────────────────────────

def test_build_arch_picks_per_axis(default_arch, synthetic_archives):
    expr_keys = ['w', 'kv', 'kvdim']
    a = build_arch(default_arch, expr_keys, synthetic_archives, np.array([1, 2, 0]))
    esm = synthetic_archives
    assert a['q']['w'] == esm['w'][1]['q']['w']
    assert a['q']['k'] == esm['kv'][2]['q']['k']
    assert a['q']['v'] == esm['kv'][2]['q']['v']
    assert a['p']['k'] == esm['kvdim'][0]['p']['k']


def test_build_arch_eff_kv_combines_q_and_p(default_arch, config):
    n_block = config['n_block']
    linears = config['linear']
    eff_subnet = {
        'q': {'w': {ln: [4] * n_block for ln in linears},
              'k': [[2, 128]] * n_block,
              'v': [[3, 128]] * n_block},
        'p': {'k': [16] * n_block, 'v': [16] * n_block},
    }
    a = build_arch(default_arch, ['eff_kv'],
                   {'eff_kv': np.array([eff_subnet], dtype=object)},
                   np.array([0]))
    assert a['q']['k'] == eff_subnet['q']['k']
    assert a['p']['k'] == eff_subnet['p']['k']


def test_build_arch_does_not_mutate_default(default_arch, synthetic_archives):
    snapshot = json.dumps(default_arch, sort_keys=True)
    build_arch(default_arch, ['w', 'kv', 'kvdim'], synthetic_archives, np.array([0, 0, 0]))
    assert json.dumps(default_arch, sort_keys=True) == snapshot


# ───────────────────────── LazyPs ─────────────────────────

def test_lazy_ps_int_slice_list_indexing():
    nd_idx = np.array([[0, 0], [1, 1], [2, 0], [0, 2]])
    ps = LazyPs(lambda r: tuple(int(x) for x in r), nd_idx)
    assert len(ps) == 4
    assert ps[2] == (2, 0)
    assert ps[1:3] == [(1, 1), (2, 0)]
    assert ps[[0, 3]] == [(0, 0), (0, 2)]


def test_lazy_ps_does_not_eagerly_build():
    """Building the wrapper must not invoke the builder once."""
    calls = {'n': 0}

    def b(row):
        calls['n'] += 1
        return None

    nd_idx = np.zeros((10_000, 2), dtype=np.int64)
    ps = LazyPs(b, nd_idx)
    assert calls['n'] == 0
    _ = ps[5]
    _ = ps[5]   # repeated access still calls — no caching by design
    assert calls['n'] == 2


# ─────────────────────── per_axis_metric / metric_over ───────────────────────

def test_per_axis_metric_wbits_strictly_increasing(synthetic_archives, config, group_size):
    pa = per_axis_metric('wbits', ['w', 'kv', 'kvdim'],
                         synthetic_archives, config, group_size, n_token=0)
    assert pa is not None
    ax, arr = pa
    assert ax == 0
    assert arr.shape == (3,)
    # 2 / 3 / 4 bit synthetic subnets → strictly increasing wbits
    assert arr[0] < arr[1] < arr[2]


def test_per_axis_metric_multi_axis_returns_none(synthetic_archives, config, group_size):
    # 'memory' depends on w + kv + kvdim → not a single-axis lookup
    assert per_axis_metric('memory', ['w', 'kv', 'kvdim'],
                           synthetic_archives, config, group_size, n_token=0) is None


def test_per_axis_metric_uses_cache(synthetic_archives, config, group_size):
    cache = {}
    pa1 = per_axis_metric('wbits', ['w', 'kv', 'kvdim'],
                          synthetic_archives, config, group_size, 0, cache)
    pa2 = per_axis_metric('wbits', ['w', 'kv', 'kvdim'],
                          synthetic_archives, config, group_size, 0, cache)
    assert pa1 is pa2  # same tuple object → cache hit


def test_metric_over_axis_matches_per_arch_fallback(default_arch, synthetic_archives,
                                                    config, group_size):
    """Vectorised lookup must produce identical values to the per-arch path."""
    expr_keys = ['w', 'kv', 'kvdim']
    nd_idx = np.array(np.meshgrid([0, 1, 2], [0, 1, 2], [0, 1, 2],
                                   indexing='ij')).reshape(3, -1).T

    via_axis = metric_over(nd_idx, 'wbits', expr_keys, synthetic_archives,
                           default_arch, config, group_size, n_token=0)
    fallback = np.array([
        get_net_info(build_arch(default_arch, expr_keys, synthetic_archives, nd_idx[i]),
                     config, group_size, n_token=0)['wbits']
        for i in range(len(nd_idx))
    ])
    np.testing.assert_allclose(via_axis, fallback)


def test_metric_over_fallback_for_memory(default_arch, synthetic_archives,
                                         config, group_size):
    """Memory is multi-axis; metric_over must build per-arch dicts."""
    expr_keys = ['w', 'kv', 'kvdim']
    nd_idx = np.array([[0, 0, 0], [2, 2, 2], [1, 0, 2]])
    out = metric_over(nd_idx, 'memory', expr_keys, synthetic_archives,
                      default_arch, config, group_size, n_token=1024)
    assert out.shape == (3,)
    assert (out > 0).all()


# ───────────────────────── draw_random ─────────────────────────

def test_draw_random_basic_unique_sorted():
    rng = np.random.default_rng(0)
    out = draw_random(20, 100, rng=rng)
    assert len(out) == 20
    assert out == sorted(out)
    assert len(set(out)) == 20
    assert all(0 <= x < 100 for x in out)


def test_draw_random_clipped_to_pool():
    out = draw_random(10_000, 50)
    assert len(out) == 50


def test_draw_random_with_exclude_disjoint_and_in_range():
    rng = np.random.default_rng(0)
    excl = {3, 7, 9, 12}
    out = draw_random(10, 20, exclude=excl, rng=rng)
    assert len(out) == 10
    assert set(out).isdisjoint(excl)
    assert all(0 <= x < 20 for x in out)


def test_draw_random_zero_inputs():
    assert draw_random(0, 100) == []
    assert draw_random(5, 0) == []


def test_draw_random_exclude_larger_than_pool():
    """Pool entirely covered by exclude → empty result."""
    out = draw_random(10, 5, exclude={0, 1, 2, 3, 4})
    assert out == []


def test_draw_random_seed_reproducibility():
    a = draw_random(20, 100, rng=np.random.default_rng(42))
    b = draw_random(20, 100, rng=np.random.default_rng(42))
    assert a == b


# ───────────────────────── select_valid_nd_idx ─────────────────────────

def test_select_full_sort_path_returns_all_sorted():
    nd_shape = (3, 4)
    nm = np.arange(12).reshape(nd_shape).astype(float)
    out = select_valid_nd_idx(nd_shape, nm, [], [], [],
                              random_sample=None, has_quantile=False,
                              has_prefer=False, verbose=False)
    assert out.shape == (12, 2)
    # Sorted ascending by metric ⇒ first row is the smallest cell
    assert tuple(out[0]) == (0, 0)
    assert tuple(out[-1]) == (2, 3)


def test_select_filter_path_keeps_only_in_range():
    nd_shape = (3, 3)
    nm = np.random.RandomState(0).rand(*nd_shape)
    comp_nd = np.arange(9).reshape(nd_shape).astype(float)
    out = select_valid_nd_idx(nd_shape, nm, [comp_nd], [2.5], [5.5],
                              random_sample=None, has_quantile=True,
                              has_prefer=False, verbose=False)
    assert len(out) == 3  # values 3, 4, 5 satisfy the bounds
    for row in out:
        assert 2.5 <= comp_nd[tuple(row)] <= 5.5


def test_select_only_random_no_filter_fast_path():
    nd_shape = (10, 10)
    nm = np.random.RandomState(0).rand(*nd_shape)
    rng = np.random.default_rng(0)
    out = select_valid_nd_idx(nd_shape, nm, [], [], [],
                              random_sample=20, has_quantile=False,
                              has_prefer=False, verbose=False, rng=rng)
    assert out.shape == (20, 2)
    flat = np.ravel_multi_index((out[:, 0], out[:, 1]), nd_shape)
    assert len(np.unique(flat)) == 20


def test_select_filter_plus_only_random_subsamples_within_filter():
    nd_shape = (10, 10)
    nm = np.random.RandomState(0).rand(*nd_shape)
    comp_nd = np.zeros(nd_shape)
    comp_nd[:5, :] = 1.0  # 50 entries pass
    rng = np.random.default_rng(0)
    out = select_valid_nd_idx(nd_shape, nm, [comp_nd], [0.5], [1.5],
                              random_sample=10, has_quantile=False,
                              has_prefer=False, verbose=False, rng=rng)
    assert len(out) == 10
    for row in out:
        assert comp_nd[tuple(row)] == 1.0


def test_select_quantile_active_does_not_subsample_filter():
    """When quantile is on, the full filtered set must reach the selector."""
    nd_shape = (10, 10)
    nm = np.random.RandomState(0).rand(*nd_shape)
    comp_nd = np.zeros(nd_shape); comp_nd[:5, :] = 1.0
    out = select_valid_nd_idx(nd_shape, nm, [comp_nd], [0.5], [1.5],
                              random_sample=10, has_quantile=True,  # ← active
                              has_prefer=False, verbose=False)
    assert len(out) == 50  # full filtered set, not 10


def test_select_returns_metric_sorted_ascending():
    nd_shape = (4, 4)
    nm = np.random.RandomState(1).rand(*nd_shape)
    out = select_valid_nd_idx(nd_shape, nm, [], [], [],
                              random_sample=None, has_quantile=False,
                              has_prefer=False, verbose=False)
    # Metric values along returned order must be non-decreasing
    metrics = nm[tuple(out.T)]
    assert (np.diff(metrics) >= 0).all()


# ───────────────────────── assemble_F ─────────────────────────

def test_assemble_F_no_comp_obj_columns():
    """Layout: [combined_metric | per-axis (metric, comp_obj) per expr_key]."""
    nd_shape = (4, 3)
    nm = np.random.RandomState(0).rand(*nd_shape)
    efm = {'w':  np.column_stack([np.arange(4) * 0.1, np.arange(4) * 1.0]),
           'kv': np.column_stack([np.arange(3) * 0.2, np.arange(3) * 2.0])}
    valid_nd_idx = np.array([[0, 0], [1, 2], [3, 1]])
    F = assemble_F(valid_nd_idx, ['w', 'kv'], efm,
                   comp_nd_list=[], new_metric_nd=nm)
    # 1 combined + 2 cols per expr_key (metric, comp_obj) × 2 keys = 5
    assert F.shape == (3, 5)
    np.testing.assert_allclose(F[:, 0], nm[(0, 1, 3), (0, 2, 1)])
    np.testing.assert_allclose(F[:, 1:3], efm['w'][[0, 1, 3]])
    np.testing.assert_allclose(F[:, 3:5], efm['kv'][[0, 2, 1]])


def test_assemble_F_with_comp_obj_appends_columns():
    nd_shape = (3, 3)
    nm = np.random.RandomState(0).rand(*nd_shape)
    efm = {'w': np.column_stack([np.arange(3), np.arange(3) * 10.0])}
    comp_nd = np.arange(9).reshape(nd_shape).astype(float)
    valid_nd_idx = np.array([[0, 1], [2, 2]])
    F = assemble_F(valid_nd_idx, ['w'], efm, [comp_nd], nm)
    # 1 combined + 2 (per-axis 'w') + 1 comp_obj = 4
    assert F.shape == (2, 4)
    np.testing.assert_allclose(F[:, -1], comp_nd[(0, 2), (1, 2)])


# ───────────────────────── quantile_select (end-to-end) ─────────────────────────

def test_quantile_select_picks_unique_in_range(default_arch, synthetic_archives,
                                               config, group_size):
    expr_keys = ['w', 'kv', 'kvdim']
    valid_nd_idx = np.array(np.meshgrid(*[range(3)] * 3,
                                         indexing='ij')).reshape(3, -1).T
    quantile_specs = {'wbits': [0.1, 0.5, 0.9], 'kvbits': [0.1, 0.5, 0.9]}

    cache = {}
    I_quant, mvals = quantile_select(quantile_specs, valid_nd_idx, expr_keys,
                                     synthetic_archives, default_arch,
                                     config, group_size, n_token=0,
                                     axis_cache=cache, verbose=False)
    assert 1 <= len(I_quant) <= 9
    assert len(set(I_quant)) == len(I_quant)
    assert all(0 <= i < len(valid_nd_idx) for i in I_quant)
    for k in quantile_specs:
        assert mvals[k].shape == (len(valid_nd_idx),)
        assert mvals[k].min() <= mvals[k].max()
    # Cache populated for every requested key
    assert set(cache) == set(quantile_specs)


def test_quantile_select_targets_match_quantiles(default_arch, synthetic_archives,
                                                 config, group_size):
    """Picks must include the architecture nearest each requested quantile."""
    expr_keys = ['w', 'kv', 'kvdim']
    valid_nd_idx = np.array(np.meshgrid(*[range(3)] * 3,
                                         indexing='ij')).reshape(3, -1).T
    qs = {'wbits': [0.0, 1.0]}  # min and max
    I_quant, mvals = quantile_select(qs, valid_nd_idx, expr_keys,
                                     synthetic_archives, default_arch,
                                     config, group_size, n_token=0,
                                     verbose=False)
    picked_w = mvals['wbits'][I_quant]
    assert mvals['wbits'].min() in picked_w
    assert mvals['wbits'].max() in picked_w
