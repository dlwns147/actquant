"""Lazy (no --expr_front) comp_obj-first enumeration == brute-force dense.

build_nd's _build_lazy_comp reuses the exact comp formulas the dense path
uses, so the only novel logic is the sparse enumeration/join in
utils.select._lazy_feasible — verified here against a full np.indices grid
+ boolean mask (what select_valid_nd_idx would return if it could fit).
"""
import numpy as np
import pytest

from utils.func import _LazyComp
from utils.select import _lazy_feasible, _lazy_assemble_F


def _lc(expr_keys, sizes, comp_specs):
    efm = {k: np.random.default_rng(len(k)).random((n, 2))
           for k, n in zip(expr_keys, sizes)}
    lc = _LazyComp(expr_keys, efm, comp_specs, ['o'] * len(comp_specs))
    lc.nd_shape = tuple(sizes)
    return lc


def _brute(sizes, comp_eval, lohi):
    g = np.indices(sizes).reshape(len(sizes), -1).T
    keep = np.ones(len(g), bool)
    for ce, (lo, hi) in zip(comp_eval, lohi):
        v = ce(g)
        keep &= (v >= lo) & (v <= hi)
    return g[keep]


def _same_set(a, b):
    if a.shape != b.shape:
        return False
    a = a[np.lexsort(a.T[::-1])]
    b = b[np.lexsort(b.T[::-1])]
    return np.array_equal(a, b)


def _run(lc, lohi):
    return _lazy_feasible(lc, [l for l, _ in lohi], [h for _, h in lohi],
                          None, False, False, np.random, False)


def test_separable_three_axis():
    rng = np.random.default_rng(1)
    sizes = (7, 9, 5)
    cv = [rng.random(n) for n in sizes]
    lc = _lc(['w', 'kv', 'kvdim'], sizes,
             [{'kind': '1d', 'axis': a, 'vals': cv[a]} for a in range(3)])
    lohi = [(.2, .8), (.1, .9), (.3, .95)]
    got = _run(lc, lohi)
    exp = _brute(sizes, [lambda g, a=a: cv[a][g[:, a]] for a in range(3)], lohi)
    assert len(got) and _same_set(got, exp)


def test_memory_additive_diagonal():
    rng = np.random.default_rng(2)
    sizes = (8, 6, 5)
    w = rng.random(sizes[0]) * 10 + 5
    kvc = rng.random((sizes[1], sizes[2])) * 8 + 2
    spec = {'kind': 'memory', 'w_axis': 0, 'w_mem': w, 'w_const': 0.0,
            'kv': {'kind': '2d', 'axes': (1, 2), 'vals': kvc}}
    lc = _lc(['w', 'kv', 'kvdim'], sizes, [spec])
    got = _run(lc, [(12., 16.)])
    exp = _brute(sizes, [lambda g: w[g[:, 0]] + kvc[g[:, 1], g[:, 2]]],
                 [(12., 16.)])
    assert len(got) and _same_set(got, exp)


def test_memory_plus_single_axis():
    rng = np.random.default_rng(3)
    sizes = (6, 7, 4)
    w = rng.random(sizes[0]) * 5 + 3
    kvc = rng.random((sizes[1], sizes[2])) * 5 + 1
    kvb = rng.random(sizes[1])
    mem = {'kind': 'memory', 'w_axis': 0, 'w_mem': w, 'w_const': 0.0,
           'kv': {'kind': '2d', 'axes': (1, 2), 'vals': kvc}}
    lc = _lc(['w', 'kv', 'kvdim'], sizes,
             [mem, {'kind': '1d', 'axis': 1, 'vals': kvb}])
    lohi = [(6., 10.), (.25, .8)]
    got = _run(lc, lohi)
    exp = _brute(sizes, [lambda g: w[g[:, 0]] + kvc[g[:, 1], g[:, 2]],
                         lambda g: kvb[g[:, 1]]], lohi)
    assert len(got) and _same_set(got, exp)


def test_eff_kvbits_2d():
    rng = np.random.default_rng(4)
    sizes = (5, 6)
    eff = rng.random(sizes) * 3 + 2
    lc = _lc(['kv', 'kvdim'], sizes,
             [{'kind': '2d', 'axes': (0, 1), 'vals': eff}])
    got = _run(lc, [(3., 4.5)])
    exp = _brute(sizes, [lambda g: eff[g[:, 0], g[:, 1]]], [(3., 4.5)])
    assert len(got) and _same_set(got, exp)


def test_memory_no_w_axis():
    rng = np.random.default_rng(5)
    sizes = (6, 5)
    kvc = rng.random(sizes) * 6 + 2
    spec = {'kind': 'memory', 'w_axis': None, 'w_mem': None, 'w_const': 7.0,
            'kv': {'kind': '2d', 'axes': (0, 1), 'vals': kvc}}
    lc = _lc(['kv', 'kvdim'], sizes, [spec])
    got = _run(lc, [(10., 13.)])
    exp = _brute(sizes, [lambda g: 7.0 + kvc[g[:, 0], g[:, 1]]], [(10., 13.)])
    assert len(got) and _same_set(got, exp)


def test_sorted_by_first_comp_and_F_has_nan_col():
    rng = np.random.default_rng(6)
    sizes = (5, 4)
    c0 = rng.random(sizes[0])
    lc = _lc(['kv', 'kvdim'], sizes,
             [{'kind': '1d', 'axis': 0, 'vals': c0}])
    got = _run(lc, [(0.0, 1.0)])
    key = c0[got[:, 0]]
    assert np.all(np.diff(key) >= 0)            # ascending by first comp_obj
    F = _lazy_assemble_F(lc, got)
    assert np.isnan(F[:, 0]).all()              # combined col is the sentinel
    assert F.shape == (len(got), 1 + 2 * 2 + 1)


def test_empty_when_excluded():
    lc = _lc(['kv'], (5,), [{'kind': '1d', 'axis': 0,
                             'vals': np.arange(5.0)}])
    got = _run(lc, [(99., 100.)])
    assert got.shape == (0, 1)


def test_no_comp_obj_raises():
    lc = _lc(['kv'], (3,), [])
    with pytest.raises(SystemExit):
        _lazy_feasible(lc, [], [], None, False, False, np.random, False)
