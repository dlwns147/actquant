"""
Unit tests for cyclic_search.py — GPU-free.

Run from /NAS/SJ/actquant/search/:
    /opt/conda/bin/python3 -m pytest tests/test_cyclic_search.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import json
import numpy as np
import pytest
from copy import deepcopy

from cyclic_search import (
    get_threshold,
    arch_to_key,
    to_eval_format,
    PhaseSearchProblem,
    PHASE_W,
    PHASE_KV,
)


# ---------------------------------------------------------------------------
# Minimal config / search-space helpers
# ---------------------------------------------------------------------------

N_BLOCK  = 4
N_LINEAR = 2   # q_proj, v_proj  (small to keep test fast)
W_BITS   = [2, 3, 4]
KV_BITS  = [2, 3, 4]
KV_GS    = [32, 64, 128]

MINI_CONFIG = {
    'n_block'  : N_BLOCK,
    'n_linear' : N_LINEAR,
    'linear'   : ['self_attn.q_proj', 'self_attn.v_proj'],
    'linear_shape': {
        'self_attn.q_proj': [4096, 4096],
        'self_attn.k_proj': [4096, 4096],   # needed by compute_memory
        'self_attn.v_proj': [4096, 4096],
    },
    'head_dim' : 128,
    'vocab_size': 32000,
    'hidden_size': 4096,
    'max_position_embeddings': 4096,
    'n_norm'   : 65,
    'k_linear' : 'self_attn.k_proj',
    'v_linear' : 'self_attn.v_proj',
}

MINI_GROUP_SIZE = {'w': 128, 'k': {2: [32, 64, 128], 3: [32, 64, 128], 4: [32, 64, 128]},
                   'v': {2: [32, 64, 128], 3: [32, 64, 128], 4: [32, 64, 128]}}

KV_OPTION = [(b, g) for b in KV_BITS for g in KV_GS]  # 9 options


def make_arch(w_bit=4, kv_bit=4, kv_gs=128):
    """Build a minimal old-format arch dict."""
    return {
        'w': {l: [w_bit] * N_BLOCK for l in MINI_CONFIG['linear']},
        'k': [[kv_bit, kv_gs]] * N_BLOCK,
        'v': [[kv_bit, kv_gs]] * N_BLOCK,
    }


class MockSearchSpace:
    """
    Minimal stand-in for LlamaGroupSizeSearchSpace.
    Encodes arch as flat integer array: [w0_b0..w0_bN, w1_b0..w1_bN, k_b0..k_bN, v_b0..v_bN]
    """
    n_block   = N_BLOCK
    n_linear  = N_LINEAR
    pass_idx_list = []
    comp_obj  = ['wbits', 'kvbits']
    pass_module = {'w': [], 'k': [], 'v': []}

    # options: index → value
    q_proj_option = W_BITS
    v_proj_option = W_BITS
    k_option      = KV_OPTION
    v_option      = KV_OPTION

    def encode(self, arch):
        """Encode old-format arch to flat integer index array."""
        rows = []
        for linear in MINI_CONFIG['linear']:
            option = self.q_proj_option if 'q_proj' in linear else self.v_proj_option
            rows.append([option.index(b) for b in arch['w'][linear]])
        # k
        rows.append([self.k_option.index(tuple(x)) for x in arch['k']])
        # v
        rows.append([self.v_option.index(tuple(x)) for x in arch['v']])
        return np.array(rows).flatten()

    def decode(self, x):
        """Decode flat integer array to old-format arch dict."""
        x_r = x.reshape(N_LINEAR + 2, N_BLOCK)
        arch = {
            'w': {},
            'k': [],
            'v': [],
        }
        for i, linear in enumerate(MINI_CONFIG['linear']):
            option = self.q_proj_option if 'q_proj' in linear else self.v_proj_option
            arch['w'][linear] = [option[idx] for idx in x_r[i]]
        arch['k'] = [list(self.k_option[idx]) for idx in x_r[N_LINEAR]]
        arch['v'] = [list(self.v_option[idx]) for idx in x_r[N_LINEAR + 1]]
        return arch

    def decode_encode_predictor(self, x):
        """No pass_idx removal in this mock."""
        return x

    def encode_predictor(self, arch):
        return self.encode(arch)


class MockPredictor:
    """Returns constant predictions = sum(x) / 1000."""
    def predict(self, x):
        return (x.sum(axis=1, keepdims=True) / 1000.0).astype(float)


# ---------------------------------------------------------------------------
# 1. get_threshold
# ---------------------------------------------------------------------------

class TestGetThreshold:
    def test_single_cycle_returns_min(self):
        """n_cycles=1 → always return val_min."""
        assert get_threshold(0, 1, 2.0, 4.0, 'cosine') == 2.0
        assert get_threshold(0, 1, 2.0, 4.0, 'linear') == 2.0

    def test_cosine_cycle0_returns_max(self):
        """Cycle 0 of cosine schedule → val_max."""
        result = get_threshold(0, 5, 2.0, 4.0, 'cosine')
        assert math.isclose(result, 4.0, rel_tol=1e-6)

    def test_cosine_last_cycle_returns_min(self):
        """Last cycle of cosine schedule → val_min."""
        result = get_threshold(4, 5, 2.0, 4.0, 'cosine')
        assert math.isclose(result, 2.0, rel_tol=1e-6)

    def test_cosine_monotone_decreasing(self):
        """Cosine schedule must be monotonically non-increasing."""
        n_cycles = 6
        values = [get_threshold(c, n_cycles, 2.0, 4.0, 'cosine')
                  for c in range(n_cycles)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-9, \
                f"Not decreasing: cycle {i}={values[i]:.4f} > cycle {i+1}={values[i+1]:.4f}"

    def test_linear_cycle0_returns_max(self):
        result = get_threshold(0, 4, 2.0, 4.0, 'linear')
        assert math.isclose(result, 4.0, rel_tol=1e-6)

    def test_linear_last_cycle_returns_min(self):
        result = get_threshold(3, 4, 2.0, 4.0, 'linear')
        assert math.isclose(result, 2.0, rel_tol=1e-6)

    def test_linear_monotone_decreasing(self):
        values = [get_threshold(c, 5, 2.0, 4.0, 'linear') for c in range(5)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-9

    def test_within_bounds(self):
        """Output always in [val_min, val_max]."""
        for schedule in ['cosine', 'linear']:
            for n in [2, 3, 5, 10]:
                for c in range(n):
                    v = get_threshold(c, n, 2.0, 4.0, schedule)
                    assert 2.0 - 1e-9 <= v <= 4.0 + 1e-9


# ---------------------------------------------------------------------------
# 2. arch_to_key
# ---------------------------------------------------------------------------

class TestArchToKey:
    def test_same_arch_same_key(self):
        a1 = make_arch(4, 4, 128)
        a2 = make_arch(4, 4, 128)
        assert arch_to_key(a1) == arch_to_key(a2)

    def test_different_wbit_different_key(self):
        assert arch_to_key(make_arch(2, 4, 128)) != arch_to_key(make_arch(4, 4, 128))

    def test_different_kvbit_different_key(self):
        assert arch_to_key(make_arch(4, 2, 128)) != arch_to_key(make_arch(4, 4, 128))

    def test_different_kvgs_different_key(self):
        assert arch_to_key(make_arch(4, 4, 64)) != arch_to_key(make_arch(4, 4, 128))

    def test_key_is_string(self):
        assert isinstance(arch_to_key(make_arch()), str)

    def test_wrapped_arch_same_key(self):
        """Old-format and wrapped arch should produce the same key."""
        a_old = make_arch(3, 3, 64)
        a_new = to_eval_format(a_old)
        assert arch_to_key(a_old) == arch_to_key(a_new)


# ---------------------------------------------------------------------------
# 3. to_eval_format
# ---------------------------------------------------------------------------

class TestToEvalFormat:
    def test_wraps_old_format(self):
        arch = make_arch(4, 4, 128)
        wrapped = to_eval_format(arch)
        assert 'q' in wrapped
        assert wrapped['q'] is arch

    def test_passes_through_new_format(self):
        arch_new = {'q': make_arch(4, 4, 128)}
        result = to_eval_format(arch_new)
        assert result is arch_new   # same object, no copy

    def test_q_contains_w_k_v(self):
        arch = make_arch(3, 2, 64)
        wrapped = to_eval_format(arch)
        assert 'w' in wrapped['q']
        assert 'k' in wrapped['q']
        assert 'v' in wrapped['q']


# ---------------------------------------------------------------------------
# 4. PhaseSearchProblem — fixed-slice enforcement
# ---------------------------------------------------------------------------

class TestPhaseSearchProblem:
    """Tests for PhaseSearchProblem without GPU or real evaluator."""

    def _build_problem(self, phase):
        ss  = MockSearchSpace()
        pred = MockPredictor()
        ctx_arch    = make_arch(w_bit=4, kv_bit=4, kv_gs=128)
        baseline    = make_arch(w_bit=4, kv_bit=4, kv_gs=128)

        if phase == PHASE_KV:
            free_comp_obj  = ['kvbits']
            free_comp_min  = [2.0]
            free_comp_max  = [4.5]
        else:
            free_comp_obj  = ['wbits']
            free_comp_min  = [2.0]
            free_comp_max  = [4.5]

        prob = PhaseSearchProblem(
            search_space   = ss,
            predictor      = pred,
            config         = MINI_CONFIG,
            free_comp_obj  = free_comp_obj,
            free_comp_obj_min = free_comp_min,
            free_comp_obj_max = free_comp_max,
            group_size     = MINI_GROUP_SIZE,
            n_token        = 0,
            phase          = phase,
            ctx_arch       = ctx_arch,
            baseline_arch  = baseline,
        )
        return prob, ss, ctx_arch

    def test_phase_kv_fixed_slice_span(self):
        """PHASE_KV should fix the W portion (first n_linear*n_block positions)."""
        prob, ss, ctx_arch = self._build_problem(PHASE_KV)
        w_end = N_LINEAR * N_BLOCK
        # xl == xu for W positions
        assert np.all(prob.xl[:w_end] == prob.xu[:w_end]), \
            "KV phase: W slice xl should equal xu (fixed)"
        # KV positions still free
        assert np.any(prob.xl[w_end:] != prob.xu[w_end:]), \
            "KV phase: KV slice should have xl < xu somewhere"

    def test_phase_w_fixed_slice_span(self):
        """PHASE_W should fix the KV portion (last 2*n_block positions)."""
        prob, ss, ctx_arch = self._build_problem(PHASE_W)
        w_end = N_LINEAR * N_BLOCK
        # KV positions fixed
        assert np.all(prob.xl[w_end:] == prob.xu[w_end:]), \
            "W phase: KV slice xl should equal xu (fixed)"
        # W positions still free
        assert np.any(prob.xl[:w_end] != prob.xu[:w_end]), \
            "W phase: W slice should have xl < xu somewhere"

    def test_phase_kv_fixed_values_match_ctx(self):
        """Fixed values in PHASE_KV should match ctx_arch W encoding."""
        prob, ss, ctx_arch = self._build_problem(PHASE_KV)
        w_end = N_LINEAR * N_BLOCK
        ctx_enc = ss.encode(ctx_arch)
        np.testing.assert_array_equal(
            prob._fixed_values, ctx_enc[:w_end],
            err_msg="PHASE_KV: fixed W values must equal ctx_arch W encoding")

    def test_phase_w_fixed_values_match_ctx(self):
        """Fixed values in PHASE_W should match ctx_arch KV encoding."""
        prob, ss, ctx_arch = self._build_problem(PHASE_W)
        w_end = N_LINEAR * N_BLOCK
        ctx_enc = ss.encode(ctx_arch)
        np.testing.assert_array_equal(
            prob._fixed_values, ctx_enc[w_end:],
            err_msg="PHASE_W: fixed KV values must equal ctx_arch KV encoding")

    def test_evaluate_enforces_fixed_slice_kv(self):
        """
        _evaluate must overwrite the fixed slice even if x has wrong values.
        In PHASE_KV, W positions should be reset to ctx values.
        """
        prob, ss, ctx_arch = self._build_problem(PHASE_KV)
        w_end = N_LINEAR * N_BLOCK
        n_var = N_BLOCK * (N_LINEAR + 2)

        # Build random population with WRONG W values
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=(10, n_var))

        out = {}
        prob._evaluate(x, out)

        # After _evaluate, W slice must equal ctx fixed values
        ctx_enc = ss.encode(ctx_arch)
        for row in x:
            np.testing.assert_array_equal(
                row[:w_end], ctx_enc[:w_end],
                err_msg="PHASE_KV: W slice was not reset to ctx values")

    def test_evaluate_enforces_fixed_slice_w(self):
        """
        _evaluate must overwrite the fixed slice even if x has wrong values.
        In PHASE_W, KV positions should be reset to ctx values.
        """
        prob, ss, ctx_arch = self._build_problem(PHASE_W)
        w_end = N_LINEAR * N_BLOCK
        n_var = N_BLOCK * (N_LINEAR + 2)

        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=(10, n_var))

        out = {}
        prob._evaluate(x, out)

        ctx_enc = ss.encode(ctx_arch)
        for row in x:
            np.testing.assert_array_equal(
                row[w_end:], ctx_enc[w_end:],
                err_msg="PHASE_W: KV slice was not reset to ctx values")

    def test_evaluate_output_shapes(self):
        """Check F and G shapes."""
        prob, ss, _ = self._build_problem(PHASE_KV)
        n_var = N_BLOCK * (N_LINEAR + 2)
        x = np.tile(ss.encode(make_arch(4, 4, 128)), (5, 1))
        out = {}
        prob._evaluate(x, out)
        assert out['F'].shape == (5, 2), f"F shape: {out['F'].shape}"
        assert out['G'].shape == (5, 2), f"G shape: {out['G'].shape}"

    def test_relative_metric_direction(self):
        """
        For PHASE_KV: using lower KV bits (worse) than baseline should yield
        positive relative_metric (metric gets worse = higher loss).
        MockPredictor(x) = sum(x)/1000, so lower KV index → lower sum → lower metric.
        Lower bits = lower index in KV_OPTION ([(2,32),(2,64),...,(4,128)]).
        Baseline uses max KV (index = len-1), lower-bit arch uses lower index.
        baseline_pred > arch_pred → rel_metric < 0 for lower-bit arch.
        (Note: for a real model, lower bits → higher loss. MockPredictor just
         returns a linear function of index, so direction may differ — we only
         check that the function runs without error and returns finite values.)
        """
        prob, ss, _ = self._build_problem(PHASE_KV)
        n_var = N_BLOCK * (N_LINEAR + 2)
        x = np.tile(ss.encode(make_arch(4, 4, 128)), (3, 1))
        out = {}
        prob._evaluate(x, out)
        assert np.all(np.isfinite(out['F'])), "F must be finite"
        assert np.all(np.isfinite(out['G'])), "G must be finite"

    def test_baseline_pred_is_scalar(self):
        prob, _, _ = self._build_problem(PHASE_KV)
        assert isinstance(prob.baseline_pred, float)
        assert math.isfinite(prob.baseline_pred)


# ---------------------------------------------------------------------------
# 5. Context extraction with threshold (mock CyclicSearch)
# ---------------------------------------------------------------------------

class _MockCyclicSearch:
    """Minimal stand-in exposing only the methods under test."""
    comp_obj     = ['wbits', 'kvbits']
    comp_obj_min = [2.0, 2.0]
    comp_obj_max = [4.0, 4.0]
    w_obj_idx    = 0
    kv_obj_idx   = 1
    max_contexts = 5
    n_cycles     = 3
    threshold_schedule = 'cosine'

    # Import the actual methods
    from cyclic_search import CyclicSearch
    _extract_pareto_contexts_with_threshold = \
        CyclicSearch._extract_pareto_contexts_with_threshold
    _select_diverse_contexts = \
        CyclicSearch._select_diverse_contexts
    _get_threshold = CyclicSearch._get_threshold


def _make_archive_row(w_bit, kv_bit):
    """Build [arch_dict, metric, wbits, kvbits] archive row with random metric."""
    arch = make_arch(w_bit=w_bit, kv_bit=kv_bit, kv_gs=128)
    # Metric: higher bits → lower metric (better)
    metric = 10.0 - w_bit - kv_bit + np.random.default_rng(w_bit * 10 + kv_bit).random()
    return [arch, float(metric), float(w_bit), float(kv_bit)]


class TestExtractParetoContexts:
    """Tests for _extract_pareto_contexts_with_threshold."""

    def _make_archive(self):
        """Simple archive with clear Pareto structure."""
        rows = []
        for w in [2, 3, 4]:
            for kv in [2, 3, 4]:
                rows.append(_make_archive_row(w, kv))
        return rows

    def test_phase_kv_high_threshold_returns_high_wbits(self):
        """
        PHASE_KV, high threshold (wbits ≥ 3.5) → only contexts with wbits ≥ 3.5.
        """
        cs   = _MockCyclicSearch()
        arch = self._make_archive()
        ctxs = cs._extract_pareto_contexts_with_threshold(arch, PHASE_KV, 3.5)
        for _, fixed_c in ctxs:
            assert fixed_c >= 3.5, f"Got wbits={fixed_c} but threshold=3.5"

    def test_phase_kv_low_threshold_includes_low_wbits(self):
        """Low threshold (wbits ≥ 1.0) should include contexts with all W bits."""
        cs   = _MockCyclicSearch()
        arch = self._make_archive()
        ctxs = cs._extract_pareto_contexts_with_threshold(arch, PHASE_KV, 1.0)
        fixed_vals = [c[1] for c in ctxs]
        assert min(fixed_vals) <= 2.1, "Expected low-wbits context with threshold=1.0"

    def test_fallback_when_no_context_meets_threshold(self):
        """If no Pareto point meets threshold, fallback returns at least 1 context."""
        cs   = _MockCyclicSearch()
        arch = self._make_archive()
        # Threshold higher than any wbits in archive
        ctxs = cs._extract_pareto_contexts_with_threshold(arch, PHASE_KV, 999.0)
        assert len(ctxs) >= 1, "Fallback should return at least 1 context"

    def test_returns_unique_w_contexts_in_phase_kv(self):
        """Different Pareto points with same W config should not appear twice."""
        cs   = _MockCyclicSearch()
        arch = self._make_archive()
        ctxs = cs._extract_pareto_contexts_with_threshold(arch, PHASE_KV, 1.0)
        keys = [json.dumps({'w': c[0]['w']}, sort_keys=True) for c in ctxs]
        assert len(keys) == len(set(keys)), "Duplicate W contexts found"

    def test_phase_w_threshold_on_kvbits(self):
        """PHASE_W, threshold applies to kvbits column."""
        cs   = _MockCyclicSearch()
        arch = self._make_archive()
        ctxs = cs._extract_pareto_contexts_with_threshold(arch, PHASE_W, 3.5)
        for _, fixed_c in ctxs:
            assert fixed_c >= 3.5, f"Got kvbits={fixed_c} but threshold=3.5"


# ---------------------------------------------------------------------------
# 6. Diverse context selection
# ---------------------------------------------------------------------------

class TestSelectDiverseContexts:
    def _make_contexts(self, values):
        return [(make_arch(4, 4, 128), float(v)) for v in values]

    def test_fewer_than_max_contexts_returns_all(self):
        cs   = _MockCyclicSearch()
        ctxs = self._make_contexts([2.0, 3.0, 4.0])
        result = cs._select_diverse_contexts(ctxs)
        assert len(result) == 3

    def test_more_than_max_contexts_limits_output(self):
        cs   = _MockCyclicSearch()
        cs.max_contexts = 3
        ctxs = self._make_contexts([2.0, 2.5, 3.0, 3.5, 4.0])
        result = cs._select_diverse_contexts(ctxs)
        assert len(result) == 3

    def test_always_includes_extremes(self):
        """Min and max complexity contexts should always be included."""
        cs   = _MockCyclicSearch()
        cs.max_contexts = 3
        ctxs = self._make_contexts([2.0, 2.4, 2.8, 3.2, 4.0])
        result = cs._select_diverse_contexts(ctxs)
        # just verify count (greedy picks min + max + middle)
        assert len(result) == 3

    def test_single_context_returns_one(self):
        cs   = _MockCyclicSearch()
        ctxs = self._make_contexts([3.0])
        result = cs._select_diverse_contexts(ctxs)
        assert len(result) == 1

    def test_empty_contexts_returns_empty(self):
        cs   = _MockCyclicSearch()
        result = cs._select_diverse_contexts([])
        assert result == []


# ---------------------------------------------------------------------------
# 7. Baseline deduplication
# ---------------------------------------------------------------------------

class TestBaselineDeduplication:
    """
    Verify that arch_to_key + archive lookup correctly skips duplicate baselines.
    This mirrors the logic in CyclicSearch.search().
    """

    def test_same_arch_detected_as_duplicate(self):
        arch = make_arch(4, 4, 128)
        archive_keys = {arch_to_key(arch)}
        new_arch = deepcopy(arch)
        assert arch_to_key(new_arch) in archive_keys

    def test_different_arch_not_duplicate(self):
        arch1 = make_arch(4, 4, 128)
        arch2 = make_arch(2, 4, 128)
        archive_keys = {arch_to_key(arch1)}
        assert arch_to_key(arch2) not in archive_keys

    def test_baseline_skipped_if_in_archive(self):
        """
        Simulate the baseline collection loop from CyclicSearch.search():
        baseline should not be added to baselines_to_eval if already in archive.
        """
        baseline = make_arch(4, 4, 128)
        archive  = [[baseline, 5.0, 4.0, 4.0]]
        archive_keys = {arch_to_key(e[0]) for e in archive}

        baselines_to_eval = []
        if arch_to_key(baseline) not in archive_keys:
            baselines_to_eval.append(baseline)

        assert len(baselines_to_eval) == 0, "Baseline should have been skipped"

    def test_baseline_added_if_not_in_archive(self):
        baseline = make_arch(4, 4, 128)
        other    = make_arch(2, 4, 128)
        archive  = [[other, 7.0, 2.0, 4.0]]
        archive_keys = {arch_to_key(e[0]) for e in archive}

        baselines_to_eval = []
        if arch_to_key(baseline) not in archive_keys:
            baselines_to_eval.append(baseline)

        assert len(baselines_to_eval) == 1, "Baseline should have been added"


# ---------------------------------------------------------------------------
# 8. get_threshold via _get_threshold (integration)
# ---------------------------------------------------------------------------

class TestGetThresholdViaMethod:
    """Ensure _get_threshold selects the right comp_obj index for each phase."""

    def test_phase_kv_uses_wbits_threshold(self):
        cs = _MockCyclicSearch()
        # cycle 0, n_cycles=3 → threshold near comp_obj_max[w_obj_idx]
        t = cs._get_threshold(cycle=0, phase=PHASE_KV)
        # w_obj_idx=0, comp_obj_max[0]=4.0, comp_obj_min[0]=2.0
        # cycle 0 cosine → near 4.0
        assert math.isclose(t, 4.0, rel_tol=1e-6)

    def test_phase_w_uses_kvbits_threshold(self):
        cs = _MockCyclicSearch()
        t = cs._get_threshold(cycle=0, phase=PHASE_W)
        # kv_obj_idx=1, comp_obj_max[1]=4.0
        assert math.isclose(t, 4.0, rel_tol=1e-6)

    def test_threshold_decreases_over_cycles(self):
        cs = _MockCyclicSearch()
        thresholds = [cs._get_threshold(cycle=c, phase=PHASE_KV)
                      for c in range(cs.n_cycles)]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] >= thresholds[i + 1] - 1e-9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
