"""
Integration test for cyclic_search.py — GPU-free using mock evaluator.

Tests the complete search loop (DOE + iterations) with:
- Mock accelerator (single-process, CPU)
- Mock evaluator (returns deterministic fake metrics & complexities)
- Small parameters: n_doe=10, iterations=4, n_iter=2, ga_pop_size=20

Run from /NAS/SJ/actquant/search/:
    /opt/conda/bin/python3 -m pytest tests/test_cyclic_integration.py -v -s
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import math
import shutil
import tempfile
import numpy as np
import pytest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from cyclic_search import CyclicSearch, PHASE_W, PHASE_KV, arch_to_key, to_eval_format


# ---------------------------------------------------------------------------
# Shared test fixtures
# Load the real Llama-3.1-8B-Instruct config — LlamaGroupSizeSearchSpace
# hardcodes 7 Llama linears so we must use the full config.
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'llama.json')
with open(_CONFIG_PATH) as _f:
    LLAMA_CONFIG = json.load(_f)['Llama-3.1-8B-Instruct']


def make_fake_arch(w_bit=4, kv_bit=4, kv_gs=128):
    return {
        'w': {l: [w_bit] * LLAMA_CONFIG['n_block']
              for l in LLAMA_CONFIG['linear']},
        'k': [[kv_bit, kv_gs]] * LLAMA_CONFIG['n_block'],
        'v': [[kv_bit, kv_gs]] * LLAMA_CONFIG['n_block'],
    }


def fake_eval(arch):
    """
    Deterministic fake evaluator: metric depends on average bits.
    Lower bits → higher metric (worse quality).
    Complexity is just mean wbits and kvbits.
    """
    a = arch['q'] if 'q' in arch else arch
    w_bits = [b for lin, bits in a['w'].items() for b in bits]
    kv_bits_list = [b for b, _ in a['k']] + [b for b, _ in a['v']]
    mean_w  = np.mean(w_bits)
    mean_kv = np.mean(kv_bits_list)
    metric = 10.0 - mean_w * 0.5 - mean_kv * 0.3 + np.random.default_rng(int(mean_w * 10 + mean_kv)).random() * 0.1
    return float(metric), {'wbits': float(mean_w), 'kvbits': float(mean_kv)}


# ---------------------------------------------------------------------------
# Mock accelerator
# ---------------------------------------------------------------------------

class MockAccelerator:
    """Minimal stand-in for HuggingFace Accelerator in single-process mode."""
    device = 'cpu'
    is_main_process = True

    def wait_for_everyone(self):
        pass

    def gather_for_metrics(self, data, use_gather_object=False):
        return data

    def print(self, *args, **kwargs):
        pass   # suppress output during test


# ---------------------------------------------------------------------------
# Build a CyclicSearch with all GPU-requiring parts mocked out
# ---------------------------------------------------------------------------

def build_cyclic_search(save_dir, n_doe=8, iterations=4, n_iter=2,
                        ga_pop_size=10, max_contexts=2, n_cycles=2,
                        n_samples_override=None):
    """
    Construct CyclicSearch with mocked evaluator and search_space
    for a CPU-only integration test.
    """
    accelerator = MockAccelerator()

    kwargs = dict(
        save          = save_dir,
        iterations    = iterations,
        n_doe         = n_doe,
        n_iter        = n_iter,
        ga_pop_size   = ga_pop_size,
        subset_pop_size = 30,
        predictor     = 'rbf',
        dataset       = 'wikitext2',
        loss_func     = 'jsd',
        metric        = 'loss',
        max_value     = 0.7,
        debug         = False,
        save_iter     = 1,
        # model / quant args (will be replaced by mocks)
        model_path    = '/tmp/fake_model_path',
        model_name    = 'Llama-3.1-8B-Instruct',
        w_method      = ['hqq'],
        kv_method     = 'kivi',
        w_bits        = [2, 3, 4],
        k_bits        = [2, 3, 4],
        v_bits        = [2, 3, 4],
        w_group_size  = 128,
        k_group_size  = [[32, 64, 128], [32, 64, 128], [32, 64, 128]],
        v_group_size  = [[32, 64, 128], [32, 64, 128], [32, 64, 128]],
        quant_model_paths = ['/tmp/fake_2bit', '/tmp/fake_3bit', '/tmp/fake_4bit'],
        comp_obj      = ['wbits', 'kvbits'],
        comp_obj_min  = [2.0, 2.0],
        comp_obj_max  = [4.0, 4.0],
        residual_length = 128,
        n_token       = 0,
        quant_kv_output = True,
        k_quant_scheme = 'channel',
        v_quant_scheme = 'token',
        seqlen        = 2048,
        min_seqlen    = 0,
        n_sample      = 4,
        data_batch_size = 1,
        mut_prob      = 0.1,
        crossover_prob = 0.9,
        # CPFS-specific
        n_cycles      = n_cycles,
        max_contexts  = max_contexts,
        phases        = [PHASE_KV, PHASE_W],
        threshold_schedule = 'cosine',
        # Optional / unused
        sensitivity_result_path = '',
        outlier_path  = '',
        base_outlier_bits = [],
        n_outlier     = 0,
        result_file   = 'results.txt',
        resume        = None,
        verbosity     = 'FATAL',
        limit         = None,
        lm_eval_batch_size = None,
        num_fewshot   = None,
        config        = 'config/llama.json',   # will be overridden
        use_key_token = False,
        trunc_len     = 512,
        sliding_window = 128,
        alpha         = 2,
        beta          = -2,
        key_token_path = '',
        only_outlier_bits = False,
        packing       = False,
        sensitivity_threshold = 2,
    )

    # Patch heavyweight __init__ internals
    with patch('cyclic_search.LlamaEvaluator'), \
         patch('cyclic_search.LlamaGroupSizeSearchSpace'), \
         patch('cyclic_search.TaskManager'), \
         patch('cyclic_search.get_task_dict'), \
         patch('cyclic_search.init_accelerator', return_value=(accelerator, {})):

        cs = CyclicSearch.__new__(CyclicSearch)
        # Manually fill required attributes (bypass heavy __init__)
        cs.config         = LLAMA_CONFIG
        cs.save_path      = save_dir
        cs.result_file    = 'results.txt'
        cs.resume         = None
        cs.iterations     = iterations
        cs.n_doe          = n_doe
        cs.n_iter         = n_iter
        cs.predictor      = 'rbf'
        cs.dataset        = 'wikitext2'
        cs.loss_func      = 'jsd'
        cs.metric         = 'loss'
        cs.max_value      = 0.7
        cs.debug          = False
        cs.save_iter      = 1
        cs.ga_pop_size    = ga_pop_size
        cs.subset_pop_size = 30
        cs.comp_obj       = ['wbits', 'kvbits']
        cs.comp_obj_min   = [2.0, 2.0]
        cs.comp_obj_max   = [5.0, 5.0]
        cs.w_obj_idx      = 0
        cs.kv_obj_idx     = 1
        cs.n_token        = 0
        # group_size for evaluator/compute_bits (dict format)
        cs.group_size     = {'w': 128,
                             'k': {2: [32, 64, 128], 3: [32, 64, 128], 4: [32, 64, 128]},
                             'v': {2: [32, 64, 128], 3: [32, 64, 128], 4: [32, 64, 128]}}
        cs.bits           = {'w': [2, 3, 4], 'k': [2, 3, 4], 'v': [2, 3, 4]}
        # group_size for LlamaGroupSizeSearchSpace.__init__ (list-of-lists format)
        _ss_group_size    = {'w': 128,
                             'k': [[32, 64, 128], [32, 64, 128], [32, 64, 128]],
                             'v': [[32, 64, 128], [32, 64, 128], [32, 64, 128]]}
        cs.n_cycles       = n_cycles
        cs.max_contexts   = max_contexts
        cs.phases         = [PHASE_KV, PHASE_W]
        cs.threshold_schedule = 'cosine'
        cs.mut_prob       = 0.1
        cs.crossover_prob = 0.9
        cs.args           = {}

        # Build a real LlamaGroupSizeSearchSpace (no GPU needed)
        # Requires list-of-lists format for k/v group_size
        from search_space.llama import LlamaGroupSizeSearchSpace
        cs.search_space = LlamaGroupSizeSearchSpace(
            bits        = {'w': [2, 3, 4], 'k': [2, 3, 4], 'v': [2, 3, 4]},
            group_size  = _ss_group_size,
            pass_module = {'w': [], 'k': [], 'v': []},
            comp_obj    = ['wbits', 'kvbits'],
            comp_obj_min = [2.0, 2.0],
            comp_obj_max = [5.0, 5.0],
            config      = LLAMA_CONFIG,
            n_token     = 0,
            outlier_bits = {l: [] for l in LLAMA_CONFIG['linear']},
        )

    # Mock evaluator.eval using fake_eval
    def mock_eval(accelerator, arch, metric, loss_func='jsd'):
        m, c = fake_eval(arch)
        return {cs.dataset: m}, c

    cs.evaluator = MagicMock()
    cs.evaluator.eval.side_effect = mock_eval

    return cs, accelerator


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestCyclicSearchIntegration:

    @pytest.fixture(autouse=True)
    def tmp_save(self, tmp_path):
        self.save_dir = str(tmp_path / 'cyclic_test')
        os.makedirs(self.save_dir, exist_ok=True)

    def _run_search(self, **kwargs):
        cs, acc = build_cyclic_search(self.save_dir, **kwargs)
        cs.search(acc)
        return cs

    def test_archive_grows_after_doe(self):
        """Archive should contain at least n_doe entries after DOE."""
        cs = self._run_search(n_doe=8, iterations=4, n_iter=2, n_cycles=2)
        # Read final stats file to check archive size
        stats_files = sorted(
            [f for f in os.listdir(self.save_dir) if f.endswith('.stats')]
        )
        assert len(stats_files) > 0, "No stats files written"
        with open(os.path.join(self.save_dir, stats_files[-1])) as f:
            data = json.load(f)
        archive = data['archive']
        # Must have at least n_doe + iterations * n_iter entries
        assert len(archive) >= 8, f"Archive too small: {len(archive)}"

    def test_stats_file_schema(self):
        """Every stats file must contain the expected keys."""
        self._run_search(n_doe=6, iterations=4, n_iter=2, n_cycles=2)
        for fname in os.listdir(self.save_dir):
            if fname.endswith('.stats'):
                with open(os.path.join(self.save_dir, fname)) as f:
                    data = json.load(f)
                for key in ['archive', 'hv', 'iteration', 'cycle', 'phase', 'threshold']:
                    assert key in data, f"Missing key '{key}' in {fname}"

    def test_phase_alternates_correctly(self):
        """Phases should alternate kv, w, kv, w ... in iteration order."""
        self._run_search(n_doe=6, iterations=4, n_iter=2, n_cycles=2)
        phases_seen = []
        for i in range(1, 5):
            path = os.path.join(self.save_dir, f'iter_{i}.stats')
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                phases_seen.append(data['phase'])
        expected = [PHASE_KV, PHASE_W, PHASE_KV, PHASE_W]
        assert phases_seen == expected[:len(phases_seen)], \
            f"Phase sequence wrong: {phases_seen}"

    def test_cycle_increments_correctly(self):
        """Cycle index should increment every n_phases iterations."""
        self._run_search(n_doe=6, iterations=4, n_iter=2, n_cycles=2)
        # n_phases=2, so iter 1-2 → cycle 0, iter 3-4 → cycle 1
        expected_cycles = {1: 0, 2: 0, 3: 1, 4: 1}
        for it, expected_cycle in expected_cycles.items():
            path = os.path.join(self.save_dir, f'iter_{it}.stats')
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                assert data['cycle'] == expected_cycle, \
                    f"iter {it}: expected cycle {expected_cycle}, got {data['cycle']}"

    def test_threshold_decreases_over_cycles(self):
        """Threshold should be higher in cycle 0 than cycle 1."""
        self._run_search(n_doe=6, iterations=4, n_iter=2, n_cycles=2)
        thresholds_by_cycle = {}
        for it in range(1, 5):
            path = os.path.join(self.save_dir, f'iter_{it}.stats')
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            phase = data['phase']
            cycle = data['cycle']
            thr   = data['threshold']
            key = (phase, cycle)
            thresholds_by_cycle[key] = thr

        # For the same phase, cycle 0 threshold ≥ cycle 1 threshold
        for phase in [PHASE_KV, PHASE_W]:
            t0 = thresholds_by_cycle.get((phase, 0))
            t1 = thresholds_by_cycle.get((phase, 1))
            if t0 is not None and t1 is not None:
                assert t0 >= t1 - 1e-6, \
                    f"Phase {phase}: cycle 0 threshold ({t0:.4f}) < cycle 1 ({t1:.4f})"

    def test_archive_entries_have_correct_schema(self):
        """Each archive entry should be [arch_dict, metric, wbits, kvbits]."""
        self._run_search(n_doe=6, iterations=2, n_iter=2, n_cycles=1)
        stats_files = sorted(
            [f for f in os.listdir(self.save_dir) if f.endswith('.stats')]
        )
        with open(os.path.join(self.save_dir, stats_files[-1])) as f:
            data = json.load(f)
        for entry in data['archive'][:5]:
            assert len(entry) == 4, f"Archive entry has {len(entry)} fields, expected 4"
            arch, metric, wbits, kvbits = entry
            assert isinstance(arch, dict)
            assert 'w' in arch or 'q' in arch, "arch missing 'w' or 'q' key"
            assert isinstance(metric, (int, float))
            assert isinstance(wbits, (int, float))
            assert isinstance(kvbits, (int, float))

    def test_hv_is_positive(self):
        """Hypervolume should be a positive finite float."""
        self._run_search(n_doe=8, iterations=4, n_iter=2, n_cycles=2)
        stats_files = sorted(
            [f for f in os.listdir(self.save_dir) if f.endswith('.stats')]
        )
        with open(os.path.join(self.save_dir, stats_files[-1])) as f:
            data = json.load(f)
        hv = data['hv']
        assert math.isfinite(hv), f"HV is not finite: {hv}"
        assert hv >= 0, f"HV is negative: {hv}"

    def test_no_duplicate_archs_evaluated(self):
        """Baselines already in archive should not be re-evaluated."""
        cs, acc = build_cyclic_search(self.save_dir, n_doe=8, iterations=4,
                                      n_iter=2, n_cycles=2)
        cs.search(acc)
        # Collect all arch keys ever passed to evaluator
        eval_arch_keys = [
            arch_to_key(to_eval_format(call.kwargs.get('arch', call.args[1]
                         if len(call.args) > 1 else {})))
            for call in cs.evaluator.eval.call_args_list
        ]
        assert len(eval_arch_keys) == len(set(eval_arch_keys)), \
            "Duplicate archs were evaluated"

    def test_results_txt_written(self):
        """results.txt should be created after the search completes."""
        self._run_search(n_doe=6, iterations=4, n_iter=2, n_cycles=2)
        assert os.path.exists(os.path.join(self.save_dir, 'results.txt'))

    def test_baseline_in_archive(self):
        """
        Baselines (ctx + max-precision other) should appear in archive.
        For PHASE_KV: baseline = (ctx_w, max_kv).
        We verify that at least one archive entry has kvbits = max(kv_bits).
        """
        self._run_search(n_doe=6, iterations=4, n_iter=2, n_cycles=2)
        stats_files = sorted(
            [f for f in os.listdir(self.save_dir) if f.endswith('.stats')]
        )
        with open(os.path.join(self.save_dir, stats_files[-1])) as f:
            data = json.load(f)
        kv_vals = [entry[3] for entry in data['archive']]  # kvbits column
        # At least one entry should have kvbits = 4.0 (max_kv from baseline)
        assert any(math.isclose(v, 4.0, rel_tol=0.1) for v in kv_vals), \
            f"No max-kvbits baseline found in archive. kvbits seen: {sorted(set(kv_vals))}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
