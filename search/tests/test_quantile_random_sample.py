"""
Verifies the index-selection logic for the three sampling modes in post_search_split.py:
  (A) QUANTILE only
  (B) QUANTILE + extra random (random samples drawn from the complement of quantile picks)
  (C) RANDOM only

The test reproduces the relevant block from post_search_split.py so it can be
exercised in isolation (no model load, no GPU).
"""
import itertools
import numpy as np
import pytest


def select_indices(metric_vals_per_key, quantile_specs, random_sample, n_ps, seed=0):
    """Mirror of post_search_split.py's I-selection block (quantile + optional extra random).

    Args:
        metric_vals_per_key: dict[str, np.ndarray] — per-arch values for each quantile key.
        quantile_specs: dict[str, list[float]] — quantile positions per key.
        random_sample: int or None — number of extra random samples to add.
        n_ps: int — total number of candidate architectures.
        seed: int — RNG seed for reproducibility.

    Returns:
        (I_quant, I_total): both sorted lists of indices into ps.
    """
    np.random.seed(seed)

    target_vals = {
        key: [np.quantile(metric_vals_per_key[key], q) for q in qs]
        for key, qs in quantile_specs.items()
    }

    I_set = set()
    keys = list(quantile_specs.keys())
    for combo in itertools.product(*[range(len(quantile_specs[k])) for k in keys]):
        targets = {k: target_vals[k][qi] for k, qi in zip(keys, combo)}
        dists = np.zeros(n_ps)
        for k, t in targets.items():
            vals = metric_vals_per_key[k]
            val_range = vals.max() - vals.min()
            dists += ((vals - t) / val_range) ** 2 if val_range > 0 else (vals - t) ** 2
        I_set.add(int(np.argmin(dists)))

    I_quant = sorted(I_set)
    I_total = list(I_quant)

    if random_sample is not None and random_sample > 0:
        quant_set = set(I_quant)
        available = np.array([j for j in range(n_ps) if j not in quant_set], dtype=np.int64)
        n_extra = int(min(random_sample, len(available)))
        if n_extra > 0:
            extra = np.random.choice(available, size=n_extra, replace=False)
            extra_list = sorted(int(e) for e in extra)
            assert quant_set.isdisjoint(extra_list)
            I_total = sorted(list(quant_set) + extra_list)

    return I_quant, I_total


@pytest.fixture
def fake_arch_metrics():
    """200 architectures with two synthetic complexity metrics."""
    rng = np.random.default_rng(42)
    n = 200
    return {
        "wbits": rng.uniform(2.0, 4.0, size=n),
        "kvbits": rng.uniform(2.0, 4.0, size=n),
    }, n


def test_mode_A_quantile_only(fake_arch_metrics):
    metrics, n = fake_arch_metrics
    specs = {"wbits": [0.1, 0.5, 0.9], "kvbits": [0.1, 0.5, 0.9]}
    I_q, I_total = select_indices(metrics, specs, random_sample=None, n_ps=n)
    assert I_q == I_total, "Mode A: total should equal quantile selection"
    assert 1 <= len(I_q) <= 9, f"3x3 cartesian → at most 9 unique picks, got {len(I_q)}"
    assert all(0 <= i < n for i in I_q)
    assert len(set(I_q)) == len(I_q), "indices must be unique"


def test_mode_B_quantile_plus_random(fake_arch_metrics):
    metrics, n = fake_arch_metrics
    specs = {"wbits": [0.1, 0.5, 0.9], "kvbits": [0.1, 0.5, 0.9]}
    n_extra = 50
    I_q, I_total = select_indices(metrics, specs, random_sample=n_extra, n_ps=n)

    # Quantile picks must remain a subset of the total
    assert set(I_q).issubset(set(I_total))
    # Extra samples must be exactly n_extra, drawn from complement
    extras = set(I_total) - set(I_q)
    assert len(extras) == n_extra
    assert extras.isdisjoint(set(I_q)), "random samples must exclude quantile picks"
    # Total count
    assert len(I_total) == len(I_q) + n_extra
    # All indices valid and unique
    assert all(0 <= i < n for i in I_total)
    assert len(set(I_total)) == len(I_total)


def test_mode_B_random_clipped_when_pool_smaller(fake_arch_metrics):
    """Requesting more random samples than available → clip to pool size."""
    metrics, n = fake_arch_metrics
    specs = {"wbits": [0.1, 0.5, 0.9], "kvbits": [0.1, 0.5, 0.9]}
    n_extra_request = 10_000  # way larger than n=200
    I_q, I_total = select_indices(metrics, specs, random_sample=n_extra_request, n_ps=n)
    # Total cannot exceed n, and must equal quantile-picks + (n - len(quantile))
    expected_total = len(I_q) + (n - len(I_q))
    assert len(I_total) == expected_total == n


def test_mode_B_reproducible_across_seeds(fake_arch_metrics):
    """Same seed → same extra random selection."""
    metrics, n = fake_arch_metrics
    specs = {"wbits": [0.1, 0.5, 0.9]}
    _, I1 = select_indices(metrics, specs, random_sample=20, n_ps=n, seed=7)
    _, I2 = select_indices(metrics, specs, random_sample=20, n_ps=n, seed=7)
    assert I1 == I2

    _, I3 = select_indices(metrics, specs, random_sample=20, n_ps=n, seed=8)
    assert I1 != I3, "different seed should yield different random extras"


def test_mode_B_zero_extra_equals_mode_A(fake_arch_metrics):
    metrics, n = fake_arch_metrics
    specs = {"wbits": [0.25, 0.75]}
    I_a, _ = select_indices(metrics, specs, random_sample=None, n_ps=n)
    I_b_quant, I_b_total = select_indices(metrics, specs, random_sample=0, n_ps=n)
    assert I_a == I_b_quant == I_b_total
