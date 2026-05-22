"""Train a per-dataset gen-performance predictor over architectures.

Uses the existing ``predictor.factory.get_predictor`` (MLP or RBF) and
the ``SearchSpace.encode`` encoder (no pass-linear filter — constant
features for forced layers cost the predictor nothing).
"""

from __future__ import annotations

import numpy as np

from search.space import SearchSpace
from predictor.factory import get_predictor
from utils.func import get_correlation


def build_search_space(args, config):
    """Construct a SearchSpace good enough for ``encode`` only.

    ``pass_linear_list`` is left empty: we only call ``encode``, which
    doesn't filter on it.
    """
    return SearchSpace(
        config=config,
        n_block=int(config['n_block']),
        n_linear=int(config['n_linear']),
        group_size=args.group_size,
        pass_linear_list=[],
        bits_range=list(args.bits_range),
    )


def _encode_archs(search_space, archs):
    return np.array([search_space.encode(a) for a in archs], dtype=np.float32)


def _holdout_metrics(predictor, x, y, trn_split=0.8, seed=0):
    """Spearman / RMSE on a fixed 80/20 split of (x, y)."""
    n = x.shape[0]
    if n < 5:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    cut = int(n * trn_split)
    vld = perm[cut:]
    if vld.size < 2:
        return float('nan'), float('nan')
    pred = np.asarray(predictor.predict(x[vld])).reshape(-1)
    target = y[vld]
    rmse, rho, _ = get_correlation(pred, target)
    return float(rmse), float(rho)


def train_gen_predictor(archs, metric_list, datasets, search_space,
                        predictor='mlp', device='cpu', verbose=True):
    """Fit one predictor per dataset on (encode(arch) -> metric_list[ds_gen]).

    Returns ``predictors`` dict keyed by dataset name and a ``stats`` dict
    of hold-out metrics for logging.
    """
    inputs = _encode_archs(search_space, archs)
    targets = {ds: np.asarray(metric_list[f"{ds}_gen"], dtype=np.float32)
               for ds in datasets}

    predictors = {
        ds: get_predictor(predictor, inputs, targets[ds], device=device)
        for ds in datasets
    }

    stats = {}
    for ds in datasets:
        rmse, rho = _holdout_metrics(predictors[ds], inputs, targets[ds])
        stats[ds] = {'rmse': rmse, 'spearman': rho}
        if verbose:
            print(f"[predictor:{ds}] hold-out rmse={rmse:.4f} spearman={rho:.4f}")
            if not np.isnan(rho) and rho < 0.5:
                print(f"[predictor:{ds}] WARNING: spearman < 0.5 — top-K picks may be noise")

    return predictors, stats


def predict_archive(predictors, search_space, archs, datasets):
    """Predict per-dataset gen-perf and the dataset-mean for every arch."""
    inputs = _encode_archs(search_space, archs)
    per_ds = {
        ds: np.asarray(predictors[ds].predict(inputs)).reshape(-1)
        for ds in datasets
    }
    mean = np.mean(np.stack([per_ds[ds] for ds in datasets], axis=0), axis=0)
    per_ds['mean'] = mean
    return per_ds
