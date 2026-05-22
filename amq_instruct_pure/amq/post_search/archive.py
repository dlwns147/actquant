"""Archive utilities for post-search pipelines.

Loads the search results JSON, extracts Pareto archs, and selects
candidates near a target bit-width via either ASF (JSD-driven) or
predicted-accuracy top-K.
"""

from __future__ import annotations

import json

import numpy as np

from pymoo.decomposition.asf import ASF
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def load_archive(path):
    """Concatenate ``archive`` + ``candidates`` (matches old amq_quantization.py)."""
    with open(path, 'r') as f:
        archive = json.load(f)['archive']
    return list(archive)


def _entry_mean_jsd(entry):
    return float(entry[-2])


def _entry_bits(entry):
    return float(entry[-1])


def extract_pareto(entries):
    """Return ``(idx, archs, mean_jsds, bits)`` for the first Pareto front
    on (mean_jsd, bits_usage). Both objectives are minimised.
    """
    archs = [e[0] for e in entries]
    mean_jsd = np.array([_entry_mean_jsd(e) for e in entries], dtype=float)
    bits = np.array([_entry_bits(e) for e in entries], dtype=float)
    F = np.column_stack((mean_jsd, bits))
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    front = np.sort(front)
    return front, [archs[i] for i in front], mean_jsd[front], bits[front]


def select_near_target_by_asf(entries, target_bits, target_bits_offset,
                              num_of_candidates, owq_offset=0.0):
    """Filter to ``bits ∈ (target ± offset)``, then pick ``num_of_candidates``
    using PyMOO's ASF on (mean_jsd, bits) toward ``target_bits``.

    Mirrors the selection block from the original amq_quantization.py.
    Returns ``(global_idx, archs, mean_jsds, bits)``.
    """
    archs = [e[0] for e in entries]
    mean_jsd = np.array([_entry_mean_jsd(e) for e in entries], dtype=float)
    bits = np.array([_entry_bits(e) for e in entries], dtype=float) + owq_offset

    sort_idx = np.argsort(mean_jsd)
    stack = np.column_stack((mean_jsd, bits))[sort_idx, :]

    flag = np.logical_and(
        stack[:, 1] > (target_bits - target_bits_offset),
        stack[:, 1] < (target_bits + target_bits_offset),
    )
    range_idx = np.argwhere(flag).flatten()
    if range_idx.size == 0:
        raise ValueError(
            f"No archive entries within target_bits {target_bits} ± {target_bits_offset}"
        )

    filtered_stack = stack[range_idx, :]
    filtered_archs = [archs[sort_idx[i]] for i in range_idx]

    weights = np.array([0, target_bits], dtype=float)
    k = min(num_of_candidates, len(range_idx))
    asf_order = ASF().do(filtered_stack, weights).argsort()[:k]

    global_idx = np.array([sort_idx[range_idx[i]] for i in asf_order], dtype=int)
    selected_archs = [filtered_archs[i] for i in asf_order]
    selected_jsd = filtered_stack[asf_order, 0]
    selected_bits = filtered_stack[asf_order, 1]
    return global_idx, selected_archs, selected_jsd, selected_bits


def filter_and_topk_by_pred(entries, pred_acc, target_bits, target_bits_offset,
                            num_of_candidates):
    """Keep entries with ``bits ∈ (target ± offset)``, then return the
    ``num_of_candidates`` with the highest predicted accuracy.

    ``pred_acc`` must be a 1-D array aligned with ``entries``.
    """
    pred_acc = np.asarray(pred_acc, dtype=float).reshape(-1)
    archs = [e[0] for e in entries]
    bits = np.array([_entry_bits(e) for e in entries], dtype=float)

    flag = np.logical_and(
        bits > (target_bits - target_bits_offset),
        bits < (target_bits + target_bits_offset),
    )
    range_idx = np.argwhere(flag).flatten()
    if range_idx.size == 0:
        raise ValueError(
            f"No archive entries within target_bits {target_bits} ± {target_bits_offset}"
        )

    sub_acc = pred_acc[range_idx]
    k = min(num_of_candidates, range_idx.size)
    order = np.argsort(sub_acc)[::-1][:k]

    global_idx = range_idx[order]
    selected_archs = [archs[i] for i in global_idx]
    selected_bits = bits[global_idx]
    selected_pred = sub_acc[order]
    return global_idx, selected_archs, selected_pred, selected_bits
