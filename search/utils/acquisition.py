"""Decision-aware candidate down-selection for the AWQ second-stage NAS.

Pure-numpy, GPU-free, unit-testable in isolation (tests/awq_correction_search/
test_acquisition.py). Extends the geometry-only `subset_select` down-selector with
two opt-in quotas motivated by the AWQ-regime findings:

  * ANCHOR quota (LCO guard, facts 5/9): the final selection needs an AWQ-measured
    anchor in every budget cell it might pick from — extrapolation-only selection
    fails (LCO ρ −0.44). So guarantee ≥ k_anchor measured archs per occupied budget
    cell before spending the rest of the batch elsewhere.

  * DECISION quota (acquisition, facts 1/10): within a (wbits, eff_kv) cell the real
    problem is ordering near-tied archs. A measurement is worth most when it can flip
    the cell's argmin — i.e. a candidate whose PREDICTED loss is at/below the cell's
    current measured-best. We score by that predicted contention. σ-uncertainty
    weighting is deferred until the Phase-0 draw-noise σ is calibrated (per the
    campaign's k-fold-coverage precondition); until then this is RANK-based only,
    which is the documented fallback and avoids the al_frac failure mode (spending
    budget on an uncalibrated acquisition signal).

The remaining budget is filled by the existing geometry selector (hole-filling
subset_select) so per-axis coverage / edge preservation is unchanged.

Default (both quotas 0) reproduces the geometry-only pick exactly.
"""
import numpy as np


def cell_ids(comp, box_min, box_max, g):
    """Assign each comp row (N, d) to a g^d budget-box cell id (int). Rows are clipped
    into [box_min, box_max]; identical binning to utils.select.even_select."""
    comp = np.asarray(comp, float)
    lo = np.asarray(box_min, float); hi = np.asarray(box_max, float)
    cell = np.clip(((comp - lo) / (hi - lo + 1e-9) * g).astype(int), 0, g - 1)
    cid = np.zeros(len(comp), dtype=np.int64)
    for d in range(comp.shape[1]):
        cid = cid * g + cell[:, d]
    return cid


def anchor_deficit_picks(cand_comp, cand_pred, arch_comp, box_min, box_max, g,
                         k_anchor, budget):
    """Indices into cand_* that fill under-anchored occupied budget cells.

    A cell is 'occupied' if it contains ≥1 candidate. For each occupied cell holding
    fewer than `k_anchor` measured (archive) archs, take its best-predicted candidates
    (lowest cand_pred first) up to the deficit. Cells with the largest deficit are
    served first. Returns ≤ budget unique candidate indices (highest priority — the
    LCO guard runs before any other quota)."""
    if k_anchor <= 0 or budget <= 0 or len(cand_comp) == 0:
        return np.array([], int)
    ccid = cell_ids(cand_comp, box_min, box_max, g)
    acid = (cell_ids(arch_comp, box_min, box_max, g) if len(arch_comp)
            else np.array([], np.int64))
    from collections import Counter
    have = Counter(int(c) for c in acid)
    order = np.argsort(cand_pred, kind='stable')          # best-predicted first
    by_cell = {}
    for i in order:
        by_cell.setdefault(int(ccid[i]), []).append(int(i))
    # deficit per occupied cell (measured count vs k_anchor)
    deficits = {c: max(0, k_anchor - have.get(c, 0)) for c in by_cell}
    picks = []
    # serve largest-deficit cells first; within a cell best-predicted first
    for c in sorted(by_cell, key=lambda c: -deficits[c]):
        if deficits[c] <= 0:
            continue
        for i in by_cell[c][:deficits[c]]:
            picks.append(i)
            if len(picks) >= budget:
                return np.array(picks, int)
    return np.array(picks, int)


def decision_scores(cand_comp, cand_pred, arch_comp, arch_loss, box_min, box_max, g):
    """Rank-based decision value per candidate: how much its measurement could change
    the per-cell argmin. For candidate i in cell c with predicted loss p_i and cell
    current measured-best m_c: score = m_c - p_i (positive ⇒ predicted to BEAT the cell
    best ⇒ a real contender worth measuring; larger ⇒ more decisive). Cells with no
    measured arch yet get m_c = +inf ⇒ every candidate there is a max contender (also
    covered by the anchor quota). Returns (scores, cell_ids)."""
    ccid = cell_ids(cand_comp, box_min, box_max, g)
    best = {}
    if len(arch_comp):
        acid = cell_ids(arch_comp, box_min, box_max, g)
        for c, l in zip(acid, np.asarray(arch_loss, float)):
            c = int(c)
            if c not in best or l < best[c]:
                best[c] = float(l)
    scores = np.array([best.get(int(c), np.inf) - p
                       for c, p in zip(ccid, cand_pred)], float)
    return scores, ccid


def decision_picks(cand_comp, cand_pred, arch_comp, arch_loss, box_min, box_max, g,
                   budget, exclude=()):
    """Top-`budget` candidate indices by decision value, at most one pass per cell in
    round-robin so the quota is spread across contested cells rather than piling into
    one. Excludes indices in `exclude` (already taken by the anchor quota)."""
    if budget <= 0 or len(cand_comp) == 0:
        return np.array([], int)
    scores, ccid = decision_scores(cand_comp, cand_pred, arch_comp, arch_loss,
                                   box_min, box_max, g)
    ex = set(int(i) for i in exclude)
    by_cell = {}
    for i in np.argsort(-scores, kind='stable'):          # most decisive first
        i = int(i)
        if i in ex or not np.isfinite(scores[i]) and scores[i] < 0:
            continue
        by_cell.setdefault(int(ccid[i]), []).append(i)
    # round-robin over contested cells (a cell is contested if its top cand can beat
    # its measured best, i.e. score > 0; inf cells are contested too)
    cells = [c for c in by_cell if by_cell[c] and scores[by_cell[c][0]] > 0]
    picks = []
    while len(picks) < budget and cells:
        for c in list(cells):
            if not by_cell[c] or scores[by_cell[c][0]] <= 0:
                cells.remove(c); continue
            picks.append(by_cell[c].pop(0))
            if len(picks) >= budget:
                break
    return np.array(picks, int)


def anchored_cells(arch_comp, box_min, box_max, g, k_anchor=1):
    """Set of g^d budget-cell ids that hold ≥ k_anchor measured archs (the cells a final
    pick is ALLOWED to come from — LCO guard for predicted/virtual candidates)."""
    if len(arch_comp) == 0:
        return set()
    from collections import Counter
    have = Counter(int(c) for c in cell_ids(arch_comp, box_min, box_max, g))
    return {c for c, n in have.items() if n >= k_anchor}


def mixed_downselect(cand_comp, cand_pred, arch_comp, arch_loss, K,
                     box_min, box_max, geometry_fn,
                     k_anchor=0, anchor_grid=8, decision_frac=0.0, decision_grid=8):
    """Combine anchor (LCO guard) + decision (acquisition) + geometry quotas → K unique
    candidate indices. `geometry_fn(remaining_K, taken_idx)` returns geometry picks
    (the existing subset_select hole-filler) for the leftover budget, excluding
    already-taken indices. Priority: anchor → decision → geometry; any shortfall is
    filled by geometry. Both quotas 0 ⇒ pure geometry (default, unchanged)."""
    N = len(cand_comp)
    if K >= N:
        return np.arange(N)
    taken, taken_set = [], set()

    def _add(idxs):
        for i in idxs:
            i = int(i)
            if i not in taken_set and len(taken) < K:
                taken_set.add(i); taken.append(i)

    if k_anchor > 0:
        _add(anchor_deficit_picks(cand_comp, cand_pred, arch_comp, box_min, box_max,
                                  anchor_grid, k_anchor, budget=K - len(taken)))
    if decision_frac > 0 and len(taken) < K:
        dbudget = min(int(round(decision_frac * K)), K - len(taken))
        _add(decision_picks(cand_comp, cand_pred, arch_comp, arch_loss,
                             box_min, box_max, decision_grid, dbudget, exclude=taken_set))
    if len(taken) < K:
        _add(geometry_fn(K - len(taken), np.array(taken, int)))
    # geometry_fn may under-fill if it can't exclude; top up by best-predicted
    if len(taken) < K:
        for i in np.argsort(cand_pred, kind='stable'):
            _add([int(i)])
            if len(taken) >= K:
                break
    return np.array(taken[:K], int)
