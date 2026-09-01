"""second_search_new.py — 2nd-stage JOINT (W × eff_kvbits) search over the ε-band
product pool. DEFAULT sampler = PSI (Predicted Staircase Improvement): a cost-aware
greedy on ONE scalar — the predicted reduction of the MEASURED archive's record
staircase.

WHAT IS BEING OPTIMIZED
    What a deployer consumes is not the point cloud but the record (attainment)
    function over the budget plane

        R_M(u, v) = min{ y_a : a ∈ M, wbits_a ≤ u, eff_kv_a ≤ v }        (y_ref if empty)

    — the best loss shippable at budget (u, v); non-increasing in both coords. The
    hypervolume this run reports is exactly its integral,
    U(M) = ∫∫ π(u,v)·[y_ref − R_M(u,v)] du dv, with π the budget prior (uniform here;
    a deployment-band weight would live in _psi_grid and nowhere else). So the exact
    marginal value of ONE measurement is

        PSI(c) = ∫∫ π · [R_M(u,v) − μ_c]₊ · 1[(u,v) ≥ (u_c, v_c)] du dv
               = (how far it beats the front) × (how wide a region it beats it over)

    which is simultaneously "beats the previous iteration's Pareto front" (the []₊
    gate) and "fills a gap the front does not cover" (the area factor). No thresholds,
    no windows; μ is the only surrogate output used (never σ).

WHY THIS IS THE RIGHT 2-D GENERALIZATION OF search.py's std-gap
    For a smooth 1-D front f, the staircase regret of points spaced Δ_k is
    ≈ ½ Σ |f'(x_k)| Δ_k², so minimizing it at fixed span gives |f'_k|·Δ_k = const:
    optimal spacing is ∝ 1/|f'| — DENSER where the front is steep, and equal spacing
    (std-gap → 0) is optimal only where the front is locally linear. Greedy PSI IS the
    greedy minimizer of that regret integral, so it is slope-aware by construction
    instead of needing a slope-weighted std-gap bolted on.

WHY THE BATCH FALLS OUT
    Δ(S | M) = U(M ∪ S) − U(M) is the volume of a UNION of staircase regions, i.e. a
    monotone submodular coverage function ⇒ (a) greedy is (1−1/e)-optimal, (b) a batch
    of cells that all improve the SAME rectangle scores once, not |S| times (the
    fantasy/Kriging-Believer update below is what enforces this), and (c) a W row is
    worth the UNION of all its front-beating KV cells, not its best one. With the AWQ
    pool's W-build reuse (utils/awq_pool groups a batch by W allocation: one build,
    many KV swaps) the cost is c(S) = build·|rows(S)| + cell·|S| — a knapsack with
    FAMILY SETUP COSTS, so cost-benefit greedy (PSI / Δcost) opens FEW W rows and gives
    each an EMERGENT, UNEQUAL number of KV cells. Nothing about that is a knob.

PRIORITY ORDER inside a batch (the design's lexicographic rule)
    1. PSI / Δcost                                            — beat the front, fill the gap
    2. intra-batch spread (maximin in the normalised plane)   — tie-break at surrogate resolution
    3. leftover budget → even coverage inside the opened rows — lowest priority
    plus two structural reserves that are NOT priorities and NOT knobs: a per-opened-row KV
    skeleton (_SKELETON = both pool extremes; measured: even sweep + corner probes recover a
    row's curve at 0.92-0.98 vs 0.5, and forcing true ends took starved high-kv RMSE
    0.019 → 0.011) and _GAP_ROWS = 1 row per iteration chosen by pure W-axis maximin
    (anti-lock-in: without it a pessimistic surrogate region is never measured, so never
    corrected). Both have one defensible value, so both are constants.

ORDINAL, ALWAYS
    PSI reads μ ONLY through sign(μ − R): it never uses HOW MUCH lower a
    cell is predicted to be, which is exactly the part a shrunk surrogate gets wrong
    (the measured HQQ→AWQ transfer is rank-preserving with slope 0.705). Formally it is
    invariant to any strictly increasing reparametrisation of the loss scale applied to
    predictions and measurements alike (fit on √y or log y and read back — same batch,
    tests/test_psi_sampler.py). It is NOT invariant to a BIAS between the predictor's
    scale and the measured scale; nothing is, and that is what the measurements correct
    — each one lowers R where the fantasy was optimistic. If the surrogate is
    uninformative the mask saturates and PSI degenerates to cost-aware 2-D gap filling,
    i.e. the pipeline cannot be WORSE than geometry because of a bad predictor. (A cardinal
    variant weighting the area by (R − μ)₊ was implemented and removed: it has neither
    invariance and would need a within-cell ρ this problem does not have — measured 0.215.)

BUDGET SHAPE — three measured facts, two opt-in switches (both default OFF)
    1. RANK CAP (--rank_cap). The (W-block × KV-block) loss table is measured near rank-2:
       on this repo's own stage-2 archive (2608231515, 99 W × 414 KV blocks, 2608 observed
       cells) ALS explains 94.5 / 98.7 / 99.5 / 99.7% of the centred variance at rank
       1/2/3/4 — an additive a(w)+b(kv) surface IS rank 2 — so r* = 2 at _RANK_TOL. A rank-r
       surface is pinned by r+1 swept rows, so builds beyond that cannot reveal a new
       direction of the surface. Each iteration re-estimates r* from the MEASURED table and
       caps builds at r*+1; the cell budget is untouched, so the saved builds become extra
       cells inside the rows that ARE open. The flag also raises the per-row skeleton to
       _RANK_SKELETON shared KV columns, without which the column factors are
       unidentifiable and the estimate can never form. Fails OPEN (no cap) while the table
       is too thin — a cap has to be earned by evidence.
    2. VERIFICATION QUOTA (--verify_frac). Between budget cells the surface is cheap to pin;
       INSIDE one it is not. Measured on the same archive: within-cell loss spread (median
       0.0032) is 0.82× a whole budget-bin step (0.0038), and best-in-cell beats
       mean-in-cell by 0.0042 — more than moving a bin. The surrogate cannot rank inside a
       cell (δ̂ ρ≈0.215), so that comparison must be MEASURED: the quota spends cells on
       direct competitors of the incumbent record points, inside rows already open (free-ish
       KV swaps). This is best-arm identification, not front construction.
    3. SATURATION is REPORTED, never acted on. The staircase utility saturates long before
       the iteration budget (on that same run, 99% of the final U by iteration 4 of 15), so
       U and ΔU/U are logged every iteration against a frozen y_ref and stored in
       iter_<it>.stats — but the run always completes --iterations. Stopping early is the
       operator's call, not the sampler's: an archive truncated by a rule is no longer
       comparable with one that ran its full budget.

BUDGET COST is DERIVED, not configured: see _build_cost(). 0 without the AWQ pool (the
    in-process evaluator rebuilds every arch, so a shared W allocation amortises nothing),
    and an online NNLS fit of this run's own (builds, cells, elapsed) history with it.

KNOBS — budgets (--n_doe / --doe_builds / --iterations / --n_iter / --companion_kv)
    + --seed, plus
    exactly three: --sampler, --verify_frac, --rank_cap. Everything else that used to be a
    flag (psi_mode / psi_grid / gap_rows / row_skeleton / rank_tol / verify_k / greedy_eps /
    stop_du / build_cost) is now a constant at the top of the class or derived from the run,
    because none of them had a value a user could calibrate.
    --companion_kv 0 or 1 = one cell per build (the skeleton switches itself off): the
    no-W-anchoring mode for HQQ, where every arch is rebuilt anyway.

SAMPLERS (--sampler)
    psi      the above (default)
    front    the previous predicted-new-front + W-gap/plane-spread sampler (kept intact)
    product  CONTROL ARM: uniform block-product sampling, cost-matched. This is the arm
             that TIED the whole 2nd-stage machine in the 2608 study — any claimed PSI
             gain must be measured against it, at equal GPU-HOURS (not equal evals: the
             W-build reuse is most of the point).

DOE = W rows EVEN over the wbits range (both box corners included, so R is defined
before any prediction) × (both KV extremes + random KV cells). FINAL = measured top-1
in band; prediction never decides deployment. Stats dump stays post_search-compatible.

Inherits SecondSearch's infrastructure (space/options derivation, evaluator/AWQ pool,
predictor stack, DOE cache, encode cache, stats format); the NSGA candidate machinery
(_next/_nsga/_companion_kv) is unused here.
"""
import os, json, argparse
import numpy as np
from time import time

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from utils.func import set_seed, get_correlation
from utils.metric_specs import protocol_dict
from utils.second_stage import _last_stats, select_eps_band, calc_hv, front_coverage, save_viz
from second_search import SecondSearch, build_parser


class FrontSearch(SecondSearch):
    _CHUNK = 200_000            # pool-prediction cells per predict call (memory guard only)
    _PSI_LEVELS = 64            # μ quantisation levels for the suffix-sum tables
    _PSI_QSAMPLE = 200_000      # subsample size when quantising a huge mask
    # ── settings that are NOT knobs: each has one defensible value, so it is a constant
    # here instead of a flag nobody can calibrate (see the module docstring) ──
    _PSI_GRID = 128             # quadrature grid side for the staircase integral
    _GAP_ROWS = 1               # W-maximin rows forced per iteration (anti-lock-in)
    _SKELETON = 2               # KV cells reserved per opened row = both pool extremes
    _RANK_SKELETON = 6          # ... raised to this under --rank_cap: the shared columns
    _RANK_TOL = 0.05            # the rank estimate needs to be identifiable at all
    _RANK_MAX = 4               # highest rank the ALS fit considers
    _VERIFY_K = 8               # incumbent record points the verify quota is spread over
    _BUILD_COST_BOOT = 5.0      # AWQ pool bootstrap: ~7.7 min build vs ~1.5 min JSD pass

    # ───────────────── pools: FULL ε-band + stage-1 y (overrides SecondSearch) ─────────────────
    def _load_pools(self, args):
        """W/KV block pools = the FULL ε-band of each 1st-stage archive (no div_k pruning:
        there is no crossover to feed — the pool is the space), deduped per block keeping
        the min measured y, box-filtered, sorted by comp ascending. Stage-1 y is kept only
        for the awq branch's plstyp warm-start features."""
        def band(expr, half, eps):
            d = json.load(open(_last_stats(expr)))
            E = d['archive'] + d.get('candidates', [])
            j = np.array([e[1] for e in E], float)
            c = np.array([e[2] for e in E], float)
            idx = select_eps_band(j, c, eps, args.front_eps_rel)
            G = self.ss.encode_batch([E[i][0] for i in idx])
            G = G[:, :self.nw] if half == 'w' else G[:, self.nw:]
            best = {}
            for g, cc, jj in zip(G, c[idx], j[idx]):        # dedup blocks: keep min-y copy
                k = tuple(g.tolist())
                if k not in best or jj < best[k][1]:
                    best[k] = (cc, jj, g)
            cs = np.array([v[0] for v in best.values()])
            ys = np.array([v[1] for v in best.values()])
            Gs = np.stack([v[2] for v in best.values()])
            o = np.argsort(cs)
            return Gs[o], cs[o], ys[o]
        self.Wg, self.w_comp, self.w_y1 = band(args.w_expr, 'w', args.w_front_eps)
        self.KVg, self.kv_comp, self.kv_y1 = band(args.eff_kv_expr, 'kv', args.kv_front_eps)
        n = len(self.comp_obj)
        box_min = list(args.comp_obj_min) if args.comp_obj_min is not None else [None] * n
        box_max = list(args.comp_obj_max) if args.comp_obj_max is not None else [None] * n
        self.comp_obj_min = [box_min[i] if box_min[i] is not None else float(c.min())
                             for i, c in enumerate((self.w_comp, self.kv_comp))]
        self.comp_obj_max = [box_max[i] if box_max[i] is not None else float(c.max())
                             for i, c in enumerate((self.w_comp, self.kv_comp))]
        self.ss.comp_obj_min, self.ss.comp_obj_max = self.comp_obj_min, self.comp_obj_max
        wm = (self.w_comp >= self.comp_obj_min[0]) & (self.w_comp <= self.comp_obj_max[0])
        km = (self.kv_comp >= self.comp_obj_min[1]) & (self.kv_comp <= self.comp_obj_max[1])
        if wm.sum() < 5 or km.sum() < 5:
            raise SystemExit(f"[budget box] too few in-box blocks (W {wm.sum()}, KV {km.sum()})")
        self.Wg, self.w_comp, self.w_y1 = self.Wg[wm], self.w_comp[wm], self.w_y1[wm]
        self.KVg, self.kv_comp, self.kv_y1 = self.KVg[km], self.kv_comp[km], self.kv_y1[km]
        self.accelerator.print(
            f"[pools] ε-band W {len(self.Wg)} blocks [{self.w_comp[0]:.3f},{self.w_comp[-1]:.3f}] | "
            f"KV {len(self.KVg)} blocks [{self.kv_comp[0]:.3f},{self.kv_comp[-1]:.3f}] | "
            f"box W[{self.comp_obj_min[0]:.3f},{self.comp_obj_max[0]:.3f}] "
            f"KV[{self.comp_obj_min[1]:.3f},{self.comp_obj_max[1]:.3f}]")

    def _build_knowledge(self, args):
        """nothing to derive — the pools ARE the space. (The predictor's active column
        set is recomputed data-driven per fit by _fit_predictor; the stage-1-y dicts
        below only feed the awq branch's plstyp warm-start features.)"""
        self.band_table = None
        self.active = np.arange(self.n_var)
        self._y1w = {tuple(g.tolist()): float(y) for g, y in zip(self.Wg, self.w_y1)}
        self._y1k = {tuple(g.tolist()): float(y) for g, y in zip(self.KVg, self.kv_y1)}
        self._y1w_mean, self._y1k_mean = float(self.w_y1.mean()), float(self.kv_y1.mean())
        self.accelerator.print(f"[pool] product cells {len(self.Wg)}×{len(self.KVg)} = "
                               f"{len(self.Wg) * len(self.KVg) / 1e6:.1f}M")

    # ───────────────── predictor input: plstyp + stage-1-y warm-start (awq branch only) ─────────────────
    def _pls_features(self, Xf):
        base = super()._pls_features(Xf)
        X = np.clip(np.round(np.asarray(Xf, float)), 0, self.xu).astype(int)
        yw = np.array([self._y1w.get(tuple(g[:self.nw].tolist()), self._y1w_mean) for g in X])
        yk = np.array([self._y1k.get(tuple(g[self.nw:].tolist()), self._y1k_mean) for g in X])
        return np.column_stack([base, np.sqrt(yw), np.sqrt(yk)])   # sqrt: match the head's scale

    # ───────────────── bookkeeping ─────────────────
    def _key(self, i, j):
        return tuple(np.concatenate([self.Wg[i], self.KVg[j]]).tolist())

    def _register(self, archive):
        """rebuild the measured-genome set + built-W ledger + measured (row, col) cells from
        the archive (resume / DOE-cache safe). Foreign archs simply don't map to a pool cell —
        they still shape the record staircase through their comp coords."""
        if not hasattr(self, '_wi'):
            self._wi = {tuple(g.tolist()): i for i, g in enumerate(self.Wg)}
            self._ki = {tuple(g.tolist()): j for j, g in enumerate(self.KVg)}
        self._measured = set()
        self._built_w = set()
        self._cell_meas = set()
        for g in self._encode_archive(archive):
            self._measured.add(tuple(g.tolist()))
            i = self._wi.get(tuple(g[:self.nw].tolist()))
            j = self._ki.get(tuple(g[self.nw:].tolist()))
            if i is not None:
                self._built_w.add(i)
                if j is not None:
                    self._cell_meas.add((i, j))

    @staticmethod
    def _points(archive):
        return [(float(x[2]), float(x[3]), float(x[1])) for x in archive if np.isfinite(x[1])]

    # ───────────────── 1. predict every pool cell ─────────────────
    def _predict_pool(self, predictor):
        """(nW, nK) predictions, chunked. Assembles ACTIVE columns only (self.active is
        sorted, so [W-active | KV-active] concatenation matches the fit's column order)
        and reuses one KV tile across chunks — the naive full-genome assembly was the
        bottleneck (~2/3 of the pool pass)."""
        nW, nK = len(self.Wg), len(self.KVg)
        aw = self.active[self.active < self.nw]
        ak = self.active[self.active >= self.nw] - self.nw
        WA = self.Wg[:, aw].astype(np.float32)
        KA = self.KVg[:, ak].astype(np.float32)
        P = np.empty((nW, nK), np.float32)
        rows_per = max(1, self._CHUNK // nK)
        tileK = np.tile(KA, (rows_per, 1))
        for s in range(0, nW, rows_per):
            e = min(s + rows_per, nW)
            X = np.concatenate([np.repeat(WA[s:e], nK, 0), tileK[:(e - s) * nK]], 1)
            P[s:e] = np.asarray(predictor.predict(X)).ravel().reshape(e - s, nK)
        return P

    # ═════════════════ PSI: record staircase + improvement integral ═════════════════
    def _psi_grid(self):
        """(gu, gv, π) — the quadrature grid the staircase integral is evaluated on.
        π is the BUDGET PRIOR (uniform: every in-box operating point is equally likely to
        be shipped). A deployment-band weight belongs here and nowhere else: the rest of
        the acquisition is unchanged by it."""
        if getattr(self, '_grid_cache', None) is None:
            G = self._PSI_GRID
            gu = np.linspace(self.comp_obj_min[0], self.comp_obj_max[0], G)
            gv = np.linspace(self.comp_obj_min[1], self.comp_obj_max[1], G)
            self._grid_cache = (gu, gv, np.full((G, G), 1.0 / (G * G)))
        return self._grid_cache

    def _psi_cells(self):
        """per pool row/col: the FIRST grid node whose budget affords it (cached)."""
        if getattr(self, '_cell_idx', None) is None:
            gu, gv, _ = self._psi_grid()
            G = len(gu)
            self._cell_idx = (np.clip(np.searchsorted(gu, self.w_comp, 'left'), 0, G - 1),
                              np.clip(np.searchsorted(gv, self.kv_comp, 'left'), 0, G - 1))
        return self._cell_idx

    def _record_grid(self, archive):
        """R on the quadrature grid: best MEASURED loss at budget ≤ each node, y_ref where
        nothing is feasible yet. Two cumulative mins = the 2-D lower-left envelope."""
        gu, gv, _ = self._psi_grid()
        G = len(gu)
        pts = self._points(archive)
        # y_ref is FROZEN after the DOE (search() sets self._y_ref) so U is comparable
        # across iterations — a growing max would make the utility series meaningless.
        y_ref = getattr(self, '_y_ref', None) or max([p[2] for p in pts], default=1.0)
        M = np.full((G, G), np.inf)
        for w, k, y in pts:
            a = int(np.searchsorted(gu, w, 'left'))
            b = int(np.searchsorted(gv, k, 'left'))
            if a < G and b < G and y < M[a, b]:
                M[a, b] = y
        R = np.minimum.accumulate(np.minimum.accumulate(M, axis=0), axis=1)
        return np.where(np.isfinite(R), R, float(y_ref))

    def _record_pool(self, archive):
        """(nW, nK) EXACT record at each pool cell's own budget coords — what a cell must
        beat to enter the front. One ascending-W sweep with a suffix-min over the KV axis
        (the same sweep _new_front uses, returning values instead of a mask). O(nW·nK)."""
        nW, nK = len(self.Wg), len(self.KVg)
        pts = sorted(self._points(archive))
        rec = np.full(nK, np.inf, np.float32)
        R = np.empty((nW, nK), np.float32)
        p = 0
        for i in range(nW):
            while p < len(pts) and pts[p][0] <= self.w_comp[i]:
                j0 = int(np.searchsorted(self.kv_comp, pts[p][1], side='left'))
                if j0 < nK:
                    np.minimum(rec[j0:], np.float32(pts[p][2]), out=rec[j0:])
                p += 1
            R[i] = rec
        return R

    def _utility(self, archive):
        """U(M) = Σ π·(y_ref − R) over the quadrature grid — the staircase utility the
        whole sampler maximises (= the box-restricted hypervolume). Reported per iteration
        reported every iteration (and stored in iter_<it>.stats) so a saturated run is
        visible: on this codebase's stage-2 runs 99% of the final U arrives by iteration 4
        of 15."""
        _, _, pi = self._psi_grid()
        Rg = self._record_grid(archive)
        y_ref = getattr(self, '_y_ref', None) or float(Rg.max())
        return float(np.sum(pi * (y_ref - Rg)))

    def _surface_rank(self, archive, tol=None, rmax=None):
        """numerical rank of the MEASURED loss surface over the pool grid.

        The (W-block × KV-block) losses form a matrix; measured on this codebase's own
        archive it is essentially rank-2 in √y (top-2 singular energy 99.98%, an additive
        a(w)+b(kv) fit explains 99.8% of the between-cell variance — the ANOVA main-effect
        result seen from the matrix side). A rank-r surface is pinned by r+1 swept rows, so
        opening more builds than that cannot reveal a new direction: the budget belongs on
        cells inside the rows already open. Returns (r*, evr) with evr[r] = fraction of the
        surface variance explained at rank r; r* = smallest r with 1 − evr[r] ≤ tol.

        Fit = ALS on the OBSERVED entries only (the table is very sparse), on rows/cols
        carrying ≥3 measurements so a rank-r row is not fit from r points. `tol`/`rmax`
        default to _RANK_TOL / _RANK_MAX.

        FAILS OPEN: returns (None, []) whenever the archive cannot support the estimate
        (too few usable cells, or fewer than rmax+2 usable rows/cols — a rank read off 3
        rows is bounded by 3 and means nothing). A cap on builds has to be EARNED by
        evidence; the early-run alternative would throttle builds on no data at all.
        `self._rank_diag = (usable_rows, usable_cols, cells)` records why it declined.

        IDENTIFIABILITY — the column side is the binding one, and it is a CONFIGURATION
        requirement, not a data accident: each opened row picks its own KV cells, so two
        rows share a KV block only where the per-row skeleton puts one. Those columns are
        the same indices in every row (that is what makes them a cross-approximation
        SKELETON), so the estimate needs _RANK_SKELETON >= rmax+2 columns — which is why
        --rank_cap raises the skeleton to that automatically. With the plain 2-cell skeleton
        the estimator keeps declining: measured in the 2609 debug runs, where 5-7 rows had
        >=3 observations but only 2 columns did."""
        tol = self._RANK_TOL if tol is None else tol
        rmax = self._RANK_MAX if rmax is None else rmax
        cells = {}
        for g, x in zip(self._encode_archive(archive), archive):
            i = self._wi.get(tuple(np.asarray(g)[:self.nw].tolist()))
            j = self._ki.get(tuple(np.asarray(g)[self.nw:].tolist()))
            if i is not None and j is not None and np.isfinite(x[1]):
                cells[(i, j)] = min(cells.get((i, j), np.inf), float(x[1]))
        self._rank_diag = (0, 0, len(cells))
        if len(cells) < 4 * (rmax + 2):
            return None, []
        ii = np.array([c[0] for c in cells]); jj = np.array([c[1] for c in cells])
        z = np.sqrt(np.array(list(cells.values()), float))          # the head's scale
        ru, rc = np.unique(ii, return_counts=True); cu, cc = np.unique(jj, return_counts=True)
        keep = np.isin(ii, ru[rc >= 3]) & np.isin(jj, cu[cc >= 3])
        ii, jj, z = ii[keep], jj[keep], z[keep]
        self._rank_diag = (len(np.unique(ii)), len(np.unique(jj)), len(z))
        if len(z) < 4 * (rmax + 2) or len(np.unique(ii)) < rmax + 2 \
                or len(np.unique(jj)) < rmax + 2:
            return None, []
        ri = {r: n for n, r in enumerate(np.unique(ii))}
        ci = {c: n for n, c in enumerate(np.unique(jj))}
        a = np.array([ri[r] for r in ii]); b = np.array([ci[c] for c in jj])
        nR, nC = len(ri), len(ci)
        z0 = z - z.mean()
        tot = float((z0 ** 2).sum()) or 1.0
        evr, rng = [], np.random.default_rng(0)
        U = np.zeros((nR, 0)); V = np.zeros((nC, 0))
        for r in range(1, rmax + 1):
            U = np.column_stack([U, rng.normal(0, .1, nR)])
            V = np.column_stack([V, rng.normal(0, .1, nC)])
            for _ in range(30):                                     # ALS on observed entries
                for M, idx, other, oidx, n in ((U, a, V, b, nR), (V, b, U, a, nC)):
                    for t in range(n):
                        m = idx == t
                        if m.sum() >= r:
                            A = other[oidx[m]]
                            M[t] = np.linalg.lstsq(A, z0[m], rcond=None)[0]
            res = float(((z0 - np.einsum('ij,ij->i', U[a], V[b])) ** 2).sum())
            evr.append(max(0.0, 1.0 - res / tot))
        r_star = next((r for r in range(1, rmax + 1) if 1.0 - evr[r - 1] <= tol), rmax)
        return r_star, evr

    def _verify_cells(self, rows, archive, n, planned, meas):
        """--verify_frac quota: cells inside the rows ALREADY OPENED this iteration that sit
        closest to the incumbent record points — direct competitors at the same operating
        budget, i.e. best-arm identification for 'which allocation ships at this budget'.

        Why this is where the money is (measured on this repo's own stage-2 archive): the
        surface between budget cells is nearly rank-2 and cheap to pin, but WITHIN a budget
        cell the loss spread (median 0.0032) is 0.82× a whole budget-bin step (0.0038), and
        picking best-in-cell over average-in-cell is worth 0.0042 — more than moving a bin.
        The surrogate cannot rank inside a cell (measured δ̂ ρ≈0.215), so these have to be
        MEASURED. Cheap here because an opened row's KV swaps need no rebuild."""
        if n <= 0 or not rows:
            return []
        F = np.column_stack([[x[c] for x in archive] for c in (1, 2, 3)])
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        fr = F[nd][:, 1:]
        o = np.argsort(fr[:, 0])
        k = self._VERIFY_K
        anchors = fr[o][np.unique(np.linspace(0, len(o) - 1, min(k, len(o))).round().astype(int))]
        ws = max(self.comp_obj_max[0] - self.comp_obj_min[0], 1e-9)
        ks = max(self.comp_obj_max[1] - self.comp_obj_min[1], 1e-9)
        cand = [(i, j) for i in rows for j in range(len(self.KVg))
                if (i, j) not in planned and not meas[i, j]]
        if not cand:
            return []
        cw = np.array([self.w_comp[i] / ws for i, _ in cand])
        ck = np.array([self.kv_comp[j] / ks for _, j in cand])
        out, taken = [], np.zeros(len(cand), bool)
        for t in range(n):                                          # round-robin over anchors
            aw, ak = anchors[t % len(anchors)] / (ws, ks)
            d = np.where(taken, np.inf, (cw - aw) ** 2 + (ck - ak) ** 2)
            x = int(np.argmin(d))
            if not np.isfinite(d[x]):
                break
            taken[x] = True; out.append(cand[x])
        return out

    def _build_cost(self):
        """cost of OPENING a W row, in units of one cell evaluation — DERIVED, never a flag.

        0 without the AWQ pool: the in-process evaluator rebuilds every arch, so sharing a
        W allocation inside a batch buys nothing and there is nothing to amortise (this is
        structural, not an estimate). With the pool — which groups a batch by W allocation,
        one build then KV swaps — the ratio is fitted online by non-negative least squares
        on THIS run's own (builds, cells, elapsed) history, bootstrapped at
        _BUILD_COST_BOOT and clamped to [0, 50]. Tests / one-off experiments can pin it by
        setting `self._bc_forced`."""
        if getattr(self, '_bc_forced', None) is not None:
            return float(self._bc_forced)
        if getattr(self, 'pool', None) is None:
            return 0.0
        H = getattr(self, '_cost_hist', [])
        if len(H) >= 3:
            try:
                from scipy.optimize import nnls
                coef, _ = nnls(np.array([[h[0], h[1]] for h in H], float),
                               np.array([h[2] for h in H], float))
                if coef[1] > 1e-9:
                    return float(np.clip(coef[0] / coef[1], 0.0, 50.0))
            except Exception:
                pass
        return self._BUILD_COST_BOOT

    def _psi_levels(self, mu):
        """μ quantisation levels for the tables (quantiles of the candidate μ distribution,
        so resolution follows where the candidates actually are). Quantiles come from a
        subsample: a real ε-band pool is 20M+ cells and np.unique on that is a full sort
        for a number we only need as a level count."""
        s = mu if len(mu) <= self._PSI_QSAMPLE else \
            np.random.default_rng(0).choice(mu, self._PSI_QSAMPLE, replace=False)
        L = int(min(self._PSI_LEVELS, max(2, len(np.unique(s)))))
        return np.unique(np.quantile(s, np.linspace(0.0, 1.0, L)))

    @staticmethod
    def _psi_tables(Rg, pi, levels):
        """per-level 2-D SUFFIX sums over the grid:
            SP[l, a, b] = Σ_{u≥a, v≥b} π·1[R > t_l]
        so one candidate's PSI is ONE lookup, SP[l, a, b]. O(L·G²) per greedy step
        (L=64, G=128 → ~1M ops ≈ 15 ms)."""
        I = (Rg[None, :, :] > np.asarray(levels, float)[:, None, None]).astype(np.float64)
        I *= pi[None, :, :]
        return np.cumsum(np.cumsum(I[:, ::-1, ::-1], axis=1), axis=2)[:, ::-1, ::-1]

    @staticmethod
    def _psi_gain(SP, lev, a, b):
        """PSI of each candidate from the table (vectorised, no per-candidate integration)."""
        return SP[lev, a, b]

    @staticmethod
    def _psi_blocks(mi):
        """np.where() returns row-major order, so mi is sorted and each row's masked cells
        are ONE contiguous block: (rows_present, block_starts). Everything per-row (row max,
        per-row best, row expansion) is a slice on these instead of a sort or a ufunc.at over
        20M+ entries."""
        rows = np.unique(mi)
        return rows, np.searchsorted(mi, rows, 'left')

    def _psi_screen(self, mi, g0, m_top, rows=None, starts=None):
        """candidate screen: the globally best m_top masked cells ∪ EVERY row's own best
        cell (so any row can still be opened). Marginal gains only shrink under the fantasy
        updates (submodularity), so a cell whose INITIAL gain is below the current best can
        never be the greedy's argmax — this is the lazy-greedy bound, not an arbitrary cut.
        Opened rows are expanded to all their masked cells inside the greedy.
        O(N): argpartition for the global top, contiguous-block argmax for the per-row best
        (a full argsort/lexsort of a 20M-cell mask costs seconds and buys nothing)."""
        if rows is None:
            rows, starts = self._psi_blocks(mi)
        m_top = int(min(max(1, m_top), len(g0)))
        top = (np.argpartition(-g0, m_top - 1)[:m_top] if m_top < len(g0)
               else np.arange(len(g0)))
        ends = np.append(starts[1:], len(g0))
        row_best = np.array([a + int(np.argmax(g0[a:b])) for a, b in zip(starts, ends)], int)
        return np.union1d(top, row_best)

    def _skeleton_cols(self, nK):
        """the per-opened-row reserve: both KV pool extremes (+ evenly spaced interior
        points under --rank_cap). Protects the row's curve / the surrogate's KV
        extrapolation; the front-chasing budget is spent by PSI on top of it.

        SIZE IS DERIVED, not a flag: 2 (both extremes) normally, _RANK_SKELETON under
        --rank_cap because the rank estimate needs that many KV columns measured across
        several rows to be identifiable at all. Disabled when --companion_kv <= that: with
        a per-row budget that small the reserve would BE the whole batch and PSI would
        never choose anything (--companion_kv 0 or 1 = 'one cell per build', the
        no-W-anchoring mode an HQQ run wants, where every arch costs the same)."""
        s = self._RANK_SKELETON if getattr(self.args, 'rank_cap', False) else self._SKELETON
        if int(getattr(self.args, 'companion_kv', 0)) <= s:
            return []
        if s <= 0 or nK == 0:
            return []
        if s == 1:
            return [0]
        return sorted({int(round(x)) for x in np.linspace(0, nK - 1, min(s, nK))})

    # ───────────────── PSI batch: cost-aware greedy ─────────────────
    def _next_psi(self, P, archive, max_rows=None):
        """-> (plan [(row, col), ...], mask, info). One cost-benefit greedy pass:

            pick argmax PSI(c | M ∪ S) / Δcost(c),   Δcost = build+1 for a new W row, 1 otherwise
            then R ← min(R, μ_c) on c's quadrant     (Kriging-Believer fantasy = the submodular
                                                      union; kills intra-batch redundancy)

        Rows are capped at --n_iter and cells at --n_iter × --companion_kv, so the batch's
        wall clock stays predictable; WITHIN that, per-row KV counts are emergent and
        deliberately unequal."""
        nW, nK = P.shape
        _, _, pi = self._psi_grid()
        iu, iv = self._psi_cells()
        cell_budget = max(1, int(self.n_iter) * max(1, int(self.args.companion_kv)))
        # the row cap is the RANK CAP when search() passes one (r*+1 swept rows pin a
        # rank-r surface); saved builds are not lost — the cell budget is unchanged, so
        # they turn into more cells inside the rows that ARE open.
        max_rows = max(1, int(self.n_iter) if max_rows is None else int(max_rows))
        build_cost = self._build_cost()
        verify_n = int(round(max(0.0, min(0.5, float(getattr(self.args, 'verify_frac', 0.0))))
                             * cell_budget))
        ws = max(self.comp_obj_max[0] - self.comp_obj_min[0], 1e-9)
        ks = max(self.comp_obj_max[1] - self.comp_obj_min[1], 1e-9)

        meas = np.zeros((nW, nK), bool)
        if self._cell_meas:
            mij = np.array(sorted(self._cell_meas), int)
            meas[mij[:, 0], mij[:, 1]] = True
        mask = (P < self._record_pool(archive)) & ~meas          # "can beat the previous front"
        info = {'front_cells': int(mask.sum()), 'front_rows': int(mask.any(1).sum()),
                'rows': [], 'row_cells': [], 'skeleton': 0, 'fill': 0, 'verify': 0,
                'psi': 0, 'psi_gain': 0.0, 'max_rows': int(max_rows)}
        mi, mj = (a.astype(np.int32) for a in np.where(mask))   # int32: a 20M-cell mask
        if len(mi) == 0:                                        # costs 320MB as int64
            return [], mask, info

        mu = P[mi, mj].astype(np.float64)
        levels = self._psi_levels(mu)
        lev = np.clip(np.searchsorted(levels, mu, 'left'), 0, len(levels) - 1)
        Rg = self._record_grid(archive)
        SP = self._psi_tables(Rg, pi, levels)
        g0 = self._psi_gain(SP, lev, iu[mi], iv[mj])
        rows_p, b_lo = self._psi_blocks(mi)
        b_hi = np.append(b_lo[1:], len(mi))
        row_gain = np.zeros(nW)
        row_gain[rows_p] = np.maximum.reduceat(g0, b_lo)     # blocks are contiguous
        row_best_idx = {int(r): int(a + np.argmax(g0[a:b]))
                        for r, a, b in zip(rows_p, b_lo, b_hi)}
        keep = self._psi_screen(mi, g0, max(2000, 20 * cell_budget), rows_p, b_lo)
        ci, cj = mi[keep].copy(), mj[keep].copy()
        cmu, clev = mu[keep].copy(), lev[keep].copy()
        taken = np.zeros(len(ci), bool)
        dmin = np.full(len(ci), np.inf)                          # dist² to nearest PLANNED cell
        pos = {(int(a), int(b)): n for n, (a, b) in enumerate(zip(ci, cj))}

        plan, planned, rows = [], set(), []
        state = {'psi': 0, 'skeleton': 0, 'fill': 0, 'verify': 0, 'psi_gain': 0.0}

        rc = {}                                                  # cells placed per opened row

        def place(i, j, kind='psi'):
            if len(plan) >= cell_budget or (i, j) in planned or meas[i, j]:
                return False
            plan.append((i, j)); planned.add((i, j))
            rc[i] = rc.get(i, 0) + 1
            state[kind] = state.get(kind, 0) + 1
            n = pos.get((i, j))
            if n is not None:
                taken[n] = True
            Rg[iu[i]:, iv[j]:] = np.minimum(Rg[iu[i]:, iv[j]:], float(P[i, j]))   # fantasy
            np.minimum(dmin, ((self.w_comp[ci] - self.w_comp[i]) / ws) ** 2
                       + ((self.kv_comp[cj] - self.kv_comp[j]) / ks) ** 2, out=dmin)
            return True

        def expand(i):
            """opening a row makes ALL its front-beating cells candidates (a row is worth the
            union of its cells, and cells 2..q of an open row cost only `cell`)."""
            nonlocal ci, cj, cmu, clev, taken, dmin
            lo = int(np.searchsorted(mi, i, 'left')); hi = int(np.searchsorted(mi, i, 'right'))
            new = np.array([n for n in range(lo, hi)
                            if (int(mi[n]), int(mj[n])) not in pos], int)
            if not len(new):
                return
            base = len(ci)
            ci = np.concatenate([ci, mi[new]]); cj = np.concatenate([cj, mj[new]])
            cmu = np.concatenate([cmu, mu[new]]); clev = np.concatenate([clev, lev[new]])
            taken = np.concatenate([taken, np.zeros(len(new), bool)])
            d = np.full(len(new), np.inf)
            for (a, b) in plan:
                np.minimum(d, ((self.w_comp[mi[new]] - self.w_comp[a]) / ws) ** 2
                           + ((self.kv_comp[mj[new]] - self.kv_comp[b]) / ks) ** 2, out=d)
            dmin = np.concatenate([dmin, d])
            for n, (a, b) in enumerate(zip(mi[new], mj[new])):    # index the NEW rows only —
                key = (int(a), int(b))                               # the existing candidates
                pos[key] = base + n                                  # already carry their state
                if key in planned:
                    taken[base + n] = True

        def open_row(i):
            """open a W row = pay a build. A build that measures NOTHING is not a build: if
            the skeleton is off (or its cells are already measured) the row still takes its
            own best masked cell, and if even that fails the row is un-opened so it neither
            consumes a --n_iter build slot nor shows up in the cost report."""
            i = int(i)
            rows.append(i)
            expand(i)
            for j in self._skeleton_cols(nK):
                place(i, int(j), 'skeleton')
            if not rc.get(i):
                n = row_best_idx.get(i)
                if n is not None:
                    place(i, int(mj[n]), 'psi')
            if not rc.get(i):
                rows.pop()

        # _GAP_ROWS: W-axis maximin rows opened BEFORE the greedy (anti-lock-in — the
        # surrogate cannot starve a wbits region of measurements and thus of correction).
        avail = rows_p
        for _ in range(min(self._GAP_ROWS, max_rows)):
            cand = [int(r) for r in avail if r not in rows]
            if not cand:
                break
            U = sorted({float(self.w_comp[i]) for i in self._built_w}
                       | {float(self.w_comp[i]) for i in rows})
            wc = self.w_comp[cand]
            d = (np.min(np.abs(wc[:, None] - np.asarray(U)[None, :]), axis=1)
                 if U else np.full(len(cand), np.inf))
            near = [c for c, dd in zip(cand, d) if dd >= d.max() - 1e-12]
            open_row(max(near, key=lambda r: row_gain[r]))           # tie-break: richest row

        # main cost-benefit greedy (stops early by verify_n so the verification quota,
        # which the surrogate cannot rank, is not eaten by front-chasing cells)
        while len(plan) < cell_budget - verify_n and len(ci):
            SP = self._psi_tables(Rg, pi, levels)
            g = self._psi_gain(SP, clev, iu[ci], iv[cj])
            g = np.where(taken, -1.0, g)
            is_open = np.isin(ci, np.array(rows, int)) if rows else np.zeros(len(ci), bool)
            if len(rows) >= max_rows:
                g = np.where(is_open, g, -1.0)                       # no build budget left
            score = g / np.where(is_open, 1.0, build_cost + 1.0)
            best = float(score.max()) if len(score) else -1.0
            if not np.isfinite(best) or best <= 0:
                break
            near = np.where(score >= best * (1.0 - 1e-9))[0]
            n = int(near[int(np.argmax(dmin[near]))])                # tie-break: intra-batch spread
            i, j = int(ci[n]), int(cj[n])
            state['psi_gain'] += float(g[n])   # accumulated PSI (the COUNT is state['psi'])
            if i not in rows:
                open_row(i)
            place(i, j, 'psi')

        # --verify_frac: within-cell best-arm identification at the incumbent budgets
        if verify_n > 0 and rows:
            for vi, vj in self._verify_cells(rows, archive, verify_n, planned, meas):
                place(int(vi), int(vj), 'verify')

        # leftover budget → lowest-priority objective: even coverage inside the opened rows
        if len(plan) < cell_budget and rows:
            fc = [(i, j) for i in rows for j in range(nK)
                  if (i, j) not in planned and not meas[i, j]]
            if fc:
                fw = np.array([self.w_comp[i] / ws for i, _ in fc])
                fk = np.array([self.kv_comp[j] / ks for _, j in fc])
                d = np.full(len(fc), np.inf)
                for (a, b) in plan:
                    np.minimum(d, (fw - self.w_comp[a] / ws) ** 2
                               + (fk - self.kv_comp[b] / ks) ** 2, out=d)
                while len(plan) < cell_budget and np.isfinite(d).any() and d.max() >= 0:
                    x = int(np.argmax(d))
                    if d[x] < 0:
                        break
                    i, j = fc[x]
                    d[x] = -1.0
                    if place(i, j, 'fill'):
                        np.minimum(d, (fw - fw[x]) ** 2 + (fk - fk[x]) ** 2, out=d)
                        d[x] = -1.0

        info['rows'] = [int(r) for r in rows]
        info['row_cells'] = [int(rc.get(r, 0)) for r in rows]
        info['skeleton'], info['fill'] = state['skeleton'], state['fill']
        info['verify'] = state['verify']
        info['max_rows'] = int(max_rows)
        info['psi'] = int(state['psi'])            # cells the greedy chose on PSI value
        info['psi_gain'] = float(state['psi_gain'])   # their accumulated predicted improvement
        return plan, mask, info

    # ───────────────── control arm: uniform block-product sampling ─────────────────
    def _next_product(self, P, archive, it):
        """--sampler product: uniform sampling of the ε-band block product, COST-MATCHED to
        psi (n_iter random W rows × companion_kv random KV cells each, so the same number of
        builds and evals). This is the arm that tied the whole 2nd-stage machine in the 2608
        study — it stays runnable so a PSI gain is a measured gain, not an assumed one."""
        nW, nK = P.shape
        rng = np.random.default_rng(int(self.args.seed) + 9973 * int(it))
        q = max(1, int(self.args.companion_kv))
        rows = rng.choice(nW, size=min(int(self.n_iter), nW), replace=False)
        budget = min(int(self.n_iter), nW) * q
        # per-row draw, then top up the shortfall from already-drawn ROWS (never new builds)
        # so the arm spends the same builds AND the same evals as psi — a control that
        # under-spends its budget is not a control.
        order = {int(r): list(rng.permutation(nK)) for r in rows}
        plan, planned = [], set()

        def draw(r, k):
            got = 0
            while got < k and order[r] and len(plan) < budget:
                cell = (r, int(order[r].pop()))
                if cell not in planned and cell not in self._cell_meas:
                    planned.add(cell); plan.append(cell); got += 1
            return got

        for r in rows:
            draw(int(r), q)
        while len(plan) < budget and any(order[int(r)] for r in rows):
            for r in rows:
                if len(plan) >= budget:
                    break
                draw(int(r), 1)
        info = {'front_cells': 0, 'front_rows': 0, 'rows': [int(r) for r in rows],
                'row_cells': [int(sum(1 for a, _ in plan if a == int(r))) for r in rows],
                'skeleton': 0, 'fill': 0, 'verify': 0, 'psi': 0, 'psi_gain': 0.0}
        return plan, np.zeros((nW, nK), bool), info

    # ═════════ legacy sampler (--sampler front): predicted-new-front + spread ═════════
    def _new_front(self, P, archive):
        """boolean (nW, nK) mask of cells on the predicted new front: μ strictly below
        every measured point and every other predicted cell with (w' ≤ w, k' ≤ k).
        One ascending-W sweep; `rec`/`run` are prefix-min arrays over the KV axis
        (archive / already-swept predicted rows), `own_prev` covers this row's own
        cheaper cells. O(nW·nK)."""
        nW, nK = P.shape
        pts = sorted(self._points(archive))
        pw = np.array([p[0] for p in pts]) if pts else np.empty(0)
        rec = np.full(nK, np.inf, np.float32)
        run = np.full(nK, np.inf, np.float32)
        mask = np.zeros((nW, nK), bool)
        pi = 0
        for i in range(nW):
            while pi < len(pts) and pw[pi] <= self.w_comp[i]:
                j0 = int(np.searchsorted(self.kv_comp, pts[pi][1], side='left'))
                if j0 < nK and pts[pi][2] < rec[-1]:
                    np.minimum(rec[j0:], np.float32(pts[pi][2]), out=rec[j0:])
                pi += 1
            row = P[i]
            own = np.minimum.accumulate(row)
            own_prev = np.concatenate([[np.float32(np.inf)], own[:-1]])
            mask[i] = row < np.minimum(np.minimum(rec, run), own_prev)
            np.minimum(run, own, out=run)
        return mask

    def _pick_rows(self, mask, P, B):
        """B rows among front-holding rows, each greedily filling the largest W-axis gap
        of {built W coords} ∪ picks (maximin distance-to-covered). Rows within one pool
        W-spacing of the best fill are geometrically indistinguishable at pool resolution
        → tie-break to the lower mean predicted curve (ordinal prediction use)."""
        cand = [int(i) for i in np.where(mask.any(axis=1))[0]]
        if not cand:
            return []
        U = sorted({float(self.w_comp[i]) for i in self._built_w})
        uw = np.unique(self.w_comp)
        w_res = float(np.median(np.diff(uw))) if len(uw) > 1 else 0.0
        picks = []
        for _ in range(min(B, len(cand))):
            wc = self.w_comp[cand]
            d = (np.min(np.abs(wc[:, None] - np.asarray(U)[None, :]), axis=1)
                 if U else np.full(len(cand), np.inf))
            near = [c for c, dd in zip(cand, d) if dd >= d.max() - w_res]
            i = min(near, key=lambda r: float(P[r].mean()))
            picks.append(i)
            U.append(float(self.w_comp[i]))
            cand.remove(i)
        return picks

    def _place_cells(self, picks, mask, archive, budget):
        """[(row, col), ...]: both pool-extreme KV cells per picked row (always), then the
        remaining budget on the picked rows' front cells, greedily at the largest plane
        gap of {measured Pareto front coords} ∪ placed cells (coords normalized by the
        budget box). Per-row counts emerge here. If the rows hold fewer front cells than
        the budget, the batch is simply smaller (no filler — logged by the caller)."""
        nK = len(self.KVg)
        plan, planned = [], set()

        def add(i, j):
            if (i, j) in planned or self._key(i, j) in self._measured:
                return
            plan.append((i, j)); planned.add((i, j))

        for i in picks:
            add(i, 0); add(i, nK - 1)
        F = np.column_stack([[x[c] for x in archive] for c in (1, 2, 3)])
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        ws = max(self.comp_obj_max[0] - self.comp_obj_min[0], 1e-9)
        ks = max(self.comp_obj_max[1] - self.comp_obj_min[1], 1e-9)
        bw = np.concatenate([F[nd, 1] / ws, [self.w_comp[i] / ws for i, _ in plan]])
        bk = np.concatenate([F[nd, 2] / ks, [self.kv_comp[j] / ks for _, j in plan]])
        cand = [(i, int(j)) for i in picks for j in np.where(mask[i])[0]
                if (i, int(j)) not in planned and self._key(i, int(j)) not in self._measured]
        if cand and len(plan) < budget:
            cw = np.array([self.w_comp[i] / ws for i, _ in cand])
            ck = np.array([self.kv_comp[j] / ks for _, j in cand])
            d = np.min((cw[:, None] - bw[None, :]) ** 2 + (ck[:, None] - bk[None, :]) ** 2, axis=1)
            taken = np.zeros(len(cand), bool)
            for _ in range(min(budget - len(plan), len(cand))):
                x = int(np.argmax(np.where(taken, -1.0, d)))
                i, j = cand[x]
                plan.append((i, j)); taken[x] = True
                np.minimum(d, (cw - cw[x]) ** 2 + (ck - ck[x]) ** 2, out=d)
        return plan

    def _next_front(self, P, archive):
        mask = self._new_front(P, archive)
        picks = self._pick_rows(mask, P, self.n_iter)
        budget = len(picks) * int(self.args.companion_kv)
        plan = self._place_cells(picks, mask, archive, budget) if picks else []
        info = {'front_cells': int(mask.sum()), 'front_rows': int(mask.any(1).sum()),
                'rows': [int(i) for i in picks],
                'row_cells': [int(sum(1 for a, _ in plan if a == int(i))) for i in picks],
                'skeleton': 0, 'fill': 0, 'verify': 0, 'psi': 0, 'psi_gain': 0.0}
        return plan, mask, info

    # ───────────────── DOE ─────────────────
    def _doe_cells(self):
        """W rows EVEN over the wbits range (linspace targets → nearest distinct pool rows,
        so BOTH box corners are in the design and the record staircase is defined over the
        whole box before any prediction) × (both KV extremes + random interior cells).

        SIZE: --n_doe is the TOTAL number of DOE archs and --doe_builds is how many W rows
        to spread them over, so cells-per-row = n_doe / doe_builds (at least 2, so both KV
        extremes always fit). The ITERATION budget (--companion_kv) does not govern the
        DOE."""
        rng = np.random.default_rng(self.args.seed)
        nW, nK = len(self.Wg), len(self.KVg)
        nrows = min(max(int(self.args.doe_builds), 1), nW)
        rows, wc = [], np.asarray(self.w_comp, float)
        for t in np.linspace(wc.min(), wc.max(), nrows):
            order = np.argsort(np.abs(wc - t))
            for r in order:
                if int(r) not in rows:
                    rows.append(int(r)); break
        q = int(np.clip(round(self.n_doe / max(nrows, 1)), 2, nK))
        out = []
        for i in rows:
            cells = [0, nK - 1]
            if q > 2 and nK > 2:
                cells += [int(x) for x in rng.choice(np.arange(1, nK - 1),
                                                     size=min(q - 2, nK - 2), replace=False)]
            out += [(int(i), int(j)) for j in cells]
        return rows, out

    # ───────────────── debug figure ─────────────────
    def _save_viz_nd(self, it, archive, plan, mask, n_doe):
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        except Exception as e:
            self.accelerator.print(f"[viz] nd panel skipped ({e})"); return
        w = np.array([r[2] for r in archive]); kv = np.array([r[3] for r in archive])
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
        ax = axes[0]
        ax.scatter(w[:n_doe], kv[:n_doe], s=6, color='0.8', label=f'DOE ({n_doe})')
        if len(archive) > n_doe:
            ax.scatter(w[n_doe:], kv[n_doe:], s=8, c=np.arange(len(archive) - n_doe),
                       cmap='viridis', label='iteration cells')
        ax.set_xlabel('wbits'); ax.set_ylabel('eff. kvbits')
        ax.set_title(f'measured cells (n={len(archive)})'); ax.legend(fontsize=8)
        ax = axes[1]
        fi, fj = np.where(mask)
        sub = np.random.default_rng(0).choice(len(fi), min(4000, len(fi)), replace=False) \
            if len(fi) else np.empty(0, int)
        ax.scatter(self.w_comp[fi[sub]], self.kv_comp[fj[sub]], s=3, color='0.75',
                   label=f'front-beating cells ({len(fi)})')
        if plan:
            ax.scatter([self.w_comp[i] for i, _ in plan], [self.kv_comp[j] for _, j in plan],
                       s=18, marker='x', color='crimson', label=f'batch ({len(plan)})')
        ax.set_xlabel('wbits'); ax.set_title(f'iter {it}: front → batch'); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_path, f'iter_{it}_nd.png'), dpi=120,
                    bbox_inches='tight')
        plt.close(fig)

    # ───────────────── main loop ─────────────────
    def search(self):
        acc = self.accelerator; main = acc.is_main_process
        sampler = getattr(self.args, 'sampler', 'psi')
        t0 = time(); start_it = 1
        self._n_doe_archs = 0
        if self.args.resume:
            rf = json.load(open(self.args.resume))
            archive = rf['archive']; start_it = rf['iteration'] + 1
            self._n_doe_archs = rf.get('n_doe_archs', 0)
            if main:
                acc.print(f"[resume] {len(archive)} archs from iter {rf['iteration']}")
        else:
            # DOE: even-W rows × (both KV extremes + random cells) — predictor-free.
            if main:
                seeded = self._load_doe([]) if self.args.doe_results else []
                self._register(seeded)
                rows, cells = self._doe_cells()
                archs, planned = [], set(self._measured)
                for i, j in cells:
                    k = self._key(i, j)
                    if k not in planned:
                        planned.add(k)
                        archs.append(self.ss.decode(np.array(k, int)))
                acc.print(f"[DOE] {len(rows)} even-W builds × "
                          f"{max(2, round(self.n_doe / max(len(rows), 1)))} KV cells "
                          f"(2 extremes + rest random) → {len(archs)} archs "
                          f"(--n_doe {self.n_doe} over --doe_builds {len(rows)})" + (f" (+{len(seeded)} cached)" if seeded else ""))
            else:
                archs, seeded = [], []
            archs = acc.gather_for_metrics(archs, use_gather_object=True)
            acc.wait_for_everyone()
            t_eval = time()
            kept, metric, comp = self._evaluate(archs) if archs else ([], [], [])
            if main:
                self._cost_hist = [(len({tuple(self.ss.encode(a)[:self.nw]) for a in archs}),
                                    len(archs), time() - t_eval)] if archs else []
                measured = [[archs[i], m, *c] for i, m, c in zip(kept, metric, comp)]
                if kept:
                    self._cache_doe([archs[i] for i in kept], metric)
                archive = seeded + measured
                self._n_doe_archs = len(archive)
                losses = [x[1] for x in archive]
                acc.print(f"[DOE] archive {len(archive)}  loss {min(losses):.4f}-{max(losses):.4f} "
                          f"({time() - t0:.1f}s)")
            else:
                archive = []
        if main:
            self._register(archive)
            ref_pt = np.array([np.max([x[i] for x in archive])
                               for i in range(1, len(self.comp_obj) + 2)])
            # FREEZE the utility reference on the DOE (or the resumed archive) so the U
            # series is comparable across iterations (a growing max would break it).
            self._y_ref = float(np.nanmax([x[1] for x in archive if np.isfinite(x[1])]))
            U_prev = self._utility(archive)
            acc.print(f"[utility] y_ref {self._y_ref:.4f}  U(DOE) = {U_prev:.4f}")
            acc.print(f"[sampler] {sampler}"
                      + (f" | build_cost={self._build_cost():.2f} (derived), "
                         f"rank_cap={'on' if self.args.rank_cap else 'off'}, "
                         f"verify_frac={self.args.verify_frac}"
                         if sampler == 'psi' else ""))
            if (self.args.rank_cap and sampler == 'psi'
                    and int(self.args.companion_kv) < 3 * self._RANK_SKELETON):
                acc.print(f"[rank] WARNING --rank_cap reserves {self._RANK_SKELETON} shared KV "
                          f"columns per opened row for identifiability, which is most of a "
                          f"--companion_kv {self.args.companion_kv} per-row budget; give it "
                          f">= {3 * self._RANK_SKELETON} or leave --rank_cap off")
        acc.wait_for_everyone()

        for it in range(start_it, self.iterations + 1):
            iter_start = time()
            if main:
                tp = time()
                pred, a_pred = self._fit_predictor(archive)
                P = self._predict_pool(pred)
                predictor_time = time() - tp
                tn = time()
                cap = None
                if sampler == 'psi' and getattr(self.args, 'rank_cap', False):
                    r_star, evr = self._surface_rank(archive)
                    if r_star is None:
                        nr, nc, ncell = getattr(self, '_rank_diag', (0, 0, 0))
                        hint = ("; --rank_cap already reserves a shared KV column skeleton, "
                                "so this should resolve once a few rows have been built"
                                if nc < self._RANK_SKELETON else "")
                        acc.print(f"[rank] cannot estimate the surface rank yet "
                                  f"({ncell} usable cells over {nr} rows x {nc} cols, need "
                                  f">=6 of each) — no build cap this iteration (fail-open)"
                                  + hint)
                    else:
                        cap = min(int(self.n_iter), r_star + 1)
                        acc.print(f"[rank] measured surface rank r*={r_star} "
                                  f"(explained {[round(e, 4) for e in evr]}) → builds capped "
                                  f"at r*+1 = {cap}"
                                  + (" (cell budget unchanged: the saved builds become cells "
                                     "inside the open rows)" if cap < int(self.n_iter)
                                     else " — not binding (>= --n_iter)"))
                if sampler == 'psi':
                    plan, mask, binfo = self._next_psi(P, archive, max_rows=cap)
                elif sampler == 'front':
                    plan, mask, binfo = self._next_front(P, archive)
                else:
                    plan, mask, binfo = self._next_product(P, archive, it)
                next_time = time() - tn
                cands = [self.ss.decode(np.array(self._key(i, j), int)) for i, j in plan]
                c_pred = np.array([float(P[i, j]) for i, j in plan])
                pts_before = self._points(archive)
                budget = int(self.n_iter) * max(1, int(self.args.companion_kv))
                acc.print(f"[iter {it}] front-beating {binfo['front_cells']} cells in "
                          f"{binfo['front_rows']} rows | builds "
                          f"{[f'{self.w_comp[i]:.2f}' for i in binfo['rows']]} "
                          f"cells/build {binfo['row_cells']} → {len(plan)} cells "
                          f"(psi {binfo['psi']} [gain {binfo.get('psi_gain', 0.0):.4f}], "
                          f"skeleton {binfo['skeleton']}, verify {binfo['verify']}, "
                          f"fill {binfo['fill']})"
                          + (f" [short of {budget}]" if len(plan) < budget else ""))
            else:
                cands = []
            acc.wait_for_everyone()
            cands = acc.gather_for_metrics(cands, use_gather_object=True)
            if not cands:
                if main:
                    acc.print(f"Iter {it}: no unmeasured front-beating cells; stop")
                break
            t_eval = time()
            kept, c_metric, c_comp = self._evaluate(cands)
            if main:
                # one (builds, cells, elapsed) row per batch -> the derived build cost
                self._cost_hist = getattr(self, '_cost_hist', []) + [
                    (max(1, len(binfo['rows'])), len(cands), time() - t_eval)]
                rmse, rho, tau = get_correlation(
                    np.concatenate([np.asarray(a_pred).ravel(), c_pred[kept]]),
                    np.array([x[1] for x in archive] + c_metric))
                for i, m, c in zip(kept, c_metric, c_comp):
                    archive.append([cands[i], m, *c])
                self._register(archive)
                # front hit-rate: measured batch cells that strictly entered the measured
                # front (the predicted-vs-realized diagnostic; purely informational)
                hits = 0
                if pts_before and kept:
                    pw = np.array([p[0] for p in pts_before])
                    pk = np.array([p[1] for p in pts_before])
                    pv = np.array([p[2] for p in pts_before])
                    for n, x in enumerate(kept):
                        i, j = plan[x]
                        msk = (pw <= self.w_comp[i]) & (pk <= self.kv_comp[j])
                        m0 = pv[msk].min() if msk.any() else np.inf
                        hits += bool(c_metric[n] < m0)
                F = np.column_stack([[x[i] for x in archive]
                                     for i in range(1, len(self.comp_obj) + 2)])
                hv = calc_hv(ref_pt, F); cov = front_coverage(archive, self.comp_obj)
                U = self._utility(archive); dU = U - U_prev; U_prev = U
                rel = dU / U if U > 0 else 0.0
                acc.print(f"[utility] U = {U:.4f}  ΔU = {dU:+.4f}  ({rel * 100:+.2f}%)"
                          f"  cost = {len(binfo['rows'])}build+{len(kept)}cell")
                acc.print(f"Iter {it}: hv = {hv:.2f}, iter time : {time() - iter_start:.2f}s, "
                          f"predictor_time : {predictor_time:.2f}, next_time : {next_time:.2f}")
                acc.print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, "
                          f"Kendall's Tau = {tau:.4f}")
                acc.print(f"[front] hit-rate {hits}/{len(kept)} measured cells entered the front")
                for obj in self.comp_obj:
                    c = cov[obj]
                    acc.print(f"  {obj} front-coverage : {c['coverage'] * 100:.1f}%  "
                              f"front=[{c['front_min']:.3f}, {c['front_max']:.3f}] / "
                              f"full=[{c['full_min']:.3f}, {c['full_max']:.3f}]")
                if it % self.save_iter == 0 or it == self.iterations:
                    os.makedirs(self.save_path, exist_ok=True)
                    with open(os.path.join(self.save_path, f"iter_{it}.stats"), 'w') as f:
                        json.dump({'archive': archive, 'candidates': archive[-len(kept):],
                                   'hv': hv, 'n_doe_archs': self._n_doe_archs,
                                   'surrogate': {'model': self.predictor, 'rmse': rmse,
                                                 'rho': rho, 'tau': tau},
                                   'utility': {'U': U, 'dU': dU, 'y_ref': self._y_ref},
                                   'ndfront': {**binfo, 'sampler': sampler,
                                               'builds': binfo['rows'],
                                               'build_wbits': [float(self.w_comp[i])
                                                               for i in binfo['rows']],
                                               'planned': len(plan), 'kept': len(kept),
                                               'hits': int(hits)},
                                   'coverage': cov, 'protocol': protocol_dict(self.args),
                                   'iteration': it}, f)
                    if self.debug:
                        save_viz(self.save_path, it, archive, c_metric,
                                 np.asarray(c_pred).ravel()[kept], c_comp, cov,
                                 self.comp_obj, self.comp_obj_min, self.comp_obj_max)
                        self._save_viz_nd(it, archive, plan, mask, self._n_doe_archs)
            acc.wait_for_everyone()

        if self.pool is not None:
            self.pool.close()
        if main:
            acc.print(f"[done] {len(archive)} archs, {time() - t0:.1f}s → {self.save_path}")
            self._write_results(archive, time() - t0)
        return archive


# flags second_search's parser defines for the NSGA candidate machinery (_next / _nsga /
# _downselect / _companion_kv / _initialize) and the pool/knowledge builders FrontSearch
# OVERRIDES — none of them is read on this entry point, so advertising them in --help is
# just noise that invites tuning something inert. Hidden from help, still present in
# vars(args) (protocol_dict and the inherited __init__ read the namespace), and a warning
# fires if one is actually set.
_DEAD_FLAGS = ('agree_frac', 'div_k', 'pop', 'mut_p_val', 'mut_p_mod', 'l0_repair',
               'pool_escape', 'grid_seed', 'cand_grid', 'subset_pop_size', 'anchor_k',
               'anchor_grid', 'decision_frac', 'decision_grid', 'companion_method',
               'companion_true_ends', 'companion_front_ends', 'companion_gap')


def warn_dead_flags(args, parser, log=print):
    """report any inherited NSGA-machinery flag the user set that this entry point ignores."""
    d = {a.dest: a.default for a in parser._actions}
    set_ = [f for f in _DEAD_FLAGS if f in d and getattr(args, f, d[f]) != d[f]]
    if set_:
        log(f"[args] IGNORED (not read by --sampler {args.sampler}; they belong to "
            f"second_search.py's NSGA machinery): " + ', '.join(f'--{f}' for f in set_))


def build_parser_new():
    p = build_parser()
    p.description = ("2nd-stage joint W×eff_kvbits search over the ε-band product pool — "
                     "cost-aware greedy on the predicted staircase improvement (PSI)")
    # surrogate/surrogate_input inherit second_search's production defaults (rbf/genome);
    # the sh's awq branch opts into sqrty_ard_gp+plstyp, mirroring scripts/second_search.sh.
    p.set_defaults(companion_kv=40, iterations=16, n_iter=5, save='save/second_search_new/run')
    for a in p._actions:                      # hide the inherited NSGA-only flags
        if a.dest in _DEAD_FLAGS:
            a.help = argparse.SUPPRESS
        elif a.dest == 'n_doe':               # same flag, entry-point-specific meaning
            a.help = ('TOTAL DOE archs, spread over --doe_builds W rows (so cells per DOE '
                      'row = n_doe / doe_builds, at least 2 so both KV extremes always fit). '
                      'The per-iteration --companion_kv does not govern the DOE.')
    p.add_argument('--doe_builds', type=int, default=12,
                   help='DOE W builds, evenly spaced over the wbits range, each × '
                        '(both KV extremes + random KV cells)')
    p.add_argument('--sampler', default='psi', choices=['psi', 'front', 'product'],
                   help="psi = cost-aware greedy on the predicted staircase improvement "
                        "(default); front = the legacy predicted-new-front + W-gap/plane-spread "
                        "sampler; product = uniform block-product CONTROL ARM (cost-matched: "
                        "n_iter random rows × companion_kv random cells)")
    p.add_argument('--verify_frac', type=float, default=0.0,
                   help='fraction of the cell budget (0-0.5) reserved for WITHIN-CELL best-arm '
                        'identification: cells in the already-opened rows nearest the incumbent '
                        'record points, i.e. direct competitors at the same operating budget. '
                        'Measured motivation: within a budget cell the loss spread (median 0.0032) '
                        'is 0.82x a whole budget-bin step (0.0038) and best-in-cell beats '
                        'mean-in-cell by 0.0042, while the surrogate cannot rank inside a cell '
                        '(delta-hat rho ~0.215) — so it must be measured. Free-ish: an opened row '
                        'sweeps KV with no rebuild. 0.25 is a sane setting.')
    p.add_argument('--rank_cap', action='store_true',
                   help='cap builds at the numerical rank of the MEASURED loss surface + 1. The '
                        '(W-block x KV-block) table is measured near rank-2 on this codebase '
                        '(ALS explains 94.5/98.7/99.5%% at rank 1/2/3; an additive a(w)+b(kv) '
                        'surface IS rank 2), and a rank-r surface is pinned by r+1 swept rows, so '
                        'further builds cannot reveal a new direction. The cell budget is '
                        'unchanged, so saved builds become extra cells inside the open rows. '
                        'Also raises the per-row KV skeleton to the shared column set the rank '
                        'estimate needs to be identifiable; fails open (no cap) until the table '
                        'can support an estimate.')
    return p


def main(args, parser=None):
    set_seed(args.seed)
    warn_dead_flags(args, parser or build_parser_new())
    config = json.load(open(args.config))[args.model_name]
    FrontSearch(config, args).search()


if __name__ == '__main__':
    _p = build_parser_new()
    main(_p.parse_args(), _p)
