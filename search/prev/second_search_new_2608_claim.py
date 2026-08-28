"""second_search_new.py — 2nd-stage JOINT (W × eff_kvbits) search, claim-based (NSGA-free).

Design (validated offline 2608, visualize/lowrank_reaudit.* + scratchpad sims; memory:
project_lowrank_reaudit_2608 items 16-18):

  * SPACE   = ε-Pareto band product of the 1st-stage per-axis archives (full band, no
    div_k thinning; the pool IS the search space — measured: NSGA machinery adds ~0
    over pool products, all value is in the pool).
  * UNIT    = a BUNDLE: 1 W-block build + a FIXED even KV row (--companion_kv cells,
    value-even over eff_kvbits, stage-1-best tie-break, box ends included). Identical
    row for every build (factorial pairing; staggered grids measured WORSE — ghost
    claims). One AWQ build amortizes the whole row (AWQEvalPool groups by W).
  * RECORD  = the measured dominance staircase m(w,k) with a τ-indifference blur,
    τ(x) = TIE / local front slope (auto from the stage-1 envelopes; the ε-quantum
    that fixes strict-dominance near-duplicate leaks).
  * SCORE   = cone-claim: predicted (log-currency) improvement of the staircase over
    the BUDGET-UNIFORM (Lebesgue) measure — the exact HV-improvement integral that
    NSGA's crowding/geometric selectors approximate. claim(i) = ∬ max(0, log m(w,k)
    − log p_i(k)) dμ over w ≥ w_i − τ_w, where p_i = the row's predicted record curve
    over the PLANNED grid cells.
  * BATCH   = fantasy-greedy (Kriging-believer style): pick argmax claim, insert the
    row's PREDICTED records into a fantasy staircase, recompute, repeat --n_iter times
    (monotone submodular ⇒ (1−1/e) greedy guarantee; naive top-B measured 25% dup).
    Fantasies are discarded — only real measurements enter the archive; an
    over-optimistic fantasy's region regains claim next iteration (automatic recall).
  * PREDICTOR = the production 2nd-search surrogate (--surrogate rbf +
    --surrogate_input genome, second_search.py defaults; the sh's awq branch may
    opt into sqrty_ard_gp+plstyp as in scripts/second_search.sh — that path adds
    TWO stage-1-y warm-start features). Refit on the FULL archive each iteration.
  * FINAL   = nothing here: the archive dump is post_search-compatible; deployment
    picks the MEASURED top-1 in-band (prediction never decides).

Inherits SecondSearch's infrastructure (space/options derivation, evaluator/AWQ pool,
plstyp predictor stack, DOE cache, encode cache, stats format) and replaces the
candidate machinery (_next/_nsga/_companion_kv are unused here).
"""
import os, json, argparse
import numpy as np
from time import time

from utils.func import set_seed, get_correlation
from utils.metric_specs import protocol_dict
from utils.second_stage import _last_stats, select_eps_band, calc_hv, front_coverage, save_viz
from second_search import SecondSearch, build_parser


def _trap_w(c, lo, hi):
    """trapezoid (Voronoi-length) weights of sorted coords c over [lo, hi], mean-1
    normalized — the budget-uniform (Lebesgue) integration measure."""
    c = np.asarray(c, float)
    e = np.concatenate([[lo], 0.5 * (c[1:] + c[:-1]), [hi]])
    w = np.maximum(np.diff(e), 1e-12)
    return w / w.mean()


def _band_thickness(comp, y, n=64):
    """median same-budget y-spread of an ε-band pool (the quality-degeneracy scale of
    'alternatives at one budget position' — the very thing τ decides to merge)."""
    lo, hi = comp[0], comp[-1]
    win = (hi - lo) / (2 * n)
    sp = [float(y[m].max() - y[m].min()) for t in np.linspace(lo, hi, n)
          if (m := (np.abs(comp - t) <= win)).sum() >= 3]
    return float(np.median(sp)) if sp else 0.0


def _local_slope(c, env, h, floor):
    """windowed |slope| of a non-increasing envelope env(c); floored (τ cap guard)."""
    s = np.empty(len(c))
    for a in range(len(c)):
        lo = min(np.searchsorted(c, c[a] - h), a)
        hi = max(np.searchsorted(c, c[a] + h, side='right') - 1, a)
        d = c[hi] - c[lo]
        s[a] = (env[lo] - env[hi]) / d if d > 1e-9 else 0.0
    return np.maximum(s, floor)


class ClaimSearch(SecondSearch):
    # ───────────────── pools: FULL ε-band + stage-1 y (overrides SecondSearch) ─────────────────
    def _load_pools(self, args):
        """W/KV block pools = the FULL ε-band of each 1st-stage archive (no div_k pruning:
        there is no crossover to feed — the pool is the space), deduped per block keeping
        the min measured y, box-filtered, sorted by comp ascending. Also keeps the stage-1
        measured y per block (grid tie-break, τ envelopes, predictor warm-start features)."""
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

    # ───────────────── claim tables: KV grid + τ + warm-start dicts (replaces _build_knowledge) ─────────────────
    def _build_knowledge(self, args):
        a = args
        # fixed even KV row: value-even targets over the box; per target the stage-1-best
        # block within a half-spacing window (alternatives at one budget are y-degenerate,
        # 0.5×TIE spread — the tie-break is free, the leftovers are the recall reserve);
        # both box-end blocks always in.
        q = max(int(a.companion_kv), 4)
        targets = np.linspace(self.kv_comp[0], self.kv_comp[-1], q)
        half = 0.5 * (targets[1] - targets[0]) if q > 1 else np.inf
        grid = {0, len(self.KVg) - 1}
        for t in targets:
            win = np.where(np.abs(self.kv_comp - t) <= half)[0]
            grid.add(int(win[np.argmin(self.kv_y1[win])]) if len(win)
                     else int(np.argmin(np.abs(self.kv_comp - t))))
        self.grid = np.array(sorted(grid), int)                 # ascending in kv_comp
        self.grid_k = self.kv_comp[self.grid]
        # per-position ALTERNATIVE lists (--kv_rep rotate): the ~43 band blocks sharing each
        # budget position are y-degenerate (0.5×TIE spread), so rotating through them per
        # build restores KV pool diversity (measured NSGA run: 1,197 distinct KV vs 40 here)
        # at ~zero JSD cost (rotate_rep_sim: regret tie, no ghost claims, losers -4).
        self._alts = []
        for gi in self.grid:
            win = np.where(np.abs(self.kv_comp - self.kv_comp[gi]) <= max(half, 0.02))[0]
            order = win[np.argsort(self.kv_y1[win])][:max(int(a.kv_rep_alts), 1)]
            alts = [int(gi)] + [int(x) for x in order if int(x) != int(gi)]
            self._alts.append(alts[:max(int(a.kv_rep_alts), 1)])
        self._n_builds = 0
        n_alt = [len(x) for x in self._alts]
        self.accelerator.print(f"[kv_rep] {a.kv_rep}: alternatives/position median "
                               f"{int(np.median(n_alt))} (cap {a.kv_rep_alts})")
        # τ-indifference quanta from the stage-1 envelopes (LOCAL slope: global-linear τ
        # under/over-merges — the front slope varies ~19× along W)
        envW = np.minimum.accumulate(self.w_y1)
        envK = np.minimum.accumulate(self.kv_y1)
        swg = max(abs(np.polyfit(self.w_comp, envW, 1)[0]), 1e-9)
        skg = max(abs(np.polyfit(self.kv_comp, envK, 1)[0]), 1e-9)
        self._sw_loc = _local_slope(self.w_comp, envW, a.tau_win_w, 0.05 * swg)
        self._sk_loc = _local_slope(self.kv_comp, envK, a.tau_win_k, 0.05 * skg)[self.grid]
        # stage-1-y warm-start lookup (all candidates are pool products → exact keys)
        self._y1w = {tuple(g.tolist()): float(y) for g, y in zip(self.Wg, self.w_y1)}
        self._y1k = {tuple(g.tolist()): float(y) for g, y in zip(self.KVg, self.kv_y1)}
        self._y1w_mean, self._y1k_mean = float(self.w_y1.mean()), float(self.kv_y1.mean())
        # claim integration subgrid over the W pool + Lebesgue weights (budget-uniform)
        self.SW = np.unique(np.r_[np.arange(0, len(self.Wg), max(int(a.claim_sub), 1)),
                                  len(self.Wg) - 1])
        self.uw = _trap_w(self.w_comp[self.SW], self.comp_obj_min[0], self.comp_obj_max[0])
        self.uk = _trap_w(self.grid_k, self.comp_obj_min[1], self.comp_obj_max[1])
        if a.claim_measure == 'count':                          # ablation: pool-density measure
            self.uw, self.uk = np.ones_like(self.uw), np.ones_like(self.uk)
        # TIE sets the τ quanta. --tie 0 = AUTO = max(stage-1 band thickness, DOE repeat Δ):
        # the band-thickness term is free and available NOW; the repeat term (added after the
        # DOE) captures eval noise that in-process determinism hides (measured: same-process
        # repeat Δ ≈ 0 → τ→0 → same-budget ghost claims, predicted 4632 vs realized 52 —
        # thickness is the floor that prevents that collapse). AWQ rebuild noise DOES show
        # up in the repeats.
        self._band_tie = max(_band_thickness(self.w_comp, self.w_y1),
                             _band_thickness(self.kv_comp, self.kv_y1))
        self._set_tie(a.tie if a.tie > 0 else max(self._band_tie, 1e-4),
                      provisional=(a.tie <= 0))
        self.band_table = None                                  # plstyp typicality off
        self.active = np.arange(self.n_var)
        self.accelerator.print(
            f"[claim] KV row = {len(self.grid)} cells value-even (+ends, stage-1-best tie-break) | "
            f"integral {len(self.SW)}×{len(self.grid)} ({a.claim_measure} measure)")

    def _set_tie(self, tie, provisional=False):
        """set/refresh the τ quanta from a TIE estimate (τ = TIE / local envelope slope,
        capped). Cheap to recompute — called again once the DOE repeats measure TIE."""
        self.tie = float(max(tie, 1e-5))
        a = self.args
        self.tau_w = np.minimum(self.tie / self._sw_loc, a.tau_cap_w)
        self.tau_k = np.minimum(self.tie / self._sk_loc, a.tau_cap_k)
        self.maskW = (self.w_comp[:, None]
                      <= self.w_comp[self.SW][None, :] + self.tau_w[self.SW][None, :]).astype(np.float32)
        self.accelerator.print(
            f"[tie] TIE={self.tie:.2e} ({'provisional — DOE repeats will calibrate' if provisional else 'set'})"
            f" → τ_w [{self.tau_w.min():.4f},{self.tau_w.max():.4f}]"
            f" τ_k [{self.tau_k.min():.4f},{self.tau_k.max():.4f}]")

    def _auto_tie(self, archive):
        """AUTO-TIE from paired repeats: duplicate-genome archive rows are repeat
        measurements; TIE = max |Δy| over pairs (conservative — matches how the 0.0016
        default was chosen from its probe: the upper end of observed deltas)."""
        ys = {}
        for g, x in zip(self._encode_archive(archive), archive):
            ys.setdefault(tuple(g.tolist()), []).append(float(x[1]))
        d = [abs(v[i] - v[0]) for v in ys.values() if len(v) > 1 for i in range(1, len(v))]
        return (float(np.max(d)) if d else None), len(d)

    # ───────────────── predictor input: plstyp + stage-1-y warm-start features ─────────────────
    def _pls_features(self, Xf):
        base = super()._pls_features(Xf)
        X = np.clip(np.round(np.asarray(Xf, float)), 0, self.xu).astype(int)
        yw = np.array([self._y1w.get(tuple(g[:self.nw].tolist()), self._y1w_mean) for g in X])
        yk = np.array([self._y1k.get(tuple(g[self.nw:].tolist()), self._y1k_mean) for g in X])
        return np.column_stack([base, np.sqrt(yw), np.sqrt(yk)])   # sqrt: match the head's scale

    # ───────────────── staircase / prediction / claim ─────────────────
    def _points(self, archive):
        """measured (w, k, y) records from the archive (foreign/DOE-cached archs contribute
        through their comp coords — the staircase never needs pool indices)."""
        return [(float(x[2]), float(x[3]), float(x[1])) for x in archive if np.isfinite(x[1])]

    def _staircase(self, pts, rows):
        """τ-blurred record staircase m(w,k) at (self.w_comp[rows] × grid) — the running
        best measured y over the dominated-budget region, blurred by τ so an indifference-
        width neighbor counts as the same position. O(P·Q + R·Q)."""
        R, Q = len(rows), len(self.grid)
        if not pts:
            return np.full((R, Q), np.inf)
        o = sorted(range(len(pts)), key=lambda i: pts[i][0])
        pw = np.array([pts[i][0] for i in o])
        pk = np.array([pts[i][1] for i in o])
        pv = np.array([pts[i][2] for i in o])
        contrib = np.where(self.grid_k[None, :] + self.tau_k[None, :] >= pk[:, None],
                           pv[:, None], np.inf)
        cum = np.minimum.accumulate(contrib, axis=0)
        L = np.searchsorted(pw, self.w_comp[rows] + self.tau_w[rows], side='right')
        return np.where(L[:, None] > 0, cum[np.maximum(L - 1, 0)], np.inf)

    def _pred_matrix(self, predictor):
        """PRED[i, t] over every pool cell (W row i × grid col t), chunked. Computed once
        per iteration (the predictor is fixed within an iteration; only records move)."""
        nW, Q = len(self.Wg), len(self.grid)
        KVrow = self.KVg[self.grid]
        P = np.empty((nW, Q), np.float64)
        chunk = max(1, int(self.args.claim_chunk) // Q)
        for s in range(0, nW, chunk):
            e = min(s + chunk, nW)
            X = np.concatenate([np.repeat(self.Wg[s:e], Q, axis=0),
                                np.tile(KVrow, (e - s, 1))], axis=1).astype(float)
            P[s:e] = np.asarray(predictor.predict(X[:, self.active])).ravel().reshape(e - s, Q)
        # row record curve p_i(k) over the PLANNED grid cells (τ_k-blurred cummin) —
        # claim never credits cells the row will not measure
        jmax = np.searchsorted(self.grid_k, self.grid_k + self.tau_k, side='right')
        Pc = np.minimum.accumulate(P, axis=1)
        self._p_row = Pc[:, np.maximum(jmax - 1, 0)]
        self._logp = np.log(np.maximum(self._p_row, 1e-9)).astype(np.float32)
        return P

    def _claims(self, pts, banned):
        """cone-claim of every pool W row against the (possibly fantasy-augmented) staircase:
        claim(i) = Σ_{s,t} 1[w_s ≥ w_i − τ_w] · max(0, log m_st − log p_it) · u_w u_k
        (inf staircase cells → --cap_cell each; a per-row bonus caps runaway inf regions)."""
        m = self._staircase(pts, self.SW)
        A = np.log(np.maximum(m, 1e-9)).astype(np.float32)       # inf stays inf
        finA = np.isfinite(A)
        with np.errstate(invalid='ignore', divide='ignore'):     # log-equivalent of m − p > TIE
            self._thr = np.where(finA, -np.log1p(-np.minimum(self.tie / np.maximum(m, 1e-9),
                                                             0.99)), 0).astype(np.float32)
        cw = (self.uw[:, None] * self.uk[None, :]).astype(np.float32)
        nW = len(self.Wg)
        claim = np.empty(nW, np.float64)
        chunk = max(1, int(self.args.claim_chunk) // max(len(self.SW), 1))
        for s in range(0, nW, chunk):
            e = min(s + chunk, nW)
            B = self._logp[s:e]                                   # (c, Q)
            # finite-record cells: (a) MATERIALITY DEADBAND — a cell counts only if the
            # predicted improvement exceeds TIE in ABSOLUTE terms (m − p > TIE). Without
            # it the log currency amplifies sub-noise absolute gaps in the saturated
            # high-W region into large relative claims (run 2608211721: 37/70 builds set
            # ZERO records, 34 of them within TIE of the record, wbits median 3.69 —
            # phantom claims 42-90 realizing 0.0). τ is the budget-axis indifference;
            # this is its value-axis twin, reusing the same TIE — no new knob.
            # (b) cap the per-cell log-gap — ">2× better than the same-budget record"
            # is predictor error, not opportunity (tiny-N ghost guard, measured).
            gap = A[None, :, :] - B[:, None, :]
            g = np.where(finA[None, :, :],
                         np.where(gap > self._thr[None, :, :],
                                  np.minimum(gap, np.float32(self.args.claim_gap_cap)), 0),
                         np.float32(self.args.cap_cell))
            claim[s:e] = np.einsum('cst,st,cs->c', g, cw, self.maskW[s:e])
        if banned:
            claim[np.fromiter(banned, int)] = 0.0
        return claim

    # ───────────────── fantasy-greedy batch pick ─────────────────
    def _pick_batch(self, archive, B):
        """B build rows via fantasy-greedy: argmax claim → insert the row's PREDICTED grid
        records into a fantasy point set → recompute → next. Rows with no unmeasured grid
        cell are banned. Returns (rows, their predicted claims)."""
        pts = self._points(archive)
        banned = set(self._built_rows)          # one build per W row (any-cell semantics)
        picks, claims = [], []
        for _ in range(B):
            cl = self._claims(pts, banned)
            i = int(np.argmax(cl))
            if cl[i] <= self.args.claim_stop and picks:
                break
            picks.append(i); claims.append(float(cl[i])); banned.add(i)
            pts = pts + [(float(self.w_comp[i]), float(self.grid_k[t]), float(self._p_row[i, t]))
                         for t in range(len(self.grid))]
        return picks, claims

    def _row_cells_clgeo(self, i, predictor, pts):
        """--kv_sel clgeo: the user-specified rule — cells of row i whose CURRENT-GP
        prediction beats the previous iterations' measured front (τ-blurred staircase at
        the cell's own eff_kvbits) by > TIE, thinned EVENLY over eff_kvbits to the
        --companion_kv budget (+ pool box ends always). Shortfall (few/no positives —
        e.g. a saturated region) falls back to the even grid so coverage never collapses.
        Returns KV POOL indices (full 3,878-block pool, not the 40-grid)."""
        n = len(self.KVg)
        X = np.concatenate([np.repeat(self.Wg[i:i + 1], n, 0), self.KVg], 1).astype(float)
        mu = np.empty(n)
        ch = max(1, int(self.args.claim_chunk))
        for s in range(0, n, ch):
            e = min(s + ch, n)
            mu[s:e] = np.asarray(predictor.predict(X[s:e][:, self.active])).ravel()
        # measured record at each pool cell's own budget coord (τ_k interpolated)
        tauk = np.interp(self.kv_comp, self.grid_k, self.tau_k)
        if pts:
            o = sorted(range(len(pts)), key=lambda x: pts[x][0])
            pw = np.array([pts[x][0] for x in o]); pk = np.array([pts[x][1] for x in o])
            pv = np.array([pts[x][2] for x in o])
            L = np.searchsorted(pw, self.w_comp[i] + self.tau_w[i], side='right')
            if L > 0:
                contrib = np.where(self.kv_comp[None, :] + tauk[None, :] >= pk[:L, None],
                                   pv[:L, None], np.inf)
                m_row = np.min(contrib, axis=0)
            else:
                m_row = np.full(n, np.inf)
        else:
            m_row = np.full(n, np.inf)
        pos = np.where(np.isfinite(m_row), mu < m_row - self.tie, True)
        C = np.where(pos)[0]
        q = max(int(self.args.companion_kv), 4)
        sel = {0, n - 1}                                     # pool box ends
        if len(C):
            take = np.unique(np.linspace(0, len(C) - 1, min(q - 2, len(C))).round().astype(int))
            sel |= {int(C[x]) for x in take}                 # C is kv-sorted (pool sorted)
        for t in range(len(self.grid)):                      # fill shortfall from even grid
            if len(sel) >= q:
                break
            sel.add(int(self.grid[t]))
        return sorted(sel), mu

    def _rep(self, t, serial):
        """KV block filling grid position t for the `serial`-th build: stage-1-best
        ('best') or cycling through the position's degenerate alternatives ('rotate').
        Rotation is LATIN-SHIFTED (phase (serial + t) mod n): a plain serial-mod would pair
        build b with rank-(b mod n) alternatives at EVERY position — a systematic (sub-TIE)
        rank×row confound; the per-position shift mixes ranks within each build and cycles
        every alternative at every position across builds (regret A/B: tie, hygiene free)."""
        alts = self._alts[t]
        return alts[(serial + t) % len(alts)] if self.args.kv_rep == 'rotate' else alts[0]

    def _cell_key(self, i, t, serial=0):
        return tuple(np.concatenate([self.Wg[i], self.KVg[self._rep(t, serial)]]).tolist())

    def _cells_todo(self, i, serial=0):
        return [t for t in range(len(self.grid))
                if self._cell_key(i, t, serial) not in self._measured]

    def _register(self, archive):
        """rebuild the measured-genome set + row ledger from the archive (resume / DOE
        cache safe). Foreign archs simply don't map — they still shape the staircase via
        their comp coords. _row_done tracks POSITIONS covered per row (any alternative)."""
        if not hasattr(self, '_wi'):
            self._wi = {tuple(g.tolist()): i for i, g in enumerate(self.Wg)}
            self._ki = {tuple(g.tolist()): int(j) for j, g in enumerate(self.KVg)}
            self._pos_of = {}                       # KV pool idx -> grid position (via alts)
            for t, alts in enumerate(self._alts):
                for j in alts:
                    self._pos_of.setdefault(int(j), t)
            self._gt = {int(a[0]): t for t, a in enumerate(self._alts)}   # rep0 map (compat)
        self._row_done = {}
        self._measured = set()
        rows_built = set()
        for g in self._encode_archive(archive):
            self._measured.add(tuple(g.tolist()))
            i = self._wi.get(tuple(g[:self.nw].tolist()))
            if i is not None:
                rows_built.add(i)
            j = self._ki.get(tuple(g[self.nw:].tolist()))
            t = self._pos_of.get(j) if j is not None else None
            if i is not None and t is not None:
                self._row_done.setdefault(i, set()).add(t)
        self._built_rows = rows_built
        self._n_builds = max(getattr(self, '_n_builds', 0), len(rows_built))

    def _save_viz_claim(self, it, archive):
        """--debug: claim-machine panel (iter_<it>_claim.png) beside the standard save_viz —
        (a) measured-cell map coloured DOE vs iteration, (b) fantasy-greedy build picks per
        iteration (size ~ claim), (c) predicted Σclaim vs realized gain. History is read
        back from this run's iter_*.stats, so it survives --resume."""
        try:
            import glob as _g, re as _re
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        except Exception as e:
            self.accelerator.print(f"[viz] claim panel skipped ({e})"); return
        stats = sorted(_g.glob(os.path.join(self.save_path, 'iter_*.stats')),
                       key=lambda p: int(_re.search(r'iter_(\d+)', p).group(1)))
        cl, sizes = {}, []
        for p in stats:
            d = json.load(open(p))
            if d.get('iteration', 0) > it:      # retro-render safety: history up to `it` only
                continue
            if 'claim' in d:
                cl[d['iteration']] = d['claim']
            sizes.append(len(d['candidates']))
        n_doe = len(archive) - sum(sizes)
        phase = np.zeros(len(archive), int)
        ofs = n_doe
        for k, n in enumerate(sizes, 1):
            phase[ofs:ofs + n] = k; ofs += n
        w = np.array([r[2] for r in archive]); kv = np.array([r[3] for r in archive])
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        ax = axes[0]; mD = phase == 0
        ax.scatter(w[mD], kv[mD], s=6, color='0.8', label=f'DOE ({int(mD.sum())})')
        if (~mD).any():
            sc = ax.scatter(w[~mD], kv[~mD], s=8, c=phase[~mD], cmap='viridis',
                            label=f'iterations ({int((~mD).sum())})')
            plt.colorbar(sc, ax=ax, fraction=0.045).set_label('iteration', fontsize=8)
        ax.set_xlabel('wbits'); ax.set_ylabel('eff. kvbits')
        ax.set_title(f'measured cells (n={len(archive)})'); ax.legend(fontsize=8)
        ax = axes[1]
        for k, c in sorted(cl.items()):
            pr = np.array(c['predicted'])
            ax.scatter(c['build_wbits'], [k] * len(pr),
                       s=20 + 130 * pr / max(pr.max(), 1e-9), color='tab:blue', alpha=0.75)
        ax.set_xlabel('build wbits'); ax.set_ylabel('iteration'); ax.grid(color='0.92')
        ax.set_title('build picks (size ~ claim)')
        ax = axes[2]
        ks = sorted(cl.keys())
        ax.plot(ks, [sum(cl[k]['predicted']) for k in ks], 'o-', color='tab:orange',
                label='predicted Σclaim')
        ax.plot(ks, [cl[k]['realized_total'] for k in ks], 's-', color='tab:green',
                label='realized gain')
        ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.grid(color='0.92')
        ax.set_title('claim calibration'); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_path, f'iter_{it}_claim.png'), dpi=120,
                    bbox_inches='tight')
        plt.close(fig)

    # ───────────────── main loop ─────────────────
    def search(self):
        acc = self.accelerator; main = acc.is_main_process
        t0 = time(); start_it = 1
        if self.args.resume:
            rf = json.load(open(self.args.resume))
            archive = rf['archive']; start_it = rf['iteration'] + 1
            if main:
                acc.print(f"[resume] {len(archive)} archs from iter {rf['iteration']}")
                if self.args.tie <= 0:      # AUTO-TIE: max(band thickness, archive pairs, stored)
                    t, npair = self._auto_tie(archive)
                    self._set_tie(max(t or 0.0, self._band_tie, rf.get('tie') or 0.0, 1e-4))
        else:
            # DOE = --doe_builds W anchors (box ends + value-even over wbits, stage-1-best
            # tie-break) × the full KV row; cached DOE archs (--doe_results) count first.
            if main:
                seeded = self._load_doe([]) if self.args.doe_results else []
                self._register(seeded)
                targets = np.linspace(self.comp_obj_min[0], self.comp_obj_max[0],
                                      max(self.args.doe_builds, 2))
                half = 0.5 * (targets[1] - targets[0])
                rows = {0, len(self.Wg) - 1}
                for t in targets:
                    win = np.where(np.abs(self.w_comp - t) <= half)[0]
                    rows.add(int(win[np.argmin(self.w_y1[win])]) if len(win)
                             else int(np.argmin(np.abs(self.w_comp - t))))
                plan = []
                for i in sorted(rows):
                    cells = self._cells_todo(i, self._n_builds)
                    if cells:
                        plan += [(i, t, self._n_builds) for t in cells]
                        self._n_builds += 1
                archs = [self.ss.decode(np.array(self._cell_key(i, t, sr), int))
                         for i, t, sr in plan]
                # AUTO-TIE repeats: re-measure a few DOE cells (spread over the plan) to
                # estimate the same-seed noise floor under THIS run's exact protocol
                n_rep = self.args.tie_repeats if self.args.tie <= 0 else 0
                if n_rep > 0 and plan:
                    step = max(1, len(plan) // n_rep)
                    archs += [self.ss.decode(np.array(self._cell_key(i, t, sr), int))
                              for i, t, sr in plan[::step][:n_rep]]
                acc.print(f"[DOE] {len(rows)} builds × ≤{len(self.grid)} cells → {len(archs)} archs"
                          + (f" (incl {n_rep} TIE repeats)" if n_rep else "")
                          + (f" (+{len(seeded)} cached)" if seeded else ""))
            else:
                archs, seeded = [], []
            archs = acc.gather_for_metrics(archs, use_gather_object=True)
            acc.wait_for_everyone()
            kept, metric, comp = self._evaluate(archs) if archs else ([], [], [])
            if main:
                measured = [[archs[i], m, *c] for i, m, c in zip(kept, metric, comp)]
                if kept:
                    self._cache_doe([archs[i] for i in kept], metric)
                archive = seeded + measured
                self._register(archive)
                losses = [x[1] for x in archive]
                acc.print(f"[DOE] archive {len(archive)}  loss {min(losses):.4f}-{max(losses):.4f} "
                          f"({time()-t0:.1f}s)")
                if self.args.tie <= 0:      # AUTO-TIE = max(band thickness, repeat Δ)
                    t, npair = self._auto_tie(archive)
                    self._set_tie(max(t or 0.0, self._band_tie, 1e-4))
                    acc.print(f"[tie] auto: band-thickness {self._band_tie:.2e} ∨ "
                              f"repeat Δ {(t or 0.0):.2e} ({npair} pairs)")
            else:
                archive = []
        if main:
            self._register(archive)
            ref_pt = np.array([np.max([x[i] for x in archive]) for i in range(1, len(self.comp_obj) + 2)])
        acc.wait_for_everyone()

        for it in range(start_it, self.iterations + 1):
            iter_start = time()
            if main:
                tp = time()
                pred, a_pred = self._fit_predictor(archive)
                self._pred_matrix(pred)
                predictor_time = time() - tp
                tn = time()
                picks, claims = self._pick_batch(archive, self.n_iter)
                plan = []                                   # (genome-key, predicted y)
                pts_now = self._points(archive)
                for i in picks:
                    if self.args.kv_sel == 'clgeo':
                        js, mu = self._row_cells_clgeo(i, pred, pts_now)
                        cells = [(tuple(np.concatenate([self.Wg[i], self.KVg[j]]).tolist()),
                                  float(mu[j])) for j in js]
                    else:
                        sr = self._n_builds
                        cells = [(self._cell_key(i, t, sr), float(self._p_row[i, t]))
                                 for t in self._cells_todo(i, sr)]
                    cells = [(k, p) for k, p in cells if k not in self._measured]
                    if cells:
                        plan += cells
                        self._n_builds += 1
                cands = [self.ss.decode(np.array(k, int)) for k, _ in plan]
                next_time = time() - tn
                pts_before = self._points(archive)
                acc.print(f"[iter {it}] builds {[f'{self.w_comp[i]:.2f}' for i in picks]} "
                          f"claims {[f'{c:.3f}' for c in claims]} → {len(cands)} cells")
            else:
                cands = []
            acc.wait_for_everyone()
            cands = acc.gather_for_metrics(cands, use_gather_object=True)
            if not cands:
                if main:
                    acc.print(f"Iter {it}: no claim-positive builds left; stop")
                break
            kept, c_metric, c_comp = self._evaluate(cands)
            if main:
                c_pred = np.array([p for _, p in plan])
                rmse, rho, tau = get_correlation(
                    np.concatenate([np.asarray(a_pred).ravel(), c_pred[kept]]),
                    np.array([x[1] for x in archive] + c_metric))
                for i, m, c in zip(kept, c_metric, c_comp):
                    archive.append([cands[i], m, *c])
                self._register(archive)
                # claim hit-rate diagnostic: realized staircase gain (log currency, same
                # measure) vs the batch's predicted claims — the free instrumentation
                m0 = self._staircase(pts_before, self.SW)
                m1 = self._staircase(self._points(archive), self.SW)
                fb = np.isfinite(m0) & np.isfinite(m1)
                realized = float(np.sum(np.where(
                    fb, np.maximum(0, np.log(np.maximum(m0, 1e-9)) - np.log(np.maximum(m1, 1e-9))), 0)
                    * self.uw[:, None] * self.uk[None, :]))
                F = np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)])
                hv = calc_hv(ref_pt, F); cov = front_coverage(archive, self.comp_obj)
                iter_time = time() - iter_start
                acc.print(f"Iter {it}: hv = {hv:.2f}, iter time : {iter_time:.2f}s, "
                          f"predictor_time : {predictor_time:.2f}, next_time : {next_time:.2f}")
                acc.print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, "
                          f"Kendall's Tau = {tau:.4f}")
                acc.print(f"[claim] predicted Σ {sum(claims):.3f} vs realized {realized:.3f} "
                          f"({len(kept)}/{len(cands)} cells kept)")
                for obj in self.comp_obj:
                    c = cov[obj]
                    acc.print(f"  {obj} front-coverage : {c['coverage']*100:.1f}%  "
                              f"front=[{c['front_min']:.3f}, {c['front_max']:.3f}] / "
                              f"full=[{c['full_min']:.3f}, {c['full_max']:.3f}]")
                if it % self.save_iter == 0 or it == self.iterations:
                    os.makedirs(self.save_path, exist_ok=True)
                    with open(os.path.join(self.save_path, f"iter_{it}.stats"), 'w') as f:
                        json.dump({'archive': archive, 'candidates': archive[-len(kept):], 'hv': hv,
                                   'tie': self.tie,
                                   'surrogate': {'model': self.predictor, 'rmse': rmse,
                                                 'rho': rho, 'tau': tau},
                                   'claim': {'builds': [int(i) for i in picks],
                                             'build_wbits': [float(self.w_comp[i]) for i in picks],
                                             'predicted': claims, 'realized_total': realized},
                                   'coverage': cov, 'protocol': protocol_dict(self.args),
                                   'iteration': it}, f)
                    if self.debug:      # search.py-style per-iter figures (+ claim panel)
                        save_viz(self.save_path, it, archive, c_metric,
                                 np.asarray(c_pred).ravel()[kept], c_comp, cov,
                                 self.comp_obj, self.comp_obj_min, self.comp_obj_max)
                        self._save_viz_claim(it, archive)
            acc.wait_for_everyone()

        if self.pool is not None:
            self.pool.close()
        if main:
            acc.print(f"[done] {len(archive)} archs, {time()-t0:.1f}s → {self.save_path}")
            self._write_results(archive, time() - t0)
        return archive


def build_parser_new():
    p = build_parser()
    p.description = "2nd-stage joint W×eff_kvbits search — claim-based (staircase + fantasy-greedy), NSGA-free"
    # surrogate/surrogate_input inherit second_search's production defaults (rbf/genome);
    # the sh's awq branch opts into sqrty_ard_gp+plstyp, mirroring scripts/second_search.sh.
    p.set_defaults(companion_kv=40, iterations=16, n_iter=5, save='save/second_search_new/run')
    p.add_argument('--doe_builds', type=int, default=12,
                   help='DOE W builds (box ends + value-even wbits anchors), each × the full KV row')
    p.add_argument('--tie', type=float, default=0.0,
                   help='quality-indifference quantum (sets τ = TIE/slope). 0 (DEFAULT) = AUTO '
                        '= max(stage-1 band same-budget thickness [free, model/protocol-'
                        'adaptive], DOE repeat |Δ| from --tie_repeats [captures eval/rebuild '
                        'noise], 1e-4 floor). Pass an explicit value to pin it.')
    p.add_argument('--tie_repeats', type=int, default=4,
                   help='AUTO-TIE: number of DOE cells re-measured for the noise-floor estimate '
                        '(~0.4%% of a 25-build DOE budget)')
    p.add_argument('--tau_cap_w', type=float, default=0.20)
    p.add_argument('--tau_cap_k', type=float, default=0.60)
    p.add_argument('--tau_win_w', type=float, default=0.20,
                   help='wbits window for the local envelope slope behind τ_w')
    p.add_argument('--tau_win_k', type=float, default=0.40)
    p.add_argument('--claim_sub', type=int, default=8,
                   help='staircase-integral W subsample stride (resolution 8×4→6×3 measured Δ<rule gaps)')
    p.add_argument('--claim_chunk', type=int, default=40_000,
                   help='max predictor/claim elements per chunk (memory guard)')
    p.add_argument('--claim_measure', default='budget', choices=['budget', 'count'],
                   help='claim integration measure: budget = Lebesgue/budget-uniform (DEFAULT; '
                        'the pre-registered A/B endpoint), count = pool-density (ablation)')
    p.add_argument('--cap_cell', type=float, default=0.05,
                   help='per-cell claim cap for unexplored (inf-record) staircase cells')
    p.add_argument('--claim_gap_cap', type=float, default=0.7,
                   help='per-cell log-gap cap for FINITE-record cells (0.7 ≈ log 2: a cell may '
                        'claim at most "2× better than the measured record" — beating the '
                        'band-product front by more at the same budget is predictor error)')
    p.add_argument('--claim_stop', type=float, default=0.0,
                   help='stop early when the best claim falls to this (0 = never; W count fixed)')
    p.add_argument('--kv_rep', default='rotate', choices=['rotate', 'best'],
                   help='which KV block fills each value-even grid position: rotate (DEFAULT) '
                        '= cycle through the position\'s y-degenerate band alternatives per '
                        'build (restores KV pool diversity — measured NSGA run used 1,197 '
                        'distinct KV vs 40 under best; JSD-regret tie, no ghost claims); '
                        'best = always the stage-1-best representative (identical rows)')
    p.add_argument('--kv_rep_alts', type=int, default=8,
                   help='rotate: max alternatives kept per grid position (window = half grid '
                        'spacing; median available ≈ 43)')
    p.add_argument('--kv_sel', default='clgeo', choices=['even', 'clgeo'],
                   help="per-build KV cell selection. clgeo (DEFAULT): from the FULL KV pool, "
                        "keep cells whose CURRENT-GP prediction beats the previous measured "
                        "front (τ-staircase) by more than TIE, pick --companion_kv of them "
                        "evenly over eff_kvbits (+pool ends); shortfall filled from the even "
                        "grid — the NSGA principle (predicted-front-exceeding ∩ geometric) on "
                        "the KV axis. even: fixed value-even grid (--kv_rep applies). DOE "
                        "always uses the even grid (no predictor yet). The HQQ run measured "
                        "rows as all-or-nothing (clgeo≡even there); AWQ is the discriminating "
                        "regime — decide by the real A/B.")
    return p


def main(args):
    set_seed(args.seed)
    config = json.load(open(args.config))[args.model_name]
    ClaimSearch(config, args).search()


if __name__ == '__main__':
    main(build_parser_new().parse_args())
