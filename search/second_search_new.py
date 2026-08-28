"""second_search_new.py — 2nd-stage JOINT (W × eff_kvbits) search: predicted-new-front
+ spread sampling (NSGA-free, threshold-free).

Per iteration:
  1. PREDICT  every cell of the ε-band product pool (full W×KV, chunked).
  2. EXTRACT  the predicted NEW Pareto front: cells strictly below every measured
     archive point AND every other predicted cell at dominated budget coords
     (weak-budget / strict-loss dominance — a pure definition: no TIE, no windows,
     no magnitudes; the ε-dominance variant was verified ≡ pymoo on subsamples).
  3. PICK W   --n_iter build rows among front-holding rows, greedily filling the
     largest W-axis gap of {measured W coords} ∪ picks; rows within one pool
     spacing of the best fill (the pool is W-dense) tie-break to the lower mean
     predicted curve — prediction is used ORDINALLY (who is lower), never as a
     score (how much lower).
  4. PICK KV  both pool-extreme KV blocks per build (always measured) + the rest
     of the per-iteration cell budget (--n_iter × --companion_kv total) placed on
     the picked rows' front cells at the largest plane gap of {measured Pareto
     front} ∪ placed cells. Per-row cell counts are EMERGENT (a row with a wide
     front segment in a thin region gets more); only the total is fixed.
  5. MEASURE  → archive. Records / front / final selection come from measurements
     only: an over-optimistic predicted cell is corrected the moment it is
     measured; unpicked fantasies re-compete against next iteration's measured
     front (bounded exposure: one batch per iteration).

DOE = random W rows × (both KV extremes + random KV cells). The only parameters
are budgets (--doe_builds / --iterations / --n_iter / --companion_kv) + --seed:
no thresholds, no tuning knobs. FINAL = measured top-1 in band (stats dump is
post_search-compatible; prediction never decides deployment).

Inherits SecondSearch's infrastructure (space/options derivation, evaluator/AWQ
pool, predictor stack, DOE cache, encode cache, stats format); the NSGA candidate
machinery (_next/_nsga/_companion_kv) is unused here.
"""
import os, json
import numpy as np
from time import time

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from utils.func import set_seed, get_correlation
from utils.metric_specs import protocol_dict
from utils.second_stage import _last_stats, select_eps_band, calc_hv, front_coverage, save_viz
from second_search import SecondSearch, build_parser


class FrontSearch(SecondSearch):
    _CHUNK = 200_000            # pool-prediction cells per predict call (memory guard only)

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
        """rebuild the measured-genome set + built-W ledger from the archive (resume /
        DOE-cache safe). Foreign archs simply don't map to a pool row — they still shape
        the record staircase through their comp coords."""
        if not hasattr(self, '_wi'):
            self._wi = {tuple(g.tolist()): i for i, g in enumerate(self.Wg)}
        self._measured = set()
        self._built_w = set()
        for g in self._encode_archive(archive):
            self._measured.add(tuple(g.tolist()))
            i = self._wi.get(tuple(g[:self.nw].tolist()))
            if i is not None:
                self._built_w.add(i)

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

    # ───────────────── 2. predicted NEW Pareto front (pure dominance) ─────────────────
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

    # ───────────────── 3. W selection: union gap filling ─────────────────
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

    # ───────────────── 4. KV placement: extremes + plane gap filling ─────────────────
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
                   label=f'predicted new front ({len(fi)})')
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
        t0 = time(); start_it = 1
        self._n_doe_archs = 0
        if self.args.resume:
            rf = json.load(open(self.args.resume))
            archive = rf['archive']; start_it = rf['iteration'] + 1
            self._n_doe_archs = rf.get('n_doe_archs', 0)
            if main:
                acc.print(f"[resume] {len(archive)} archs from iter {rf['iteration']}")
        else:
            # DOE: random W rows × (both KV extremes + random cells) — predictor-free.
            if main:
                seeded = self._load_doe([]) if self.args.doe_results else []
                self._register(seeded)
                rng = np.random.default_rng(self.args.seed)
                nK = len(self.KVg)
                nrows = min(max(int(self.args.doe_builds), 1), len(self.Wg))
                rows = rng.choice(len(self.Wg), size=nrows, replace=False)
                q = max(int(self.args.companion_kv), 2)
                archs, planned = [], set(self._measured)
                for i in rows:
                    cells = [0, nK - 1]
                    if q > 2 and nK > 2:
                        cells += [int(x) for x in rng.choice(np.arange(1, nK - 1),
                                                             size=min(q - 2, nK - 2),
                                                             replace=False)]
                    for j in cells:
                        k = self._key(int(i), int(j))
                        if k not in planned:
                            planned.add(k)
                            archs.append(self.ss.decode(np.array(k, int)))
                acc.print(f"[DOE] {nrows} random builds × (2 extremes + {q - 2} random) "
                          f"→ {len(archs)} archs" + (f" (+{len(seeded)} cached)" if seeded else ""))
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
        acc.wait_for_everyone()

        for it in range(start_it, self.iterations + 1):
            iter_start = time()
            if main:
                tp = time()
                pred, a_pred = self._fit_predictor(archive)
                P = self._predict_pool(pred)
                predictor_time = time() - tp
                tn = time()
                mask = self._new_front(P, archive)
                picks = self._pick_rows(mask, P, self.n_iter)
                budget = len(picks) * int(self.args.companion_kv)
                plan = self._place_cells(picks, mask, archive, budget) if picks else []
                next_time = time() - tn
                cands = [self.ss.decode(np.array(self._key(i, j), int)) for i, j in plan]
                c_pred = np.array([float(P[i, j]) for i, j in plan])
                pts_before = self._points(archive)
                acc.print(f"[iter {it}] predicted front: {int(mask.sum())} cells in "
                          f"{int(mask.any(1).sum())} rows | builds "
                          f"{[f'{self.w_comp[i]:.2f}' for i in picks]} → {len(plan)} cells"
                          + (f" (short of {budget})" if len(plan) < budget else ""))
            else:
                cands = []
            acc.wait_for_everyone()
            cands = acc.gather_for_metrics(cands, use_gather_object=True)
            if not cands:
                if main:
                    acc.print(f"Iter {it}: no unmeasured predicted-front cells; stop")
                break
            kept, c_metric, c_comp = self._evaluate(cands)
            if main:
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
                                   'ndfront': {'builds': [int(i) for i in picks],
                                               'build_wbits': [float(self.w_comp[i]) for i in picks],
                                               'front_cells': int(mask.sum()),
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


def build_parser_new():
    p = build_parser()
    p.description = ("2nd-stage joint W×eff_kvbits search — predicted-new-front + spread "
                     "sampling (NSGA-free, threshold-free)")
    # surrogate/surrogate_input inherit second_search's production defaults (rbf/genome);
    # the sh's awq branch opts into sqrty_ard_gp+plstyp, mirroring scripts/second_search.sh.
    p.set_defaults(companion_kv=40, iterations=16, n_iter=5, save='save/second_search_new/run')
    p.add_argument('--doe_builds', type=int, default=12,
                   help='random DOE W builds, each × (both KV extremes + random KV cells)')
    return p


def main(args):
    set_seed(args.seed)
    config = json.load(open(args.config))[args.model_name]
    FrontSearch(config, args).search()


if __name__ == '__main__':
    main(build_parser_new().parse_args())
