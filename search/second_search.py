"""second_search.py — 2nd-stage JOINT (W × eff_kvbits) NAS, HQQ-based, NSGA-III.

Structure mirrors search.py (Search): DOE → _fit_predictor → _next → _evaluate →
iter_<it>.stats loop; archive = [arch, metric, *comp_obj]. Uses the SAME building
blocks as search.py — LlamaSearchSpace (encode/decode/encode_predictor) and
LlamaEvaluator (eval). Joint-specific operators live in utils/joint_search.py.

  * genome = ss.encode(arch); a (W-block, KV-block) assembled from the 1st-stage
    per-axis Pareto fronts (--w_expr / --eff_kv_expr).
  * CROSSOVER unit = whole axis block (W rows / KV rows; additive W⊥KV).
  * MUTATION = importance-weighted by 1st-stage lever strength (ε-floor → coverage;
    direction free → non-monotone allocations stay reachable; NO monotone repair).
  * PREDICTOR input = ss.encode_predictor (arch input, like search.py).
  * objectives = (loss, wbits, eff_kvbits); loss = real HQQ JSD via LlamaEvaluator.eval.
Output iter_<it>.stats is search.py-compatible (post_search-consumable).
"""
import os, json, argparse
import numpy as np
from time import time

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import Hypervolume

from search_space.llama import LlamaSearchSpace
from predictor.factory import get_predictor
from utils.func import set_seed, get_correlation, get_net_info
from utils.select import maximin_extras
from utils.joint_search import (
    encoding_xu, nw_split, load_block_pools, gene_weights, freeze_mask, block_segments, derive_options, derive_qeft,
    FrontierProductSampling, AxisBlockCrossover, WeightedIntMutation, KnowledgeMutation, JointAuxProblem, JointComp)


class SecondSearch:
    """Joint 2nd-stage search — same loop as search.py::Search; joint operators."""
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.save_path = args.save
        self.iterations, self.n_doe, self.n_iter = args.iterations, args.n_doe, args.n_iter
        self.predictor, self.ga_pop_size = args.surrogate, args.pop
        self._pred_device = getattr(args, 'predictor_device', 'auto'); self._pred_dev_logged = False
        self.save_iter, self.debug = max(1, args.save_iter), args.debug
        # comp objectives + budget box, taken at once like search.py (--comp_obj / --comp_obj_min/max).
        # Any unset edge (default None) is AUTO-DERIVED from the 1st-stage pools' achievable comp
        # range below (the input files decide the range); pass --comp_obj_min/max only to narrow.
        self.comp_obj = list(args.comp_obj)
        n = len(self.comp_obj)
        self._box_min = list(args.comp_obj_min) if args.comp_obj_min is not None else [None] * n
        self._box_max = list(args.comp_obj_max) if args.comp_obj_max is not None else [None] * n
        self.n_token, self.attn_sink = args.n_token, args.attn_sink

        # unified joint search space — options AUTO-DERIVED from the 1st-stage archives so
        # ss.encode() covers every arch (model/archive-general; no hardcoded grids).
        wbits, kvbits, gs_lists, kprune, vprune = derive_options(args.w_expr, args.eff_kv_expr)
        # QEFT-on-HQQ: if the W archive used a (bits, n_outlier) outlier-column axis, rebuild the
        # SAME ladder so ss.encode covers it (else W entries [bits,n_out] vs scalar options mismatch).
        n_qeft_column, qeft_bits = derive_qeft(args.w_expr, args.eff_kv_expr)
        self._uses_qeft = qeft_bits is not None
        self.wbits, self.kvbits = wbits, kvbits
        print(f"[options] w_bits={wbits} kv_bits={kvbits} gs={gs_lists} k_prune={kprune} v_prune={vprune}"
              + (f" | QEFT n_outlier={n_qeft_column} on bits={qeft_bits}" if qeft_bits else ""))
        self.ss = LlamaSearchSpace(
            bits={'w': wbits, 'k': kvbits, 'v': kvbits},
            group_size={'w': args.w_group_size, 'k': [list(g) for g in gs_lists], 'v': [list(g) for g in gs_lists]},
            pass_module={'w': [], 'k': [], 'v': []}, config=config,
            comp_obj=self.comp_obj, comp_obj_min=[0.0, 0.0], comp_obj_max=[1e9, 1e9],  # placeholder; real box set after pools (ss box only gates ss.sample, unused here)
            n_token=self.n_token, k_pruning_dim=kprune, v_pruning_dim=vprune,
            n_qeft_column=n_qeft_column, qeft_outlier_bits=qeft_bits)
        self.xu = encoding_xu(self.ss); self.nw = nw_split(self.ss); self.n_var = len(self.xu)
        self._comp = JointComp(self.ss)   # vectorized comp_obj (batch over genomes; skips decode+get_net_info)
        self.ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        self.segments = block_segments(self.ss)

        # 1st-stage block pools + per-gene lever weights
        self.Wg, self.KVg, _, _, w_comp, kv_comp = load_block_pools(
            args.w_expr, args.eff_kv_expr, self.ss, w_eps=args.w_front_eps, kv_eps=args.kv_front_eps,
            eps_rel=args.front_eps_rel, div_k=args.div_k, seed=args.seed)
        # AUTO-derive any unspecified budget-box edge from the pool's achievable comp range
        # (wbits = W-block only, eff_kvbits = KV-block only) so the box follows the input files.
        self.comp_obj_min = [self._box_min[i] if self._box_min[i] is not None else float(c.min())
                             for i, c in enumerate((w_comp, kv_comp))]
        self.comp_obj_max = [self._box_max[i] if self._box_max[i] is not None else float(c.max())
                             for i, c in enumerate((w_comp, kv_comp))]
        self.ss.comp_obj_min, self.ss.comp_obj_max = self.comp_obj_min, self.comp_obj_max
        box_src = 'auto' if all(v is None for v in self._box_min + self._box_max) else 'user/auto'
        print(f"[budget box] wbits [{self.comp_obj_min[0]:.3f}, {self.comp_obj_max[0]:.3f}] | "
              f"eff_kvbits [{self.comp_obj_min[1]:.3f}, {self.comp_obj_max[1]:.3f}]  ({box_src})")
        # filter pools to the comp budget box so DOE + search stay in-box (each axis filtered
        # independently → every W×KV combination stays feasible). No-op when the box is auto.
        wm = (w_comp >= self.comp_obj_min[0]) & (w_comp <= self.comp_obj_max[0])
        km = (kv_comp >= self.comp_obj_min[1]) & (kv_comp <= self.comp_obj_max[1])
        if wm.sum() < 5 or km.sum() < 5:
            raise SystemExit(f"[budget box] too few in-box blocks (W {wm.sum()}, KV {km.sum()}); "
                             f"widen --comp_obj_min/max or raise --front_eps_rel")
        if wm.sum() < len(self.Wg) or km.sum() < len(self.KVg):
            print(f"[budget box] W {len(self.Wg)}→{int(wm.sum())} | KV {len(self.KVg)}→{int(km.sum())}")
        self.Wg, self.KVg = self.Wg[wm], self.KVg[km]
        self.w_comp, self.kv_comp = w_comp[wm], kv_comp[km]   # in-box comp (for endpoint corners)
        self.w = gene_weights(self.Wg, self.KVg)
        # L2 direct space reduction: FREEZE genome cells the 1st-stage blocks agree on (modal
        # fraction ≥ --agree_frac) by zeroing their mutation lever weight → mutation only
        # explores the CONTESTED cells; sampling/crossover already keep agreed cells at the
        # pool consensus. New-arch exploration is thus confined to the uncertain dims (the
        # decided dims are frozen). Verified loss-free in tests/joint_reabsorption.py
        # (mut_contested == mut in HV). agree_frac > 1.0 disables (freezes nothing).
        self.contested, mf = freeze_mask(self.Wg, self.KVg, args.agree_frac)
        self.w = self.w * self.contested
        n_frozen = int((~self.contested).sum())
        print(f"[L2 freeze] agree_frac={args.agree_frac}: frozen/agreed cells {n_frozen}/{self.n_var} "
              f"→ free(contested) dims {int(self.contested.sum())} "
              f"(W {int(self.contested[:self.nw].sum())}/{self.nw}, "
              f"KV {int(self.contested[self.nw:].sum())}/{self.n_var-self.nw})")
        # predictor active set is recomputed data-driven per fit (_fit_predictor): cols that
        # vary in the current archive, capped below N for RBF (else singular → nan).
        self.active = np.where(self.w > 0.05)[0]
        print(f"[ss] n_var={self.n_var} (W-block {self.nw}, KV-block {self.n_var-self.nw}) | "
              f"W-blocks={len(self.Wg)} KV-blocks={len(self.KVg)} | "
              f"contested~{len(self.active)} levers(>0.5)={int((self.w>0.5).sum())}")

        self._build_evaluator(args)

    def _build_evaluator(self, args):
        import torch
        from evaluator import LlamaEvaluator
        from utils.func import init_accelerator, process_dtype
        self.accelerator, device_map = init_accelerator(args.gpu_id, self.config)
        # QEFT-on-HQQ: archs with n_outlier>0 need the multi-rank outlier dict so the evaluator
        # can insert the FP16 columns per-arch. Required whenever the W archive used outliers.
        outlier = torch.load(args.outlier_path) if getattr(args, 'outlier_path', '') else None
        if outlier is None and getattr(self, '_uses_qeft', False):
            raise SystemExit("[QEFT] the W archive uses outlier columns (n_outlier>0) but no "
                             "--outlier_path was given; pass the extract_outidx.py multi-rank dict.")
        self.evaluator = LlamaEvaluator(
            self.config, accelerator=self.accelerator, device_map=device_map,
            model_id=f'{args.model_path}/{args.model_name}',
            method={'w': args.w_method, 'kv': args.kv_method}, quant_model_paths=args.quant_model_paths,
            outlier=outlier,
            seqlen=args.seqlen, n_sample=args.n_sample, datasets=[args.dataset], dtype=process_dtype(args.dtype),
            bits={'w': self.wbits, 'k': self.kvbits, 'v': self.kvbits}, group_size=self.ss.group_size,
            residual_length=args.residual_length, attn_sink=args.attn_sink,
            k_quant_scheme=args.k_quant_scheme, v_quant_scheme=args.v_quant_scheme,
            loss_func=args.loss_func, last_tokens=args.last_tokens)

    # ───────────────── evaluation (real HQQ JSD via LlamaEvaluator.eval) ─────────────────
    def _evaluate(self, archs):
        metric, comp = [], []
        for arch in archs:
            ni = get_net_info(arch, self.ss.config, self.ss.group_size, n_token=self.n_token, attn_sink=self.attn_sink)
            comp.append([ni[o] for o in self.comp_obj])
            m, _ = self.evaluator.eval(self.accelerator, arch, 'loss', loss_func=self.args.loss_func,
                                       stride=self.args.stride, prefill_prompt=self.args.prefill_prompt)
            metric.append(float(list(m.values())[0]))
        return metric, comp

    # ───────────────── surrogate fit (arch input via encode_predictor, like search.py) ─────────────────
    def _fit_predictor(self, archive):
        Xf = np.array([self.ss.encode(x[0]) for x in archive], float)        # full (N, n_var) encoding
        targets = np.array([x[1] for x in archive])
        # active = cols that actually VARY in THIS data (constant-in-sample cols make the
        # RBF tail block singular → nan); for RBF cap dim below N (exact interpolation).
        var = Xf.var(0); active = np.where(var > 1e-9)[0]
        if self.predictor == 'rbf':
            # RBF interpolates exactly: it needs n_active ≤ N-1 (tail block ntail=n_active+1
            # must be ≤ N) else it asserts. Cap below N with a 30-pt margin, but never above
            # N-1 — the floor must not exceed the archive size (was max(10,…) → crashed for
            # tiny archives where 10 > N-1).
            cap = max(1, min(len(archive) - 1, len(archive) - 30))
            if len(active) > cap:
                active = np.sort(active[np.argsort(-var[active])][:cap])
        self.active = active                                                 # _next reads this (set before it)
        inputs = Xf[:, active]
        kwargs = {'lb': np.zeros(len(active)), 'ub': self.xu[active].astype(float)} if self.predictor == 'rbf' else {}
        # surrogate runs its linear algebra on GPU when available (rbf/gp/ard_gp/mlp are
        # torch-backed; device='auto'→cuda). Matters at large archive N (RBF fit is the N×N solve).
        if not self._pred_dev_logged:
            import torch
            res = 'cuda' if (self._pred_device in ('auto', 'cuda') and torch.cuda.is_available()) else \
                  ('cpu' if self._pred_device in ('auto', 'cuda') else self._pred_device)
            print(f"[predictor] {self.predictor} on {res} (requested '{self._pred_device}')")
            self._pred_dev_logged = True
        pred = get_predictor(self.predictor, inputs, targets, device=self._pred_device, **kwargs)
        return pred, pred.predict(inputs)

    # ───────────────── candidate generation (NSGA-III + joint operators) ─────────────────
    def _next(self, archive, predictor, K):
        problem = JointAuxProblem(self.ss, predictor, self.active, self.xu, self.comp_obj,
                                  self.comp_obj_min, self.comp_obj_max, self.n_token, self.attn_sink,
                                  comp=self._comp)
        if self.args.mutation == 'knowledge':
            mut = KnowledgeMutation(self.w, self.xu, self.Wg, self.KVg, self.nw, self.segments,
                                    p_val=self.args.mut_p_val, p_mod=self.args.mut_p_mod)
        else:
            mut = WeightedIntMutation(self.w, self.xu)
        algo = NSGA3(pop_size=self.ga_pop_size, ref_dirs=self.ref_dirs,
                     sampling=FrontierProductSampling(self.Wg, self.KVg),
                     crossover=AxisBlockCrossover(self.nw, self.n_var),
                     mutation=mut, eliminate_duplicates=False)
        res = minimize(problem, algo, ('n_gen', 20), seed=self.args.seed, save_history=True, verbose=True)
        seen = {tuple(self.ss.encode(x[0])) for x in archive}
        Xc = np.unique(np.clip(np.round(res.pop.get('X')), 0, self.xu).astype(int), axis=0)
        Xc = np.array([g for g in Xc if tuple(g) not in seen])
        if len(Xc) == 0:
            return [], np.zeros((0,))
        if len(Xc) > K:                                  # diversity-select K across comp space (maximin)
            comp = self._comp.batch(Xc, self.comp_obj)   # vectorized (was per-genome get_net_info)
            z = (comp - comp.mean(0)) / (comp.std(0) + 1e-9)
            Xc = Xc[maximin_extras(z, anchor_idx=[], K=K, seed=self.args.seed)]
        cands = [self.ss.decode(g) for g in Xc]
        return cands, predictor.predict(Xc[:, self.active].astype(float))

    def _front_coverage(self, archive):
        F = np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)])
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        cov = {}
        for j, obj in enumerate(self.comp_obj):
            col, full = F[nd, 1 + j], F[:, 1 + j]
            denom = full.max() - full.min()
            cov[obj] = {'front_min': float(col.min()), 'front_max': float(col.max()),
                        'full_min': float(full.min()), 'full_max': float(full.max()),
                        'coverage': float((col.max() - col.min()) / denom) if denom > 0 else 1.0}
        return cov

    @staticmethod
    def _calc_hv(ref_pt, F):
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        rp = 1.01 * ref_pt
        return float(Hypervolume(ref_point=rp).do(F[nd]) / np.prod(rp))

    def _save_viz(self, it, archive, c_metric, c_pred, c_comp, cov):
        """Per-save_iter figures (search.py --debug style + a joint panel):
          - one panel PER comp_obj (wbits, eff_kvbits): loss vs that comp axis — archive (blue)
            + 1st Pareto-front line (black) + this iter's candidates (evaluated red / predicted
            green), x fixed to the budget box so iters compare;
          - a final JOINT panel: wbits × eff_kvbits scatter coloured by loss with the non-
            dominated front ringed (the W×KV tradeoff the per-axis panels can't show)."""
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[viz] skipped (matplotlib unavailable: {e})"); return
        plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12.5, 'axes.titleweight': 'semibold',
                             'axes.edgecolor': '#444', 'axes.linewidth': 0.9, 'figure.facecolor': 'white',
                             'axes.facecolor': '#fbfbfd', 'legend.framealpha': 0.92})
        eps = 1e-4
        n_obj = len(self.comp_obj)
        comp_c = np.array(c_comp); c_pred = np.clip(np.asarray(c_pred).ravel(), eps, None)
        c_metric = np.clip(np.asarray(c_metric), eps, None)
        perf = np.clip(np.array([x[1] for x in archive]), eps, None)
        arch_F = np.column_stack([[x[k] for x in archive] for k in range(1, n_obj + 2)])
        nd_idx = NonDominatedSorting().do(arch_F, only_non_dominated_front=True)
        joint = (n_obj == 2)
        ncol = n_obj + (1 if joint else 0)
        fig, axes = plt.subplots(1, ncol, figsize=(5.7 * ncol, 5.3), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for i in range(n_obj):
            ax = axes[i]; obj = self.comp_obj[i]
            comp = np.array([x[i + 2] for x in archive])
            # archive cloud: light, soft, rasterized → no black overplot mush
            ax.scatter(comp, perf, s=9, c='#9fb3d1', alpha=0.35, edgecolors='none',
                       rasterized=True, zorder=1, label=f'archive (n={len(archive)})')
            # achievable frontier = best loss within budget (running min of loss as comp grows)
            o = np.argsort(comp); env = np.minimum.accumulate(perf[o])
            ax.plot(comp[o], env, color='#1f4e8c', lw=2.4, zorder=3, label='best-loss envelope')
            # this iter's candidates: measured vs predicted (surrogate accuracy at a glance)
            ax.scatter(comp_c[:, i], c_metric, s=42, marker='o', c='#2ca02c', edgecolors='white',
                       lw=0.6, zorder=6, label='cand · measured')
            ax.scatter(comp_c[:, i], c_pred, s=46, marker='x', c='#ff7f0e', lw=1.5,
                       zorder=5, label='cand · predicted')
            fmin, fmax = self.comp_obj_min[i], self.comp_obj_max[i]
            pad = 0.03 * (fmax - fmin) if fmax > fmin else 0.1
            ax.set_xlim(fmin - pad, fmax + pad)
            ax.axvspan(fmin, fmax, color='#000000', alpha=0.025, zorder=0)
            ax.set_xlabel(obj); ax.grid(True, which='both', ls=':', c='0.8', lw=0.6); ax.set_axisbelow(True)
            ax.set_title(f"{obj}  ·  front-coverage {cov[obj]['coverage'] * 100:.0f}%")
        axes[0].set_ylabel('loss')
        axes[0].legend(loc='upper right', fontsize=8.5, ncol=1)
        if joint:                                         # joint W × eff_kvbits loss landscape
            ax = axes[n_obj]
            wx = np.array([x[2] for x in archive]); ky = np.array([x[3] for x in archive])
            # every arch as a point coloured by its loss (linear) — the actual sampled landscape;
            # this iter's measured candidates ringed so you can see where the search just probed.
            sc = ax.scatter(wx, ky, c=perf, s=16, cmap='viridis', alpha=0.7,
                            edgecolors='none', rasterized=True, zorder=1)
            ax.scatter(comp_c[:, 0], comp_c[:, 1], s=46, marker='o', facecolors='none',
                       edgecolors='#d62728', lw=1.4, zorder=4, label='cand · this iter')
            ax.set_xlabel(self.comp_obj[0]); ax.set_ylabel(self.comp_obj[1])
            ax.set_xlim(self.comp_obj_min[0], self.comp_obj_max[0])
            ax.set_ylim(self.comp_obj_min[1], self.comp_obj_max[1])
            ax.set_title(f'joint  {self.comp_obj[0]} × {self.comp_obj[1]}  ·  colour = loss')
            ax.legend(loc='upper right', fontsize=9); ax.grid(True, ls=':', c='0.8', lw=0.6); ax.set_axisbelow(True)
            fig.colorbar(sc, ax=ax, label='loss', shrink=0.92)
        fig.suptitle(f'2nd-stage joint W×eff_kvbits search — iter {it}', fontsize=13.5, fontweight='bold')
        fig.savefig(os.path.join(self.save_path, f'iter_{it}.png'), dpi=130)
        plt.close(fig)

    def _initialize(self):
        """DOE seed, mirroring search_space.llama.initialize: enumerate the BOUNDARY corners
        first, then fill the remaining budget with random samples → n_doe total. Here the
        boundary corners are the {W min/max comp} × {KV min/max comp} block combos (both edges
        of both comp axes), always included by default; the random fill is the front-block
        product. Returns (archs, n_corner, n_random)."""
        ws = sorted({int(np.argmin(self.w_comp)), int(np.argmax(self.w_comp))})   # W block extremes
        ks = sorted({int(np.argmin(self.kv_comp)), int(np.argmax(self.kv_comp))}) # KV block extremes
        corners = [np.concatenate([self.Wg[wi], self.KVg[ki]]) for wi in ws for ki in ks]
        rand = FrontierProductSampling(self.Wg, self.KVg)._do(None, max(self.n_doe - len(corners), 0))
        doe_g = np.concatenate([np.array(corners), rand], axis=0) if len(rand) else np.array(corners)
        return [self.ss.decode(g) for g in doe_g], len(corners), len(rand)

    # ───────────────── main loop (mirrors search.py::Search.search) ─────────────────
    def search(self):
        t0 = time(); start_it = 1
        if self.args.resume:                                  # resume from an iter_<it>.stats
            rf = json.load(open(self.args.resume))
            archive = rf['archive']; start_it = rf['iteration'] + 1
            print(f"[resume] {len(archive)} archs from iter {rf['iteration']} → start iter {start_it}")
        else:
            archs, n_corner, n_rand = self._initialize()
            metric, comp = self._evaluate(archs)
            archive = [[a, m, *c] for a, m, c in zip(archs, metric, comp)]
            print(f"[DOE] {len(archive)} archs ({n_corner} budget corners + {n_rand} random)  "
                  f"loss {min(metric):.4f}-{max(metric):.4f}  ({time()-t0:.1f}s)")
        ref_pt = np.array([np.max([x[i] for x in archive]) for i in range(1, len(self.comp_obj) + 2)])
        print(f'data preparation time : {time() - t0:.2f}s')

        for it in range(start_it, self.iterations + 1):
            iter_start = time()
            # construct accuracy predictor surrogate model from archive
            predictor_start = time()
            pred, a_pred = self._fit_predictor(archive)
            predictor_time = time() - predictor_start
            # search for the next set of candidates for high-fidelity evaluation
            next_start = time()
            cands, c_pred = self._next(archive, pred, self.n_iter)
            next_time = time() - next_start
            if not cands:
                print(f"Iter {it}: no new candidates; stop"); break
            c_metric, c_comp = self._evaluate(cands)
            # check accuracy predictor's performance
            rmse, rho, tau = get_correlation(
                np.concatenate([np.asarray(a_pred).ravel(), np.asarray(c_pred).ravel()]),
                np.array([x[1] for x in archive] + c_metric))
            for a, m, c in zip(cands, c_metric, c_comp):
                archive.append([a, m, *c])
            F = np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)])
            hv = self._calc_hv(ref_pt, F); cov = self._front_coverage(archive)
            iter_time = time() - iter_start
            # print iteration-wise statistics (search.py format)
            print(f"Iter {it}: hv = {hv:.2f}, iter time : {iter_time:.2f}s, "
                  f"predictor_time : {predictor_time:.2f}, next_time : {next_time:.2f}")
            print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall's Tau = {tau:.4f}")
            for obj in self.comp_obj:
                c = cov[obj]
                print(f"  {obj} front-coverage : {c['coverage']*100:.1f}%  "
                      f"front=[{c['front_min']:.3f}, {c['front_max']:.3f}] / "
                      f"full=[{c['full_min']:.3f}, {c['full_max']:.3f}]")
            print(f'iteration time : {iter_time:.2f}s')
            # dump stats + per-save_iter visualization (also always on the final iter)
            if it % self.save_iter == 0 or it == self.iterations:
                os.makedirs(self.save_path, exist_ok=True)
                with open(os.path.join(self.save_path, f"iter_{it}.stats"), 'w') as f:
                    json.dump({'archive': archive, 'candidates': archive[-self.n_iter:], 'hv': hv,
                               'surrogate': {'model': self.predictor, 'rmse': rmse, 'rho': rho, 'tau': tau},
                               'coverage': cov, 'iteration': it}, f)
                if self.debug:
                    self._save_viz(it, archive, c_metric, c_pred, c_comp, cov)
        print(f"[done] {len(archive)} archs, {time()-t0:.1f}s → {self.save_path}")
        return archive


def build_parser():
    p = argparse.ArgumentParser(description="2nd-stage joint W×eff_kvbits NAS (HQQ, NSGA-III, LlamaSearchSpace)")
    p.add_argument('--config', default='config/llama.json'); p.add_argument('--model_name', default='Llama-3.1-8B-Instruct')
    p.add_argument('--w_expr', required=True, help='1st-stage W-axis archive dir or iter_N.stats')
    p.add_argument('--eff_kv_expr', required=True, help='1st-stage eff_kvbits archive dir or iter_N.stats')
    p.add_argument('--surrogate', default='rbf', help='arch-input predictor (rbf/gp/ard_gp/carts)')
    p.add_argument('--predictor_device', default='auto', help="surrogate compute device: 'auto' (cuda if available) / 'cuda' / 'cuda:N' / 'cpu'")
    p.add_argument('--iterations', type=int, default=8); p.add_argument('--n_doe', type=int, default=512)
    p.add_argument('--n_iter', type=int, default=60); p.add_argument('--pop', type=int, default=92)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--save', default='save/second_search/run')
    p.add_argument('--resume', default=None, help='path to an iter_<it>.stats to resume from')
    p.add_argument('--save_iter', type=int, default=1, help='dump iter_<it>.stats (+ viz if --debug) every save_iter iters (and on the last)')
    p.add_argument('--debug', action='store_true', help='also save per-save_iter scatter plots iter_<it>.png (like search.py --debug)')
    # comp objectives + budget box (taken at once, like search.py). comp_obj_min/max default
    # None per edge → AUTO-derived from the input archives' achievable comp range; pass an
    # explicit value only to NARROW an axis. comp_obj[0]=W axis, comp_obj[1]=KV axis.
    p.add_argument('--comp_obj', nargs='+', default=['wbits', 'eff_kvbits'])
    p.add_argument('--comp_obj_min', nargs='+', type=float, default=None)
    p.add_argument('--comp_obj_max', nargs='+', type=float, default=None)
    p.add_argument('--n_token', type=int, default=0); p.add_argument('--attn_sink', type=int, default=8)
    # building-block selector from 1st-stage archives: adaptive-ε band → structural-diversity
    p.add_argument('--w_front_eps', type=float, default=0.0); p.add_argument('--kv_front_eps', type=float, default=0.0)
    p.add_argument('--front_eps_rel', type=float, default=0.0, help='relative ε band = front_jsd·(1+rel); scale-free, auto-wider in the high-loss corner')
    p.add_argument('--div_k', type=int, default=0, help='keep div_k structurally-diverse blocks per axis (maximin; 0=off)')
    p.add_argument('--agree_frac', type=float, default=0.95, help='L2 freeze: cells where ≥ this fraction of 1st-stage blocks agree are frozen at consensus (mutation skips them); >1.0 disables. 0.95 is loss-free per tests/joint_reabsorption.py')
    # mutation: 1st-stage-knowledge-guided (value-resample + module-transplant) vs lever-weighted ±1
    p.add_argument('--mutation', default='knowledge', choices=['knowledge', 'weighted'])
    p.add_argument('--mut_p_val', type=float, default=0.5, help='prob a mutated cell takes a 1st-stage value (else ±1)')
    p.add_argument('--mut_p_mod', type=float, default=0.15, help='prob/indiv of a 1st-stage module-transplant')
    # search-space options (must cover both 1st-stage archives)
    p.add_argument('--w_bits', type=int, nargs='+', default=[2, 3, 4]); p.add_argument('--k_bits', type=int, nargs='+', default=[2, 3, 4])
    p.add_argument('--v_bits', type=int, nargs='+', default=[2, 3, 4]); p.add_argument('--w_group_size', type=int, default=128)
    p.add_argument('--k_pruning_dim', type=int, nargs='+', default=[0, 16, 32, 48, 64])
    p.add_argument('--v_pruning_dim', type=int, nargs='+', default=[0, 16, 32, 48, 64])
    # hqq-mode model/quant args (mirror search.py / evaluator.py)
    p.add_argument('--gpu_id', default='0'); p.add_argument('--model_path', default='/SSD/huggingface/meta-llama')
    p.add_argument('--dtype', default='bfloat16'); p.add_argument('--w_method', nargs='+', default=['hqq'])
    p.add_argument('--kv_method', nargs='+', default=['kivi', 'think']); p.add_argument('--quant_model_paths', nargs='+', default=[])
    p.add_argument('--outlier_path', default='', help='QEFT-on-HQQ: extract_outidx.py multi-rank outlier dict {key:{n_out:[idx]}}; required when the W archive used n_outlier>0 (auto-detected from the archive)')
    p.add_argument('--residual_length', type=int, default=128); p.add_argument('--k_quant_scheme', default='channel')
    p.add_argument('--v_quant_scheme', default='token'); p.add_argument('--dataset', default='wikitext2')
    p.add_argument('--n_sample', type=int, default=128); p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--loss_func', default='jsd'); p.add_argument('--stride', type=int, default=128)
    p.add_argument('--prefill_prompt', action='store_true'); p.add_argument('--last_tokens', type=int, default=512)
    return p


def main(args):
    set_seed(args.seed)
    config = json.load(open(args.config))[args.model_name]
    SecondSearch(config, args).search()


if __name__ == '__main__':
    main(build_parser().parse_args())
