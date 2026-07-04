"""second_search.py — 2nd-stage JOINT (W × eff_kvbits) NAS, HQQ-based, NSGA-III.

Structure mirrors search.py (Search): DOE → _fit_predictor → _next → _evaluate →
iter_<it>.stats loop; archive = [arch, metric, *comp_obj]. Uses the SAME building
blocks as search.py — LlamaSearchSpace (encode/decode/encode_predictor) and
LlamaEvaluator (eval). Joint-specific operators + auxiliary helpers (supply seeding,
coverage/HV stats, per-iter viz) live in utils/second_stage.py; generic candidate
subset selectors (even_select / moo_subset_select) in utils/select.py.

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

from search_space.llama import LlamaSearchSpace
from predictor.factory import get_predictor
from utils.func import set_seed, get_correlation, get_net_info
from utils.select import maximin_extras, even_select, moo_subset_select
from utils.second_stage import (
    encoding_xu, nw_split, load_block_pools, load_band_blocks, load_extra_parts,
    gene_weights, freeze_mask, block_segments, derive_options, derive_qeft,
    FrontierProductSampling, AxisBlockCrossover, WeightedIntMutation, KnowledgeMutation, JointAuxProblem, JointComp,
    grid_side, grid_seed, calc_hv, front_coverage, save_viz)


class SecondSearch:
    """Joint 2nd-stage search — same loop as search.py::Search; joint operators."""
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.save_path = args.save
        self.result_file = getattr(args, 'result_file', 'results.txt')
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
        self._uses_qeft = qeft_bits is not None; self._n_qeft_column = n_qeft_column
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
        # seeding pools beyond the div_k-pruned crossover pools (--seed_pool full / --seed_expr):
        # full ε-band blocks (densest nearest-comp matching) + extra archives' W/KV parts.
        self.Wseed = self.KVseed = None
        if args.seed_pool == 'full':
            Wb, wbc, KVb, kbc = load_band_blocks(args.w_expr, args.eff_kv_expr, self.ss,
                                                 w_eps=args.w_front_eps, kv_eps=args.kv_front_eps,
                                                 eps_rel=args.front_eps_rel)
            wm2 = (wbc >= self.comp_obj_min[0]) & (wbc <= self.comp_obj_max[0])
            km2 = (kbc >= self.comp_obj_min[1]) & (kbc <= self.comp_obj_max[1])
            self.Wseed, self.wseed_comp = Wb[wm2], wbc[wm2]
            self.KVseed, self.kvseed_comp = KVb[km2], kbc[km2]
            print(f"[seed_pool full] band blocks W {len(self.Wseed)} | KV {len(self.KVseed)} (in-box)")
        if args.seed_expr:
            xW, xwc, xK, xkc, n_skip = load_extra_parts(args.seed_expr, self.ss, self._comp)
            bW, bwc = (self.Wseed, self.wseed_comp) if self.Wseed is not None else (self.Wg, self.w_comp)
            bK, bkc = (self.KVseed, self.kvseed_comp) if self.KVseed is not None else (self.KVg, self.kv_comp)
            self.Wseed = np.vstack([bW, xW]); self.wseed_comp = np.concatenate([bwc, xwc])
            self.KVseed = np.vstack([bK, xK]); self.kvseed_comp = np.concatenate([bkc, xkc])
            print(f"[seed_expr] +{len(xW)} W / +{len(xK)} KV parts from {len(args.seed_expr)} "
                  f"archives (skipped {n_skip} non-encodable)")
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
        outlier = None
        if getattr(args, 'outlier_path', ''):
            outlier = torch.load(args.outlier_path); print(f"[QEFT] outlier dict: {args.outlier_path}")
        elif getattr(self, '_uses_qeft', False):
            # auto-locate the extract_outidx.py dict under the repo's outlier/ tree (path scheme
            # mirrors search.sh: outlier/<model>/w16_r<ranks>_<dataset>/outlier.pth). Resolved
            # relative to this file so it works regardless of cwd.
            ranks = '_'.join(str(c) for c in self._n_qeft_column if c > 0)
            root = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.join(root, 'outlier', args.model_name, f'w16_r{ranks}_{args.dataset}', 'outlier.pth')
            if os.path.isfile(cand):
                outlier = torch.load(cand); print(f"[QEFT] auto-loaded outlier dict: {cand}")
            else:
                raise SystemExit(f"[QEFT] W archive uses outlier columns (n_outlier={self._n_qeft_column}) "
                                 f"but no --outlier_path given and none at {cand}; pass the "
                                 f"extract_outidx.py multi-rank dict via --outlier_path.")
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

    # ───────────────── incremental encode cache (append-only archive) ─────────────────
    def _encode_archive(self, archive):
        """ss.encode() with an append-only cache: the archive only GROWS (the loop appends, never
        reorders), so encode ONLY the new tail each iter instead of re-encoding all N. Encoding is
        ~3ms/arch pure-python and was run over the FULL archive twice per iter (_fit_predictor +
        _next) — the dominant per-iter cost at large N (up to ~31s at N=10.5k). Returns the int
        (N, n_var) genome matrix (callers cast to float where needed)."""
        cache = getattr(self, '_enc_cache', None)
        n = len(archive)
        if cache is None or cache.shape[0] > n:              # first call / resume / (defensive) shrink
            self._enc_cache = np.array([self.ss.encode(x[0]) for x in archive]) if n else np.empty((0, self.n_var), int)
        elif cache.shape[0] < n:                             # append only the freshly-added tail
            new = np.array([self.ss.encode(archive[i][0]) for i in range(cache.shape[0], n)])
            self._enc_cache = np.vstack([cache, new])
        return self._enc_cache

    # ───────────────── surrogate fit (arch input via encode_predictor, like search.py) ─────────────────
    def _fit_predictor(self, archive):
        Xf = self._encode_archive(archive).astype(float)                     # full (N, n_var) encoding (cached)
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
        # surrogate device: DEFAULT is CPU now ('auto'→cpu in predictor/factory._resolve_device) —
        # GPU cuSOLVER returns garbage on the ill-conditioned RBF saddle system. Pass
        # --predictor_device cuda to opt into GPU (guarded by RBF's CPU-lstsq fallback).
        if not self._pred_dev_logged:
            res = 'cpu' if self._pred_device in ('auto', None) else self._pred_device
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
        Ga = self._encode_archive(archive)                          # archive genomes (cached; reused below)
        seen = {tuple(gg) for gg in Ga}
        g = grid_side(K, self.args.cand_grid)
        pool = res.pop.get('X')
        if self.args.grid_seed:                          # even SUPPLY: block-product seeds per box cell
            # base = div_k crossover pools ('first') or the denser seed pools ('full'/--seed_expr)
            if self.Wseed is not None:
                Wp, wc, KVp, kc = self.Wseed, self.wseed_comp, self.KVseed, self.kvseed_comp
            else:
                Wp, wc, KVp, kc = self.Wg, self.w_comp, self.KVg, self.kv_comp
            if self.args.seed_pool != 'first':
                # augment with the ARCHIVE's W/KV sub-blocks (incl. 2nd-stage mutants) so seeds
                # improve as the search discovers better parts; per-part comp comes free from
                # the stored archive comp columns.
                aw = np.array([x[2] for x in archive]); ak = np.array([x[3] for x in archive])
                uW, iW = np.unique(Ga[:, :self.nw], axis=0, return_index=True)
                uK, iK = np.unique(Ga[:, self.nw:], axis=0, return_index=True)
                Wp = np.vstack([Wp, uW]); wc = np.concatenate([wc, aw[iW]])
                KVp = np.vstack([KVp, uK]); kc = np.concatenate([kc, ak[iK]])
            pool = np.vstack([pool, grid_seed(Wp, KVp, wc, kc,
                                              self.comp_obj_min, self.comp_obj_max, g)])
        Xc = np.unique(np.clip(np.round(pool), 0, self.xu).astype(int), axis=0)
        Xc = np.array([gg for gg in Xc if tuple(gg) not in seen])
        if len(Xc) == 0:
            return [], np.zeros((0,))
        pred = np.asarray(predictor.predict(Xc[:, self.active].astype(float))).ravel()
        # optional ACTIVE-LEARNING quota (--al_frac): reserve part of K for the candidates
        # FARTHEST (standardized active-dims distance) from every measured arch — a model-free
        # uncertainty proxy (safe with rbf: no jackknife, no uncertainty-AL clustering pathology).
        # Default 0: in-house evidence says AL payoff is tail-only and cov_rad already explores.
        n_al = int(round(self.args.al_frac * K)) if len(Xc) > K else 0
        K_sel = K - n_al
        if len(Xc) > K:
            comp = self._comp.batch(Xc, self.comp_obj)   # (N,2) [wbits, eff_kvbits], vectorized
            if self.args.cand_even == 'maximin':         # extent coverage (legacy default)
                z = (comp - comp.mean(0)) / (comp.std(0) + 1e-9)
                idx = np.asarray(maximin_extras(z, anchor_idx=[], K=K_sel, seed=self.args.seed), int)
            elif self.args.cand_even == 'grid':          # per-axis-even quota over the box
                idx = even_select(comp, pred, K_sel, g, self.comp_obj_min, self.comp_obj_max)
            elif self.args.cand_even == 'moo':           # 2/3-obj (loss × coverage [× gap-std]) → knee
                idx = moo_subset_select(comp, pred, K_sel, self.comp_obj_min, self.comp_obj_max, g,
                                        algo=self.args.moo_algo, pop=self.args.moo_pop,
                                        n_gen=self.args.moo_gen, seed=self.args.seed,
                                        gap_std=self.args.moo_gap_std,
                                        coverage=self.args.moo_coverage)
            else:                                        # hybrid: front pressure + grid-even coverage
                k_even = int(round(self.args.even_frac * K_sel)); k_front = K_sel - k_even
                front = np.argsort(pred)[:k_front]       # low predicted loss = keep the front moving
                rest = np.setdiff1d(np.arange(len(Xc)), front)
                even = (rest[even_select(comp[rest], pred[rest], k_even, g,
                                         self.comp_obj_min, self.comp_obj_max)]
                        if len(rest) else np.array([], int))
                idx = np.concatenate([front, even]).astype(int)
            if n_al > 0:                                 # AL quota: farthest-from-archive extras
                rest = np.setdiff1d(np.arange(len(Xc)), idx)
                if len(rest):
                    act = self.active
                    mu = Ga[:, act].mean(0); sd = Ga[:, act].std(0); sd[sd < 1e-9] = 1.0
                    A = (Ga[:, act] - mu) / sd
                    Rz = (Xc[rest][:, act] - mu) / sd
                    from scipy.spatial.distance import cdist
                    dmin = cdist(Rz, A).min(1)
                    idx = np.concatenate([idx, rest[np.argsort(-dmin)[:n_al]]]).astype(int)
            Xc, pred = Xc[idx], pred[idx]
        cands = [self.ss.decode(gg) for gg in Xc]
        return cands, pred

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
            hv = calc_hv(ref_pt, F); cov = front_coverage(archive, self.comp_obj)
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
                    save_viz(self.save_path, it, archive, c_metric, c_pred, c_comp, cov,
                             self.comp_obj, self.comp_obj_min, self.comp_obj_max)
        print(f"[done] {len(archive)} archs, {time()-t0:.1f}s → {self.save_path}")
        self._write_results(archive, time() - t0)
        return archive

    def _write_results(self, archive, total_time):
        """Final run summary → <save>/<result_file> (mirrors search.py::search): all args +
        total time, plus a search summary (archive/front size, best loss, per-obj coverage)."""
        os.makedirs(self.save_path, exist_ok=True)
        losses = [x[1] for x in archive]
        F = np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)])
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        cov = front_coverage(archive, self.comp_obj)
        lines = [f"{k}: {v}\n" for k, v in vars(self.args).items()]
        lines.append(f"\nTotal time: {total_time:.2f}s\n")
        lines.append(f"archive: {len(archive)} archs | front: {len(nd)} | "
                     f"loss {min(losses):.4f}-{max(losses):.4f} (best {min(losses):.4f})\n")
        for obj in self.comp_obj:
            c = cov[obj]
            lines.append(f"  {obj}: front=[{c['front_min']:.3f}, {c['front_max']:.3f}] "
                         f"full=[{c['full_min']:.3f}, {c['full_max']:.3f}] coverage {c['coverage']*100:.1f}%\n")
        with open(os.path.join(self.save_path, self.result_file), 'w') as f:
            f.writelines(lines)
        print(f"[results] {os.path.join(self.save_path, self.result_file)}")


def build_parser():
    p = argparse.ArgumentParser(description="2nd-stage joint W×eff_kvbits NAS (HQQ, NSGA-III, LlamaSearchSpace)")
    p.add_argument('--config', default='config/llama.json'); p.add_argument('--model_name', default='Llama-3.1-8B-Instruct')
    p.add_argument('--w_expr', required=True, help='1st-stage W-axis archive dir or iter_N.stats')
    p.add_argument('--eff_kv_expr', required=True, help='1st-stage eff_kvbits archive dir or iter_N.stats')
    p.add_argument('--surrogate', default='rbf', help='arch-input predictor (rbf/gp/ard_gp/carts)')
    p.add_argument('--predictor_device', default='auto', help="surrogate compute device: 'auto' (=cpu; GPU cuSOLVER is unreliable on the RBF saddle system) / 'cuda' / 'cuda:N' / 'cpu'")
    p.add_argument('--iterations', type=int, default=8); p.add_argument('--n_doe', type=int, default=512)
    p.add_argument('--n_iter', type=int, default=60); p.add_argument('--pop', type=int, default=92)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--save', default='save/second_search/run')
    p.add_argument('--resume', default=None, help='path to an iter_<it>.stats to resume from')
    p.add_argument('--save_iter', type=int, default=1, help='dump iter_<it>.stats (+ viz if --debug) every save_iter iters (and on the last)')
    p.add_argument('--result_file', default='results.txt', help='filename (under --save) for the final run summary (args + total time + search summary), mirrors search.py')
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
    # per-iteration candidate down-select across the (wbits × eff_kvbits) budget box.
    # maximin = extent coverage (legacy). grid = per-axis-even quota (bucket the box into a
    # g×g grid, round-robin cells, best predicted loss within a cell). hybrid = even_frac of K
    # on grid-even coverage + the rest on lowest-predicted-loss front pressure. moo = 2-obj
    # (mean pred-loss × box covering-radius) subset GA → knee (principled explore↔exploit
    # balance, dominates hybrid's hard split; --moo_algo nsga3 ≥ nsga2). --grid_seed additionally
    # injects nearest-block genomes per cell so the high-comp corner NSGA drops (right-end
    # collapse) still gets sampled — pair it with grid/hybrid/moo. Defaults keep legacy behaviour.
    p.add_argument('--cand_even', default='maximin', choices=['maximin', 'grid', 'hybrid', 'moo'])
    p.add_argument('--cand_grid', type=int, default=0, help='grid side for grid/hybrid/moo even-select (0=auto=ceil(sqrt(n_iter)))')
    p.add_argument('--even_frac', type=float, default=0.5, help='hybrid: fraction of K spent on grid-even coverage (rest = front pressure)')
    p.add_argument('--grid_seed', action='store_true', help='seed nearest-block genomes per box cell into the candidate pool each iter (guarantees high-comp supply)')
    p.add_argument('--seed_pool', default='archive', choices=['first', 'archive', 'full'],
                   help="block source for --grid_seed: 'first' = div_k-pruned 1st-stage pools; "
                        "'archive' = + the current archive's W/KV sub-blocks (2nd-stage mutants) "
                        "— seeds adapt as the search finds better parts; 'full' = + the FULL "
                        "ε-band 1st-stage blocks (no div_k pruning — densest nearest-comp match)")
    p.add_argument('--seed_expr', nargs='*', default=[],
                   help='extra iter_N.stats archives (e.g. prior 2nd-stage runs) whose W/KV '
                        'sub-blocks join the seed pool (QEFT [b,0] entries auto-normalized; '
                        'n_outlier>0 archs skipped for a scalar-bit space)')
    p.add_argument('--al_frac', type=float, default=0.0,
                   help='fraction of K reserved for ACTIVE-LEARNING picks = candidates farthest '
                        '(standardized active-dims distance) from all measured archs; model-free '
                        'uncertainty proxy. 0 = off (in-house evidence: AL payoff is tail-only)')
    p.add_argument('--moo_algo', default='nsga3', choices=['nsga3', 'nsga2'], help='moo down-select MOO solver (nsga3 = reference-direction, recommended)')
    p.add_argument('--moo_pop', type=int, default=80, help='moo down-select subset-GA population')
    p.add_argument('--moo_gen', type=int, default=80, help='moo down-select subset-GA generations')
    p.add_argument('--moo_coverage', default='rad', choices=['rad', 'gap'],
                   help="moo 2nd objective: 'rad' = box covering radius (2D reach), "
                        "'gap' = max-axis spacing gap-std (1D marginal evenness)")
    p.add_argument('--moo_gap_std', action='store_true',
                   help='moo: add a 3rd objective = max-axis std of consecutive sorted-coordinate '
                        'gaps of the subset (per-axis spacing evenness, as in '
                        "coverage_subset_nsga2_extras marginal/max)")
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
