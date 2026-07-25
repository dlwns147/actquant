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
from utils.select import subset_select
from utils.second_stage import (
    encoding_xu, nw_split, load_block_pools, load_band_blocks,
    gene_weights, freeze_mask, block_segments, derive_options, derive_qeft,
    FrontierProductSampling, AxisBlockCrossover, KnowledgeMutation, JointAuxProblem, JointComp,
    ArchFeatures, BandTable, grid_side, stair_seed, calc_hv, front_coverage, save_viz)


class SecondSearch:
    """Joint 2nd-stage search — same loop as search.py::Search; joint operators.

    __init__ is a 4-step orchestrator; each step is its own method:
      _build_space      ss + encodings, options auto-derived from the 1st-stage archives
      _load_pools       crossover block pools + budget box + denser seed pools
      _build_knowledge  lever weights, L2 freeze, P1/P2 BandTable (+corner genomes)
      _build_evaluator  LlamaEvaluator (or the --eval_workers arch-parallel pool)
    """
    def __init__(self, config, args):
        self.config, self.args = config, args
        from utils.func import init_accelerator
        self.accelerator, self._device_map = init_accelerator(args.gpu_id, config)  # early: enables accelerator.print
        self.save_path = args.save
        self.result_file = getattr(args, 'result_file', 'results.txt')
        self.iterations, self.n_doe, self.n_iter = args.iterations, args.n_doe, args.n_iter
        self.predictor, self.ga_pop_size = args.surrogate, args.pop
        self._pred_device = getattr(args, 'predictor_device', 'auto'); self._pred_dev_logged = False
        self.save_iter, self.debug = max(1, args.save_iter), args.debug
        self.comp_obj = list(args.comp_obj)      # comp_obj[0] = W axis, comp_obj[1] = KV axis
        self.n_token, self.attn_sink = args.n_token, args.attn_sink
        self._build_space(args)
        self._load_pools(args)
        self._build_knowledge(args)
        self._build_evaluator(args)

    def _build_space(self, args):
        """unified joint search space — options AUTO-DERIVED from the 1st-stage archives so
        ss.encode() covers every arch (model/archive-general; no hardcoded grids). QEFT-on-HQQ:
        a W archive with (bits, n_outlier) entries rebuilds the SAME option ladder."""
        wbits, kvbits, gs_lists, kprune, vprune = derive_options(args.w_expr, args.eff_kv_expr)
        # --w_bits is a CONSISTENCY CHECK, not a control: the archive decides the options;
        # this guards the silent mismatch where the HQQ banks passed via --quant_model_paths
        # (built from the script's W_BITS) don't cover the archive's bit ladder.
        if args.w_bits and sorted(args.w_bits) != wbits:
            raise SystemExit(f"[options] --w_bits {sorted(args.w_bits)} != archive-derived {wbits}; "
                             f"the archives decide the search space — fix W_BITS / the bank list.")
        n_qeft_column, qeft_bits = derive_qeft(args.w_expr, args.eff_kv_expr)
        self._uses_qeft = qeft_bits is not None; self._n_qeft_column = n_qeft_column
        self.wbits, self.kvbits = wbits, kvbits
        self.accelerator.print(f"[options] w_bits={wbits} kv_bits={kvbits} gs={gs_lists} k_prune={kprune} v_prune={vprune}"
              + (f" | QEFT n_outlier={n_qeft_column} on bits={qeft_bits}" if qeft_bits else ""))
        self.ss = LlamaSearchSpace(
            bits={'w': wbits, 'k': kvbits, 'v': kvbits},
            group_size={'w': args.w_group_size, 'k': [list(g) for g in gs_lists], 'v': [list(g) for g in gs_lists]},
            pass_module={'w': [], 'k': [], 'v': []}, config=self.config,
            comp_obj=self.comp_obj, comp_obj_min=[0.0, 0.0], comp_obj_max=[1e9, 1e9],  # placeholder; real box set in _load_pools (ss box only gates ss.sample, unused here)
            n_token=self.n_token, k_pruning_dim=kprune, v_pruning_dim=vprune,
            n_qeft_column=n_qeft_column, qeft_outlier_bits=qeft_bits)
        self.xu = encoding_xu(self.ss); self.nw = nw_split(self.ss); self.n_var = len(self.xu)
        self._comp = JointComp(self.ss)   # vectorized comp_obj (batch over genomes; skips decode+get_net_info)
        self._feats = ArchFeatures(self.ss, self._comp)  # 15d mean13+comp input (--surrogate_input feat/cv)
        self.ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        self.segments = block_segments(self.ss)

    def _load_pools(self, args):
        """1st-stage CROSSOVER pools (ε-band → div_k diversity) + budget box.
        Box: any unset --comp_obj_min/max edge is AUTO-derived from the pools' achievable
        comp range (the input files decide the range); pools are then filtered to the box
        per axis (every W×KV combination stays feasible)."""
        self.Wg, self.KVg, _, _, w_comp, kv_comp = load_block_pools(
            args.w_expr, args.eff_kv_expr, self.ss, w_eps=args.w_front_eps, kv_eps=args.kv_front_eps,
            eps_rel=args.front_eps_rel, div_k=args.div_k, seed=args.seed)
        n = len(self.comp_obj)
        box_min = list(args.comp_obj_min) if args.comp_obj_min is not None else [None] * n
        box_max = list(args.comp_obj_max) if args.comp_obj_max is not None else [None] * n
        self.comp_obj_min = [box_min[i] if box_min[i] is not None else float(c.min())
                             for i, c in enumerate((w_comp, kv_comp))]
        self.comp_obj_max = [box_max[i] if box_max[i] is not None else float(c.max())
                             for i, c in enumerate((w_comp, kv_comp))]
        self.ss.comp_obj_min, self.ss.comp_obj_max = self.comp_obj_min, self.comp_obj_max
        box_src = 'auto' if all(v is None for v in box_min + box_max) else 'user/auto'
        self.accelerator.print(f"[budget box] wbits [{self.comp_obj_min[0]:.3f}, {self.comp_obj_max[0]:.3f}] | "
              f"eff_kvbits [{self.comp_obj_min[1]:.3f}, {self.comp_obj_max[1]:.3f}]  ({box_src})")
        wm = (w_comp >= self.comp_obj_min[0]) & (w_comp <= self.comp_obj_max[0])
        km = (kv_comp >= self.comp_obj_min[1]) & (kv_comp <= self.comp_obj_max[1])
        if wm.sum() < 5 or km.sum() < 5:
            raise SystemExit(f"[budget box] too few in-box blocks (W {wm.sum()}, KV {km.sum()}); "
                             f"widen --comp_obj_min/max or raise --front_eps_rel")
        if wm.sum() < len(self.Wg) or km.sum() < len(self.KVg):
            self.accelerator.print(f"[budget box] W {len(self.Wg)}→{int(wm.sum())} | KV {len(self.KVg)}→{int(km.sum())}")
        self.Wg, self.KVg = self.Wg[wm], self.KVg[km]
        self.w_comp, self.kv_comp = w_comp[wm], kv_comp[km]   # in-box comp (for endpoint corners)

    def _build_knowledge(self, args):
        """1st-stage knowledge wired into the operators:
        - per-gene lever weights (mutation WHERE-probability),
        - L2 freeze: cells ≥ agree_frac of the pool agrees on are zeroed out of mutation
          (loss-free per tests/joint_reabsorption.py; agree_frac > 1.0 disables),
        - P1/P2 BandTable: band-conditional mutation value draws + staircase supply seeding
          (built from the FULL ε-band; + corner genomes since the staircase cannot reach
          the extreme corners — see BandTable docstring for the measured basis)."""
        self.w = gene_weights(self.Wg, self.KVg)
        self.contested, _ = freeze_mask(self.Wg, self.KVg, args.agree_frac)
        self.w = self.w * self.contested
        n_frozen = int((~self.contested).sum())
        self.accelerator.print(f"[L2 freeze] agree_frac={args.agree_frac}: frozen/agreed cells {n_frozen}/{self.n_var} "
              f"→ free(contested) dims {int(self.contested.sum())} "
              f"(W {int(self.contested[:self.nw].sum())}/{self.nw}, "
              f"KV {int(self.contested[self.nw:].sum())}/{self.n_var-self.nw})")
        Wb, wbc, KVb, kbc = load_band_blocks(args.w_expr, args.eff_kv_expr, self.ss,
                                             w_eps=args.w_front_eps, kv_eps=args.kv_front_eps,
                                             eps_rel=args.front_eps_rel)
        self.band_table = BandTable(Wb, wbc, KVb, kbc, self.xu, self.nw)   # band counts auto (~300/band)
        ws = sorted({int(np.argmin(self.w_comp)), int(np.argmax(self.w_comp))})
        ks = sorted({int(np.argmin(self.kv_comp)), int(np.argmax(self.kv_comp))})
        self._corner_genomes = [np.concatenate([self.Wg[a], self.KVg[b]]) for a in ws for b in ks]
        self.accelerator.print(f"[band table] auto bands W {len(self.band_table.we) - 1} × {len(Wb)} blocks | "
              f"KV {len(self.band_table.ke) - 1} × {len(KVb)} blocks (+{len(self._corner_genomes)} corners)")
        # predictor active set is recomputed data-driven per fit (_fit_predictor): cols that
        # vary in the current archive, capped below N for RBF (else singular → nan).
        self.active = np.where(self.w > 0.05)[0]
        self.accelerator.print(f"[ss] n_var={self.n_var} (W-block {self.nw}, KV-block {self.n_var-self.nw}) | "
              f"W-blocks={len(self.Wg)} KV-blocks={len(self.KVg)} | "
              f"contested~{len(self.active)} levers(>0.5)={int((self.w>0.5).sum())}")

    def _build_evaluator(self, args):
        import torch
        from evaluator import LlamaEvaluator
        from utils.func import process_dtype
        # arch-parallel eval pool (--eval_workers > 0): whole archs farmed to persistent
        # per-GPU workers — for expensive backends (awq: per-arch ~8min build dominates,
        # which accelerate-DP would DUPLICATE on every rank instead of parallelize).
        # Main process then builds NO local evaluator (workers own model + teacher logits).
        self.pool = None
        if getattr(args, 'eval_workers', 0) > 0:
            assert self.accelerator.num_processes == 1, \
                '--eval_workers requires num_processes=1 (pool owns the GPUs)'
            from utils.awq_pool import AWQEvalPool
            cfg = dict(seed=args.seed, config=args.config, model_name=args.model_name,
                       model_path=args.model_path,
                       method={'w': args.w_method, 'kv': args.kv_method},
                       quant_model_paths=args.quant_model_paths,
                       seqlen=args.seqlen, n_sample=args.n_sample, dataset=args.dataset,
                       dtype=args.dtype, bits={'w': self.wbits, 'k': self.kvbits, 'v': self.kvbits},
                       group_size=self.ss.group_size, residual_length=args.residual_length,
                       attn_sink=args.attn_sink, k_quant_scheme=args.k_quant_scheme,
                       v_quant_scheme=args.v_quant_scheme, loss_func=args.loss_func,
                       last_tokens=args.last_tokens, stride=args.stride,
                       prefill_prompt=args.prefill_prompt,
                       # QUIET WORKERS (default): each worker redirects its OS-level stdout/stderr
                       # (fd 1/2 — catches run_awq/tqdm/C-extension spew) to a per-worker log file
                       # under <save>/awq_logs/, so the terminal shows only the main-process
                       # [awq_pool] X/N aggregate progress (results flow over the queue, unaffected).
                       # --awq_verbose_workers keeps the old inline spew (None = no redirect).
                       worker_log_dir=None if args.awq_verbose_workers
                                      else os.path.join(self.save_path, 'awq_logs'))
            gpus = [g.strip() for g in args.worker_gpus.split(',') if g.strip()]
            self.pool = AWQEvalPool(gpus[:args.eval_workers] if args.eval_workers <= len(gpus) else gpus,
                                    cfg, recycle_after=args.worker_recycle, log=self.accelerator.print)
            self.evaluator = None
            return
        device_map = self._device_map                      # accelerator already built in __init__
        # QEFT-on-HQQ: archs with n_outlier>0 need the multi-rank outlier dict so the evaluator
        # can insert the FP16 columns per-arch. Required whenever the W archive used outliers.
        outlier = None
        if getattr(args, 'outlier_path', ''):
            outlier = torch.load(args.outlier_path); self.accelerator.print(f"[QEFT] outlier dict: {args.outlier_path}")
        elif getattr(self, '_uses_qeft', False):
            # auto-locate the extract_outidx.py dict under the repo's outlier/ tree (path scheme
            # mirrors search.sh: outlier/<model>/w16_r<ranks>_<dataset>/outlier.pth). Resolved
            # relative to this file so it works regardless of cwd.
            ranks = '_'.join(str(c) for c in self._n_qeft_column if c > 0)
            root = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.join(root, 'outlier', args.model_name, f'w16_r{ranks}_{args.dataset}', 'outlier.pth')
            if os.path.isfile(cand):
                outlier = torch.load(cand); self.accelerator.print(f"[QEFT] auto-loaded outlier dict: {cand}")
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

    # ───────────────── evaluation (real JSD via LlamaEvaluator.eval) ─────────────────
    def _evaluate(self, archs):
        """-> (kept_idx, metric, comp). Pool mode (--eval_workers) farms whole archs to
        the per-GPU workers; archs whose eval hard-fails (nan after retries) are DROPPED
        (a max_value-style penalty would poison the exact-interpolating rbf surrogate),
        so callers must subset by kept_idx. Legacy in-process mode keeps every arch."""
        comp = []
        for arch in archs:
            ni = get_net_info(arch, self.ss.config, self.ss.group_size, n_token=self.n_token, attn_sink=self.attn_sink)
            comp.append([ni[o] for o in self.comp_obj])
        if self.pool is not None:
            ys = self.pool.map(archs)
            kept = [i for i, y in enumerate(ys) if y is not None and np.isfinite(y)]
            if len(kept) < len(archs):
                self.accelerator.print(f"[awq_pool] dropped {len(archs) - len(kept)}/{len(archs)} failed archs")
            return kept, [ys[i] for i in kept], [comp[i] for i in kept]
        metric = []
        for arch in archs:
            m, _ = self.evaluator.eval(self.accelerator, arch, 'loss', loss_func=self.args.loss_func,
                                       stride=self.args.stride, prefill_prompt=self.args.prefill_prompt)
            metric.append(float(list(m.values())[0]))
        return list(range(len(archs))), metric, comp

    def _load_seed_results(self, archive):
        """Pre-measured archs (e.g. tests/awq_alloc_flip AWQ pilots) appended to the DOE
        archive: joins *specs*.json (idx->arch) with *results*.jsonl (idx->y_awq) under
        each --seed_results dir. Free archive mass — but the seed measurements' protocol
        (dataset/n_sample/stride/prefill/last_tokens/attn_sink AND w_method) must match
        this run's, else the surrogate trains on mixed scales."""
        import glob
        seen = {tuple(self.ss.encode(x[0]).tolist()) for x in archive}
        out, n_skip = [], 0
        for d in self.args.seed_results:
            os.makedirs(d, exist_ok=True)   # auto-create a missing seed/cache dir (no hard fail)
            specs = {}
            for p in sorted(glob.glob(os.path.join(d, '*specs*.json'))):
                for s in json.load(open(p)):
                    specs[s['idx']] = s
            for p in sorted(glob.glob(os.path.join(d, '*results*.jsonl'))):
                for line in open(p):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    s = specs.get(r['idx'])
                    y = r.get('y_awq', float('nan'))
                    if s is None or not np.isfinite(y):
                        n_skip += 1; continue
                    try:
                        key = tuple(self.ss.encode(s['arch']).tolist())
                    except Exception:
                        n_skip += 1; continue
                    if key in seen:
                        n_skip += 1; continue
                    ni = get_net_info(s['arch'], self.ss.config, self.ss.group_size,
                                      n_token=self.n_token, attn_sink=self.attn_sink)
                    c = [ni[o] for o in self.comp_obj]
                    if any(cc < lo - 1e-9 or cc > hi + 1e-9 for cc, lo, hi in
                           zip(c, self.comp_obj_min, self.comp_obj_max)):
                        n_skip += 1; continue
                    seen.add(key)
                    out.append([s['arch'], float(y), *c])
        if n_skip:
            self.accelerator.print(f"[seed_results] skipped {n_skip} (dup / out-of-box / non-encodable / nan)")
        return out

    def _cache_seed(self, tag, archs, metrics):
        """Persist freshly-measured (arch -> loss) pairs into the FIRST --seed_results dir as
        a run-tagged specs/results pair, so a future run of the SAME protocol warm-starts from
        them (AWQ ~509s/arch — measurements are worth caching). Format matches
        _load_seed_results (idx-keyed specs list + {idx, y_awq} jsonl); string idx = run+tag+i
        can't collide across runs; dedup-on-READ by encoded arch makes cross-run duplicates
        harmless. No-op when --seed_results is unset."""
        if not getattr(self.args, 'seed_results', None):
            return
        d = self.args.seed_results[0]
        os.makedirs(d, exist_ok=True)
        run = os.path.basename(self.save_path.rstrip('/'))
        rows = [(f"{run}_{tag}_{i}", a, float(m)) for i, (a, m) in enumerate(zip(archs, metrics))
                if np.isfinite(m)]
        if not rows:
            return
        with open(os.path.join(d, f"{run}_{tag}_specs.json"), 'w') as f:
            json.dump([{'idx': idx, 'arch': a} for idx, a, _ in rows], f)
        with open(os.path.join(d, f"{run}_{tag}_results.jsonl"), 'w') as f:
            for idx, _, y in rows:
                f.write(json.dumps({'idx': idx, 'y_awq': y}) + "\n")
        self.accelerator.print(f"[seed_cache] +{len(rows)} measured archs -> {d}/{run}_{tag}_*")

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
    def _fit_genome(self, Xf, targets):
        """genome-input predictor (legacy path) → (pred, active); pred.predict expects
        X[:, active]. active = cols that actually VARY in THIS data (constant-in-sample
        cols make the RBF tail block singular → nan); for RBF cap dim below N (exact
        interpolation needs n_active ≤ N-1; 30-pt margin, floor never above N-1)."""
        var = Xf.var(0); active = np.where(var > 1e-9)[0]
        if self.predictor == 'rbf':
            cap = max(1, min(len(targets) - 1, len(targets) - 30))
            if len(active) > cap:
                active = np.sort(active[np.argsort(-var[active])][:cap])
        kwargs = {'lb': np.zeros(len(active)), 'ub': self.xu[active].astype(float)} if self.predictor == 'rbf' else {}
        pred = get_predictor(self.predictor, Xf[:, active], targets, device=self._pred_device, **kwargs)
        return pred, active

    def _fit_feat(self, Xf, targets):
        """low-dim arch-summary predictor: ArchFeatures mean13+comp (15d) → ard_gp.
        Sample-efficient at AWQ-archive sizes (band ρ .947 from N≈30) and measurement-
        free for mutants — see ArchFeatures docstring for the input-choice evidence.
        Returned pred.predict takes the FULL genome matrix (caller sets active=all)."""
        inner = get_predictor('ard_gp', self._feats(Xf), targets, device=self._pred_device)
        feats = self._feats
        class _FeatPred:
            name = 'feat_ard_gp'
            def predict(self, X):
                return np.asarray(inner.predict(feats(np.asarray(X, float)))).ravel()
        return _FeatPred()

    def _pls_features(self, Xf):
        """'plstyp' input: HQQ-SUPERVISED low-rank embedding (PLS, 8 comps/axis, fit ONCE
        on the FULL 1st-stage per-axis archives' (block, per-axis JSD) pairs — N≈10k
        supports the per-layer input) + comp (EXACT wbits/eff_kvbits: a denoised copy of
        the PLS budget direction, |corr(PLS1, comp)|≈0.997; small consistent corner gain
        +0.002–0.035). Transfer-learning frame: cheap HQQ supervision learns the
        representation, the expensive AWQ archive fits the head — recovers within-cell
        allocation ranking that mean13 washes out.

        BandTable-P1 typicality (mean own-band log-prob, 2d) is OPTIONAL, OFF by default
        (--plstyp_typ to add): a disjoint-split ablation on the 538-arch AWQ archive found
        it NEUTRAL (corner Δ −0.012…+0.006, within the n≈15 noise) — a hand-crafted score
        that neither earns its place nor clearly hurts; dropped for leanness. (An earlier
        ablation showing it 'harmful' was a train/test-overlap artifact — see
        tests/awq_alloc_flip/embedding_input_check.py.)"""
        X = np.clip(np.round(np.asarray(Xf, float)), 0, self.xu).astype(int)
        use_typ = getattr(self, 'band_table', None) is not None and getattr(self.args, 'plstyp_typ', False)
        if getattr(self, '_pls', None) is None:
            from sklearn.cross_decomposition import PLSRegression
            from utils.second_stage import _last_stats
            def pairs(expr, half):
                e = json.load(open(_last_stats(expr)))
                arcs = e['archive'] + e.get('candidates', [])
                G = np.array([self.ss.encode(a[0]) for a in arcs])
                return (G[:, :self.nw] if half == 'w' else G[:, self.nw:]), \
                    np.array([a[1] for a in arcs], float)
            GW, mW = pairs(self.args.w_expr, 'w')
            GK, mK = pairs(self.args.eff_kv_expr, 'kv')
            self._pls = (PLSRegression(8).fit(GW, mW), PLSRegression(8).fit(GK, mK))
            self.accelerator.print(
                f"[pls] embedding fit: {len(mW)} W / {len(mK)} KV HQQ pairs → 8+8 comps + comp"
                + (" + typicality" if use_typ else ""))
        pw, pk = self._pls
        C = self._comp.batch(X, ['wbits', 'eff_kvbits'])
        cols = [pw.transform(X[:, :self.nw]), pk.transform(X[:, self.nw:]), C]
        if use_typ:
            bt = self.band_table
            bw = bt.band(C[:, 0], 'w'); bk = bt.band(C[:, 1], 'kv')
            nk = self.n_var - self.nw
            tw = np.array([np.log(np.maximum(bt.Pw[bw[i], np.arange(self.nw), X[i, :self.nw]], 1e-9)).mean()
                           for i in range(len(X))])
            tk = np.array([np.log(np.maximum(bt.Pk[bk[i], np.arange(nk), X[i, self.nw:]], 1e-9)).mean()
                           for i in range(len(X))])
            cols.append(np.column_stack([tw, tk]))
        return np.column_stack(cols)

    def _fit_plstyp(self, Xf, targets):
        """pls+typ family — measured best WITHIN-CELL input on the 538-arch AWQ archive
        (tests/awq_alloc_flip/embedding_input_check.py): B_low(w2.8) OOS ρ .51 vs feat
        .29, C_kv18 .77 vs .34, cell-mean .63 vs .46, global tied (.996). Kitchen-sink
        pls+feat+typ was WORSE (.55 — feature dilution at this N): keep the family lean."""
        inner = get_predictor('ard_gp', self._pls_features(Xf), targets, device=self._pred_device)
        featf = self._pls_features
        class _PlsPred:
            name = 'plstyp_ard_gp'
            def predict(self, X):
                return np.asarray(inner.predict(featf(np.asarray(X, float)))).ravel()
        return _PlsPred()

    def _fit_selfpls(self, Xf, targets):
        """SELF-bootstrapped PLS input: PLS(16) fit on the CURRENT archive's own
        (genome → measured loss) pairs + comp → ard_gp head. Needs NO 1st-stage
        archive and its supervision IS the target backend (no HQQ-prior ceiling).
        Measured (tests/awq_alloc_flip/selfpls_check.py, 538-arch AWQ archive,
        fold-internal PLS — no leakage): cell-mean OOS ρ .64 = TIES archive-plstyp
        (.63) and beats feat15 (.46); learning curve crosses feat at N≈200
        (N=100 .34 / N=200 .51 / N=430 .60). Cross-model transfer of a foreign
        embedding never beats this or mean-features in any regime
        (crossmodel_transfer_check.py) — self-supervision + generic cold-start
        features are the archive-free answer."""
        from sklearn.cross_decomposition import PLSRegression
        p = PLSRegression(min(16, max(2, len(targets) // 12))).fit(Xf, targets)
        comp = self._comp
        def featf(X):
            X = np.asarray(X, float)
            return np.column_stack([p.transform(X), comp.batch(
                np.clip(np.round(X), 0, self.xu).astype(int), ['wbits', 'eff_kvbits'])])
        inner = get_predictor('ard_gp', featf(Xf), targets, device=self._pred_device)
        class _SelfPred:
            name = 'selfpls_ard_gp'
            def predict(self, X):
                return np.asarray(inner.predict(featf(X))).ravel()
        return _SelfPred()

    def _hist_features(self, Xf):
        """(module, option) OCCUPANCY-HISTOGRAM input (+comp): the PRINCIPLED
        (non-hand-crafted) representation — under the measured additivity
        (v5 Sobol ~99%, MCKP additive R² .979) the per-row option-fraction vector
        is the sufficient statistic of the additive part (y ≈ Σ count·d), and it
        preserves the bit-MIX that per-module MEANS destroy (30%@2b+70%@4b vs
        100%@3b: same mean, different histograms → within-cell signal). NOTE the
        head must be NONLINEAR: hist+ridge (= pure additive fit) FAILS within-cell
        (cell-mean .14-.29 — same-cell archs have near-equal additive predictions;
        within-cell ranking lives in the residual interactions the GP captures)."""
        X = np.clip(np.round(np.asarray(Xf, float)), 0, self.xu).astype(int)
        nrow = self.ss.n_linear + 4
        Xr = X.reshape(len(X), nrow, self.ss.n_block)
        xur = self.xu.reshape(nrow, self.ss.n_block)
        cols = []
        for row in range(nrow):
            for o in range(int(xur[row].max()) + 1):
                cols.append((Xr[:, row, :] == o).mean(1))
        return np.column_stack(cols + [self._comp.batch(X, ['wbits', 'eff_kvbits'])])

    def _fit_hist(self, Xf, targets):
        """hist+GP — measured (tests/awq_alloc_flip/hist_input_check.py, fold-subsample
        learning curve on the 538-arch AWQ archive, cell-mean OOS ρ):
          N=100: hist .42 > selfpls .34 > feat .32 | N=200: hist .57 > selfpls .51 >
          feat .32 | N=430: selfpls .60 ≈ hist .54 > feat .46; global ≥.986 from N=50.
        One fixed representation from DOE onward — no family switching."""
        inner = get_predictor('ard_gp', self._hist_features(Xf), targets, device=self._pred_device)
        featf = self._hist_features
        class _HistPred:
            name = 'hist_ard_gp'
            def predict(self, X):
                return np.asarray(inner.predict(featf(np.asarray(X, float)))).ravel()
        return _HistPred()

    def _kfold_rho(self, Xf, targets, which, k=5):
        """OUT-of-sample 5-fold Spearman of a predictor family on the archive (rbf
        interpolates exactly, so in-sample rho is meaningless for this comparison)."""
        from scipy.stats import spearmanr
        n = len(targets); idx = np.random.default_rng(0).permutation(n)
        pred = np.full(n, np.nan)
        for f in range(k):
            te = np.sort(idx[f::k]); tr = np.setdiff1d(np.arange(n), te)
            if which == 'feat':
                m = self._fit_feat(Xf[tr], targets[tr])
                pred[te] = m.predict(Xf[te])
            elif which == 'plstyp':
                m = self._fit_plstyp(Xf[tr], targets[tr])
                pred[te] = m.predict(Xf[te])
            else:
                m, act = self._fit_genome(Xf[tr], targets[tr])
                pred[te] = np.asarray(m.predict(Xf[te][:, act])).ravel()
        return float(spearmanr(pred, targets).statistic)

    def _fit_predictor(self, archive):
        Xf = self._encode_archive(archive).astype(float)                     # full (N, n_var) encoding (cached)
        targets = np.array([x[1] for x in archive])
        # surrogate device: DEFAULT is GPU-when-visible ('auto'→cuda in predictor/factory.
        # _resolve_device) — RBF's ridge-stabilised _robust_solve makes the cuSOLVER LU solve
        # reliable on the ill-conditioned saddle. Pass --predictor_device cpu to force CPU.
        if not self._pred_dev_logged:
            import torch
            auto = 'cuda' if torch.cuda.is_available() else 'cpu'
            res = auto if self._pred_device in ('auto', None) else self._pred_device
            self.accelerator.print(f"[predictor] {self.predictor} on {res} (requested '{self._pred_device}')")
            self._pred_dev_logged = True
        # input mode: 'genome' = legacy per-cell encoding (needs large N — right for the
        # 10k+ HQQ archive); 'feat' = 15d ArchFeatures (right for small AWQ archives);
        # 'self' = FIXED archive-free schedule (RECOMMENDED for awq runs): feat15 below
        # --selfpls_n0 (cold start — generic mean-features win there in BOTH tests),
        # self-bootstrapped PLS(16)+comp above it (ties archive-plstyp within-cell with
        # NO 1st-stage archive). Deterministic — replaces the per-iter cv bake-off,
        # whose winner criterion (GLOBAL OOS rho) is saturated ~0.99 across families
        # and thus effectively noise w.r.t. the within-cell quality that matters;
        # 'cv' / 'plstyp' are kept for research.
        mode = getattr(self.args, 'surrogate_input', 'genome')
        if mode == 'hist':
            self.active = np.arange(self.n_var)
            pred = self._fit_hist(Xf, targets)
            return pred, pred.predict(Xf)
        if mode == 'self':
            n0 = int(getattr(self.args, 'selfpls_n0', 200))
            if len(targets) >= n0:
                self.active = np.arange(self.n_var)
                pred = self._fit_selfpls(Xf, targets)
                return pred, pred.predict(Xf)
            mode = 'feat'
        if mode == 'cv':
            if len(targets) >= 30:
                rhos = {'genome': self._kfold_rho(Xf, targets, 'genome'),
                        'feat':   self._kfold_rho(Xf, targets, 'feat')}
                try:                       # plstyp needs the 1st-stage archives (skip if absent)
                    rhos['plstyp'] = self._kfold_rho(Xf, targets, 'plstyp')
                except Exception as e:     # noqa: BLE001 — cv degrades gracefully
                    self.accelerator.print(f"[surrogate_cv] plstyp unavailable ({type(e).__name__}: {e})")
                mode = max(rhos, key=rhos.get)
                self.accelerator.print("[surrogate_cv] " + "  ".join(
                    f"{k} {v:.3f}" for k, v in rhos.items()) + f" → {mode}")
            else:
                mode = 'feat'          # tiny archive: the low-dim input is the safe default
        if mode in ('feat', 'plstyp'):
            self.active = np.arange(self.n_var)      # these preds consume the full genome
            pred = self._fit_feat(Xf, targets) if mode == 'feat' else self._fit_plstyp(Xf, targets)
            return pred, pred.predict(Xf)
        pred, active = self._fit_genome(Xf, targets)
        self.active = active                                                 # _next reads this (set before it)
        return pred, pred.predict(Xf[:, active])

    # ───────────────── candidate generation (NSGA-III + joint operators) ─────────────────
    def _next(self, archive, predictor, K):
        """one candidate round: NSGA-III generation → supply seeding (--grid_seed) →
        dedup vs archive → predict → down-select K (--cand_even [+ --al_frac quota])."""
        res = self._nsga(predictor)
        Ga = self._encode_archive(archive)                          # archive genomes (cached)
        g = grid_side(K, self.args.cand_grid)
        pool = res.pop.get('X')
        seen = {tuple(gg) for gg in Ga}
        if self.args.grid_seed:
            # P2 even-supply (the right-end-collapse fix: NSGA's dominance drops high-comp
            # candidates where loss is flat): one FRESH staircase/in-band genome per box
            # grid point (self-regulating vs `seen` — see stair_seed) + corner genomes.
            pool = np.vstack([pool, stair_seed(self.band_table, self.comp_obj_min,
                                               self.comp_obj_max, g, seen=seen,
                                               corners=self._corner_genomes)])
        Xc = np.unique(np.clip(np.round(pool), 0, self.xu).astype(int), axis=0)
        Xc = np.array([gg for gg in Xc if tuple(gg) not in seen])
        if len(Xc) == 0:
            return [], np.zeros((0,))
        pred = np.asarray(predictor.predict(Xc[:, self.active].astype(float))).ravel()
        if len(Xc) > K:
            idx = self._downselect(Xc, archive, K)
            Xc, pred = Xc[idx], pred[idx]
        return [self.ss.decode(gg) for gg in Xc], pred

    def _nsga(self, predictor):
        """NSGA-III over the surrogate problem with the joint operators (block-product
        sampling, axis-block crossover, knowledge mutation — band-conditional under P1)."""
        problem = JointAuxProblem(self.ss, predictor, self.active, self.xu, self.comp_obj,
                                  self.comp_obj_min, self.comp_obj_max, self.n_token, self.attn_sink,
                                  comp=self._comp)
        mut = KnowledgeMutation(self.w, self.xu, self.Wg, self.KVg, self.nw, self.segments,
                                p_val=self.args.mut_p_val, p_mod=self.args.mut_p_mod,
                                band_table=self.band_table, comp=self._comp, comp_obj=self.comp_obj)
        algo = NSGA3(pop_size=self.ga_pop_size, ref_dirs=self.ref_dirs,
                     sampling=FrontierProductSampling(self.Wg, self.KVg),
                     crossover=AxisBlockCrossover(self.nw, self.n_var),
                     mutation=mut, eliminate_duplicates=False)
        return minimize(problem, algo, ('n_gen', 20), seed=self.args.seed, verbose=True)

    def _downselect(self, Xc, archive, K):
        """pick K candidates with the subset selector: union(archive front, picks)
        std-of-gaps GA — hole-filling, keeps edge candidates (an isolated edge candidate
        splits the front's largest gap and is KEPT, where subset-only geometry drops it).
        The only selector that survived the right-end replay post-mortem (29/30 injected
        edge candidates kept vs moo 13/30 / maximin 16/30 — baseline_search 2607100451/0638);
        the losing variants (maximin/grid/hybrid/moo) and the measured-negative --al_frac
        AL quota were removed 2026-07 (utils/select.py keeps the selectors for the
        visualize/ studies)."""
        comp = self._comp.batch(Xc, self.comp_obj)       # (N,2) [wbits, eff_kvbits], vectorized
        Fa = np.column_stack([[x[i] for x in archive] for i in (1, 2, 3)])
        fr_nd = NonDominatedSorting().do(Fa, only_non_dominated_front=True)
        return subset_select(comp, Fa[fr_nd][:, 1:], K, self.args.subset_pop_size,
                             endpoints=np.array([self.comp_obj_min, self.comp_obj_max], float),
                             seed=self.args.seed)

    def _initialize(self, n_doe=None):
        """DOE seed, mirroring search_space.llama.initialize: enumerate the BOUNDARY corners
        first, then fill the remaining budget with random samples → n_doe total. Here the
        boundary corners are the {W min/max comp} × {KV min/max comp} block combos (both edges
        of both comp axes), always included by default; the random fill is the front-block
        product. Returns (archs, n_corner, n_random)."""
        ws = sorted({int(np.argmin(self.w_comp)), int(np.argmax(self.w_comp))})   # W block extremes
        ks = sorted({int(np.argmin(self.kv_comp)), int(np.argmax(self.kv_comp))}) # KV block extremes
        corners = [np.concatenate([self.Wg[wi], self.KVg[ki]]) for wi in ws for ki in ks]
        n_doe = self.n_doe if n_doe is None else n_doe
        rand = FrontierProductSampling(self.Wg, self.KVg)._do(None, max(n_doe - len(corners), 0))
        doe_g = np.concatenate([np.array(corners), rand], axis=0) if len(rand) else np.array(corners)
        return [self.ss.decode(g) for g in doe_g], len(corners), len(rand)

    # ───── main loop — MULTI-PROCESS SAFE (mirrors search.py's accelerator structure) ─────
    # With `accelerate launch --num_processes N>1`, candidate GENERATION (DOE/_fit_predictor/
    # _next), printing and iter_<it>.stats dumps run on the MAIN rank only; the arch/candidate
    # lists are broadcast with gather_for_metrics so every rank EVALUATES the same set (the
    # evaluator shards the calibration batches across ranks and gathers → ~N× eval speedup).
    # Without these guards every rank reruns the whole loop → duplicated output / redundant
    # search / iter_<it>.stats file-races. Reduces to the single-process loop at N=1
    # (is_main always True, gather/barrier no-ops). NOTE: needs #calibration batches >= N.
    def search(self):
        acc = self.accelerator; main = acc.is_main_process
        t0 = time(); start_it = 1
        if self.args.resume:                                  # resume from an iter_<it>.stats
            rf = json.load(open(self.args.resume))
            archive = rf['archive']; start_it = rf['iteration'] + 1
            if main:
                acc.print(f"[resume] {len(archive)} archs from iter {rf['iteration']} → start iter {start_it}")
        else:
            # cached measurements count toward N_DOE: load them first, then MEASURE ONLY the
            # shortfall (N_DOE - cached). If the cache (folder already exists) already has
            # >= N_DOE archs, skip DOE measurement entirely and use them as-is.
            if main:
                seeded = self._load_seed_results([]) if self.args.seed_results else []
                n_need = max(0, self.n_doe - len(seeded))
                archs, n_corner, n_rand = self._initialize(n_need) if n_need > 0 else ([], 0, 0)
                if seeded and archs:                       # don't re-measure already-cached archs
                    _seen = {tuple(self.ss.encode(x[0]).tolist()) for x in seeded}
                    archs = [a for a in archs if tuple(self.ss.encode(a).tolist()) not in _seen]
                if seeded:
                    acc.print(f"[seed_results] {len(seeded)} cached (dir exists) → DOE measures "
                              f"{'0 (>= N_DOE, skip)' if n_need == 0 else str(len(archs)) + ' more'} "
                              f"(target N_DOE={self.n_doe})")
            else:
                archs, n_corner, n_rand = [], 0, 0
            archs = acc.gather_for_metrics(archs, use_gather_object=True)   # → all ranks
            acc.wait_for_everyone()
            kept, metric, comp = self._evaluate(archs) if archs else ([], [], [])
            if main:
                measured = [[archs[i], m, *c] for i, m, c in zip(kept, metric, comp)]
                if kept:
                    self._cache_seed('doe', [archs[i] for i in kept], metric)   # persist new measurements
                archive = seeded + measured
                losses = [x[1] for x in archive]
                acc.print(f"[DOE] {len(archive)} archs = {len(seeded)} cached + {len(measured)} measured"
                      + (f" ({n_corner} corners + {n_rand} random)" if measured else " (N_DOE met by cache)")
                      + f"  loss {min(losses):.4f}-{max(losses):.4f}  ({time()-t0:.1f}s)")
            else:
                archive = []
        if main:
            ref_pt = np.array([np.max([x[i] for x in archive]) for i in range(1, len(self.comp_obj) + 2)])
            acc.print(f'data preparation time : {time() - t0:.2f}s')
        acc.wait_for_everyone()

        for it in range(start_it, self.iterations + 1):
            iter_start = time()
            # construct accuracy predictor + next candidates on the main rank only
            if main:
                predictor_start = time()
                pred, a_pred = self._fit_predictor(archive)
                predictor_time = time() - predictor_start
                next_start = time()
                cands, c_pred = self._next(archive, pred, self.n_iter)
                next_time = time() - next_start
            else:
                cands = []
            acc.wait_for_everyone()
            cands = acc.gather_for_metrics(cands, use_gather_object=True)   # broadcast to all ranks
            if not cands:
                if main:
                    acc.print(f"Iter {it}: no new candidates; stop")
                break
            kept, c_metric, c_comp = self._evaluate(cands)    # data-parallel across ranks
            if main:
                # check accuracy predictor's performance (c_pred subset to the kept archs)
                rmse, rho, tau = get_correlation(
                    np.concatenate([np.asarray(a_pred).ravel(), np.asarray(c_pred).ravel()[kept]]),
                    np.array([x[1] for x in archive] + c_metric))
                for i, m, c in zip(kept, c_metric, c_comp):
                    archive.append([cands[i], m, *c])
                self._cache_seed(f'it{it}', [cands[i] for i in kept], c_metric)
                F = np.column_stack([[x[i] for x in archive] for i in range(1, len(self.comp_obj) + 2)])
                hv = calc_hv(ref_pt, F); cov = front_coverage(archive, self.comp_obj)
                iter_time = time() - iter_start
                # print iteration-wise statistics (search.py format)
                acc.print(f"Iter {it}: hv = {hv:.2f}, iter time : {iter_time:.2f}s, "
                      f"predictor_time : {predictor_time:.2f}, next_time : {next_time:.2f}")
                acc.print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall's Tau = {tau:.4f}")
                for obj in self.comp_obj:
                    c = cov[obj]
                    acc.print(f"  {obj} front-coverage : {c['coverage']*100:.1f}%  "
                          f"front=[{c['front_min']:.3f}, {c['front_max']:.3f}] / "
                          f"full=[{c['full_min']:.3f}, {c['full_max']:.3f}]")
                acc.print(f'iteration time : {iter_time:.2f}s')
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
            acc.wait_for_everyone()

        if self.pool is not None:
            self.pool.close()
        if main:
            acc.print(f"[done] {len(archive)} archs, {time()-t0:.1f}s → {self.save_path}")
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
        self.accelerator.print(f"[results] {os.path.join(self.save_path, self.result_file)}")


def build_parser():
    p = argparse.ArgumentParser(description="2nd-stage joint W×eff_kvbits NAS (HQQ, NSGA-III, LlamaSearchSpace)")
    p.add_argument('--config', default='config/llama.json'); p.add_argument('--model_name', default='Llama-3.1-8B-Instruct')
    p.add_argument('--w_expr', required=True, help='1st-stage W-axis archive dir or iter_N.stats')
    p.add_argument('--eff_kv_expr', required=True, help='1st-stage eff_kvbits archive dir or iter_N.stats')
    p.add_argument('--surrogate', default='rbf', help='arch-input predictor (rbf/gp/ard_gp/carts)')
    p.add_argument('--selfpls_n0', type=int, default=200,
                   help="'self' input mode: archive size at which the surrogate input "
                        "switches feat15 → self-bootstrapped PLS (measured crossover "
                        "N≈200, tests/awq_alloc_flip/selfpls_check.py)")
    p.add_argument('--surrogate_input', default='genome',
                   choices=['genome', 'feat', 'plstyp', 'self', 'hist', 'cv'],
                   help="predictor input: 'genome' = per-cell encoding (needs large N — the 10k+ "
                        "HQQ-archive regime); 'feat' = 15d ArchFeatures per-module means + comp "
                        "(sample-efficient — ties measured per-axis-JSD input within-band, works "
                        "from N~30; use for small AWQ archives); 'plstyp' = HQQ-supervised PLS "
                        "embedding + comp (best within-cell at small N — the AWQ default); "
                        "'cv' = 5-fold out-of-sample compare per iter, use the winner (logged)")
    p.add_argument('--plstyp_typ', action='store_true',
                   help="plstyp: append the 2d BandTable typicality features (OFF by default — "
                        "ablation-neutral on the AWQ archive, corner Δ within noise; kept as a "
                        "research toggle)")
    p.add_argument('--predictor_device', default='auto', help="surrogate compute device: 'auto' (=cuda when visible; the RBF saddle solve is ridge-stabilised) / 'cuda' / 'cuda:N' / 'cpu'")
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
    # per-iteration candidate down-select = subset selector (union(archive front, picks)
    # std-of-gaps GA — hole-filling, keeps edge candidates: 29/30 injected right-end kept
    # vs moo 13/30 / maximin 16/30, baseline_search 2607100451/0638 post-mortem; the losing
    # selector variants and the measured-negative --al_frac AL quota were removed 2026-07).
    p.add_argument('--subset_pop_size', type=int, default=100, help='down-select subset-GA population size')
    p.add_argument('--cand_grid', type=int, default=0, help='seed grid side over the budget box (0=auto=ceil(sqrt(n_iter)))')
    p.add_argument('--grid_seed', action='store_true', help='inject staircase even-supply genomes per box grid cell into the candidate pool each iter (guarantees high-comp supply)')
    # mutation = 1st-stage-knowledge-guided (value-resample + module-transplant); value draws
    # are BAND-CONDITIONAL (P1) and supply seeds are staircase-decoded (P2) — the legacy
    # global-draw / nearest-block arms were removed 2026-07 after losing the 2-seed A/B
    # pilot (HV .2523/.2620 vs .2481/.2580; per-cell best-loss 14-6/14-8; corner better).
    # Band counts and seed freshness are AUTO (BandTable._auto_bands / stair_seed seen-retry).
    p.add_argument('--mut_p_val', type=float, default=0.5, help='prob a mutated cell takes a 1st-stage value (else ±1)')
    p.add_argument('--mut_p_mod', type=float, default=0.15, help='prob/indiv of a 1st-stage module-transplant')
    # search-space options are AUTO-DERIVED from the 1st-stage archives (derive_options).
    # --w_bits is kept as a consistency CHECK against the HQQ bank list (not a control);
    # the W group size is the only free space knob here.
    p.add_argument('--w_bits', type=int, nargs='+', default=[],
                   help='expected W bit ladder — verified against the archive-derived options '
                        '(guards a quant_model_paths/archive mismatch); empty = skip check')
    p.add_argument('--w_group_size', type=int, default=128)
    # arch-parallel eval pool (for expensive backends: --w_method awq)
    p.add_argument('--eval_workers', type=int, default=0,
                   help='N arch-parallel GPU eval workers (persistent LlamaEvaluator each). '
                        'Use with --w_method awq where the ~8min per-arch build dominates '
                        '(accelerate-DP would duplicate it per rank, not parallelize). '
                        '0 = legacy in-process eval. Requires num_processes=1.')
    p.add_argument('--worker_gpus', default='0,1,2,3',
                   help='comma-separated GPU ids for --eval_workers')
    p.add_argument('--worker_recycle', type=int, default=32,
                   help='recycle each eval worker after this many archs (run_awq inter-build leak defense)')
    p.add_argument('--awq_verbose_workers', action='store_true',
                   help='keep each AWQ eval worker printing its run_awq/tqdm build output inline '
                        '(default: workers redirect stdout/stderr to <save>/awq_logs/worker*.log '
                        'so the terminal shows only the main [awq_pool] X/N aggregate progress)')
    p.add_argument('--seed_results', nargs='*', default=[],
                   help='dirs of pre-measured results (*specs*.json + *results*.jsonl, e.g. '
                        'save/awq_alloc_flip = the curated 88-arch pilot+round-0 seed set) '
                        'appended to the DOE archive; measurement protocol AND w_method must '
                        'match this run. NOTE: every *specs*.json in the dir is globbed and must '
                        "carry an 'idx' key — keep probe/other-format files out of the seed dir")
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
