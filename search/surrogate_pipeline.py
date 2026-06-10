"""surrogate_pipeline.py — three-mode SLURM-friendly pipeline for fitting
surrogate models with random / GA-based / active-learning sample picks,
sharing a single quantile warm-start and a separate held-out validation
pool for fair cross-method comparison.

A successor to ``sample_surrogate.py`` for the comparison study. Keeps
post_search.py compatibility — ``--mode aggregate`` writes per-method
``results_<method>.csv`` files that ``post_search.load_sample_csv`` can
read as ``--sample_path`` unchanged.

Modes
-----
* ``--mode sample``  Build / extend ``archs.csv`` (one row per arch). CPU-only.
    ``--method quantile``   Initial warm-start anchors (run once, round 0).
    ``--method random``     Append ``--batch`` random extras.
    ``--method ga``         Append ``--batch`` coverage_nsga2 extras.
    ``--method al --acq A`` Append ``--batch`` active-learning extras using
                             acquisition ``A`` (refits on completed
                             ``result_*.json`` each round). ``--acq`` ∈
                               ei    EVI (conformal σ) — lowest-JSD arch (obj B)
                               ucb   LCB = μ − κσ            — obj B
                               alm   max posterior σ          — obj A
                               imse  ALC integrated-var drop  — obj A (GP only)
                               maximin farthest-point (model-free) — obj A
                               qbc   bootstrap-ensemble σ (tps-friendly) — obj A
                               rank  ensemble rank-disagreement — obj A
                             obj A = global surrogate ranking accuracy;
                             obj B = find the best arch within the band.
                             σ source: GP posterior std when --surrogate ard_gp,
                             else bootstrap ensemble (so rbf/tps also work).
    ``--method al_ei``      Back-compat alias of ``--method al --acq ei``.
    ``--validation``        (orthogonal) Build / extend
                             ``validation_archs.csv`` with uniform random
                             picks under ``--val_seed``. Used by all
                             surrogates for the same held-out comparison.

* ``--mode eval --idx N``  Evaluate row N of archs.csv (or
    validation_archs.csv when ``--validation`` set). Writes
    ``result_<N>.json`` / ``validation_result_<N>.json``. Idempotent.

* ``--mode aggregate``  Scans all completed ``result_*.json`` into:
    - ``results_<method>.csv`` — column-per-arch layout consumed by
      ``post_search.load_sample_csv``. One file per method (random / ga /
      al_ei), each including the shared quantile anchors.
    - ``validation_metrics.csv`` — per-method R² / Spearman / Kendall on
      the shared held-out validation pool.
    - ``learning_curve.csv`` — (AL only) per-round R² / Spearman / Kendall
      on the validation pool, for the AL convergence plot.

archs.csv columns
-----------------
``idx, round, source, nd_idx_json, arch_json, <comp_keys...>,
metric_<axis>..., combined_metric``

EI acquisition
--------------
Mean = ``ard_gp.predict()`` (mean only). σ = conformal:
``σ_conf = q_{0.95}(|LOO residual|)`` brute-force over the train pool.
Baseline ``B_ε = y_min - eps``; ``EVI = (B_ε - μ)·Φ(t) + σ·φ(t)``,
``t = (B_ε - μ) / σ``. Top-K greedy in EVI (no batch penalisation;
predictor refits each round so successive rounds see the previous batch).
"""
import os
import json
import csv
import argparse
import warnings

import numpy as np
import torch
import scipy.stats as stats

from evaluator import LlamaEvaluator
from utils.func import (build_expr_map, build_nd, evaluate_metric,
                        comp_key_order, get_net_info)
from utils.select import (build_arch, draw_random, assemble_F,
                          select_valid_nd_idx, quantile_select, axis_of_map,
                          coverage_subset_nsga2_extras)

from correlation import _build_ctx
from post_search import _make_surrogate, _resolve_surrogate_device
from predictor.factory import all_surrogates as _all_surrogates, _strip_transform

warnings.simplefilter("ignore")

SURROGATES = _all_surrogates()
# Acquisitions selectable via --acq (all driven by --method al; al_ei kept as a
# back-compat alias of --method al --acq ei). Objective tag: 'A' = global
# surrogate-ranking accuracy (uncertainty / coverage / disagreement), 'B' =
# find the lowest-JSD arch within the band (improvement-based).
ACQS = ['ei', 'ucb', 'alm', 'imse', 'maximin', 'qbc', 'rank']
ACQ_OBJECTIVE = {'ei': 'B', 'ucb': 'B', 'alm': 'A', 'imse': 'A',
                 'maximin': 'A', 'qbc': 'A', 'rank': 'A'}
METHODS = ['quantile', 'random', 'ga', 'al', 'al_ei']


# ════════════════════════════════════════════════════════════════════════════
# archs.csv helpers
# ════════════════════════════════════════════════════════════════════════════
def _archs_path(args):
    return os.path.join(args.save or '.',
                        'validation_archs.csv' if args.validation
                        else 'archs.csv')


def _result_path(args, idx):
    prefix = 'validation_result' if args.validation else 'result'
    return os.path.join(args.save or '.', f'{prefix}_{idx}.json')


def _meta_path(args):
    return os.path.join(args.save or '.',
                        'validation_meta.json' if args.validation
                        else 'sample_meta.json')


def _load_archs_csv(path):
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        return list(reader), header


def _existing_pool_positions(rows, valid_nd_idx):
    """Map archs.csv rows' nd_idx_json back to their position in valid_nd_idx.
    Rows whose nd-tuple is no longer in the current pool (comp_obj filter
    changed) are silently dropped — they don't constrain the new picks."""
    if not rows:
        return np.array([], dtype=np.int64)
    pool_lookup = {tuple(int(x) for x in row): pos
                   for pos, row in enumerate(valid_nd_idx)}
    out = []
    for r in rows:
        t = tuple(json.loads(r['nd_idx_json']))
        p = pool_lookup.get(t)
        if p is not None:
            out.append(p)
    return np.asarray(out, dtype=np.int64)


def _append_rows(path, header, rows):
    if not rows:
        return
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a' if exists else 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _archs_header(comp_keys, expr_keys):
    return (['idx', 'round', 'source', 'nd_idx_json', 'arch_json']
            + list(comp_keys)
            + [f'metric_{k}' for k in expr_keys]
            + ['combined_metric'])


def _build_row(*, idx, round_id, source, nd_idx, arch, comp_keys, comp_vals,
               expr_keys, per_axis_vals, combined):
    row = dict(
        idx=idx, round=round_id, source=source,
        nd_idx_json=json.dumps([int(x) for x in nd_idx]),
        arch_json=json.dumps(arch, separators=(',', ':')),
        combined_metric=float(combined),
    )
    for ck, cv in zip(comp_keys, comp_vals):
        row[ck] = float(cv) if isinstance(cv, (int, float, np.integer,
                                               np.floating)) else cv
    for k, v in zip(expr_keys, per_axis_vals):
        row[f'metric_{k}'] = float(v)
    return row


def _load_arch_row(path, idx):
    rows, _ = _load_archs_csv(path)
    for r in rows:
        if int(r['idx']) == idx:
            return r
    raise SystemExit(f"idx {idx} not in {path} ({len(rows)} rows)")


# ════════════════════════════════════════════════════════════════════════════
# Candidate pool (shared by every cmd_sample branch)
# ════════════════════════════════════════════════════════════════════════════
def _build_pool(args, ctx):
    expr_map = build_expr_map(args, ctx)
    nd = build_nd(args, ctx, expr_map)
    expr_keys, _esm, _efm = nd.expr_keys, nd.esm, nd.efm
    K_axes = len(expr_keys)

    valid_nd_idx = select_valid_nd_idx(
        nd.nd_shape, nd.new_metric_nd, nd.comp_nd_list,
        comp_obj_min=args.comp_obj_min, comp_obj_max=args.comp_obj_max,
        random_sample=None,
        has_quantile=True,      # forces full feasible pool to be materialised
        has_prefer=False)
    if len(valid_nd_idx) == 0:
        raise SystemExit(
            "[surrogate_pipeline] 0 candidates after comp_obj filter — "
            "widen --comp_obj_min / --comp_obj_max.")

    F = assemble_F(valid_nd_idx, expr_keys, _efm,
                   nd.comp_nd_list, nd.new_metric_nd)
    M_valid = F[:, [1 + 2 * i for i in range(K_axes)]]
    comp_keys = comp_key_order(ctx.config, ctx.group_size)
    return (valid_nd_idx, F, M_valid,
            expr_keys, _esm, _efm, nd, comp_keys)


# ════════════════════════════════════════════════════════════════════════════
# Completed results loader (AL training data)
# ════════════════════════════════════════════════════════════════════════════
def _load_completed_results(save_dir, rows, *, dataset_idx=0,
                            validation=False):
    """Return dict {row_idx → measured_metric_float}. Drops in-progress /
    NaN entries. ``dataset_idx`` picks which dataset's metric to use as y."""
    out = {}
    prefix = 'validation_result' if validation else 'result'
    for r in rows:
        idx = int(r['idx'])
        path = os.path.join(save_dir, f'{prefix}_{idx}.json')
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                res = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        mm = res.get('measured_metric')
        if not isinstance(mm, list) or not mm:
            continue
        v = mm[dataset_idx] if dataset_idx < len(mm) else mm[0]
        if not isinstance(v, (int, float)) or np.isnan(v):
            continue
        out[idx] = float(v)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Active-learning EI acquisition (conformal σ on top of ard_gp mean)
# ════════════════════════════════════════════════════════════════════════════
def _loocv_residuals(args, X, y, M_bounds):
    """Brute-force leave-one-out residuals using the chosen surrogate.
    ~N refits; for N≈30-60 with ard_gp this is ~1-3 min one-time."""
    N = len(y)
    res = np.zeros(N, dtype=float)
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        m = _make_surrogate(args, X[mask], y[mask], M_bounds)
        yhat = float(np.asarray(m.predict(X[[i]])).reshape(-1)[0])
        res[i] = float(y[i]) - yhat
    return res


def _ei_acquire(args, X_train, y_train, M_cand, K, *, eps=1e-6):
    """Conformal-σ EI. Returns (selected positions into M_cand, info dict).
    Minimization objective: lower y = better. Top-K greedy on EVI."""
    M_bounds = np.vstack([X_train, M_cand]) if len(M_cand) else X_train
    loo_res = _loocv_residuals(args, X_train, y_train, M_bounds)
    sigma = max(float(np.quantile(np.abs(loo_res), 0.95)), 1e-9)

    m = _make_surrogate(args, X_train, y_train, M_bounds)
    mu = np.asarray(m.predict(M_cand)).reshape(-1).astype(float)

    y_min = float(np.min(y_train))
    B_eps = y_min - eps * max(abs(float(np.mean(y_train))), 1e-9)
    diff = B_eps - mu
    t = diff / sigma
    evi = diff * stats.norm.cdf(t) + sigma * stats.norm.pdf(t)

    K = int(min(K, len(evi)))
    sel = np.argsort(-evi, kind='stable')[:K]
    return (np.asarray(sel, dtype=np.int64),
            dict(sigma_conf=sigma, B_eps=B_eps, y_min=y_min,
                 mu_min=float(mu.min()), mu_max=float(mu.max()),
                 evi_max=float(evi.max()), n_train=int(len(y_train)),
                 n_cand=int(len(mu))))


# ════════════════════════════════════════════════════════════════════════════
# Uncertainty signals (shared by alm / ucb / qbc / rank)
# ════════════════════════════════════════════════════════════════════════════
def _gp_mu_std(args, X_tr, y_tr, M_cand):
    """Fit a plain ARD-GP (matching --ard_kernel) and return its posterior
    (mu, std) over M_cand. Used as the query-dependent σ for variance-based
    acquisitions when the surrogate base is a GP."""
    from predictor.ard_gp import ARDGP
    m = ARDGP(kernel=args.ard_kernel, n_restarts=args.gp_n_restarts,
              device=_resolve_surrogate_device(args.surrogate_device))
    m.fit(np.asarray(X_tr, float), np.asarray(y_tr, float))
    mu, std = m.predict(np.asarray(M_cand, float), return_std=True)
    return np.asarray(mu, float).ravel(), np.asarray(std, float).ravel(), m


def _bootstrap_preds(args, X_tr, y_tr, M_cand, B, seed=0):
    """B bootstrap refits of the chosen surrogate → (B, n_cand) predictions.
    Model-agnostic query-dependent uncertainty (the tps-native AL signal):
    works for rbf/tps interpolants that have no posterior variance."""
    X_tr = np.asarray(X_tr, float); y_tr = np.asarray(y_tr, float)
    M_cand = np.asarray(M_cand, float)
    rng = np.random.default_rng(seed)
    n = len(y_tr)
    bounds = np.vstack([X_tr, M_cand]) if len(M_cand) else X_tr
    P = np.empty((B, len(M_cand)), dtype=float)
    for b in range(B):
        bi = rng.integers(0, n, size=n)
        mdl = _make_surrogate(args, X_tr[bi], y_tr[bi], bounds)
        P[b] = np.asarray(mdl.predict(M_cand)).reshape(-1)
    return P


def _mu_sigma(args, X_tr, y_tr, M_cand, *, force_bootstrap=False):
    """(mu, sigma) over M_cand. GP posterior std when the base surrogate is
    ard_gp (unless force_bootstrap), else bootstrap-ensemble mean/std."""
    base = _strip_transform(args.surrogate)[1]
    if base == 'ard_gp' and not force_bootstrap:
        mu, sd, _ = _gp_mu_std(args, X_tr, y_tr, M_cand)
        return mu, sd
    P = _bootstrap_preds(args, X_tr, y_tr, M_cand, int(args.al_qbc_B))
    return P.mean(0), P.std(0)


def _maximin_pick(Xs_c, Xs_seed, K):
    """Greedy farthest-point: pick K rows of Xs_c maximizing min-distance to
    Xs_seed ∪ already-picked. Xs_* are standardized. Returns positions in Xs_c."""
    if len(Xs_seed):
        dmin = np.min(np.linalg.norm(Xs_c[:, None] - Xs_seed[None], axis=2), axis=1)
    else:
        dmin = np.full(len(Xs_c), np.inf)
    sel = []
    for _ in range(int(min(K, len(Xs_c)))):
        j = int(np.argmax(dmin))
        sel.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(Xs_c - Xs_c[j], axis=1))
        dmin[j] = -1.0
    return np.asarray(sel, dtype=np.int64)


def _topk_diverse(scores, M_cand, K, enable, mult=4):
    """Top-K by score; if `enable`, spread the batch by maximin among the
    top mult·K scorers (high-score AND diverse — avoids a clumped batch that
    a single sbatch array would waste on near-identical archs)."""
    order = np.argsort(-np.asarray(scores), kind='stable')
    K = int(min(K, len(order)))
    if not enable or len(order) <= K:
        return order[:K]
    pool = order[:min(len(order), mult * K)]
    Xp = np.asarray(M_cand, float)[pool]
    mu_, sd_ = Xp.mean(0), Xp.std(0); sd_[sd_ < 1e-12] = 1.0
    Xs = (Xp - mu_) / sd_
    # seed maximin with the single highest scorer so it stays score-anchored.
    sub = _maximin_pick(Xs, Xs[:1], K - 1)
    return pool[np.concatenate([[0], sub])][:K]


# ════════════════════════════════════════════════════════════════════════════
# Unified acquisition dispatch (all --acq choices)
# ════════════════════════════════════════════════════════════════════════════
def _acquire(args, X_train, y_train, M_cand, K, acq):
    """Return (selected positions into M_cand, info dict) for the chosen
    acquisition. Minimization objective (lower JSD = better)."""
    K = int(min(K, len(M_cand)))
    info = dict(acq=acq, objective=ACQ_OBJECTIVE[acq],
                n_train=int(len(y_train)), n_cand=int(len(M_cand)))
    diverse = bool(args.al_diverse)

    # Improvement-2: transform y for the σ/cov-fitting acquisitions so the
    # heavy-tailed JSD extreme does not dominate (ei keeps raw y — its baseline
    # B_ε is defined in y-space).
    tr = getattr(args, 'al_transform', 'none')
    if acq != 'ei' and tr != 'none':
        y_train = np.asarray(y_train, float)
        y_train = (np.sqrt(np.clip(y_train, 0, None)) if tr == 'sqrt'
                   else np.log(np.clip(y_train, 1e-12, None)))
        info['transform'] = tr

    if acq == 'ei':
        sel, ei_info = _ei_acquire(args, X_train, y_train, M_cand, K)
        info.update(ei_info)
        return sel, info

    if acq == 'ucb':
        mu, sigma = _mu_sigma(args, X_train, y_train, M_cand)
        lcb = mu - float(args.al_ucb_kappa) * sigma     # optimistic lower bound
        sel = np.argsort(lcb, kind='stable')[:K]        # smallest LCB = best
        info.update(kappa=float(args.al_ucb_kappa),
                    lcb_min=float(lcb.min()), sigma_max=float(sigma.max()))
        return np.asarray(sel, np.int64), info

    if acq in ('alm', 'qbc'):
        _, sigma = _mu_sigma(args, X_train, y_train, M_cand,
                             force_bootstrap=(acq == 'qbc'))
        sel = _topk_diverse(sigma, M_cand, K, diverse)
        info.update(sigma_max=float(sigma.max()),
                    signal='gp_std' if (acq == 'alm' and
                            _strip_transform(args.surrogate)[1] == 'ard_gp')
                            else 'bootstrap_std')
        return np.asarray(sel, np.int64), info

    if acq == 'rank':
        P = _bootstrap_preds(args, X_train, y_train, M_cand, int(args.al_qbc_B))
        ranks = P.argsort(1).argsort(1).astype(float)   # per-member ranks
        score = ranks.std(0)                             # rank disagreement
        sel = _topk_diverse(score, M_cand, K, diverse)
        info.update(rank_std_max=float(score.max()), n_members=int(P.shape[0]))
        return np.asarray(sel, np.int64), info

    if acq == 'maximin':
        X_all = np.vstack([np.asarray(X_train, float), np.asarray(M_cand, float)])
        mu_, sd_ = X_all.mean(0), X_all.std(0); sd_[sd_ < 1e-12] = 1.0
        Xs_tr = (np.asarray(X_train, float) - mu_) / sd_
        Xs_c = (np.asarray(M_cand, float) - mu_) / sd_
        sel = _maximin_pick(Xs_c, Xs_tr, K)
        info.update(signal='model_free_distance')
        return np.asarray(sel, np.int64), info

    if acq == 'imse':
        if _strip_transform(args.surrogate)[1] != 'ard_gp':
            print(f"[al imse] surrogate '{args.surrogate}' has no posterior "
                  f"cov; falling back to alm (bootstrap σ).")
            return _acquire(args, X_train, y_train, M_cand, K, 'alm')
        cap = int(args.al_pool_cap)
        n = len(M_cand)
        if n > cap:
            rng = np.random.default_rng(int(args.seed))
            idx = rng.choice(n, cap, replace=False)
            print(f"[al imse] candidate/reference pool capped {n}→{cap} "
                  f"(random; logged, not silent).")
        else:
            idx = np.arange(n)
        _, _, gp = _gp_mu_std(args, X_train, y_train, M_cand[idx])
        cov = gp.predict_cov(np.asarray(M_cand, float)[idx])
        jit = 1e-12
        # one-step ALC: adding c lowers every ref j's var by cov(c,j)^2/var(c).
        delta = (cov ** 2).sum(1) / (np.diag(cov) + jit)
        order = idx[np.argsort(-delta)[:K]]
        info.update(pool_cap=cap, delta_max=float(delta.max()))
        return np.asarray(order, np.int64), info

    raise SystemExit(f"unknown --acq {acq}")


# ════════════════════════════════════════════════════════════════════════════
# cmd_sample
# ════════════════════════════════════════════════════════════════════════════
def cmd_sample(args):
    # Validation mode reseeds for clean disjoint draws from train.
    if args.validation:
        args.seed = int(args.val_seed)

    ctx = _build_ctx(args)
    (valid_nd_idx, F, M_valid,
     expr_keys, _esm, _efm, nd, comp_keys) = _build_pool(args, ctx)

    archs_csv = _archs_path(args)
    existing_rows, _ = _load_archs_csv(archs_csv)
    existing_pos = _existing_pool_positions(existing_rows, valid_nd_idx)
    next_idx = len(existing_rows)
    print(f"[sample] pool=|{len(valid_nd_idx)}|  existing={len(existing_rows)}"
          f"  mode={'validation' if args.validation else args.method}"
          f"  round={args.round}")

    # ────── pick new positions (in valid_nd_idx coords) + source tag ──────
    new_positions, new_sources = [], []

    if args.validation:
        rng = np.random.default_rng(int(args.val_seed))
        pos = draw_random(int(args.n_val), len(valid_nd_idx),
                          exclude=existing_pos.tolist(), rng=rng)
        new_positions += [int(p) for p in pos]
        new_sources += ['validation'] * len(pos)

    elif args.method == 'quantile':
        if not args.quantile_sample:
            raise SystemExit(
                "--method quantile requires --quantile_sample "
                "(e.g. 'metric_w#0.01,0.5,0.99 metric_kv#0.01,0.5,0.99').")
        specs = {}
        for s in args.quantile_sample:
            k, v = s.split('#')
            specs[k] = [float(q) for q in v.split(',')]
        _axis_map = axis_of_map(expr_keys)
        _flag = {'w': '--w_expr', 'kv': '--kv_expr',
                 'kvdim': '--kvdim_expr', 'eff_kv': '--eff_kv_expr'}
        for k in specs:
            ax = _axis_map.get(k)
            if ax is not None and ax not in expr_keys:
                raise SystemExit(
                    f"[quantile] '{k}' depends on axis '{ax}' but "
                    f"{_flag.get(ax, ax)} was not provided.")
        I_quant, _ = quantile_select(
            specs, valid_nd_idx, expr_keys, _esm, ctx.default_arch,
            ctx.config, ctx.group_size, args.n_token,
            axis_cache={}, efm=_efm)
        existing_set = set(existing_pos.tolist())
        I_quant = [int(i) for i in I_quant if int(i) not in existing_set]
        new_positions += I_quant
        new_sources += ['quantile'] * len(I_quant)
        print(f"[quantile] {len(I_quant)} new anchors (de-duped against "
              f"{len(existing_set)} existing)")

    elif args.method == 'random':
        rng = np.random.default_rng(int(args.seed) + 1000 * int(args.round))
        pos = draw_random(int(args.batch), len(valid_nd_idx),
                          exclude=existing_pos.tolist(), rng=rng)
        new_positions += [int(p) for p in pos]
        new_sources += ['random'] * len(pos)

    elif args.method == 'ga':
        fit_mode = args.sampling_method.replace('coverage_nsga2_', '')
        I_extra = coverage_subset_nsga2_extras(
            valid_nd_idx, _efm, expr_keys,
            anchor_idx=existing_pos.tolist(),
            K=int(args.batch), fitness=fit_mode,
            coord=args.coverage_coord,
            per_axis_agg=args.coverage_per_axis_agg,
            pareto_select=args.coverage_pareto_select,
            seed=int(args.seed) + int(args.round), verbose=False)
        I_extra = [int(p) for p in I_extra]
        new_positions += I_extra
        new_sources += ['ga'] * len(I_extra)

    elif args.method in ('al', 'al_ei'):
        # al_ei is the back-compat alias of `--method al --acq ei`; otherwise
        # the acquisition is whatever --acq selects.
        acq = 'ei' if args.method == 'al_ei' else args.acq
        src_tag = 'al_ei' if args.method == 'al_ei' else f'al_{acq}'
        completed = _load_completed_results(
            args.save or '.', existing_rows,
            dataset_idx=int(args.al_dataset_idx))
        if not completed:
            raise SystemExit(
                f"[{src_tag}] no completed result_*.json — the warm-start "
                "round (method=quantile / random / ga) must finish evaluating "
                "before an AL round ≥ 1.")
        # Train (X, y) from completed rows only, X taken from per-axis
        # metric columns already stored in archs.csv.
        X_train, y_train = [], []
        used_pos = []
        for r in existing_rows:
            idx = int(r['idx'])
            if idx not in completed:
                continue
            X_train.append([float(r[f'metric_{k}']) for k in expr_keys])
            y_train.append(completed[idx])
            t = tuple(json.loads(r['nd_idx_json']))
            # not every row needs to be in current pool, but for the mask
            # we use existing_pos derived above (also includes pending rows).
            del t  # silence linter
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        mask = np.ones(len(valid_nd_idx), dtype=bool)
        if len(existing_pos):
            mask[existing_pos] = False
        cand_pos = np.where(mask)[0]
        # Pool-based AL: the full feasible combo space can be tens of millions,
        # so scoring every candidate (esp. bootstrap qbc/rank) is intractable.
        # Subsample the candidate set to --al_pool_cap (seeded per round; logged,
        # not silent). Applies uniformly to every acq so the comparison is fair.
        cap = int(args.al_pool_cap)
        if cap > 0 and len(cand_pos) > cap:
            rng_c = np.random.default_rng(int(args.seed) + 7919 * int(args.round))
            cand_pos = np.sort(rng_c.choice(cand_pos, cap, replace=False))
            print(f"[{src_tag}] candidate pool subsampled "
                  f"{int(mask.sum())}→{cap} (random, seed-dependent).")
        M_cand = M_valid[cand_pos]
        if len(M_cand) < int(args.batch):
            print(f"[{src_tag}] only {len(M_cand)} candidates left, "
                  f"reducing batch from {args.batch}")
        sel, info = _acquire(args, X_train, y_train, M_cand,
                             K=int(args.batch), acq=acq)
        I_extra = [int(cand_pos[s]) for s in sel]
        print(f"[{src_tag}] acq={acq} objective={info.get('objective')} "
              f"train={info['n_train']} cand={info['n_cand']} "
              f"surrogate={args.surrogate}")
        print(f"[{src_tag}] info=" + ", ".join(
            f"{k}={v}" for k, v in info.items()
            if k not in ('acq', 'objective', 'n_train', 'n_cand')))
        new_positions += I_extra
        new_sources += [src_tag] * len(I_extra)
    else:
        raise SystemExit(f"unknown --method {args.method}")

    if not new_positions:
        print("[sample] nothing to add (all picks already in archs.csv).")
        return

    # ────── materialise rows + append ──────
    header = _archs_header(comp_keys, expr_keys)
    new_rows = []
    for p, src in zip(new_positions, new_sources):
        nd_idx_row = valid_nd_idx[p]
        arch = build_arch(ctx.default_arch, expr_keys, _esm, nd_idx_row)
        complexity = get_net_info(arch, ctx.config, ctx.group_size,
                                  n_token=args.n_token)
        comp_vals = [complexity[ck] for ck in comp_keys]
        per_axis = [float(F[p, 1 + 2 * i]) for i in range(len(expr_keys))]
        new_rows.append(_build_row(
            idx=next_idx, round_id=int(args.round), source=src,
            nd_idx=nd_idx_row, arch=arch,
            comp_keys=list(comp_keys), comp_vals=comp_vals,
            expr_keys=expr_keys, per_axis_vals=per_axis,
            combined=float(F[p, 0])))
        next_idx += 1

    _append_rows(archs_csv, header, new_rows)
    print(f"[sample] appended {len(new_rows)} rows ({new_sources[0]}…) → "
          f"{archs_csv}  (total now {next_idx})")

    # meta.json (write once)
    meta_path = _meta_path(args)
    if not os.path.exists(meta_path):
        meta = dict(
            model_name=args.model_name, model_path=args.model_path,
            config=args.config, expr_keys=list(expr_keys),
            comp_keys=list(comp_keys),
            w_expr=args.w_expr, kv_expr=args.kv_expr,
            kvdim_expr=args.kvdim_expr, eff_kv_expr=args.eff_kv_expr,
            expr_front=args.expr_front, n_token=args.n_token,
            comp_obj=args.comp_obj, comp_obj_min=args.comp_obj_min,
            comp_obj_max=args.comp_obj_max,
            seed=args.seed, val_seed=args.val_seed,
            datasets=args.datasets, metric=args.metric,
            loss_func=args.loss_func, surrogate=args.surrogate,
        )
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[sample] wrote {meta_path}")


# ════════════════════════════════════════════════════════════════════════════
# cmd_eval
# ════════════════════════════════════════════════════════════════════════════
def cmd_eval(args):
    ctx = _build_ctx(args)
    archs_csv = _archs_path(args)
    if not os.path.exists(archs_csv):
        raise SystemExit(
            f"{archs_csv} missing — run --mode sample first.")
    row = _load_arch_row(archs_csv, args.idx)
    arch = json.loads(row['arch_json'])

    result_path = _result_path(args, args.idx)
    if os.path.exists(result_path) and not args.force:
        with open(result_path) as f:
            existing = json.load(f)
        mm = existing.get('measured_metric')
        if isinstance(mm, list) and mm and all(
                isinstance(v, (int, float)) and not np.isnan(v) for v in mm):
            print(f"[eval] idx={args.idx} already done — skip "
                  f"(--force to rerun)")
            return

    # expr_keys for per_axis_metric ordering
    meta_path = os.path.join(args.save or '.',
                             'validation_meta.json' if args.validation
                             else 'sample_meta.json')
    if not os.path.exists(meta_path):
        raise SystemExit(f"{meta_path} missing")
    with open(meta_path) as f:
        meta = json.load(f)
    expr_keys = meta['expr_keys']

    model_id = f'{args.model_path}/{args.model_name}'
    if 'hqq' not in args.w_method:
        args.quant_model_paths = []

    # ThinK pruning_dim scalar fallback (per-arch arch['p'] overrides)
    kpd = args.k_pruning_dim[0] if args.k_pruning_dim else 0
    vpd = args.v_pruning_dim[0] if args.v_pruning_dim else 0

    evaluator = LlamaEvaluator(
        ctx.config, accelerator=ctx.accelerator, model_id=model_id,
        method={'w': args.w_method, 'kv': args.kv_method},
        quant_model_paths=args.quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=args.seqlen, min_seqlen=args.min_seqlen, n_sample=args.n_sample,
        datasets=args.datasets, device_map=ctx.device_map, dtype=ctx.dtype,
        bits={'w': args.w_bits, 'k': args.k_bits, 'v': args.v_bits},
        group_size=ctx.group_size, residual_length=args.residual_length,
        k_quant_scheme=args.k_quant_scheme, v_quant_scheme=args.v_quant_scheme,
        k_pruning_dim=kpd, v_pruning_dim=vpd,
        loss_func=args.loss_func, last_tokens=args.last_tokens,
        use_key_token=args.use_key_token, trunc_len=args.trunc_len,
        sliding_window=args.sliding_window, alpha=args.alpha, beta=args.beta,
        key_token_path=args.key_token_path)

    complexity = get_net_info(arch, ctx.config, ctx.group_size,
                              n_token=args.n_token)
    print(f"[eval] idx={args.idx} source={row['source']} "
          f"round={row['round']}")
    print(f"[eval] complexity: {complexity}")
    ctx.accelerator.print(f"[eval] arch: {arch}")

    model = evaluator.sample(arch)
    metric_dict = evaluate_metric(args, arch, model, evaluator,
                                  ctx.accelerator)
    measured = [float(v) for v in metric_dict.values()]
    print(f"[eval] {args.metric}: {measured}  "
          f"per_axis={[float(row[f'metric_{k}']) for k in expr_keys]}  "
          f"combined={row['combined_metric']}")

    result = dict(
        idx=int(args.idx),
        arch=arch,
        complexity={k: float(v) for k, v in complexity.items()},
        datasets=list(args.datasets),
        measured_metric=measured,
        per_axis_metric=[float(row[f'metric_{k}']) for k in expr_keys],
        combined_metric=float(row['combined_metric']),
        source=row['source'],
        round=int(row['round']),
    )
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[eval] wrote {result_path}")


# ════════════════════════════════════════════════════════════════════════════
# cmd_aggregate
# ════════════════════════════════════════════════════════════════════════════
def _validation_metrics(args, X_tr, y_tr, X_val, y_val):
    M_bounds = np.vstack([X_tr, X_val]) if len(X_val) else X_tr
    m = _make_surrogate(args, X_tr, y_tr, M_bounds)
    yp = np.asarray(m.predict(X_val)).reshape(-1).astype(float)
    ss_r = float(np.sum((y_val - yp) ** 2))
    ss_t = float(np.sum((y_val - y_val.mean()) ** 2))
    r2 = 1.0 - ss_r / max(ss_t, 1e-30)
    rho = float(stats.spearmanr(yp, y_val).correlation)
    tau = float(stats.kendalltau(yp, y_val).correlation)
    return r2, rho, tau


def cmd_aggregate(args):
    save = args.save or '.'
    archs_csv = os.path.join(save, 'archs.csv')
    train_rows, _ = _load_archs_csv(archs_csv)
    if not train_rows:
        raise SystemExit(f"no rows in {archs_csv}")
    meta_path = os.path.join(save, 'sample_meta.json')
    if not os.path.exists(meta_path):
        raise SystemExit(f"{meta_path} missing")
    with open(meta_path) as f:
        meta = json.load(f)
    expr_keys = meta['expr_keys']
    comp_keys = meta['comp_keys']
    datasets = meta['datasets']
    n_datasets = max(1, len(datasets))

    # ── load completed train results ──
    train_data = []
    for r in train_rows:
        idx = int(r['idx'])
        path = os.path.join(save, f'result_{idx}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            res = json.load(f)
        mm = res.get('measured_metric')
        if not isinstance(mm, list) or not mm:
            continue
        train_data.append((idx, r, res))
    print(f"[aggregate] train: {len(train_data)}/{len(train_rows)} completed")

    sources_present = {r['source'] for _, r, _ in train_data}
    # Each non-quantile source becomes its own method, always paired with the
    # shared quantile warm-start. Discovers every al_<acq> source dynamically
    # (al_ei, al_alm, al_imse, al_maximin, al_qbc, al_rank) plus random / ga.
    method_groups = {
        src: [s for s in ('quantile', src) if s in sources_present]
        for src in sorted(sources_present) if src != 'quantile'
    }

    # ── results_<method>.csv (post_search.load_sample_csv layout) ──
    for method, srcs in method_groups.items():
        rs = [(i, r, res) for i, r, res in train_data if r['source'] in srcs]
        if not rs:
            continue
        out_path = os.path.join(save, f'results_{method}.csv')
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            # n_comp rows (complexity)
            for ck in comp_keys:
                w.writerow([res['complexity'].get(ck, '')
                            for _, _, res in rs])
            # n_datasets rows (measured metric)
            for di in range(n_datasets):
                w.writerow([res['measured_metric'][di]
                            if di < len(res['measured_metric']) else ''
                            for _, _, res in rs])
            # 1 row: combined predicted metric
            w.writerow([float(r['combined_metric']) for _, r, _ in rs])
            # n_axes rows: per-axis search metric
            for k in expr_keys:
                w.writerow([float(r[f'metric_{k}']) for _, r, _ in rs])
        print(f"[aggregate] wrote {out_path}  (n_archs={len(rs)} "
              f"sources={srcs})")

    # ── validation pool ──
    val_archs = os.path.join(save, 'validation_archs.csv')
    val_data = []
    if os.path.exists(val_archs):
        val_rows, _ = _load_archs_csv(val_archs)
        for r in val_rows:
            idx = int(r['idx'])
            path = os.path.join(save, f'validation_result_{idx}.json')
            if not os.path.exists(path):
                continue
            with open(path) as f:
                res = json.load(f)
            mm = res.get('measured_metric')
            if not isinstance(mm, list) or not mm:
                continue
            val_data.append((idx, r, res))
        print(f"[aggregate] validation: {len(val_data)}/{len(val_rows)} "
              f"completed")
    if not val_data:
        print("[aggregate] no validation completions — skipping "
              "validation_metrics.csv + learning_curve.csv")
        return

    X_val = np.asarray([[float(r[f'metric_{k}']) for k in expr_keys]
                        for _, r, _ in val_data], dtype=float)
    y_val = np.asarray([res['measured_metric'][0]
                        for _, _, res in val_data], dtype=float)

    # ── validation_metrics.csv (per method) ──
    val_metrics_rows = []
    for method, srcs in method_groups.items():
        rs = [(i, r, res) for i, r, res in train_data if r['source'] in srcs]
        if len(rs) < 5:
            print(f"[aggregate] {method}: <5 train rows ({len(rs)}), "
                  f"skipping validation fit")
            continue
        X_tr = np.asarray([[float(r[f'metric_{k}']) for k in expr_keys]
                           for _, r, _ in rs], dtype=float)
        y_tr = np.asarray([res['measured_metric'][0] for _, _, res in rs],
                          dtype=float)
        try:
            r2, rho, tau = _validation_metrics(args, X_tr, y_tr, X_val, y_val)
        except Exception as e:                              # noqa: BLE001
            print(f"[aggregate] {method}: surrogate fit failed: {e!r}")
            continue
        val_metrics_rows.append((method, len(rs), r2, rho, tau))

    out_path = os.path.join(save, 'validation_metrics.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'n_train', 'n_val', 'R2', 'Spearman', 'Kendall'])
        for method, n_tr, r2, rho, tau in val_metrics_rows:
            w.writerow([method, n_tr, len(val_data),
                        f"{r2:.6f}", f"{rho:.6f}", f"{tau:.6f}"])
    print(f"[aggregate] wrote {out_path}")
    print(f"[aggregate] validation summary (n_val={len(val_data)}):")
    for method, n_tr, r2, rho, tau in val_metrics_rows:
        print(f"  {method:<8}  n_tr={n_tr:<3}  R²={r2:+.4f}  "
              f"ρ={rho:+.4f}  τ={tau:+.4f}")

    # ── learning_curve.csv (per non-quantile source, cumulative-by-round) ──
    # Covers al_<acq> AND the random / ga baselines so every method has a curve.
    al_sources = sorted(s for s in sources_present if s != 'quantile')
    curve = []   # (source, round, n_train, r2, rho, tau)
    for al_src in al_sources:
        al_rs = [(i, r, res) for i, r, res in train_data
                 if r['source'] in ('quantile', al_src)]
        if not al_rs:
            continue
        for R in sorted({int(r['round']) for _, r, _ in al_rs}):
            rs = [(i, r, res) for i, r, res in al_rs if int(r['round']) <= R]
            if len(rs) < 5:
                continue
            X_tr = np.asarray([[float(r[f'metric_{k}']) for k in expr_keys]
                               for _, r, _ in rs], dtype=float)
            y_tr = np.asarray([res['measured_metric'][0] for _, _, res in rs],
                              dtype=float)
            try:
                r2, rho, tau = _validation_metrics(args, X_tr, y_tr, X_val, y_val)
            except Exception:                               # noqa: BLE001
                continue
            curve.append((al_src, R, len(rs), r2, rho, tau))
    if not curve:
        return

    out_path = os.path.join(save, 'learning_curve.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['acq', 'round', 'n_train', 'n_val', 'R2', 'Spearman',
                    'Kendall'])
        for acq_src, R, n_tr, r2, rho, tau in curve:
            w.writerow([acq_src, R, n_tr, len(val_data),
                        f"{r2:.6f}", f"{rho:.6f}", f"{tau:.6f}"])
    print(f"[aggregate] wrote {out_path}")
    print(f"[aggregate] AL learning curve (n_val={len(val_data)}):")
    for acq_src, R, n_tr, r2, rho, tau in curve:
        print(f"  {acq_src:<10} round {R}  n_tr={n_tr:<3}  R²={r2:+.4f}  "
              f"ρ={rho:+.4f}  τ={tau:+.4f}")


# ════════════════════════════════════════════════════════════════════════════
# Argparse
# ════════════════════════════════════════════════════════════════════════════
def build_parser():
    p = argparse.ArgumentParser(
        description='Three-mode surrogate-comparison pipeline '
                    '(random / GA / AL-EI) with held-out validation.')
    p.add_argument('--mode', choices=['sample', 'eval', 'aggregate'],
                   required=True)
    # ── method / round / batch ──
    p.add_argument('--method', choices=METHODS, default='quantile',
                   help='(sample) which method drives the new rows. '
                        '"quantile" writes warm-start anchors once (round 0); '
                        '"random"/"ga"/"al"/"al_ei" append --batch extras. '
                        '"al" uses --acq; "al_ei" == --method al --acq ei.')
    p.add_argument('--acq', choices=ACQS, default='ei',
                   help='(sample --method al) acquisition. Objective A (global '
                        'surrogate ranking): alm/imse/maximin/qbc/rank. '
                        'Objective B (lowest-JSD arch in band): ei/ucb. '
                        'imse needs --surrogate ard_gp (else falls back to alm). '
                        'maximin/qbc/rank are model-free / interpolant-friendly '
                        '(work with --surrogate rbf tps).')
    p.add_argument('--al_ucb_kappa', type=float, default=2.0,
                   help='(acq ucb) LCB = mu - kappa*sigma exploration weight.')
    p.add_argument('--al_qbc_B', type=int, default=20,
                   help='(acq qbc/rank, and alm/ucb when surrogate is not a GP) '
                        'bootstrap-ensemble size for the disagreement σ.')
    p.add_argument('--al_pool_cap', type=int, default=3000,
                   help='(acq imse) cap on the candidate/reference pool for the '
                        'O(n^2) posterior-cov ALC step (random subsample, logged).')
    p.add_argument('--al_diverse', action='store_true',
                   help='(acq alm/qbc/rank) spread the batch by maximin among '
                        'the top scorers so one sbatch array is not wasted on '
                        'near-identical archs. EI/UCB keep pure top-K.')
    p.add_argument('--al_transform', choices=['none', 'sqrt', 'log'],
                   default='none',
                   help='target transform applied to y when fitting the '
                        'acquisition surrogate (alm/imse/qbc/rank/ucb). sqrt '
                        'stabilises heavy-tailed JSD so the σ/cov signal is not '
                        'dominated by the high-JSD extreme (diagnosis: raw-Y '
                        'GP chases the tail). Does not affect ei.')
    p.add_argument('--round', type=int, default=0,
                   help='(sample) round id stored on each new row. AL '
                        'increments by 1 each acquisition step.')
    p.add_argument('--batch', type=int, default=8,
                   help='(sample) extras to add per call '
                        '(random / ga / al_ei).')
    p.add_argument('--validation', action='store_true',
                   help='(sample / eval) operate on validation_archs.csv / '
                        'validation_result_<idx>.json with --val_seed.')
    p.add_argument('--n_val', type=int, default=100,
                   help='(sample --validation) number of held-out archs.')
    p.add_argument('--val_seed', type=int, default=1000,
                   help='(sample --validation / eval --validation) RNG seed; '
                        'kept disjoint from --seed.')
    p.add_argument('--al_dataset_idx', type=int, default=0,
                   help='(sample al_ei / aggregate) which --datasets column '
                        'to use as y. Default 0 = first dataset.')
    p.add_argument('--idx', type=int, default=-1,
                   help='(eval) row idx in archs.csv to evaluate.')
    p.add_argument('--force', action='store_true',
                   help='(eval) recompute even if already in result_<idx>.json')

    # ── model / config ──
    p.add_argument('--model_path', type=str, default='')
    p.add_argument('--model_name', type=str, default='')
    p.add_argument('--config', type=str, default='config/llama.json')
    p.add_argument('--dtype', type=str, default='auto',
                   choices=['float16', 'float', 'fp16', 'bfloat16', 'bfloat',
                            'bf16', 'auto'])
    p.add_argument('--gpu_id', type=str, default='0')
    p.add_argument('--seed', type=int, default=0)

    # ── quant methods / bits ──
    p.add_argument('--w_method', type=str, nargs='+', default=[],
                   choices=['fp16', 'awq', 'gptq', 'qeft', 'hqq'])
    p.add_argument('--kv_method', type=str, nargs='+', default=['kivi'],
                   choices=['fp16', 'hqq', 'kivi', 'think'])
    p.add_argument('--quant_model_paths', type=str, nargs='+', default=[])
    p.add_argument('--w_bits', type=int, nargs='+', default=[])
    p.add_argument('--k_bits', type=int, nargs='+', default=[2, 4])
    p.add_argument('--v_bits', type=int, nargs='+', default=[2, 4])
    p.add_argument('--w_group_size', type=int, default=128)
    p.add_argument('--k_group_size', type=int, nargs='+', action='append',
                   default=[])
    p.add_argument('--v_group_size', type=int, nargs='+', action='append',
                   default=[])
    p.add_argument('--residual_length', type=int, default=128)
    p.add_argument('--k_quant_scheme', type=str, choices=['channel', 'token'])
    p.add_argument('--v_quant_scheme', type=str, choices=['channel', 'token'])
    p.add_argument('--k_pruning_dim', type=int, nargs='+', default=None)
    p.add_argument('--v_pruning_dim', type=int, nargs='+', default=None)
    p.add_argument('--outlier_path', type=str, default='')

    # ── calibration data / metric (eval mode) ──
    p.add_argument('--datasets', type=str, nargs='+', default=[])
    p.add_argument('--metric', type=str, default='loss')
    p.add_argument('--loss_func', type=str, default='jsd')
    p.add_argument('--stride', type=int, default=None)
    p.add_argument('--last_tokens', type=int, default=None)
    p.add_argument('--prefill_prompt', action='store_true')
    p.add_argument('--n_sample', type=int, default=128)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--min_seqlen', type=int, default=0)
    p.add_argument('--data_batch_size', type=int, default=1)
    p.add_argument('--n_token', type=int, default=0)

    # ── expr archives + combined-metric scales (sample mode) ──
    p.add_argument('--w_expr', type=str, default='')
    p.add_argument('--kv_expr', type=str, default='')
    p.add_argument('--kvdim_expr', type=str, default='')
    p.add_argument('--eff_kv_expr', type=str, default='')
    p.add_argument('--expr_front', action='store_true')
    p.add_argument('--sqrt', action='store_true')
    p.add_argument('--w_scale', type=float, default=1.0)
    p.add_argument('--kv_scale', type=float, default=1.0)
    p.add_argument('--kvdim_scale', type=float, default=1.0)
    p.add_argument('--eff_kv_scale', type=float, default=1.0)

    # ── comp_obj pre-filter ──
    p.add_argument('--comp_obj', type=str, nargs='+', default=[])
    p.add_argument('--comp_obj_min', type=float, nargs='+', default=[])
    p.add_argument('--comp_obj_max', type=float, nargs='+', default=[])

    # ── quantile / coverage_nsga2 knobs ──
    p.add_argument('--quantile_sample', type=str, nargs='+', default=[])
    p.add_argument('--sampling_method', type=str,
                   default='coverage_nsga2_combined',
                   choices=['random', 'coverage_nsga2_joint',
                            'coverage_nsga2_marginal',
                            'coverage_nsga2_combined'])
    p.add_argument('--coverage_coord', type=str, default='rank',
                   choices=['z', 'rank'])
    p.add_argument('--coverage_per_axis_agg', type=str, default='max',
                   choices=['max', 'sum', 'pareto'])
    p.add_argument('--coverage_pareto_select', type=str, default='knee',
                   choices=['auto', 'strategy3', 'knee'])

    # ── surrogate (AL EI + aggregate validation) ──
    p.add_argument('--surrogate', type=str, default='ard_gp', choices=SURROGATES,
                   help='Used by (a) AL EI mean predictor and σ_conf LOOCV, '
                        '(b) per-method aggregate validation fit. Default '
                        'ard_gp matches the planned EI study.')
    p.add_argument('--rbf_kernel', type=str, default='tps',
                   choices=['cubic', 'tps', 'linear'])
    p.add_argument('--ard_kernel', type=str, default='matern32',
                   choices=['rbf', 'matern52', 'matern32', 'rq'])
    p.add_argument('--gp_n_restarts', type=int, default=10)
    p.add_argument('--surrogate_device', type=str, default='auto')

    # ── long-ppl key-token options (eval mode) ──
    p.add_argument('--use_key_token', action='store_true')
    p.add_argument('--trunc_len', type=int, default=512)
    p.add_argument('--sliding_window', type=int, default=128)
    p.add_argument('--alpha', type=int, default=2)
    p.add_argument('--beta', type=int, default=-2)
    p.add_argument('--key_token_path', type=str, default='')

    # ── output ──
    p.add_argument('--save', type=str, default='',
                   help='output dir (archs.csv / result_*.json / '
                        'validation_archs.csv / results_<method>.csv / …).')
    return p


def main():
    args = build_parser().parse_args()
    if args.mode == 'sample':
        cmd_sample(args)
    elif args.mode == 'eval':
        if args.idx < 0:
            raise SystemExit("--mode eval requires --idx >= 0")
        cmd_eval(args)
    elif args.mode == 'aggregate':
        cmd_aggregate(args)
    else:
        raise SystemExit(f"unknown --mode {args.mode}")


if __name__ == '__main__':
    main()
