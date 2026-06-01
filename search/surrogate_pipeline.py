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
    ``--method al_ei``      Append ``--batch`` active-learning EI extras.
                             Refits the surrogate on completed ``result_*.json``.
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
from post_search import _make_surrogate
from predictor.factory import all_surrogates as _all_surrogates

warnings.simplefilter("ignore")

SURROGATES = _all_surrogates()
METHODS = ['quantile', 'random', 'ga', 'al_ei']


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

    elif args.method == 'al_ei':
        completed = _load_completed_results(
            args.save or '.', existing_rows,
            dataset_idx=int(args.al_dataset_idx))
        if not completed:
            raise SystemExit(
                "[al_ei] no completed result_*.json — the warm-start round "
                "(method=quantile / random / ga) must finish evaluating "
                "before AL round ≥ 1.")
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
        M_cand = M_valid[cand_pos]
        if len(M_cand) < int(args.batch):
            print(f"[al_ei] only {len(M_cand)} candidates left, "
                  f"reducing batch from {args.batch}")
        sel, info = _ei_acquire(args, X_train, y_train, M_cand,
                                K=int(args.batch))
        I_extra = [int(cand_pos[s]) for s in sel]
        print(f"[al_ei] train={info['n_train']} cand={info['n_cand']} "
              f"surrogate={args.surrogate}")
        print(f"[al_ei] σ_conf={info['sigma_conf']:.5f}  "
              f"B_ε={info['B_eps']:.5f}  y_min={info['y_min']:.5f}  "
              f"μ_range=[{info['mu_min']:.5f},{info['mu_max']:.5f}]  "
              f"max_EVI={info['evi_max']:.5f}")
        new_positions += I_extra
        new_sources += ['al_ei'] * len(I_extra)
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
    method_groups = {
        'random': [s for s in ('quantile', 'random') if s in sources_present],
        'ga':     [s for s in ('quantile', 'ga')     if s in sources_present],
        'al_ei':  [s for s in ('quantile', 'al_ei')  if s in sources_present],
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

    # ── learning_curve.csv (AL only, cumulative-by-round) ──
    al_rs = [(i, r, res) for i, r, res in train_data
             if r['source'] in ('quantile', 'al_ei')]
    if not al_rs:
        return
    rounds = sorted({int(r['round']) for _, r, _ in al_rs})
    curve = []
    for R in rounds:
        rs = [(i, r, res) for i, r, res in al_rs
              if int(r['round']) <= R]
        if len(rs) < 5:
            continue
        X_tr = np.asarray([[float(r[f'metric_{k}']) for k in expr_keys]
                           for _, r, _ in rs], dtype=float)
        y_tr = np.asarray([res['measured_metric'][0] for _, _, res in rs],
                          dtype=float)
        try:
            r2, rho, tau = _validation_metrics(args, X_tr, y_tr, X_val, y_val)
        except Exception:                                   # noqa: BLE001
            continue
        curve.append((R, len(rs), r2, rho, tau))

    out_path = os.path.join(save, 'learning_curve.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['round', 'n_train', 'n_val', 'R2', 'Spearman', 'Kendall'])
        for R, n_tr, r2, rho, tau in curve:
            w.writerow([R, n_tr, len(val_data),
                        f"{r2:.6f}", f"{rho:.6f}", f"{tau:.6f}"])
    print(f"[aggregate] wrote {out_path}")
    print(f"[aggregate] AL learning curve (n_val={len(val_data)}):")
    for R, n_tr, r2, rho, tau in curve:
        print(f"  round {R}  n_tr={n_tr:<3}  R²={r2:+.4f}  "
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
                        '"random"/"ga"/"al_ei" append --batch extras.')
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
