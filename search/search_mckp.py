"""search_mckp.py — Stage-1 per-method Pareto-frontier search by MEASURED DP-MCKP.

Drop-in alternative to search.py for ONE axis (--comp_obj wbits|kvbits|eff_kvbits|kvdim)
or for the JOINT 2-axis baseline (--comp_obj wbits eff_kvbits). Same LlamaEvaluator /
protocol as search.py (real JSD, --stride/--prefill_prompt/--last_tokens), so every
number below is a REAL measurement, not a prediction.

Procedure (exact MCKP under measured additive rate-distortion):
  1. reference arch = every searchable module at its MAX-quality option
     (max bits / no prune); measure its JSD once.
  2. for each module m and each lower-quality option o, measure the MARGINAL
     distortion d_m(o)=JSD(ref with m:=o)-JSD(ref) and the comp-cost delta.
     -> n_module*(n_option-1) real evals (vs NSGA's n_doe+iter*n_iter).
  3. size-aware DP-MCKP over the comp budget -> per-budget optimal allocation
     (exact, recovers non-convex frontier points).
  4. MEASURE each frontier arch's real JSD and write iter_mckp.stats
     (same layout as search.py, post_search-consumable).

JOINT mode (--comp_obj wbits eff_kvbits) — the additive-MCKP BASELINE for the
(loss, wbits, eff_kvbits) 3-objective search: because the two budget axes are
module-disjoint (wbits <- W linears only, eff_kvbits <- KV layers only) and the
distortion is additive (v5 decomposition), the joint problem decomposes EXACTLY
into two independent per-axis DP-MCKPs; the predicted 3D front is the product of
the two 1D envelopes, base + d_W(B_w) + d_KV(B_kv). Every product point is then
MEASURED (real JSD), and the measured-vs-additive gap (RMSE/rho/tau) is reported
— the additivity error IS the baseline's known failure mode (aggressive corner).
On the eff_kvbits axis each KV layer's option set is the (bits,gs) x ThinK-prune
PRODUCT (cost couples them multiplicatively, so they must be one MCKP module);
requires --kv_method to include 'think' when pruning is searched.

SPLIT 3-axis mode (--comp_obj wbits kvbits kvdim) — KV bits and ThinK dims as
SEPARATE budget axes (module-disjoint: kvbits <- per-layer (bits,gs) with prune
at ref 0, kvdim <- per-layer prune with bits at ref) -> three independent DPs,
6-pt/axis product front by default. The within-layer bits x prune interaction is
DROPPED here (it is measured exactly in the folded eff_kvbits mode) — comparing
the two modes quantifies that interaction.

Right-edge guards: marginal distortions are clamped at 0 inside the DP (d<0 =
noise; the DP otherwise harvests it into a below-ref envelope that truncates the
high-comp end — measured 3.52/4.25 on eff_kvbits) and the exact ref corner is
anchored onto each axis envelope. Raw (unclamped) marginals are persisted in
iter_mckp.stats['marginals'] so the DP/front can be re-solved offline without
re-measuring.

So the frontier in iter_mckp.stats carries MEASURED JSD and can be compared to a
search.py / second_search.py archive on actual values at matched comp.

Launch via scripts/search_mckp.sh (mirrors scripts/search.sh).
"""
import os, json, argparse, itertools
from copy import deepcopy
import numpy as np
from time import time as _t

from search import Search
from utils.func import get_net_info, set_seed, init_accelerator, get_correlation

import warnings
warnings.simplefilter("ignore")


# ───────────────────────── per-axis module / option model ─────────────────────────
def build_axis_modules(engine, axis):
    """Return (ref_arch, modules) where modules is a list of dicts:
       {kind, loc, options(list of non-ref values), ref, apply(arch,val)}.
    Only modules whose option set has >1 entry are searchable; the rest stay at ref.
    eff_kvbits: per-layer option = ((bits,gs), prune) PRODUCT when pruning is
    searched — eff cost is (bits+32/gs)*(1-p/head_dim), multiplicative across the
    two knobs, so they must be a single MCKP module to keep costs exact/additive.
    """
    cfg = engine.config
    nb = cfg['n_block']
    linears = cfg['linear']
    ss = engine.search_space

    # option lists from the search space
    w_opts = {lin: sorted(getattr(ss, f"{lin.split('.')[-1]}_option")) for lin in linears}
    k_opts = [list(o) for o in ss.k_option]
    v_opts = [list(o) for o in ss.v_option]
    kp_opts = sorted(ss.k_pruning_dim_option)
    vp_opts = sorted(ss.v_pruning_dim_option)
    ref_w = {lin: max(w_opts[lin]) for lin in linears}
    ref_k = max(k_opts, key=lambda t: (t[0], t[1]))
    ref_v = max(v_opts, key=lambda t: (t[0], t[1]))
    ref_kp, ref_vp = min(kp_opts), min(vp_opts)        # 0 = no prune = max quality

    # QEFT weight options are (bits, n_outlier) tuples; compute_bits/compute_memory
    # accept a LIST [bits, n_outlier] (not a tuple), so normalize on write — mirrors
    # the search space's sample() storing list(mx). Plain int bits pass through.
    _wnorm = lambda v: list(v) if isinstance(v, tuple) else v

    ref_arch = {'q': {'w': {lin: [_wnorm(ref_w[lin]) for _ in range(nb)] for lin in linears},
                      'k': [list(ref_k) for _ in range(nb)],
                      'v': [list(ref_v) for _ in range(nb)]},
                'p': {'k': [ref_kp] * nb, 'v': [ref_vp] * nb}}

    modules = []
    if axis == 'wbits':
        for lin in linears:
            others = [b for b in w_opts[lin] if b != ref_w[lin]]
            if not others:
                continue
            for blk in range(nb):
                modules.append(dict(kind='w', loc=(lin, blk), ref=ref_w[lin],
                                    options=others,
                                    apply=lambda a, val, _l=lin, _b=blk: a['q']['w'][_l].__setitem__(_b, _wnorm(val))))
    elif axis in ('kvbits', 'eff_kvbits'):
        # eff_kvbits folds ThinK pruning into the rate (include_pruning=True), so when a
        # prune grid is given the per-layer module options are the (bits,gs) x prune product.
        prune_searched = axis == 'eff_kvbits' and (len(kp_opts) > 1 or len(vp_opts) > 1)
        for layer in range(nb):
            for kv, opts, ref, p_opts, p_ref in (('k', k_opts, ref_k, kp_opts, ref_kp),
                                                 ('v', v_opts, ref_v, vp_opts, ref_vp)):
                if prune_searched:
                    ref_combo = (tuple(ref), p_ref)
                    others = [(tuple(o), p) for o in opts for p in p_opts
                              if (tuple(o), p) != ref_combo]
                    if not others:
                        continue
                    modules.append(dict(kind=kv + 'e', loc=layer, ref=ref_combo, options=others,
                                        apply=lambda a, val, _kv=kv, _l=layer: (
                                            a['q'][_kv].__setitem__(_l, list(val[0])),
                                            a['p'][_kv].__setitem__(_l, val[1]))))
                else:
                    others = [o for o in opts if o != ref]
                    if not others:
                        continue
                    modules.append(dict(kind=kv, loc=layer, ref=ref, options=others,
                                        apply=lambda a, val, _kv=kv, _l=layer: a['q'][_kv].__setitem__(_l, list(val))))
    elif axis == 'kvdim':
        for layer in range(nb):
            for kv, opts, ref in (('k', kp_opts, ref_kp), ('v', vp_opts, ref_vp)):
                others = [o for o in opts if o != ref]
                if not others:
                    continue
                modules.append(dict(kind=kv + 'p', loc=layer, ref=ref, options=others,
                                    apply=lambda a, val, _kv=kv, _l=layer: a['p'][_kv].__setitem__(_l, val)))
    else:
        raise SystemExit(f"search_mckp supports comp_obj in wbits/kvbits/eff_kvbits/kvdim, got {axis}")
    return ref_arch, modules


def arch_with(ref_arch, mod, val):
    a = deepcopy(ref_arch)
    mod['apply'](a, val)
    return a


def apply_choices(arch, modules, choices):
    """Apply a DP choice vector (index 0 == ref) onto `arch` IN PLACE."""
    for m, j in enumerate(choices):
        if j > 0:
            modules[m]['apply'](arch, modules[m]['options'][j - 1])
    return arch


# ───────────────────────── DP-MCKP over the comp budget ─────────────────────────
def dp_mckp(d, cost, res):
    """d[m][j], cost[m][j] (comp delta vs ref, <=0). Each module picks one option
    (index 0 == ref: d=0,cost=0). DP over binned cumulative comp delta.
    Returns {binned_delta: (min_total_distortion, [chosen j per module])}."""
    cur = {0.0: (0.0, [])}
    for m in range(len(d)):
        nxt = {}
        for key, (dist, ch) in cur.items():
            for j in range(len(d[m])):
                nk = round((key + cost[m][j]) / res) * res
                nd = dist + d[m][j]
                if nk not in nxt or nd < nxt[nk][0]:
                    nxt[nk] = (nd, ch + [j])
        cur = nxt
    return cur


def solve_axis(d, cost, comp_ref, dp_res, lo, hi):
    """Per-axis DP-MCKP -> non-dominated (comp, distortion, choices) envelope,
    filtered to the [lo, hi] budget box (falls back to the full envelope if the
    box is empty). Two right-edge guards (measured: without them the eff_kvbits
    envelope truncated at 3.52/4.25 — the DP harvested noise-negative marginals
    into a below-ref distortion that dominated the whole high-comp band):
      * marginals are clamped at 0 — quality cannot improve by quantizing or
        pruning MORE, so d<0 is measurement noise, not signal;
      * the exact ref corner (comp_ref, 0, all-ref) is anchored onto the
        envelope so the achievable right edge is always represented
        (anchor_endpoints analogue).
    Returns (envelope, res, n_dp_states)."""
    d = [[max(0.0, x) for x in row] for row in d]
    span = max(1e-9, -sum(min(c) for c in cost))   # min(c) <= 0 (index 0 = ref = 0)
    res = dp_res if dp_res > 0 else max(span / 400.0, 1e-6)
    dp = dp_mckp(d, cost, res)
    pts = sorted((comp_ref + k, v[0], v[1]) for k, v in dp.items())
    # non-dominated lower envelope (comp asc, distortion strictly decreasing as comp rises)
    env, best = [], np.inf
    for comp, dist, ch in pts:
        if dist < best - 1e-12:
            best = dist; env.append((comp, dist, ch))
    env = [e for e in env if lo - 1e-9 <= e[0] <= hi + 1e-9] or env
    if lo - 1e-9 <= comp_ref <= hi + 1e-9 and comp_ref - env[-1][0] > 1e-9:
        env.append((comp_ref, 0.0, [0] * len(d)))
    return env, res, len(dp)


def subsample_env(env, n_points):
    """Evenly thin an envelope to n_points (<=0 keeps all)."""
    if n_points > 0 and len(env) > n_points:
        idx = np.linspace(0, len(env) - 1, n_points).round().astype(int)
        env = [env[i] for i in sorted(set(idx))]
    return env


def main():
    args = build_parser().parse_args()
    # capture before Search (it pops keys out of the kwargs dict we pass)
    save_dir, front_points, dp_res = args.save, args.mckp_front_points, args.dp_res
    set_seed(args.seed)
    with open(args.config) as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    engine = Search(config=config, accelerator=accelerator,
                         device_map=device_map, kwargs=dict(vars(args)))
    axes = list(engine.comp_obj)
    if not 1 <= len(axes) <= 3:
        raise SystemExit(f"search_mckp supports 1-3 comp_obj axes, got {axes}")
    if 'eff_kvbits' in axes and ({'kvbits', 'kvdim'} & set(axes)):
        raise SystemExit("[search_mckp] eff_kvbits claims the same KV modules as kvbits/kvdim — "
                         "use EITHER the folded axis (wbits eff_kvbits) OR the split axes "
                         "(wbits kvbits kvdim), not both.")
    accelerator.print(f"[search_mckp] axes={axes}  n_token={engine.n_token}")

    # ThinK gate guard: prune-searching archs silently NO-OP unless 'think' is in
    # kv_method (enable_think), corrupting every prune marginal — fail fast instead.
    ss = engine.search_space
    if ({'eff_kvbits', 'kvdim'} & set(axes)) \
            and (max(ss.k_pruning_dim_option) > 0 or max(ss.v_pruning_dim_option) > 0) \
            and 'think' not in engine.method['kv']:
        raise SystemExit("[search_mckp] the prune axis needs ThinK but --kv_method lacks "
                         "'think' — prune archs would silently no-op. Pass --kv_method kivi think.")

    ref_arch, mods_per_axis = None, []
    for axis in axes:
        ra, mods = build_axis_modules(engine, axis)
        ref_arch = ra if ref_arch is None else ref_arch   # identical construction per axis
        mods_per_axis.append(mods)
        accelerator.print(f"[search_mckp] axis={axis}: {len(mods)} searchable modules; "
                          f"{sum(len(m['options']) for m in mods)} marginal evals")

    comp_ref = {axis: get_net_info(ref_arch, config, engine.group_size, n_token=engine.n_token,
                                   attn_sink=engine.attn_sink)[axis] for axis in axes}

    # ---- measure baseline + every marginal (real JSD via engine._evaluate) ----
    index = []                                   # (axis_idx, m, j_in_d, arch) per marginal
    d = [[[0.0] for _ in mods] for mods in mods_per_axis]     # d[ai][m][0]=ref distortion 0
    cost = [[[0.0] for _ in mods] for mods in mods_per_axis]  # cost[ai][m][0]=ref delta 0
    for ai, (axis, mods) in enumerate(zip(axes, mods_per_axis)):
        for m, mod in enumerate(mods):
            for val in mod['options']:
                a = arch_with(ref_arch, mod, val)
                comp = get_net_info(a, config, engine.group_size, n_token=engine.n_token,
                                    attn_sink=engine.attn_sink)[axis]
                d[ai][m].append(None)            # filled after measuring
                cost[ai][m].append(comp - comp_ref[axis])
                index.append((ai, m, len(d[ai][m]) - 1, a))
    accelerator.print(f"[search_mckp] measuring {1 + len(index)} archs "
                      f"(1 ref + {len(index)} marginals)…")
    t0 = _t()
    metrics, _ = engine._evaluate(archs=[ref_arch] + [a for *_, a in index],
                                  accelerator=accelerator)
    accelerator.print(f"[search_mckp] marginal measurement done ({_t()-t0:.1f}s)")
    base = metrics[0]
    for (ai, m, j, _a), mval in zip(index, metrics[1:]):
        d[ai][m][j] = mval - base                # measured marginal distortion
    neg = [x for dd in d for row in dd for x in row if x < 0]
    if neg:
        accelerator.print(f"[search_mckp] {len(neg)} negative marginals (min {min(neg):.2e}) "
                          f"= noise floor; clamped to 0 inside the DP (raw values kept in stats)")

    # ---- per-axis DP-MCKP -> frontier envelopes within [comp_obj_min, comp_obj_max] ----
    # front_points <= 0  ->  single-axis: the ENTIRE DP-MCKP frontier (no subsampling);
    #                        joint: 16 pts per axis (the full product would explode).
    envs = []
    for ai, axis in enumerate(axes):
        lo, hi = engine.comp_obj_min[ai], engine.comp_obj_max[ai]
        t0 = _t()
        env, res, n_states = solve_axis(d[ai], cost[ai], comp_ref[axis], dp_res, lo, hi)
        accelerator.print(f"[search_mckp] axis={axis}: DP-MCKP solved {n_states} comp-states "
                          f"({len(env)} frontier pts) in {(_t()-t0)*1000:.0f} ms (res={res:.4g})")
        n_pts = front_points
        if len(axes) > 1 and n_pts <= 0:
            n_pts = {2: 16, 3: 6}[len(axes)]   # measured combos = n_pts ** n_axes
            accelerator.print(f"[search_mckp] joint mode: thinning {axis} envelope to {n_pts} pts "
                              f"(set --mckp_front_points to override)")
        envs.append(subsample_env(env, n_pts))

    # ---- assemble frontier archs (product over axes; additive prediction) + MEASURE ----
    front_archs, pred = [], []
    for combo in itertools.product(*envs):
        a = deepcopy(ref_arch)
        for mods, (_comp, _dist, ch) in zip(mods_per_axis, combo):
            apply_choices(a, mods, ch)
        front_archs.append(a)
        pred.append(base + sum(dist for _comp, dist, _ch in combo))
    accelerator.print(f"[search_mckp] measuring {len(front_archs)} frontier archs (real JSD)…")
    fmetrics, fcomp = engine._evaluate(archs=front_archs, accelerator=accelerator)

    archive = []
    for a, mtr, cmp in zip(front_archs, fmetrics, fcomp):
        archive.append([a, float(mtr), *[float(c) for c in cmp]])

    # additive-vs-measured gap on the frontier — in joint mode this quantifies the
    # cross-axis interaction the additive baseline ignores (its known failure mode).
    rmse, rho, tau = get_correlation(np.array(pred), np.array(fmetrics))

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, 'iter_mckp.stats')
        with open(out, 'w') as f:
            json.dump({'archive': archive, 'candidates': [], 'iteration': 'mckp',
                       'axis': axes[0] if len(axes) == 1 else axes, 'axes': axes,
                       'comp_ref': comp_ref[axes[0]] if len(axes) == 1 else comp_ref,
                       'base_jsd': base, 'n_marginal_evals': 1 + len(index),
                       'additive_pred': [float(p) for p in pred],
                       'additivity': {'rmse': float(rmse), 'rho': float(rho), 'tau': float(tau)},
                       # RAW (unclamped) per-module marginal measurements — the expensive part.
                       # Lets the DP / envelope / front grid be re-solved OFFLINE (different box,
                       # dp_res, front_points, clamping...) without re-measuring ~2k archs.
                       'marginals': [
                           {'axis': axis,
                            'modules': [{'kind': mod['kind'], 'loc': mod['loc'], 'ref': mod['ref'],
                                         'options': mod['options'],
                                         'd': [float(x) for x in dd[1:]],
                                         'cost': [float(x) for x in cc[1:]]}
                                        for mod, dd, cc in zip(mods, d[ai], cost[ai])]}
                           for ai, (axis, mods) in enumerate(zip(axes, mods_per_axis))],
                       'surrogate': {'model': 'measured_dp_mckp'}}, f)
        accelerator.print(f"[search_mckp] wrote {len(archive)} MEASURED frontier "
                          f"archs -> {out}")
        accelerator.print("\n" + " | ".join(f"{axis:>12}" for axis in axes)
                          + f" | {'measured JSD':>12} | {'additive pred':>13}")
        for (a, mtr, *cmp), p in zip(archive, pred):
            accelerator.print(" | ".join(f"{c:>12.4f}" for c in cmp)
                              + f" | {mtr:>12.5f} | {p:>13.5f}")
        accelerator.print(f"\n[evals] MCKP total = {1+len(index)+len(front_archs)} "
                          f"(vs NSGA n_doe+iter*n_iter). base_jsd={base:.5f}")
        accelerator.print(f"[additivity] measured vs additive on the frontier: "
                          f"RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall's Tau = {tau:.4f}")


def build_parser():
    # mirrors search.py's parser (subset needed by Search) + MCKP knobs
    p = argparse.ArgumentParser()
    p.add_argument('--save', type=str, default='save/mckp')
    p.add_argument('--gpu_id', type=str, default='0')
    p.add_argument('--model_path', type=str, default='')
    p.add_argument('--model_name', type=str, default='')
    p.add_argument('--dtype', type=str, default='auto')
    p.add_argument('--quant_model_paths', type=str, nargs='+', default=[])
    p.add_argument('--w_method', type=str, nargs='+', default=[])
    p.add_argument('--kv_method', type=str, nargs='+', default=['kivi'])
    p.add_argument('--w_bits', type=int, nargs='+', default=[])
    p.add_argument('--k_bits', type=int, nargs='+', default=[2, 4])
    p.add_argument('--v_bits', type=int, nargs='+', default=[2, 4])
    p.add_argument('--w_group_size', type=int, default=128)
    p.add_argument('--k_group_size', type=int, nargs='+', action='append', default=[])
    p.add_argument('--v_group_size', type=int, nargs='+', action='append', default=[])
    p.add_argument('--residual_length', type=int, default=128)
    p.add_argument('--attn_sink', type=int, default=0,
                   help='Keep first S KV tokens in FP (KVSink). 0=off. Read by Search via vars(args).')
    p.add_argument('--quant_kv_output', action='store_true')
    p.add_argument('--k_quant_scheme', type=str, default='channel')
    p.add_argument('--v_quant_scheme', type=str, default='token')
    p.add_argument('--comp_obj', type=str, nargs='+', default=['wbits'],
                   help='1 axis (wbits/kvbits/eff_kvbits/kvdim) or a JOINT baseline: '
                        '"wbits eff_kvbits" (KV bits x prune folded, per-layer product options) '
                        'or "wbits kvbits kvdim" (split axes; kvbits marginals at prune=0, kvdim '
                        'at 4bit — the within-layer bits x prune interaction is dropped). '
                        'Per-axis DP-MCKPs, additive product front, every point re-measured.')
    p.add_argument('--comp_obj_min', type=float, nargs='+', default=[2])
    p.add_argument('--comp_obj_max', type=float, nargs='+', default=[5])
    p.add_argument('--k_pruning_dim', type=int, nargs='+', default=None)
    p.add_argument('--v_pruning_dim', type=int, nargs='+', default=None)
    # QEFT outlier-column axis (folds into the 'wbits' rate via compute_bits).
    # When n_qeft_column has a >0 entry, each weight linear's option becomes a
    # (w_bits, n_outlier) tuple; Search reads these out of vars(args).
    p.add_argument('--outlier_path', type=str, default='')
    p.add_argument('--base_outlier_bits', type=int, nargs='+', default=[])
    p.add_argument('--n_outlier', type=int, default=0)
    p.add_argument('--n_qeft_column', type=int, nargs='+', default=[0])
    p.add_argument('--only_outlier_bits', action='store_true')
    p.add_argument('--dataset', type=str, default='wikitext2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_sample', type=int, default=128)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--min_seqlen', type=int, default=0)
    p.add_argument('--metric', type=str, default='loss')
    p.add_argument('--data_batch_size', type=int, default=1)
    p.add_argument('--config', type=str, default='config/llama.json')
    p.add_argument('--loss_func', type=str, default='jsd')
    p.add_argument('--max_value', type=float, default=0.7)
    p.add_argument('--n_token', type=int, default=0)
    p.add_argument('--stride', type=int, default=0)
    p.add_argument('--last_tokens', type=int, default=None)
    p.add_argument('--prefill_prompt', action='store_true')
    p.add_argument('--verbosity', type=str, default='FATAL')
    # MCKP knobs
    p.add_argument('--mckp_front_points', type=int, default=0,
                   help='frontier budgets to MEASURE per axis; <=0 = ENTIRE frontier '
                        '(single-axis) / 16 per axis (2-axis joint) / 6 per axis (3-axis); '
                        'measured combos = product over axes')
    p.add_argument('--dp_res', type=float, default=0.0,
                   help='comp-budget DP bin width (0 = auto = span/400, per axis)')
    return p


if __name__ == '__main__':
    main()
