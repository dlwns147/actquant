"""search_mckp.py — Stage-1 per-method Pareto-frontier search by MEASURED DP-MCKP.

Drop-in alternative to search.py for ONE method/axis (--comp_obj wbits|kvbits|kvdim).
Same LlamaEvaluator / protocol as search.py (real JSD, --stride/--prefill_prompt/
--last_tokens), so every number below is a REAL measurement, not a prediction.

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

So the frontier in iter_mckp.stats carries MEASURED JSD and can be compared to a
search.py NSGA archive on actual values at matched comp.

Launch via scripts/search_mckp.sh (mirrors scripts/search.sh).
"""
import os, json, argparse, time
from copy import deepcopy
import numpy as np
from time import time as _t

from search import SearchThink
from utils.func import get_net_info, set_seed, init_accelerator

import warnings
warnings.simplefilter("ignore")


# ───────────────────────── per-axis module / option model ─────────────────────────
def build_axis_modules(engine, axis):
    """Return (ref_arch, modules) where modules is a list of dicts:
       {kind, loc, options(list of non-ref values), ref, apply(arch,val)}.
    Only modules whose option set has >1 entry are searchable; the rest stay at ref.
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
        for layer in range(nb):
            for kv, opts, ref in (('k', k_opts, ref_k), ('v', v_opts, ref_v)):
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
        raise SystemExit(f"search_mckp supports comp_obj in wbits/kvbits/kvdim, got {axis}")
    return ref_arch, modules


def arch_with(ref_arch, mod, val):
    a = deepcopy(ref_arch)
    mod['apply'](a, val)
    return a


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


def main():
    args = build_parser().parse_args()
    # capture before SearchThink (it pops keys out of the kwargs dict we pass)
    save_dir, front_points, dp_res = args.save, args.mckp_front_points, args.dp_res
    set_seed(args.seed)
    with open(args.config) as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    engine = SearchThink(config=config, accelerator=accelerator,
                         device_map=device_map, kwargs=dict(vars(args)))
    axis = engine.comp_obj[0]
    accelerator.print(f"[search_mckp] axis={axis}  n_token={engine.n_token}")

    ref_arch, modules = build_axis_modules(engine, axis)
    accelerator.print(f"[search_mckp] {len(modules)} searchable modules; "
                      f"{sum(len(m['options']) for m in modules)} marginal evals")

    comp_ref = get_net_info(ref_arch, config, engine.group_size, n_token=engine.n_token)[axis]

    # ---- measure baseline + every marginal (real JSD via engine._evaluate) ----
    eval_archs = [ref_arch]
    index = []                               # (m, j_in_d, val) for each marginal arch
    d = [[0.0] for _ in modules]             # d[m][0]=ref distortion 0
    cost = [[0.0] for _ in modules]          # cost[m][0]=ref delta 0
    for m, mod in enumerate(modules):
        for val in mod['options']:
            a = arch_with(ref_arch, mod, val)
            comp = get_net_info(a, config, engine.group_size, n_token=engine.n_token)[axis]
            d[m].append(None)                # filled after measuring
            cost[m].append(comp - comp_ref)
            index.append((m, len(d[m]) - 1, a))
    accelerator.print(f"[search_mckp] measuring {1 + len(index)} archs "
                      f"(1 ref + {len(index)} marginals)…")
    t0 = _t()
    metrics, _ = engine._evaluate(archs=eval_archs + [a for _, _, a in index],
                                  accelerator=accelerator)
    accelerator.print(f"[search_mckp] marginal measurement done ({_t()-t0:.1f}s)")
    base = metrics[0]
    for (m, j, _a), mval in zip(index, metrics[1:]):
        d[m][j] = mval - base                # measured marginal distortion

    # ---- DP-MCKP -> frontier allocations at a grid of comp budgets ----
    span = max(1e-9, comp_ref - min(comp_ref + sum(min(c) for c in cost), comp_ref))
    res = dp_res if dp_res > 0 else max(span / 400.0, 1e-6)
    t0 = _t(); dp = dp_mckp(d, cost, res); dp_t = _t() - t0
    pts = sorted((comp_ref + k, v[0], v[1]) for k, v in dp.items())
    # non-dominated lower envelope (comp asc, distortion strictly decreasing as comp rises)
    env, best = [], np.inf
    for comp, dist, ch in pts:
        if dist < best - 1e-12:
            best = dist; env.append((comp, dist, ch))
    accelerator.print(f"[search_mckp] DP-MCKP solved {len(dp)} comp-states "
                      f"({len(env)} frontier pts) in {dp_t*1000:.0f} ms (res={res:.4g})")

    # pick the frontier budgets to MEASURE within [comp_obj_min, comp_obj_max].
    # front_points <= 0  ->  use the ENTIRE DP-MCKP frontier (no subsampling).
    lo, hi = engine.comp_obj_min[0], engine.comp_obj_max[0]
    env = [e for e in env if lo - 1e-9 <= e[0] <= hi + 1e-9] or env
    if front_points > 0 and len(env) > front_points:
        idx = np.linspace(0, len(env) - 1, front_points).round().astype(int)
        env = [env[i] for i in sorted(set(idx))]

    # ---- MEASURE the real JSD of each frontier arch ----
    front_archs = []
    for comp, _dist, ch in env:
        a = deepcopy(ref_arch)
        for m, j in enumerate(ch):
            if j > 0:
                modules[m]['apply'](a, modules[m]['options'][j - 1])
        front_archs.append(a)
    accelerator.print(f"[search_mckp] measuring {len(front_archs)} frontier archs (real JSD)…")
    fmetrics, fcomp = engine._evaluate(archs=front_archs, accelerator=accelerator)

    archive = []
    for a, mtr, cmp in zip(front_archs, fmetrics, fcomp):
        archive.append([a, float(mtr), *[float(c) for c in cmp]])

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, 'iter_mckp.stats')
        with open(out, 'w') as f:
            json.dump({'archive': archive, 'candidates': [], 'iteration': 'mckp',
                       'axis': axis, 'comp_ref': comp_ref, 'base_jsd': base,
                       'n_marginal_evals': 1 + len(index),
                       'surrogate': {'model': 'measured_dp_mckp'}}, f)
        accelerator.print(f"[search_mckp] wrote {len(archive)} MEASURED frontier "
                          f"archs -> {out}")
        accelerator.print(f"\n{'comp':>12} | {'measured JSD':>12}")
        for a, mtr, cmp in zip(front_archs, fmetrics, fcomp):
            accelerator.print(f"{cmp[0]:>12.4f} | {mtr:>12.5f}")
        accelerator.print(f"\n[evals] MCKP total = {1+len(index)+len(front_archs)} "
                          f"(vs NSGA n_doe+iter*n_iter). base_jsd={base:.5f}")


def build_parser():
    # mirrors search.py's parser (subset needed by SearchThink) + MCKP knobs
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
                   help='Keep first S KV tokens in FP (KVSink). 0=off. Read by SearchThink via vars(args).')
    p.add_argument('--quant_kv_output', action='store_true')
    p.add_argument('--k_quant_scheme', type=str, default='channel')
    p.add_argument('--v_quant_scheme', type=str, default='token')
    p.add_argument('--comp_obj', type=str, nargs='+', default=['wbits'])
    p.add_argument('--comp_obj_min', type=float, nargs='+', default=[2])
    p.add_argument('--comp_obj_max', type=float, nargs='+', default=[5])
    p.add_argument('--k_pruning_dim', type=int, nargs='+', default=None)
    p.add_argument('--v_pruning_dim', type=int, nargs='+', default=None)
    # QEFT outlier-column axis (folds into the 'wbits' rate via compute_bits).
    # When n_qeft_column has a >0 entry, each weight linear's option becomes a
    # (w_bits, n_outlier) tuple; SearchThink reads these out of vars(args).
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
                   help='how many frontier budgets to MEASURE; <=0 = ENTIRE frontier')
    p.add_argument('--dp_res', type=float, default=0.0,
                   help='comp-budget DP bin width (0 = auto = span/400)')
    return p


if __name__ == '__main__':
    main()
