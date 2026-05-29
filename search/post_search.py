"""post_search.py — stage 2 of the two-stage post-search.

Reads the results.csv produced by sample_surrogate.py, fits a surrogate via
predictor.factory.get_predictor (--surrogate: rbf / gp / mlp / carts / as),
re-ranks the full combo space with the surrogate's prediction, selects the
final architecture(s) under the COMP_OBJ range (prefer / high_tradeoff /
top-n), then evaluates + benchmarks them.

If --sample_path is omitted, the additive combined metric (args scales,
same as the old post_search_split.py) is used instead of a fitted surrogate.
"""
import os
import json
import csv
import argparse
import warnings
from time import time

import torch
import numpy as np
import scipy.stats as stats
from pymoo.decomposition.asf import ASF
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tqdm import tqdm

from evaluator import LlamaEvaluator
from predictor.factory import get_predictor
from utils.func import (init_run, build_expr_map, build_nd, comp_key_order,
                        evaluate_metric, configure_model_cache, get_net_info,
                        clean_up, _LazyComp)
from utils.select import (LazyPs, build_arch, assemble_F, select_valid_nd_idx,
                          per_axis_metric)
from utils.eval import eval_zeroshot
from utils.longbench import pred_longbench, eval_longbench
from utils.data import get_tokenizer
from utils.ruler import eval_ruler
from utils.longeval import eval_longeval_lines, generate_lines_testcases
from utils.minilongbench import pred_minilongbench, eval_minilongbench

warnings.simplefilter("ignore")

SURROGATES = ('rbf', 'gp', 'mlp', 'carts', 'as', 'ard_gp',
              'badd_quad', 'gam', 'sqrty_gp')


def _resolve_surrogate_device(spec):
    """'auto' → cuda when visible else cpu; otherwise pass spec through
    (the rbf/ard_gp predictors fall back to cpu themselves if cuda is
    requested but unavailable)."""
    if spec == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return spec


def _make_surrogate(args, X, y, M_valid):
    """Fit the chosen surrogate via predictor.factory.get_predictor
    (ard_gp lives in predictor/ard_gp.py). The pure-PyTorch rbf / ard_gp
    surrogates run on --surrogate_device (GPU/CPU selectable); the other
    predictors ignore the device kwarg."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    device = _resolve_surrogate_device(args.surrogate_device)
    kw = {}
    if args.surrogate == 'rbf':
        lb = np.minimum(X.min(0), M_valid.min(0))
        ub = np.maximum(X.max(0), M_valid.max(0))
        kw = dict(kernel=args.rbf_kernel, tail='linear',
                  lb=lb, ub=np.where(ub > lb, ub, lb + 1e-9))
    elif args.surrogate == 'ard_gp':
        kw = dict(ard_kernel=args.ard_kernel, gp_n_restarts=args.gp_n_restarts)
    elif args.surrogate == 'sqrty_gp':
        # Same MLE knobs as ard_gp (kernel + restarts).
        kw = dict(ard_kernel=args.ard_kernel, gp_n_restarts=args.gp_n_restarts)
    return get_predictor(args.surrogate, X, y, device=device, **kw)


# ───────────────────────── results.csv reader ─────────────────────────
def load_sample_csv(path, n_comp, n_axes, n_datasets=None):
    """Parse a sample_surrogate results.csv → (X, y, y_combined, n_valid).

    X (N, n_axes) per-axis search metric, y (N,) measured metric on the first
    dataset. NaN-metric columns (in-progress runs) are dropped, matching
    analysis/v5/_common.extract_xy.

    n_datasets is auto-detected from the CSV row count
    (total rows = n_comp + n_datasets + 1 + n_axes) so post_search.py can use
    a different --datasets set than the one used at sampling time. Pass an
    explicit value only to override.
    """
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    ncol = max(len(r) for r in rows)
    M = np.full((len(rows), ncol), np.nan)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            try:
                M[i, j] = float(v)
            except ValueError:
                pass
    n_rows = M.shape[0]
    if n_datasets is None:
        n_datasets = n_rows - n_comp - 1 - n_axes
        if n_datasets < 1:
            raise SystemExit(
                f"[load_sample_csv] CSV at {path} has {n_rows} rows but the "
                f"layout requires n_comp({n_comp}) + n_datasets(>=1) + 1 "
                f"combined + n_axes({n_axes}) = at least "
                f"{n_comp + 2 + n_axes} rows. Either the CSV is truncated or "
                f"n_comp/n_axes don't match how it was generated.")
        print(f"[load_sample_csv] auto-detected n_datasets={n_datasets} "
              f"from CSV rows={n_rows} (n_comp={n_comp}, n_axes={n_axes})")
    y_row = n_comp                    # first --datasets metric
    comb_row = n_comp + n_datasets    # pf column 0 (combined predicted)
    axis0 = comb_row + 1              # per-axis metrics start here
    y_all = M[y_row, :ncol]
    valid = ~np.isnan(y_all)
    X = M[axis0:axis0 + n_axes, :ncol].T[valid]
    y = y_all[valid]
    y_comb = M[comb_row, :ncol][valid] if M.shape[0] > comb_row else X.sum(1)
    return dict(X=X, y=y, y_combined=y_comb, n_valid=int(valid.sum()))


# ───────────────────────── LongEval testcase gen ─────────────────────────
def maybe_generate_testcases(args):
    """Handle --generate_testcases; returns True if the caller should exit."""
    if not args.generate_testcases:
        return False
    print("Generating LongEval testcases...")
    generate_lines_testcases(
        num_lines_list=args.generate_testcases_num_lines,
        num_test_samples=args.generate_testcases_num_samples,
        line_idx_opt=args.generate_testcases_line_idx_opt,
        output_dir=args.generate_testcases_output_dir)
    print("Testcase generation completed.")
    return bool(args.generate_testcases_only)


# ───────────────────────── benchmarks ─────────────────────────
def run_benchmarks(args, model, model_id):
    """passkey / zeroshot / longbench / minilongbench / ruler / longeval for
    one (already sampled) architecture. Each block needs the KV cache on."""
    if args.pass_key_file:
        clean_up()
        configure_model_cache(args, model, use_cache=True)
        print("-----------------------------------")
        enc = get_tokenizer(model_id)
        for line in open(args.pass_key_file, "r"):
            clean_up()
            torch.cuda.reset_max_memory_allocated()
            example = json.loads(line)
            prompt_postfix = "What is the pass key? The pass key is "
            prompt = example["input"] + prompt_postfix
            input_ids = enc(prompt, return_tensors="pt").input_ids.cuda()
            print("-----------------------------------")
            print(f"#Tokens of Prompt:", input_ids.shape[1], end=" ")
            print("Passkey target:", example["target"])
            tokens = model.generate(input_ids,
                                    max_new_tokens=len(example["target"]))
            answer = prompt_postfix + enc.decode(
                tokens[0].tolist()[input_ids.shape[1]:],
                skip_special_tokens=True)
            answer = f"[ {answer.replace(chr(10), chr(92) + 'n')} ]"
            peak = torch.cuda.max_memory_allocated()
            print(answer)
            print(f"Mem: {peak / 1024 / 1024 / 1024:.3f} GB")
            print("-----------------------------------\n")

    if args.zeroshot:
        clean_up()
        configure_model_cache(args, model, use_cache=True)
        results = eval_zeroshot(model, tokenizer=get_tokenizer(model_id),
                                task_list=args.tasks,
                                batch_size=args.lm_eval_batch_size)
        total_result = []
        print(f'task : {list(results.keys())}')
        for task, result in results.items():
            new_result = {k: float(v) for k, v in result.items()
                          if k in ['em,none', 'exact_match,strict-match',
                                   'exact_match,flexible-extract',
                                   'bleu_max,none', 'bleu_acc,none',
                                   'acc,none']}
            print(f'task: {task}, result: {list(new_result.keys())}, '
                  f'{list(new_result.values())}')
            total_result += list(new_result.values())
        print(f'total_result: {total_result}')

    if args.longbench:
        clean_up()
        configure_model_cache(args, model, use_cache=True)
        t0 = time()
        pred_longbench(model, tokenizer=get_tokenizer(model_id),
                       save_path=args.longbench_result_path,
                       longbench_config=args.longbench_config,
                       e=args.longbench_e, model_name=args.model_name)
        eval_longbench(args.longbench_result_path, args.longbench_e)
        sentences = [f"{k}: {v}\n" for k, v in vars(args).items()]
        sentences += [f'Longbench Time: {time() - t0:.2f}s', "\n"]
        with open(os.path.join(args.longbench_result_path,
                               "pred_e" if args.longbench_e else "pred",
                               'result.txt'), 'w') as f:
            f.writelines(sentences)

    if args.minilongbench:
        clean_up()
        configure_model_cache(args, model, use_cache=True)
        t0 = time()
        pred_minilongbench(
            model, tokenizer=get_tokenizer(model_id),
            save_path=args.minilongbench_result_path,
            longbench_config=args.longbench_config,
            data_dir=args.minilongbench_data_dir or None,
            model_name=args.model_name)
        eval_minilongbench(args.minilongbench_result_path)
        mlb_time = time() - t0
        print(f'MiniLongBench Time: {mlb_time:.2f}s')
        sentences = [f"{k}: {v}\n" for k, v in vars(args).items()]
        sentences.append(f'MiniLongBench Time: {mlb_time:.2f}s\n')
        with open(os.path.join(args.minilongbench_result_path, "pred",
                               "result.txt"), 'w') as f:
            f.writelines(sentences)

    if args.ruler:
        clean_up()
        configure_model_cache(args, model, use_cache=True)
        t0 = time()
        eval_ruler(model, tokenizer=get_tokenizer(model_id), model_id=model_id,
                   tasks=args.ruler_task, yaml_path=args.ruler_yaml_path,
                   batch_size=args.ruler_batch_size, length=args.ruler_length,
                   nsample=args.ruler_sample, gen_toks=args.ruler_gen_toks,
                   result_path=args.ruler_result_path, seed=args.seed)
        print(f'RULER Time: {time() - t0:.2f}s')

    if args.longeval:
        clean_up()
        configure_model_cache(args, model, use_cache=True)
        t0 = time()
        if args.longeval_result_path:
            d = os.path.dirname(args.longeval_result_path)
            os.makedirs(d if d else '.', exist_ok=True)
            result_path = args.longeval_result_path
        else:
            result_path = ''
        eval_longeval_lines(
            model=model, tokenizer=get_tokenizer(model_id),
            test_dir=args.longeval_test_dir, model_name_or_path=model_id,
            num_lines_list=args.longeval_num_lines,
            eval_shortest_only=args.longeval_shortest_only,
            result_path=result_path, use_cache=True)
        longeval_time = time() - t0
        print(f'LongEval Lines Task Time: {longeval_time:.2f}s')
        if args.longeval_result_path:
            sentences = [f"{k}: {v}\n" for k, v in vars(args).items()]
            sentences += [f'LongEval Time: {longeval_time:.2f}s', "\n"]
            with open(args.longeval_result_path.replace('.json',
                                                        '_summary.txt'),
                      'w') as f:
                f.writelines(sentences)


def main(args):
    print(args)
    ctx = init_run(args)
    if maybe_generate_testcases(args):
        return

    n_comp_obj = len(args.comp_obj)
    assert n_comp_obj == len(args.comp_obj_min) == len(args.comp_obj_max)

    expr_map = build_expr_map(args, ctx)
    nd = build_nd(args, ctx, expr_map)
    _lazy = getattr(nd, 'lazy', False)
    print(f"[post_search] expr_front={args.expr_front} → "
          f"{'lazy comp_obj-pruned (no NDS, full archives)' if _lazy else 'dense'} "
          f"path  (n_total={nd.n_total:.3e})  "
          f"ranking={'surrogate' if args.sample_path else 'no-surrogate (see criterion below)'}")
    expr_keys, _esm, _efm = nd.expr_keys, nd.esm, nd.efm
    K = len(expr_keys)

    # ─────────────────────── candidate set under COMP_OBJ ───────────────────────
    # Separable per-axis-argmin fast path: no surrogate AND every comp_obj is
    # single-axis (wbits/kvbits/kvdim …). Those axes are independent and their
    # per-axis JSDs come from SEPARATE searches against different references —
    # summing them into one scalar is meaningless. Instead, per comp_obj/axis
    # pick the lowest-JSD subnet WITHIN that comp_obj's range and combine the
    # per-axis winners. No Cartesian product (no _LAZY_MAX_FEASIBLE blow-up).
    # -n N → the N combinations pairing each axis's k-th best (k=0..N-1).
    def _per_axis_comp(o, i):
        lc = nd.comp_nd_list
        if isinstance(lc, _LazyComp):                 # reuse prebuilt piece
            s = lc.comp_specs[i]
            return ((s['axis'], np.asarray(s['vals'], float))
                    if s['kind'] == '1d' else None)
        return per_axis_metric(o, expr_keys, _esm, ctx.config,
                               ctx.group_size, args.n_token)

    _pa = ([_per_axis_comp(o, i) for i, o in enumerate(args.comp_obj)]
           if args.comp_obj else [])
    separable = (not args.sample_path and bool(args.comp_obj)
                 and all(p is not None for p in _pa))

    if separable:
        ranked = {}                       # axis -> subnet idx, best→worst JSD
        for i, o in enumerate(args.comp_obj):
            ax, comp = _pa[i]
            lo, hi = args.comp_obj_min[i], args.comp_obj_max[i]
            m = _efm[expr_keys[ax]][:, 0]
            feas = np.where((comp >= lo) & (comp <= hi))[0]
            if len(feas) == 0:
                raise SystemExit(
                    f"[post_search] comp_obj '{o}' range [{lo:.4g},{hi:.4g}] "
                    f"excludes every subnet on axis '{expr_keys[ax]}' "
                    f"(achievable [{comp.min():.4g},{comp.max():.4g}]). "
                    f"Widen --comp_obj_min/--comp_obj_max.")
            ranked[ax] = feas[np.argsort(m[feas], kind='stable')]
        for a, k in enumerate(expr_keys):             # uncovered axis: global
            if a not in ranked:                       # lowest JSD, no window
                ranked[a] = np.argsort(_efm[k][:, 0], kind='stable')
        n_sel = min(int(args.n), min(len(v) for v in ranked.values()))
        valid_nd_idx = np.empty((n_sel, K), np.int64)
        for a in range(K):
            valid_nd_idx[:, a] = ranked[a][:n_sel]
        F = assemble_F(valid_nd_idx, expr_keys, _efm, nd.comp_nd_list,
                       nd.new_metric_nd)
        pf = F
        ps = LazyPs(lambda row: build_arch(ctx.default_arch, expr_keys,
                                           _esm, row), valid_nd_idx)
        I = list(range(n_sel))
        sel_mode = 'per-axis-argmin (no surrogate, separable comp_obj)'
        for i, o in enumerate(args.comp_obj):
            ax = _pa[i][0]
            print(f"[post_search] {o}: best within "
                  f"[{args.comp_obj_min[i]:.4g},{args.comp_obj_max[i]:.4g}] "
                  f"→ '{expr_keys[ax]}' subnet {valid_nd_idx[0, ax]} "
                  f"JSD={_efm[expr_keys[ax]][valid_nd_idx[0, ax], 0]:.5f}")
        print(f"[post_search] no --sample_path + separable comp_obj "
              f"{args.comp_obj} → per-axis lowest-JSD combination "
              f"(NO Σ; -n={args.n} → {n_sel} combo(s))")
    else:
        valid_nd_idx = select_valid_nd_idx(
            nd.nd_shape, nd.new_metric_nd, nd.comp_nd_list,
            comp_obj_min=args.comp_obj_min, comp_obj_max=args.comp_obj_max,
            random_sample=None, has_quantile=False, has_prefer=False)
        if len(valid_nd_idx) == 0:
            msg = ["[post_search] no architecture satisfies the COMP_OBJ range — "
                   "0 candidates after filtering."]
            for i, obj in enumerate(args.comp_obj):
                nd_i = np.asarray(nd.comp_nd_list[i])
                msg.append(f"  {obj}: achievable [{nd_i.min():.4g}, {nd_i.max():.4g}]"
                           f"  requested [{args.comp_obj_min[i]:.4g}, "
                           f"{args.comp_obj_max[i]:.4g}]")
            msg.append("Adjust --comp_obj_min/--comp_obj_max (or COMP_OBJ_VAL in "
                       "the script) into the achievable range. Note: model/config "
                       f"is {args.model_name} but expr archives may be from another "
                       "model — cross-model adapts layer count and shifts the "
                       "achievable memory range.")
            raise SystemExit("\n".join(msg))
        F = assemble_F(valid_nd_idx, expr_keys, _efm, nd.comp_nd_list,
                       nd.new_metric_nd)
        # per-axis search-metric columns of F (col 0 of each (metric,comp) pair)
        M_valid = F[:, [1 + 2 * i for i in range(K)]]

        # ─────────────── surrogate → combined metric ───────────────
        if args.sample_path:
            smp = load_sample_csv(
                args.sample_path,
                n_comp=len(comp_key_order(ctx.config, ctx.group_size)),
                n_axes=K)
            if smp['X'].shape[1] != K:
                raise SystemExit(
                    f"sample CSV has {smp['X'].shape[1]} per-axis columns but "
                    f"{K} expr axes were provided; they must match "
                    f"(expr_keys={expr_keys}).")
            print(f"[surrogate] training data: N={smp['n_valid']} axes={expr_keys}")
            model = _make_surrogate(args, smp['X'], smp['y'], M_valid)
            yp_tr = np.asarray(model.predict(np.asarray(smp['X'], float))).reshape(-1)
            ss_r = float(np.sum((smp['y'] - yp_tr) ** 2))
            ss_t = float(np.sum((smp['y'] - smp['y'].mean()) ** 2))
            r2 = 1.0 - ss_r / max(ss_t, 1e-30)
            rho, _ = stats.spearmanr(yp_tr, smp['y'])
            tau, _ = stats.kendalltau(yp_tr, smp['y'])
            print(f"[surrogate={args.surrogate}] train R2={r2:.4f} "
                  f"rho={rho:.4f} tau={tau:.4f}")
            pred = np.asarray(model.predict(M_valid)).reshape(-1).astype(float)
            F[:, 0] = pred
            # tripwire: any NaN in the surrogate prediction (fit failure /
            # bad M_valid) must abort rather than silently rank on NaN.
            n_nan = int(np.isnan(F[:, 0]).sum())
            if n_nan:
                raise SystemExit(
                    f"[post_search] {n_nan}/{len(F)} predicted metrics are NaN "
                    f"after surrogate override (surrogate={args.surrogate}). "
                    f"Refusing to rank on NaN — check the surrogate fit / "
                    f"M_valid columns / sample CSV.")
        elif not args.comp_obj:
            pred = F[:, 0].astype(float)
            print("[post_search] no --sample_path, no comp_obj → additive "
                  "args-scale combined metric")
        else:
            # coupled comp_obj (e.g. memory): one budget over all axes →
            # rank by scale-1 Σ per-axis JSD within the budget.
            pred = F[:, [1 + 2 * i for i in range(K)]].sum(1).astype(float)
            print(f"[post_search] no --sample_path, coupled comp_obj "
                  f"{args.comp_obj} → scale-1 Σ per-axis JSD within budget")

        order = np.argsort(pred)
        F = F[order]
        valid_nd_idx = valid_nd_idx[order]
        pf = F
        ps = LazyPs(lambda row: build_arch(ctx.default_arch, expr_keys,
                                           _esm, row), valid_nd_idx)
        I = list(range(min(args.n, len(valid_nd_idx))))
        sel_mode = f'top-{args.n}'

    # ─────────────────────── final architecture selection ───────────────────────
    # NOTE: --prefer (ASF) and --high_tradeoff (NonDominatedSorting) final
    # picks are intentionally DISABLED (shoved aside, do not use). Selection is
    # the separable per-axis-argmin combo(s) or plain top-n. Re-enable the
    # block below only if a weighted / non-dominated pick is explicitly needed.
    # if args.prefer:
    #     preferences = {}
    #     for p in args.prefer:
    #         k, v = p.split('#')
    #         preferences[k] = float(v)
    #     weights = np.fromiter(preferences.values(), dtype=float)
    #     I = ASF().do(pf[:, [0, *range(-n_comp_obj, 0)]],
    #                  weights).argsort()[:args.n].reshape(args.n)
    #     sel_mode = 'prefer'
    # elif args.high_tradeoff:
    #     I = NonDominatedSorting().do(pf[:, [0, *range(-n_comp_obj, 0)]],
    #                                  only_non_dominated_front=True)
    #     sel_mode = 'high_tradeoff'

    print(f'[selection] mode={sel_mode}  |I|={len(I)}  '
          f'|candidates|={len(valid_nd_idx)}  (n_total={nd.n_total})')

    # ─────────────────────── evaluate + benchmark final arch ───────────────────────
    model_id = f'{args.model_path}/{args.model_name}'
    if 'hqq' not in args.w_method:
        args.quant_model_paths = []
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
        loss_func=args.loss_func, last_tokens=args.last_tokens,
        use_key_token=args.use_key_token, trunc_len=args.trunc_len,
        sliding_window=args.sliding_window, alpha=args.alpha, beta=args.beta,
        key_token_path=args.key_token_path)

    for idx in tqdm(I):
        arch = ps[idx]
        complexity = get_net_info(arch, ctx.config, ctx.group_size,
                                  n_token=args.n_token)
        print(f'complexity: {list(complexity.keys())}')
        print(f'complexity: {list(complexity.values())}')
        ctx.accelerator.print(f'arch: {arch}')
        model = evaluator.sample(arch)

        if args.datasets:
            metric = evaluate_metric(args, arch, model, evaluator,
                                     ctx.accelerator)
            print(f'[{idx}] {args.metric}: {[p for p in metric.values()]}, '
                  f'pred_metric: {pf[idx, 0]}, per_axis_metric: '
                  f'{pf[idx, [1 + 2 * i for i in range(K)]].tolist()}')

        run_benchmarks(args, model, model_id)

        if ('awq' in args.w_method or 'gptq' in args.w_method
                or 'qeft' in args.w_method):
            del model, evaluator.model
            clean_up()

    print(args)


def build_parser():
    p = argparse.ArgumentParser(
        description='Stage 2: fit surrogate from results.csv, pick final '
                    'architecture under COMP_OBJ, evaluate + benchmark.')
    # model / config
    p.add_argument('--model_path', type=str, default='')
    p.add_argument('--model_name', type=str, default='')
    p.add_argument('--config', type=str, default='config/llama.json')
    p.add_argument('--dtype', type=str, default='auto',
                   choices=['float16', 'float', 'fp16', 'bfloat16', 'bfloat',
                            'bf16', 'auto'])
    p.add_argument('--gpu_id', type=str, default='0')
    p.add_argument('--seed', type=int, default=0)
    # quant methods / bits
    p.add_argument('--w_method', type=str, nargs='+', default=[],
                   choices=['fp16', 'awq', 'gptq', 'qeft', 'hqq'])
    p.add_argument('--kv_method', type=str, nargs='+', default=['kivi'],
                   choices=['fp16', 'hqq', 'kivi', 'think'],
                   help="space-separated list (e.g. 'kivi think'). Matches "
                        "search.py.")
    p.add_argument('--quant_model_paths', type=str, nargs='+', default=[])
    p.add_argument('--w_bits', type=int, nargs='+', default=[])
    p.add_argument('--k_bits', type=int, nargs='+', default=[2, 4])
    p.add_argument('--v_bits', type=int, nargs='+', default=[2, 4])
    p.add_argument('--w_group_size', type=int, default=128)
    p.add_argument('--k_group_size', type=int, default=128)
    p.add_argument('--v_group_size', type=int, default=128)
    p.add_argument('--residual_length', type=int, default=128)
    p.add_argument('--k_quant_scheme', type=str, choices=['channel', 'token'])
    p.add_argument('--v_quant_scheme', type=str, choices=['channel', 'token'])
    p.add_argument('--outlier_path', type=str, default='')
    # calibration data / metric
    p.add_argument('--datasets', type=str, nargs='+', default=[])
    p.add_argument('--metric', type=str, default='ppl')
    p.add_argument('--loss_func', type=str, default='cross_entropy')
    p.add_argument('--stride', type=int, default=None)
    p.add_argument('--last_tokens', type=int, default=None)
    p.add_argument('--prefill_prompt', action='store_true')
    p.add_argument('--n_sample', type=int, default=128)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--min_seqlen', type=int, default=0)
    p.add_argument('--data_batch_size', type=int, default=1)
    p.add_argument('--n_token', type=int, default=0)
    # expr archives + additive-metric scales (fallback when no sample CSV)
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
    # surrogate (predictor.factory.get_predictor)
    p.add_argument('--sample_path', type=str, default='',
                   help='results.csv from sample_surrogate.py (surrogate '
                        'training data). If omitted, additive metric is used.')
    p.add_argument('--surrogate', type=str, default='rbf', choices=SURROGATES,
                   help='rbf | gp | mlp | carts | as (predictor.factory) | '
                        'ard_gp (sklearn ARD-GP, analysis/v5)')
    p.add_argument('--rbf_kernel', type=str, default='tps',
                   choices=['cubic', 'tps', 'linear'],
                   help='RBF kernel when --surrogate rbf (v5: tps best)')
    p.add_argument('--ard_kernel', type=str, default='matern32',
                   choices=['rbf', 'matern52', 'matern32', 'rq'],
                   help='ARD kernel when --surrogate ard_gp (v5: matern32)')
    p.add_argument('--gp_n_restarts', type=int, default=10,
                   help='n_restarts_optimizer for --surrogate ard_gp')
    p.add_argument('--surrogate_device', type=str, default='auto',
                   help="device for the pure-PyTorch rbf / ard_gp surrogate: "
                        "'auto' (cuda if visible else cpu), 'cpu', 'cuda', "
                        "'cuda:N'. Other surrogates ignore this.")
    # comp_obj range + final selection
    p.add_argument('--comp_obj', type=str, nargs='+', default=[])
    p.add_argument('--comp_obj_min', type=float, nargs='+', default=[])
    p.add_argument('--comp_obj_max', type=float, nargs='+', default=[])
    p.add_argument('--prefer', type=str, nargs='+', default=[],
                   help='preferences (metric#0.0 memory#6.2e9)')
    p.add_argument('--high_tradeoff', type=str, nargs='+', default=[])
    p.add_argument('-n', type=int, default=1,
                   help='number of architectures desired')
    # long-ppl key-token options (consumed by the evaluator)
    p.add_argument('--use_key_token', action='store_true')
    p.add_argument('--trunc_len', type=int, default=512)
    p.add_argument('--sliding_window', type=int, default=128)
    p.add_argument('--alpha', type=int, default=2)
    p.add_argument('--beta', type=int, default=-2)
    p.add_argument('--key_token_path', type=str, default='')
    # output
    p.add_argument('--save', type=str, default='')
    p.add_argument('--results_csv_file', type=str, default='results.csv')
    # benchmarks
    p.add_argument('--latency', action='store_true')
    p.add_argument('--zeroshot', action='store_true')
    p.add_argument('--tasks', type=str, nargs='+',
                   default=['coqa', 'gsm8k', 'truthfulqa'])
    p.add_argument('--lm_eval_batch_size', type=int, default=None)
    p.add_argument('--longbench', action='store_true')
    p.add_argument('--longbench_e', action='store_true')
    p.add_argument('--longbench_result_path', type=str, default='')
    p.add_argument('--longbench_config', type=str, default='')
    p.add_argument('--longbench_task', type=str, nargs='+', default=[])
    p.add_argument('--pass_key_file', type=str, default='')
    p.add_argument('--minilongbench', action='store_true')
    p.add_argument('--minilongbench_result_path', type=str, default='')
    p.add_argument('--minilongbench_data_dir', type=str, default='')
    p.add_argument('--ruler', action='store_true')
    p.add_argument('--ruler_task', type=str, nargs='+', default=None,
                   choices=["niah_single_1", "niah_single_2", "niah_single_3",
                            "niah_multikey_1", "niah_multikey_2",
                            "niah_multikey_3", "niah_multivalue",
                            "niah_multiquery", "ruler_vt", "ruler_cwe",
                            "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot"])
    p.add_argument('--ruler_length', type=int, nargs='+', default=[4096])
    p.add_argument('--ruler_yaml_path', type=str, default='')
    p.add_argument('--ruler_sample', type=int, default=50)
    p.add_argument('--ruler_gen_toks', type=int, default=None)
    p.add_argument('--ruler_batch_size', type=int, default=1)
    p.add_argument('--ruler_result_path', type=str, default='')
    p.add_argument('--longeval', action='store_true')
    p.add_argument('--longeval_test_dir', type=str, default='')
    p.add_argument('--longeval_num_lines', type=int, nargs='+',
                   default=[200, 300, 400, 500, 600, 680, 700, 800, 900, 1000,
                            1100, 1200, 1350])
    p.add_argument('--longeval_shortest_only', action='store_true')
    p.add_argument('--longeval_result_path', type=str, default='')
    # LongEval testcase generation passthrough
    p.add_argument('--generate_testcases', action='store_true')
    p.add_argument('--generate_testcases_only', action='store_true')
    p.add_argument('--generate_testcases_num_lines', type=int, nargs='+',
                   default=[200, 300, 400, 500, 600, 680, 700, 800, 900, 1000,
                            1100, 1200, 1350])
    p.add_argument('--generate_testcases_num_samples', type=int, default=50)
    p.add_argument('--generate_testcases_line_idx_opt', type=str, default='LRT',
                   choices=['LRT', 'LRT-ABCindex', 'LRT-UUID', 'LRT-NL'])
    p.add_argument('--generate_testcases_output_dir', type=str,
                   default='evaluation')
    return p


if __name__ == '__main__':
    main(build_parser().parse_args())
