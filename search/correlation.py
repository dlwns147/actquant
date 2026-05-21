"""correlation.py — Loss/PPL ↔ LongBench/LongBench-E/RULER correlation harness.

Two stages, both invoked via this single file:

* `--mode sample`  (stage 1)
    Build the joint combo space from the per-axis search archives
    (`--w_expr / --kv_expr / --kvdim_expr / --eff_kv_expr`), random-sample
    `--n_archs` architectures (optionally inside `--comp_obj_min/--comp_obj_max`)
    and write each architecture as a single row to `<save>/archs.csv`. Pareto-
    near sampling is achieved upstream: pass `--expr_front` to keep only the
    per-axis Pareto frontier of each archive (matches scripts/sample_surrogate.sh).

* `--mode eval`  (stage 2, run once per `--idx`)
    Load `archs.csv`, pick the architecture at `--idx`, and evaluate all
    requested calibration metrics + long-context benchmarks on it.
    Idempotent: results are written to `<save>/result_<idx>.json`; re-running
    only fills in the metrics that are still missing (or in `--force`).

Calibration metrics (`--metrics`):
    c4_ppl              C4 PPL (test split, n_sample=128, seqlen=2048)
    wt2_jsd             wikitext2 JSD, n_sample=128 seqlen=2048,
                        prefill_prompt=False, stride=None
    wt2_jsd_s512        wikitext2 JSD, … stride=512
    wt2_jsd_pp512_s128  wikitext2 JSD, … prefill_prompt=True last_tokens=512
                        stride=128
    gov_jsd             gov_report JSD, n_sample=8 seqlen=8196 min_seqlen=8192
                        (trunc=256 sw=64 alpha=1 beta=-1; no key-token)
    gov_jsd_kt          gov_report JSD, … with `--key_token_path`

Long-context benchmarks:
    longbench / longbench_e / ruler  (same as post_search.py block)

Use `--metrics all` (default) to enable everything, or pass a subset
(comma-separated or space-separated).

The same `--*_expr / --model_* / --w_method / --kv_method / --k_bits / ...`
flags as post_search.py and sample_surrogate.py are accepted, so a typical
run-pair looks like:

    accelerate launch correlation.py --mode sample --save save/correlation/llama \\
        --model_path .../meta-llama --model_name Llama-3.1-8B-Instruct \\
        --w_expr <stats> --kv_expr <stats> --kvdim_expr <stats> --expr_front \\
        --n_archs 50 --seed 0

    accelerate launch correlation.py --mode eval --save save/correlation/llama \\
        --idx 0 --model_path … --model_name … (same quant args) \\
        --longbench_config utils/longbench_config \\
        --ruler_yaml_path utils/ruler_utils --ruler_task niah_single_1 \\
        --ruler_length 16384 --ruler_sample 50 \\
        --key_token_path key_token/Qwen2.5-72B-Instruct_gov_report_test_8sample_8192seqlen_8192min_256trunc_64sw_1alpha_-1beta
"""
import os
import json
import csv
import argparse
import traceback
import warnings
from copy import deepcopy
from time import time

import numpy as np
import torch
from tqdm import tqdm

from evaluator import LlamaEvaluator
from utils.func import (init_run, build_expr_map, build_nd, comp_key_order,
                        configure_model_cache, get_net_info, clean_up,
                        set_seed, init_accelerator, process_dtype, RunCtx)
from utils.select import (build_arch, select_valid_nd_idx, assemble_F,
                          LazyPs, draw_random, quantile_select, axis_of_map,
                          coverage_subset_nsga2_extras)
from utils.eval import eval_metric, eval_loss
from utils.data import get_tokenizer
from utils.longbench import pred_longbench, eval_longbench
from utils.ruler import eval_ruler

warnings.simplefilter("ignore")


# ════════════════════════════════════════════════════════════════════════════
# Calibration metric specs
# ════════════════════════════════════════════════════════════════════════════
# Each task references one evaluator GROUP (shared datasets / n_sample / seqlen
# / loss_func / use_key_token / key_token_path — i.e. things you cannot change
# without rebuilding the FP-model dense_logits). Multiple tasks inside a group
# only differ in stride / prefill_prompt / last_tokens, which can be varied
# per-call without rebuilding.

GROUPS = {
    'A': dict(  # wikitext2 / c4 base — full-sequence JSD, no key-token
        datasets=['wikitext2', 'c4'], n_sample=128, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=False, last_tokens=None,
        trunc_len=512, sliding_window=128, alpha=2, beta=-2,
    ),
    'A_pp': dict(  # wikitext2 only — answer-phase (last 512 tokens) JSD
        # get_logits' last_tokens at init MUST match eval_loss' last_tokens
        # per-call (dense_logits is pre-masked). Putting wt2_jsd_pp512_s128
        # in its own group with last_tokens=512 keeps it consistent.
        datasets=['wikitext2'], n_sample=128, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=False, last_tokens=512,
        trunc_len=512, sliding_window=128, alpha=2, beta=-2,
    ),
    'A_lt128': dict(  # wikitext2 — single-pass JSD on last 128 tokens
        # quant_kv_output=True is implicit because the single-pass path
        # (stride=0, prefill_prompt=False) sets use_cache=False, which in
        # turn flips configure_model_cache to quant_kv_output=True.
        datasets=['wikitext2'], n_sample=128, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=False, last_tokens=128,
        trunc_len=512, sliding_window=128, alpha=2, beta=-2,
    ),
    'B_pp': dict(  # gov_report — answer-phase JSD with prefill_prompt
        # last_tokens=512 makes dense_logits tiny (~1 GB) so the standard
        # eval_loss path fits without stream_dense gymnastics.
        datasets=['gov_report'], n_sample=8, seqlen=8196, min_seqlen=8192,
        loss_func='jsd', use_key_token=False, last_tokens=512,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'D': dict(  # gsm8k — short answer-only loss, JSD
        datasets=['gsm8k'], n_sample=8, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=False, last_tokens=None,
        trunc_len=512, sliding_window=128, alpha=2, beta=-2,
    ),
    'B': dict(  # gov_report long, no key-token
        datasets=['gov_report'], n_sample=8, seqlen=8196, min_seqlen=8192,
        loss_func='jsd', use_key_token=False, last_tokens=None,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'C': dict(  # gov_report long, with key-token
        datasets=['gov_report'], n_sample=8, seqlen=8196, min_seqlen=8192,
        loss_func='jsd', use_key_token=True, last_tokens=None,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
}

# (key, group, dataset, eval_kwargs)
#   eval_kwargs forwarded to eval_metric (stride, prefill_prompt, last_tokens,
#   metric, loss_func). dataset=None marks tasks handled by a custom path
#   (needle_nll generates its own prompts; see _run_needle_nll).
METRIC_TASKS = [
    ('c4_ppl',            'A', 'c4',
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('wt2_jsd',           'A', 'wikitext2',
        dict(metric='loss', loss_func='jsd',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('wt2_jsd_s512',      'A', 'wikitext2',
        dict(metric='loss', loss_func='jsd',
             stride=512, prefill_prompt=False, last_tokens=None)),
    ('wt2_jsd_pp512_s128', 'A_pp', 'wikitext2',
        dict(metric='loss', loss_func='jsd',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('wt2_jsd_lt128',     'A_lt128', 'wikitext2',
        # single-pass JSD on last 128 tokens. last_tokens is set at evaluator
        # init (Group A_lt128) and matches the eval_loss mask.
        dict(metric='loss', loss_func='jsd',
             stride=0, prefill_prompt=False, last_tokens=128)),
    ('needle_nll',        'A', None,
        dict(kind='needle_nll')),
    ('gsm8k_jsd',         'D', 'gsm8k',
        # Standard path. The padded-input KIVI bug is now fixed at the
        # source (quant/kivi_utils/new_pack.py:fake_quant handles 2D
        # HF padding masks via _kivi_mask_to_bnh11t1).
        dict(metric='loss', loss_func='jsd',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('gov_jsd',           'B', 'gov_report',
        # Standard path. _move_all_dense_logits_to_cpu has already replaced
        # evaluator.dense_logits['gov_report'] with a _LazyGpuList shim, so
        # the 16 GiB of dense logits no longer sit on GPU.
        dict(metric='loss', loss_func='jsd',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('gov_jsd_s512',      'B', 'gov_report',
        # Same Group B + cpu-shim, plus stride=512 chunked forward to
        # bound peak activation memory at 8K context.
        dict(metric='loss', loss_func='jsd',
             stride=512, prefill_prompt=False, last_tokens=None)),
    ('gov_jsd_pp512_s128', 'B_pp', 'gov_report',
        # answer-phase JSD (prefill_prompt + last_tokens=512 + stride=128).
        # dense_logits is tiny under last_tokens=512 → standard path is OK.
        dict(metric='loss', loss_func='jsd',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('gov_jsd_kt',        'C', 'gov_report',
        dict(metric='loss', loss_func='jsd',
             stride=0, prefill_prompt=False, last_tokens=None)),
]
METRIC_KEYS = [t[0] for t in METRIC_TASKS]
BENCH_KEYS = ['longbench', 'longbench_e', 'ruler']
ALL_KEYS = METRIC_KEYS + BENCH_KEYS


# ════════════════════════════════════════════════════════════════════════════
# Sample mode — build combo space, random-sample N archs, write archs.csv
# ════════════════════════════════════════════════════════════════════════════
def cmd_sample(args):
    ctx = _build_ctx(args)
    expr_map = build_expr_map(args, ctx)
    nd = build_nd(args, ctx, expr_map)
    expr_keys, _esm, _efm = nd.expr_keys, nd.esm, nd.efm
    print(f"[correlation/sample] expr_front={args.expr_front} → "
          f"{'lazy comp_obj-pruned' if getattr(nd, 'lazy', False) else 'dense'} path  "
          f"(n_total={nd.n_total:.3e}, expr_keys={expr_keys})")

    if args.n_archs is None or args.n_archs <= 0:
        raise SystemExit("[correlation/sample] --n_archs must be > 0")

    # ── Parse --quantile_sample (same syntax as sample_surrogate.py) ──
    # "metric_w#0.01,0.5,0.99 metric_kv#0.05,0.95" → {key: [q1,q2,...]}.
    # Quantile anchors hit ARCHS at specific (per-metric) percentiles —
    # extremes by default, so the correlation regression sees the full range
    # of each axis instead of a random clump.
    quantile_specs = {}
    for spec in (args.quantile_sample or []):
        k, v = spec.split('#')
        quantile_specs[k] = [float(q) for q in v.split(',')]
    if quantile_specs:
        _axis_map = axis_of_map(expr_keys)
        _expr_flag = {'w': '--w_expr', 'kv': '--kv_expr',
                      'kvdim': '--kvdim_expr', 'eff_kv': '--eff_kv_expr'}
        _missing = [(k, _axis_map.get(k)) for k in quantile_specs
                    if _axis_map.get(k) is not None
                    and _axis_map.get(k) not in expr_keys]
        if _missing:
            for k, ax in _missing:
                print(f"[quantile_sample] ERROR: metric '{k}' depends on "
                      f"axis '{ax}' but {_expr_flag.get(ax, ax)} was not "
                      f"provided; quantile would collapse to a constant.")
            raise SystemExit(1)

    # ── Candidate filter (comp_obj range + optional random pre-sample) ──
    # has_quantile/has_prefer drive select_valid_nd_idx's branching:
    # without quantile anchors, the random_sample knob pre-samples here;
    # with quantile anchors, we want the FULL feasible set so the quantile
    # picks have something to hit (extras drawn below).
    valid_nd_idx = select_valid_nd_idx(
        nd.nd_shape, nd.new_metric_nd, nd.comp_nd_list,
        comp_obj_min=args.comp_obj_min, comp_obj_max=args.comp_obj_max,
        random_sample=(args.n_archs if not quantile_specs else None),
        has_quantile=bool(quantile_specs), has_prefer=False)
    if len(valid_nd_idx) == 0:
        raise SystemExit(
            "[correlation/sample] 0 candidates after comp_obj filter — "
            "widen --comp_obj_min/--comp_obj_max.")

    # ── Pick final indices I (mirrors sample_surrogate.main()) ──
    if quantile_specs:
        I_quant, metric_vals = quantile_select(
            quantile_specs, valid_nd_idx, expr_keys, _esm,
            ctx.default_arch, ctx.config, ctx.group_size, args.n_token,
            axis_cache={}, efm=_efm)
        print(f"[quantile_sample] anchors selected: {len(I_quant)} "
              f"(out of {len(valid_nd_idx)} candidates)")

        # n_extras = n_archs - len(I_quant), clamped to >= 0
        n_extras = max(0, int(args.n_archs) - len(I_quant))
        I_extra = []
        if n_extras > 0:
            if args.sampling_method == 'random':
                I_extra = draw_random(n_extras, len(valid_nd_idx),
                                      exclude=I_quant)
                samp_desc = f'random (+{len(I_extra)} extras, '\
                            f'pool={len(valid_nd_idx) - len(I_quant)})'
            else:
                fit_mode = args.sampling_method.replace('coverage_nsga2_', '')
                I_extra = coverage_subset_nsga2_extras(
                    valid_nd_idx, _efm, expr_keys, anchor_idx=I_quant,
                    K=n_extras, fitness=fit_mode,
                    coord=args.coverage_coord,
                    per_axis_agg=args.coverage_per_axis_agg,
                    pareto_select=args.coverage_pareto_select,
                    seed=args.seed, verbose=False)
                samp_desc = (f"{args.sampling_method} "
                             f"(coord={args.coverage_coord}, "
                             f"per_axis_agg={args.coverage_per_axis_agg}, "
                             f"pareto_select={args.coverage_pareto_select}) "
                             f"+{len(I_extra)} extras")
        else:
            samp_desc = 'quantile-only (n_extras=0)'

        I_set = sorted(set(I_quant) | set(I_extra))
        assert len(I_set) == len(I_quant) + len(I_extra), \
            'quantile and extras must be disjoint'
        # Reorder valid_nd_idx so row i==I_set[i] (so archs.csv idx == I_set[i])
        valid_nd_idx = valid_nd_idx[I_set]
        n_final = len(I_set)
    else:
        # Pure random; select_valid_nd_idx already sub-sampled to n_archs.
        n_final = len(valid_nd_idx)
        samp_desc = f'random (n_archs={n_final})'

    print(f"[correlation/sample] final |I|={n_final}  method={samp_desc}")

    F = assemble_F(valid_nd_idx, expr_keys, _efm, nd.comp_nd_list,
                   nd.new_metric_nd)
    # F columns: [combined_metric | (metric_axis_i, comp_axis_i) * K | comp_obj_i]
    per_axis_metric_cols = [F[:, 1 + 2 * i] for i in range(len(expr_keys))]

    comp_keys = comp_key_order(ctx.config, ctx.group_size)
    metric_col_names = [f"metric_{k}" for k in expr_keys]
    header = (['idx', 'arch_json'] + comp_keys + metric_col_names
              + ['combined_metric'])

    save_dir = args.save or '.'
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'archs.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, row in enumerate(valid_nd_idx):
            arch = build_arch(ctx.default_arch, expr_keys, _esm, row)
            comp = get_net_info(arch, ctx.config, ctx.group_size,
                                n_token=args.n_token)
            comp_vals = [comp[k] for k in comp_keys]
            metric_vals = [float(per_axis_metric_cols[ax][i])
                           for ax in range(len(expr_keys))]
            w.writerow([i, json.dumps(arch, separators=(',', ':'))]
                       + comp_vals + metric_vals + [float(F[i, 0])])
    print(f"[correlation/sample] wrote {len(valid_nd_idx)} archs → {csv_path}")

    # also save the meta (so eval mode can sanity-check the model/expr context)
    meta = {
        'model_name': args.model_name, 'model_path': args.model_path,
        'config': args.config, 'expr_keys': list(expr_keys),
        'w_expr': args.w_expr, 'kv_expr': args.kv_expr,
        'kvdim_expr': args.kvdim_expr, 'eff_kv_expr': args.eff_kv_expr,
        'expr_front': args.expr_front, 'n_token': args.n_token,
        'comp_obj': args.comp_obj, 'comp_obj_min': args.comp_obj_min,
        'comp_obj_max': args.comp_obj_max, 'n_archs': args.n_archs,
        'seed': args.seed,
    }
    with open(os.path.join(save_dir, 'sample_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════
# Eval mode helpers
# ════════════════════════════════════════════════════════════════════════════
def _load_arch_row(csv_path, idx):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if int(r['idx']) == idx:
            return json.loads(r['arch_json']), r
    raise SystemExit(f"idx {idx} not in {csv_path} ({len(rows)} rows)")


def _resolve_metric_set(arg):
    if not arg or arg == ['all']:
        return ALL_KEYS
    flat = []
    for x in arg:
        flat.extend(p for p in x.replace(',', ' ').split() if p)
    bad = [k for k in flat if k not in ALL_KEYS]
    if bad:
        raise SystemExit(f"--metrics: unknown keys {bad}. Valid: {ALL_KEYS}")
    return flat


class _LazyGpuList:
    """List-of-lists of tensors held on CPU; per-tensor `__getitem__` moves
    to GPU lazily. Drop-in for `evaluator.dense_logits[dataset]` so the
    standard utils.eval.eval_loss path works unchanged:

      `dense_logits_list[batch_idx][seq_idx].contiguous()`
        → _LazyGpuList[batch_idx]      → _LazyGpuBatch
        →                  [seq_idx]   → CPU→GPU upload of that one tensor

    Saves ~16 GB GPU for the gov_report group (8 × 8195 × 128k vocab fp16)
    and ~4 GB for the wikitext2+c4 group — total slack matters because the
    quant model + 3 HQQ template copies + KV cache + transient JSD compute
    eat the rest of 48 GB.
    """

    class _Batch:
        def __init__(self, seqs, device):
            self._seqs = seqs
            self._device = device

        def __getitem__(self, j):
            t = self._seqs[j]
            return t.to(self._device, non_blocking=True) if t.device.type == 'cpu' else t

        def __len__(self):
            return len(self._seqs)

    def __init__(self, batches, device):
        self._batches = batches
        self._device = device

    def __getitem__(self, i):
        return _LazyGpuList._Batch(self._batches[i], self._device)

    def __len__(self):
        return len(self._batches)


def _move_all_dense_logits_to_cpu(evaluator):
    """For every dataset present in evaluator.dense_logits, replace the
    GPU-resident list-of-lists with a _LazyGpuList shim backed by CPU
    tensors. Run once right after _build_evaluator + before any eval_loss
    call. eval_loss's interface is unchanged (it reads `[bi][si].contiguous()`)."""
    target_device = evaluator.model.device if evaluator.model is not None else 'cuda'
    n_moved = 0
    for dataset, batches in list(evaluator.dense_logits.items()):
        if batches is None:
            continue
        if isinstance(batches, _LazyGpuList):
            continue   # already wrapped
        cpu_batches = [[t.detach().to('cpu', copy=False) for t in batch]
                       for batch in batches]
        evaluator.dense_logits[dataset] = _LazyGpuList(cpu_batches, target_device)
        n_moved += sum(len(b) for b in cpu_batches)
    clean_up()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"[dense_logits→cpu] moved {n_moved} per-seq tensors to CPU, "
              f"GPU free={free/1e9:.2f}GB / {total/1e9:.2f}GB")


def _build_evaluator(args, ctx, *, datasets, n_sample, seqlen, min_seqlen,
                     loss_func, use_key_token, key_token_path,
                     trunc_len, sliding_window, alpha, beta,
                     last_tokens=None):
    """One LlamaEvaluator with the requested data-side config. `last_tokens`
    here is set on the evaluator at init so dense_logits gets pre-masked to
    the last N positions — must match the eval_loss last_tokens used at
    metric-call time (eval_loss compares len-N logits vs len-N dense).
    Dense_logits is moved to CPU right after build (see _move_all_dense_logits_to_cpu)."""
    model_id = f'{args.model_path}/{args.model_name}'
    quant_model_paths = args.quant_model_paths if 'hqq' in args.w_method else []
    # Scalar fallback for replace_kv_cache (per-arch arch['p'] overrides this
    # at sample() time); pick the first option from the CLI list.
    kpd = args.k_pruning_dim[0] if args.k_pruning_dim else 0
    vpd = args.v_pruning_dim[0] if args.v_pruning_dim else 0
    evaluator = LlamaEvaluator(
        ctx.config, accelerator=ctx.accelerator, model_id=model_id,
        method={'w': args.w_method, 'kv': args.kv_method},
        quant_model_paths=quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=seqlen, min_seqlen=min_seqlen, n_sample=n_sample,
        datasets=datasets, device_map=ctx.device_map, dtype=ctx.dtype,
        bits={'w': args.w_bits, 'k': args.k_bits, 'v': args.v_bits},
        group_size=ctx.group_size, residual_length=args.residual_length,
        k_quant_scheme=args.k_quant_scheme, v_quant_scheme=args.v_quant_scheme,
        k_pruning_dim=kpd, v_pruning_dim=vpd,
        loss_func=loss_func, last_tokens=last_tokens,
        use_key_token=use_key_token, trunc_len=trunc_len,
        sliding_window=sliding_window, alpha=alpha, beta=beta,
        key_token_path=key_token_path)
    _move_all_dense_logits_to_cpu(evaluator)
    return evaluator


def _build_needle_loader(args, model_id, device):
    """Materialise NIAH prompts as a list of (input_ids, attention_mask,
    labels) batches matching the utils.data loader contract:
      - input_ids = tokenized(prompt + ' ' + answer), batch=1
      - attention_mask = all-ones (no padding; each batch is variable length)
      - labels = input_ids with -100 on the prompt span so eval_loss's
        get_loss_mask (labels != -100) selects only the answer tokens

    Tensors are placed on `device` upfront because utils.eval.eval_loss
    does `model(inputs, attention_mask=attention_mask)` with NO further
    `.to(device)` (other loaders pass through `accelerator.prepare()` which
    auto-moves; this custom iterable doesn't, so we move here).
    """
    from utils.ruler_utils import niah_utils as _niah
    import random as _random

    if not hasattr(_niah, args.needle_task):
        raise SystemExit(
            f"[needle_nll] unknown --needle_task '{args.needle_task}'. "
            f"Valid: niah_single_1/2/3, niah_multikey_1/2/3.")
    tokenizer = get_tokenizer(model_id)
    # niah's generate_input_output only seeds the needles SHUFFLE; magic-
    # number / word / depth picks use the *global* random module which has
    # been advanced by LlamaEvaluator init by now. Re-seed so every arch
    # in a sweep sees IDENTICAL needle prompts.
    _random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    t0 = time()
    data = getattr(_niah, args.needle_task)(
        model=model_id,
        max_seq_lengths=[int(args.needle_seqlen)],
        num_samples=int(args.needle_n_sample))['test']
    print(f"[needle_nll] generated {len(data)} {args.needle_task} prompts "
          f"in {time() - t0:.1f}s (seed={args.seed})")

    batches = []
    for ex in data:
        prompt = ex['input'] + ' ' + ex['gen_prefix']
        answer = ex['outputs'][0] if isinstance(ex['outputs'], list) else ex['outputs']
        enc_prompt = tokenizer(prompt, return_tensors='pt',
                               add_special_tokens=True).input_ids
        enc_full = tokenizer(prompt + ' ' + answer, return_tensors='pt',
                             add_special_tokens=True).input_ids
        if enc_full.shape[1] <= enc_prompt.shape[1]:
            continue
        labels = enc_full.clone()
        labels[:, :enc_prompt.shape[1]] = -100
        attention_mask = torch.ones_like(enc_full)
        batches.append((enc_full.to(device),
                        attention_mask.to(device),
                        labels.to(device)))

    class _NeedleLoader:
        def __iter__(self):
            return iter(batches)

        def __len__(self):
            return len(batches)

    return _NeedleLoader()


def _run_needle_nll(args, ctx, evaluator, model_id):
    """Cheap NIAH cross-entropy NLL via utils.eval.eval_loss. The loader
    yields (input_ids, attention_mask, labels) with labels=-100 on prompt
    tokens, so eval_loss's get_loss_mask naturally restricts CE to the
    answer span. No bespoke forward / loss loop — same metric math as all
    other cross_entropy calibration paths."""
    loader = _build_needle_loader(args, model_id, evaluator.model.device)
    if len(loader) == 0:
        raise RuntimeError("[needle_nll] no usable NIAH prompts after tokenisation")
    configure_model_cache(args, evaluator.model, use_cache=False)
    return eval_loss(model=evaluator.model, accelerator=ctx.accelerator,
                     loader=loader, seqlen=int(args.needle_seqlen),
                     loss_func='cross_entropy', dense_logits_list=None,
                     key_token_list=None, stride=0, last_tokens=None,
                     prefill_prompt=False)


def _run_calibration_task(args, ctx, evaluator, dataset, eval_kwargs):
    """Run one calibration metric on an already-prepared evaluator+model.

    Bypasses LlamaEvaluator.eval() (which loops over all loaders in the
    evaluator) so we can pick one dataset + run with task-specific
    stride / prefill_prompt / last_tokens.
    """
    model = evaluator.model
    use_cache = (eval_kwargs.get('stride') or 0) > 0 or eval_kwargs.get('prefill_prompt')
    configure_model_cache(args, model, use_cache=use_cache)

    if eval_kwargs['metric'] == 'ppl':
        loader = evaluator.test_loaders[dataset]
    else:
        loader = evaluator.train_loaders[dataset]
    dense_logits = (evaluator.dense_logits.get(dataset)
                    if eval_kwargs.get('loss_func') in ('jsd', 'kld', 'topk')
                    else None)
    key_token_list = (evaluator.key_token_list.get(dataset)
                      if evaluator.use_key_token else None)
    return eval_metric(
        model=model, accelerator=ctx.accelerator,
        metric=eval_kwargs['metric'], loader=loader, seqlen=evaluator.seqlen,
        loss_func=eval_kwargs.get('loss_func', 'cross_entropy'),
        dense_logits_list=dense_logits, key_token_list=key_token_list,
        stride=eval_kwargs.get('stride') or 0,
        last_tokens=eval_kwargs.get('last_tokens'),
        prefill_prompt=bool(eval_kwargs.get('prefill_prompt')),
        tokenizer=evaluator.tokenizer)


def _run_benchmark_block(args, model, model_id, which):
    """which ∈ {'longbench', 'longbench_e', 'ruler'}; mirrors post_search.run_benchmarks
    but isolated per block so eval mode can run only what was requested."""
    if which in ('longbench', 'longbench_e'):
        e = (which == 'longbench_e')
        if not args.longbench_config:
            raise SystemExit(f"--{which}: --longbench_config is required")
        result_path = (args.longbench_e_result_path if e
                       else args.longbench_result_path)
        if not result_path:
            raise SystemExit(f"--{which}: pass --longbench{'_e' if e else ''}_result_path")
        clean_up()
        configure_model_cache(args, model, use_cache=True)
        t0 = time()
        pred_longbench(model, tokenizer=get_tokenizer(model_id),
                       save_path=result_path,
                       longbench_config=args.longbench_config,
                       e=e, model_name=args.model_name)
        eval_longbench(result_path, e)
        score_path = os.path.join(result_path, 'pred_e' if e else 'pred',
                                  'result.json')
        with open(score_path) as f:
            scores = json.load(f)
        scores['_time'] = time() - t0
        return scores

    if which == 'ruler':
        if not args.ruler_yaml_path or not args.ruler_task:
            raise SystemExit("--ruler: --ruler_yaml_path and --ruler_task required")
        clean_up()
        configure_model_cache(args, model, use_cache=True)
        t0 = time()
        result_path = args.ruler_result_path or ''
        if result_path:
            os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
        eval_ruler(model, tokenizer=get_tokenizer(model_id), model_id=model_id,
                   tasks=args.ruler_task, yaml_path=args.ruler_yaml_path,
                   batch_size=args.ruler_batch_size, length=args.ruler_length,
                   nsample=args.ruler_sample, gen_toks=args.ruler_gen_toks,
                   result_path=result_path, seed=args.seed)
        if result_path and os.path.exists(result_path):
            with open(result_path) as f:
                scores = json.load(f)
        else:
            scores = {}
        scores['_time'] = time() - t0
        return scores
    raise ValueError(which)


# ════════════════════════════════════════════════════════════════════════════
# Eval mode — evaluate one arch (by --idx) on the requested metrics
# ════════════════════════════════════════════════════════════════════════════
def cmd_eval(args):
    ctx = _build_ctx(args)
    archs_csv = (args.archs_csv
                 or os.path.join(args.save or '.', 'archs.csv'))
    if not os.path.exists(archs_csv):
        raise SystemExit(f"archs.csv not found at {archs_csv} — run --mode sample first")
    arch, row = _load_arch_row(archs_csv, args.idx)
    print(f"[correlation/eval] idx={args.idx}  arch keys={list(arch['q'].keys())}")
    print(f"[correlation/eval] complexity: " + ", ".join(
        f"{k}={row.get(k, '?')}" for k in
        ('wbits', 'kvbits', 'kvdim', 'eff_kvbits', 'memory')))

    requested = _resolve_metric_set(args.metrics)
    print(f"[correlation/eval] requested metrics: {requested}")

    out_dir = args.save or os.path.dirname(archs_csv) or '.'
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, f'result_{args.idx}.json')

    results = {}
    if os.path.exists(result_path) and not args.force:
        with open(result_path) as f:
            results = json.load(f)

    def _done(k):
        # An entry stored as {'error': ...} is a previous failure — retry it.
        # An entry stored as a scalar float / int / a benchmark scores dict
        # (without 'error') is genuinely done.
        v = results.get(k)
        if v is None:
            return False
        if isinstance(v, dict) and 'error' in v:
            return False
        return True

    pending_calib = [t for t in METRIC_TASKS
                     if t[0] in requested and (args.force or not _done(t[0]))]
    pending_bench = [k for k in BENCH_KEYS
                     if k in requested and (args.force or not _done(k))]
    skipped = [k for k in requested
               if k in {t[0] for t in METRIC_TASKS} | set(BENCH_KEYS)
               and _done(k) and not args.force]
    if skipped:
        print(f"[correlation/eval] skipping (done): {skipped}")
    retried = [k for k in requested
               if isinstance(results.get(k), dict) and 'error' in results[k]]
    if retried:
        print(f"[correlation/eval] retrying previous failures: {retried}")
    if not pending_calib and not pending_bench:
        print("[correlation/eval] nothing to do — all requested metrics already present "
              "(pass --force to recompute).")
        return

    results.setdefault('idx', args.idx)
    results.setdefault('arch', arch)
    results.setdefault('complexity', {k: row.get(k) for k in row
                                      if k not in ('idx', 'arch_json')})

    model_id = f'{args.model_path}/{args.model_name}'

    # ── Calibration metrics, grouped to share evaluator builds ──
    groups_needed = sorted({t[1] for t in pending_calib})
    for g in groups_needed:
        spec = dict(GROUPS[g])
        if g == 'C':
            if not args.key_token_path:
                raise SystemExit(
                    "gov_jsd_kt requested but --key_token_path is empty. "
                    "Either pass --key_token_path or drop gov_jsd_kt from --metrics.")
            spec['key_token_path'] = args.key_token_path
        else:
            spec['key_token_path'] = ''
        print(f"\n[correlation/eval] === group {g}: datasets={spec['datasets']} "
              f"n_sample={spec['n_sample']} seqlen={spec['seqlen']} "
              f"use_key_token={spec['use_key_token']} ===")
        evaluator = _build_evaluator(args, ctx, **spec)
        model = evaluator.sample(arch)

        for key, group, dataset, eval_kwargs in pending_calib:
            if group != g:
                continue
            print(f"[correlation/eval] → metric '{key}' on '{dataset}' kwargs={eval_kwargs}")
            t0 = time()
            try:
                kind = eval_kwargs.get('kind')
                if kind == 'needle_nll':
                    value = _run_needle_nll(args, ctx, evaluator, model_id)
                else:
                    value = _run_calibration_task(args, ctx, evaluator,
                                                  dataset, eval_kwargs)
                if isinstance(value, torch.Tensor):
                    value = value.item()
                results[key] = float(value)
                print(f"[correlation/eval]   {key} = {results[key]:.6f}  "
                      f"({time() - t0:.1f}s)")
            except Exception as e:                              # noqa: BLE001
                tb = traceback.format_exc()
                results[key] = {'error': repr(e), 'traceback': tb}
                print(f"[correlation/eval]   {key} FAILED: {e!r}\n{tb}")
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)

        # Free this evaluator before building the next group's
        del evaluator
        clean_up()

    # ── Long-context benchmarks ──
    if pending_bench:
        # A minimal evaluator just to load the quant model + KV cache. No
        # datasets / dense_logits — so loss_func='cross_entropy' (skips FP forward).
        print(f"\n[correlation/eval] === benchmarks: {pending_bench} ===")
        bench_evaluator = _build_evaluator(
            args, ctx, datasets=[], n_sample=128, seqlen=2048, min_seqlen=0,
            loss_func='cross_entropy', use_key_token=False, key_token_path='',
            trunc_len=512, sliding_window=128, alpha=2, beta=-2)
        model = bench_evaluator.sample(arch)
        for which in pending_bench:
            print(f"[correlation/eval] → benchmark '{which}'")
            try:
                results[which] = _run_benchmark_block(args, model, model_id, which)
                print(f"[correlation/eval]   {which} = {results[which]}")
            except Exception as e:                              # noqa: BLE001
                results[which] = {'error': repr(e)}
                print(f"[correlation/eval]   {which} FAILED: {e!r}")
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
        del bench_evaluator
        clean_up()

    print(f"\n[correlation/eval] wrote {result_path}")


# ════════════════════════════════════════════════════════════════════════════
# Aggregate mode — scan result_*.json files into a wide CSV for plotting
# ════════════════════════════════════════════════════════════════════════════
def cmd_aggregate(args):
    archs_csv = (args.archs_csv
                 or os.path.join(args.save or '.', 'archs.csv'))
    out_dir = args.save or os.path.dirname(archs_csv) or '.'
    if not os.path.exists(archs_csv):
        raise SystemExit(f"archs.csv not found at {archs_csv}")
    with open(archs_csv) as f:
        archs_rows = list(csv.DictReader(f))
    arch_header = list(archs_rows[0].keys()) if archs_rows else []

    # Discover the column set across all result files
    rows = []
    for r in archs_rows:
        idx = int(r['idx'])
        result_path = os.path.join(out_dir, f'result_{idx}.json')
        merged = {k: r[k] for k in arch_header if k != 'arch_json'}
        if os.path.exists(result_path):
            with open(result_path) as f:
                res = json.load(f)
            for mk in METRIC_KEYS:
                if mk in res:
                    v = res[mk]
                    merged[mk] = v if isinstance(v, (int, float)) else str(v)
            for bk in BENCH_KEYS:
                if bk in res and isinstance(res[bk], dict):
                    for sk, sv in res[bk].items():
                        if sk.startswith('_'):
                            continue
                        merged[f'{bk}__{sk}'] = sv
                        if isinstance(sv, dict):
                            # LongBench-E returns {'0-4k':…, '4-8k':…, '8k+':…}
                            for sub_k, sub_v in sv.items():
                                merged[f'{bk}__{sk}__{sub_k}'] = sub_v
        rows.append(merged)

    all_cols = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); all_cols.append(k)
    out_csv = os.path.join(out_dir, 'correlation.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[correlation/aggregate] wrote {out_csv} "
          f"({len(rows)} rows, {len(all_cols)} cols)")


# ════════════════════════════════════════════════════════════════════════════
# Argparse — single parser, --mode selects the sub-command
# ════════════════════════════════════════════════════════════════════════════
def build_parser():
    p = argparse.ArgumentParser(
        description='Loss/PPL ↔ LongBench/LongBench-E/RULER correlation harness.')
    p.add_argument('--mode', choices=['sample', 'eval', 'aggregate'],
                   required=True)
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
                   help="space-separated list (e.g. 'kivi think' enables "
                        "ThinK pruning on top of KIVI). Matches search.py.")
    p.add_argument('--quant_model_paths', type=str, nargs='+', default=[])
    p.add_argument('--w_bits', type=int, nargs='+', default=[])
    p.add_argument('--k_bits', type=int, nargs='+', default=[2, 4])
    p.add_argument('--v_bits', type=int, nargs='+', default=[2, 4])
    p.add_argument('--w_group_size', type=int, default=128)
    # Same parser as search.py — repeated `--k_group_size 32 64 128 …`
    # builds a list-of-lists where each call is the gs option set for
    # one bit-width slot (search.py / scripts/search.sh convention).
    # E.g. K_BITS="2 3 4" + K_GROUP_SIZE=("32 64 128" "32 64 128" "128")
    # → 2-bit: {32,64,128}, 3-bit: {32,64,128}, 4-bit: {128}.
    p.add_argument('--k_group_size', type=int, nargs='+', action='append',
                   default=[])
    p.add_argument('--v_group_size', type=int, nargs='+', action='append',
                   default=[])
    p.add_argument('--residual_length', type=int, default=128)
    p.add_argument('--k_quant_scheme', type=str, choices=['channel', 'token'])
    p.add_argument('--v_quant_scheme', type=str, choices=['channel', 'token'])
    # ThinK channel-pruning options (pruned-channel count; matches search.py
    # convention — anything *prune* = removed count). In correlation.py
    # archs come from --kvdim_expr archives so arch['p'] overrides per-arch;
    # this list is the scalar fallback used by LlamaEvaluator's
    # replace_kv_cache call for layers without an override.
    p.add_argument('--k_pruning_dim', type=int, nargs='+', default=None,
                   help="K pruning dim options (# of head_dim channels to "
                        "prune; 0 = no pruning).")
    p.add_argument('--v_pruning_dim', type=int, nargs='+', default=None,
                   help="V pruning dim options. See --k_pruning_dim.")
    p.add_argument('--outlier_path', type=str, default='')
    p.add_argument('--n_token', type=int, default=0)
    # expr archives + combined-metric scales (sample mode)
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
    # optional pre-filter
    p.add_argument('--comp_obj', type=str, nargs='+', default=[])
    p.add_argument('--comp_obj_min', type=float, nargs='+', default=[])
    p.add_argument('--comp_obj_max', type=float, nargs='+', default=[])
    # sample mode
    p.add_argument('--n_archs', '--n_samples', dest='n_archs', type=int, default=50,
                   help='(sample mode) number of architectures (rows in '
                        'archs.csv) to sample. NOT to be confused with the '
                        'per-loader n_sample (data examples per metric, set '
                        'inside GROUPS).')
    # ── quantile + coverage-NSGA2 sampling (mirrors sample_surrogate.py) ──
    p.add_argument('--quantile_sample', type=str, nargs='+', default=[],
                   help='(sample) per-metric quantile anchors. Syntax: '
                        '"metric_w#0.01,0.5,0.99 metric_kv#0.05,0.95". '
                        'Picks one arch per quantile point; extras (up to '
                        '--n_archs total) drawn via --sampling_method. '
                        'Empty → pure random.')
    p.add_argument('--sampling_method', type=str,
                   default='coverage_nsga2_combined',
                   choices=['random', 'coverage_nsga2_joint',
                            'coverage_nsga2_marginal',
                            'coverage_nsga2_combined'],
                   help='(sample) how to draw extras on top of the quantile '
                        'anchors. Default coverage_nsga2_combined = 2-obj '
                        'GA (cov_rad, std_max) → balances extent (reach '
                        'every region) and per-axis uniformity. '
                        'Ignored when --quantile_sample empty.')
    p.add_argument('--coverage_coord', type=str, default='rank',
                   choices=['z', 'rank'],
                   help='(sample) coverage GA coordinate space.')
    p.add_argument('--coverage_per_axis_agg', type=str, default='max',
                   choices=['max', 'sum', 'pareto'],
                   help='(sample) per-axis std aggregator for '
                        'coverage_nsga2_marginal.')
    p.add_argument('--coverage_pareto_select', type=str, default='auto',
                   choices=['auto', 'strategy3', 'knee'],
                   help='(sample) how to collapse a multi-obj Pareto front '
                        'to K picks.')
    # eval mode
    p.add_argument('--archs_csv', type=str, default='',
                   help='(eval / aggregate) path to archs.csv (default: <save>/archs.csv)')
    p.add_argument('--idx', type=int, default=-1,
                   help='(eval) row index in archs.csv to evaluate')
    p.add_argument('--metrics', type=str, nargs='+', default=['all'],
                   help=f'(eval) subset of {ALL_KEYS}, or "all" (default).')
    p.add_argument('--force', action='store_true',
                   help='(eval) recompute even if already in result_<idx>.json')
    # gov_jsd_kt key-token archive (consumed in eval mode only)
    p.add_argument('--key_token_path', type=str, default='',
                   help='dir containing per-dataset key-token archives '
                        '(required for gov_jsd_kt; see actquant/search/key_token/…)')
    # needle_nll prompt generation knobs — kept small (8 prompts × 2048 ctx
    # ≈ 16k tokens; ~3s on Llama-3.1-8B) so it doesn't dominate the suite.
    p.add_argument('--needle_n_sample', type=int, default=8,
                   help='(needle_nll) number of NIAH prompts')
    p.add_argument('--needle_seqlen', type=int, default=2048,
                   help='(needle_nll) target context length per prompt')
    p.add_argument('--needle_task', type=str, default='niah_multikey_2',
                   choices=['niah_single_1', 'niah_single_2', 'niah_single_3',
                            'niah_multikey_1', 'niah_multikey_2',
                            'niah_multikey_3'],
                   help='(needle_nll) NIAH variant. Default niah_multikey_2: '
                        'distractor-needle haystack (much harder than '
                        'niah_single_1\'s repeat haystack). multivalue/'
                        'multiquery are excluded (multi-answer format).')
    # benchmarks (eval mode)
    p.add_argument('--longbench_config', type=str, default='utils/longbench_config')
    p.add_argument('--longbench_result_path', type=str, default='')
    p.add_argument('--longbench_e_result_path', type=str, default='')
    p.add_argument('--ruler_task', type=str, nargs='+', default=None,
                   choices=["niah_single_1", "niah_single_2", "niah_single_3",
                            "niah_multikey_1", "niah_multikey_2",
                            "niah_multikey_3", "niah_multivalue",
                            "niah_multiquery", "ruler_vt", "ruler_cwe",
                            "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot"])
    p.add_argument('--ruler_length', type=int, nargs='+', default=[16384])
    p.add_argument('--ruler_yaml_path', type=str, default='utils/ruler_utils')
    p.add_argument('--ruler_sample', type=int, default=50)
    p.add_argument('--ruler_gen_toks', type=int, default=None)
    p.add_argument('--ruler_batch_size', type=int, default=1)
    p.add_argument('--ruler_result_path', type=str, default='')
    # output
    p.add_argument('--save', type=str, default='',
                   help='output dir (archs.csv / sample_meta.json / result_*.json)')
    return p


def _build_ctx(args):
    """Replicates utils.func.init_run but bypasses its `min(args.k_group_size)`
    fallback, which fails on the list-of-lists shape produced by search.py-
    style `--k_group_size 32 64 128 --k_group_size 128` (nargs+action='append').
    We pick the first scalar in the flattened list for default_arch (the
    per-layer (bits, gs) in arch['q']['k'/'v'][i] overrides this at
    sample() time, so the choice only matters as a no-op fallback).
    """
    set_seed(args.seed)
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    dtype = process_dtype(args.dtype)

    group_size = {'w': args.w_group_size,
                  'k': args.k_group_size,
                  'v': args.v_group_size}
    n_block = config['n_block']
    w_linears = config['linear']
    default_w_bits = max(args.w_bits) if args.w_bits else 16
    default_k_bits = max(args.k_bits) if args.k_bits else 4
    default_v_bits = max(args.v_bits) if args.v_bits else 4
    # Flat-first scalar for default_arch (search.py-style list-of-lists
    # arrives here; fall back to 128 if --k/v_group_size not supplied).
    def _first_int(xs):
        if not xs:
            return 128
        v = xs[0]
        return v[0] if isinstance(v, (list, tuple)) and v else (v if isinstance(v, int) else 128)
    k_gs = _first_int(args.k_group_size)
    v_gs = _first_int(args.v_group_size)
    default_arch = {
        'q': {
            'w': {linear: [default_w_bits] * n_block for linear in w_linears},
            'k': [[default_k_bits, k_gs]] * n_block,
            'v': [[default_v_bits, v_gs]] * n_block,
        },
        'p': {'k': [0] * n_block, 'v': [0] * n_block},
    }
    return RunCtx(config, accelerator, device_map, dtype, group_size,
                  default_arch, n_block)


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
