"""correlation.py — Loss/PPL ↔ LongBench/LongBench-E/RULER correlation harness.

Two stages, both invoked via this single file:

* `--mode sample`  (stage 1)
    Build the joint combo space from the per-axis search archives
    (`--w_expr / --kv_expr / --kvdim_expr / --eff_kv_expr`), random-sample
    `--n_archs` architectures (optionally inside `--comp_obj_min/--comp_obj_max`)
    and write each architecture as a single row to `<save>/archs.csv`. Pareto-
    near sampling is achieved upstream: pass `--expr_front` to keep only the
    per-axis Pareto frontier of each archive (matches scripts/sample_surrogate.sh),
    or `--front_eps_rel R` to keep the ε-BAND around that frontier instead —
    metric ≤ front(comp)·(1+R), the same near-front shell second_search.py
    builds its block pools from (R=0 → strict front, unchanged).

    `--grid_sample` switches to a PAIRED FACTORIAL design instead: draw
    `--grid_n` blocks per axis from each axis's Pareto front (random, or
    comp-stratified with `--grid_stratify`) and emit the full cartesian
    product (e.g. 12x12 = 144 rows for w x eff_kv). Row order is axis-major
    and per-axis block ids are written as `grid_<axis>` columns, so a two-way
    ANOVA / interaction analysis can reconstruct the design directly from
    archs.csv. Requires `--expr_front`; ignores --n_archs / --quantile_sample.

* `--mode eval`  (stage 2, run once per `--idx`)
    Load `archs.csv`, pick the architecture at `--idx`, and evaluate all
    requested calibration metrics + long-context benchmarks on it.
    Idempotent: results are written to `<save>/result_<idx>.json`; re-running
    only fills in the metrics that are still missing (or in `--force`).

Calibration metrics (`--metrics`) — the full list is METRIC_TASKS in
utils/metric_specs.py; a few landmarks:
    c4_ppl              C4 PPL (test split, n_sample=128, seqlen=2048)
    wt2_ppl_sl8192      wikitext2 PPL at an 8192-token window (35 windows);
                        c4_ppl_sl8192 too. The ladder is 2048 and 8192 only.
    wt2_ppl_sl8192_s512 8192 windows through the REAL KV-cache path (stride)
    gov_ppl             gov_report PPL, 128 real documents @ 8192
                        (+ gov_ppl_sl2048, gov_ppl_s512, gov_ppl_pp512_s128)
    nqa_ppl             LongBench narrativeqa (novels/scripts) PPL, 128 docs
                        @ 8192 (+ nqa_ppl_sl2048, nqa_ppl_s512)
    qmsum_ppl           LongBench qmsum (meeting transcripts), 128 docs @ 8192
                        (+ qmsum_ppl_sl2048, qmsum_ppl_s512)

    All three long-document corpora span 2048 and 8192, so DOMAIN is
    separable from LENGTH: at 2048 they are directly comparable to wt2/c4_ppl
    (same window, different text), and within a corpus only the window changes.
    LongBench GRADES qmsum, so qmsum_ppl* are long-context REPORTING metrics —
    measured next to benchmark accuracy on the same documents, which is exactly
    what makes "PPL barely moved, accuracy dropped" a confound-free statement,
    but not evidence that they PREDICT that benchmark (correlation.py lists
    those cells in correlation_contamination.txt).

    EVERY PPL corpus × window additionally has the answer-phase protocol
    `<base>_pp128_s32` (prefill, then score the last 128 tokens in 32-token
    chunks — the PPL twin of gov_jsd_pp128_s32), alongside the full-sequence
    base task.
    The long-document corpora (gov_report / longbench:*) SELECT documents, so
    their groups pin `data_seed=0`: the document set is a property of the metric
    name, not of --seed. Windows stop at 8192 by design.
    wt2_jsd_pp32_s8     tightest answer window: prefill, then 32 tokens in
    gov_jsd_pp32_s8     8-token chunks (closest stand-in for decode)
    wt2_jsd             wikitext2 JSD, n_sample=128 seqlen=2048,
                        prefill_prompt=False, stride=None
    wt2_jsd_s512        wikitext2 JSD, … stride=512
    wt2_jsd_pp512_s128  wikitext2 JSD, … prefill_prompt=True last_tokens=512
                        stride=128
    gov_jsd             gov_report JSD, n_sample=8 seqlen=8192 min_seqlen=8192
                        (trunc=256 sw=64 alpha=1 beta=-1; no key-token)
    gov_jsd_kt          gov_report JSD, … with `--key_token_path`

Long-context benchmarks:
    longbench / longbench_e / ruler  (same as post_search.py block)

    Both persist the GENERATION of every example, not just the aggregate:
      RULER     → <ruler_result_path>[/len<L>]/per_example_s<seed>.jsonl —
                  task, sample_index, seed, requested/sample length,
                  input_sha256, context/generated token counts, prediction,
                  references, score. Keyed by seed because the seed decides
                  which samples exist (eval_ruler re-seeds before building the
                  dataset), so seeds never overwrite each other.
      LongBench → <longbench_result_path>/pred[_e]/<dataset>.jsonl — pred,
                  answers, all_classes, length + dataset, _id, context/generated
                  tokens, input_sha256 and the per-example `score` (stamped in
                  by eval_longbench_preds, so it is the same number the reported
                  average is computed from)
    Prompts are NOT stored: they are regenerable from (dataset, _id) /
    (seed, task, sample_index), and input_sha256 proves a regenerated prompt is
    the one the model saw.

    `--ruler_length` takes one or MORE context lengths and is independent of
    `--n_token` (memory accounting). Each length is run as its own full
    `--ruler_sample` sweep — a single eval_ruler call with several lengths
    would shuffle them together and keep only nsample samples in total — and
    multi-length results are reported per length as `<task>_len<L>` with raw
    artefacts under `<ruler_result_path>/len<L>/scores.json`.

`--metrics all` (default) runs all calibration metrics (PPL/loss). Keys
listed explicitly on --metrics are force-rerun; keys expanded by 'all'
are incremental (skip if already done).

Benchmarks are toggled via dedicated flags: `--ruler`, `--longbench`,
`--longbench_e`. Each flag opts the benchmark in and force-reruns it
(snapshotting the previous result under ${SAVE}/archive/<ts>/ first).

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
        --key_token_path key_token

`--key_token_path` is the DIRECTORY the archives live in (default `key_token`).
The root itself is derived per metric:
    <dir>/kt_eval-<evaluator>_tgt-<target>_<layout>/<corpus>_<protocol>
`evaluator` comes from the metric NAME (`..._q72b` / `..._l8b`), `target` is the
model being measured, and `layout` is `raw` or `chat-a<answer window>`. So the
metric name pins which key tokens were used, and a run cannot silently pair a
metric with another evaluator's archive. The archive is per (evaluator, target, corpus, seqlen, min_seqlen,
seed, trunc_len, sliding_window) -- meta.json records all of it and a mismatch
is a hard error, never a silent fallback.
"""
import os
import glob
import hashlib
import json
import csv
import shutil
import argparse
import traceback
import warnings
from copy import deepcopy
from time import time, strftime

import numpy as np
import torch
from tqdm import tqdm

from evaluator import LlamaEvaluator
from utils.func import (init_run, build_expr_map, build_nd, comp_key_order,
                        configure_model_cache, get_net_info, clean_up,
                        set_seed, init_accelerator, process_dtype, RunCtx,
                        arch_sha8, bench_stamp, stamp_artifact_dir)
from utils.metric_specs import key_token_root, spec_sha8, TASKS_BY_NAME as _TASKS
from utils.select import (build_arch, select_valid_nd_idx, assemble_F,
                          LazyPs, draw_random, quantile_select, axis_of_map,
                          coverage_subset_nsga2_extras, per_axis_metric)
from utils.eval import eval_metric, eval_loss, LazyGpuList
from utils.data import get_tokenizer
from utils.longbench import (pred_longbench, eval_longbench_preds,
                             LONGBENCH_DATASETS, LONGBENCH_E_DATASETS)
from utils.ruler import eval_ruler

warnings.simplefilter("ignore")


# ════════════════════════════════════════════════════════════════════════════
# Calibration metric specs
# ════════════════════════════════════════════════════════════════════════════
# The registry (GROUPS / METRIC_TASKS and every metric NAME) lives in
# utils/metric_specs.py so correlation.py and post_search.py measure the same
# thing when they are given the same name. Imported here under the original
# names; this module's own code is unchanged.
from utils.metric_specs import (GROUPS, METRIC_TASKS, METRIC_KEYS, BENCH_KEYS,
                                ALL_KEYS, TASKS_BY_NAME, precompute_groups,
                                move_dense_to_cpu, run_task)


# ════════════════════════════════════════════════════════════════════════════
# Overlap check — is a calibration corpus ALSO graded by the benchmark?
# ════════════════════════════════════════════════════════════════════════════
# INFORMATIONAL, not a gate. Some metrics here are long-context REPORTING
# metrics (a final-table number next to benchmark accuracy) rather than proxy
# candidates, and for those the overlap is intentional — nothing is fit to the
# text (the search objective is wikitext2 JSD; post_search.select_joint picks on
# the archive's stored loss and measures these afterwards), and PPL over a
# document is a different target than accuracy over the answer generated from
# it. Same-corpus is in fact the confound-free way to say "PPL barely moved,
# accuracy dropped X%".
# What the overlap DOES cost is the predictive reading of that one cell: an
# overlapping (metric, benchmark) pair shares per-architecture, per-document
# noise, so its correlation is optimistic as evidence that the metric PREDICTS
# the benchmark. Hence: measure everything, report the overlap, and let the
# reader (or --corr_drop_contaminated) decide. Measured on this box
# (Llama-3.1 tokenizer):
#   longbench:qmsum  → 128/128 calibration documents are ALSO graded by
#                      LongBench (same THUDM/LongBench qmsum test split);
#                      registered on purpose as a reporting metric
#   gov_report       → CLEAN: LongBench's gov_report subset comes from the
#                      GovReport *validation* split (192/200 match there,
#                      4/200 in test), while the calibration loader reads
#                      *test* → 0/128 selected docs appear in the benchmark
#   longbench:narrativeqa, wikitext2, c4, gsm8k → not scored by either list
# ════════════════════════════════════════════════════════════════════════════
# Measurement directory — one folder per measurement CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
# archs.csv is the ARCH POOL and is shared; the numbers measured on that pool
# are only comparable to each other when the model + quantization config that
# produced them is the same. aggregate() puts every result_<idx>.json into ONE
# table and correlates across rows, so a config change mid-sweep would silently
# mix incomparable rows. Keying the directory by the config makes that
# structurally impossible instead of something to detect afterwards:
#
#   <save>/archs.csv, sample_meta.json      the pool (one, shared)
#   <save>/m_<cfg_sha8>/meta.json           the config, in full and readable
#   <save>/m_<cfg_sha8>/result_<idx>.json   values + per-metric spec hashes
#   <save>/m_<cfg_sha8>/longbench_<idx>/…   benchmark artefacts (P4-stamped)
#
# The name is a short hash, not the config spelled out: this repo has already
# hit the 255-byte directory-name limit. What must NOT enter the key: which
# metrics a run happened to measure (the harness fills metrics in incrementally,
# one at a time) — that is handled per entry by spec_sha8.
_CONFIG_KEYS = ('model_name', 'dtype', 'w_method', 'kv_method', 'w_bits',
                'k_bits', 'v_bits', 'w_group_size', 'k_group_size',
                'v_group_size', 'residual_length', 'attn_sink',
                'k_quant_scheme', 'v_quant_scheme', 'seed')


def measurement_config(args):
    """The axes whose change invalidates EVERY number in a directory."""
    return {k: getattr(args, k, None) for k in _CONFIG_KEYS}


def measurement_dir(args, create=False):
    """Resolve (and optionally create+stamp) this run's measurement directory.

    Back-compat: a save dir that already holds result_*.json at its ROOT and no
    m_* subdirectory keeps using the root — the 670 existing result files stay
    exactly where they are and remain readable.
    """
    save = args.save or '.'
    if getattr(args, 'measure_dir', ''):
        mdir = args.measure_dir
    else:
        legacy = (glob.glob(os.path.join(save, 'result_*.json'))
                  and not glob.glob(os.path.join(save, 'm_*')))
        cfg = measurement_config(args)
        sha = hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode('utf-8')
        ).hexdigest()[:8]
        mdir = save if legacy else os.path.join(save, f'm_{sha}')
    if create:
        os.makedirs(mdir, exist_ok=True)
        if mdir != save:
            stamp_artifact_dir(mdir, dict(kind='measurement',
                                          **measurement_config(args)))
    return mdir


def reroot(path, save, mdir):
    """Move a benchmark artefact path from <save>/x into <mdir>/x.

    The shell builds these paths from SAVE (it cannot know the config hash), so
    correlation re-roots anything that sits directly under SAVE into the
    measurement directory. Paths outside SAVE are left alone.
    """
    if not path or mdir == save:
        return path
    save_n = os.path.normpath(save)
    path_n = os.path.normpath(path)
    if path_n == save_n or not path_n.startswith(save_n + os.sep):
        return path
    return os.path.join(mdir, os.path.relpath(path_n, save_n))


def _bench_spec(args, which):
    """Definition fingerprint of a BENCHMARK entry — what would make two
    stored scores incomparable (task list, context lengths, sample count)."""
    if which == 'ruler':
        cfg = dict(task=sorted(args.ruler_task or []),
                   length=sorted(int(L) for L in (args.ruler_length or [])),
                   nsample=args.ruler_sample, gen_toks=args.ruler_gen_toks)
    else:
        from utils.longbench import LONGBENCH_DATASETS, LONGBENCH_E_DATASETS
        cfg = dict(datasets=sorted(LONGBENCH_E_DATASETS if which == 'longbench_e'
                                   else LONGBENCH_DATASETS))
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:8]


def _metric_corpus(metric, needle_task=''):
    """The corpus a calibration metric reads, as a comparable key."""
    t = TASKS_BY_NAME.get(metric)
    if t is None:
        return None
    _k, _g, ds, kw = t
    if ds:
        return ds.split(':')[-1]
    if str(kw.get('kind', '')).startswith('needle'):
        # needle_* prompts come from RULER's own niah generators, so they
        # collide with a RULER task of the SAME name (and length/seed).
        return f'ruler:{needle_task}' if needle_task else None
    return None


def _bench_col_corpora(col, all_bench_cols):
    """Which corpora does one aggregate column score? `<bk>__avg` covers every
    dataset of that benchmark, so an overlap in ONE dataset contaminates the
    average too."""
    bk, _, rest = col.partition('__')
    name = rest.split('__')[0]              # strip the LongBench-E length bucket
    if bk == 'longbench':
        return set(LONGBENCH_DATASETS) if name == 'avg' else {name}
    if bk == 'longbench_e':
        return set(LONGBENCH_E_DATASETS) if name == 'avg' else {name}
    if bk == 'ruler':
        import re as _re
        strip = lambda t: _re.sub(r'_len\d+$', '', t)   # noqa: E731
        if name == 'avg':
            return {f'ruler:{strip(c.split("__", 1)[1])}' for c in all_bench_cols
                    if c.startswith('ruler__') and not c.endswith('__avg')}
        return {f'ruler:{strip(name)}'}
    return set()


# Same dataset NAME, disjoint DOCUMENTS — measured, not assumed, so the
# name-based rule below does not raise a false alarm on it.
DISJOINT_BY_SPLIT = {
    ('gov_report', 'longbench_e'):
        "LongBench-E's gov_report is the GovReport VALIDATION split "
        "(192/200 of its documents match launch/gov_report[validation] on a "
        "120-char probe, only 4/200 match [test]); the calibration loader reads "
        "split='test' → 0/128 selected documents are benchmark documents",
}


def contaminated_pairs(metric_cols, bench_cols, needle_task=''):
    """[(metric, bench column)] measured over documents the benchmark grades.

    NOT an error list — see the module comment: for a reporting metric the
    overlap is intended, it only disqualifies that cell as evidence of
    PREDICTION. Name-based, minus the measured DISJOINT_BY_SPLIT exemptions.
    Over-flagging is the safe direction: for needle_* vs a RULER task of the
    same name the prompts come from one generator and one seed, so they are
    literally identical whenever the context length matches too, and drawn from
    the same needle/haystack pool even when it does not."""
    corpus = {m: _metric_corpus(m, needle_task) for m in metric_cols}
    out = []
    for b in bench_cols:
        bk = b.partition('__')[0]
        scored = _bench_col_corpora(b, bench_cols)
        out += [(m, b) for m in metric_cols
                if corpus[m] is not None and corpus[m] in scored
                and (corpus[m], bk) not in DISJOINT_BY_SPLIT]
    return out



# ════════════════════════════════════════════════════════════════════════════
# Sample mode — build combo space, random-sample N archs, write archs.csv
# ════════════════════════════════════════════════════════════════════════════
def cmd_sample(args):
    ctx = _build_ctx(args)
    expr_map = build_expr_map(args, ctx)
    nd = build_nd(args, ctx, expr_map)
    expr_keys, _esm, _efm = nd.expr_keys, nd.esm, nd.efm
    _pool_desc = (f"ε-band(rel={args.front_eps_rel})" if args.front_eps_rel > 0
                  else f"expr_front={args.expr_front}")
    print(f"[correlation/sample] pool={_pool_desc} → "
          f"{'lazy comp_obj-pruned' if getattr(nd, 'lazy', False) else 'dense'} path  "
          f"(per-axis {dict(zip(expr_keys, nd.nd_shape))}, "
          f"n_total={nd.n_total:.3e}, expr_keys={expr_keys})")

    # ── Factorial grid mode (--grid_sample): random/stratified per-axis
    # Pareto-front blocks → FULL cartesian product. Bypasses the quantile /
    # coverage selection entirely (paired design for interaction analysis).
    if args.grid_sample:
        valid_nd_idx, grid_meta, samp_desc = _grid_sample(args, nd, expr_keys,
                                                          ctx)
        _write_sample_outputs(args, ctx, nd, expr_keys, valid_nd_idx,
                              samp_desc, grid_meta)
        return

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

    _write_sample_outputs(args, ctx, nd, expr_keys, valid_nd_idx, samp_desc)


def _grid_sample(args, nd, expr_keys, ctx):
    """--grid_sample: paired factorial design over the per-axis Pareto fronts.

    Draws --grid_n blocks per axis (pure random by default; with
    --grid_stratify: sort the axis front by its own comp, split into grid_n
    quantile bins, one random member per bin — pure random from a front
    oversamples whichever comp region the front is densest in), then emits
    the FULL cartesian product as valid_nd_idx in axis-major order, so the
    (block_i, block_j) cell structure is recoverable from the row index (and
    explicitly from the grid_<axis> CSV columns).

    Single-axis --comp_obj bounds (wbits / eff_kvbits / …) filter their own
    axis's pool; multi-axis objectives (memory) are rejected — they cannot
    bound one axis independently of the others.
    """
    if getattr(nd, 'lazy', False):
        raise SystemExit("[grid_sample] requires --expr_front (per-axis "
                         "Pareto fronts) — the lazy full-archive path has no "
                         "per-axis front to grid over.")
    K = len(expr_keys)
    grid_n = list(args.grid_n or [12])
    if len(grid_n) == 1:
        grid_n = grid_n * K
    if len(grid_n) != K:
        raise SystemExit(f"[grid_sample] --grid_n takes 1 or {K} values "
                         f"(expr axes {list(expr_keys)}); got {grid_n}")
    if not (len(args.comp_obj) == len(args.comp_obj_min)
            == len(args.comp_obj_max)):
        raise SystemExit("[grid_sample] comp_obj / comp_obj_min / "
                         "comp_obj_max lengths must match")

    # single-axis comp_obj bounds → per-axis pool filter
    amap = axis_of_map(expr_keys)
    axis_bounds = {}
    for o, lo, hi in zip(args.comp_obj, args.comp_obj_min, args.comp_obj_max):
        ax_key = amap.get(o)
        if ax_key is None or ax_key not in expr_keys:
            raise SystemExit(f"[grid_sample] comp_obj '{o}' does not reduce "
                             f"to one expr axis of {list(expr_keys)}; only "
                             f"single-axis objectives can bound a grid pool.")
        axis_bounds.setdefault(ax_key, []).append((o, lo, hi))

    rng = np.random.default_rng(args.seed)
    pa_cache = {}
    chosen, grid_meta = [], {}
    for ax, key in enumerate(expr_keys):
        efm_k = np.asarray(nd.efm[key])
        n_front = len(efm_k)
        pool = np.arange(n_front)
        for o, lo, hi in axis_bounds.get(key, []):
            _, comp_o = per_axis_metric(o, expr_keys, nd.esm, ctx.config,
                                        ctx.group_size, args.n_token,
                                        cache=pa_cache)
            pool = pool[(comp_o[pool] >= lo) & (comp_o[pool] <= hi)]
        if len(pool) == 0:
            raise SystemExit(f"[grid_sample] axis '{key}': comp_obj filter "
                             f"left an empty pool (front size {n_front})")
        comp_own = efm_k[:, 1]          # the axis archive's own comp column
        n_want = grid_n[ax]
        if len(pool) <= n_want:
            print(f"[grid_sample] axis '{key}': pool {len(pool)} <= "
                  f"grid_n {n_want} — taking the whole pool")
            picks = pool
        elif args.grid_stratify:
            # comp-sorted quantile bins, one member per bin; the two endpoint
            # bins are pinned to the exact min/max-comp members so the budget
            # corners are always in the design (a random member of a wide
            # outer bin can sit far from the true corner).
            order = pool[np.argsort(comp_own[pool])]
            bins = np.array_split(order, n_want)
            picks = np.array(
                [b[0] if bi == 0 else
                 b[-1] if bi == len(bins) - 1 else rng.choice(b)
                 for bi, b in enumerate(bins)])
        else:
            picks = rng.choice(pool, size=n_want, replace=False)
        picks = picks[np.argsort(comp_own[picks])]   # comp-ascending rows
        chosen.append(picks)
        grid_meta[key] = {
            'subnet_idx': [int(i) for i in picks],
            'comp': [float(comp_own[i]) for i in picks],
            'metric': [float(efm_k[i, 0]) for i in picks],
        }
        print(f"[grid_sample] axis '{key}': front={n_front} pool={len(pool)} "
              f"picked={len(picks)} "
              f"comp=[{comp_own[picks].min():.3f}, {comp_own[picks].max():.3f}]"
              + (' stratified' if args.grid_stratify else ' random'))

    mesh = np.meshgrid(*chosen, indexing='ij')
    valid_nd_idx = np.stack([m.ravel() for m in mesh], axis=1)
    desc = ('factorial grid '
            + 'x'.join(str(len(c)) for c in chosen)
            + f' = {len(valid_nd_idx)} archs'
            + (' (stratified)' if args.grid_stratify else ' (random)'))
    return valid_nd_idx, grid_meta, desc


def _write_sample_outputs(args, ctx, nd, expr_keys, valid_nd_idx, samp_desc,
                          grid_meta=None):
    """Shared archs.csv + sample_meta.json writer for both cmd_sample paths.
    With grid_meta, per-axis `grid_<axis>` block-id columns (the subnet index
    into each axis front) are appended so the factorial design is
    reconstructable from the CSV alone (eval/aggregate use DictReader, so
    extra columns are transparent to them)."""
    print(f"[correlation/sample] final |I|={len(valid_nd_idx)}  "
          f"method={samp_desc}")

    F = assemble_F(valid_nd_idx, expr_keys, nd.efm, nd.comp_nd_list,
                   nd.new_metric_nd)
    # F columns: [combined_metric | (metric_axis_i, comp_axis_i) * K | comp_obj_i]
    per_axis_metric_cols = [F[:, 1 + 2 * i] for i in range(len(expr_keys))]

    comp_keys = comp_key_order(ctx.config, ctx.group_size)
    metric_col_names = [f"metric_{k}" for k in expr_keys]
    header = (['idx', 'arch_json'] + comp_keys + metric_col_names
              + ['combined_metric'])
    if grid_meta is not None:
        header += [f"grid_{k}" for k in expr_keys]

    save_dir = args.save or '.'
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'archs.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, row in enumerate(valid_nd_idx):
            arch = build_arch(ctx.default_arch, expr_keys, nd.esm, row)
            comp = get_net_info(arch, ctx.config, ctx.group_size,
                                n_token=args.n_token,
                                attn_sink=args.attn_sink)
            comp_vals = [comp[k] for k in comp_keys]
            metric_vals = [float(per_axis_metric_cols[ax][i])
                           for ax in range(len(expr_keys))]
            out = ([i, json.dumps(arch, separators=(',', ':'))]
                   + comp_vals + metric_vals + [float(F[i, 0])])
            if grid_meta is not None:
                out += [int(v) for v in row]
            w.writerow(out)
    print(f"[correlation/sample] wrote {len(valid_nd_idx)} archs → {csv_path}")

    # also save the meta (so eval mode can sanity-check the model/expr context)
    meta = {
        'model_name': args.model_name, 'model_path': args.model_path,
        'config': args.config, 'expr_keys': list(expr_keys),
        'w_expr': args.w_expr, 'kv_expr': args.kv_expr,
        'kvdim_expr': args.kvdim_expr, 'eff_kv_expr': args.eff_kv_expr,
        'expr_front': args.expr_front, 'front_eps_rel': args.front_eps_rel,
        'n_token': args.n_token,
        'attn_sink': args.attn_sink,
        'comp_obj': args.comp_obj, 'comp_obj_min': args.comp_obj_min,
        'comp_obj_max': args.comp_obj_max,
        'n_archs': (len(valid_nd_idx) if grid_meta is not None
                    else args.n_archs),
        'seed': args.seed,
    }
    if grid_meta is not None:
        meta['grid'] = {'stratify': bool(args.grid_stratify),
                        'grid_n': [len(grid_meta[k]['subnet_idx'])
                                   for k in expr_keys],
                        'axes': grid_meta}
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


def _archive_existing_results(args, result_path, results, rerunning_keys):
    """If any key about to be (re)computed already has a NON-error entry in
    `results`, snapshot result_<IDX>.json + MOVE the matching raw benchmark
    artefacts under `${SAVE}/archive/<ts>/` before they get overwritten.

    No-op when keys are being added for the first time (the common
    add-a-new-metric path) — only fires when an actual overwrite is about to
    happen. raw artefacts are moved (not copied) because longbench dirs in
    particular are large and disk usage would balloon otherwise.

    aggregate mode scans only `result_*.json` directly under `<save>/`, so
    archived snapshots are invisible to it by design (history, not state).
    """
    if args.no_archive:
        return

    def _valid(k):
        v = results.get(k)
        return v is not None and not (isinstance(v, dict) and 'error' in v)

    overwriting = [k for k in rerunning_keys if _valid(k)]
    if not overwriting:
        return

    ts = strftime('%y%m%d_%H%M%S')
    save_dir = os.path.dirname(result_path) or '.'
    archive_dir = os.path.join(save_dir, 'archive', ts)
    os.makedirs(archive_dir, exist_ok=True)

    # Snapshot the JSON (copy, since we keep writing to the live one).
    shutil.copy2(result_path,
                 os.path.join(archive_dir, os.path.basename(result_path)))

    # Move raw bench artefacts for the bench keys actually being rerun.
    moved = []
    bench_artefacts = {
        'ruler': args.ruler_result_path,
        'longbench': args.longbench_result_path,
        'longbench_e': args.longbench_e_result_path,
    }
    for k, src in bench_artefacts.items():
        if k not in overwriting or not src or not os.path.exists(src):
            continue
        dst = os.path.join(archive_dir, os.path.basename(src.rstrip('/')))
        shutil.move(src, dst)
        moved.append(os.path.basename(src.rstrip('/')))

    print(f"[archive] snapshotted result_<idx>.json + moved {len(moved)} raw "
          f"artefact(s) → {archive_dir}\n"
          f"          rerunning: {overwriting}"
          + (f"  moved: {moved}" if moved else ""))


def _resolve_metric_set(arg):
    """Return (keys, rerun_set) for CALIBRATION metrics only (PPL/loss).

    `'all'` (or empty --metrics) expands to all METRIC_KEYS; `'none'` requests
    benchmark-only evaluation. Benchmarks
    (ruler/longbench/longbench_e) are rejected here with a redirect to
    `--benchmarks` — they are not metrics by this script's taxonomy.

    Any token listed explicitly (i.e. not the magic `'all'`) is added to
    `rerun_set` → cmd_eval treats those specific keys as force-rerun, so
    `--metrics wt2_jsd` re-evaluates wt2_jsd even when an entry exists.
    Keys expanded by `'all'` are NOT in rerun_set → they keep the
    add-only-if-missing behaviour.
    """
    if not arg:
        return list(METRIC_KEYS), set()

    tokens = [tok for x in arg
              for tok in x.replace(',', ' ').split() if tok]
    if 'none' in tokens:
        if len(tokens) != 1:
            raise SystemExit("--metrics none is exclusive; do not combine it "
                             "with calibration metric names")
        return [], set()

    expanded, rerun_set, seen = [], set(), set()
    for x in arg:
        for tok in (p for p in x.replace(',', ' ').split() if p):
            if tok == 'all':
                for k in METRIC_KEYS:
                    if k not in seen:
                        seen.add(k)
                        expanded.append(k)
                continue
            if tok in BENCH_KEYS:
                raise SystemExit(
                    f"--metrics: '{tok}' is a benchmark, not a metric. "
                    f"Pass it via --benchmarks instead.")
            if tok not in METRIC_KEYS:
                raise SystemExit(
                    f"--metrics: unknown key '{tok}'. "
                    f"Valid: {METRIC_KEYS} (or 'all').")
            if tok not in seen:
                seen.add(tok)
                expanded.append(tok)
            rerun_set.add(tok)
    return expanded, rerun_set


def _benchmarks_from_args(args):
    """Return (keys, rerun_set) for benchmarks. Each benchmark is toggled
    by its own boolean flag (`--ruler`, `--longbench`, `--longbench_e`);
    flagged-on benchmarks are added to rerun_set (force-rerun on next call).
    Empty by default → no benchmarks run unless a flag is set.
    """
    keys, rerun_set = [], set()
    for k, flag in (('longbench',   args.longbench),
                    ('longbench_e', args.longbench_e),
                    ('ruler',       args.ruler)):
        if flag:
            keys.append(k)
            rerun_set.add(k)
    return keys, rerun_set


# The CPU-offload shim and the one-FP-pass group builder live in
# utils/metric_specs.py (shared with post_search.py). Thin aliases keep this
# module's call sites unchanged.
_LazyGpuList = LazyGpuList
_move_all_dense_logits_to_cpu = move_dense_to_cpu


def _precompute_group_data(args, ctx, model_id, group_items, tasks=None):
    """One FP-teacher pass for every group of this idx → the shared builder.
    store_device='cpu' also removes the transient GPU pile-up the old local
    version had (it built on GPU, then moved).

    `tasks` (the pending calibration tasks) restricts the teacher pass to the
    (group, dataset) pairs actually consumed. Without it every dataset of a
    divergence group is materialised even when nothing reads it — e.g. asking
    only for c4_ppl used to build BOTH wikitext2 and c4 full-sequence teacher
    logits for Group A (128 x 2048 x vocab fp16 ≈ 67 GB each, on CPU).
    """
    return precompute_groups(ctx.accelerator, model_id, group_items,
                             seed=args.seed, dtype=ctx.dtype,
                             device_map=ctx.device_map, store_device='cpu',
                             tasks=tasks, fail_soft=True)


def _build_evaluator(args, ctx, *, datasets, n_sample, seqlen, min_seqlen,
                     loss_func, use_key_token, key_token_path,
                     trunc_len, sliding_window, alpha, beta,
                     last_tokens=None, precomputed=None, sides=None,
                     data_seed=None):
    """One LlamaEvaluator with the requested data-side config. `last_tokens`
    here is set on the evaluator at init so dense_logits gets pre-masked to
    the last N positions — must match the eval_loss last_tokens used at
    metric-call time (eval_loss compares len-N logits vs len-N dense).
    Dense_logits is moved to CPU right after build (see _move_all_dense_logits_to_cpu).
    `sides` (which loader sides to build) and `data_seed` (pinned document
    selection for long-doc groups) are GROUP-spec keys consumed by
    precompute_groups; accepted and ignored here because the loaders arrive
    pre-built via `precomputed`."""
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
        attn_sink=args.attn_sink,
        k_quant_scheme=args.k_quant_scheme, v_quant_scheme=args.v_quant_scheme,
        k_pruning_dim=kpd, v_pruning_dim=vpd,
        loss_func=loss_func, last_tokens=last_tokens,
        use_key_token=use_key_token, trunc_len=trunc_len,
        sliding_window=sliding_window, alpha=alpha, beta=beta,
        key_token_path=key_token_path,
        precomputed_train_loaders=(precomputed or {}).get('train_loaders'),
        precomputed_test_loaders=(precomputed or {}).get('test_loaders'),
        precomputed_dense_logits=(precomputed or {}).get('dense_logits'),
        precomputed_key_token_list=(precomputed or {}).get('key_token_list'))
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


def _run_needle_nll(args, ctx, evaluator, model_id, *,
                    stride=0, prefill_prompt=False, last_tokens=None):
    """Cheap NIAH cross-entropy NLL via utils.eval.eval_loss. The loader
    yields (input_ids, attention_mask, labels) with labels=-100 on prompt
    tokens, so eval_loss's get_loss_mask naturally restricts CE to the
    answer span. No bespoke forward / loss loop — same metric math as all
    other cross_entropy calibration paths.

    stride / prefill_prompt / last_tokens forward through eval_loss
    untouched, so needle_nll_s512 / needle_nll_pp512_s128 reuse this path
    by varying these args.
    """
    loader = _build_needle_loader(args, model_id, evaluator.model.device)
    if len(loader) == 0:
        raise RuntimeError("[needle_nll] no usable NIAH prompts after tokenisation")
    use_cache = stride > 0 or prefill_prompt
    configure_model_cache(args, evaluator.model, use_cache=use_cache)
    return eval_loss(model=evaluator.model, accelerator=ctx.accelerator,
                     loader=loader, seqlen=int(args.needle_seqlen),
                     loss_func='cross_entropy', dense_logits_list=None,
                     key_token_list=None, stride=stride,
                     last_tokens=last_tokens, prefill_prompt=prefill_prompt)


# Module-level in-process cache for FP teacher needle logits — keyed by all
# inputs that affect the prompts AND the FP forward. Loaded from disk on
# first access per process so re-runs across archs don't rebuild it.
_NEEDLE_DENSE_CACHE = {}


def _needle_dense_cache_path(args, last_tokens):
    model_basename = os.path.basename(args.model_name.rstrip('/'))
    key = (f"{model_basename}_{args.needle_task}_{args.needle_seqlen}_"
           f"{args.needle_n_sample}_seed{args.seed}_lt{last_tokens}")
    save_dir = args.save or '.'
    cache_dir = os.path.join(save_dir, '_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'needle_dense_{key}.pt'), key


def _get_needle_dense_logits(args, ctx, model_id, device, last_tokens=512):
    """Lazy compute + cache FP teacher dense_logits on the (deterministic)
    needle prompts. Cached in-memory per process AND on disk so re-runs
    across archs (each a fresh process) don't rebuild it.

    Returns a _LazyGpuList (CPU-backed, per-position upload to `device` on
    use) — drop-in for evaluator.dense_logits[dataset] in eval_loss.
    """
    cache_path, key = _needle_dense_cache_path(args, last_tokens)
    if key in _NEEDLE_DENSE_CACHE:
        return _LazyGpuList(_NEEDLE_DENSE_CACHE[key], device)
    if os.path.exists(cache_path):
        print(f"[needle_jsd] loading cached FP dense_logits ← {cache_path}")
        dense_cpu = torch.load(cache_path, map_location='cpu', weights_only=False)
        _NEEDLE_DENSE_CACHE[key] = dense_cpu
        return _LazyGpuList(dense_cpu, device)

    # Cache miss: load FP teacher briefly to compute dense_logits.
    from utils.func import get_hfmodel
    from utils.eval import get_logits
    print(f"[needle_jsd] FP teacher logits cache miss — computing one-time "
          f"(task={args.needle_task}, n={args.needle_n_sample}, "
          f"seqlen={args.needle_seqlen}, seed={args.seed})…")
    t0 = time()
    fp_model = get_hfmodel(model_id, dtype=ctx.dtype, device_map=ctx.device_map)
    fp_model.eval()
    needle_loader = _build_needle_loader(args, model_id, fp_model.device)
    if len(needle_loader) == 0:
        del fp_model
        clean_up()
        raise RuntimeError("[needle_jsd] no needle prompts after tokenization")
    dense_gpu = get_logits(fp_model, needle_loader, key_token_list=None,
                           last_tokens=last_tokens)
    # Move to CPU for storage (matches _move_all_dense_logits_to_cpu pattern).
    dense_cpu = [[t.detach().to('cpu', copy=False) for t in batch]
                 for batch in dense_gpu]
    del fp_model, dense_gpu
    clean_up()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.save(dense_cpu, cache_path)
    print(f"[needle_jsd] computed + saved → {cache_path}  "
          f"({time() - t0:.1f}s)")
    _NEEDLE_DENSE_CACHE[key] = dense_cpu
    return _LazyGpuList(dense_cpu, device)


def _run_needle_jsd(args, ctx, evaluator, model_id, *,
                    stride=0, prefill_prompt=False, last_tokens=512):
    """JSD variant of needle eval. Mirrors _run_needle_nll but uses cached
    FP-teacher dense_logits + loss_func='jsd'. The needle prompts are
    seed-deterministic, so the same dense_logits applies to every arch.
    """
    loader = _build_needle_loader(args, model_id, evaluator.model.device)
    if len(loader) == 0:
        raise RuntimeError("[needle_jsd] no usable NIAH prompts after tokenisation")
    dense_logits = _get_needle_dense_logits(
        args, ctx, model_id, evaluator.model.device,
        last_tokens=last_tokens if last_tokens is not None else int(args.needle_seqlen))
    use_cache = stride > 0 or prefill_prompt
    configure_model_cache(args, evaluator.model, use_cache=use_cache)
    return eval_loss(model=evaluator.model, accelerator=ctx.accelerator,
                     loader=loader, seqlen=int(args.needle_seqlen),
                     loss_func='jsd', dense_logits_list=dense_logits,
                     key_token_list=None, stride=stride,
                     last_tokens=last_tokens, prefill_prompt=prefill_prompt)


def _build_gsm8k_unpadded_loader(evaluator, device):
    """Re-yield gsm8k batches with trailing padding stripped per example.
    The existing Group D `train_loaders['gsm8k']` was built by
    get_gsm8k_trainenc which right-pads each example to seqlen=2048; here
    we use the attention_mask sum as the real length and slice to it.

    Why: for prefill_prompt + stride mode to be meaningful, the answer
    tokens must lie at the END of the sequence (so prefill covers the
    question and stride covers the answer). With padding, the answer sits
    in the middle and the `last_tokens` window misses it.

    Dense_logits sharing: evaluator.dense_logits['gsm8k'] (computed by
    get_logits on the padded loader at labels != -100 positions) is
    IDENTICAL to what we'd get from the unpadded loader — causal attention
    means the FP logits at the answer positions depend only on tokens
    [0:answer_pos], which are the same in both versions. So no
    recomputation needed.
    """
    padded = evaluator.train_loaders['gsm8k']
    batches = []
    for ids, attn, lab in padded:
        for i in range(ids.shape[0]):
            n = int(attn[i].sum().item())
            if n < 2:
                continue
            batches.append((ids[i:i + 1, :n].to(device),
                            attn[i:i + 1, :n].to(device),
                            lab[i:i + 1, :n].to(device)))

    class _UnpaddedGsm8k:
        def __iter__(self):
            return iter(batches)

        def __len__(self):
            return len(batches)

    return _UnpaddedGsm8k()


def _run_gsm8k_unpad_pp(args, ctx, evaluator, *,
                       stride=128, prefill_prompt=True, last_tokens=512):
    """gsm8k JSD with answer-end + prefill_prompt + stride. Builds an
    unpadded view of Group D's padded loader, reuses Group D's dense_logits
    (causal attention → identical FP logits at answer positions), delegates
    to utils.eval.eval_loss."""
    loader = _build_gsm8k_unpadded_loader(evaluator, evaluator.model.device)
    if len(loader) == 0:
        raise RuntimeError("[gsm8k_jsd_pp_s128] no usable examples")
    dense_logits = evaluator.dense_logits.get('gsm8k')
    use_cache = stride > 0 or prefill_prompt
    configure_model_cache(args, evaluator.model, use_cache=use_cache)
    return eval_loss(model=evaluator.model, accelerator=ctx.accelerator,
                     loader=loader, seqlen=2048,
                     loss_func='jsd', dense_logits_list=dense_logits,
                     key_token_list=None, stride=stride,
                     last_tokens=last_tokens, prefill_prompt=prefill_prompt)


def _run_calibration_task(args, ctx, evaluator, dataset, eval_kwargs):
    """One calibration metric on the prepared evaluator+model → shared runner."""
    return run_task(args, ctx.accelerator, evaluator, dataset, eval_kwargs)


def _quant_meta(args):
    """The quantization configuration a benchmark artefact was produced under —
    the identity half of meta.json (see utils.func.stamp_artifact_dir)."""
    return dict(model=args.model_name, w_method=args.w_method,
                kv_method=args.kv_method, w_bits=args.w_bits,
                k_bits=args.k_bits, v_bits=args.v_bits,
                w_group_size=args.w_group_size,
                residual_length=args.residual_length, attn_sink=args.attn_sink,
                k_quant_scheme=args.k_quant_scheme,
                v_quant_scheme=args.v_quant_scheme, seed=args.seed)


def _run_benchmark_block(args, model, model_id, which, arch=None, idx=None):
    """which ∈ {'longbench', 'longbench_e', 'ruler'}; mirrors post_search.run_benchmarks
    but isolated per block so eval mode can run only what was requested.

    Every artefact carries provenance twice: `stamp` on each per-example row
    (run_id / arch_sha8 / idx) and meta.json on the directory, which rotates the
    directory away if it already holds a different configuration."""
    stamp = bench_stamp(arch, idx)
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
        stamp_artifact_dir(result_path,
                           dict(benchmark=which, **_quant_meta(args),
                                **{k: v for k, v in stamp.items() if k != 'run_id'}))
        t0 = time()
        preds = pred_longbench(model, tokenizer=get_tokenizer(model_id),
                               save_path=result_path,
                               longbench_config=args.longbench_config,
                               e=e, model_name=args.model_name, stamp=stamp,
                               topk_logits=args.longbench_topk_logits)
        # Score this run's predictions in memory (still writes result.json);
        # avoids re-reading the dir, which could mix in stale .jsonl files.
        # This also rewrites <dataset>.jsonl with the per-example `score`.
        scores = dict(eval_longbench_preds(preds, e, save_path=result_path))
        print(f"[{which}] per-example generations → {result_path}/"
              f"pred{'_e' if e else ''}/<dataset>.jsonl "
              f"({sum(len(v) for v in preds.values())} rows)")
        scores['_time'] = time() - t0
        return scores

    if which == 'ruler':
        if not args.ruler_yaml_path or not args.ruler_task:
            raise SystemExit("--ruler: --ruler_yaml_path and --ruler_task required")
        clean_up()
        configure_model_cache(args, model, use_cache=True)
        t0 = time()
        # ── ONE eval_ruler call PER context length ──
        # eval_ruler's builders yield `nsample` samples per requested length and
        # CHAIN them, then `.shuffle().select(range(nsample))` keeps nsample
        # MIXED across lengths — so passing several lengths at once used to give
        # one per-task score averaged over a random length mixture (and only
        # nsample/len(lengths) samples per length). Looping here keeps every
        # length at its full `--ruler_sample` and reports the lengths separately.
        lengths = [int(L) for L in (args.ruler_length or [])]
        if not lengths:
            raise SystemExit("--ruler: --ruler_length needs at least one value")
        multi = len(lengths) > 1
        # --ruler_result_path is a FOLDER (parity with longbench's dir layout);
        # eval_ruler wants a single JSON path. With several lengths each gets
        # its own `len<L>/scores.json` so the raw artefacts stay separable and
        # _archive_existing_results can still move the whole folder as one unit.
        result_dir = args.ruler_result_path or ''
        scores = {}
        for L in lengths:
            scores_file = ''
            if result_dir:
                sub = os.path.join(result_dir, f'len{L}') if multi else result_dir
                stamp_artifact_dir(sub, dict(
                    benchmark='ruler', ruler_task=list(args.ruler_task or []),
                    length=L, nsample=args.ruler_sample,
                    batch_size=args.ruler_batch_size,
                    gen_toks=args.ruler_gen_toks, **_quant_meta(args),
                    **{k: v for k, v in stamp.items() if k != 'run_id'}))
                scores_file = os.path.join(sub, 'scores.json')
            # Per-example generations are dumped by default (nsample x tasks
            # rows — tiny) next to this length's scores.json, keyed by SEED:
            # the seed decides which samples exist (eval_ruler re-seeds before
            # dataset construction), so two seeds are two different sample sets
            # and must not overwrite each other. An explicit
            # --ruler_per_example_path still wins; with several lengths it gets
            # the _len<L> suffix so the runs stay separable.
            if args.ruler_per_example_path:
                per_example_path = args.ruler_per_example_path
                if multi:
                    base, ext = os.path.splitext(per_example_path)
                    per_example_path = f'{base}_len{L}{ext or ".jsonl"}'
            elif scores_file:
                per_example_path = os.path.join(
                    os.path.dirname(scores_file),
                    f'per_example_s{int(args.seed)}.jsonl')
            else:
                per_example_path = ''
            print(f"[correlation/eval]   ruler @ length={L} "
                  f"(tasks={args.ruler_task}, nsample={args.ruler_sample})")
            one = eval_ruler(model, tokenizer=get_tokenizer(model_id),
                             model_id=model_id,
                             tasks=args.ruler_task, yaml_path=args.ruler_yaml_path,
                             batch_size=args.ruler_batch_size, length=[L],
                             nsample=args.ruler_sample, gen_toks=args.ruler_gen_toks,
                             result_path=scores_file,
                             per_example_path=per_example_path,
                             stamp=stamp, topk_logits=args.topk_logits,
                             seed=args.seed)
            # eval_ruler returns THIS call's scores; the file it writes is merged
            # with whatever a previous run left in that dir, so the return value
            # is the authoritative one (file read kept as a fallback).
            if not one and scores_file and os.path.exists(scores_file):
                with open(scores_file) as f:
                    one = json.load(f)
            one = {k: v for k, v in (one or {}).items() if k != 'time'}
            # Single length → flat {task: score} (unchanged layout, so existing
            # `ruler__<task>` columns keep working). Several lengths → suffixed
            # `<task>_len<L>` keys; NO cross-length average is invented here
            # (aggregate's ruler__avg still means "mean over all reported cells").
            for task, v in one.items():
                scores[f'{task}_len{L}' if multi else task] = v
        scores['_lengths'] = lengths
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

    calib_keys, calib_rerun = _resolve_metric_set(args.metrics)
    bench_keys, bench_rerun = _benchmarks_from_args(args)
    requested = calib_keys + bench_keys
    rerun_set = calib_rerun | bench_rerun
    print(f"[correlation/eval] requested: {requested}"
          + (f"  explicit rerun: {sorted(rerun_set)}" if rerun_set else ""))

    # Contamination is checked HERE, not only at aggregate time: a benchmark
    # costs 29-283 min per idx, so a metric that reads documents the benchmark
    # grades should be visible before that is spent, not after.
    if calib_keys and bench_keys:
        pseudo_cols = [f'{b}__avg' for b in bench_keys]
        bad_now = contaminated_pairs(calib_keys, pseudo_cols, args.needle_task)
        if bad_now:
            offenders = sorted({m for m, _ in bad_now})
            print(f"[correlation/eval] note: {offenders} are measured over "
                  f"documents {sorted({b.split('__')[0] for _, b in bad_now})} "
                  f"grades. Fine as long-context REPORTING metrics (nothing is "
                  f"fit to them); just do not read their correlation with that "
                  f"benchmark as evidence of prediction — the pair shares "
                  f"per-arch, per-document noise. Logged at aggregate time.")

    save_dir = args.save or os.path.dirname(archs_csv) or '.'
    out_dir = measurement_dir(args, create=True)
    if out_dir != save_dir:
        print(f"[correlation/eval] measurement dir: {out_dir} "
              f"(one folder per model+quant config; archs.csv stays shared)")
    # the shell builds artefact paths from SAVE, which cannot know the config
    # hash — re-root them so a config change cannot reuse another's artefacts
    for _a in ('longbench_result_path', 'longbench_e_result_path',
               'ruler_result_path', 'ruler_per_example_path'):
        setattr(args, _a, reroot(getattr(args, _a, ''), save_dir, out_dir))
    result_path = os.path.join(out_dir, f'result_{args.idx}.json')

    # Always load the existing file (if any) so unrequested metrics are
    # preserved AND so the archive step has the full pre-overwrite snapshot.
    results = {}
    if os.path.exists(result_path):
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

    # Per-key rerun: keys listed explicitly on --metrics are force-rerun;
    # keys from `'all'` expansion only run if not already done. --force
    # forces everything regardless.
    def _spec(k):
        """Definition fingerprint of one entry: registry spec for a metric,
        benchmark configuration for a benchmark."""
        if k in _TASKS:
            return spec_sha8(k)
        return _bench_spec(args, k)

    def _stale_spec(k):
        old = (results.get('_specs') or {}).get(k)
        return old is not None and old != _spec(k)

    def _should_run(k):
        # A stored value whose spec hash no longer matches was produced by a
        # DIFFERENT definition (someone edited the group/task, or the benchmark
        # config changed) — recompute instead of silently keeping it.
        return args.force or k in rerun_set or not _done(k) or _stale_spec(k)

    pending_calib = [t for t in METRIC_TASKS
                     if t[0] in requested and _should_run(t[0])]
    pending_bench = [k for k in BENCH_KEYS
                     if k in requested and _should_run(k)]
    skipped = [k for k in requested
               if k in {t[0] for t in METRIC_TASKS} | set(BENCH_KEYS)
               and _done(k) and not (args.force or k in rerun_set)]
    if skipped:
        print(f"[correlation/eval] skipping (done): {skipped}")
    redefined = [k for k in requested if _done(k) and _stale_spec(k)]
    if redefined:
        print(f"[correlation/eval] re-measuring (definition changed since the "
              f"stored value): {redefined}")
    retried = [k for k in requested
               if isinstance(results.get(k), dict) and 'error' in results[k]]
    if retried:
        print(f"[correlation/eval] retrying previous failures: {retried}")
    if not pending_calib and not pending_bench:
        print("[correlation/eval] nothing to do — all requested metrics already present "
              "(pass --force to recompute).")
        return

    # Archive existing entries we're about to overwrite (no-op when only
    # ADDING new metrics, which is the common case). Snapshots result_<IDX>.json
    # + moves raw bench artefacts to ${SAVE}/archive/<ts>/.
    if os.path.exists(result_path):
        rerunning = ([t[0] for t in pending_calib]
                     + [k for k in pending_bench])
        _archive_existing_results(args, result_path, results, rerunning)

    results.setdefault('idx', args.idx)
    results.setdefault('arch', arch)
    results.setdefault('complexity', {k: row.get(k) for k in row
                                      if k not in ('idx', 'arch_json')})

    model_id = f'{args.model_path}/{args.model_name}'

    # AWQ/GPTQ/QEFT produce an IDENTICAL quantized model for every metric group
    # and benchmark within ONE idx (same arch), yet rebuilding the evaluator per
    # group re-runs the expensive AWQ scale search (~8 min) each time. Build the
    # quant-weight model ONCE and reuse it across all groups + benchmarks; only
    # the data side (loaders + FP-teacher dense_logits, which legitimately
    # differ per group) is rebuilt. Mirrors post_search.py, which quantizes once
    # and reuses one model for the metric AND every benchmark.
    # HQQ is unaffected — its per-group build is a cheap pre-quant disk load +
    # layer swap, so we leave that path on the default sample() each group.
    _reuse_w = any(m in args.w_method for m in ('awq', 'gptq', 'qeft'))
    _shared_qmodel = {'model': None}

    def _sample_or_reuse(ev, arch):
        """First call builds the quant model (AWQ runs once); later calls within
        this idx reattach the same model object — no re-quantization. The KV
        cache wrapper + kivi_config set at first build persist on the model and
        stay valid (arch is constant across this idx's groups)."""
        if _reuse_w and _shared_qmodel['model'] is not None:
            ev.model = _shared_qmodel['model']
            return _shared_qmodel['model']
        m = ev.sample(arch)
        if _reuse_w:
            _shared_qmodel['model'] = m
        return m

    # ── Calibration metrics, grouped to share evaluator builds ──
    groups_needed = sorted({t[1] for t in pending_calib})
    group_items = []
    for g in groups_needed:
        spec = dict(GROUPS[g])
        # any key-token GROUP, not just 'C' — a hardcoded name silently left a
        # second key-token group with key_token_path='' (groups_for() already
        # keys off the flag; this is the same check)
        if spec.get('use_key_token'):
            if not args.key_token_path:
                raise SystemExit(
                    f"a key-token metric (group '{g}') was requested but "
                    f"--key_token_path is empty. Either pass --key_token_path or "
                    f"drop the *_kt metrics from --metrics.")
            # DERIVED, not configured: the evaluator comes from the metric
            # name (its group), the target is the model being measured, and the
            # layout from the group -- so a run cannot pair a metric with
            # another evaluator's archive. --key_token_path is just the
            # directory those roots live in.
            spec['key_token_path'] = key_token_root(
                args.key_token_path, spec, args.model_name)
        else:
            spec['key_token_path'] = ''
        group_items.append((g, spec))

    # One FP-teacher pass builds every group's dense_logits (+ key tokens) up
    # front and stashes them on CPU (dense is arch-independent → one pass serves
    # all groups). Keeps the FP teacher and the (later, reused) quant model off
    # the GPU together.
    precomp = (_precompute_group_data(args, ctx, model_id, group_items,
                                      tasks=pending_calib)
               if group_items else {})

    for g, spec in group_items:
        # fail_soft precompute: a group whose corpus could not be built carries
        # an error payload. Stamp that error on the group's own metrics (same
        # shape as a per-metric failure below) and move on — the other groups
        # and the benchmarks still run.
        payload = precomp.get(g)
        if payload is not None and 'error' in payload:
            print(f"\n[correlation/eval] === group {g} SKIPPED — data build "
                  f"failed: {payload['error']} ===")
            for key, group, dataset, eval_kwargs in pending_calib:
                if group == g:
                    results[key] = {'error': payload['error'],
                                    'traceback': payload.get('traceback', '')}
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            continue
        print(f"\n[correlation/eval] === group {g}: datasets={spec['datasets']} "
              f"n_sample={spec['n_sample']} seqlen={spec['seqlen']} "
              f"use_key_token={spec['use_key_token']} ===")
        evaluator = _build_evaluator(args, ctx, precomputed=payload, **spec)
        model = _sample_or_reuse(evaluator, arch)

        for key, group, dataset, eval_kwargs in pending_calib:
            if group != g:
                continue
            print(f"[correlation/eval] → metric '{key}' on '{dataset}' kwargs={eval_kwargs}")
            t0 = time()
            try:
                kind = eval_kwargs.get('kind')
                if kind == 'needle_nll':
                    value = _run_needle_nll(
                        args, ctx, evaluator, model_id,
                        stride=eval_kwargs.get('stride', 0),
                        prefill_prompt=eval_kwargs.get('prefill_prompt', False),
                        last_tokens=eval_kwargs.get('last_tokens'))
                elif kind == 'needle_jsd':
                    value = _run_needle_jsd(
                        args, ctx, evaluator, model_id,
                        stride=eval_kwargs.get('stride', 0),
                        prefill_prompt=eval_kwargs.get('prefill_prompt', False),
                        last_tokens=eval_kwargs.get('last_tokens', 512))
                elif kind == 'gsm8k_unpad_pp':
                    value = _run_gsm8k_unpad_pp(
                        args, ctx, evaluator,
                        stride=eval_kwargs.get('stride', 128),
                        prefill_prompt=eval_kwargs.get('prefill_prompt', True),
                        last_tokens=eval_kwargs.get('last_tokens', 512))
                else:
                    value = _run_calibration_task(args, ctx, evaluator,
                                                  dataset, eval_kwargs)
                if isinstance(value, torch.Tensor):
                    value = value.item()
                results[key] = float(value)
                results.setdefault('_specs', {})[key] = _spec(key)
                print(f"[correlation/eval]   {key} = {results[key]:.6f}  "
                      f"({time() - t0:.1f}s)")
            except Exception as e:                              # noqa: BLE001
                tb = traceback.format_exc()
                results[key] = {'error': repr(e), 'traceback': tb}
                print(f"[correlation/eval]   {key} FAILED: {e!r}\n{tb}")
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)

        # Free this evaluator AND this group's precomputed dense_logits before
        # the next group — each group is evaluated exactly once, so its CPU
        # tensors are dead weight afterwards. Bounds CPU RAM to ~one group while
        # keeping the single up-front FP-teacher pass.
        precomp.pop(g, None)
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
        model = _sample_or_reuse(bench_evaluator, arch)
        for which in pending_bench:
            print(f"[correlation/eval] → benchmark '{which}'")
            try:
                results[which] = _run_benchmark_block(args, model, model_id,
                                                      which, arch=arch,
                                                      idx=args.idx)
                results.setdefault('_specs', {})[which] = _spec(which)
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
def _to_float(x):
    """Coerce a CSV cell / JSON scalar to float; return NaN on failure
    (strings, dicts, None, empty cell)."""
    if x is None or x == '':
        return float('nan')
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def _compute_correlations(rows, metric_cols, bench_cols, min_samples=5):
    """For each (metric, bench_subcol) pair, compute Pearson, Spearman, and
    Kendall tau-b over rows where BOTH values are numeric. Returns:
        {metric: {bench: dict(pearson, spearman, kendall, n)}}

    Pairs with fewer than `min_samples` overlapping numeric rows return NaN
    (correlation on n<5 is too noisy to interpret). Kendall tau-b is more
    robust to small-n and ties than Spearman; it's slower (O(n²)) but n=50
    is trivial.
    """
    from scipy.stats import spearmanr, kendalltau

    out = {m: {} for m in metric_cols}
    metric_vals = {m: np.array([_to_float(r.get(m)) for r in rows])
                   for m in metric_cols}
    bench_vals = {b: np.array([_to_float(r.get(b)) for r in rows])
                  for b in bench_cols}
    for m in metric_cols:
        mv = metric_vals[m]
        for b in bench_cols:
            bv = bench_vals[b]
            mask = ~(np.isnan(mv) | np.isnan(bv))
            n = int(mask.sum())
            if n < min_samples or np.unique(mv[mask]).size < 2 \
                    or np.unique(bv[mask]).size < 2:
                pr, sr, kt = float('nan'), float('nan'), float('nan')
            else:
                pr = float(np.corrcoef(mv[mask], bv[mask])[0, 1])
                sr = float(spearmanr(mv[mask], bv[mask]).correlation)
                kt = float(kendalltau(mv[mask], bv[mask]).correlation)
            out[m][b] = {'pearson': pr, 'spearman': sr,
                         'kendall': kt, 'n': n}
    return out


def _write_corr_matrix(path, corr, metric_cols, bench_cols, kind):
    """Write a wide CSV: rows=metrics, cols=bench sub-cols, cells=`kind` (pearson/spearman)."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric'] + bench_cols)
        for m in metric_cols:
            row = [m]
            for b in bench_cols:
                v = corr[m][b][kind]
                row.append(f"{v:.4f}" if not np.isnan(v) else '')
            w.writerow(row)


def _print_corr_summary(corr, metric_cols, bench_cols, top_k=3,
                        out_path=None, exclude=()):
    """Per benchmark column, list the top-K calibration metrics by |Pearson|
    plus their Spearman. Writes to stdout AND (if `out_path` given) to a
    text file for offline review. Same content goes to both sinks.

    `exclude` = (metric, bench) pairs whose correlation is circular (the metric
    reads documents the benchmark scores). They stay in the full matrices —
    the number is still a fact — but they are kept OUT of the ranking, which is
    where a contaminated pair would otherwise be read as "best proxy".
    """
    if not metric_cols or not bench_cols:
        return
    exclude = set(exclude)
    lines = []
    lines.append(f"Top-{top_k} calibration metrics per benchmark column "
                 f"(by |Pearson r|):")
    lines.append(f"  {'benchmark':<40}  {'rank':<4}  {'metric':<24}  "
                 f"{'pearson':>8}  {'spearman':>9}  {'kendall':>8}  {'n':>4}")
    lines.append(f"  {'-' * 40}  {'-' * 4}  {'-' * 24}  {'-' * 8}  "
                 f"{'-' * 9}  {'-' * 8}  {'-' * 4}")
    for b in bench_cols:
        # Rank metrics by |pearson| (NaN sorts last), minus the circular ones.
        rankable = [m for m in metric_cols if (m, b) not in exclude]
        dropped = len(metric_cols) - len(rankable)
        if dropped:
            lines.append(f"  {b:<40}  [{dropped} contaminated metric(s) "
                         f"excluded from this ranking]")
        ranked = sorted(
            rankable,
            key=lambda m: (np.isnan(corr[m][b]['pearson']),
                           -abs(corr[m][b]['pearson'])
                           if not np.isnan(corr[m][b]['pearson']) else 0.0))
        for i, m in enumerate(ranked[:top_k]):
            c = corr[m][b]
            pr_s = f"{c['pearson']:+.4f}" if not np.isnan(c['pearson']) else '   nan'
            sr_s = f"{c['spearman']:+.4f}" if not np.isnan(c['spearman']) else '    nan'
            kt_s = f"{c['kendall']:+.4f}" if not np.isnan(c['kendall']) else '   nan'
            bname = b if i == 0 else ''
            lines.append(f"  {bname:<40}  {i + 1:<4}  {m:<24}  "
                         f"{pr_s:>8}  {sr_s:>9}  {kt_s:>8}  {c['n']:>4}")

    print(f"\n[correlation/aggregate] " + lines[0])
    for ln in lines[1:]:
        print(ln)
    if out_path:
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"[correlation/aggregate] wrote {out_path}")


def cmd_aggregate(args):
    archs_csv = (args.archs_csv
                 or os.path.join(args.save or '.', 'archs.csv'))
    save_dir = args.save or os.path.dirname(archs_csv) or '.'
    out_dir = measurement_dir(args)
    if not os.path.exists(archs_csv):
        raise SystemExit(f"archs.csv not found at {archs_csv}")
    others = [d for d in sorted(glob.glob(os.path.join(save_dir, 'm_*')))
              if os.path.isdir(d) and os.path.normpath(d) != os.path.normpath(out_dir)]
    if others:
        print(f"[correlation/aggregate] reading ONE measurement config: "
              f"{out_dir}\n  other configs in this pool (not mixed in): "
              f"{[os.path.basename(d) for d in others]}"
              f"\n  pass --measure_dir to aggregate one of those instead.")
    with open(archs_csv) as f:
        archs_rows = list(csv.DictReader(f))
    arch_header = list(archs_rows[0].keys()) if archs_rows else []

    # Discover the column set across all result files
    rows, stale, spec_seen = [], [], {}
    for r in archs_rows:
        idx = int(r['idx'])
        result_path = os.path.join(out_dir, f'result_{idx}.json')
        merged = {k: r[k] for k in arch_header if k != 'arch_json'}
        if os.path.exists(result_path):
            with open(result_path) as f:
                res = json.load(f)
            # `idx` is a ROW NUMBER, not an identity: re-running --mode sample
            # with a different pool/seed rewrites archs.csv while the old
            # result_<idx>.json files stay, and every measurement would then be
            # silently attributed to whatever arch now sits at that row. The
            # results carry the arch they were measured on, so compare hashes.
            want = arch_sha8(json.loads(r['arch_json']))
            got = arch_sha8(res['arch']) if isinstance(res.get('arch'), dict) else None
            if got is not None and got != want:
                stale.append((idx, want, got))
                res = {}          # treat exactly like an unmeasured arch
            for mk, sh in (res.get('_specs') or {}).items():
                spec_seen.setdefault(mk, {}).setdefault(sh, []).append(idx)
            for mk in METRIC_KEYS:
                if mk in res:
                    v = res[mk]
                    merged[mk] = v if isinstance(v, (int, float)) else str(v)
            for bk in BENCH_KEYS:
                if bk in res and isinstance(res[bk], dict):
                    # Track all numeric leaf values that belong to this
                    # benchmark so we can derive `<bk>__avg` (overall score).
                    leaf_values = []
                    for sk, sv in res[bk].items():
                        # Skip wall-clock timings (ruler returns a top-level
                        # 'time' field alongside scores; it's not a benchmark
                        # score and its positive correlation with loss would
                        # contaminate ruler__avg).
                        if sk.startswith('_') or sk == 'time':
                            continue
                        merged[f'{bk}__{sk}'] = sv
                        if isinstance(sv, dict):
                            # LongBench-E returns {'0-4k':…, '4-8k':…, '8k+':…}
                            # → flatten to <bk>__<task>__<bucket> AND collect
                            # numeric leaves for the average.
                            for sub_k, sub_v in sv.items():
                                merged[f'{bk}__{sk}__{sub_k}'] = sub_v
                                if isinstance(sub_v, (int, float)):
                                    leaf_values.append(float(sub_v))
                        elif isinstance(sv, (int, float)):
                            leaf_values.append(float(sv))
                    if leaf_values:
                        merged[f'{bk}__avg'] = sum(leaf_values) / len(leaf_values)
        rows.append(merged)

    all_cols = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); all_cols.append(k)
    # per-metric definition drift WITHIN one config folder (old files written
    # before a registry edit): the folder key covers model/quant, not the
    # metric specs, so this is where a redefined metric would show up.
    drift = {}
    for m, seen in spec_seen.items():
        if len(seen) > 1:
            drift[m] = {k: sorted(v)[:4] for k, v in seen.items()}
    if drift:
        print(f"[correlation/aggregate] ⚠ {len(drift)} metric(s) have rows "
              f"measured under DIFFERENT definitions — re-run those idx "
              f"(--metrics <name>) before comparing: "
              + ', '.join(f'{m}: ' + ' vs '.join(f'{h}@idx{v}' for h, v in d.items())
                          for m, d in list(drift.items())[:5]))

    if stale:
        print(f"[correlation/aggregate] ⚠ DROPPED {len(stale)} result files "
              f"measured on a DIFFERENT arch than archs.csv now has at that row "
              f"(archs.csv was regenerated after they were written): "
              f"{[i for i, _, _ in stale[:10]]}"
              + (' …' if len(stale) > 10 else '')
              + ". Re-run those idx or restore the matching archs.csv.")

    out_csv = os.path.join(out_dir, 'correlation.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[correlation/aggregate] wrote {out_csv} "
          f"({len(rows)} rows, {len(all_cols)} cols)")

    # ── Correlation: each calibration metric ↔ each benchmark sub-column ──
    # metric_cols = the METRIC_KEYS that actually appear in the table.
    # bench_cols  = all `<bk>__<sk>` (and deeper) sub-columns that are numeric;
    #               the `<bk>__<sk>` parent column for LongBench-E carries the
    #               raw dict and is non-numeric, so it's filtered out below.
    metric_cols = [c for c in all_cols if c in METRIC_KEYS]
    bench_prefixes = tuple(f'{bk}__' for bk in BENCH_KEYS)
    bench_cols_all = [c for c in all_cols if c.startswith(bench_prefixes)]
    # Keep only numeric bench cols (skip stringified dicts etc.)
    bench_cols = [b for b in bench_cols_all
                  if any(not np.isnan(_to_float(r.get(b))) for r in rows)]
    if not metric_cols:
        print("[correlation/aggregate] no calibration metrics found — skipping correlation.")
    elif not bench_cols:
        print("[correlation/aggregate] no benchmark columns found — skipping correlation.")
    else:
        corr = _compute_correlations(rows, metric_cols, bench_cols,
                                     min_samples=args.corr_min_samples)
        pearson_path = os.path.join(out_dir, 'correlation_pearson.csv')
        spearman_path = os.path.join(out_dir, 'correlation_spearman.csv')
        kendall_path = os.path.join(out_dir, 'correlation_kendall.csv')
        _write_corr_matrix(pearson_path, corr, metric_cols, bench_cols, 'pearson')
        _write_corr_matrix(spearman_path, corr, metric_cols, bench_cols, 'spearman')
        _write_corr_matrix(kendall_path, corr, metric_cols, bench_cols, 'kendall')
        print(f"[correlation/aggregate] wrote {pearson_path}")
        print(f"[correlation/aggregate] wrote {spearman_path}")
        print(f"[correlation/aggregate] wrote {kendall_path}")
        # ── contamination: metric corpus ∩ benchmark corpus ──
        # REPORT-ONLY by default: overlap is recorded, never acted on. Pass
        # --corr_drop_contaminated to also keep those pairs out of the top-K
        # ranking (the full matrices always keep every number either way).
        bad = contaminated_pairs(metric_cols, bench_cols, args.needle_task)
        if bad:
            by_metric = {}
            for m, b in bad:
                by_metric.setdefault(m, []).append(b)
            cpath = os.path.join(out_dir, 'correlation_contamination.txt')
            with open(cpath, 'w') as f:
                f.write(
                    "(metric, benchmark) pairs measured over the SAME documents.\n\n"
                    "This is not an error: a long-context REPORTING metric is\n"
                    "supposed to sit next to benchmark accuracy on the same test\n"
                    "documents, and nothing in the pipeline is fit to that text\n"
                    "(search objective = wikitext2 JSD; post_search selects on the\n"
                    "archive's stored loss and measures these afterwards).\n"
                    "What the overlap costs is the PREDICTIVE reading of these\n"
                    "cells: the pair shares per-architecture, per-document noise,\n"
                    "so its correlation overstates how well the metric would\n"
                    "predict the benchmark on unseen documents. Every number is\n"
                    "in the correlation matrices; --corr_drop_contaminated also\n"
                    "keeps these pairs out of the top-K ranking.\n\n")
                for m, bs in sorted(by_metric.items()):
                    f.write(f"{m}\n" + ''.join(f"    {b}\n" for b in sorted(bs)))
            print(f"\n[correlation/aggregate] note: {len(bad)} (metric, "
                  f"benchmark) pairs share documents — {sorted(by_metric)} are "
                  f"measured over text the benchmark also grades. "
                  + (f"Excluded from the ranking (--corr_drop_contaminated); "
                     if args.corr_drop_contaminated else "Reported only; ")
                  + f"see {cpath}")
        summary_path = os.path.join(out_dir, 'correlation_summary.txt')
        _print_corr_summary(corr, metric_cols, bench_cols,
                            top_k=args.corr_top_k, out_path=summary_path,
                            exclude=bad if args.corr_drop_contaminated else ())


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
    p.add_argument('--attn_sink', type=int, default=0,
                   help='Keep first S KV tokens in FP (KVSink). 0=off. Match the search-time value.')
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
    # ε-band (near-front shell) instead of the strict per-axis Pareto front —
    # same envelope rule as second_search.py --front_eps_rel
    # (utils/second_stage.select_eps_band): keep metric <= front(comp)*(1+rel).
    # >0 implies front filtering (--expr_front not required). The per-axis pools
    # grow ~8x at rel=0.05, so the combo product grows ~64x — check the printed
    # n_total before adding a comp_obj-free wide band.
    p.add_argument('--front_eps_rel', type=float, default=0.0,
                   help='(sample) relative ε-band around each per-axis front: '
                        'keep archs with metric <= front(comp)*(1+rel). '
                        'Scale-free (wider in the high-loss corner). '
                        '0 = strict front (--expr_front), unchanged behaviour.')
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
    # ── factorial grid sampling (paired W×KV design for interaction ANOVA) ──
    p.add_argument('--grid_sample', action='store_true',
                   help='(sample) paired factorial design: draw --grid_n '
                        'blocks per expr axis from its Pareto front and emit '
                        'the FULL cartesian product (axis-major rows; '
                        'per-axis block ids in grid_<axis> columns). '
                        'Requires --expr_front; ignores --n_archs, '
                        '--quantile_sample and --sampling_method. '
                        'Single-axis --comp_obj bounds filter their own '
                        'axis pool.')
    p.add_argument('--grid_n', type=int, nargs='+', default=[12],
                   help='(sample, --grid_sample) blocks per axis: one value '
                        'broadcast to all expr axes, or one per axis in '
                        'expr_keys order (e.g. --grid_n 12 12 for w × '
                        'eff_kv).')
    p.add_argument('--grid_stratify', action='store_true',
                   help='(sample, --grid_sample) comp-stratified pick '
                        'instead of pure random: sort the axis front by its '
                        'own comp, split into grid_n quantile bins, one '
                        'random member per bin; endpoint bins are pinned to '
                        'the exact min/max-comp members (budget corners '
                        'always in the design). Guards against fronts that '
                        'are dense in one comp region (recommended for '
                        'ANOVA).')
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
    p.add_argument('--measure_dir', type=str, default='',
                   help='(eval / aggregate) measurement directory. Default: '
                        '<save>/m_<config sha8> — one folder per model+quant '
                        'config, so rows in one folder are comparable BY '
                        'CONSTRUCTION (archs.csv stays shared at <save>). A save '
                        'dir that already holds result_*.json at its root keeps '
                        'using the root. Pass this to aggregate a specific '
                        'config.')
    p.add_argument('--corr_min_samples', type=int, default=5,
                   help='(aggregate) min overlapping numeric samples needed '
                        'to compute a correlation; below this the cell is NaN.')
    p.add_argument('--corr_drop_contaminated', action='store_true',
                   help='(aggregate) keep (metric, benchmark) pairs that share '
                        'documents OUT of the top-K ranking. Off by default: '
                        'the overlap is written to '
                        'correlation_contamination.txt and every number stays '
                        'in the correlation matrices.')
    p.add_argument('--corr_top_k', type=int, default=3,
                   help='(aggregate) per benchmark column, print top-K '
                        'calibration metrics by |Pearson r|.')
    p.add_argument('--idx', type=int, default=-1,
                   help='(eval) row index in archs.csv to evaluate')
    p.add_argument('--metrics', type=str, nargs='+', default=['all'],
                   help='(eval) calibration metrics (PPL/loss) to evaluate. '
                        '"all" (default) = all METRIC_KEYS. Explicitly listed '
                        'keys are force-rerun; "none" runs benchmarks only. '
                        'Benchmarks are NOT accepted '
                        'here — use --ruler / --longbench / --longbench_e. '
                        f'Valid: {METRIC_KEYS} (or "all").')
    # Benchmark toggles: each flag opts the corresponding benchmark in.
    # Listing the flag implies force-rerun (overwrite any existing entry).
    p.add_argument('--ruler', action='store_true',
                   help='(eval) run RULER benchmark.')
    p.add_argument('--longbench', action='store_true',
                   help='(eval) run LongBench benchmark.')
    p.add_argument('--longbench_e', action='store_true',
                   help='(eval) run LongBench-E benchmark.')
    p.add_argument('--force', action='store_true',
                   help='(eval) recompute even if already in result_<idx>.json')
    p.add_argument('--no_archive', action='store_true',
                   help='(eval) skip archiving when an existing metric/'
                        'benchmark entry is about to be overwritten. By '
                        'default, result_<idx>.json + raw bench artefacts '
                        '(ruler_*.json, longbench_*/) are moved under '
                        '${SAVE}/archive/<timestamp>/ before re-running, so '
                        'previous results are recoverable.')
    # gov_jsd_kt key-token archive (consumed in eval mode only)
    p.add_argument('--key_token_path', type=str, default='key_token',
                   help='DIRECTORY the key-token archives live in. The root is '
                        'derived per metric: <dir>/kt_eval-<evaluator>_tgt-<target>'
                        '_<layout>, with the evaluator taken from the metric name '
                        '(..._q72b / ..._l8b)')
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
    p.add_argument('--ruler_length', type=int, nargs='+', default=[16384],
                   help='(eval) RULER context length(s). Independent of '
                        '--n_token (which is the memory-accounting token '
                        'count). Several values run one FULL --ruler_sample '
                        'sweep per length (NOT a mixed sample): scores are '
                        'then reported as <task>_len<L> and the raw artefacts '
                        'land in <ruler_result_path>/len<L>/scores.json. A '
                        'single value keeps the flat <task> layout.')
    p.add_argument('--ruler_yaml_path', type=str, default='utils/ruler_utils',
                   help='dir holding ruler yaml configs AND cached qa JSON '
                        '(hotpot_dev_distractor_v1.json, dev-v2.0.json)')
    p.add_argument('--ruler_sample', type=int, default=50)
    p.add_argument('--ruler_gen_toks', type=int, default=None)
    p.add_argument('--ruler_batch_size', type=int, default=1)
    p.add_argument('--ruler_result_path', type=str, default='')
    p.add_argument('--topk_logits', type=int, default=5,
                   help='(eval) per generated RULER token, store the top-K '
                        'next-token candidates (raw logits → logprobs) plus the '
                        'chosen token logprob and the top1-top2 margin in the '
                        'per-example JSONL. 0 = off. Cheap here (answers are a '
                        'few tokens); it is what turns "wrong answer" into "how '
                        'wrong".')
    p.add_argument('--longbench_topk_logits', type=int, default=0,
                   help='(eval) top-K record for LongBench generations. OFF by '
                        'default because this harness sweeps MANY idx: measured '
                        '34.5 MB per pass (2200 examples, 225k generated '
                        'tokens), so a 200-idx sweep would add ~6.9 GB. Set 5 '
                        'for the handful of idx you want to inspect. RULER is '
                        'on by default at 0.7 MB per sweep.')
    p.add_argument('--ruler_per_example_path', type=str, default='',
                   help='(eval) JSONL path for sample-level RULER generations, '
                        'references, scores and prompt hashes. Default: '
                        '<ruler_result_path>[/len<L>]/per_example_s<seed>.jsonl '
                        '— always written (a few hundred rows), and keyed by '
                        'seed because the seed decides which samples exist.')
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
