"""utils/metric_specs.py — the ONE registry of named calibration metrics.

A metric name (e.g. ``wt2_jsd_pp512_s32``, ``gov_jsd_pp128_s32``, ``c4_ppl``)
fully determines what gets measured: which dataset, how the calibration data is
built, and how the forward pass is run. correlation.py and post_search.py both
resolve names from here, so a number reported by one is the SAME measurement as
the number reported by the other.

Two levels:

* **GROUP** — the data side: ``datasets / n_sample / seqlen / min_seqlen /
  last_tokens / use_key_token``. These decide the FP-teacher ``dense_logits``,
  which ``get_logits`` pre-masks to ``last_tokens``, so changing ANY of them
  requires another teacher pass. Tasks sharing a group share one pass.
* **TASK** — the forward side: ``metric / loss_func / stride / prefill_prompt /
  last_tokens``. Tasks in the same group differ only here and cost one extra
  forward each, no extra teacher pass.

Helpers: :func:`resolve_tasks` (names → task tuples), :func:`groups_for`
(tasks → the group specs to build), :func:`precompute_groups` (ONE FP-teacher
pass for every group, parked on CPU) and :func:`apply_group` (point an existing
evaluator's data side at a group — no model rebuild).
"""
import os
from time import time

# NOTE: nothing heavy at module level — the registry is plain dicts/lists, so
# importing it costs ~0.04s. torch / utils.data / utils.eval (datasets,
# transformers, lm_eval) cost ~4.4s and are only needed by the runtime helpers,
# which import them when CALLED. (The SAVE-dir tag is built in the shell from
# the task name or the knobs; see scripts/metric_tag.sh — no lookup needed.)


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
    'A_lt128': dict(  # wikitext2 — last-128-token JSD (dense_logits masked to
        # the last 128 positions). Serves both wt2_jsd_lt128 (single-pass) and
        # wt2_jsd_pp128_s32 (answer-phase prefill+stride) — the FP-teacher
        # dense_logits depend only on last_tokens, not the forward strategy.
        # quant_kv_output=True is implicit for the single-pass path
        # (stride=0, prefill_prompt=False) sets use_cache=False, which in
        # turn flips configure_model_cache to quant_kv_output=True; the
        # answer-phase metric re-configures the cache per-call (use_cache=True).
        datasets=['wikitext2'], n_sample=128, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=False, last_tokens=128,
        trunc_len=512, sliding_window=128, alpha=2, beta=-2,
    ),
    'B_pp': dict(  # gov_report — answer-phase JSD with prefill_prompt
        # last_tokens=512 makes dense_logits tiny (~1 GB) so the standard
        # eval_loss path fits without stream_dense gymnastics.
        datasets=['gov_report'], n_sample=8, seqlen=8192, min_seqlen=8192,
        loss_func='jsd', use_key_token=False, last_tokens=512,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'B_lt128': dict(  # gov_report — last-128-token JSD (shared by the
        # single-pass gov_jsd_lt128 and answer-phase gov_jsd_pp128_s32).
        datasets=['gov_report'], n_sample=8, seqlen=8192, min_seqlen=8192,
        loss_func='jsd', use_key_token=False, last_tokens=128,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'D': dict(  # gsm8k — short answer-only loss, JSD
        datasets=['gsm8k'], n_sample=8, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=False, last_tokens=None,
        trunc_len=512, sliding_window=128, alpha=2, beta=-2,
    ),
    'B': dict(  # gov_report long, no key-token
        datasets=['gov_report'], n_sample=8, seqlen=8192, min_seqlen=8192,
        loss_func='jsd', use_key_token=False, last_tokens=None,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'C': dict(  # gov_report long, with key-token
        datasets=['gov_report'], n_sample=8, seqlen=8192, min_seqlen=8192,
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
    ('c4_ppl_pp512_s128', 'A', 'c4',
        # Answer-phase PPL: prefill prompt + stride answer (s128) over the
        # last 512 tokens. eval_ppl now supports last_tokens / prefill_prompt;
        # dense_logits is unused for PPL so Group A's last_tokens=None is fine.
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('c4_ppl_pp128_s32',  'A', 'c4',
        # Same answer-phase PPL with a shorter last_tokens=128 window and
        # finer stride=32 (4× the chunks of s128).
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('wt2_ppl',           'A', 'wikitext2',
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('wt2_ppl_pp512_s128', 'A', 'wikitext2',
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('wt2_ppl_pp128_s32', 'A', 'wikitext2',
        # last_tokens=128 answer window, finer stride=32 (PPL → no dense_logits,
        # so Group A's last_tokens=None is fine).
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('wt2_jsd',           'A', 'wikitext2',
        dict(metric='loss', loss_func='jsd',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('wt2_jsd_s512',      'A', 'wikitext2',
        dict(metric='loss', loss_func='jsd',
             stride=512, prefill_prompt=False, last_tokens=None)),
    ('wt2_jsd_pp512_s128', 'A_pp', 'wikitext2',
        dict(metric='loss', loss_func='jsd',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('wt2_jsd_pp512_s32', 'A_pp', 'wikitext2',
        # Same Group A_pp + answer-phase mask, finer stride=32 for denser
        # answer-token coverage (4× the chunks of s128 → ~4× eval time).
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=512)),
    ('wt2_jsd_pp128_s32', 'A_lt128', 'wikitext2',
        # Answer-phase JSD on the last 128 tokens (prefill_prompt + stride=32).
        # Group A_lt128 (last_tokens=128) supplies the matching pre-masked
        # dense_logits — it is arch- and forward-strategy-independent, so it is
        # shared with wt2_jsd_lt128 (no extra FP-teacher pass).
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('wt2_jsd_lt128',     'A_lt128', 'wikitext2',
        # single-pass JSD on last 128 tokens. last_tokens is set at evaluator
        # init (Group A_lt128) and matches the eval_loss mask.
        dict(metric='loss', loss_func='jsd',
             stride=0, prefill_prompt=False, last_tokens=128)),
    ('needle_nll',        'A', None,
        dict(kind='needle_nll',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('needle_nll_s512',   'A', None,
        # chunked forward (use_cache=True path); answer-tokens loss unchanged.
        dict(kind='needle_nll',
             stride=512, prefill_prompt=False, last_tokens=None)),
    ('needle_nll_pp512_s128', 'A', None,
        # prefill prompt then stride answer in 128-chunks. last_tokens=512
        # bounds the answer span; label=-100 already restricts loss to the
        # actual answer tokens (which lie at the very end of the prompt).
        dict(kind='needle_nll',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('needle_nll_pp512_s32', 'A', None,
        # Same as needle_nll_pp512_s128 with finer stride=32 over answer span.
        dict(kind='needle_nll',
             stride=32, prefill_prompt=True, last_tokens=512)),
    ('needle_nll_pp128_s32', 'A', None,
        # Shorter last_tokens=128 answer window, finer stride=32.
        dict(kind='needle_nll',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('needle_jsd_pp512_s128', 'A', None,
        # JSD variant: FP teacher dense_logits cached per-process + on disk
        # (needle prompts are seed-deterministic). Higher SNR than CE since
        # it compares the full output distribution at answer positions.
        dict(kind='needle_jsd',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('needle_jsd_pp512_s32', 'A', None,
        dict(kind='needle_jsd',
             stride=32, prefill_prompt=True, last_tokens=512)),
    ('needle_jsd_pp128_s32', 'A', None,
        # Shorter last_tokens=128 answer window, finer stride=32. The FP-teacher
        # dense_logits are cached separately (keyed by _lt128), so this does not
        # collide with the lt512 needle_jsd cache.
        dict(kind='needle_jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
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
    ('gov_jsd_pp512_s32', 'B_pp', 'gov_report',
        # Same Group B_pp, finer stride=32 over the 512-token answer span.
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=512)),
    ('gov_jsd_pp128_s32', 'B_lt128', 'gov_report',
        # Answer-phase JSD on the last 128 tokens (prefill_prompt + stride=32);
        # Group B_lt128 (last_tokens=128) supplies the matching pre-masked
        # dense_logits (shared with gov_jsd_lt128).
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('gov_jsd_lt128',     'B_lt128', 'gov_report',
        # single-pass JSD on last 128 tokens. Group B_lt128's dense_logits
        # is also trimmed to last 128 so the mask matches.
        dict(metric='loss', loss_func='jsd',
             stride=0, prefill_prompt=False, last_tokens=128)),
    ('gsm8k_jsd_pp_s128', 'D', 'gsm8k',
        # gsm8k unpadded → answer is at the end → prefill question, stride
        # answer in 128-chunks. last_tokens=512 caps the answer span; the
        # label=-100 mask filters out any question tokens that fall inside
        # the last 512 window. Dense_logits is reused from Group D (padded
        # forward gives identical FP logits at the answer positions because
        # causal attention only sees past tokens).
        dict(kind='gsm8k_unpad_pp',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('gsm8k_jsd_pp_s32', 'D', 'gsm8k',
        # Same Group D + unpadded path, finer stride=32 over the answer
        # span (4× chunks of s128). Same dense_logits reuse.
        dict(kind='gsm8k_unpad_pp',
             stride=32, prefill_prompt=True, last_tokens=512)),
    ('gov_jsd_kt',        'C', 'gov_report',
        dict(metric='loss', loss_func='jsd',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('gov_jsd_kt_s512',   'C', 'gov_report',
        # stride=512 chunked forward over the same key-token archive as
        # gov_jsd_kt — bounds peak activation memory at 8K context while
        # keeping the key-token weighting identical.
        dict(metric='loss', loss_func='jsd',
             stride=512, prefill_prompt=False, last_tokens=None)),
]
METRIC_KEYS = [t[0] for t in METRIC_TASKS]
BENCH_KEYS = ['longbench', 'longbench_e', 'ruler']
ALL_KEYS = METRIC_KEYS + BENCH_KEYS


TASKS_BY_NAME = {t[0]: t for t in METRIC_TASKS}


def resolve_tasks(names):
    """Metric names → their (key, group, dataset, eval_kwargs) tuples, in the
    order given, de-duplicated. Unknown names abort with the valid list."""
    out, seen = [], set()
    for n in names:
        for tok in (p for p in str(n).replace(',', ' ').split() if p):
            if tok in seen:
                continue
            if tok not in TASKS_BY_NAME:
                raise SystemExit(
                    f"unknown metric task '{tok}'. Valid names:\n  "
                    + '\n  '.join(METRIC_KEYS))
            seen.add(tok)
            out.append(TASKS_BY_NAME[tok])
    return out


def groups_for(tasks, key_token_path=''):
    """The group specs `tasks` need, as [(group_name, spec)] in first-use order.
    Group 'C' is the key-token group and needs --key_token_path."""
    out, seen = [], set()
    for _key, g, _ds, _kw in tasks:
        if g in seen:
            continue
        seen.add(g)
        spec = dict(GROUPS[g])
        if spec.get('use_key_token'):
            if not key_token_path:
                raise SystemExit(
                    f"metric group '{g}' needs key tokens → pass --key_token_path.")
            spec['key_token_path'] = key_token_path
        else:
            spec['key_token_path'] = ''
        out.append((g, spec))
    return out


DIVERGENCE_LOSSES = ('jsd', 'kld', 'topk', 'forward_kl')


def needs_dense(eval_kwargs):
    """Does this task consume FP-teacher logits? Custom-loader tasks (`kind`)
    are assumed to (they read the group's dense through their own path)."""
    if eval_kwargs.get('kind'):
        return True
    return (eval_kwargs.get('metric') != 'ppl'
            and eval_kwargs.get('loss_func') in DIVERGENCE_LOSSES)


def precompute_groups(accelerator, model_id, group_items, *, seed=0, dtype='auto',
                      device_map='auto', store_device='cpu', batch_size=1,
                      tasks=None):
    """ONE FP-teacher pass that builds every group's data side.

    dense_logits do NOT depend on the arch, so a single pass serves every group
    AND every architecture that is measured afterwards; the teacher is freed
    before the caller builds its quantized model, so the two never sit on the
    GPU together. With store_device='cpu' (default) the logits are parked off-GPU
    as they are produced — eval_loss uploads one sequence at a time.

    The SAME loader objects used for get_logits are returned, so dense_logits[i]
    aligns with eval_loss's loader[i] exactly.

    `tasks` (optional) restricts the teacher pass to the (group, dataset) pairs
    the requested tasks actually consume. Without it every dataset of every
    divergence group is built — which for a PPL-only request on a jsd group means
    a full-sequence dense set nobody reads (wikitext2 n128 @2048 = 67 GB).

    Returns {group: dict(train_loaders, test_loaders, dense_logits,
                         key_token_list, spec)}.
    """
    from utils.data import get_loader
    from utils.eval import get_logits, get_tokenizer
    from utils.func import clean_up, get_hfmodel
    from utils.loss import get_key_token_list

    wanted = None
    if tasks is not None:
        wanted = {(g, ds) for _k, g, ds, kw in tasks if needs_dense(kw)}
    out, pending = {}, []
    for g, spec in group_items:
        def _loaders(train):
            return {d: accelerator.prepare(get_loader(
                        d, model=model_id, n_sample=spec['n_sample'],
                        batch_size=batch_size, train=train, seed=seed,
                        seqlen=spec['seqlen'], min_seqlen=spec['min_seqlen']))
                    for d in spec['datasets']}
        is_div = spec['loss_func'] in DIVERGENCE_LOSSES
        dense_ds = [d for d in spec['datasets']
                    if is_div and (wanted is None or (g, d) in wanted)]
        out[g] = dict(train_loaders=_loaders(True), test_loaders=_loaders(False),
                      dense_logits={d: None for d in dense_ds},
                      key_token_list={d: None for d in spec['datasets']}
                      if spec['use_key_token'] else {},
                      spec=spec)
        if dense_ds or spec['use_key_token']:
            pending.append((g, spec))

    if not pending:
        return out

    print(f"[metric_specs] one FP-teacher pass for group(s) "
          f"{[g for g, _ in pending]} (dense_logits are arch-independent) …")
    t0 = time()
    fp = get_hfmodel(model_id, dtype=dtype, device_map=device_map)
    fp.eval()
    tok = get_tokenizer(model_id, use_fast=True)
    for g, spec in pending:
        r = out[g]
        if spec['use_key_token']:
            r['key_token_list'] = {
                d: get_key_token_list(
                    evaluator_model=fp, evaluator_tokenizer=tok, loader=loader,
                    trunc_len=spec['trunc_len'], sliding_window=spec['sliding_window'],
                    alpha=spec['alpha'], beta=spec['beta'],
                    load_path=os.path.join(spec.get('key_token_path', ''), d),
                    mode='offline')
                for d, loader in r['train_loaders'].items()}
        if not r['dense_logits']:
            continue
        for d in list(r['dense_logits']):
            loader = r['train_loaders'][d]
            r['dense_logits'][d] = get_logits(
                fp, loader,
                key_token_list=r['key_token_list'].get(d) if spec['use_key_token'] else None,
                last_tokens=spec['last_tokens'], store_device=store_device)
            clean_up()
            print(f"[metric_specs] {g}/{d}: dense_logits ready "
                  f"({len(r['dense_logits'][d])} batches, on {store_device or 'gpu'})")
    del fp
    clean_up()
    print(f"[metric_specs] FP-teacher pass done ({time() - t0:.1f}s)")
    return out


def apply_group(evaluator, payload):
    """Point an EXISTING evaluator's data side at one precomputed group.

    Swaps only what a group owns (loaders / teacher logits / key tokens / the
    answer window + seqlen the loss side reads). The quantized model, KV config
    and everything else are untouched, so N groups cost N forward passes and
    ZERO model rebuilds. `LlamaEvaluator.__init__` is not involved.
    """
    spec = payload['spec']
    evaluator.train_loaders = payload['train_loaders']
    evaluator.test_loaders = payload['test_loaders']
    evaluator.dense_logits = payload['dense_logits']
    evaluator.key_token_list = payload['key_token_list']
    evaluator.use_key_token = bool(spec['use_key_token'])
    evaluator.last_tokens = spec['last_tokens']
    evaluator.loss_func = spec['loss_func']
    evaluator.seqlen = evaluator.loss_seqlen = evaluator.ppl_seqlen = spec['seqlen']
    return evaluator


def move_dense_to_cpu(evaluator):
    """Replace GPU-resident dense_logits with CPU tensors behind a LazyGpuList.
    For data that was ALREADY built on the GPU; new code should pass
    ``get_logits(store_device='cpu')`` instead."""
    import torch
    from utils.eval import LazyGpuList
    from utils.func import clean_up

    target = evaluator.model.device if evaluator.model is not None else 'cuda'
    n = 0
    for dataset, batches in list(evaluator.dense_logits.items()):
        if batches is None or isinstance(batches, LazyGpuList):
            continue
        cpu = [[t.detach().to('cpu', copy=False) for t in b] for b in batches]
        evaluator.dense_logits[dataset] = LazyGpuList(cpu, target)
        n += sum(len(b) for b in cpu)
    clean_up()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"[dense_logits→cpu] moved {n} per-seq tensors to CPU, "
              f"GPU free={free/1e9:.2f}GB / {total/1e9:.2f}GB")


def run_task(args, accelerator, evaluator, dataset, eval_kwargs):
    """Measure ONE task on an already-prepared evaluator + model.

    Bypasses LlamaEvaluator.eval() (which loops over every loader on that side)
    so a task can pick its ONE dataset and run with its own stride /
    prefill_prompt / last_tokens. The evaluator's data side must already be the
    task's group (see apply_group). `args` supplies residual_length / kv_method
    for the cache configuration only.
    """
    from utils.eval import eval_metric
    from utils.func import configure_model_cache

    model = evaluator.model
    use_cache = (eval_kwargs.get('stride') or 0) > 0 or eval_kwargs.get('prefill_prompt')
    configure_model_cache(args, model, use_cache=use_cache)

    loader = (evaluator.test_loaders[dataset] if eval_kwargs['metric'] == 'ppl'
              else evaluator.train_loaders[dataset])
    dense_logits = (evaluator.dense_logits.get(dataset)
                    if eval_kwargs.get('loss_func') in ('jsd', 'kld', 'topk', 'forward_kl')
                    else None)
    key_token_list = (evaluator.key_token_list.get(dataset)
                      if evaluator.use_key_token else None)
    return eval_metric(
        model=model, accelerator=accelerator,
        metric=eval_kwargs['metric'], loader=loader, seqlen=evaluator.seqlen,
        loss_func=eval_kwargs.get('loss_func', 'cross_entropy'),
        dense_logits_list=dense_logits, key_token_list=key_token_list,
        stride=eval_kwargs.get('stride') or 0,
        last_tokens=eval_kwargs.get('last_tokens'),
        prefill_prompt=bool(eval_kwargs.get('prefill_prompt')),
        tokenizer=evaluator.tokenizer)


# ── measurement protocol, for embedding in an archive ───────────────────────
# The SAVE-dir tag (scripts/metric_tag.sh) is a short IDENTITY label — dir names
# sit near the 255-byte limit, so the numbers can't live there. results.txt does
# hold every arg, but only for a run that finished. iter_<it>.stats is the file
# that actually travels to second_search / post_search, so the protocol goes in
# there: an archive should be self-describing about how its loss was measured.
PROTOCOL_KEYS = ('dataset', 'datasets', 'n_sample', 'seqlen', 'min_seqlen',
                 'data_batch_size', 'metric', 'loss_func', 'stride',
                 'prefill_prompt', 'last_tokens', 'use_key_token',
                 'attn_sink', 'residual_length')


def protocol_dict(args):
    """Measurement-protocol subset of `args` (a dict or a Namespace), skipping
    keys the caller doesn't have."""
    get = args.get if hasattr(args, 'get') else (lambda k, d=None: getattr(args, k, d))
    return {k: get(k) for k in PROTOCOL_KEYS if get(k) is not None}
