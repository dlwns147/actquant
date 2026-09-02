"""utils/metric_specs.py — the ONE registry of named calibration metrics.

A metric name (e.g. ``wt2_jsd_pp512_s32``, ``gov_jsd_pp128_s32``, ``c4_ppl``)
fully determines what gets measured: which dataset, how the calibration data is
built, and how the forward pass is run. correlation.py and post_search.py both
resolve names from here, so a number reported by one is the SAME measurement as
the number reported by the other.

Two levels:

* **GROUP** — the data side: ``datasets / n_sample / seqlen / min_seqlen /
  last_tokens / use_key_token`` (+ optional ``sides``, which drops the loader
  side a group never reads — PPL-only groups skip the train build).
  These decide the FP-teacher ``dense_logits``,
  which ``get_logits`` pre-masks to ``last_tokens``, so changing ANY of them
  requires another teacher pass. Tasks sharing a group share one pass.
* **TASK** — the forward side: ``metric / loss_func / stride / prefill_prompt /
  last_tokens``. Tasks in the same group differ only here and cost one extra
  forward each, no extra teacher pass.

``last_tokens`` is the ANSWER WINDOW: the prefill/answer split point, where the
``chat:`` loader puts the assistant header (``seqlen - last_tokens``), and — under
the default ``score='last'`` — the positions that are scored. ``score='full'``
keeps the split and scores everything instead; that is the shape key-token
metrics need, because key tokens are sparse and an AND with a small window often
scores nothing. So a ``_pp<N>_s<M>`` name means: prefill up to ``seqlen - N``,
push the last ``N`` through the cache in ``M``-token chunks, score those ``N``
positions — unless the metric sets ``score='full'``, which scores every position
the label/key masks allow.

Helpers: :func:`resolve_tasks` (names → task tuples), :func:`groups_for`
(tasks → the group specs to build), :func:`precompute_groups` (ONE FP-teacher
pass for every group, parked on CPU) and :func:`apply_group` (point an existing
evaluator's data side at a group — no model rebuild).
"""
import os
import json
import hashlib
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

# ── Long-document corpora (gov_report, longbench:*) — FIXED evaluation set ──
# Unlike wikitext2/c4, whose test loaders deterministically window a fixed token
# stream, a long-document corpus SELECTS documents: shuffle(seed) → keep the
# first n_sample that clear the window. Both knobs are pinned here so the
# document set is a property of the METRIC NAME, not of the run:
#   * LONG_DOC_N_SAMPLE — the default sample count for long-doc groups.
#   * LONG_DOC_DATA_SEED — group-owned `data_seed`, applied to every long-doc
#     group below (see the loop after GROUPS) and used by precompute_groups
#     INSTEAD of the run's --seed for document selection only. Every script
#     already runs SEED=0, so this changes no existing number; it stops a
#     --seed sweep (which is about arch sampling / needle prompts) from
#     silently swapping the evaluation documents under a fixed metric name.
LONG_DOC_N_SAMPLE = 128
LONG_DOC_DATA_SEED = 0

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
    # KEEP IN MIND when reading pp32_s8: under residual_length=128 the whole
    # 32-token window is inside the FP residual, so none of the SCORED tokens'
    # own KV is quantised -- it probes the quantised PREFIX, not the
    # KV-accumulates-as-you-decode behaviour the name suggests. MEASURED scored
    # positions: wt2 128 (wikitext2 yields 4 windows, not 128 -- the loader
    # joins 128 text ROWS then reslices) and gov 256, against 2048/4096 for the
    # pp512_s128 pair, which is also the only window wider than the residual
    # (384 of its 512 positions carry quantised KV). It is the tightest
    # decode-shaped probe available; just do not read it as the most realistic.
    'A_lt32': dict(  # wikitext2 — last-32-token answer window (pp32_s8).
        # Prefill 2016 tokens, then score 32 in 8-token chunks (4 cache-appending
        # steps). dense_logits are tiny (32 positions x vocab fp16).
        datasets=['wikitext2'], n_sample=128, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=False, last_tokens=32,
        trunc_len=512, sliding_window=128, alpha=2, beta=-2,
    ),
    'B_lt32': dict(  # gov_report — last-32-token answer window (pp32_s8).
        # 8k prefill + 32 scored tokens in 8-token chunks; same shape as A_lt32.
        datasets=['gov_report'], n_sample=8, seqlen=8192, min_seqlen=8192,
        loss_func='jsd', use_key_token=False, last_tokens=32,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
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
    # gov_jsd_pp128_s32 n_sample × seqlen sweep — same last-128 answer window as
    # B_lt128, but n_sample (8→4) and seqlen (8192→4096→2048) vary for a cost /
    # context-length ablation. n_sample + seqlen + min_seqlen are group-owned
    # (they set the loaders + FP-teacher dense_logits), so each combo needs its
    # own group. min_seqlen tracks seqlen (standard shorter-context gov_report);
    # dense_logits stay tiny (n_sample × 128 × vocab fp16) thanks to last_tokens=128.
    'B_lt128_n8_sl4096': dict(  # gov_report — n8, seqlen 4096
        datasets=['gov_report'], n_sample=8, seqlen=4096, min_seqlen=4096,
        loss_func='jsd', use_key_token=False, last_tokens=128,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'B_lt128_n8_sl2048': dict(  # gov_report — n8, seqlen 2048
        datasets=['gov_report'], n_sample=8, seqlen=2048, min_seqlen=2048,
        loss_func='jsd', use_key_token=False, last_tokens=128,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'B_lt128_n4_sl8192': dict(  # gov_report — n4, seqlen 8192
        datasets=['gov_report'], n_sample=4, seqlen=8192, min_seqlen=8192,
        loss_func='jsd', use_key_token=False, last_tokens=128,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'B_lt128_n4_sl4096': dict(  # gov_report — n4, seqlen 4096
        datasets=['gov_report'], n_sample=4, seqlen=4096, min_seqlen=4096,
        loss_func='jsd', use_key_token=False, last_tokens=128,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'B_lt128_n4_sl2048': dict(  # gov_report — n4, seqlen 2048
        datasets=['gov_report'], n_sample=4, seqlen=2048, min_seqlen=2048,
        loss_func='jsd', use_key_token=False, last_tokens=128,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'B_lt128_n128_sl8192': dict(  # gov_report — n128 (LONG_DOC_N_SAMPLE), 8192
        # The long-doc JSD group at the 128-document default. Only the
        # last_tokens=128 window can afford it: dense_logits scale with
        # n_sample, so at n=128 this group stores 128 x 128 x vocab fp16 =
        # 4.2 GB, while the full-sequence groups B / C would need 268 GB and
        # B_pp (last 512) 16.8 GB. That is why the existing gov_jsd* groups
        # stay at n=8 — and why raising THEM would silently redefine numbers
        # already measured under those names; this is a separate name instead.
        datasets=['gov_report'], n_sample=LONG_DOC_N_SAMPLE,
        seqlen=8192, min_seqlen=8192,
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
        key_token_suffix='_raw',
    ),
    # ── chat-templated twins (`chat:<corpus>`) ────────────────────────────
    # Same corpus, same window, but each sample is wrapped in ONE user turn with
    # the assistant header at seqlen - last_tokens, so the scored tail is
    # produced in ASSISTANT position -- the format deployment actually feeds an
    # Instruct model. seqlen/min_seqlen stay TOTAL; the document budget shrinks
    # by the per-model affix overhead (Llama-3.1 35 tokens, Qwen2.5 29,
    # Gemma-3 9, Mistral-v0.3 4), so these are NOT the same token spans as the
    # raw twins -- pair them by name, not by assuming identical text.
    # sides=('train',): the chat loader is loss-side only (PPL over template
    # boilerplate is not a metric), so the test build is skipped entirely.
    'A_lt128_chat': dict(
        datasets=['chat:wikitext2'], n_sample=128, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=False, last_tokens=128, sides=('train',),
        trunc_len=512, sliding_window=128, alpha=2, beta=-2,
    ),
    'A_pp_chat': dict(
        datasets=['chat:wikitext2'], n_sample=128, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=False, last_tokens=512, sides=('train',),
        trunc_len=512, sliding_window=128, alpha=2, beta=-2,
    ),
    'B_lt128_chat': dict(
        datasets=['chat:gov_report'], n_sample=8, seqlen=8192, min_seqlen=8192,
        loss_func='jsd', use_key_token=False, last_tokens=128, sides=('train',),
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'B_pp_chat': dict(
        datasets=['chat:gov_report'], n_sample=8, seqlen=8192, min_seqlen=8192,
        loss_func='jsd', use_key_token=False, last_tokens=512, sides=('train',),
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'C_pp': dict(  # gov_report long, key-token, 512-token answer window
        # gov_jsd_pp512_s128's key-token twin (group B_pp + the flag).
        datasets=['gov_report'], n_sample=8, seqlen=8192, min_seqlen=8192,
        loss_func='jsd', use_key_token=True, last_tokens=512,
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
        key_token_suffix='_raw',
    ),
    'C_wt2_pp': dict(  # wikitext2, key-token, 512-token answer window
        # The FIRST key-token group on a non-gov corpus, so it needs its own
        # archive. trunc_len/sliding_window are gov's RATIOS, not gov's numbers:
        # trunc_len is the "no long context" baseline, so what matters is its
        # share of the window. gov is 256/8192 = 3.1% with 124 short forwards per
        # document; at 2048 that is 64/16 -- same share, same window count, same
        # cost per document.
        # MEASURED (Llama-3.1-8B self-evaluated, 4 docs x 2048), hit rate =
        # intervals / candidate positions:
        #     512/128  ->   32 intervals  0.52%   (the original, far too coarse)
        #     256/64   ->   56            0.78%
        #      64/16   ->  166            2.09%   <- gov's 2.38%
        #      32/8    ->  326            4.04%   (looser than gov's ratio: a
        #                                          32-token baseline calls any
        #                                          token needing ~50 tokens key)
        # alpha/beta keep C's thresholds (1/-1).
        # ⚠️ MEASURED AND THIN. wikitext2 is far sparser than gov_report:
        # n_sample=128 yields only 4 windows (the loader joins 128 text ROWS then
        # reslices), carrying 15 key tokens in total. With last_tokens=512 just
        # 7 survive, spread [4, 0, 1, 2] -- ONE DOCUMENT CONTRIBUTES NOTHING and
        # is skipped by eval_loss. Compare gov_jsd_kt_pp512_s128: 119 survivors
        # over 8 documents, none empty. Treat this metric as a very-low-N probe;
        # to make it substantive raise n_sample (n_sample=2048 -> 65 windows),
        # loosen alpha/beta, or drop the window (score every key token, as the
        # last_tokens=None group C tasks do).
        datasets=['wikitext2'], n_sample=128, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=True, last_tokens=512,
        trunc_len=64, sliding_window=16, alpha=1, beta=-1,
        key_token_suffix='_raw',
    ),
    # ── chat-templated key-token groups ───────────────────────────────────
    # `key_token_on_sample`: the archive is computed on the WHOLE chat sample --
    # affixes included -- not on the bare document, because that sample IS the
    # input being measured. The sample text depends on the answer window (with
    # answer_tokens=A the document is laid out pre+context+post+tail, so `post`
    # moves): MEASURED, all of chat A=0 / A=128 / A=512 and the bare document
    # are four DIFFERENT strings. So each window needs its own archive, named by
    # `key_token_suffix`. Key tokens that land on the affixes are dropped by the
    # label mask (the chat loader sets -100 there), so they cost nothing.
    # Same TOTAL seqlen as the raw twins (the chat convention: the affixes come
    # out of the DOCUMENT budget), so the archive is computed on the shorter
    # body and is NOT the raw archive. Generate it with
    # Llama-3.1 (overhead 35): --dataset chatdoc:gov_report --seqlen 8157.
    # precompute_groups files it under the BASE corpus and maps the intervals
    # onto the document range(s), so <key_token_path>/gov_report is the path.
    'C_pp_chat': dict(
        datasets=['chat:gov_report'], n_sample=8, seqlen=8192, min_seqlen=8192,
        loss_func='jsd', use_key_token=True, last_tokens=512, sides=('train',),
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
        key_token_suffix='_chat-a512', key_token_on_sample=True,
    ),
    'C_wt2_pp_chat': dict(
        # wikitext2 is a STREAM corpus: the chat body is built with
        # add_special_tokens=False, so it differs from the raw loader by the BOS
        # and the two archives are NOT interchangeable (MEASURED: the raw loader
        # is rejected against a chat-body archive; the windows sit one token
        # apart). Generate with --dataset chatdoc:wikitext2 --seqlen 2013.
        # trunc/sliding_window follow C_wt2_pp (64/16 = gov's ratios at 2048).
        # Inherits C_wt2_pp's thinness warning -- 4 windows, ~15 key tokens.
        datasets=['chat:wikitext2'], n_sample=128, seqlen=2048, min_seqlen=0,
        loss_func='jsd', use_key_token=True, last_tokens=512, sides=('train',),
        trunc_len=64, sliding_window=16, alpha=1, beta=-1,
        key_token_suffix='_chat-a512', key_token_on_sample=True,
    ),
    # ── PPL-only groups ────────────────────────────────────────────────────
    # PPL reads test_loaders and never touches dense_logits, so loss_func=
    # 'cross_entropy' here means precompute_groups runs NO FP-teacher pass for
    # these groups, and `sides=('test',)` skips the unused train-loader build.
    # seqlen is group-owned (it re-windows the corpus) ⇒ one group per length.
    #
    # Measured window counts (Llama-3.1 tokenizer, this box):
    #   wikitext2 test  = 289,077 tok → 141 / 35 windows @ 2048 / 8192
    #   c4 (1100 val docs joined) = 517,864 tok → 252 / 63 windows
    # The ladder is 2048 and 8192 only: 4096 was a pure midpoint (the two ends
    # already give the trend) and 16384 would leave wikitext2 with 17 windows. CAVEAT: both corpora are CONCATENATIONS of independent
    # texts, so a longer window buys context LENGTH, not long-range DEPENDENCY
    # — an 8192-token c4 window spans ~16 unrelated web documents. Use 'E_ppl'
    # (gov_report, real single documents) for genuine long-context PPL.
    'A_ppl_sl8192': dict(
        datasets=['wikitext2', 'c4'], n_sample=128, seqlen=8192, min_seqlen=0,
        loss_func='cross_entropy', use_key_token=False, last_tokens=None,
        sides=('test',),
        trunc_len=512, sliding_window=128, alpha=2, beta=-2,
    ),
    'E_ppl': dict(  # gov_report PPL — 128 real documents truncated to 8192 tok
        # get_loader('gov_report', train=False) is get_gov_report(split='test'),
        # i.e. the SAME split/seed the JSD groups B/B_pp/B_lt* use — so gov PPL
        # is a different METRIC on the same documents, not an independent test
        # set. That is fine for measuring quantization damage (both are probes
        # of one model) but it is not a held-out generalization number.
        #
        # n_sample=128, NOT the JSD groups' 8: the 8 exists only because their
        # FP-teacher dense_logits are 16.8 GB at n=8, and PPL stores none. 8 docs
        # would be 65k scored tokens — 4.4x fewer than wikitext2 PPL (289k) — and
        # with 8 documents the DOCUMENT-sampling variance dominates the number
        # (it cancels for arch-vs-arch comparisons, which are paired on the same
        # docs, but not for the absolute PPL). Measured capacity of the split:
        # 973 test docs, median 8,245 tok, 491 >= 8192 → collecting 128 scans 262
        # docs (~4 s of tokenizer time). The n=8 doc set is a strict PREFIX of
        # this one (same seed/shuffle/floor), so gov_ppl and gov_jsd still share
        # their first 8 documents.
        # COST: 128 x 8192 = 1.05M scored tokens per gov_ppl* task (3.6x
        # wt2_ppl_sl8192) — the most expensive metrics in the registry. Drop to
        # n_sample=32 (66 docs scanned) if a big --metrics all sweep needs it.
        datasets=['gov_report'], n_sample=LONG_DOC_N_SAMPLE,
        seqlen=8192, min_seqlen=8192,
        loss_func='cross_entropy', use_key_token=False, last_tokens=None,
        sides=('test',),
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    # ── LongBench long-DOCUMENT PPL (utils/data.get_longbench_ppl) ──────────
    # narrativeqa's `context` is ONE coherent document, so an 8k window carries
    # real long-range dependency — unlike wikitext2/c4 (independent texts
    # joined) or the LongBench multi-hop / passage subsets (unrelated passages
    # concatenated), which give length without dependency.
    # Measured (Llama-3.1 tokenizer, FULL 200-doc test splits, raw `context`):
    #   narrativeqa  median 31,284 tok — 192 docs ≥ 8192, 139 ≥ 16384
    #   qmsum        median 12,934 tok — 160 docs ≥ 8192,  52 ≥ 16384
    # n_sample=128 at 8192 needs 134 (narrativeqa) / 160 (qmsum) documents
    # scanned, i.e. qmsum uses ALL of its qualifying documents — raising
    # n_sample or min_seqlen past that is a hard error from the loader.
    #
    # ⚠️ qmsum is a LONG-CONTEXT REPORTING metric, not a proxy candidate.
    # LongBench GRADES qmsum (utils/longbench.LONGBENCH_DATASETS), so all 128
    # calibration documents are documents the benchmark also reads. That is
    # deliberate and fine for the role: nothing is FIT to this text (the search
    # objective is wikitext2 JSD and post_search.select_joint picks on the
    # archive's stored loss — these tasks are measured after the arch is fixed),
    # and PPL over the raw context and LongBench accuracy over the generated
    # answer are different targets on the same documents. Same-corpus is in fact
    # the cleanest way to say "PPL barely moved, accuracy dropped X%" without a
    # domain confound. What it is NOT is evidence that the metric PREDICTS the
    # benchmark: that pair shares per-architecture, per-document noise, so
    # correlation.py reports it in correlation_contamination.txt.
    # narrativeqa carries no such caveat — it is in NEITHER benchmark list
    # (LONGBENCH_DATASETS 8 / LONGBENCH_E_DATASETS 13), a property of the
    # CURRENT benchmark config (full 21-subset LongBench would grade it too;
    # tests/audit_corpus_contamination.py re-checks).
    'LB_ppl_sl8192': dict(
        datasets=['longbench:narrativeqa', 'longbench:qmsum'],
        n_sample=LONG_DOC_N_SAMPLE, seqlen=8192, min_seqlen=8192,
        loss_func='cross_entropy', use_key_token=False, last_tokens=None,
        sides=('test',),
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    # 2048 tier — the long-document corpora at wikitext2/c4's OWN window. This
    # is the CONTROL that separates the two effects that are otherwise
    # confounded: {wt2,c4}_ppl vs {gov,nqa,qmsum}_ppl_sl2048 differ only in
    # DOMAIN (same 2048 window), while _sl2048 → 8192 within one corpus differs
    # only in LENGTH (same documents, longer window). Capacity is a non-issue:
    # gov_report 960/973, narrativeqa 200/200, qmsum ~199/200 clear 2048.
    # (16384 is not offered: only narrativeqa has the documents for it — 139
    # vs qmsum 52 / gov_report 100 — so it would be a single-corpus outlier.
    # It is executable if ever wanted: measured peak 24.6 GB with the FP16
    # model, ~4.3 GB over the 8192 run.)
    'E_ppl_sl2048': dict(
        datasets=['gov_report'], n_sample=LONG_DOC_N_SAMPLE,
        seqlen=2048, min_seqlen=2048,
        loss_func='cross_entropy', use_key_token=False, last_tokens=None,
        sides=('test',),
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
    'LB_ppl_sl2048': dict(
        datasets=['longbench:narrativeqa', 'longbench:qmsum'],
        n_sample=LONG_DOC_N_SAMPLE, seqlen=2048, min_seqlen=2048,
        loss_func='cross_entropy', use_key_token=False, last_tokens=None,
        sides=('test',),
        trunc_len=256, sliding_window=64, alpha=1, beta=-1,
    ),
}


def is_long_doc(spec):
    """Does this group SELECT documents (rather than window a fixed stream)?
    True for gov_report and every `longbench:<subset>` corpus — their loaders
    shuffle the split and keep the first n_sample documents that clear the
    window, so the seed decides WHICH documents are measured."""
    return any(d == 'gov_report' or d.startswith('longbench:')
               for d in spec['datasets'])


# The chat layout only says something when there IS an answer window: with
# answer_tokens=0 the assistant header just trails the document, which is not a
# protocol anyone measures -- and it would need its own archive (the sample text
# depends on the window). Fail loudly rather than let one be added by accident.
for _g, _g_spec in GROUPS.items():
    if (_g_spec.get('use_key_token')
            and any(str(d).startswith('chat:') for d in _g_spec['datasets'])
            and not _g_spec.get('last_tokens')):
        raise SystemExit(
            f"[metric_specs] chat key-token group '{_g}' has no answer window "
            f"(last_tokens={_g_spec.get('last_tokens')}). The chat layout is only "
            f"used with prefill_prompt + an answer window; give it one or drop "
            f"the group.")


# Pin the document-selection seed on every long-doc group (see LONG_DOC_DATA_SEED)
# AND on every key-token group: an archive is bound to the exact documents by a
# per-slice text hash, so a run with a different --seed would select other
# documents and utils/loss.py would reject the archive outright.
# Done as a loop, not 14 copies of `data_seed=0`, so a new long-doc group is
# covered the moment it is added. A group may still override it explicitly.
for _g_spec in GROUPS.values():
    if is_long_doc(_g_spec) or _g_spec.get('use_key_token'):
        _g_spec.setdefault('data_seed', LONG_DOC_DATA_SEED)
del _g_spec

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
    ('wt2_ppl_sl8192',    'A_ppl_sl8192', 'wikitext2',
        # 35 windows — the longest wikitext2 PPL window this corpus supports
        # without the sample count collapsing (16384 would leave 17).
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('c4_ppl_sl8192',     'A_ppl_sl8192', 'c4',
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('wt2_ppl_sl8192_s512', 'A_ppl_sl8192', 'wikitext2',
        # Same 8192 windows through the REAL KV-cache path (stride=512 ⇒
        # use_cache=True), not the single-shot quant_kv_output=True path. This
        # is the variant that actually exercises KV quantization at 8k.
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=512, prefill_prompt=False, last_tokens=None)),
    ('gov_ppl',           'E_ppl', 'gov_report',
        # Long-document PPL: 8 gov_report docs @ 8192, single forward pass.
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('gov_ppl_s512',      'E_ppl', 'gov_report',
        # Chunked forward (bounds peak activation memory at 8k AND runs the
        # real cache path, so KV quantization is actually exercised).
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=512, prefill_prompt=False, last_tokens=None)),
    ('nqa_ppl',           'LB_ppl_sl8192', 'longbench:narrativeqa',
        # LongBench narrativeqa (novels / screenplays) @ 8192, single forward.
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('nqa_ppl_s512',      'LB_ppl_sl8192', 'longbench:narrativeqa',
        # Same windows through the REAL KV-cache path (stride ⇒ use_cache=True).
        # Positions still see full left context, so it is the same PPL — but
        # measured with the cache actually quantized, which is the point here.
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=512, prefill_prompt=False, last_tokens=None)),
    ('qmsum_ppl',         'LB_ppl_sl8192', 'longbench:qmsum',
        # LongBench qmsum (meeting transcripts) @ 8192 — a spoken-register,
        # speaker-turn corpus, deliberately unlike report/wiki prose. REPORTING
        # metric: the benchmark grades these documents (see the group comment).
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('qmsum_ppl_s512',    'LB_ppl_sl8192', 'longbench:qmsum',
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=512, prefill_prompt=False, last_tokens=None)),
    ('nqa_ppl_sl2048',    'LB_ppl_sl2048', 'longbench:narrativeqa',
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('qmsum_ppl_sl2048',  'LB_ppl_sl2048', 'longbench:qmsum',
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('gov_ppl_sl2048',    'E_ppl_sl2048', 'gov_report',
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=0, prefill_prompt=False, last_tokens=None)),
    ('gov_ppl_pp512_s128', 'E_ppl', 'gov_report',
        # Answer-phase PPL: prefill 7680 tokens, score the last 512 in
        # 128-chunks (the PPL twin of gov_jsd_pp512_s128).
        dict(metric='ppl',  loss_func='cross_entropy',
             stride=128, prefill_prompt=True, last_tokens=512)),
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
    ('wt2_jsd_pp32_s8',   'A_lt32', 'wikitext2',
        # Tightest answer-phase JSD: prefill 2016 tokens, then score the last 32
        # in 8-token chunks. Group A_lt32 supplies the last-32 pre-masked dense.
        dict(metric='loss', loss_func='jsd',
             stride=8, prefill_prompt=True, last_tokens=32)),
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
    ('needle_nll_pp32_s8', 'A', None,
        # Tightest needle answer window: prefill, then 32 tokens in 8-chunks.
        dict(kind='needle_nll',
             stride=8, prefill_prompt=True, last_tokens=32)),
    ('needle_jsd_pp32_s8', 'A', None,
        # JSD twin; its FP-teacher dense cache is keyed by _lt32 so it does not
        # collide with the lt512 / lt128 needle caches.
        dict(kind='needle_jsd',
             stride=8, prefill_prompt=True, last_tokens=32)),
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
    # gov_jsd_pp128_s32 n_sample × seqlen sweep (see the B_lt128_n*_sl* groups).
    # Same forward side as gov_jsd_pp128_s32 (jsd, stride=32, prefill,
    # last_tokens=128); only the group's n_sample / seqlen / min_seqlen differ.
    # Every SWEEP name spells out BOTH knobs (_n{N}_sl{L}); only the base
    # gov_jsd_pp128_s32 (= n8 / sl8192) is left implicit, for back-compat.
    ('gov_jsd_pp128_s32_n8_sl4096', 'B_lt128_n8_sl4096', 'gov_report',
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('gov_jsd_pp128_s32_n8_sl2048', 'B_lt128_n8_sl2048', 'gov_report',
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('gov_jsd_pp128_s32_n4_sl8192', 'B_lt128_n4_sl8192', 'gov_report',
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('gov_jsd_pp128_s32_n4_sl4096', 'B_lt128_n4_sl4096', 'gov_report',
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('gov_jsd_pp128_s32_n4_sl2048', 'B_lt128_n4_sl2048', 'gov_report',
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('gov_jsd_pp128_s32_n128_sl8192', 'B_lt128_n128_sl8192', 'gov_report',
        # Same measurement as gov_jsd_pp128_s32 at the 128-document long-doc
        # default (16x the documents ⇒ 16x the cost + a 4.2 GB teacher set).
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('gov_jsd_pp32_s8',   'B_lt32', 'gov_report',
        # 8k prefill + last-32 answer window scored in 8-token chunks (the
        # gov_report twin of wt2_jsd_pp32_s8; Group B_lt32 pre-masks dense).
        dict(metric='loss', loss_func='jsd',
             stride=8, prefill_prompt=True, last_tokens=32)),
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
    # chat-templated twins of wt2/gov _pp{128,512}_s32 (see the *_chat groups).
    # No key tokens: the same answer-phase protocol as the raw twins, with every
    # sample wrapped in one user turn and the assistant header at
    # seqlen - last_tokens. They are the control for the chat LAYOUT on its own.
    ('wt2_jsd_pp128_s32_chat', 'A_lt128_chat', 'chat:wikitext2',
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('wt2_jsd_pp512_s32_chat', 'A_pp_chat', 'chat:wikitext2',
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=512)),
    ('wt2_jsd_pp512_s128_chat', 'A_pp_chat', 'chat:wikitext2',
        # same group as _s32 -> shares its FP-teacher pass; only the chunk size
        # differs (128 vs 32 cache-appending steps over the same 512 window).
        # This is the chat twin of wt2_jsd_pp512_s128.
        dict(metric='loss', loss_func='jsd',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('gov_jsd_pp128_s32_chat', 'B_lt128_chat', 'chat:gov_report',
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=128)),
    ('gov_jsd_pp512_s32_chat', 'B_pp_chat', 'chat:gov_report',
        dict(metric='loss', loss_func='jsd',
             stride=32, prefill_prompt=True, last_tokens=512)),
    ('gov_jsd_pp512_s128_chat', 'B_pp_chat', 'chat:gov_report',
        # chat twin of gov_jsd_pp512_s128; shares B_pp_chat's teacher with _s32.
        dict(metric='loss', loss_func='jsd',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('gov_jsd_kt_pp512_s128', 'C_pp', 'gov_report',
        # gov_jsd_pp512_s128's key-token twin. 119 of gov_report's 1729 key
        # tokens fall in the last 512 positions (7,10,29,11,24,11,4,23) --
        # 4.6x the pp128 window, non-empty on every document.
        dict(metric='loss', loss_func='jsd',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('wt2_jsd_kt_pp512_s128', 'C_wt2_pp', 'wikitext2',
        # wt2_jsd_pp512_s128's key-token twin.
        dict(metric='loss', loss_func='jsd',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('gov_jsd_kt_pp512_s128_chat', 'C_pp_chat', 'chat:gov_report',
        dict(metric='loss', loss_func='jsd',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('wt2_jsd_kt_pp512_s128_chat', 'C_wt2_pp_chat', 'chat:wikitext2',
        dict(metric='loss', loss_func='jsd',
             stride=128, prefill_prompt=True, last_tokens=512)),
    ('gov_jsd_kt_s512',   'C', 'gov_report',
        # stride=512 chunked forward over the same key-token archive as
        # gov_jsd_kt — bounds peak activation memory at 8K context while
        # keeping the key-token weighting identical.
        dict(metric='loss', loss_func='jsd',
             stride=512, prefill_prompt=False, last_tokens=None)),
]

# ── answer-phase PPL grid ───────────────────────────────────────────────────
# Every PPL corpus × window also gets the two answer-phase protocols
#     _pp128_s32 : prefill the prompt, then score the last 128 tokens in
#                  32-token chunks — the PPL twin of gov_jsd_pp128_s32.
# The 512-token window was dropped: two answer windows per corpus x window
# doubled the grid for a difference that is one scoring span, and 128 is the
# one that matches the JSD side actually in use. (The JSD family keeps its own
# _pp512_s32 tasks — those are separate, pre-existing metrics.) They cost ONE
# extra forward each — PPL stores no teacher logits, so no group and no FP pass
# is added — and they run the real KV-cache path (prefill + stride), unlike the
# single-shot base task which takes the quant_kv_output=True path.
#
# Generated rather than spelled out (24 near-identical tuples). The names follow
# `<base>_pp<window>_s32`, so e.g. nqa_ppl → nqa_ppl_pp512_s32 and
# wt2_ppl_sl8192 → wt2_ppl_sl8192_pp128_s32; `resolve_tasks` prints the full
# resolved list on an unknown name, and METRIC_KEYS holds them all.
_PPL_ANSWER_WINDOWS = (128,)
_PPL_BASES = [  # (base task name, group, dataset) — one per corpus × window
    ('wt2_ppl',          'A',              'wikitext2'),
    ('c4_ppl',           'A',              'c4'),
    ('wt2_ppl_sl8192',   'A_ppl_sl8192',   'wikitext2'),
    ('c4_ppl_sl8192',    'A_ppl_sl8192',   'c4'),
    ('gov_ppl_sl2048',   'E_ppl_sl2048',   'gov_report'),
    ('gov_ppl',          'E_ppl',          'gov_report'),
    ('nqa_ppl_sl2048',   'LB_ppl_sl2048',  'longbench:narrativeqa'),
    ('qmsum_ppl_sl2048', 'LB_ppl_sl2048',  'longbench:qmsum'),
    ('nqa_ppl',          'LB_ppl_sl8192',  'longbench:narrativeqa'),
    ('qmsum_ppl',        'LB_ppl_sl8192',  'longbench:qmsum'),
]
_existing = {t[0] for t in METRIC_TASKS}
for _base, _grp, _ds in _PPL_BASES:
    assert (_base, _grp, _ds) in [(t[0], t[1], t[2]) for t in METRIC_TASKS], \
        f'answer-phase grid: base task {_base} is not defined above'
    for _lt in _PPL_ANSWER_WINDOWS:
        _name = f'{_base}_pp{_lt}_s32'
        if _name in _existing:            # already spelled out above
            continue
        _existing.add(_name)
        METRIC_TASKS.append((_name, _grp, _ds,
                             dict(metric='ppl', loss_func='cross_entropy',
                                  stride=32, prefill_prompt=True, last_tokens=_lt)))
del _base, _grp, _ds, _lt, _name, _existing

# ── one metric (and one group) per key-token EVALUATOR ──────────────────────
# The evaluator decides WHICH tokens are key, and swapping it moves the value:
# MEASURED on gov_report, Qwen2.5-72B vs Llama-3.1-8B pick only ~half the same
# tokens (jaccard 0.38) and the JSD differs by 10-16%. A single name like
# `gov_jsd_kt` would therefore mean two different numbers depending on which
# archive a run happened to point at, so the evaluator goes in the NAME and the
# archive path is DERIVED from it (no --key_token_path root to get wrong).
# Groups double too: dense_logits are pre-masked to the key positions, so a
# different key set needs its own FP-teacher pass.
_KT_EVALS = {           # name tag -> evaluator model (the `eval-` in the path)
    'q72b': 'Qwen2.5-72B-Instruct',
    'q7b':  'Qwen2.5-7B-Instruct',   # same family as q72b, 10x cheaper to run
    'l8b':  'Llama-3.1-8B-Instruct',
}

_kt_base_groups = [g for g, sp in GROUPS.items() if sp.get('use_key_token')]
for _g in _kt_base_groups:
    _spec = GROUPS.pop(_g)
    for _tag, _ev in _KT_EVALS.items():
        GROUPS[f'{_g}_{_tag}'] = dict(_spec, key_token_eval=_ev)

_expanded = []
for _t in METRIC_TASKS:
    if _t[1] in _kt_base_groups:
        for _tag in _KT_EVALS:
            _expanded.append((f'{_t[0]}_{_tag}', f'{_t[1]}_{_tag}', _t[2], dict(_t[3])))
    else:
        _expanded.append(_t)
METRIC_TASKS = _expanded
del _kt_base_groups, _expanded, _g, _spec, _tag, _ev, _t


def key_token_root(base_dir, spec, target_model):
    """<base_dir>/kt_eval-<evaluator>_tgt-<target>_<layout> for one group.

    Derived, not configured: the evaluator comes from the metric name (via its
    group) and the target is the model being measured, so a run cannot pair a
    metric with another evaluator's archive.
    """
    import os
    return os.path.join(base_dir, f"kt_eval-{spec['key_token_eval']}"
                                  f"_tgt-{target_model}{spec['key_token_suffix']}")


def key_token_archive_status(base_dir, spec, target_model, dataset=None):
    """(ok, reason) — is a usable key-token archive present for this group?

    Cheap and structural: the derived root exists, the per-corpus protocol
    directory exists, it holds slices and a meta.json, and that manifest's
    protocol matches the group. It deliberately does NOT open the loaders to
    check the text hash — that needs the corpus built and is what the manifest
    check does at measurement time.

    Callers use it to SKIP a metric they cannot measure rather than abort the
    whole run; `reason` is printed so a skip is never silent.
    """
    import os
    from utils.data import key_token_dir, key_token_dirname
    if not base_dir:
        return False, 'no --key_token_path'
    d = dataset or spec['datasets'][0]
    root = key_token_root(base_dir, spec, target_model)
    if not os.path.isdir(root):
        return False, 'no archive root %s' % os.path.basename(root)
    args = (spec['n_sample'], spec['seqlen'], spec['min_seqlen'], spec['trunc_len'],
            spec['sliding_window'], spec['alpha'], spec['beta'],
            spec.get('data_seed', 0))
    path = key_token_dir(root, d, *args)
    if not path or not os.path.isdir(path):
        return False, 'missing %s/%s' % (os.path.basename(root),
                                         key_token_dirname(d, *args))
    if not any(f.startswith('slice_') for f in os.listdir(path)):
        return False, '%s holds no slice files' % os.path.basename(path)
    mp = os.path.join(path, 'meta.json')
    if not os.path.exists(mp):
        return False, '%s has no meta.json' % os.path.basename(path)
    try:
        man = json.load(open(mp))
    except Exception as e:                                          # noqa: BLE001
        return False, 'unreadable meta.json (%r)' % e
    bad = [k for k in ('trunc_len', 'sliding_window', 'alpha', 'beta')
           if man.get(k) is not None and man.get(k) != spec[k]]
    if bad:
        return False, 'manifest protocol differs on %s' % ','.join(bad)
    return True, os.path.basename(path)


METRIC_KEYS = [t[0] for t in METRIC_TASKS]
BENCH_KEYS = ['longbench', 'longbench_e', 'ruler']
ALL_KEYS = METRIC_KEYS + BENCH_KEYS


TASKS_BY_NAME = {t[0]: t for t in METRIC_TASKS}


def spec_sha8(name):
    """Hash of everything that defines what a metric NAME measures.

    A name is a promise ("this number means the same thing everywhere"), but the
    definition behind it lives in code and can be edited. Storing this hash next
    to a measured value turns the promise into something checkable: a number
    whose spec hash no longer matches was produced by a DIFFERENT definition and
    must be re-measured, not compared.

    Covers the group (datasets / n_sample / seqlen / min_seqlen / last_tokens /
    loss_func / use_key_token / data_seed / sides) and the task (metric /
    loss_func / stride / prefill_prompt / last_tokens / kind). NOT covered:
    --key_token_path (a run-time path, injected by groups_for) and the model —
    those belong to the measurement CONFIG, which is the directory key.
    """
    _k, g, ds, kw = TASKS_BY_NAME[name]
    payload = {'dataset': ds, 'group': GROUPS[g], 'task': kw}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:8]


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


# ── one named metric → the flat knobs a single-metric script takes ──────────
# correlation.py / post_search.py measure MANY names per run, so they drive the
# evaluator through groups_for → precompute_groups → apply_group → run_task.
# The single-metric scripts (search.py, second_search*.py, baseline_search.py,
# sample_surrogate.py) instead carry ONE protocol on the CLI
# (--dataset/--n_sample/--seqlen/--stride/--last_tokens/…), and the protocol had
# to be retyped by hand in each wrapper — which is exactly how a search archive
# ends up not being the measurement the correlation table judging it reports.
#
# `task_knobs` closes that WITHOUT touching those entry points: it reads the same
# registry entry out as flat knobs, and `scripts/metric_task.sh` (via the
# __main__ emitter at the bottom of this file) turns them into the wrapper's own
# shell variables. So the python CLIs stay exactly as they are — they still take
# only knobs — and `METRIC_TASK=wt2_jsd_pp512_s128_chat` in a script is the same
# measurement as the hand-set knobs by construction, dir tags included.
KNOB_KEYS = ('dataset', 'n_sample', 'seqlen', 'min_seqlen', 'metric', 'loss_func',
             'stride', 'prefill_prompt', 'last_tokens', 'score', 'use_key_token',
             'trunc_len', 'sliding_window', 'alpha', 'beta', 'key_token_path')


def task_knobs(name, *, key_token_path='', target_model=None,
               require=None, context='this entry point'):
    """One registry name → `{knob: value}` for a single-metric CLI.

    Consumed by `scripts/metric_task.sh` (through this module's `--shell`
    emitter), which assigns them to the wrapper script's variables before it
    builds its arg list. Nothing in search.py / second_search*.py resolves a
    metric name: they keep taking raw knobs, and the shell is what knows the
    registry.

    Mirrors apply_group + run_task exactly: the GROUP owns the data side
    (dataset / n_sample / seqlen / min_seqlen / use_key_token + the key-token
    protocol), the TASK owns the forward side (metric / loss_func / stride /
    prefill_prompt / last_tokens / score, defaulting to 'last' as run_task
    does). `0` is the CLI's OFF for stride/last_tokens — argparse cannot pass
    None — matching utils/func.metric_protocol and LlamaEvaluator.

    Also returned, as metadata rather than knobs: `metric_task` (the name),
    `metric_spec` (spec_sha8 — the DEFINITION hash of what the name meant when
    it was resolved; the wrapper prints it), `group` and `data_seed`.

    Two things it refuses rather than mis-measure:
      * custom-loader tasks (`kind=needle_*` / `gsm8k_unpad_pp`) — they build
        their own prompts inside correlation.py and are not expressible as a
        dataset + protocol (post_search rejects them for the same reason);
      * a key-token metric whose archive is missing — the evaluator would
        otherwise fail deep inside the FP-teacher pass. `key_token_path` is the
        BASE directory here (e.g. `key_token`); the per-metric root
        `kt_eval-<evaluator>_tgt-<target>_<layout>` is DERIVED from the group
        (the evaluator comes from the name's `_q72b`/`_q7b`/`_l8b` suffix), so a
        run cannot pair a metric with another evaluator's archive.
    """
    (key, g, ds, kw), = resolve_tasks([name])
    if kw.get('kind'):
        raise SystemExit(
            f"metric task '{key}' is a custom-loader task (kind={kw['kind']}): it "
            f"generates its own prompts inside correlation.py and has no "
            f"--dataset + protocol form. Measure it with correlation.py.")
    spec = GROUPS[g]
    # `require` = what the CALLER can honour, checked HERE — before the key-token block
    # below asks for an archive — so an entry point with no key-token plumbing says
    # exactly that instead of demanding a path it would refuse to use anyway.
    for rk, want in (require or {}).items():
        got = (bool(spec.get('use_key_token')) if rk == 'use_key_token'
               else kw.get(rk, spec.get(rk)))
        if rk == 'metric':
            got = kw.get('metric', 'loss')
        if got != want:
            raise SystemExit(
                f"metric task '{key}' has {rk}={got!r}, but {context} can only measure "
                f"{rk}={want!r}. Pick a metric with {rk}={want!r} "
                f"(`python -m utils.metric_specs` lists them), or measure this one with "
                f"correlation.py / post_search.py.")
    out = dict(
        metric_task=key, metric_spec=spec_sha8(key), group=g,
        data_seed=int(spec.get('data_seed', 0)),
        dataset=ds,
        n_sample=int(spec['n_sample']),
        seqlen=int(spec['seqlen']),
        min_seqlen=int(spec['min_seqlen']),
        metric=kw.get('metric', 'loss'),
        loss_func=kw.get('loss_func', spec.get('loss_func', 'cross_entropy')),
        stride=int(kw.get('stride') or 0),
        prefill_prompt=bool(kw.get('prefill_prompt')),
        last_tokens=int(kw.get('last_tokens') or 0),
        score=kw.get('score', 'last'),
        use_key_token=bool(spec.get('use_key_token')),
        trunc_len=int(spec.get('trunc_len', 512)),
        sliding_window=int(spec.get('sliding_window', 128)),
        alpha=int(spec.get('alpha', 2)),
        beta=int(spec.get('beta', -2)),
        key_token_path='',
    )
    if out['use_key_token']:
        if not (key_token_path and target_model):
            raise SystemExit(
                f"metric task '{key}' is key-token weighted: pass --key_token_path "
                f"(the BASE archive directory, e.g. 'key_token'; the "
                f"kt_eval-<evaluator>_tgt-<model> root is derived from the metric "
                f"name)" + ('' if target_model else " and --model_name"))
        out['key_token_path'] = key_token_root(key_token_path, spec, target_model)
        ok, why = key_token_archive_status(key_token_path, spec, target_model, ds)
        if not ok:
            raise SystemExit(
                f"metric task '{key}': no usable key-token archive ({why}).\n"
                f"  generate it with gen_key_token.py --metrics {key} "
                f"--model_name {spec['key_token_eval']} --target_model {target_model} "
                f"--save_path {key_token_path}")
    return out


def groups_for(tasks, key_token_path='', target_model=None):
    """The group specs `tasks` need, as [(group_name, spec)] in first-use order.

    A key-token group needs `key_token_path` (the DIRECTORY the archives live
    in). With `target_model` the per-group root is derived from it —
    `kt_eval-<evaluator>_tgt-<target>_<layout>`, the evaluator coming from the
    metric name — which is what a caller measuring one model should pass.
    Without it the bare directory is handed through (legacy callers)."""
    out, seen = [], set()
    for _key, g, _ds, _kw in tasks:
        if g in seen:
            continue
        seen.add(g)
        spec = dict(GROUPS[g])
        if spec.get('use_key_token'):
            if not key_token_path:
                # Defensive: correlation.py never reaches this — it drops
                # unmeasurable key-token TASKS before building groups (see
                # correlation.drop_key_token_tasks), so no such group survives.
                # A caller that lands here handed key-token tasks straight
                # through without that filter.
                raise SystemExit(
                    f"metric group '{g}' needs key tokens → pass --key_token_path, "
                    f"or filter the key-token tasks out first "
                    f"(correlation.drop_key_token_tasks does this and reports "
                    f"which metrics it skipped).")
            spec['key_token_path'] = (key_token_root(key_token_path, spec, target_model)
                                      if target_model else key_token_path)
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
                      tasks=None, fail_soft=False):
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

    `fail_soft` turns a per-group DATA-BUILD failure (a corpus that cannot be
    loaded — missing file, offline hub, …) into a {'error', 'traceback'} payload
    for that group instead of an exception, so one bad corpus cannot take down
    every other group's metrics (a narrativeqa cache-miss once killed all 26
    metrics of 400 correlation jobs). The FP-teacher pass and spec-consistency
    errors stay hard either way.

    Returns {group: dict(train_loaders, test_loaders, dense_logits,
                         key_token_list, spec)}; with fail_soft, failed groups
    map to dict(error=..., traceback=...) instead.
    """
    from utils.data import get_loader, chat_spans_for, key_token_corpus, key_token_dir
    from utils.eval import get_logits, get_tokenizer
    from utils.func import clean_up, get_hfmodel
    from utils.loss import get_key_token_list

    wanted = None
    if tasks is not None:
        wanted = {(g, ds) for _k, g, ds, kw in tasks if needs_dense(kw)}
    out, pending = {}, []
    for g, spec in group_items:
        # Long-doc groups pin `data_seed` so the DOCUMENT SET is fixed by the
        # metric name; everything else follows the run's --seed as before.
        data_seed = spec.get('data_seed')
        data_seed = int(seed) if data_seed is None else int(data_seed)
        if data_seed != int(seed):
            print(f"[metric_specs] group '{g}': document selection pinned to "
                  f"data_seed={data_seed} (run --seed {seed} not used here)")

        def _loaders(train, spec=spec, data_seed=data_seed):
            return {d: accelerator.prepare(get_loader(
                        d, model=model_id, n_sample=spec['n_sample'],
                        batch_size=batch_size, train=train, seed=data_seed,
                        seqlen=spec['seqlen'], min_seqlen=spec['min_seqlen'],
                        # a chat: corpus with an answer window splits the DOCUMENT, so its span is a
                        # GROUP property (it changes the data, not just the forward)
                        answer_tokens=spec['last_tokens']))
                    for d in spec['datasets']}
        # `sides` (optional) restricts which loader sides are built: 'loss'
        # metrics read train_loaders, 'ppl' reads test_loaders, so a PPL-only
        # group can skip the train build entirely (the gov_report loader in
        # particular tokenizes documents until n_sample long ones are found).
        sides = tuple(spec.get('sides') or ('train', 'test'))
        is_div = spec['loss_func'] in DIVERGENCE_LOSSES
        dense_ds = [d for d in spec['datasets']
                    if is_div and (wanted is None or (g, d) in wanted)]
        if dense_ds and 'train' not in sides:
            raise SystemExit(f"[metric_specs] group '{g}': dense_logits are "
                             f"computed on the train side, but sides={sides}.")
        try:
            out[g] = dict(train_loaders=_loaders(True) if 'train' in sides else {},
                          test_loaders=_loaders(False) if 'test' in sides else {},
                          dense_logits={d: None for d in dense_ds},
                          key_token_list={d: None for d in spec['datasets']}
                          if spec['use_key_token'] else {},
                          spec=spec)
        except Exception as e:                                    # noqa: BLE001
            if not fail_soft:
                raise
            import traceback
            out[g] = dict(error=repr(e), traceback=traceback.format_exc())
            print(f"[metric_specs] group '{g}' data build FAILED "
                  f"(datasets={spec['datasets']}): {e!r} — its metrics will be "
                  f"recorded as errors; the other groups continue.")
            continue
        if dense_ds or spec['use_key_token']:
            pending.append((g, spec))

    if not pending:
        return out

    print(f"[metric_specs] one FP-teacher pass for group(s) "
          f"{[g for g, _ in pending]} (dense_logits are arch-independent) …")
    t0 = time()
    fp = get_hfmodel(model_id, dtype=dtype, device_map=device_map)
    fp.eval()
    # only the key-token path needs a tokenizer here; loading one for a
    # dense-logits-only group is a pointless read (and made the mocked test
    # reach the hub)
    tok = (get_tokenizer(model_id, use_fast=True)
           if any(sp['use_key_token'] for _, sp in pending) else None)
    for g, spec in pending:
        r = out[g]
        if spec['use_key_token']:
            # A pre-flight check cannot see a text-hash mismatch (that needs the
            # loaders), so the BUILD is fail-soft too when the caller asked for
            # it: one unusable archive becomes an error entry for its group
            # instead of taking down every other metric in the run.
            # Two chat-corpus corrections, mirroring LlamaEvaluator:
            #  * the archive is computed on the RAW document, so it is filed under
            #    the BASE corpus -- `chat:gov_report` must look in
            #    <key_token_path>/gov_report, not <…>/chat:gov_report;
            #  * the character intervals address the DOCUMENT, so the offsets must
            #    be built over the document range(s) only. Without this the
            #    template affixes shift every offset and the manifest check
            #    rejects the archive (which is the good failure, but a hard stop).
            _ans = spec.get('answer_span') or spec['last_tokens']
            try:
              r['key_token_list'] = {
                d: get_key_token_list(
                    evaluator_model=fp, evaluator_tokenizer=tok, loader=loader,
                    trunc_len=spec['trunc_len'], sliding_window=spec['sliding_window'],
                    alpha=spec['alpha'], beta=spec['beta'],
                    load_path=key_token_dir(
                        spec.get('key_token_path', ''), d, spec['n_sample'],
                        spec['seqlen'], spec['min_seqlen'], spec['trunc_len'],
                        spec['sliding_window'], spec['alpha'], spec['beta'],
                        spec.get('data_seed', 0)),
                    mode='offline',
                    # doc_spans maps intervals computed on the BARE document
                    # into a chat sample. An on-sample archive needs no mapping:
                    # its offsets already refer to the whole sequence.
                    doc_spans=(None if spec.get('key_token_on_sample')
                               else chat_spans_for(d, tok, spec['seqlen'], _ans)))
                for d, loader in r['train_loaders'].items()}
            except Exception as e:                                # noqa: BLE001
                # An archive can be structurally fine and still not match the
                # documents this loader produced (the text-hash check). Callers
                # that asked for fail_soft get an error entry for this group and
                # keep every other metric; the rest still raise.
                if not fail_soft:
                    raise
                import traceback
                out[g] = dict(error=repr(e), traceback=traceback.format_exc())
                print(f"[metric_specs] group '{g}' key-token build FAILED: {e!r} "
                      f"— its metrics will be recorded as errors; the other "
                      f"groups continue.")
                continue
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
        score=eval_kwargs.get('score', 'last'),
        tokenizer=evaluator.tokenizer)


# ── measurement protocol, for embedding in an archive ───────────────────────
# The SAVE-dir tag (scripts/metric_tag.sh) is a short IDENTITY label — dir names
# sit near the 255-byte limit, so the numbers can't live there. results.txt does
# hold every arg, but only for a run that finished. iter_<it>.stats is the file
# that actually travels to second_search / post_search, so the protocol goes in
# there: an archive should be self-describing about how its loss was measured.
PROTOCOL_KEYS = ('dataset', 'datasets', 'n_sample', 'seqlen', 'min_seqlen',
                 'data_batch_size', 'metric', 'loss_func', 'stride',
                 'prefill_prompt', 'last_tokens', 'score', 'use_key_token',
                 'attn_sink', 'residual_length')


def protocol_dict(args):
    """Measurement-protocol subset of `args` (a dict or a Namespace), skipping
    keys the caller doesn't have."""
    get = args.get if hasattr(args, 'get') else (lambda k, d=None: getattr(args, k, d))
    return {k: get(k) for k in PROTOCOL_KEYS if get(k) is not None}


# ── CLI: name → knobs, for shells that must name the same metric ────────────
# `python -m utils.metric_specs --shell <name>` prints the search.sh variables
# for one metric, so the SAVE-dir name is DERIVED from the registry instead of
# being hand-typed next to it (a dir name that disagrees with the measurement is
# the failure mode this whole module exists to prevent). `--json` gives the raw
# knob dict; no argument lists every valid name.
if __name__ == '__main__':
    import argparse

    _ap = argparse.ArgumentParser(
        description='resolve a named calibration metric into flat CLI knobs')
    _ap.add_argument('name', nargs='?', help='metric name (omit to list them all)')
    _ap.add_argument('--shell', action='store_true',
                     help='emit KEY=VALUE lines for `eval` in scripts/search.sh')
    _ap.add_argument('--json', action='store_true', help='emit the knob dict as JSON')
    _ap.add_argument('--key_token_path', default='',
                     help='BASE key-token archive dir (the kt_eval-*_tgt-* root is derived)')
    _ap.add_argument('--model_name', default='', help='target model (key-token metrics)')
    _ap.add_argument('--loss_only', action='store_true',
                     help="refuse anything the 2nd-stage entry points cannot measure "
                          "(metric='loss', no key tokens) — scripts/second_search*.sh")
    _a = _ap.parse_args()

    if not _a.name:
        print('\n'.join(METRIC_KEYS))
        raise SystemExit(0)

    _k = task_knobs(_a.name, key_token_path=_a.key_token_path,
                    target_model=_a.model_name or None,
                    require={'metric': 'loss', 'use_key_token': False} if _a.loss_only else None,
                    context='the 2nd-stage search')
    if _a.shell and _k['data_seed'] != 0:
        # The wrapper scripts build their loaders at LlamaEvaluator's default seed (none
        # of them passes one), so a group pinning a different document-selection seed
        # cannot be honoured from a shell. No group does today (LONG_DOC_DATA_SEED = 0);
        # this is the guard for the one that gets added. correlation.py/post_search.py
        # DO honour data_seed, hence the check lives on the shell path only.
        raise SystemExit(
            f"metric task '{_a.name}' pins data_seed={_k['data_seed']}, but the wrapper "
            f"scripts build their loaders at seed 0. Measure it with correlation.py / "
            f"post_search.py, which honour the group's data_seed.")
    if _a.json or not _a.shell:
        print(json.dumps(_k, indent=2, sort_keys=True))
        raise SystemExit(0)

    # search.sh keeps the RAW corpus and the chat layout in two variables (the
    # dir tag `_ct` and the key-token archive layout both key off the latter),
    # so the `chat:` prefix is split back out here rather than in the shell.
    _ds, _chat = _k['dataset'], False
    if _ds.startswith('chat:'):
        _ds, _chat = _ds[len('chat:'):], True
    _sh = [
        ('DATASET', _ds), ('USE_CHAT_TEMPLATE', _chat),
        ('N_SAMPLE', _k['n_sample']), ('SEQLEN', _k['seqlen']),
        ('MIN_SEQLEN', _k['min_seqlen']),
        ('METRIC', _k['metric']), ('LOSS_FUNC', _k['loss_func']),
        ('STRIDE', _k['stride']), ('PREFILL_PROMPT', _k['prefill_prompt']),
        ('LAST_TOKENS', _k['last_tokens']), ('SCORE', _k['score']),
        ('USE_KEY_TOKEN', _k['use_key_token']),
        ('TRUNC_LEN', _k['trunc_len']), ('SLIDING_WINDOW', _k['sliding_window']),
        ('ALPHA', _k['alpha']), ('BETA', _k['beta']),
        # the DERIVED kt_eval-<evaluator>_tgt-<model> root — what the script passes
        # as --key_token_path (no python entry point resolves the name, so nothing
        # downstream would derive it)
        ('KEY_TOKEN_PATH', _k['key_token_path']),
        ('METRIC_SPEC', _k['metric_spec']),
    ]
    for _n, _v in _sh:
        if isinstance(_v, bool):
            _v = 'True' if _v else 'False'
        print(f"{_n}='{_v}'")
