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
                        configure_model_cache, get_net_info, clean_up)
from utils.select import (build_arch, select_valid_nd_idx, assemble_F)
from utils.eval import eval_metric
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
        # custom path: unpad each example (KIVI 4D-mask bug on padded inputs)
        dict(kind='jsd_custom', unpad=True, stream_dense=False)),
    ('gov_jsd',           'B', 'gov_report',
        # custom path: keep dense_logits on CPU, stream per-seq to GPU
        dict(kind='jsd_custom', unpad=False, stream_dense=True)),
    ('gov_jsd_s512',      'B', 'gov_report',
        # full-seq JSD with stride=512 chunked forward + stream_dense
        # (same Group B as gov_jsd; only differs in forward mode)
        dict(kind='jsd_custom', unpad=False, stream_dense=True,
             stride=512, prefill_prompt=False)),
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
    ctx = init_run(args)
    expr_map = build_expr_map(args, ctx)
    nd = build_nd(args, ctx, expr_map)
    expr_keys, _esm, _efm = nd.expr_keys, nd.esm, nd.efm
    print(f"[correlation/sample] expr_front={args.expr_front} → "
          f"{'lazy comp_obj-pruned' if getattr(nd, 'lazy', False) else 'dense'} path  "
          f"(n_total={nd.n_total:.3e}, expr_keys={expr_keys})")

    if args.n_archs is None or args.n_archs <= 0:
        raise SystemExit("[correlation/sample] --n_archs must be > 0")

    valid_nd_idx = select_valid_nd_idx(
        nd.nd_shape, nd.new_metric_nd, nd.comp_nd_list,
        comp_obj_min=args.comp_obj_min, comp_obj_max=args.comp_obj_max,
        random_sample=args.n_archs, has_quantile=False, has_prefer=False)
    if len(valid_nd_idx) == 0:
        raise SystemExit(
            "[correlation/sample] 0 candidates after comp_obj filter — "
            "widen --comp_obj_min/--comp_obj_max.")

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


def _build_evaluator(args, ctx, *, datasets, n_sample, seqlen, min_seqlen,
                     loss_func, use_key_token, key_token_path,
                     trunc_len, sliding_window, alpha, beta,
                     last_tokens=None):
    """One LlamaEvaluator with the requested data-side config. `last_tokens`
    here is set on the evaluator at init so dense_logits gets pre-masked to
    the last N positions — must match the eval_loss last_tokens used at
    metric-call time (eval_loss compares len-N logits vs len-N dense)."""
    model_id = f'{args.model_path}/{args.model_name}'
    quant_model_paths = args.quant_model_paths if 'hqq' in args.w_method else []
    return LlamaEvaluator(
        ctx.config, accelerator=ctx.accelerator, model_id=model_id,
        method={'w': args.w_method, 'kv': args.kv_method},
        quant_model_paths=quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=seqlen, min_seqlen=min_seqlen, n_sample=n_sample,
        datasets=datasets, device_map=ctx.device_map, dtype=ctx.dtype,
        bits={'w': args.w_bits, 'k': args.k_bits, 'v': args.v_bits},
        group_size=ctx.group_size, residual_length=args.residual_length,
        k_quant_scheme=args.k_quant_scheme, v_quant_scheme=args.v_quant_scheme,
        loss_func=loss_func, last_tokens=last_tokens,
        use_key_token=use_key_token, trunc_len=trunc_len,
        sliding_window=sliding_window, alpha=alpha, beta=beta,
        key_token_path=key_token_path)


def _run_jsd_custom(args, ctx, evaluator, dataset, *, unpad=False,
                    stream_dense=False, stride=0, prefill_prompt=False,
                    ignore_index=-100):
    """Custom JSD eval — same metric as eval_loss's jsd path, with knobs the
    stock loop doesn't have:

    - unpad=True: per-example, slice input_ids/attention_mask to the actual
      length BEFORE forward. Workaround for the KIVI fake_quant bug that
      crashes (`attention_mask[:, :, -1]` IndexError) when HF builds a 4D
      mask for inputs containing padding zeros. Used by gsm8k_jsd.
    - stream_dense=True: keep evaluator.dense_logits on CPU and move only
      the active per-seq tensor to GPU just before the JSD call. Saves
      ~16 GiB for gov_report-scale dense_logits (8 × 8195 × 128k vocab fp16).
    - stride>0 (and not prefill_prompt): chunked forward through the full
      sequence with past_key_values (use_cache=True). Mirrors eval_loss's
      stride path. Needed for gov_jsd_s512 so we can both stream dense AND
      avoid a single 8K-context activation peak.
    - prefill_prompt=True (uses evaluator.last_tokens): prefill the prompt
      [0:total-last_tokens] in one forward, then stride through the answer
      span. Currently NOT needed by any task that uses the custom path
      (gov_jsd_pp512_s128 uses the standard path because dense_logits is
      small under last_tokens=512), but supported for symmetry.

    Mirrors eval_loss numerics: total loss = Σ_seq (mean_jsd_per_token ×
    n_tokens) / Σ_seq n_tokens; mask = (labels != ignore) & (optional
    last_tokens window).
    """
    from utils.eval import get_loss_mask
    from utils.loss import JSD
    model = evaluator.model
    use_cache = stride > 0 or prefill_prompt
    configure_model_cache(args, model, use_cache=use_cache)
    loader = evaluator.train_loaders[dataset]
    dense_list = evaluator.dense_logits.get(dataset)
    if dense_list is None:
        raise RuntimeError(f"evaluator has no dense_logits for '{dataset}'")
    if stream_dense:
        # Replace the GPU tensors with CPU copies IN-PLACE so the originals
        # are deref'd and clean_up() actually frees their GPU memory (saves
        # ~16 GiB for gov_report-scale; without this rebind the GPU copies
        # stay alive via evaluator.dense_logits[dataset] and forward still
        # OOMs).
        new_batches = []
        for batch in dense_list:
            new_seqs = []
            for t in batch:
                new_seqs.append(t.detach().to('cpu'))
            new_batches.append(new_seqs)
        evaluator.dense_logits[dataset] = new_batches
        dense_list = new_batches
        del new_batches, new_seqs
        clean_up()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"[jsd_custom/stream] dense_logits → CPU, "
              f"GPU free={free/1e9:.2f}GB / {total/1e9:.2f}GB")
    last_tokens = getattr(evaluator, 'last_tokens', None)
    device = model.device
    jsd_fn = JSD()

    total_loss = 0.0
    total_tokens = 0
    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"jsd_custom/{dataset}")
    for batch_idx, (inputs, attention_mask, labels) in pbar:
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        for seq_idx in range(inputs.shape[0]):
            seq_in = inputs[seq_idx:seq_idx + 1]
            seq_attn = (attention_mask[seq_idx:seq_idx + 1]
                        if attention_mask is not None else None)
            seq_lab = labels[seq_idx]

            if unpad and seq_attn is not None:
                real_len = int(seq_attn.sum().item())
                if real_len < 2:
                    continue
                seq_in = seq_in[:, :real_len]
                seq_attn = None    # all-ones now → no 4D mask, no KIVI bug
                seq_lab = seq_lab[:real_len]

            T = seq_in.shape[1]
            if prefill_prompt and last_tokens is not None and 0 < last_tokens < T:
                # prefill prompt + stride answer (mirrors eval_loss path 1)
                prompt_len = T - last_tokens
                ans_stride = stride if stride and stride > 0 else last_tokens
                chunks, pkv = [], None
                p_out = model(seq_in[:, :prompt_len],
                              attention_mask=(seq_attn[:, :prompt_len] if seq_attn is not None else None),
                              past_key_values=pkv, use_cache=True)
                chunks.append(p_out.logits)
                pkv = p_out.past_key_values
                for s_pos in range(prompt_len, T, ans_stride):
                    e_pos = min(s_pos + ans_stride, T)
                    c_attn = seq_attn[:, :e_pos] if seq_attn is not None else None
                    c_out = model(seq_in[:, s_pos:e_pos], attention_mask=c_attn,
                                  past_key_values=pkv, use_cache=True)
                    chunks.append(c_out.logits)
                    pkv = c_out.past_key_values
                lm_logits = torch.cat(chunks, dim=1)
                del chunks, pkv
            elif stride and stride > 0:
                # chunked forward through full sequence (mirrors eval_loss path 2)
                chunks, pkv = [], None
                for s_pos in range(0, T, stride):
                    e_pos = min(s_pos + stride, T)
                    c_attn = seq_attn[:, :e_pos] if seq_attn is not None else None
                    c_out = model(seq_in[:, s_pos:e_pos], attention_mask=c_attn,
                                  past_key_values=pkv, use_cache=True)
                    chunks.append(c_out.logits)
                    pkv = c_out.past_key_values
                lm_logits = torch.cat(chunks, dim=1)
                del chunks, pkv
            else:
                lm_logits = model(seq_in, attention_mask=seq_attn).logits
            shift_logits = lm_logits[0, :-1].contiguous()       # (T-1, V) on GPU
            shift_labels = seq_lab[1:].contiguous()
            del lm_logits
            mask = get_loss_mask(shift_labels, key_tokens=None,
                                 last_tokens=last_tokens,
                                 ignore_index=ignore_index, device=device)
            n_tok = int(mask.sum().item())
            if n_tok == 0:
                continue
            dense_seq = dense_list[batch_idx][seq_idx].contiguous()
            if dense_seq.device != device:
                dense_seq = dense_seq.to(device, non_blocking=True)
            loss = jsd_fn(shift_logits[mask], dense_seq)
            total_loss += float(loss.detach().item()) * n_tok
            total_tokens += n_tok
            # release per-seq intermediates before the next forward
            del shift_logits, dense_seq
        clean_up()
        pbar.set_postfix(jsd=f"{total_loss / max(total_tokens, 1):.5f}",
                         toks=total_tokens)
    return total_loss / max(total_tokens, 1)


def _run_needle_nll(args, ctx, model, model_id):
    """Cheap Needle-in-a-Haystack NLL (RULER-style prompts at ~wikitext
    cost: --needle_n_sample samples × --needle_seqlen tokens).

    Computes cross-entropy on the answer-needle tokens only — cross_entropy
    NOT JSD, so no FP dense_logits needed. Returns mean per-token NLL over
    the answer span.

    `--needle_task` selects the NIAH variant. niah_multikey_2 is the default:
    the haystack is filled with LOOK-ALIKE distractor needles
    ("special magic number for <random word> is <random number>") so the
    model must distinguish the queried key from many fakes. Substantially
    harder than the repetitive-haystack niah_single_1. Other supported
    variants: niah_single_1/2/3, niah_multikey_1/3 (all return a single
    answer; multivalue/multiquery are skipped because their multi-answer
    format breaks the single-answer NLL path).
    """
    from utils.ruler_utils import niah_utils as _niah
    import random as _random
    tokenizer = get_tokenizer(model_id)
    configure_model_cache(args, model, use_cache=False)

    task_name = args.needle_task
    if not hasattr(_niah, task_name):
        raise SystemExit(f"[needle_nll] unknown --needle_task '{task_name}'. "
                         f"Valid: niah_single_1, niah_single_2, niah_single_3, "
                         f"niah_multikey_1, niah_multikey_2, niah_multikey_3")
    task_fn = getattr(_niah, task_name)

    seqlen = int(args.needle_seqlen)
    n_sample = int(args.needle_n_sample)
    # niah's generate_input_output only seeds the needles SHUFFLE
    # (random.Random(random_seed).shuffle); the actual magic-number /
    # word / depth / insertion picks use the *global* random module.
    # By the time we reach this function the global RNG has been advanced
    # by LlamaEvaluator init / dataloader / arch sampling and the niah
    # data would silently differ across archs. Re-seed here so every arch
    # in a sweep sees IDENTICAL needle prompts — necessary for the eval
    # to be a clean comparison across the rows of archs.csv.
    _random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    print(f"[needle_nll] generating {n_sample} {task_name} prompts "
          f"at seqlen={seqlen} (seed={args.seed}) …")
    t0 = time()
    data = task_fn(model=model_id, max_seq_lengths=[seqlen],
                   num_samples=n_sample)['test']
    print(f"[needle_nll] prompt generation took {time() - t0:.1f}s "
          f"({len(data)} samples)")

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    total_tokens = 0
    device = model.device
    pbar = tqdm(data, desc='needle_nll')
    for ex in pbar:
        prompt = ex['input'] + ' ' + ex['gen_prefix']
        answer = ex['outputs'][0] if isinstance(ex['outputs'], list) else ex['outputs']
        enc_prompt = tokenizer(prompt, return_tensors='pt',
                               add_special_tokens=True).input_ids
        enc_full = tokenizer(prompt + ' ' + answer, return_tensors='pt',
                             add_special_tokens=True).input_ids
        p_len = enc_prompt.shape[1]
        if enc_full.shape[1] <= p_len:
            continue
        input_ids = enc_full.to(device)
        labels = input_ids.clone()
        labels[:, :p_len] = -100
        with torch.inference_mode():
            logits = model(input_ids).logits
        shift_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = loss_fn(shift_logits.float(), shift_labels)
        valid = (shift_labels != -100).sum().item()
        if valid > 0:
            total_loss += loss.item()
            total_tokens += valid
        pbar.set_postfix(nll=f"{total_loss / max(total_tokens, 1):.4f}",
                         toks=total_tokens)
    return total_loss / max(total_tokens, 1)


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
    ctx = init_run(args)
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
                    value = _run_needle_nll(args, ctx, evaluator.model, model_id)
                elif kind == 'jsd_custom':
                    value = _run_jsd_custom(
                        args, ctx, evaluator, dataset,
                        unpad=eval_kwargs.get('unpad', False),
                        stream_dense=eval_kwargs.get('stream_dense', False))
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
    p.add_argument('--kv_method', type=str, default='kivi',
                   choices=['fp16', 'hqq', 'kivi'])
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
