import argparse
import os
import shutil
import json
import torch
import warnings
from time import time, strftime
from accelerate import Accelerator
from utils.func import init_accelerator, set_seed, get_hfmodel, clean_up
from utils.data import (get_loader, get_tokenizer, chat_overhead,
                        key_token_dirname, CHAT_PREFIX, CHATDOC_PREFIX)
from utils.loss import get_key_token_list

warnings.simplefilter("ignore")

def protocol_from_metrics(names, target_id=None):
    """The archive protocol the named correlation.py metrics need.

    An archive is a function of (corpus, n_sample, seqlen, min_seqlen, trunc_len,
    sliding_window, alpha, beta, data_seed) — NOT of last_tokens / stride /
    prefill_prompt / score, which are eval-time knobs (MEASURED: the same
    archive serves last_tokens 1024/512/256/128 and both score modes). So
    metrics that differ only there share one archive, and asking for them
    together is the point of this flag.
    """
    from utils.metric_specs import METRIC_TASKS, GROUPS
    KEYS = ('n_sample', 'seqlen', 'min_seqlen', 'trunc_len',
            'sliding_window', 'alpha', 'beta', 'data_seed')
    by_name = {t[0]: t for t in METRIC_TASKS}
    protos = {}
    for n in names:
        if n not in by_name:
            raise SystemExit(f"unknown metric '{n}'. Registered key-token metrics: "
                             + ', '.join(t[0] for t in METRIC_TASKS
                                         if GROUPS[t[1]].get('use_key_token')))
        _, group, ds, _kw = by_name[n]
        g = GROUPS[group]
        if not g.get('use_key_token'):
            raise SystemExit(f"metric '{n}' does not use key tokens (group {group})")
        protos.setdefault((ds,) + tuple(g.get(k) for k in KEYS), []).append(n)
    if len(protos) > 1:
        lines = [f"  {p[0]}: " + ', '.join(f'{k}={v}' for k, v in zip(KEYS, p[1:]))
                 + f"   <- {ns}" for p, ns in protos.items()]
        raise SystemExit("these metrics need DIFFERENT archives; generate them one "
                         "protocol at a time:\n" + "\n".join(lines))
    proto, names = next(iter(protos.items()))
    out = dict(zip(KEYS, proto[1:])); out['dataset'] = proto[0]
    ds_seed = out.pop('data_seed')
    out['seed'] = 0 if ds_seed is None else int(ds_seed)

    # A chat: group wraps every sample in one user turn, and the affixes come out
    # of the DOCUMENT budget -- so the archive is computed on `chatdoc:<corpus>`
    # at seqlen - overhead, not on the group's TOTAL seqlen. The overhead is the
    # TARGET's (its template, its tokenizer), which is why this needs
    # --target_model. For document corpora chatdoc: is the raw loader at that
    # length; for the stream corpora (wikitext2/c4) it is NOT -- the chat body is
    # built with add_special_tokens=False and differs from raw by the BOS.
    if str(out['dataset']).startswith(CHAT_PREFIX):
        g0 = GROUPS[by_name[names[0]][1]]
        if g0.get('key_token_on_sample'):
            # The archive is the WHOLE chat sample -- the thing actually
            # measured -- so the loader stays chat: at the group's full seqlen,
            # and the answer window is part of the input (it moves `post`).
            out['answer_tokens'] = g0.get('last_tokens') or 0
            print(f"[gen_key_token] chat group -> the chat SAMPLE itself "
                  f"(seqlen {out['seqlen']}, answer_tokens {out['answer_tokens']})")
        else:
            if not target_id:
                raise SystemExit(f"{names} are chat: groups — pass --target_model so "
                                 f"the chat affix overhead can be resolved")
            ov = chat_overhead(get_tokenizer(target_id, use_fast=True))
            out['dataset'] = CHATDOC_PREFIX + out['dataset'].split(':', 1)[1]
            out['seqlen'] = int(out['seqlen']) - ov
            out['min_seqlen'] = max(0, int(out['min_seqlen']) - ov) if out['min_seqlen'] else 0
            print(f"[gen_key_token] chat group -> document budget {out['seqlen']} "
                  f"(total - {ov} affix tokens)")
    return out, names


def main(args):
    if args.metrics:
        _tgt = (f'{args.target_model_path}/{args.target_model}'
                if args.target_model_path else args.target_model) if args.target_model else None
        proto, served = protocol_from_metrics(args.metrics, _tgt)
        for k, v in proto.items():
            if v is not None:
                setattr(args, k, v)
        # every key-token group is a LOSS-side group -> train loader
        args.train = True
        print(f"[gen_key_token] --metrics {served} -> {proto}")

    set_seed(args.seed)

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    
    # Initialize accelerator
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)
    
    # Load model and tokenizer
    model_id = f'{args.model_path}/{args.model_name}' if args.model_path else args.model_name
    accelerator.print(f"Loading model from: {model_id}")
    
    model = get_hfmodel(
        model_id, 
        dtype=args.dtype, 
        device_map=device_map
    )
    tokenizer = get_tokenizer(model_id, use_fast=True)
    # tokenizer = get_tokenizer(model_id)

    # ── two tokenizer ROLES, one model each ──
    # evaluator (this model): re-tokenizes the TEXT inside find_key_token and
    #   decides which tokens are key. Never sees anyone else's ids.
    # target   (--target_model): owns the LOADER — which documents clear
    #   min_seqlen and where seqlen truncates — plus the decode that produces
    #   the text and the offset_mapping the character intervals are mapped back
    #   onto. That is the text correlation.py/post_search will actually measure.
    # Leaving --target_model empty keeps the old behaviour (evaluator does both),
    # which makes the archive depend on the evaluator's tokenization: the
    # shipped gov_report archives were cut ~2-6% shorter than the Llama loader
    # cuts, and only survived because the shorter text is a PREFIX of the longer.
    target_id = (f'{args.target_model_path}/{args.target_model}'
                 if args.target_model_path else args.target_model) \
        if args.target_model else model_id
    target_tokenizer = (get_tokenizer(target_id, use_fast=True)
                        if target_id != model_id else tokenizer)
    accelerator.print(f"evaluator: {model_id}\n     target: {target_id}"
                      + ("  (same — archive follows the EVALUATOR's tokenization)"
                         if target_id == model_id else ""))

    # Create data loader
    accelerator.print(f"Creating data loader for dataset: {args.dataset}")
    # answer_tokens is part of the INPUT for a chat: sample (it moves the
    # assistant header, and with it `post`), so an on-sample archive is specific
    # to one window. --metrics sets it; it is inert for the raw corpora.
    _ans = int(getattr(args, 'answer_tokens', 0) or 0)
    loader = get_loader(
        args.dataset,
        model=target_id,
        n_sample=args.n_sample,
        batch_size=args.data_batch_size,
        train=args.train,
        seed=args.seed,
        seqlen=args.seqlen,
        min_seqlen=args.min_seqlen,
        **({'answer_tokens': _ans} if _ans and str(args.dataset).startswith('chat:') else {})
    )
    loader = accelerator.prepare(loader)
    
    # Prepare save path
    if args.save_path:
        save_path = args.save_path
    else:
        # Generate default save path based on parameters
        save_path = f"key_token/{args.model_name}_{args.n_sample}sample_{args.seqlen}seqlen_{args.min_seqlen}min_{args.trunc_len}trunc_{args.sliding_window}sw_{args.alpha}alpha_{args.beta}beta"
    
    # Write into a staging directory and swap it in only once the whole run
    # succeeded. Writing in place destroys the archive that is already there the
    # moment anything fails: the 16384 run died on document 20 of 50 with a CUDA
    # error and left slices 0-19 from the new run next to 20-49 from the old one,
    # under the OLD manifest -- a mixture nothing can detect except by hash.
    from utils.data import key_token_corpus
    # filed under the BASE corpus so the consumer (evaluator -> key_token_corpus)
    # finds it at <key_token_path>/<corpus> for chat:/chat:/chatdoc: alike
    _dirname = key_token_dirname(args.dataset, args.n_sample, args.seqlen,
                                 args.min_seqlen, args.trunc_len,
                                 args.sliding_window, args.alpha, args.beta,
                                 args.seed)
    final_save_path = os.path.join(save_path, _dirname)
    staging_root = save_path.rstrip('/') + '.partial'
    # A chatdoc:/chat: dataset is filed under its BASE corpus so the consumer
    # (evaluator -> key_token_corpus) finds it at <key_token_path>/<corpus>.
    dataset_save_path = os.path.join(staging_root, _dirname)
    protocol = dict(evaluator=model_id, target=target_id, dataset=args.dataset,
                    n_sample=args.n_sample, seqlen=args.seqlen,
                    min_seqlen=args.min_seqlen, trunc_len=args.trunc_len,
                    sliding_window=args.sliding_window, alpha=args.alpha,
                    beta=args.beta, seed=args.seed, train=bool(args.train))
    if accelerator.is_main_process:
        # The lock is checked BEFORE anything is deleted: a run without
        # --resume used to wipe the staging directory (lock included) and only
        # then look for the lock, so it happily destroyed a live run's work.
        lock_path = os.path.join(staging_root, 'lock.json')
        if os.path.exists(lock_path):
            try:
                with open(lock_path) as f:
                    other = json.load(f)
                os.kill(int(other['pid']), 0)
                alive = True
            except Exception:
                alive = False
            if alive:
                raise RuntimeError(
                    f"another generation (pid {other.get('pid')}, started "
                    f"{other.get('started')}) is already writing {staging_root}. "
                    f"Wait for it or use a different --save_path.")
        stale = os.path.exists(staging_root)
        if stale and args.resume:
            # only reuse staged slices that were produced by THIS protocol
            ppath = os.path.join(staging_root, 'protocol.json')
            try:
                with open(ppath) as f:
                    stale = json.load(f) != protocol
            except Exception:
                stale = True
            if stale:
                accelerator.print(f"[key_token] {staging_root} is from a different "
                                  f"protocol — discarding it")
        if stale:
            shutil.rmtree(staging_root)
        os.makedirs(dataset_save_path, exist_ok=True)
        with open(os.path.join(staging_root, 'protocol.json'), 'w') as f:
            json.dump(protocol, f, indent=2)
        with open(lock_path, 'w') as f:
            json.dump(dict(pid=os.getpid(), started=strftime('%Y-%m-%d %H:%M:%S')), f)
    accelerator.wait_for_everyone()

    accelerator.print(f"Saving key tokens to: {dataset_save_path}"
                      f"\n  (swapped into {final_save_path} on success)")
    
    # Generate key token list
    accelerator.print("Generating key token list...")
    start_time = time()
    key_token_list = get_key_token_list(
        evaluator_model=model,
        evaluator_tokenizer=tokenizer,
        loader=loader,
        tokenizer=target_tokenizer,
        trunc_len=args.trunc_len,
        sliding_window=args.sliding_window,
        alpha=args.alpha,
        beta=args.beta,
        save_path=dataset_save_path,
        mode='online',
        resume=args.resume,
        verbosity=args.verbosity,
        manifest_meta=dict(evaluator_model=model_id, target_model=target_id,
                           dataset=args.dataset, train=bool(args.train),
                           n_sample=args.n_sample, seqlen=args.seqlen,
                           min_seqlen=args.min_seqlen, seed=args.seed,
                           data_batch_size=args.data_batch_size,
                           answer_tokens=_ans)
    )
    end_time = time()
    accelerator.print(f"Time taken to generate key token list: {(end_time - start_time):.2f} seconds")
    
    # Count total key tokens
    # key_token_list is NESTED [batch][seq] (see utils/loss.get_key_token_list)
    flat_key_tokens = [k for batch in key_token_list for k in batch]
    n_key_token = sum(len(k) if k is not None else 0 for k in flat_key_tokens)
    n_key_token = sum(accelerator.gather_for_metrics([n_key_token], use_gather_object=True))
    
    accelerator.print(f'Dataset: {args.dataset}, Total key tokens: {n_key_token}')
    
    # Decode key tokens back to text and print
    accelerator.print("Decoding key tokens to text...")
    _slice_base = 0
    for batch_idx, (inputs, attention_mask, labels) in enumerate(loader):
        if _slice_base >= len(flat_key_tokens):
            break

        # running counter, NOT batch_idx * bs: the last batch is short whenever
        # n_sample % batch_size != 0, and `bs` below is that short size.
        cur_bs = inputs.shape[0]          # true size of THIS batch (last one is short)
        batch_key_tokens = flat_key_tokens[_slice_base:_slice_base + cur_bs]
        for seq_idx in range(len(batch_key_tokens)):
            slice_idx = _slice_base + seq_idx
            key_tokens = batch_key_tokens[seq_idx]
            if key_tokens is None or len(key_tokens) == 0:
                continue

            input_ids = inputs[seq_idx]
            if attention_mask is not None:
                mask = attention_mask[seq_idx]
                actual_length = mask.sum().item()
                input_ids = input_ids[:actual_length]

            # key_tokens are indices for shift_logits (predicting token at idx+1)
            # So actual input_ids index is idx + 1
            # Filter valid indices and convert to input_ids indices
            key_token_ids = []
            for kt in key_tokens:
                actual_idx = kt + 1  # Convert from shift_logits index to input_ids index
                if 0 <= actual_idx < input_ids.shape[0]:
                    key_token_ids.append(input_ids[actual_idx].item())
            
            if len(key_token_ids) == 0:
                continue

            key_text = tokenizer.decode(key_token_ids, skip_special_tokens=True)

            # accelerator.print(f"[batch {batch_idx} seq {seq_idx}] key token indices (shift_logits): {key_tokens}")
            # accelerator.print(f"[batch {batch_idx} seq {seq_idx}] key token indices (input_ids): {valid_indices}")
            accelerator.print(f"[Slice {slice_idx}] key token text: {key_text if len(key_text) < 200 else key_text[:200] + '...'}")

        _slice_base += cur_bs

    # ── swap the finished archive in ───────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        n_written = len([f for f in os.listdir(dataset_save_path)
                         if f.startswith('slice_') and f.endswith('.txt')])
        expected = len(loader.dataset) if hasattr(loader, 'dataset') else n_written
        if n_written != expected:
            raise RuntimeError(
                f"staging holds {n_written} slices but the loader has {expected} "
                f"documents — refusing to replace {final_save_path}")
        if not os.path.exists(os.path.join(dataset_save_path, 'meta.json')):
            raise RuntimeError(
                f"staging has no meta.json — refusing to replace {final_save_path}")
        backup = None
        if os.path.exists(final_save_path):
            backup = final_save_path + '.replaced'
            if os.path.exists(backup):
                shutil.rmtree(backup)
            os.replace(final_save_path, backup)
        os.makedirs(os.path.dirname(final_save_path) or '.', exist_ok=True)
        os.replace(dataset_save_path, final_save_path)
        shutil.rmtree(staging_root, ignore_errors=True)
        if backup:
            shutil.rmtree(backup, ignore_errors=True)
        accelerator.print(f"[key_token] archive complete: {final_save_path} "
                          f"({n_written} slices)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate key token list from model')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to model directory')
    parser.add_argument('--model_name', type=str, default='',
                        help='model name (e.g., Llama-3.1-8B-Instruct)')
    parser.add_argument('--target_model', type=str, default='',
                        help='model whose TOKENIZER owns the loader (document '
                             'selection + seqlen truncation), the decode and the '
                             'offset_mapping — i.e. the model these key tokens '
                             'will be USED with. The evaluator still judges with '
                             'its own tokenizer (find_key_token re-tokenizes the '
                             'text). Empty = evaluator does both (legacy).')
    parser.add_argument('--target_model_path', type=str, default='',
                        help='directory holding --target_model')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset name (wikitext2, c4, gsm8k, etc.)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--config', type=str, default='',
                        help='path to config json file')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='number of samples to process')
    parser.add_argument('--data_batch_size', type=int, default=1,
                        help='batch size for data loader')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='sequence length')
    parser.add_argument('--min_seqlen', type=int, default=0,
                        help='minimum sequence length')
    parser.add_argument('--trunc_len', type=int, default=512,
                        help='truncation length for long PPL/JSD calculation')
    parser.add_argument('--sliding_window', type=int, default=128,
                        help='sliding_window length for long PPL/JSD calculation')
    parser.add_argument('--alpha', type=int, default=2,
                        help='Long-short distance (LSD) threshold for long PPL/JSD calculation')
    parser.add_argument('--beta', type=int, default=-2,
                        help='Long context likelihood (LCL) threshold for long PPL/JSD calculation')
    parser.add_argument('--save_path', type=str, default='',
                        help='path to save key tokens (default: auto-generated)')
    # parser.add_argument('--load_path', type=str, default='',
    #                     help='path to load precomputed key tokens (for offline mode)')
    # parser.add_argument('--save_list', action='store_true',
    #                     help='save key_token_list as pickle file')
    parser.add_argument('--dtype', type=str, default='auto',
                        help='model dtype (auto, float16, bfloat16, etc.)')
    parser.add_argument('--verbosity', action='store_true',
                        help='')
    parser.add_argument('--train', action='store_true',
                        help='')
    parser.add_argument('--answer_tokens', type=int, default=0,
                        help='chat: answer window — part of the INPUT (it moves the '
                             'assistant header), so an on-sample archive is specific '
                             'to one value. --metrics sets it automatically.')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='derive the archive protocol from correlation.py metric '
                             'names (e.g. gov_jsd_kt gov_jsd_kt_s512 gov_jsd_kt_pp512_s128). '
                             'Overrides --dataset/--n_sample/--seqlen/--min_seqlen/'
                             '--trunc_len/--sliding_window/--alpha/--beta/--seed. '
                             'Metrics that differ only in last_tokens/stride/score '
                             'share ONE archive; mixing protocols is an error.')
    parser.add_argument('--resume', action='store_true',
                        help='reuse slices already staged in <save_path>.partial '
                             'from an interrupted run with the same protocol')
    
    args = parser.parse_args()
    main(args)
