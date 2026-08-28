"""Make an existing key-token archive verifiable, without re-running the evaluator.

The shipped archives predate meta.json, so nothing checks that slice_<i>.txt
belongs to the document a loader hands over — a wrong archive loads silently.
Regenerating them needs the 72B evaluator's weights; the manifest does not. Its
TOKENIZER is enough to rebuild the exact text those intervals were computed on
(gen_key_token used the evaluator for the loader too, which is what made the
archive text a PREFIX of the target's), hash it, and record the protocol.

Every slice is validated before anything is written: the intervals must fit
inside the reconstructed text AND their boundaries must fall on that
tokenizer's token boundaries — they came from its offset_mapping, so a wrong
reconstruction cannot line up.

Usage:  python retrofit_key_token_manifest.py key_token/<archive> [--apply]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

from utils.loss import load_key_token, write_key_token_manifest   # noqa: E402

NAME_RE = re.compile(
    r'^(?P<evaluator>.+?)_(?P<dataset>[a-z0-9_]+?)_(?P<split>train|test|validation)_'
    r'(?P<n_sample>\d+)sample_(?P<seqlen>\d+)seqlen_(?P<min_seqlen>\d+)min_'
    r'(?P<trunc_len>\d+)trunc_(?P<sliding_window>\d+)sw_'
    r'(?P<alpha>-?\d+)alpha_(?P<beta>-?\d+)beta$')


def parse_archive_name(path):
    m = NAME_RE.match(os.path.basename(path.rstrip('/')))
    if not m:
        raise SystemExit(f"cannot parse the protocol out of {path!r}; the "
                         f"directory name is the only record a pre-manifest "
                         f"archive has.")
    d = m.groupdict()
    for k in ('n_sample', 'seqlen', 'min_seqlen', 'trunc_len',
              'sliding_window', 'alpha', 'beta'):
        d[k] = int(d[k])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive')
    ap.add_argument('--evaluator_path', default='Qwen',
                    help='HF org/dir holding the evaluator tokenizer')
    ap.add_argument('--apply', action='store_true',
                    help='write meta.json (default: validate only)')
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from utils.data import get_loader

    p = parse_archive_name(args.archive)
    ev_id = f"{args.evaluator_path}/{p['evaluator']}"
    slice_dir = os.path.join(args.archive, p['dataset'])
    print(f"archive   {args.archive}\nprotocol  {p}\nevaluator {ev_id}")

    tok = AutoTokenizer.from_pretrained(ev_id, use_fast=True)
    # the legacy generator built the loader with the EVALUATOR's tokenizer
    loader = get_loader(p['dataset'], model=ev_id, n_sample=p['n_sample'],
                        batch_size=1, train=(p['split'] != 'test'), seed=0,
                        seqlen=p['seqlen'], min_seqlen=p['min_seqlen'])

    texts, bad = [], []
    for i, (ids, attn, _) in enumerate(loader):
        text = tok.decode(ids[0][:int(attn[0].sum())], skip_special_tokens=True)
        texts.append(text)
        f = os.path.join(slice_dir, f'slice_{i}.txt')
        if not os.path.exists(f):
            bad.append((i, 'missing slice file')); continue
        iv = load_key_token(f) or []
        if not iv:
            continue
        if max(b for _, b in iv) > len(text):
            bad.append((i, f'intervals reach {max(b for _, b in iv)} > text '
                           f'{len(text)} chars')); continue
        # boundaries must be token boundaries of THIS tokenization
        off = tok(text, return_offsets_mapping=True)['offset_mapping']
        edges = {int(a) for a, _ in off} | {int(b) for _, b in off}
        miss = [(a, b) for a, b in iv if a not in edges or b not in edges]
        if miss:
            bad.append((i, f'{len(miss)}/{len(iv)} interval edges are not token '
                           f'boundaries, e.g. {miss[:2]}'))
    n_sl = len(os.listdir(slice_dir)) if os.path.isdir(slice_dir) else 0
    print(f"slices    {n_sl} files / {len(texts)} loader docs")
    if bad:
        for i, why in bad[:5]:
            print(f"  slice_{i}: {why}")
        raise SystemExit(f"VALIDATION FAILED on {len(bad)} slice(s) — the "
                         f"reconstruction does not match; not writing anything.")
    print("validated : every interval fits the reconstructed text and lands on "
          "its token boundaries")
    if not args.apply:
        print("(dry run — pass --apply to write meta.json)")
        return
    write_key_token_manifest(slice_dir, texts, dict(
        evaluator_model=ev_id, target_model=ev_id, dataset=p['dataset'],
        train=(p['split'] != 'test'), n_sample=p['n_sample'],
        seqlen=p['seqlen'], min_seqlen=p['min_seqlen'], seed=0,
        trunc_len=p['trunc_len'], sliding_window=p['sliding_window'],
        alpha=p['alpha'], beta=p['beta'], data_batch_size=1,
        retrofitted=True,
        note=('manifest reconstructed from the archive name + the evaluator '
              'tokenizer; the evaluator also owned the loader, so this text is '
              'a PREFIX of what a target-tokenizer loader produces')))
    print(f"wrote {os.path.join(slice_dir, 'meta.json')}")


if __name__ == '__main__':
    main()
