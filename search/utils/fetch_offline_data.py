"""Fetch every vendored data payload the offline containers need — ONCE, on a
machine WITH internet (e.g. the slurm login node). The data files are NOT in
git (only this script and the loaders are): running this populates the repo
checkout in place, and since the checkout is what the containers mount, they
see the files with no HF-cache baking and no image rebuild.

    python utils/fetch_offline_data.py                 # everything missing
    python utils/fetch_offline_data.py --only longbench
    python utils/fetch_offline_data.py --only ruler minilongbench
    python utils/fetch_offline_data.py --force         # re-download

What lands where (all relative to this file, so cwd does not matter):
  longbench      utils/longbench_data/<config>.jsonl   (~260 MB, 24 configs)
                 read by utils/data.load_longbench_split → get_longbench_ppl
                 ('longbench:<subset>' PPL corpora) + longbench.pred_longbench
  ruler          utils/ruler_utils/paul_graham_essays.jsonl (~3 MB) +
                 utils/ruler_utils/nltk_data/tokenizers/punkt_tab/english
                 read by ruler_utils/prepare_niah.py (essay haystack, punkt).
                 SQuAD/HotpotQA JSONs are git-tracked already — not fetched.
  minilongbench  utils/minilongbench_data/data/<subset>.jsonl (~9 MB)
                 read by utils/minilongbench.py + utils/data.get_minilongbench

Stdlib-only on purpose (urllib/zipfile/json): login nodes may have no ML venv.
The one exception is the essays parquet, which prefers pyarrow/datasets when
present and otherwise falls back to the HF datasets-server rows API (still
stdlib). Row ORDER is preserved everywhere — LongBench document selection
(.shuffle(seed) = default_rng(seed).permutation(num_rows) over jsonl line
order) and the essay haystack word order depend on it.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# `python utils/fetch_offline_data.py` puts utils/ at sys.path[0], where
# utils/select.py shadows the stdlib `select` that urllib needs — drop it.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != HERE]

import argparse            # noqa: E402
import io                  # noqa: E402
import json                # noqa: E402
import shutil              # noqa: E402
import tempfile            # noqa: E402
import urllib.request      # noqa: E402
import zipfile             # noqa: E402

# ── LongBench ──────────────────────────────────────────────────────────────
# The union of what the code loads (keep in sync with utils/longbench.py
# LONGBENCH_DATASETS / LONGBENCH_E_DATASETS and utils/data.py
# LONGBENCH_PPL_SUBSETS; those modules import torch, so the lists are
# duplicated here to keep this script stdlib-only).
LONGBENCH_URL = ('https://huggingface.co/datasets/zai-org/LongBench/'
                 'resolve/main/data.zip')
LONGBENCH_FULL = ['narrativeqa', 'qmsum', 'gov_report', 'multifieldqa_en',
                  'qasper', 'multi_news', 'lcc', 'repobench-p',
                  'triviaqa', 'trec', 'samsum']
LONGBENCH_E = ['qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa',
               'gov_report', 'multi_news', 'trec', 'triviaqa', 'samsum',
               'passage_count', 'passage_retrieval_en', 'lcc', 'repobench-p']
LONGBENCH_CONFIGS = LONGBENCH_FULL + [f'{d}_e' for d in LONGBENCH_E]
LONGBENCH_DIR = os.path.join(HERE, 'longbench_data')

# ── RULER ──────────────────────────────────────────────────────────────────
ESSAYS_PARQUET_URL = ('https://huggingface.co/datasets/baber/'
                      'paul_graham_essays/resolve/main/essays.parquet')
ESSAYS_ROWS_API = ('https://datasets-server.huggingface.co/rows?dataset='
                   'baber%2Fpaul_graham_essays&config=default&split=train')
ESSAYS_OUT = os.path.join(HERE, 'ruler_utils', 'paul_graham_essays.jsonl')
PUNKT_URL = ('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/'
             'packages/tokenizers/punkt_tab.zip')
NLTK_DATA_DIR = os.path.join(HERE, 'ruler_utils', 'nltk_data')

# ── MiniLongBench ──────────────────────────────────────────────────────────
MINILB_SUBSETS = ['2wikimqa', 'dureader', 'gov_report', 'hotpotqa', 'lcc',
                  'lsht', 'multi_news', 'multifieldqa_en', 'multifieldqa_zh',
                  'musique', 'narrativeqa', 'passage_count',
                  'passage_retrieval_en', 'passage_retrieval_zh', 'qasper',
                  'qmsum', 'repobench-p', 'samsum', 'trec', 'triviaqa',
                  'vcsum']
MINILB_URL = ('https://huggingface.co/datasets/linggm/MiniLongBench/'
              'resolve/main/data/{name}.jsonl')
MINILB_DIR = os.path.join(HERE, 'minilongbench_data', 'data')


def _download(url, dest, desc):
    """Stream url to dest atomically (tmp file + rename)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'actquant-fetch'})
    with urllib.request.urlopen(req, timeout=120) as r:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest))
        try:
            with os.fdopen(fd, 'wb') as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            os.replace(tmp, dest)
        except BaseException:
            os.unlink(tmp)
            raise
    print(f'  {desc}: {os.path.getsize(dest):,} B  ← {url}')


def _count_lines(path):
    with open(path, 'rb') as f:
        return sum(1 for _ in f)


def fetch_longbench(force):
    missing = [c for c in LONGBENCH_CONFIGS
               if force or not os.path.isfile(
                   os.path.join(LONGBENCH_DIR, f'{c}.jsonl'))]
    if not missing:
        print(f'[longbench] all {len(LONGBENCH_CONFIGS)} configs present — skip')
        return
    print(f'[longbench] fetching data.zip for {len(missing)} config(s) …')
    os.makedirs(LONGBENCH_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix='.zip', dir=LONGBENCH_DIR)
    try:
        req = urllib.request.Request(LONGBENCH_URL,
                                     headers={'User-Agent': 'actquant-fetch'})
        with urllib.request.urlopen(req, timeout=300) as r, \
                os.fdopen(fd, 'wb') as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        with zipfile.ZipFile(tmp) as z:
            for c in missing:
                dest = os.path.join(LONGBENCH_DIR, f'{c}.jsonl')
                with z.open(f'data/{c}.jsonl') as src, open(dest, 'wb') as out:
                    shutil.copyfileobj(src, out)
                print(f'  {c}.jsonl: {os.path.getsize(dest):,} B, '
                      f'{_count_lines(dest)} rows')
    finally:
        os.unlink(tmp)


def _essay_texts():
    """The 218 essay texts, in dataset row order. pyarrow/datasets when
    available; else the HF datasets-server rows API (stdlib)."""
    fd, tmp = tempfile.mkstemp(suffix='.parquet')
    os.close(fd)
    try:
        _download(ESSAYS_PARQUET_URL, tmp, 'essays.parquet')
        try:
            import pyarrow.parquet as pq
            return pq.read_table(tmp).column('text').to_pylist()
        except ImportError:
            pass
        try:
            from datasets import load_dataset
            return load_dataset('parquet', data_files=tmp,
                                split='train')['text']
        except ImportError:
            pass
    finally:
        os.unlink(tmp)
    print('  (no pyarrow/datasets — falling back to the rows API)')
    texts, offset = [], 0
    while True:
        url = f'{ESSAYS_ROWS_API}&offset={offset}&length=100'
        req = urllib.request.Request(url,
                                     headers={'User-Agent': 'actquant-fetch'})
        with urllib.request.urlopen(req, timeout=120) as r:
            rows = json.load(r)['rows']
        if not rows:
            return texts
        assert [r['row_idx'] for r in rows] == list(
            range(offset, offset + len(rows))), 'rows API order broken'
        texts += [r['row']['text'] for r in rows]
        offset += len(rows)


def fetch_ruler(force):
    if force or not os.path.isfile(ESSAYS_OUT):
        print('[ruler] fetching Paul Graham essays …')
        texts = _essay_texts()
        with open(ESSAYS_OUT, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')
        print(f'  paul_graham_essays.jsonl: {len(texts)} essays, '
              f'{os.path.getsize(ESSAYS_OUT):,} B')
    else:
        print('[ruler] paul_graham_essays.jsonl present — skip')

    punkt_dir = os.path.join(NLTK_DATA_DIR, 'tokenizers', 'punkt_tab')
    if not force and os.path.isdir(os.path.join(punkt_dir, 'english')):
        print('[ruler] nltk punkt_tab/english present — skip')
        return
    print('[ruler] fetching nltk punkt_tab (keeping english only) …')
    req = urllib.request.Request(PUNKT_URL,
                                 headers={'User-Agent': 'actquant-fetch'})
    with urllib.request.urlopen(req, timeout=120) as r:
        buf = io.BytesIO(r.read())
    n = 0
    with zipfile.ZipFile(buf) as z:
        for name in z.namelist():
            # zip root is punkt_tab/; keep english/ + top-level README
            rel = name.split('punkt_tab/', 1)[-1]
            if not (rel.startswith('english/') or rel == 'README') or \
                    name.endswith('/'):
                continue
            dest = os.path.join(punkt_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(name) as src, open(dest, 'wb') as out:
                shutil.copyfileobj(src, out)
            n += 1
    print(f'  punkt_tab/english: {n} files → {punkt_dir}')


def fetch_minilongbench(force):
    missing = [s for s in MINILB_SUBSETS
               if force or not os.path.isfile(
                   os.path.join(MINILB_DIR, f'{s}.jsonl'))]
    if not missing:
        print(f'[minilongbench] all {len(MINILB_SUBSETS)} subsets present — skip')
        return
    print(f'[minilongbench] fetching {len(missing)} subset(s) …')
    for s in missing:
        _download(MINILB_URL.format(name=s),
                  os.path.join(MINILB_DIR, f'{s}.jsonl'), f'{s}.jsonl')


FETCHERS = {'longbench': fetch_longbench, 'ruler': fetch_ruler,
            'minilongbench': fetch_minilongbench}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(FETCHERS),
                    default=sorted(FETCHERS),
                    help='fetch only these payloads (default: all)')
    ap.add_argument('--force', action='store_true',
                    help='re-download even when the files exist')
    args = ap.parse_args()
    for name in args.only:
        FETCHERS[name](args.force)
    print('done.')


if __name__ == '__main__':
    sys.exit(main())
