"""Pre-download all LongBench dataset configs used by utils/longbench.py.

longbench.pred_longbench() calls
    load_dataset('THUDM/LongBench', <config>, split='test')
at run time. Running this script once populates the HuggingFace datasets
cache so the search / post_search run can work fully offline.

Usage:
    python -m utils.prefetch_longbench              # both e=True and e=False sets
    python utils/prefetch_longbench.py --only-e     # only the LongBench-E (`_e`) configs
    python utils/prefetch_longbench.py --only-full  # only the non-e configs

After prefetching you can run the actual job offline:
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 accelerate launch post_search.py ...
"""
import argparse
from datasets import load_dataset

# Mirrors longbench.pred_longbench() exactly.
DATASETS_E = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
]
DATASETS_FULL = [
    "triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p",
    "qmsum", "multi_news",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-e", action="store_true",
                        help="prefetch only the LongBench-E (`_e`) configs")
    parser.add_argument("--only-full", action="store_true",
                        help="prefetch only the non-e configs")
    args = parser.parse_args()

    configs = []
    if not args.only_full:
        configs += [f"{d}_e" for d in DATASETS_E]
    if not args.only_e:
        configs += list(DATASETS_FULL)

    # de-dup while preserving order
    seen = set()
    configs = [c for c in configs if not (c in seen or seen.add(c))]

    print(f"Prefetching {len(configs)} LongBench configs into HF cache...")
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] THUDM/LongBench :: {cfg}")
        ds = load_dataset("THUDM/LongBench", cfg, split="test")
        print(f"    -> {len(ds)} rows cached")
    print("Done. Re-run jobs with HF_DATASETS_OFFLINE=1 to use the cache.")


if __name__ == "__main__":
    main()
