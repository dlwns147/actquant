# LongBench data (NOT in git — fetch once per cluster)

`<config>.jsonl` files from `data.zip` of the LongBench dataset repo
(https://huggingface.co/datasets/zai-org/LongBench, formerly
`THUDM/LongBench`; `data/<config>.jsonl` inside the zip, byte-identical, laid
FLAT here because `search/.gitignore` swallows any directory literally named
`data/`). The jsonls themselves are gitignored — populate them by running

    python utils/fetch_offline_data.py --only longbench

once on a machine with internet (e.g. the slurm login node). The repo checkout
is what the offline containers mount, so after that every container sees the
files — no HF-cache baking, no image rebuild.

Consumed local-first by `utils/data.py::load_longbench_split` (used by both
`get_longbench_ppl` — the `longbench:<subset>` PPL corpora — and
`utils/longbench.py::pred_longbench`, the benchmark). A config with no file
here falls back to `load_dataset('THUDM/LongBench', ...)`, which needs network
or an already-prepared HF cache and no longer works at all on `datasets>=3.0`
(script datasets were removed) — so the fix for a missing config is fetching
its jsonl here, never re-baking a docker image's cache.

Row order matters and is preserved: the hub script materialises the same jsonl
line by line (`_generate_examples`), and `.shuffle(seed)` permutes by
`(seed, num_rows)` only, so long-doc document selection (`LONG_DOC_DATA_SEED`)
is identical between the fetched and hub paths.

Fetched set = the union of what the code loads (~260 MB): the 8
`LONGBENCH_PPL_SUBSETS` + narrativeqa, the 8 `LONGBENCH_DATASETS` full configs
and the 13 `LONGBENCH_E_DATASETS` `_e` configs. The zh/unused configs were
left out; extend `LONGBENCH_FULL` in `utils/fetch_offline_data.py` if one ever
gets registered.
