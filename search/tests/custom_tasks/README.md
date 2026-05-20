# Custom tasks: `ifeval_pp` and `apps`

Two custom lm-evaluation-harness tasks. Location:
`/NAS/SJ/actquant/search/tests/custom_tasks`. lm-eval (`lm_eval==0.4.5`) is
used as the installed package; this dir is just the task definitions, loaded
via `--include_path`.

## Dependencies

`ifeval_pp` reuses the upstream IFEval instruction checker, which needs the
`lm_eval[ifeval]` extras (not pulled in by a bare lm-eval install):

```bash
pip install langdetect immutabledict "nltk>=3.9.1"
```

NLTK's `punkt_tab` is downloaded automatically by IFEval on first use. `apps`
has no extra deps beyond the standard lm-eval / datasets stack.

## Tasks

| Task | Dataset | Type | Metrics |
|------|---------|------|---------|
| `ifeval_pp` | [`RebeccaYU920/ifeval-pp`](https://huggingface.co/datasets/RebeccaYU920/ifeval-pp) (5410 rows, `train` split) | `generate_until` | `prompt_level_strict_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`, `inst_level_loose_acc` |
| `apps` | [`codeparrot/apps`](https://huggingface.co/datasets/codeparrot/apps) (`all`, `test` split) | `generate_until` | `strict_acc` (all test cases pass), `test_case_acc` (mean fraction of cases passed) |
| `apps_introductory` / `apps_interview` / `apps_competition` | APPS filtered by `difficulty` | `generate_until` | same as `apps` |

`ifeval_pp` is the perturbed/paraphrased IFEval variant. Its schema matches
IFEval, so scoring delegates to the upstream `lm_eval.tasks.ifeval` instruction
registry (no logic duplicated).

`apps` extracts the model's Python code (last ```python fenced block, else text
after `ANSWER:`) and runs it against every test case. Standard-input problems
run as a script with stdin/stdout comparison; call-based problems (`fn_name`)
run through a generated driver.

## Run with Llama 3.1 8B Instruct

Standalone Python runner (`run.py`) — depends only on `lm_eval` /
`transformers` / `datasets`, no actquant code:

```bash
cd /NAS/SJ/actquant/search/tests/custom_tasks
CUDA_VISIBLE_DEVICES=2,3 python run.py --limit 20    # quick smoke
CUDA_VISIBLE_DEVICES=2,3 python run.py               # full run
python run.py --tasks ifeval_pp --batch-size 8       # single task
```

Equivalent shell wrapper around the `lm_eval` CLI (`run_llama31_8b.sh`):

```bash
CUDA_VISIBLE_DEVICES=2,3 LIMIT=20 bash run_llama31_8b.sh
```

Or the `lm_eval` CLI directly:

```bash
python -m lm_eval --model hf \
  --model_args pretrained=/SSD/huggingface/meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16 \
  --tasks ifeval_pp,apps \
  --include_path /NAS/SJ/actquant/search/tests/custom_tasks \
  --apply_chat_template --batch_size auto --limit 20 --log_samples \
  --output_path /NAS/SJ/actquant/search/tests/custom_tasks/results
```

`--apply_chat_template` is required: Llama 3.1 8B is an instruct model and both
tasks depend on instruction-following behaviour.

## Tests

```bash
cd /NAS/SJ/actquant/search/tests/custom_tasks
python test_custom_tasks.py --offline                              # no network, fast
python test_custom_tasks.py                                        # + dataset-backed checks
CUDA_VISIBLE_DEVICES=2,3 RUN_MODEL=1 python test_custom_tasks.py    # + Llama smoke (slow)
```

## Security note

The `apps` task **executes model-generated Python**. Each run is isolated in a
separate process group with a wall-clock timeout and SIGKILL on timeout, but
this is not a hardened sandbox (no seccomp/network/fs jailing) — same trust
model as upstream APPS. Only run on a machine where this is acceptable.
