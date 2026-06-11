#!/usr/bin/env bash
# Usage: bash scripts/surrogate/aggregate.sh <SAVE>
# Walks <SAVE>/{archs.csv, result_*.json, validation_archs.csv,
# validation_result_*.json} and writes:
#   - results_<method>.csv     (post_search.load_sample_csv compat)
#   - validation_metrics.csv   (per-method R²/Spearman/Kendall on val pool)
#   - learning_curve.csv       (AL only; per-round metrics on val pool)
# CPU-only; safe to run mid-sweep (in-progress archs are NaN'd out).

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
source "${HERE}/_config.sh"
cd "${ROOT}"

SAVE=${1:?"need <SAVE>"}

ARGS="--mode aggregate --save ${SAVE}"
ARGS+=" $(build_model_args)"
ARGS+=" --datasets ${DATASETS} --metric ${METRIC} --loss_func ${LOSS_FUNC}"
ARGS+=" --surrogate ${SURROGATE} --ard_kernel ${ARD_KERNEL} --gp_n_restarts ${GP_N_RESTARTS}"
# region split of the val pool (validation_region_metrics.csv)
ARGS+=" --val_bands ${VAL_BANDS:-6}"
[ -n "${VAL_BAND_OBJ:-}" ] && ARGS+=" --val_band_obj ${VAL_BAND_OBJ}"

python surrogate_pipeline.py ${ARGS}
