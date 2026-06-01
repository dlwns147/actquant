#!/usr/bin/env bash
# Usage: bash scripts/surrogate/sample.sh <SAVE> <METHOD> [<ROUND> <BATCH>]
#   SAVE     output dir (archs.csv / sample_meta.json / result_*.json land here)
#   METHOD   quantile | random | ga | al_ei | validation
#   ROUND    (default 0) round_id stored on the new rows
#   BATCH    (default $N_EXTRAS for random/ga, $AL_BATCH for al_ei,
#            $N_VAL for validation; ignored for quantile)
#
# CPU-only (no GPU acquired). Safe to run on the login node.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
source "${HERE}/_config.sh"
cd "${ROOT}"

SAVE=${1:?"need <SAVE>"}
METHOD=${2:?"need <METHOD>  (quantile|random|ga|al_ei|validation)"}
ROUND=${3:-0}
BATCH_ARG=${4:-}

mkdir -p "${SAVE}"

ARGS="--mode sample --save ${SAVE} --round ${ROUND}"
ARGS+=" $(build_model_args)"
ARGS+=" $(build_sample_args)"

case "${METHOD}" in
    quantile)
        ARGS+=" --method quantile --quantile_sample ${QUANTILE_SAMPLE}"
        ;;
    random)
        BATCH=${BATCH_ARG:-${N_EXTRAS}}
        ARGS+=" --method random --batch ${BATCH}"
        ;;
    ga)
        BATCH=${BATCH_ARG:-${N_EXTRAS}}
        ARGS+=" --method ga --batch ${BATCH}"
        ;;
    al_ei)
        BATCH=${BATCH_ARG:-${AL_BATCH}}
        # al_ei needs the calibration y-column to refit on completed results
        ARGS+=" --method al_ei --batch ${BATCH} --datasets ${DATASETS}"
        ;;
    validation)
        NV=${BATCH_ARG:-${N_VAL}}
        ARGS+=" --validation --n_val ${NV} --val_seed ${VAL_SEED}"
        ;;
    *)
        echo "ERROR: unknown METHOD '${METHOD}'"
        exit 1
        ;;
esac

echo "[sample.sh] SAVE=${SAVE}  METHOD=${METHOD}  ROUND=${ROUND}"
python surrogate_pipeline.py ${ARGS}
