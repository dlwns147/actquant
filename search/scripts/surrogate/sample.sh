#!/usr/bin/env bash
# Usage: bash scripts/surrogate/sample.sh <SAVE> <METHOD> [<ROUND> <BATCH>]
#   SAVE     output dir (archs.csv / sample_meta.json / result_*.json land here)
#   METHOD   quantile | random | ga | validation
#            | al_ei (back-compat) | al  (uses $ACQ from _config.sh)
#            | al_<acq>  where acq ∈ ei ucb alm imse maximin qbc rank
#   ROUND    (default 0) round_id stored on the new rows
#   BATCH    (default $N_EXTRAS for random/ga, $AL_BATCH for al/al_*,
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
    maximin)
        # validated best global-representation sampler (model-free coverage)
        BATCH=${BATCH_ARG:-${N_EXTRAS}}
        ARGS+=" --method maximin --batch ${BATCH}"
        ;;
    al_ei)
        BATCH=${BATCH_ARG:-${AL_BATCH}}
        # al_ei needs the calibration y-column to refit on completed results
        ARGS+=" --method al_ei --batch ${BATCH} --datasets ${DATASETS}"
        ;;
    ga_imse)
        # improvements 1+2+3: GA-coverage + front/sqrt IMSE-refine
        BATCH=${BATCH_ARG:-${AL_BATCH}}
        ARGS+=" --method ga_imse --batch ${BATCH} --datasets ${DATASETS}"
        ;;
    al_fimse)
        # front+sqrt IMSE (improvements 1+2 on standalone imse; AL_CAND/AL_TRANSFORM
        # from _config supply front/sqrt). Distinct dir from raw al_imse.
        BATCH=${BATCH_ARG:-${AL_BATCH}}
        ARGS+=" --method al --acq imse --batch ${BATCH} --datasets ${DATASETS}"
        ;;
    al|al_*)
        # al → use $ACQ from _config.sh; al_<acq> → that acq explicitly.
        if [ "${METHOD}" = "al" ]; then ACQ_NAME=${ACQ}; else ACQ_NAME=${METHOD#al_}; fi
        BATCH=${BATCH_ARG:-${AL_BATCH}}
        ARGS+=" --method al --acq ${ACQ_NAME} --batch ${BATCH} --datasets ${DATASETS}"
        ;;
    validation)
        NV=${BATCH_ARG:-${N_VAL}}
        ARGS+=" --validation --n_val ${NV} --val_seed ${VAL_SEED}"
        ;;
    validation_band)
        # stratified-uniform: NV split evenly over VAL_BANDS bands of the
        # band comp axis (equal per-band precision; random val is bulk-weighted)
        NV=${BATCH_ARG:-${N_VAL}}
        ARGS+=" --validation --val_method band --n_val ${NV} --val_seed ${VAL_SEED}"
        ARGS+=" --val_bands ${VAL_BANDS:-6}"
        [ -n "${VAL_BAND_OBJ:-}" ] && ARGS+=" --val_band_obj ${VAL_BAND_OBJ}"
        ;;
    validation_front)
        # per-(wbits×band)-cell argmin-predicted picks = the post_search
        # selection locus ("Pareto-front combos")
        NV=${BATCH_ARG:-${N_VAL}}
        ARGS+=" --validation --val_method front --n_val ${NV} --val_seed ${VAL_SEED}"
        ARGS+=" --val_bands ${VAL_BANDS:-6} --val_front_wbands ${VAL_FRONT_WBANDS:-4}"
        [ -n "${VAL_BAND_OBJ:-}" ] && ARGS+=" --val_band_obj ${VAL_BAND_OBJ}"
        ;;
    *)
        echo "ERROR: unknown METHOD '${METHOD}'"
        exit 1
        ;;
esac

echo "[sample.sh] SAVE=${SAVE}  METHOD=${METHOD}  ROUND=${ROUND}"
python surrogate_pipeline.py ${ARGS}
