#!/usr/bin/env bash
# Usage: bash scripts/surrogate/al_driver.sh <SAVE_AL> [<N_ROUNDS> <AL_BATCH>]
#
# Login-node driver for the active-learning loop:
#   round 0: assumes <SAVE_AL>/archs.csv already has quantile anchors
#            (i.e. `sample.sh <SAVE_AL> quantile 0` was called first)
#            AND that round-0 SBATCH eval array has finished.
#   round R (1..N_ROUNDS): sample.sh al_ei R → sbatch --wait eval array
#                          → next round refits on the augmented pool.
#
# Uses `sbatch --wait` so the driver blocks until each round's array
# completes. Quit the driver = quit the AL loop (the previous batch's
# results are still saved and the next round can resume by reading
# completed result_*.json).
#
# This script must keep running until completion. Run inside `tmux`
# / `screen` / a long-lived shell.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
source "${HERE}/_config.sh"
cd "${ROOT}"

SAVE=${1:?"need <SAVE_AL>"}
N_ROUNDS=${2:-${AL_ROUNDS}}
BATCH=${3:-${AL_BATCH}}

if [ ! -f "${SAVE}/archs.csv" ]; then
    echo "ERROR: ${SAVE}/archs.csv missing — run sample.sh quantile + the"
    echo "       round-0 eval array first before launching the AL driver."
    exit 1
fi

# Sanity: at least one completed result_*.json so EI has training data.
N_DONE=$(ls "${SAVE}"/result_*.json 2>/dev/null | wc -l)
if [ "${N_DONE}" -lt 5 ]; then
    echo "ERROR: only ${N_DONE} completed result_*.json in ${SAVE}; need ≥5"
    echo "       for EI to have a stable train pool. Wait for the round-0"
    echo "       eval array to finish (or grow the quantile/seed pool)."
    exit 1
fi

for R in $(seq 1 "${N_ROUNDS}"); do
    echo "════════════════ AL round ${R}/${N_ROUNDS} ════════════════"

    # 1) pick BATCH new EI points and append to archs.csv (CPU-only)
    bash "${HERE}/sample.sh" "${SAVE}" al_ei "${R}" "${BATCH}"

    # 2) figure out the idx range we just appended
    LAST_IDX=$(awk -F, 'NR>1 {print $1}' "${SAVE}/archs.csv" | tail -1)
    FROM=$(( LAST_IDX - BATCH + 1 ))
    TO=${LAST_IDX}
    echo "[al_driver] round ${R}: submitting eval array ${FROM}-${TO}%${SLURM_ARRAY_CONCURRENCY}"

    # 3) sbatch --wait blocks until the entire array finishes
    sbatch --wait \
        --array=${FROM}-${TO}%${SLURM_ARRAY_CONCURRENCY} \
        "${HERE}/eval.sbatch" "${SAVE}"

    # 4) optional mid-sweep aggregate (cheap; updates learning_curve.csv).
    bash "${HERE}/aggregate.sh" "${SAVE}" || true
done

echo "[al_driver] done. Final aggregate:"
bash "${HERE}/aggregate.sh" "${SAVE}"
