#!/usr/bin/env bash
# Usage: bash scripts/surrogate/run_compare.sh [<TAG>]
#
# Top-level driver — submits the entire 4-track comparison study:
#   SAVE_VAL  : validation pool        (uniform random N_VAL @ val_seed)
#   SAVE_RAND : quantile + random      (one-shot N_EXTRAS, round 0)
#   SAVE_GA   : quantile + coverage_GA (one-shot N_EXTRAS, round 0)
#   SAVE_AL   : quantile + AL EI loop  (AL_BATCH × AL_ROUNDS, round 0..R)
# All four share the same quantile warm-start anchors (cp'd from SAVE_RAND).
#
# Layout under save/surrogate/<TAG>/:
#   validation/   archs.csv + 100 validation_result_*.json
#   random/       archs.csv + result_*.json + results_random.csv
#   ga/           archs.csv + result_*.json + results_ga.csv
#   al/           archs.csv + result_*.json + results_al_ei.csv
#                                + learning_curve.csv
#
# Each method dir gets its own validation symlink + aggregate at the end
# so validation_metrics.csv is method-local.
#
# This driver itself is short — every step uses sbatch --wait so the
# user can ctrl-C between steps without losing intermediate state.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
source "${HERE}/_config.sh"
cd "${ROOT}"

TAG=${1:-$(date +%y%m%d%H%M)_${MODEL_NAME}}
ROOT_SAVE="save/surrogate/${TAG}"
SAVE_VAL="${ROOT_SAVE}/validation"
SAVE_RAND="${ROOT_SAVE}/random"
SAVE_GA="${ROOT_SAVE}/ga"
SAVE_AL="${ROOT_SAVE}/al"

mkdir -p "${SAVE_VAL}" "${SAVE_RAND}" "${SAVE_GA}" "${SAVE_AL}" logs
echo "[run_compare] tag=${TAG}"
echo "[run_compare] root=${ROOT_SAVE}"

# ── Stage A: validation pool (one-shot) ────────────────────────────────────
echo "════════════════ Stage A: validation pool (N_VAL=${N_VAL}) ════════════"
bash "${HERE}/sample.sh" "${SAVE_VAL}" validation 0 "${N_VAL}"
N_VAL_ACTUAL=$(( $(wc -l < "${SAVE_VAL}/validation_archs.csv") - 1 ))
echo "[run_compare] validation archs.csv has ${N_VAL_ACTUAL} rows"

VAL_JOBID=$(sbatch --parsable \
    --array=0-$(( N_VAL_ACTUAL - 1 ))%${SLURM_ARRAY_CONCURRENCY} \
    "${HERE}/eval.sbatch" "${SAVE_VAL}" --validation)
echo "[run_compare] validation eval array submitted: ${VAL_JOBID}"

# ── Stage B: shared quantile warm-start for the three method dirs ──────────
# Build once under SAVE_RAND then copy to SAVE_GA / SAVE_AL to guarantee
# byte-identical warm-start across methods (same arch order + idx).
echo "════════════════ Stage B: shared quantile warm-start ══════════════════"
bash "${HERE}/sample.sh" "${SAVE_RAND}" quantile 0
cp "${SAVE_RAND}/archs.csv"        "${SAVE_GA}/archs.csv"
cp "${SAVE_RAND}/archs.csv"        "${SAVE_AL}/archs.csv"
cp "${SAVE_RAND}/sample_meta.json" "${SAVE_GA}/sample_meta.json"
cp "${SAVE_RAND}/sample_meta.json" "${SAVE_AL}/sample_meta.json"
N_QUANT=$(( $(wc -l < "${SAVE_RAND}/archs.csv") - 1 ))
echo "[run_compare] quantile anchors: ${N_QUANT}"

# ── Stage C1: random extras (one-shot) + array submit ──────────────────────
echo "════════════════ Stage C1: random extras (N_EXTRAS=${N_EXTRAS}) ═══════"
bash "${HERE}/sample.sh" "${SAVE_RAND}" random 0
N_RAND=$(( $(wc -l < "${SAVE_RAND}/archs.csv") - 1 ))
RAND_JOBID=$(sbatch --parsable \
    --array=0-$(( N_RAND - 1 ))%${SLURM_ARRAY_CONCURRENCY} \
    "${HERE}/eval.sbatch" "${SAVE_RAND}")
echo "[run_compare] random eval array submitted: ${RAND_JOBID}"

# ── Stage C2: GA extras (one-shot) + array submit ──────────────────────────
echo "════════════════ Stage C2: GA extras (N_EXTRAS=${N_EXTRAS}) ═══════════"
bash "${HERE}/sample.sh" "${SAVE_GA}" ga 0
N_GA=$(( $(wc -l < "${SAVE_GA}/archs.csv") - 1 ))
GA_JOBID=$(sbatch --parsable \
    --array=0-$(( N_GA - 1 ))%${SLURM_ARRAY_CONCURRENCY} \
    "${HERE}/eval.sbatch" "${SAVE_GA}")
echo "[run_compare] GA eval array submitted: ${GA_JOBID}"

# ── Stage C3: AL EI loop ───────────────────────────────────────────────────
# Submit the round-0 eval array for the quantile anchors first, then hand
# off to al_driver.sh which loops rounds 1..AL_ROUNDS with sbatch --wait.
echo "════════════════ Stage C3: AL EI (round 0 then ${AL_ROUNDS} rounds) ══"
AL_R0_JOBID=$(sbatch --parsable \
    --array=0-$(( N_QUANT - 1 ))%${SLURM_ARRAY_CONCURRENCY} \
    "${HERE}/eval.sbatch" "${SAVE_AL}")
echo "[run_compare] AL round 0 eval array submitted: ${AL_R0_JOBID}"

# Wait for AL round 0 before starting the AL driver (so EI has training data).
# Other arrays (validation, random, GA) run in parallel; we don't need to
# wait for them yet.
echo "[run_compare] waiting on AL round 0 (${AL_R0_JOBID}) so EI has training data…"
sbatch --wait --dependency=afterany:${AL_R0_JOBID} \
    --output=logs/_wait_al_r0_%j.out \
    --wrap="echo 'AL round 0 done — driver can start'"

# Launch the AL driver inline (blocking). User can ctrl-C between rounds.
bash "${HERE}/al_driver.sh" "${SAVE_AL}" "${AL_ROUNDS}" "${AL_BATCH}"

# ── Stage D: wait on remaining arrays, then aggregate ──────────────────────
echo "════════════════ Stage D: wait + aggregate ════════════════════════════"
for J in ${VAL_JOBID} ${RAND_JOBID} ${GA_JOBID}; do
    sbatch --wait --dependency=afterany:${J} \
        --output=logs/_wait_${J}_%j.out \
        --wrap="echo 'array ${J} done'"
done

# Validation pool is shared across methods → symlink so each method's
# aggregate sees it.
for D in "${SAVE_RAND}" "${SAVE_GA}" "${SAVE_AL}"; do
    [ -e "${D}/validation_archs.csv" ] || ln -sf "$(realpath "${SAVE_VAL}/validation_archs.csv")" "${D}/validation_archs.csv"
    for vr in "${SAVE_VAL}"/validation_result_*.json; do
        bn=$(basename "${vr}")
        [ -e "${D}/${bn}" ] || ln -sf "$(realpath "${vr}")" "${D}/${bn}"
    done
done

echo "[run_compare] running per-method aggregate…"
bash "${HERE}/aggregate.sh" "${SAVE_RAND}"
bash "${HERE}/aggregate.sh" "${SAVE_GA}"
bash "${HERE}/aggregate.sh" "${SAVE_AL}"

echo ""
echo "[run_compare] DONE. Compare results under ${ROOT_SAVE}/{random,ga,al}/:"
echo "  - results_<method>.csv   (post_search.py --sample_path)"
echo "  - validation_metrics.csv (R² / Spearman / Kendall on shared val pool)"
echo "  - learning_curve.csv     (al/ only)"
