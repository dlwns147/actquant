#!/usr/bin/env bash
# SLURM-free per-acquisition comparison (HQQ): build a shared quantile warm-start
# + a shared validation pool, then for EACH method run AL rounds of --batch and
# measure the surrogate on the held-out validation pool each round
# (learning_curve.csv per method). All methods share the SAME warm-start + the
# SAME validation pool, so the only difference is which extras each acquisition
# picks.
#
# Usage: bash scripts/surrogate/run_compare_local.sh <TAG> [<ROUNDS> <BATCH> <N_VAL> <METHODS...>]
#   METHODS default: random al_ei al_ucb al_alm al_imse al_maximin al_qbc al_rank
#
# Resumable: every eval is idempotent, so re-running continues where it stopped.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
source "${HERE}/_config.sh"
cd "${ROOT}"

TAG=${1:?"need <TAG>"}
ROUNDS=${2:-2}
BATCH=${3:-10}
NVAL=${4:-40}
shift $(( $# < 4 ? $# : 4 )) || true
METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS=(random al_ei al_ucb al_alm al_imse al_maximin al_qbc al_rank)
fi

ROOT_SAVE="save/surrogate/${TAG}"
SEED_DIR="${ROOT_SAVE}/seed"
VAL_DIR="${ROOT_SAVE}/validation"
mkdir -p "${SEED_DIR}" "${VAL_DIR}" logs
echo "[compare_local] TAG=${TAG} rounds=${ROUNDS} batch=${BATCH} n_val=${NVAL}"
echo "[compare_local] methods: ${METHODS[*]}"

last_idx() { awk -F, 'NR>1{print $1}' "$1" | tail -1; }

# ── Stage A: shared validation pool ────────────────────────────────────────
if [ ! -f "${VAL_DIR}/validation_archs.csv" ]; then
    bash "${HERE}/sample.sh" "${VAL_DIR}" validation 0 "${NVAL}"
fi
NV=$(( $(wc -l < "${VAL_DIR}/validation_archs.csv") - 1 ))
echo "[compare_local] validation pool = ${NV}; evaluating…"
bash "${HERE}/eval_local.sh" "${VAL_DIR}" 0 $(( NV - 1 )) --validation

# ── Stage B: shared warm-start = quantile [+ INIT_RANDOM random] ────────────
# INIT_RANDOM env (default 0) appends N random archs to the quantile anchors so
# the shared initial = "quantile + random N" (all round 0), per the LC protocol.
if [ ! -f "${SEED_DIR}/archs.csv" ]; then
    bash "${HERE}/sample.sh" "${SEED_DIR}" quantile 0
    if [ "${INIT_RANDOM:-0}" -gt 0 ]; then
        bash "${HERE}/sample.sh" "${SEED_DIR}" random 0 "${INIT_RANDOM}"
    fi
fi
NSEED=$(( $(wc -l < "${SEED_DIR}/archs.csv") - 1 ))
echo "[compare_local] warm-start (quantile + ${INIT_RANDOM:-0} random) = ${NSEED}; evaluating…"
bash "${HERE}/eval_local.sh" "${SEED_DIR}" 0 $(( NSEED - 1 ))

# ── Stage C: per-method AL rounds ───────────────────────────────────────────
for M in "${METHODS[@]}"; do
    D="${ROOT_SAVE}/${M}"
    mkdir -p "${D}"
    # seed each method dir with the SAME warm-start (archs + results + meta)
    cp -n "${SEED_DIR}/archs.csv"        "${D}/archs.csv"        2>/dev/null || true
    cp -n "${SEED_DIR}/sample_meta.json" "${D}/sample_meta.json" 2>/dev/null || true
    for rj in "${SEED_DIR}"/result_*.json; do
        bn=$(basename "${rj}"); [ -e "${D}/${bn}" ] || cp "${rj}" "${D}/${bn}"
    done
    # share the validation pool (cp — this FS does not support symlinks)
    [ -e "${D}/validation_archs.csv" ] || cp "${VAL_DIR}/validation_archs.csv" "${D}/validation_archs.csv"
    for vr in "${VAL_DIR}"/validation_result_*.json; do
        bn=$(basename "${vr}"); [ -e "${D}/${bn}" ] || cp "${vr}" "${D}/${bn}"
    done

    echo "════════ method=${M} ════════"
    for R in $(seq 1 "${ROUNDS}"); do
        bash "${HERE}/sample.sh" "${D}" "${M}" "${R}" "${BATCH}"
        LAST=$(last_idx "${D}/archs.csv")
        FROM=$(( LAST - BATCH + 1 ))
        echo "[compare_local] ${M} round ${R}: eval idx ${FROM}..${LAST}"
        bash "${HERE}/eval_local.sh" "${D}" "${FROM}" "${LAST}"
    done
    bash "${HERE}/aggregate.sh" "${D}" || true
done

echo ""
echo "[compare_local] DONE. Per-method learning curves:"
for M in "${METHODS[@]}"; do
    lc="${ROOT_SAVE}/${M}/learning_curve.csv"
    [ -f "${lc}" ] && { echo "─ ${M}"; cat "${lc}"; }
done
