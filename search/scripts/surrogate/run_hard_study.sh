#!/usr/bin/env bash
# Hard-regime AL study driver (llama_effkv_hard, eff_kvbits ∈ [2.0, 2.7]):
# 1000-point region validation (random600 + band240 + front160, pre-sampled)
# + 5 tracks × 17 rounds × batch 10 (n_train 29 → 199 ≈ 200, per user budget).
# Adds a 2-pass retry sweep (transient HF-hub 504s leave holes; eval_local is
# idempotent/fast-resume) and a final val-copy refresh + re-aggregate.
# In-band-SAMPLING upper bound (band known in advance): exports HARD_BAND=1 so
# _config.sh restricts the sampling pool to eff_kvbits ∈ [2.0, 2.7]. The
# deployment protocol (global-train → band-val) lives in run_band_protocol.sh.

set -uo pipefail
export HARD_BAND=1

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
cd "${ROOT}"

TAG=${1:-llama_effkv_hard}
ROUNDS=${2:-17}
BATCH=${3:-10}
# ga = the deployed QS+GA (coverage_nsga2) baseline — the target to beat.
METHODS=(random ga maximin al_neyman al_cvvor al_mepe)

INIT_RANDOM=20 bash scripts/surrogate/run_compare_local.sh "${TAG}" "${ROUNDS}" "${BATCH}" 60 "${METHODS[@]}"

# ── retry sweep: fill any holes left by transient eval failures ──
for pass in 1 2; do
    NV=$(( $(wc -l < "save/surrogate/${TAG}/validation/validation_archs.csv") - 1 ))
    bash scripts/surrogate/eval_local.sh "save/surrogate/${TAG}/validation" 0 $(( NV - 1 )) --validation
    for M in "${METHODS[@]}"; do
        D="save/surrogate/${TAG}/${M}"
        [ -f "${D}/archs.csv" ] || continue
        N=$(( $(wc -l < "${D}/archs.csv") - 1 ))
        bash scripts/surrogate/eval_local.sh "${D}" 0 $(( N - 1 ))
    done
done

# ── refresh per-method validation copies (run_compare_local copied them
# BEFORE the retry pass could fill holes) + final aggregate ──
for M in "${METHODS[@]}"; do
    D="save/surrogate/${TAG}/${M}"
    [ -d "${D}" ] || continue
    cp "save/surrogate/${TAG}/validation/validation_archs.csv" "${D}/"
    for vr in "save/surrogate/${TAG}/validation"/validation_result_*.json; do
        bn=$(basename "${vr}"); [ -e "${D}/${bn}" ] || cp "${vr}" "${D}/${bn}"
    done
    bash scripts/surrogate/aggregate.sh "${D}" || true
done
echo "[hard_study] DONE"
