#!/usr/bin/env bash
# DEPLOYMENT-protocol band study (global-train → band-local validate) + the
# in-band-sampling upper bound, sequenced on the local 4 GPUs:
#   P1  evaluate the hard-band region validation pool (1000 archs, 2 retry
#       passes for transient HF-hub failures; eval is idempotent).
#   P2  extend the GLOBAL llama_effkv_lc tracks rounds 8..17 (n_train 99→199,
#       per-user budget 200): random ga maximin al_neyman al_cvvor.
#   P3  CPU: global-trained surrogates → hard-band val-1000 region metrics
#       (tests/global_train_band_val.py) + per-dir re-aggregate.
#   P4  in-band-sampling upper bound: HARD_BAND=1 run_hard_study.sh
#       (its Stage A val eval fast-skips; 6 tracks × 17 rounds in-band).
# Phase markers ([P1] DONE … [protocol] DONE) are greppable by the monitor.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
cd "${ROOT}"

GTAG=llama_effkv_lc
HTAG=llama_effkv_hard
GMETHODS=(random ga maximin al_neyman al_cvvor)
EXT_ROUNDS_FROM=8
EXT_ROUNDS_TO=17
BATCH=10

last_idx() { awk -F, 'NR>1{print $1}' "$1" | tail -1; }

echo "[P1] hard-band validation eval (1000 archs)"
for pass in 1 2; do
    bash scripts/surrogate/eval_local.sh "save/surrogate/${HTAG}/validation" 0 999 --validation
done
NVDONE=$(ls save/surrogate/${HTAG}/validation/validation_result_*.json 2>/dev/null | wc -l)
echo "[P1] DONE (${NVDONE}/1000)"

echo "[P2] global track extension rounds ${EXT_ROUNDS_FROM}..${EXT_ROUNDS_TO}"
for M in "${GMETHODS[@]}"; do
    D="save/surrogate/${GTAG}/${M}"
    [ -f "${D}/archs.csv" ] || { echo "[P2] ${M}: no archs.csv, skip"; continue; }
    for R in $(seq "${EXT_ROUNDS_FROM}" "${EXT_ROUNDS_TO}"); do
        # skip rounds that already exist (resumable)
        if awk -F, -v r="${R}" 'NR>1 && $2==r {found=1} END{exit !found}' "${D}/archs.csv"; then
            continue
        fi
        bash scripts/surrogate/sample.sh "${D}" "${M}" "${R}" "${BATCH}"
        LAST=$(last_idx "${D}/archs.csv")
        FROM=$(( LAST - BATCH + 1 ))
        bash scripts/surrogate/eval_local.sh "${D}" "${FROM}" "${LAST}"
    done
done
# retry sweep for the extension evals
for M in "${GMETHODS[@]}"; do
    D="save/surrogate/${GTAG}/${M}"
    [ -f "${D}/archs.csv" ] || continue
    N=$(( $(wc -l < "${D}/archs.csv") - 1 ))
    bash scripts/surrogate/eval_local.sh "${D}" 0 $(( N - 1 ))
done
echo "[P2] DONE"

echo "[P3] global-train → band-val analysis + re-aggregate"
source scripts/surrogate/_config.sh
CUDA_VISIBLE_DEVICES="" python tests/global_train_band_val.py \
    $(build_model_args) $(build_sample_args) --mode sample \
    --save "save/surrogate/${GTAG}" || true
for M in "${GMETHODS[@]}"; do
    bash scripts/surrogate/aggregate.sh "save/surrogate/${GTAG}/${M}" || true
done
echo "[P3] DONE"

echo "[P4] in-band-sampling upper bound (HARD_BAND=1)"
bash scripts/surrogate/run_hard_study.sh
echo "[P4] DONE"

echo "[protocol] DONE"
