#!/usr/bin/env bash
# A: y measurement noise floor — re-evaluate the SAME archs under different
# calibration seeds (wikitext2 trainenc is shuffle(seed=…)-sampled, so --seed
# changes the 128 calibration segments = deployment-relevant protocol noise).
# The duplicate --seed appended AFTER build_model_args wins (argparse last-wins).
# Usage: bash scripts/surrogate/noise_eval.sh "411 412 410 102 872 992 740 920" "1 2 3"

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
source "${HERE}/_config.sh"
cd "${ROOT}"
mkdir -p logs

IDX_LIST=${1:?"need idx list"}
SEEDS=${2:-"1 2 3"}
VD=save/surrogate/llama_effkv_hard/validation

for S in ${SEEDS}; do
    D=save/surrogate/noise_floor/s${S}
    mkdir -p "${D}"
    [ -f "${D}/archs.csv" ] || cp "${VD}/validation_archs.csv" "${D}/archs.csv"
    [ -f "${D}/sample_meta.json" ] || cp "${VD}/validation_meta.json" "${D}/sample_meta.json"
done

COMMON="$(build_model_args) $(build_eval_args)"
GPUS=(0 1 2 3)
declare -A SLOT_PID=()

launch() {
    local IDX=$1 S=$2 G=$3
    local PORT=$(( (RANDOM % 20000) + 20000 ))
    CUDA_VISIBLE_DEVICES=${G} accelerate launch --num_processes=1 --num_machines=1 \
        --main_process_port=${PORT} surrogate_pipeline.py \
        --mode eval --idx ${IDX} --save save/surrogate/noise_floor/s${S} \
        --gpu_id 0 ${COMMON} --seed ${S} \
        > "logs/noise_s${S}_${IDX}.log" 2>&1 &
    SLOT_PID[$G]=$!
}

for S in ${SEEDS}; do
    for IDX in ${IDX_LIST}; do
        rf="save/surrogate/noise_floor/s${S}/result_${IDX}.json"
        if [ -f "${rf}" ] && grep -q "measured_metric" "${rf}" 2>/dev/null; then
            continue
        fi
        placed=0
        while [ ${placed} -eq 0 ]; do
            for G in "${GPUS[@]}"; do
                pid=${SLOT_PID[$G]:-}
                if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
                    launch "${IDX}" "${S}" "${G}"
                    echo "[noise] idx=${IDX} seed=${S} -> gpu ${G}"
                    placed=1
                    break
                fi
            done
            [ ${placed} -eq 0 ] && sleep 3
        done
    done
done
wait
echo "[noise] DONE"
