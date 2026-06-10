#!/usr/bin/env bash
# SLURM-free replacement for eval.sbatch: evaluate idx FROM..TO of
# <SAVE>/archs.csv (or validation_archs.csv) across all local GPUs in parallel,
# keeping at most one job per GPU. Idempotent (surrogate_pipeline skips done idx).
#
# Usage: bash scripts/surrogate/eval_local.sh <SAVE> <FROM> <TO> [--validation] [GPUS]
#   GPUS  space/comma list, default "0 1 2 3"

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
source "${HERE}/_config.sh"
cd "${ROOT}"
mkdir -p logs

SAVE=${1:?"need <SAVE>"}
FROM=${2:?"need FROM"}
TO=${3:?"need TO"}
VAL_FLAG=${4:-}
GPUS_ARG=${5:-${SURR_GPUS:-"0 1 2 3"}}
read -r -a GPUS <<< "${GPUS_ARG//,/ }"
NG=${#GPUS[@]}

COMMON="$(build_model_args) $(build_eval_args)"
[ "${VAL_FLAG}" = "--validation" ] && COMMON+=" --validation"

declare -A SLOT_PID=()    # gpu -> pid of running job (if any)

launch() {
    local IDX=$1 G=$2
    local PORT=$(( (RANDOM % 20000) + 20000 ))
    local tag=$(echo "${SAVE}" | tr '/' '_')
    CUDA_VISIBLE_DEVICES=${G} accelerate launch --num_processes=1 --num_machines=1 \
        --main_process_port=${PORT} surrogate_pipeline.py \
        --mode eval --idx ${IDX} --save ${SAVE} --gpu_id 0 ${COMMON} \
        > "logs/eval_${tag}_${IDX}.log" 2>&1 &
    SLOT_PID[$G]=$!
}

t0=$(date +%s)
for IDX in $(seq "${FROM}" "${TO}"); do
    placed=0
    while [ ${placed} -eq 0 ]; do
        for G in "${GPUS[@]}"; do
            pid=${SLOT_PID[$G]:-}
            if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
                launch "${IDX}" "${G}"
                echo "[eval_local] idx=${IDX} -> gpu ${G} (pid ${SLOT_PID[$G]})"
                placed=1
                break
            fi
        done
        [ ${placed} -eq 0 ] && sleep 3
    done
done
wait
echo "[eval_local] ${SAVE} idx ${FROM}..${TO} done in $(( $(date +%s) - t0 ))s"
