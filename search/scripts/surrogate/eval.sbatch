#!/usr/bin/env bash
#SBATCH --job-name=surrogate-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/surrogate-eval_%A_%a.out
#SBATCH --error=logs/surrogate-eval_%A_%a.err
#
# Usage:
#   sbatch --array=<FROM>-<TO>%<CONCURRENCY> \
#          scripts/surrogate/eval.sbatch <SAVE> [<--validation>]
#
# One SLURM task evaluates one row of <SAVE>/archs.csv (or
# <SAVE>/validation_archs.csv when the second arg is `--validation`).
# IDX = SLURM_ARRAY_TASK_ID. Writes <SAVE>/result_<IDX>.json (or
# validation_result_<IDX>.json). Idempotent: skips when already done.
#
# Add cluster-specific options at the top (partition, account, qos,
# constraint, …) by editing this file.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
source "${HERE}/_config.sh"
cd "${ROOT}"
mkdir -p logs

SAVE=${1:?"need <SAVE> as first sbatch arg"}
VAL_FLAG=${2:-}     # pass "--validation" to eval validation_archs.csv
IDX=${SLURM_ARRAY_TASK_ID:?"this script must be launched as an array job"}

ARGS="--mode eval --idx ${IDX} --save ${SAVE} --gpu_id 0"
ARGS+=" $(build_model_args)"
ARGS+=" $(build_eval_args)"
[ "${VAL_FLAG}" = "--validation" ] && ARGS+=" --validation"

# accelerate launch isolates per-task; the SLURM gres allocation already
# limits CUDA_VISIBLE_DEVICES to a single GPU on this node.
PORT_NUM=$(( ( RANDOM % 10000 ) + 10000 ))
accelerate launch --num_processes=1 --num_machines=1 \
    --main_process_port=${PORT_NUM} \
    surrogate_pipeline.py ${ARGS}
