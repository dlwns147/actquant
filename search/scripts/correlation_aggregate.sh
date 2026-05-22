#!/usr/bin/env bash
# Usage: bash scripts/correlation_aggregate.sh [<SAVE_DIR>]
# Walks every result_<idx>.json in SAVE_DIR (or the newest save/correlation/*
# if omitted) and writes correlation.csv — a wide CSV with one row per arch,
# the per-axis search metric / complexity columns, every calibration metric
# (c4_ppl, wt2_jsd, …, needle_nll), and every benchmark sub-score
# (longbench__*, longbench_e__*__<bucket>, ruler__*).
#
# Safe to run mid-sweep: cells for archs that haven't been eval'd yet are
# simply left blank.

SAVE_ARG=${1:-}
if [ -z "${SAVE_ARG}" ]; then
    SAVE_ARG=$(ls -dt save/correlation/*/ 2>/dev/null | head -1 | sed 's:/*$::')
    if [ -z "${SAVE_ARG}" ]; then
        echo "ERROR: no save/correlation/* dir found; pass SAVE_DIR explicitly."
        exit 1
    fi
    echo "[auto] SAVE_DIR=${SAVE_ARG}"
fi

python correlation.py --mode aggregate --save "${SAVE_ARG}"
