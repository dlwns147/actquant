#!/bin/bash
# Launch 3 search runs in parallel with jsd-bin stratification.
# GPU 분배: seed 42 → GPUs 2,3 / seed 0 → GPUs 4,5 / seed 100 → GPUs 6,7
#
# Usage: bash scripts/amq_search_jsd_3seeds.sh

set -e
REPO_DIR=/NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure
cd ${REPO_DIR}

mkdir -p logs

bash scripts/amq_search.sh 4,5 0   JSD  > logs/jsd_seed0.log   2>&1 &
PID_0=$!
bash scripts/amq_search.sh 2,3 42  JSD  > logs/jsd_seed42.log  2>&1 &
PID_42=$!
bash scripts/amq_search.sh 6,7 100 JSD  > logs/jsd_seed100.log 2>&1 &
PID_100=$!

echo "started PIDs: seed42=${PID_42} seed0=${PID_0} seed100=${PID_100}"
echo "logs in ${REPO_DIR}/logs/jsd_seed{42,0,100}.log"

wait ${PID_42} ${PID_0} ${PID_100}
echo "all 3 jsd-stratified searches finished"
