#!/bin/bash
set -u

cd /NAS/SJ/actquant/poc/benchmark_proxy/amq_instruct_pure

# Usage: bash launch_all_seeds.sh "0,1,2,3,4,5,6,7"
GPU_LIST_STR=${1:-"0,1,2,3,4,5,6,7"}
IFS=',' read -ra GPUS <<< "$GPU_LIST_STR"
N_GPU=${#GPUS[@]}

LOG_DIR=scripts/launch_logs_$(date +%y%m%d%H%M)
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/main.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

log "GPUs: ${GPUS[*]} (N=$N_GPU)"

# LOAD_DIR=$(ls -1dt amq/results/search/*/ 2>/dev/null | head -1)
# LOAD_DIR=${LOAD_DIR%/}
# log "LOAD_DIR for tmp_100: $LOAD_DIR"

BITS=(2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.2)

# Round-robin: bit b -> GPUS[b % N_GPU]
declare -A GPU_BITS
for b in "${!BITS[@]}"; do
  g=${GPUS[$((b % N_GPU))]}
  GPU_BITS[$g]+="${BITS[$b]} "
done

for g in "${GPUS[@]}"; do
  log "GPU $g will run bits: ${GPU_BITS[$g]}"
done

for g in "${GPUS[@]}"; do
  (
    for BIT in ${GPU_BITS[$g]}; do
      # log "GPU $g bit $BIT: tmp_0 start"
      # bash scripts/amq_quantization_gen_tmp_0.sh   "$g" "$BIT"
      # log "GPU $g bit $BIT: tmp_1 start"
      # bash scripts/amq_quantization_gen_tmp_1.sh   "$g" "$BIT"
      # log "GPU $g bit $BIT: tmp_2 start"
      # bash scripts/amq_quantization_gen_tmp_2.sh   "$g" "$BIT"
      log "GPU $g bit $BIT: tmp_42 start"
      bash scripts/amq_quantization_gen_tmp_42.sh  "$g" "$BIT"
      # log "GPU $g bit $BIT: tmp_100 start"
      # bash scripts/amq_quantization_gen_tmp_100.sh "$g" "$BIT"
      log "GPU $g bit $BIT: done"
    done
  ) > "$LOG_DIR/gpu${g}.log" 2>&1 &
done

wait
log "All jobs finished."
