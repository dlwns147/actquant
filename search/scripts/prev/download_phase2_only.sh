#!/usr/bin/env bash
# Phase 2 only — non-gated additions to the router pool.
# Phase 1 entries are skipped here; if they already exist on disk,
# snapshot_download is a no-op so this is safe to re-run.
set -u
cd "$(dirname "$0")/.."
LOG=/SSD/huggingface/_router_download.log

download() {
    local path="$1"; local name="$2"
    {
        echo "=== $(date -Iseconds)  ${path}/${name} ==="
        df -h /SSD | tail -1
    } | tee -a "$LOG"
    python download_hf.py --model_path "$path" --model_name "$name" 2>&1 | tee -a "$LOG"
    echo "--- done: ${path}/${name}" | tee -a "$LOG"
}

download Qwen        Qwen3-14B
download pytorch     gemma-3-27b-it-AWQ-INT4
download cyankiwi    Qwen3-Next-80B-A3B-Instruct-AWQ-4bit
download ibm-granite granite-4.1-30b

{
    echo "=== $(date -Iseconds)  PHASE 2 DONE ==="
    df -h /SSD | tail -1
} | tee -a "$LOG"
