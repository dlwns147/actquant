#!/usr/bin/env bash
#
# AvengersPro routing-study model pool — Phase 1 download queue.
#
# Designed to be re-runnable: snapshot_download skips files already present.
# Logs progress + free-space per model to /SSD/huggingface/_router_download.log.
#
# Gated models (google/gemma-3-*) require HF_TOKEN — handled by a separate
# section at the bottom; export HF_TOKEN=... before running them.
#
# Phase 1 footprint with allow/ignore patterns (post-slim):
#   ibm-granite/granite-4.1-3b               ~7  GB
#   ibm-granite/granite-4.1-8b               ~18 GB
#   Qwen/Qwen3-1.7B                          ~4  GB
#   Qwen/Qwen3-8B                            ~16 GB
#   Qwen/Qwen3-32B-AWQ                       ~19 GB
#   openai/gpt-oss-20b                       ~14 GB (MXFP4 only)
#   openai/gpt-oss-120b                      ~63 GB (MXFP4 only)
#   RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16  ~40 GB
#   RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16         ~40 GB
#   ---------- non-gated total ~221 GB
#   google/gemma-3-1b-it     ~2  GB  [GATED]
#   google/gemma-3-4b-it     ~9  GB  [GATED]

set -u
cd "$(dirname "$0")/.."
LOG=/SSD/huggingface/_router_download.log

HF_TOKEN=hf
huggingface-cli login --token ${HF_TOKEN}

download() {
    local path="$1"; local name="$2"
    {
        echo "=== $(date -Iseconds)  ${path}/${name} ==="
        df -h /SSD | tail -1
    } | tee -a "$LOG"
    python download_hf.py --model_path "$path" --model_name "$name" 2>&1 | tee -a "$LOG"
    echo "--- done: ${path}/${name}" | tee -a "$LOG"
}

# ----- Non-gated (no HF_TOKEN required) -----
download ibm-granite granite-4.1-3b
download Qwen        Qwen3-1.7B
download ibm-granite granite-4.1-8b
download Qwen        Qwen3-8B
download Qwen        Qwen3-32B-AWQ
download openai      gpt-oss-20b
download RedHatAI    Llama-3.3-70B-Instruct-quantized.w4a16
download RedHatAI    DeepSeek-R1-Distill-Llama-70B-quantized.w4a16
download openai      gpt-oss-120b

# ----- Phase 2 (all non-gated; ~154 GB) -----
download Qwen        Qwen3-14B
download pytorch     gemma-3-27b-it-AWQ-INT4
download cyankiwi    Qwen3-Next-80B-A3B-Instruct-AWQ-4bit
download ibm-granite granite-4.1-30b

# ----- Gated (require HF_TOKEN; comment out or run after `huggingface-cli login`) -----
download google      gemma-3-1b-it
download google      gemma-3-4b-it

{
    echo "=== $(date -Iseconds)  ALL DONE ==="
    df -h /SSD | tail -1
    du -sh /SSD/huggingface/* 2>/dev/null
} | tee -a "$LOG"
