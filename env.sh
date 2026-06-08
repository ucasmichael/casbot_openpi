#!/usr/bin/env bash

set -u

# ===== 加速配置 =====
export HF_ENDPOINT=https://hf-mirror.com

retry_cmd() {
    local cmd="$1"
    local name="$2"
    local attempt=1

    while true; do
        echo "[$name] Attempt $attempt..."
        eval "$cmd"
        if [ $? -eq 0 ]; then
            echo "[$name] Success ✅"
            break
        else
            echo "[$name] Failed ❌, retrying in 5s..."
            attempt=$((attempt+1))
            sleep 5
        fi
    done
}

echo "===== Stage 1 ====="
retry_cmd "GIT_LFS_SKIP_SMUDGE=1 uv sync" "uv sync"

echo "===== Stage 2 ====="
retry_cmd "GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ." "uv install"

echo "🎉 Done"