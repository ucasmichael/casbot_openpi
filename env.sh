#!/usr/bin/env bash

set -u

# Keep the project tools and all model/data caches in predictable locations.
export PATH="/home/lyx/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/data/lyx/casbot_openpi/.venv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HOME="${HF_HOME:-/data/lyx/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HF_HOME/openpi}"

# ===== 加速配置 =====
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

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
retry_cmd "GIT_LFS_SKIP_SMUDGE=1 uv pip install --python \"$UV_PROJECT_ENVIRONMENT/bin/python\" --no-deps -e ." "uv install"

echo "🎉 Done"
