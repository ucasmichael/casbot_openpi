#!/usr/bin/env bash

# Source this file from /data/lyx/casbot_openpi before conversion or training.
export UV_PROJECT_ENVIRONMENT="/data/lyx/casbot_openpi/.venv"
export UV_CACHE_DIR="/data/lyx/.cache/uv"
export UV_PYTHON_INSTALL_DIR="/data/lyx/.cache/uv/python"
export PIP_CACHE_DIR="/data/lyx/.cache/pip"
export XDG_CACHE_HOME="/data/lyx/.cache"
export HF_HOME="/data/lyx/hf_cache"
export HF_HUB_CACHE="/data/lyx/hf_cache/hub"
export HF_DATASETS_CACHE="/data/lyx/hf_cache/datasets"
export HF_LEROBOT_HOME="/data/lyx/dataset/lerobot"
export OPENPI_DATA_HOME="/data/lyx/hf_cache"
export TORCH_HOME="/data/lyx/.cache/torch"
export WANDB_CACHE_DIR="/data/lyx/.cache/wandb"
export JAX_COMPILATION_CACHE_DIR="/data/lyx/.cache/jax"
export TMPDIR="/data/lyx/tmp/casbot_openpi"

# The lock pins PyAV 14.4, whose source build requires FFmpeg 7 development
# headers. Reuse the existing data-disk FFmpeg runtime while keeping Python in
# this repository's .venv.
A3_FFMPEG_ROOT="/data/lyx/hand_obj_tracking/.conda/envs/mint-inference"
export PKG_CONFIG_PATH="$A3_FFMPEG_ROOT/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$A3_FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LDFLAGS="${LDFLAGS:+$LDFLAGS }-Wl,-rpath,$A3_FFMPEG_ROOT/lib"

mkdir -p \
  "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" \
  "$PIP_CACHE_DIR" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_LEROBOT_HOME" \
  "$TORCH_HOME" \
  "$WANDB_CACHE_DIR" \
  "$JAX_COMPILATION_CACHE_DIR" \
  "$TMPDIR"
