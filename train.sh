export HF_DATASETS_CACHE="/data/lyx/hf_cache"
export HF_LEROBOT_HOME="/data/lyx/dataset/lerobot"
export OPENPI_DATA_HOME="/data/lyx/hf_cache/openpi"
cd /data/lyx/casbot_openpi
source .venv/bin/activate
export WANDB_MODE=disabled
export WANDB_SILENT=true
export OPENPI_SKIP_WEIGHT_LOAD="${OPENPI_SKIP_WEIGHT_LOAD:-1}"
set -e
# python examples/casbot_real/convert2lerobot_statecut.py --raw-dir /data/lyx/casbot_dataset/local_dataset/hdf5_dataset/casbot01 --repo-id local_dataset/casbot_white_board
# CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name casbot_fold_cloth
# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py casbot_fold_cloth --exp-name=casbot_fold_cloth_white_board_0523 --overwrite

# /home/lyx/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
DATASET_ROOT="$HF_LEROBOT_HOME/local_dataset/a3_0813_both"
RAW_A3_DIR="${RAW_A3_DIR:-/data/lyx/dataset/data0819_flat}"
FORCE_A3_CONVERT="${FORCE_A3_CONVERT:-1}"
if [ "$FORCE_A3_CONVERT" = "1" ] || [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
  .venv/bin/python examples/a3_real/convert_a3_mcap_to_lerobot.py \
    --raw-dir "$RAW_A3_DIR" \
    --repo-id local_dataset/a3_0813_both \
    --lerobot-root "$HF_LEROBOT_HOME" \
    --joint-mode both \
    --mode video \
    --overwrite \
    --task "stack cups"
else
  echo "Found existing LeRobot dataset: $DATASET_ROOT"
fi

NORM_STATS="assets/pi05_a3_both/local_dataset/a3_0813_both/norm_stats.json"
if [ "$FORCE_A3_CONVERT" = "1" ] || [ ! -f "$NORM_STATS" ]; then
  .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_a3_both
else
  echo "Found existing norm stats: $NORM_STATS"
fi

TRAIN_STEPS="${TRAIN_STEPS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
.venv/bin/python scripts/train.py pi05_a3_both \
  --exp-name=pi05_a3_both_stack_cups_chunk50 \
  --model.paligemma-variant=dummy \
  --model.action-expert-variant=dummy \
  --num-train-steps="$TRAIN_STEPS" \
  --log-interval=1 \
  --save-interval=1 \
  --no-wandb-enabled \
  --overwrite

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_a3_both --exp-name=pi05_a3_both_stack_cups_chunk50 --overwrite
