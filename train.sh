export HF_DATASETS_CACHE="/data/lyx/hf_cache"
export HF_LEROBOT_HOME="/data/lyx/dataset/lerobot"
cd /data/lyx/casbot_openpi
source .venv/bin/activate
export WANDB_MODE=disabled
export WANDB_SILENT=true
# python examples/casbot_real/convert2lerobot_statecut.py --raw-dir /data/lyx/casbot_dataset/local_dataset/hdf5_dataset/casbot01 --repo-id local_dataset/casbot_white_board
# CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name casbot_fold_cloth
# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py casbot_fold_cloth --exp-name=casbot_fold_cloth_white_board_0523 --overwrite

# /home/lyx/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
.venv/bin/python examples/a3_real/convert_a3_mcap_to_lerobot.py   --raw-dir /data/lyx/dataset/data0813/data0813_flat   --repo-id local_dataset/a3_0813_both   --lerobot-root "$HF_LEROBOT_HOME"   --joint-mode both   --mode video   --overwrite --task "stack cups"
uv run scripts/compute_norm_stats.py --config-name pi05_a3_both
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_a3_both --exp-name=pi05_a3_both_stack_cups_chunk50 --overwrite