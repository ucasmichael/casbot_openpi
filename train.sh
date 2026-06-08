export HF_DATASETS_CACHE="/data/lyx/hf_cache"
export HF_LEROBOT_HOME="/data/lyx/casbot_dataset"
cd /data/lyx/casbot_openpi
source .venv/bin/activate
python examples/casbot_real/convert2lerobot_statecut.py --raw-dir /data/lyx/casbot_dataset/local_dataset/hdf5_dataset/casbot01 --repo-id local_dataset/casbot_white_board
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name casbot_fold_cloth
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py casbot_fold_cloth --exp-name=casbot_fold_cloth_white_board_0523 --overwrite

# /home/lyx/.cache/openpi/openpi-assets/checkpoints/pi05_base/params