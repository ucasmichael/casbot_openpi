# A3 Sim2Real retarget data to LeRobot

This directory converts the processed retarget outputs under
`/data/lyx/hand_obj_tracking/data` into local LeRobot v2.1 datasets used by
the A3 OpenPI training configs.

## Data contract

The executable source is `a3_qpose.json`, not `eef_qpose_37d.json`. The source
contains waist and arm positions in radians and OmniHand actuator commands in
the integer range `[0, 4095]`.

The converter changes the source wire order:

```text
waist[yaw, roll, pitch] + left_arm7 + right_arm7 + left_hand10 + right_hand10
```

to the existing OpenPI A3 training order:

```text
left_arm7 + right_arm7 + left_hand10 + right_hand10 + waist[roll, pitch, yaw]
```

Both `observation.state` and `action` contain the same absolute executable 37D
qpose. The OpenPI dataloader builds the future action chunk from `action` using
the configured horizon.

`put_spoon_20260913` is marked as right-hand-only by all of its source files.
The two dual-arm datasets are `put_pass_bowl_20260913` and
`put_yangtao_spoon_20260913`.

## Environment

Run from the repository root. `UV_PROJECT_ENVIRONMENT` pins the virtual
environment to `/data/lyx/casbot_openpi/.venv`; caches and temporary files are
routed to `/data`.

The lock pins PyAV 14.4, which needs FFmpeg 7 when built from source. The
environment script points its build/runtime library path at the existing
`/data/lyx/hand_obj_tracking/.conda/envs/mint-inference` FFmpeg 7.1.1 files;
Python itself still runs entirely from this repository's `.venv`.

```bash
cd /data/lyx/casbot_openpi
source examples/a3_sim2real/env.sh
uv python install 3.11
uv sync --frozen --managed-python --python 3.11
```

Download only the JAX/Orbax pi05 base parameters if they are not already present:

```bash
.venv/bin/python -c 'from openpi.shared import download; print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params"))'
```

The expected parameter path is:

```text
/data/lyx/hf_cache/openpi-assets/checkpoints/pi05_base/params
```

## Preflight

Preflight validates schemas, action dimensions and ranges, source FPS, frame
uniqueness, and the existence of every selected RGB/depth pair. It does not
write a dataset.

```bash
.venv/bin/python examples/a3_sim2real/convert_a3_retarget_to_lerobot.py \
  --raw-dir /data/lyx/hand_obj_tracking/data/move_cup_20260913 \
  --repo-id local_dataset/a3_sim2real_move_cup_20260913 \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --task "move the paper cup" \
  --dry-run
```

For a small write-path smoke test, use a separate repo id:

```bash
.venv/bin/python examples/a3_sim2real/convert_a3_retarget_to_lerobot.py \
  --raw-dir /data/lyx/hand_obj_tracking/data/move_cup_20260913 \
  --repo-id local_dataset/a3_sim2real_smoke \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --task "move the paper cup" \
  --max-episodes 1 \
  --max-frames-per-episode 8 \
  --overwrite
```

## Full conversion

Image mode is the default and preserves the source `uint16` depth PNGs. Each
command fails before creating output if preflight finds an invalid episode.

```bash
.venv/bin/python examples/a3_sim2real/convert_a3_retarget_to_lerobot.py \
  --raw-dir /data/lyx/hand_obj_tracking/data/move_cup_20260913 \
  --repo-id local_dataset/a3_sim2real_move_cup_20260913 \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --task "move the paper cup" \
  --overwrite

.venv/bin/python examples/a3_sim2real/convert_a3_retarget_to_lerobot.py \
  --raw-dir /data/lyx/hand_obj_tracking/data/hang_blue_cup_20260914 \
  --repo-id local_dataset/a3_sim2real_hang_blue_cup_20260914 \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --task "hang the blue cup on the stand" \
  --overwrite

.venv/bin/python examples/a3_sim2real/convert_a3_retarget_to_lerobot.py \
  --raw-dir /data/lyx/hand_obj_tracking/data/put_spoon_20260913 \
  --repo-id local_dataset/a3_sim2real_put_spoon_20260913 \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --task "put the spoon in the bowl" \
  --overwrite

.venv/bin/python examples/a3_sim2real/convert_a3_retarget_to_lerobot.py \
  --raw-dir /data/lyx/hand_obj_tracking/data/put_yangtao_20260913 \
  --repo-id local_dataset/a3_sim2real_put_yangtao_20260913 \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --task "put the star fruit on the plate" \
  --overwrite

.venv/bin/python examples/a3_sim2real/convert_a3_retarget_to_lerobot.py \
  --raw-dir /data/lyx/hand_obj_tracking/data/put_pass_bowl_20260913 \
  --repo-id local_dataset/a3_sim2real_put_pass_bowl_20260913 \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --task "pass the bowl between hands" \
  --overwrite

.venv/bin/python examples/a3_sim2real/convert_a3_retarget_to_lerobot.py \
  --raw-dir /data/lyx/hand_obj_tracking/data/put_yangtao_spoon_20260913 \
  --repo-id local_dataset/a3_sim2real_put_yangtao_spoon_20260913 \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --task "put the star fruit and spoon in the bowl" \
  --overwrite
```

Do not pass `--skip-invalid` for a formal dataset. That option exists only for
diagnostics and records skipped sessions in `meta/conversion_manifest.json`.

## Normalization and training

Compute normalization statistics after conversion:

```bash
for config in \
  pi05_a3_sim2real_move_cup \
  pi05_a3_sim2real_hang_blue_cup \
  pi05_a3_sim2real_put_spoon \
  pi05_a3_sim2real_put_yangtao \
  pi05_a3_sim2real_put_pass_bowl \
  pi05_a3_sim2real_put_yangtao_spoon; do
  .venv/bin/python scripts/compute_norm_stats.py --config-name "$config"
done
```

Example training command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
.venv/bin/python scripts/train.py pi05_a3_sim2real_move_cup \
  --exp-name=pi05_a3_sim2real_move_cup_h50 \
  --overwrite
```

The raw preprocess trajectories are format-compatible but are not all proven
safe for real-robot playback. In particular, formal `move_cup` training should
prefer independently accepted sessions from `move_cup_20260913_refined` when
that dataset is ready.
