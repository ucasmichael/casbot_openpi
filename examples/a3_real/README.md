# A3 MCAP to LeRobot

这个目录用于把 `collect_data/collect_qpose_images.py` 采集的 A3 MCAP 数据转换成 LeRobot v2.1，并配套 `openpi` 训练配置。

## 原始数据结构

示例数据：`/data/lyx/dataset/a3/0730_test`

```text
0730_test/
  session_meta.json
  episode_001/
    prompt.txt
    metadata.json
    adu_all_YYYYMMDD_HHMMSS/
      metadata.yaml
      *.mcap
```

当前样例 MCAP 中实际写入的 topic 和 shape：

| Topic | ROS type | 频率/数量示例 | Shape |
| --- | --- | --- | --- |
| `/hal/chest_front_d457_camera/rgb` | `sensor_msgs/Image`, `bgr8` | 约 30 Hz | `720 x 1280 x 3`, uint8，转换后为 RGB |
| `/hal/chest_front_d457_camera/depth` | `sensor_msgs/Image`, `16UC1` | 约 30 Hz | `720 x 1280`, uint16 |
| `/motion/control/arm_joint_state` | `sensor_msgs/JointState` | 约 100 Hz | `14` |
| `/motion/control/hand_joint_state` | `sensor_msgs/JointState` | 约 100 Hz | `20` |
| `/motion/control/neck_joint_state` | `sensor_msgs/JointState` | 约 100 Hz | `2` |

训练数据只写入手臂和手的 qpose，不写入 neck/depth。转换脚本支持三种关节模式：

| `--joint-mode` | qpose 组成 | dim | 对应训练 config |
| --- | --- | --- | --- |
| `left` | 左臂 7 + 左手 10 | `17` | `pi05_a3_left` |
| `right` | 右臂 7 + 右手 10 | `17` | `pi05_a3_right` |
| `both` | 左臂 7 + 左手 10 + 右臂 7 + 右手 10 | `34` | `pi05_a3_both` |

`neck 2` 和未选中的另一侧 qpose 不会进入 LeRobot 训练 feature；转换结束时会打印这些未进入训练数据的 qpose 均值，便于检查数据分布。

样例第一帧关节名：

```text
arm:
left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint,
left_elbow_joint, left_wrist_roll_joint, left_wrist_pitch_joint, left_wrist_yaw_joint,
right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint,
right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint

hand:
left_thumb_roration_0, left_thumb_wiggles_1, left_thumb_bent_2,
left_index_wiggles_0, left_index_bent_1, left_middle_bent,
left_ring_wiggles_0, left_ring_bent_1, left_pinky_wiggles_0, left_pinky_bent_1,
right_thumb_rotation_0, right_thumb_wiggles_1, right_thumb_bent_2,
right_index_wiggles_0, right_index_bent_1, right_middle_bent,
right_ring_wiggles_0, right_ring_bent_1, right_pinky_wiggles_0, right_pinky_bent_1
```

## 转换

先确保依赖在项目环境中：

```bash
cd /data/lyx/casbot_openpi
uv pip install --python .venv/bin/python -r examples/a3_real/requirements.txt
```

转换为本地 LeRobot 数据集：

```bash
export HF_LEROBOT_HOME=/data/lyx/dataset/lerobot

.venv/bin/python examples/a3_real/convert_a3_mcap_to_lerobot.py \
  --raw-dir /data/lyx/dataset/a3/0730_test \
  --repo-id local_dataset/a3_0730_test_both \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --joint-mode both \
  --mode video \
  --overwrite
```

默认输出 feature：

```text
observation.state: (17,) 或 (34,), float32
action: (17,) 或 (34,), float32
observation.images.chest_front_rgb: (720, 1280, 3), video/image
task: 来自每个 episode 的 prompt.txt 或 metadata.json
```

转换左臂+左手：

```bash
.venv/bin/python examples/a3_real/convert_a3_mcap_to_lerobot.py \
  --raw-dir /data/lyx/dataset/a3/0730_test \
  --repo-id local_dataset/a3_0730_test_left \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --joint-mode left \
  --mode video \
  --overwrite
```

转换右臂+右手：

```bash
.venv/bin/python examples/a3_real/convert_a3_mcap_to_lerobot.py \
  --raw-dir /data/lyx/dataset/a3/0730_test \
  --repo-id local_dataset/a3_0730_test_right \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --joint-mode right \
  --mode video \
  --overwrite
```

Depth 和 neck 保留在原始 MCAP 中，不进入训练数据。

转换结束后，脚本会在 LeRobot 数据集的 `meta/qpose_means.json` 写入 `training_qpose` 和 `excluded_qpose` 的逐维均值、关节名和帧数；终端也会同步打印同样的均值，方便快速检查。

## 训练

新增配置名：

```text
pi05_a3_left
pi05_a3_right
pi05_a3_both
```

计算归一化统计：

```bash
cd /data/lyx/casbot_openpi
export HF_LEROBOT_HOME=/data/lyx/dataset/lerobot
export HF_DATASETS_CACHE=/data/lyx/hf_cache
uv run scripts/compute_norm_stats.py --config-name pi05_a3_right 
```

开始训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_a3_right --exp-name=pi05_a3_right_pick_cube --overwrite
```

如果使用 PyTorch 训练入口：

```bash
.venv/bin/python scripts/train_pytorch.py pi05_a3_both \
  --exp_name a3_0730_test
```

## 训练数据如何进入模型

转换脚本把每个 RGB 帧作为时间基准，并在 `80 ms` 默认容差内找最近的 arm/hand qpose。由于采集脚本目前只有 qpose，没有单独的 commanded action，LeRobot 的 `action` 保存为同一时刻选中关节的 qpose；训练 dataloader 会按 `action_horizon` 读取未来 qpose chunk 作为目标动作序列。

`src/openpi/policies/a3_policy.py` 的默认映射：

```text
LeRobot observation.images.chest_front_rgb -> OpenPI base_0_rgb
OpenPI left_wrist_0_rgb/right_wrist_0_rgb -> zero padding, mask=false
LeRobot observation.state/action -> 17 或 34 维 A3 arm+hand qpose
```
