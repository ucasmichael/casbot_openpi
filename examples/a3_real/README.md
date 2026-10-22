# A3 MCAP / IPC 数据转换为 LeRobot

本目录支持两种 A3 原始数据，输出项目所用的 **LeRobot v2.1**：

- `convert_a3_mcap_to_lerobot.py`：ROS MCAP。
- `convert_a3_ipc_to_lerobot.py`：`egodex.ipc_collection.v1`，即 `teleop_data` 中的 SQLite + HDF5 + JSONL 数据，无需先转成 MCAP。

## 当前输出契约与语义

**MCAP 和 IPC 均默认使用真实反馈 state、独立的 command action，维度为 37，只支持 `--joint-mode both`。** 旧文档的 17/34 维、只存 RGB、不存 depth 的描述不再适用于当前转换器。

| Feature | 类型与 shape | 内容 |
| --- | --- | --- |
| `observation.state` | float32 `(37,)` | 手臂/手部/腰部真实反馈；缺失时 warning 并回退到对应组的历史 command |
| `action` | float32 `(37,)` | 绝对控制目标，同一图像时刻的 command |
| `observation.images.chest_front_rgb` | RGB uint8 `(H,W,3)`，image | MCAP 胸部 RGB 或 IPC 的 D455 RGB |
| `observation.images.chest_front_depth` | uint16 PNG `(H,W,1)`，image | 毫米深度；IPC 使用对齐到 RGB 的 `depth_aligned` |
| task 文本 | LeRobot task 表 / `task_index` | 读取时由 LeRobot 返回 `task` |
| 索引与时间 | LeRobot 标准字段 | `episode_index`, `frame_index`, `index`, `timestamp` 等 |

37 维的固定顺序：

```text
[0:7]   左臂 7
[7:14]  右臂 7
[14:24] 左手 10
[24:34] 右手 10
[34:37] waist_roll, waist_pitch, waist_yaw
```

手臂和腰部使用 rad，手部保留原始编码器计数（正常范围 0–4095），不转换为 rad，也不在转换阶段做归一化。完整维度名写入 `meta/info.json` 的 feature names。

| 信号 | MCAP / IPC 对应 topic |
| --- | --- |
| arm command | `/motion/control/pnc/arm_joint_command` |
| hand command | `/motion/control/hand_joint_command` |
| waist command | `/motion/control/move_waist/pb_3Aaimdk_2Eprotocol_2EMotionControlMoveWaistChannel` |
| arm feedback | `/motion/control/arm_joint_state`，**没有 pnc 前缀** |
| hand feedback | `/motion/control/hand_joint_state` |
| waist feedback | `/motion/control/waist_joint_state` |

### state 默认反馈，缺失时回退历史 command

**两个入口都默认 `--state-source feedback`**：state 取图像时刻最近的实际反馈。按 arm（14）、hand（20）、waist（3）三组分别处理，反馈可用的组保留实测值；某组缺 topic、没有样本、同步超时、数值/关节布局异常（IPC 还包括时钟不确定度超限），则输出 `WARNING` 并改用该组在图像时刻或之前的最新 command。

回退 command 的年龄也不能超过 `--sync-tolerance-ms`（两个入口默认 100 ms）。**绝不使用未来 command 回填 state，也不无限延用过旧命令。** 历史 command 也不可用时，报告该帧无效：MCAP 跳过该帧；IPC 裁掉无效头尾，若无效帧位于中间则跳过整个 episode，以免压缩时间。

`action` 始终独立来自 command。两个入口都支持 `--command-alignment nearest|previous`，默认保留 `nearest`；它可以选到图像之后的 command。使用 `previous` 可限制 action 为不晚于图像的最后一条命令。这个参数**不影响 state 的回退规则**：回退始终使用 previous。因此，即使某组 state 发生回退，也不保证与该组 action 相等。

需要复现旧的 command-as-state 数据时，显式传 `--state-source command`，此时 state 与 action 数值相同。默认策略已改变，旧数据不会自动更新，训练时需要重新转换并计算归一化统计。

两个 reader 共用关节排序和手部边界检查：arm 按固定左右关节顺序、hand 按固定 20 维协议顺序、waist 按 roll/pitch/yaw 排列。手部 command 没有名称时使用协议顺序；反馈中的 `left_thumb_roration_0` 规范为 `left_thumb_rotation_0`，腰反馈的 `_joint` 后缀也统一处理。所有单位保持一致。

## 依赖

使用项目锁定的 LeRobot 版本，避免用最新版 LeRobot 替换 v2.1 接口：

```bash
cd /data/lyx/casbot_openpi
# 已建立项目 .venv 的情况下补齐 reader 依赖：
uv pip install --python .venv/bin/python mcap mcap-ros2-support h5py
```

`requirements.txt` 列出了转换依赖。IPC reader 使用标准库 SQLite；只有实际读取 MCAP 才需要导入 MCAP decoder。

## IPC / teleop_data 转换

原始目录可传集合根目录、session 或单个 episode，脚本递归发现 episode，保留完整来源路径，不会混淆不同 session 的同名 `episode_001`。

```text
move_cup_yaocao_1021/
  move_cup_20261021_.../
    session_meta.json
    episode_001/
      metadata.json
      prompt.txt
      camera.json
      clock_sync.json
      frames.jsonl
      alignment.jsonl
      robot.sqlite3
      rgbd.h5              # rgb, depth_raw, depth_aligned
```

先检查全部输入和时间匹配，不写 LeRobot 图像：

```bash
.venv/bin/python examples/a3_real/convert_a3_ipc_to_lerobot.py \
  --raw-dir /data/lyx/data/teleop_data/move_cup_yaocao_1021 \
  --dry-run \
  --report-path /tmp/a3_ipc_audit.json
```

按两个入口统一的默认 feedback 语义转换：

```bash
export HF_LEROBOT_HOME=/data/lyx/dataset/lerobot

.venv/bin/python examples/a3_real/convert_a3_ipc_to_lerobot.py \
  --raw-dir /data/lyx/data/teleop_data/move_cup_yaocao_1021 \
  --repo-id local_dataset/a3_teleop_move_cup_feedback \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --task "move the paper cup" \
  --mode image
```

如需 action 也只取历史 command，并把同步容差收紧到 80 ms：

```bash
.venv/bin/python examples/a3_real/convert_a3_ipc_to_lerobot.py \
  --raw-dir /data/lyx/data/teleop_data/move_cup_yaocao_1021 \
  --repo-id local_dataset/a3_teleop_move_cup_feedback \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --task "move the paper cup" \
  --state-source feedback \
  --command-alignment previous \
  --sync-tolerance-ms 80
```

`--task` 在 IPC 入口覆盖全部任务文本；未指定时依次读取 `prompt.txt`、`metadata.json.prompt`、`task_name`（下划线转为空格）。本批 prompt 均为空，建议显式写清实际任务。`--max-episodes 2 --max-frames-per-episode 60` 可生成小样本；`--overwrite` 会替换指定的已有输出目录。

### 时间、质量与图像处理

- 机器人消息使用 SQLite 的校准后 `ipc_time_ns`，图像使用 `frames.jsonl` 的 `ipc_time_ns` / `depth_ipc_time_ns`。原始 ROS header 与 RealSense wall clock 有偏移，不能直接混用。
- 为兼容 MCAP 的 nearest 规则，脚本从原始消息重新匹配；不直接套用 `alignment.jsonl` 中固定的 previous-command 引用。最终消息 ID、时间及偏差均保留。
- 默认限制单路匹配偏差 100 ms、消息时钟不确定度 5 ms、RGB/depth 偏差 10 ms，分别可通过 `--sync-tolerance-ms`、`--max-clock-uncertainty-ms`、`--max-rgb-depth-skew-ms` 调整。反馈和 command 都受同步容差约束。
- IPC 的采集总标记 `quality_passed=False` 会打印 warning 并记录，仍逐帧执行下述验证，以便处理因缺反馈而未通过采集检查的数据；未完成的 episode 或明确的采集 error 仍被拒绝。
- 先尝试 feedback，再按上述规则回退。回退后仍无效、action 不可用或 RGB/depth 时间不匹配的连续头尾帧被裁掉；中间出现无效帧则跳过整个 episode 并报告，避免把动作时间缺口压缩掉。
- 有效图像时间区间建立固定 `--fps` 网格（默认 30 Hz），选择最近图像，再使用该图像时刻的 state/action。默认网格与图像偏差最多 20 ms，可用 `--image-tolerance-ms` 调整。保留重复源帧映射；帧数可能与源视频不同。LeRobot 时间从 0 开始为 `frame_index / fps`，绝对 IPC 时间另存。
- 对手部值允许默认 1 count 的边界误差。feedback 的 4096 显式限幅至 4095 并逐帧计数；更大异常反馈会触发历史 command 回退，未落入匹配窗口的异常反馈不会污染输出。command 值保持 MCAP 的原始数值语义；严格边界检查可用 `--hand-range-tolerance 0`。
- RGB 已是 RGB8，不交换红蓝通道。深度按 `depth_scale_m` 转为 uint16 毫米（四舍五入、饱和至 65535）；约 0.001 m 的源数据逐值保留，包括 0 空洞。深度只做单位转换，不填洞。
- IPC 只支持 `--mode image`：RGB 和深度存为无损 PNG，并复用 MCAP 的外部图像写入方式。普通视频编码不能保证 uint16 深度精度。

### 与 MCAP 图像格式的差别

本批源相机是 **D455，640×480**；MCAP 样例是胸部 D457，1280×720。IPC 默认保留原生 shape `(480,640,3)` / `(480,640,1)`，模型的现有 RGB resize/pad 可直接接受。要与某个 1280×720 MCAP 数据集的 feature shape 完全相同，增加：

```bash
--image-size 1280 720
```

RGB 用双线性、深度用最近邻缩放。该选项直接改变宽高比，不恢复原相机视场；几何计算须按实际缩放更新内参。报告保留原始 calibration 和输入/输出 shape。

`chest_front_*` 只是现有训练接口使用的键名，不能证明 D455 安装于胸部，也不能把它标成 D457。没有该相机到躯干的标定，IPC **不执行旧 D457 的 FK 投影裁剪**。格式、关节单位和 command 语义可以对齐；视角、内外参以及深度像素对应关系不能靠改文件格式变得相同。

### 可追溯输出

除 LeRobot 标准 `meta/info.json`、task/episode/stat 文件、Parquet 与 PNG 外，还写入：

- `meta/ipc_conversion.json`：语义模式、参数、关节顺序和单位、逐 episode 相机标定、裁剪/重采样统计、各组回退计数和跳过原因。
- `meta/source_frames/episode_XXXXXX.jsonl`：每个输出帧对应的 HDF5 帧、IPC 时间、机器人消息 ID、同步偏差和反馈限幅计数。其中 `state_messages` 按 arm/hand/waist 记录实际来源（`feedback` / `historical_command` / `command`）、消息 ID、时间及回退原因；与 action 的消息引用分别保存。
- `meta/skipped_episodes.json`：未转换的 episode 及原因；全部无效时退出并报错。
- `meta/qpose_means.json`：state 的逐维均值，结构沿用 MCAP 输出。
- `meta/action_means.json`：action 的独立逐维均值。

`--dry-run --report-path ...` 检查所有选择的 episode 的结构、消息值和时间映射，不逐帧解码 HDF5 图像，也不代表已生成完整数据集。实际转换按帧读取图像，避免一次加载全部 RGBD。

## MCAP 转换

原始 episode 包含 `prompt.txt` / `metadata.json` 和下面目录中的 `*.mcap`。当前版本要求 RGB、depth、arm command、hand command、move_waist 五个 topic 齐全。

```bash
.venv/bin/python examples/a3_real/convert_a3_mcap_to_lerobot.py \
  --raw-dir /data/lyx/dataset/data0819_flat \
  --repo-id local_dataset/a3_0819_both_depth \
  --lerobot-root "$HF_LEROBOT_HOME" \
  --joint-mode both \
  --mode image \
  --no-projection-trim \
  --task "stack cups"
```

MCAP 入口的 `--task` 是 prompt 缺失时的 fallback。state 默认来自 feedback，缺失时按组 warning 并回退历史 command；action 默认取最近 command，可传 `--command-alignment previous`。RGB 为基准，输出逐帧时间由设定的 fps 生成；它没有 IPC 入口的规则时间网格重采样。

MCAP 额外写入 `meta/state_sources.json`，记录每个输出 episode 的策略、回退计数，以及每帧的源 RGB 序号、时间和各组 state 来源；`meta/action_means.json` 保存独立的 action 均值。反馈 topic 不再是必需输入，历史 command 能覆盖时仍可转换并报告回退。

不传 `--no-projection-trim` 时，旧入口默认启用其固定 D457 标定和 URDF 的投影裁剪，命令行默认最少保留 400 帧。仅在相机标定确实匹配时使用。该入口仍保留历史 `video` 参数，但含 uint16 深度的数据应使用 `image`。

## 训练接入

`src/openpi/policies/a3_policy.py` / `LeRobotA3DataConfig` 的双臂接口为 37 维：

```text
observation.images.chest_front_rgb -> base_0_rgb -> resize/pad 至 224×224
left_wrist_0_rgb / right_wrist_0_rgb -> zero padding，mask=false
observation.state -> state
当前及未来 action -> action_horizon × 37 的绝对动作序列
```

当前 A3 模型不读取 depth。转换格式兼容不意味着旧 normalization 或任务配置可以直接复用；需要给所选语义模式独立的 repo_id 和统计。

`pi05_a3_both` 已是 37 维，但仓库中它的 repo_id 仍指向旧数据，默认 prompt 也是旧任务。训练前应在 `src/openpi/training/config.py` 中为新数据添加/调整 `make_a3_train_config`，例如以下参数，并保留 `use_delta_joint_actions=False`：

```python
make_a3_train_config(
    name="pi05_a3_realteleop_move_cup",
    repo_id="local_dataset/a3_teleop_move_cup_feedback",
    action_dim=a3_policy.A3_BOTH_ACTION_DIM,
    prompt="move the paper cup",
    steps=30_000,
    action_horizon=50,
    batch_size=32,
    save_interval=10_000,
)
```

上例是待添加的训练配置，不是本次新增的配置名。配置就绪后再执行：

```bash
export HF_LEROBOT_HOME=/data/lyx/dataset/lerobot
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_a3_realteleop_move_cup
.venv/bin/python scripts/train.py pi05_a3_realteleop_move_cup --exp-name move_cup
```

历史 `pi05_a3_left/right` 是 17 维接口，不应直接用于此处的 37 维输出。

## 本批数据验证（2026-10-22）

输入 `/data/lyx/data/teleop_data/move_cup_yaocao_1021`：8 个 session、58 个 episode、18,457 个源 RGB 帧。

| 参数 | 通过 episode | 有效源帧 | 30 Hz 输出帧 | 处理 |
| --- | ---: | ---: | ---: | --- |
| 默认 feedback + nearest，100 ms | 58/58 | 18,383 | 18,446 | 7 段尾部共 74 帧过旧 action 被裁掉；有效帧均有反馈，回退 0 次 |
| feedback + previous，80 ms | 58/58 | 18,377 | 18,440 | 7 段尾部共 80 帧被裁掉；12 个反馈边界值限幅 |

这些数字来自实际全量预检；规则重采样会重复少量源图像，因此输出帧数可以大于有效源帧数。测试覆盖真实反馈、部分/全部反馈缺失、超时/无效反馈、禁止未来 command 回填，以及 MCAP/IPC 数值一致性和 LeRobot 写入回读。未执行全量图像导出或模型训练。

回归检查：

```bash
.venv/bin/pytest -q examples/a3_real/test_convert_a3_ipc_to_lerobot.py examples/a3_real/test_a3_feedback_alignment.py
```
