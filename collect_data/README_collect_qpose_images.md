# collect_qpose_images.py 使用说明

这个脚本在 ADU 上运行，用来同步采集：

- ADU 胸部相机 RGB/Depth（当前仍为原始 Image topic）
- HDU 头部相机 H.265 压缩视频流
- HDU 关节 qpose / joint state

脚本会为每条数据保存 prompt、metadata 和 rosbag。rosbag 使用 MCAP 格式，并且所有 bag 都只写在 ADU。HDU 不再启动 recorder，也不会持久化保存数据。

## 运行前准备

1. 确认 ADU 和 HDU 都已经启动机器人 ROS 环境、相机和遥操作相关节点。
2. 在 ADU 上确认能看到胸部、头部 H.265 和关节 topic：

```bash
source /opt/ros/jazzy/setup.bash
source /agibot/software/v0/entry/env/env.sh
ros2 topic list | grep -E 'head_front.*h265|chest_front|joint_state'
```

3. 在 ADU 上进入脚本目录：

```bash
cd /agibot/data/home/agi/Desktop/collect_data
```

## 最常用命令

采集 10 条数据，启动后只输入一次固定 prompt：

```bash
./collect_qpose_images.py \
  --episodes 10 \
  --output-root /agibot/data/home/agi/Desktop/bags
```

也可以直接在命令行指定固定 prompt：

```bash
./collect_qpose_images.py \
  --episodes 10 \
  --prompt "pick up the cup and place it on the table" \
  --output-root /agibot/data/home/agi/Desktop/bags
```


## 交互流程

脚本启动后会先显示将要采集的 topic。

如果没有传 `--prompt`，脚本会问一次：

```text
本次采集固定 prompt:
```

之后每条 episode 都会复用这个 prompt，不需要重复输入。

每条数据的操作流程：

```text
[1/10] prompt: ...
准备好后按 Enter 开始采集。
采集中。按空格、Enter 或 q 结束本条；Ctrl+C 会先停止当前录制再退出。
```

也就是：

- 按 Enter：开始当前条采集
- 按空格、Enter 或 q：结束当前条采集
- Ctrl+C：停止当前录制并退出脚本

## 默认采集 topic

ADU 本地采集：

| Topic | 含义 |
| --- | --- |
| `/hal/chest_front_d457_camera/rgb` | ADU 胸部 D457 RGB 图像 |
| `/hal/chest_front_d457_camera/depth` | ADU 胸部 D457 Depth 深度图 |

HDU 来源 topic（由 ADU 本地 recorder 订阅并存储）：

| Topic | 含义 |
| --- | --- |
| `/hal/head_front_dual_fisheye_camera/h265` | HDU 头部双鱼眼 H.265 压缩视频流；消息类型为 `foxglove_msgs/msg/CompressedVideo` |
| `/motion/control/arm_joint_state` | 手臂关节状态，qpose 主要来源之一 |
| `/motion/control/hand_joint_state` | 手部关节状态，qpose 主要来源之一 |
| `/motion/control/neck_joint_state` | 颈部关节状态，qpose 主要来源之一 |

## 输出结构

默认输出根目录是：

```text
/agibot/data/home/agi/Desktop/bags
```

一次采集会生成一个 session 目录：

```text
collection_YYYYMMDD_HHMMSS/
  session_meta.json
  episode_001/
    prompt.txt
    metadata.json
    adu_all_YYYYMMDD_HHMMSS/
      metadata.yaml
      *.mcap
    adu_all.log
  episode_002/
    ...
```

每条 episode 的 `metadata.json` 会记录：

- prompt
- 开始和结束时间
- 录制时长
- ADU 上的 bag 路径
- ADU topic 和 HDU 来源 topic 列表
- 实际记录的 topic 列表

## 常用参数

查看完整参数：

```bash
./collect_qpose_images.py --help
```

只采集图像，不采集关节：

```bash
./collect_qpose_images.py --episodes 5 --no-joints
```

如需回退记录原始头部左相机 Image：

```bash
./collect_qpose_images.py \
  --episodes 5 \
  --head-cameras head_front_left_fisheye_camera \
  --head-stream image
```

传完整 topic 名也可以：

```bash
./collect_qpose_images.py \
  --episodes 5 \
  --head-cameras /hal/head_front_left_fisheye_camera/image \
  --chest-cameras /hal/chest_front_d457_camera/rgb
```

注意：如果 `--chest-cameras` 传的是相机名，会自动展开为 `/rgb` 和 `/depth`；如果传的是完整 `/topic`，就只记录这个 topic。

## 查看 MCAP 里有哪些 topic

录制完成后脚本会自动打印 `ros2 bag info`。也可以手动查看：

```bash
ros2 bag info /path/to/bag_directory
```

例如：

```bash
ros2 bag info /agibot/data/home/agi/Desktop/bags/collection_YYYYMMDD_HHMMSS/episode_001/adu_all_YYYYMMDD_HHMMSS
```


## 排错

如果 ADU 本地 recorder 起不来，先确认胸部相机 topic 是否存在：

```bash
source /opt/ros/jazzy/setup.bash
source /agibot/software/v0/entry/env/env.sh
ros2 topic list | grep chest_front_d457
```

如果脚本提示日志中有异常，查看对应 episode 目录里的：

```text
adu_all.log
```
