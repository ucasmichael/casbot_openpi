# 0902 move cup normfix 部署

`--normfix` 默认关闭。仅对使用 0902 normfix 规则训练的权重显式开启；
输入归一化和输出反归一化会一起切换。无需修改 checkpoint 自带的 `norm_stats.json`。

## 工控机 server

```bash
cd /home/agilex/cobot_magic/aloha-devel/casbot_openpi

XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
venv/bin/python -u scripts/serve_policy.py \
  --port 8000 \
  --default-prompt "move cup" \
  --normfix \
  --record \
  policy:checkpoint \
  --policy.config pi05_a3_0902_move_cup_both \
  --policy.dir "$PWD/checkpoint/move_cup_0902_normfix/19999"
```

启动日志和 WebSocket metadata 的 `normalization` 中应出现 `mode: normfix`、
`use_quantiles: true`、`quantile_range_eps: 0.001`、`norm_abs_clip: 3.0`，
以及 state/actions 各自的冻结维度。`--record` 将输入和返回的动作保存到当前工作目录的
`policy_records/`，其中返回动作已经反归一化，但尚未经过 ADU 的关节限幅。

## ADU dry-run

先将初始化包转换为当前加载器支持的 `schema_version: 2`，确认以下文件已存在：

```text
/agibot/data/home/agi/Desktop/policy/casbot_openpi/deploy/a3/assets/0902_move_cup/init_pose_v2.json
```

在 ADU 上执行：

```bash
cd /agibot/data/home/agi/Desktop/policy/casbot_openpi

./deploy/a3/deploy_a3.sh adu \
  --policy-host 10.42.10.50 \
  --policy-port 8000 \
  --prompt "move cup" \
  --init-pose-file deploy/a3/assets/0902_move_cup/init_pose_v2.json \
  --two-stage-reset \
  --dry-run
```

此模式不创建控制指令 publisher，不需要实际运动确认 token，且不能同时传入非零
`--max-actions`。报告存放在 ADU 的 `~/a3_openpi_deploy_logs/` 下。

当前 ADU dry-run 使用 startup command 和实时 RGB 请求一个完整的 50 动作块。
它不会将机器人移动到 0902 初始化姿态，也没有完整模拟两阶段初始化与复位的所有插值段。
因此还需使用 0902 初始化包中的首帧 RGB 和 state 单独检查模型输出；
dry-run 通过不能证明机器人已经到位或运动路径不会碰到桌面。

实际运行时，同样传入上述 `--prompt`、`--init-pose-file`、`--two-stage-reset`。
`--normfix` 是工控机 server 的参数，不传给 ADU。

## 离线推理入口

`a3_inference.py` 也支持默认关闭的 `--normfix`，使用本次权重时需要明确指定：

```bash
venv/bin/python a3_inference.py \
  --config-name pi05_a3_0902_move_cup_both \
  --checkpoint "$PWD/checkpoint/move_cup_0902_normfix/19999" \
  --dataset-root "/实际路径/local_dataset/a3_0902_move_cup_50traj_both" \
  --prompt "move cup" \
  --normfix \
  --episode-index 24 \
  --start-frame 0 \
  --num-frames 10
```

该入口当前的误差汇总只比较每个预测块的第一个 action，不是整个动作块的误差。

## 计算规则与回归检查

规则与用户提供的 `0902_transforms.tar.gz` 一致：

- Quantile 范围小于 `0.001` 的维度在 Normalize 中返回 0，在 Unnormalize 中返回 q01。
- 正常维度使用实际范围，不添加旧实现的 `1e-6`。
- Normalize 计算后裁剪到 `[-3, 3]`，Unnormalize 计算前也裁剪到 `[-3, 3]`。
- 非 quantile 路径在显式开启 normfix 时使用对应训练版规则：`std < 1e-6` 的维度冻结为 mean。
- 关闭 normfix 时，继续使用原始计算规则。

在当前工控机环境中，按服务入口的原生库导入顺序运行相关测试：

```bash
JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  scripts/serve_policy_test.py \
  src/openpi/policies/policy_config_test.py \
  src/openpi/transforms_test.py
```
