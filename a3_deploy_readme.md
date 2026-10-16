# a3 deploy

0902 move cup normfix 权重的初始化、归一化开关和 dry-run 用法见
[0902 部署说明](docs/a3_0902_normfix.md)。

## server deploy
``` bash
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
> venv/bin/python -u scripts/serve_policy.py \
>   --port 8000 \
>   --default-prompt "stack cups" \
>   policy:checkpoint \
>   --policy.config pi05_a3_both \
>   --policy.dir /home/agilex/cobot_magic/aloha-devel/casbot_openpi/checkpoint/stack_cups_0819/29999
```

## ADU policy observation and step limits

The ADU executor uses fresh `/arm_joint_state`, `/hand_joint_state`, and
`/waist_joint_state` feedback as the policy state by default. This policy input
does not use `arm_state_to_command.json`. In the current compliance-control
deployment, startup and reset poses also use measured state directly, so the
executor does not load that projection file.

The policy step defaults are:

```text
arm:  0.02 rad/action
hand: 200 encoder counts/action
waist: 0.003 rad/action
```

Do not keep the old `--max-arm-step 0.01` argument when the intended arm limit
is `0.02`. Either omit both options to use the defaults, or specify them
explicitly:

```bash
./deploy/a3/deploy_a3.sh adu \
  ... \
  --observation-state-source measured \
  --max-arm-step 0.02 \
  --max-hand-step 200
```

Each real-motion run writes `run_config.json` and `commands.csv` under `logs/`.
The CSV includes the policy observation, returned action, scoped action,
published command, and measured feedback. Use
`--observation-state-source command` only for a legacy A/B comparison.

## RGB observation modes

`adu-d457` remains the default and preserves the existing deployment path: the
ADU reads `/hal/chest_front_d457_camera/rgb` and sends the image with the 37D
state to the policy server. Existing commands do not need a new argument.

```bash
# Policy server (workstation)
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
venv/bin/python -u scripts/serve_policy.py \
  --port 8000 \
  --camera-source adu-d457 \
  --default-prompt "move the paper cup" \
  policy:checkpoint \
  --policy.config pi05_a3_sim2real_move_cup \
  --policy.dir /path/to/checkpoint/20000

# Actor (ADU); --camera-source can be omitted because adu-d457 is the default.
./deploy/a3/deploy_a3.sh adu \
  --policy-host 10.42.10.50 \
  --policy-port 8000 \
  --camera-source adu-d457 \
  --prompt "move the paper cup" \
  --init-pose-file ./deploy/a3/assets/move_cup_20260913/init_pose_v2.json \
  --active-sides right \
  --no-lock-lower \
  --dry-run
```

`external-d455` captures RGB on the policy-server workstation. Install the
optional RealSense binding in the server environment once:

```bash
uv pip install --python venv/bin/python -r requirements-a3-realsense.txt
```

Start both ends with the same explicit camera source:

```bash
# Policy server (workstation)
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
venv/bin/python -u scripts/serve_policy.py \
  --port 8000 \
  --camera-source external-d455 \
  --camera-serial 318122300934 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --default-prompt "move the paper cup" \
  policy:checkpoint \
  --policy.config pi05_a3_sim2real_move_cup \
  --policy.dir /path/to/checkpoint/20000

# Actor (ADU), validation only: this never creates motion publishers.
./deploy/a3/deploy_a3.sh adu \
  --policy-host 10.42.10.50 \
  --policy-port 8000 \
  --camera-source external-d455 \
  --prompt "move the paper cup" \
  --init-pose-file ./deploy/a3/assets/move_cup_20260913/init_pose_v2.json \
  --active-sides right \
  --no-lock-lower \
  --validate-only
```

In external mode the ADU does not transmit an image. It sends bounded,
source-stamped arm/hand/waist histories. The server calibrates the two monotonic
clocks with seven minimum-RTT probes, requires uncertainty at most 5 ms, and
selects a D455 `global_time` RGB frame bracketed by all three state streams with
a total error bound at most 20 ms. Any stale state, camera restart, clock step,
wrong camera mode, or failed alignment stops inference instead of falling back
to receipt time. The matched state order remains
`arm14 + hand20 + waist[roll,pitch,yaw]`.

Each real-motion run also writes `inference.jsonl`. For external D455 runs it
contains the selected frame id, per-stream skew, clock uncertainty, frame age,
camera delivery time, exact aligned 37D state, and server inference timing for
post-run audit.
