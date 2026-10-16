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
