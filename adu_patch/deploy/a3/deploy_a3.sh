#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$DEPLOY_DIR/../.." && pwd)"
ADU_HOST="${ADU_HOST:-10.42.10.11}"
ADU_USER="${ADU_USER:-lyx}"
ADU_REPO="${ADU_REPO:-/agibot/data/home/agi/Desktop/policy/casbot_openpi}"
POLICY_HOST="${POLICY_HOST:-10.42.10.50}"
POLICY_PORT="${POLICY_PORT:-8000}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_a3_both}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/agilex/cobot_magic/aloha-devel/casbot_openpi/checkpoint/stack_cups_0826_teleop_21traj/20000}"
CAMERA_SOURCE="${CAMERA_SOURCE:-adu-d457}"

usage() {
    cat <<'EOF'
Usage:
  deploy/a3/deploy_a3.sh setup-adu
  deploy/a3/deploy_a3.sh extract [extract_init_pose.py arguments]
  deploy/a3/deploy_a3.sh adu [executor.py arguments]
  deploy/a3/deploy_a3.sh validate [executor.py arguments]
  deploy/a3/deploy_a3.sh launch [executor.py arguments]
  deploy/a3/deploy_a3.sh server [serve_policy.py arguments]

Environment overrides:
  ADU_HOST ADU_USER ADU_REPO POLICY_HOST POLICY_PORT POLICY_CONFIG CHECKPOINT_DIR CAMERA_SOURCE
EOF
}

setup_ros_environment() {
    set +u
    source /opt/ros/jazzy/setup.bash
    source /agibot/software/v0/entry/env/env.sh
    set -u
    local aima_prefix="/opt/agibot/share/ros2_package/aima_protocol_ros2_package"
    local aimrt_prefix="/opt/agibot/share/ros2_package/aimrt_protocol_ros2_package"
    export AMENT_PREFIX_PATH="$aima_prefix:$aimrt_prefix:${AMENT_PREFIX_PATH:-}"
    export PYTHONPATH="$REPO_ROOT:$aima_prefix/lib/python3.12/site-packages:$aimrt_prefix/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:$aima_prefix/lib:$aimrt_prefix/lib:${LD_LIBRARY_PATH:-}"
}

setup_adu_venv() {
    if [[ ! -x "$DEPLOY_DIR/.venv/bin/python" ]]; then
        /usr/bin/python3 -m venv --system-site-packages "$DEPLOY_DIR/.venv"
    fi
    "$DEPLOY_DIR/.venv/bin/python" -m pip install --disable-pip-version-check \
        -r "$DEPLOY_DIR/requirements-adu.txt"
}

run_remote() {
    local mode="$1"
    shift
    local remote=(
        "$ADU_REPO/deploy/a3/deploy_a3.sh"
        adu
        --policy-host "$POLICY_HOST"
        --policy-port "$POLICY_PORT"
    )
    if [[ "$mode" == "validate" ]]; then
        remote+=(--validate-only)
    fi
    remote+=("$@")
    local command
    printf -v command '%q ' "${remote[@]}"
    ssh -tt "$ADU_USER@$ADU_HOST" "$command"
}

command="${1:-}"
if [[ -z "$command" ]]; then
    usage
    exit 2
fi
shift

case "$command" in
    setup-adu)
        setup_adu_venv
        ;;
    extract)
        setup_ros_environment
        exec /usr/bin/python3 -u "$DEPLOY_DIR/extract_init_pose.py" "$@"
        ;;
    adu)
        setup_ros_environment
        if [[ ! -x "$DEPLOY_DIR/.venv/bin/python" ]]; then
            echo "Missing $DEPLOY_DIR/.venv; run '$0 setup-adu' first." >&2
            exit 1
        fi
        exec "$DEPLOY_DIR/.venv/bin/python" -u -m deploy.a3.executor "$@"
        ;;
    validate)
        timeout 5 bash -c ": > /dev/tcp/$POLICY_HOST/$POLICY_PORT"
        run_remote validate "$@"
        ;;
    launch)
        timeout 5 bash -c ": > /dev/tcp/$POLICY_HOST/$POLICY_PORT"
        run_remote launch "$@"
        ;;
    server)
        cd "$REPO_ROOT"
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export WANDB_MODE=disabled
        exec "$REPO_ROOT/venv/bin/python" -u scripts/serve_policy.py \
            --port "$POLICY_PORT" \
            --default-prompt "stack cups" \
            --camera-source "$CAMERA_SOURCE" \
            policy:checkpoint \
            --policy.config "$POLICY_CONFIG" \
            --policy.dir "$CHECKPOINT_DIR" \
            "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $command" >&2
        usage >&2
        exit 2
        ;;
esac
