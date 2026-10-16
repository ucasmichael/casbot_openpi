#!/usr/bin/env python3
"""Interactive, guarded A3 executor for a remote pi05_a3_both policy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from dataclasses import replace
import fcntl
import json
import logging
import os
from pathlib import Path
import select
import signal
import sys
import termios
import time
import tty
from typing import Any

import numpy as np
import rclpy

from .core import DEFAULT_POLICY_ARM_STEP
from .core import DEFAULT_POLICY_HAND_STEP
from .core import DEFAULT_TRANSITION_ARM_STEP
from .core import DEFAULT_TRANSITION_HAND_STEP
from .core import DEFAULT_TRANSITION_WAIST_STEP
from .core import HARD_MAX_ARM_STEP
from .core import HARD_MAX_HAND_STEP
from .core import HARD_MAX_WAIST_STEP
from .core import MODEL_NAMES
from .core import Pose
from .core import analyze_action_chunk
from .core import bounded_action
from .core import interpolate_poses
from .core import load_init_pose_config
from .core import transition_step_count
from .core import validate_action_chunk
from .policy_client import CAMERA_SOURCE_ADU_D457
from .policy_client import CAMERA_SOURCE_EXTERNAL_D455
from .policy_client import InferenceWorker
from .policy_client import PolicyClient
from .policy_client import PolicyRequest
from .ros_node import A3DeploymentNode
from .ros_node import LowerLock
from .side_scope import A3SideScope
from .side_scope import parse_active_sides

CONFIRM_TOKEN = "A3_DEPLOY_REAL_MOTION"
DEFAULT_INIT_POSE = Path(__file__).with_name("init_pose.json")
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

INTERMEDIATE_ARM_STATE = np.asarray(
    [
        0.265156,
        0.418402,
        -0.187052,
        0.066849,
        -0.542635,
        -0.224250,
        -0.050413,
        0.333822,
        -0.405280,
        0.187662,
        0.026261,
        0.353727,
        -0.281928,
        0.152953,
    ],
    dtype=np.float64,
)
INTERMEDIATE_HAND_STATE = np.asarray(
    [
        3198,
        3693,
        50,
        3965,
        2,
        2,
        3968,
        4,
        65,
        4,
        3194,
        119,
        25,
        37,
        1,
        2,
        3622,
        7,
        3919,
        11,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class MotionResult:
    pose: Pose
    fault: str | None = None
    operator_stopped: bool = False


@dataclass(frozen=True)
class PolicyObservation:
    state_pose: Pose
    measured_pose: Pose
    state_ages_ms: tuple[float, float, float]
    source: str
    source_state_window: dict[str, Any] | None = None


class KeyReader:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.original: list[Any] | None = None

    def __enter__(self) -> KeyReader:
        if not sys.stdin.isatty():
            raise RuntimeError("interactive motion requires a TTY; use ssh -tt")
        self.original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.original is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)

    def poll(self) -> str | None:
        readable, _, _ = select.select([self.fd], [], [], 0.0)
        if not readable:
            return None
        char = os.read(self.fd, 1).decode("utf-8", errors="ignore")
        if char.lower() == "q":
            return "q"
        if char in ("\r", "\n"):
            return "enter"
        return char


def _json_arg(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, tuple):
        return list(value)
    return value


def build_run_config(
    args: argparse.Namespace,
    policy_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_args = {
        key: ("<provided>" if key == "confirm_token" and value else _json_arg(value))
        for key, value in vars(args).items()
    }
    return {
        "schema_version": 1,
        "argv": list(sys.argv),
        "effective_args": effective_args,
        "effective_step_limits": {
            "arm": args.max_arm_step,
            "hand": args.max_hand_step,
            "waist": args.max_waist_step,
        },
        "hard_step_limits": {
            "arm": HARD_MAX_ARM_STEP,
            "hand": HARD_MAX_HAND_STEP,
            "waist": HARD_MAX_WAIST_STEP,
        },
        "observation_state_source": args.observation_state_source,
        "observation_state_semantics": (
            "raw_measured_arm_hand_waist_feedback"
            if args.observation_state_source == "measured"
            else "last_published_command"
        ),
        "uses_arm_state_to_command_projection": False,
        "policy_metadata": policy_metadata,
    }


class CommandLog:
    def __init__(self, root: Path, *, run_config: dict[str, Any]) -> None:
        self.run_dir = root.expanduser() / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.file = (self.run_dir / "commands.csv").open("w", encoding="utf-8", newline="", buffering=1)
        self.writer = csv.writer(self.file)
        self.inference_file = (self.run_dir / "inference.jsonl").open("w", encoding="utf-8", buffering=1)
        self.writer.writerow(
            [
                "wall_time",
                "state",
                "chunk",
                "step",
                "round_trip_ms",
                "requested_arm_max_delta",
                "requested_hand_max_delta",
                "requested_waist_max_delta",
                "bounded_arm_max_delta",
                "bounded_hand_max_delta",
                "bounded_waist_max_delta",
                "arm_state_age_ms",
                "hand_state_age_ms",
                "waist_state_age_ms",
                "configured_arm_max_step",
                "configured_hand_max_step",
                "configured_waist_max_step",
                "observation_arm_state_age_ms",
                "observation_hand_state_age_ms",
                "observation_waist_state_age_ms",
                *[f"command:{name}" for name in MODEL_NAMES],
                *[f"feedback:{name}" for name in MODEL_NAMES],
                *[f"tracking_reference:{name}" for name in MODEL_NAMES],
                *[f"tracking_error:{name}" for name in MODEL_NAMES],
                *[f"observation:{name}" for name in MODEL_NAMES],
                *[f"raw_action:{name}" for name in MODEL_NAMES],
                *[f"scoped_action:{name}" for name in MODEL_NAMES],
            ]
        )

    def write_action(
        self,
        *,
        chunk: int,
        step: int,
        round_trip_ms: float,
        stats: dict[str, float],
        pose: Pose,
        feedback: Pose,
        tracking_reference: Pose,
        state_ages_ms: tuple[float, float, float],
        configured_steps: tuple[float, float, float],
        observation: PolicyObservation,
        raw_action: np.ndarray,
        scoped_action: np.ndarray,
    ) -> None:
        keys = (
            "requested_arm_max_delta",
            "requested_hand_max_delta",
            "requested_waist_max_delta",
            "bounded_arm_max_delta",
            "bounded_hand_max_delta",
            "bounded_waist_max_delta",
        )
        self.writer.writerow(
            [time.time(), "RUNNING", chunk, step, round_trip_ms]
            + [stats[key] for key in keys]
            + list(state_ages_ms)
            + list(configured_steps)
            + list(observation.state_ages_ms)
            + pose.state().tolist()
            + feedback.state().tolist()
            + tracking_reference.state().tolist()
            + (feedback.state() - tracking_reference.state()).tolist()
            + observation.state_pose.state().tolist()
            + np.asarray(raw_action, dtype=np.float64).tolist()
            + np.asarray(scoped_action, dtype=np.float64).tolist()
        )

    def close(self) -> None:
        self.file.close()
        self.inference_file.close()

    def write_inference(
        self,
        *,
        chunk: int,
        response: Any,
        observation: PolicyObservation,
    ) -> None:
        assert response.result is not None
        record = {
            "wall_time": time.time(),
            "chunk": chunk,
            "round_trip_ms": response.round_trip_ms,
            "request_bytes": response.request_bytes,
            "response_bytes": response.response_bytes,
            "observation_state_source": observation.source,
            "observation_state_ages_ms": list(observation.state_ages_ms),
            "observation_state": observation.state_pose.state().tolist(),
            "server_timing": response.result.get("server_timing"),
            "observation_alignment": response.result.get("observation_alignment"),
        }
        self.inference_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="10.42.10.50")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument(
        "--camera-source",
        choices=(CAMERA_SOURCE_ADU_D457, CAMERA_SOURCE_EXTERNAL_D455),
        default=CAMERA_SOURCE_ADU_D457,
        help="adu-d457 preserves the existing robot camera path; external-d455 aligns server RGB with ADU state",
    )
    parser.add_argument("--prompt", default="stack cups")
    parser.add_argument("--init-pose-file", type=Path, default=DEFAULT_INIT_POSE)
    parser.add_argument("--init-pose-id", default=None)
    parser.add_argument(
        "--active-sides",
        default="left,right",
        help="Comma-separated active A3 sides: left, right, or left,right",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="request and simulate one action chunk without creating command publishers",
    )
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument(
        "--two-stage-reset",
        action="store_true",
        help="route initialization and reset through the safe intermediate pose",
    )
    parser.add_argument("--confirm-token", default="")
    parser.add_argument("--allow-command-conflicts", action="store_true")
    parser.add_argument(
        "--lock-lower",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="continuously hold startup neck and leg poses",
    )
    parser.add_argument("--wait-ready-timeout", type=float, default=10.0)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--infer-timeout", type=float, default=60.0)
    parser.add_argument("--max-state-age", type=float, default=0.15)
    parser.add_argument("--max-image-age", type=float, default=0.25)
    parser.add_argument(
        "--observation-state-source",
        choices=("measured", "command"),
        default="measured",
        help=(
            "measured uses fresh arm/hand/waist feedback directly; "
            "command preserves the legacy last-published-command input"
        ),
    )
    parser.add_argument("--control-rate", type=float, default=30.0)
    parser.add_argument("--hold-rate", type=float, default=50.0)
    parser.add_argument("--init-duration", type=float, default=2.0)
    parser.add_argument("--reset-duration", type=float, default=2.0)
    parser.add_argument("--max-arm-step", type=float, default=DEFAULT_POLICY_ARM_STEP)
    parser.add_argument("--max-hand-step", type=float, default=DEFAULT_POLICY_HAND_STEP)
    parser.add_argument("--max-waist-step", type=float, default=HARD_MAX_WAIST_STEP)
    parser.add_argument("--max-actions", type=int, default=0, help="0 runs until q")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path("/tmp/openpi_a3_deployment.lock"),
    )
    args = parser.parse_args()
    try:
        args.active_sides = parse_active_sides(args.active_sides)
    except ValueError as error:
        parser.error(str(error))
    if args.policy_port <= 0 or args.policy_port > 65535:
        parser.error("--policy-port must be in [1,65535]")
    if (
        min(
            args.wait_ready_timeout,
            args.connect_timeout,
            args.infer_timeout,
            args.max_state_age,
            args.max_image_age,
            args.control_rate,
            args.hold_rate,
        )
        <= 0
    ):
        parser.error("timeouts, ages, and rates must be positive")
    if args.init_duration < 0 or args.reset_duration < 0:
        parser.error("transition durations cannot be negative")
    if not 0 < args.max_arm_step <= HARD_MAX_ARM_STEP:
        parser.error(f"--max-arm-step must be in (0,{HARD_MAX_ARM_STEP}]")
    if not 0 < args.max_hand_step <= HARD_MAX_HAND_STEP:
        parser.error(f"--max-hand-step must be in (0,{HARD_MAX_HAND_STEP}]")
    if not 0 < args.max_waist_step <= HARD_MAX_WAIST_STEP:
        parser.error(f"--max-waist-step must be in (0,{HARD_MAX_WAIST_STEP}]")
    if args.max_actions < 0:
        parser.error("--max-actions cannot be negative")
    if sum((args.validate_only, args.dry_run, args.init_only)) > 1:
        parser.error("--validate-only, --dry-run, and --init-only are mutually exclusive")
    if args.dry_run and args.max_actions:
        parser.error("--dry-run always validates one complete 50-action chunk; omit --max-actions")
    read_only = args.validate_only or args.dry_run
    if not read_only and args.confirm_token != CONFIRM_TOKEN:
        parser.error(f"real motion requires --confirm-token {CONFIRM_TOKEN}")
    if not read_only and not args.allow_command_conflicts:
        parser.error("real motion requires --allow-command-conflicts")
    return args


def _timed_tick(started: float, rate_hz: float) -> None:
    time.sleep(max(0.0, 1.0 / rate_hz - (time.monotonic() - started)))


def publish_hold_once(
    node: A3DeploymentNode,
    pose: Pose,
    *,
    candidate: Any,
    lower_lock: LowerLock | None,
    side_scope: A3SideScope,
) -> Pose:
    node.assert_spin_healthy()
    return node.publish_pose(
        pose,
        arm_template=candidate.arm_template,
        hand_template=candidate.hand_template,
        waist_height=candidate.waist_height,
        lower_lock=lower_lock,
        side_scope=side_scope,
    )


def transition(
    node: A3DeploymentNode,
    start: Pose,
    target: Pose,
    *,
    candidate: Any,
    lower_lock: LowerLock | None,
    duration_s: float,
    args: argparse.Namespace,
    keys: KeyReader,
    label: str,
    side_scope: A3SideScope,
) -> MotionResult:
    steps = transition_step_count(
        start,
        target,
        rate_hz=args.hold_rate,
        min_duration_s=duration_s,
        max_arm_step=min(DEFAULT_TRANSITION_ARM_STEP, args.max_arm_step),
        max_hand_step=min(DEFAULT_TRANSITION_HAND_STEP, args.max_hand_step),
        max_waist_step=min(DEFAULT_TRANSITION_WAIST_STEP, args.max_waist_step),
    )
    logging.warning("state=%s steps=%d minimum_estimated_s=%.2f", label, steps, steps / args.hold_rate)
    frames = interpolate_poses(
        start,
        target,
        rate_hz=args.hold_rate,
        min_duration_s=duration_s,
        max_arm_step=min(DEFAULT_TRANSITION_ARM_STEP, args.max_arm_step),
        max_hand_step=min(DEFAULT_TRANSITION_HAND_STEP, args.max_hand_step),
        max_waist_step=min(DEFAULT_TRANSITION_WAIST_STEP, args.max_waist_step),
    )
    current = start
    for frame in frames:
        started = time.monotonic()
        if keys.poll() == "q":
            logging.warning("q received during %s; holding current command", label)
            return MotionResult(current, operator_stopped=True)
        try:
            node.measured_pose()
        except (RuntimeError, ValueError) as error:
            message = f"{label}: {error}"
            logging.error("safety stop during %s: %s; holding current command", label, error)
            return MotionResult(current, fault=message)
        current = publish_hold_once(
            node,
            frame,
            candidate=candidate,
            lower_lock=lower_lock,
            side_scope=side_scope,
        )
        _timed_tick(started, args.hold_rate)
    return MotionResult(current)


def wait_with_hold(
    node: A3DeploymentNode,
    pose: Pose,
    *,
    candidate: Any,
    lower_lock: LowerLock | None,
    keys: KeyReader,
    args: argparse.Namespace,
    prompt: str,
    accepted: set[str],
    side_scope: A3SideScope,
) -> str:
    print(prompt, flush=True)
    while rclpy.ok():
        started = time.monotonic()
        key = keys.poll()
        if key in accepted:
            return key
        publish_hold_once(node, pose, candidate=candidate, lower_lock=lower_lock, side_scope=side_scope)
        _timed_tick(started, args.hold_rate)
    raise RuntimeError("ROS shutdown while holding pose")


def hold_for(
    node: A3DeploymentNode,
    pose: Pose,
    *,
    candidate: Any,
    lower_lock: LowerLock | None,
    args: argparse.Namespace,
    duration_s: float,
    side_scope: A3SideScope,
) -> None:
    deadline = time.monotonic() + duration_s
    while rclpy.ok() and time.monotonic() < deadline:
        started = time.monotonic()
        publish_hold_once(node, pose, candidate=candidate, lower_lock=lower_lock, side_scope=side_scope)
        _timed_tick(started, args.hold_rate)


def capture_policy_observation(
    node: A3DeploymentNode,
    command_pose: Pose,
    args: argparse.Namespace,
    camera_source: str,
) -> PolicyObservation:
    if camera_source == CAMERA_SOURCE_EXTERNAL_D455 and args.observation_state_source != "measured":
        raise RuntimeError("external-d455 requires --observation-state-source measured")
    measured, timestamps = node.measured_sample()
    sampled_at = time.monotonic()
    state_ages_ms = tuple((sampled_at - timestamp) * 1000.0 for timestamp in timestamps)
    state_pose = measured if args.observation_state_source == "measured" else command_pose
    return PolicyObservation(
        state_pose=state_pose,
        measured_pose=measured,
        state_ages_ms=state_ages_ms,
        source=args.observation_state_source,
        source_state_window=(node.source_state_window() if camera_source == CAMERA_SOURCE_EXTERNAL_D455 else None),
    )


def make_policy_request(
    node: A3DeploymentNode,
    sample: PolicyObservation,
    args: argparse.Namespace,
    side_scope: A3SideScope,
    camera_source: str,
) -> PolicyRequest:
    include_image = camera_source == CAMERA_SOURCE_ADU_D457
    observation = node.observation(
        sample.state_pose,
        args.prompt,
        active_sides=side_scope.active_sides,
        include_image=include_image,
    )
    return PolicyRequest(
        observation=observation,
        source_state_window=sample.source_state_window,
    )


def warm_policy(
    node: A3DeploymentNode,
    command_pose: Pose,
    args: argparse.Namespace,
    side_scope: A3SideScope,
) -> tuple[np.ndarray, dict[str, Any], PolicyObservation, str]:
    client = PolicyClient(
        args.policy_host,
        args.policy_port,
        connect_timeout_s=args.connect_timeout,
        infer_timeout_s=args.infer_timeout,
        expected_camera_source=args.camera_source,
    )
    try:
        started = time.perf_counter()
        metadata = client.open()
        camera_source = client.camera_source
        sample = capture_policy_observation(node, command_pose, args, camera_source)
        request = make_policy_request(node, sample, args, side_scope, camera_source)
        result, request_bytes, response_bytes = client.infer(request)
        actions = validate_action_chunk(result.get("actions"))
        logging.info(
            "validate inference shape=%s rtt_ms=%.1f request_bytes=%d response_bytes=%d metadata=%s",
            actions.shape,
            (time.perf_counter() - started) * 1000,
            request_bytes,
            response_bytes,
            metadata,
        )
        logging.info("policy camera_source=%s", camera_source)
        return actions, metadata, sample, camera_source
    finally:
        client.close()


def capture_startup_command(node: A3DeploymentNode) -> tuple[Pose, str]:
    return node.measured_pose(), "live_measured_feedback"


def build_intermediate_pose(startup: Pose) -> Pose:
    return Pose(
        arm=INTERMEDIATE_ARM_STATE.copy(),
        hand=np.clip(INTERMEDIATE_HAND_STATE, 0.0, 4095.0),
        waist=startup.waist.copy(),
    )


def final_reset_pose(startup: Pose, intermediate: Pose) -> Pose:
    return Pose(startup.arm.copy(), startup.hand.copy(), intermediate.waist.copy())


def execute_policy(
    node: A3DeploymentNode,
    initial_command: Pose,
    *,
    candidate: Any,
    lower_lock: LowerLock | None,
    keys: KeyReader,
    args: argparse.Namespace,
    command_log: CommandLog,
    side_scope: A3SideScope,
    camera_source: str,
) -> MotionResult:
    current = initial_command
    worker = InferenceWorker(
        PolicyClient(
            args.policy_host,
            args.policy_port,
            connect_timeout_s=args.connect_timeout,
            infer_timeout_s=args.infer_timeout,
            expected_camera_source=camera_source,
        )
    )
    worker.start()
    chunk = 0
    action_count = 0
    try:
        while rclpy.ok():
            try:
                observation_holder: list[PolicyObservation] = []

                def build_external_request(
                    current_command: Pose = current,
                    holder: list[PolicyObservation] = observation_holder,
                ) -> PolicyRequest:
                    sample = capture_policy_observation(node, current_command, args, camera_source)
                    holder.append(sample)
                    return make_policy_request(node, sample, args, side_scope, camera_source)

                if camera_source == CAMERA_SOURCE_EXTERNAL_D455:
                    inference_request = build_external_request
                else:
                    policy_observation = capture_policy_observation(node, current, args, camera_source)
                    observation_holder.append(policy_observation)
                    inference_request = make_policy_request(node, policy_observation, args, side_scope, camera_source)
            except (RuntimeError, ValueError) as error:
                message = f"observation: {error}"
                logging.error("observation safety stop: %s; holding current command", error)
                return MotionResult(current, fault=message)
            worker.submit(chunk, inference_request)
            response = None
            while response is None:
                started = time.monotonic()
                if keys.poll() == "q":
                    logging.warning("q received; stopping inference and action execution")
                    return MotionResult(current, operator_stopped=True)
                try:
                    node.measured_pose()
                except (RuntimeError, ValueError) as error:
                    message = f"inference wait feedback: {error}"
                    logging.error("inference-wait feedback safety stop: %s; holding current command", error)
                    return MotionResult(current, fault=message)
                publish_hold_once(node, current, candidate=candidate, lower_lock=lower_lock, side_scope=side_scope)
                response = worker.poll()
                _timed_tick(started, args.hold_rate)
            if response.error is not None:
                message = f"policy inference: {response.error}"
                logging.error("policy inference failed: %s; holding current command", response.error)
                return MotionResult(current, fault=message)
            assert response.result is not None
            if not observation_holder:
                message = "policy inference completed without a captured observation"
                logging.error(message)
                return MotionResult(current, fault=message)
            policy_observation = observation_holder[-1]
            try:
                actions = validate_action_chunk(response.result.get("actions"))
            except ValueError as error:
                message = f"invalid policy action chunk: {error}"
                logging.error("invalid policy action chunk: %s; holding current command", error)
                return MotionResult(current, fault=message)
            logging.info(
                "chunk=%d rtt_ms=%.1f request_bytes=%d response_bytes=%d",
                chunk,
                response.round_trip_ms,
                response.request_bytes,
                response.response_bytes,
            )
            command_log.write_inference(
                chunk=chunk,
                response=response,
                observation=policy_observation,
            )
            for step, requested in enumerate(actions):
                started = time.monotonic()
                if keys.poll() == "q":
                    logging.warning("q received; stopping at chunk=%d step=%d", chunk, step)
                    return MotionResult(current, operator_stopped=True)
                try:
                    live_pose, state_timestamps = node.measured_sample()
                    sampled_at = time.monotonic()
                    state_ages_ms = tuple((sampled_at - timestamp) * 1000.0 for timestamp in state_timestamps)
                    tracking_reference = current
                    raw_action = np.asarray(requested, dtype=np.float64).copy()
                    scoped_action = side_scope.hold_state(raw_action, live_pose.state())
                    next_command, stats = bounded_action(
                        current,
                        scoped_action,
                        max_arm_step=args.max_arm_step,
                        max_hand_step=args.max_hand_step,
                        max_waist_step=args.max_waist_step,
                    )
                    next_command = side_scope.hold_pose(next_command, live_pose)
                except (RuntimeError, ValueError) as error:
                    message = f"action chunk={chunk} step={step}: {error}"
                    logging.error(
                        "action safety stop at chunk=%d step=%d: %s; holding current command",
                        chunk,
                        step,
                        error,
                    )
                    return MotionResult(current, fault=message)
                current = publish_hold_once(
                    node,
                    next_command,
                    candidate=candidate,
                    lower_lock=lower_lock,
                    side_scope=side_scope,
                )
                command_log.write_action(
                    chunk=chunk,
                    step=step,
                    round_trip_ms=response.round_trip_ms,
                    stats=stats,
                    pose=current,
                    feedback=live_pose,
                    tracking_reference=tracking_reference,
                    state_ages_ms=state_ages_ms,
                    configured_steps=(
                        args.max_arm_step,
                        args.max_hand_step,
                        args.max_waist_step,
                    ),
                    observation=policy_observation,
                    raw_action=raw_action,
                    scoped_action=scoped_action,
                )
                action_count += 1
                if args.max_actions and action_count >= args.max_actions:
                    logging.warning("reached --max-actions=%d", args.max_actions)
                    return MotionResult(current)
                _timed_tick(started, args.control_rate)
            chunk += 1
    finally:
        worker.stop()
    return MotionResult(current)


def run_policy_dry_run(
    actions: np.ndarray,
    startup_command: Pose,
    *,
    startup_source: str,
    candidate: Any,
    conflicts: dict[str, list[str]],
    metadata: dict[str, Any],
    policy_observation: PolicyObservation,
    args: argparse.Namespace,
    intermediate_pose: Pose,
    side_scope: A3SideScope,
) -> int:
    actions = side_scope.hold_actions(actions, startup_command.state())
    inactive_parts = []
    if "left" in side_scope.inactive_sides:
        inactive_parts.extend((np.arange(0, 7), np.arange(14, 24)))
    if "right" in side_scope.inactive_sides:
        inactive_parts.extend((np.arange(7, 14), np.arange(24, 34)))
    inactive_indices = (
        np.sort(np.concatenate(inactive_parts).astype(np.int64, copy=False))
        if inactive_parts
        else np.empty(0, dtype=np.int64)
    )
    startup_state = startup_command.state()
    init_state = candidate.pose.state()
    intermediate_state = intermediate_pose.state()
    inactive_action_error = (
        float(np.max(np.abs(actions[:, inactive_indices] - startup_state[inactive_indices])))
        if inactive_indices.size
        else 0.0
    )
    _, action_report = analyze_action_chunk(
        startup_command,
        actions,
        max_arm_step=args.max_arm_step,
        max_hand_step=args.max_hand_step,
        max_waist_step=args.max_waist_step,
    )
    init_steps = transition_step_count(
        startup_command,
        candidate.pose,
        rate_hz=args.hold_rate,
        min_duration_s=args.init_duration,
        max_arm_step=min(DEFAULT_TRANSITION_ARM_STEP, args.max_arm_step),
        max_hand_step=min(DEFAULT_TRANSITION_HAND_STEP, args.max_hand_step),
        max_waist_step=min(DEFAULT_TRANSITION_WAIST_STEP, args.max_waist_step),
    )
    report = {
        "mode": "dry_run",
        "publishers_created": False,
        "policy": f"{args.policy_host}:{args.policy_port}",
        "policy_metadata": metadata,
        "init_pose_id": candidate.pose_id,
        "two_stage_reset": args.two_stage_reset,
        "active_sides": list(side_scope.active_sides),
        "inactive_sides": list(side_scope.inactive_sides),
        "inactive_output": "live_measured_state",
        "side_scope_check": {
            "inactive_action_max_error": inactive_action_error,
            "init_inactive_max_error": (
                float(np.max(np.abs(init_state[inactive_indices] - startup_state[inactive_indices])))
                if inactive_indices.size
                else 0.0
            ),
            "intermediate_inactive_max_error": (
                float(np.max(np.abs(intermediate_state[inactive_indices] - startup_state[inactive_indices])))
                if inactive_indices.size
                else 0.0
            ),
        },
        "startup_state": startup_state.tolist(),
        "scoped_init_state": init_state.tolist(),
        "scoped_intermediate_state": intermediate_state.tolist(),
        "startup_source": startup_source,
        "observation_state_source": policy_observation.source,
        "observation_state": policy_observation.state_pose.state().tolist(),
        "observation_measured_state": policy_observation.measured_pose.state().tolist(),
        "observation_state_ages_ms": list(policy_observation.state_ages_ms),
        "command_state_tracking": "disabled",
        "runtime_config": build_run_config(args, metadata),
        "existing_command_publishers": conflicts,
        "init_transition": {
            "two_stage": args.two_stage_reset,
            "steps": init_steps,
            "rate_hz": args.hold_rate,
            "estimated_s": init_steps / args.hold_rate,
            "max_arm_step": min(DEFAULT_TRANSITION_ARM_STEP, args.max_arm_step),
            "max_hand_step": min(DEFAULT_TRANSITION_HAND_STEP, args.max_hand_step),
            "max_waist_step": min(DEFAULT_TRANSITION_WAIST_STEP, args.max_waist_step),
        },
        "intermediate_pose": intermediate_pose.state().tolist(),
        "reset_waist_matches_startup": bool(np.array_equal(intermediate_pose.waist, startup_command.waist)),
        "action_chunk": action_report,
    }
    run_dir = args.log_dir.expanduser() / (time.strftime("%Y%m%d_%H%M%S") + f"_dry_run_{os.getpid()}")
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "dry_run.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rejected = (
        action_report["raw_limit_violation_count"]["arm"] + action_report["raw_limit_violation_count"]["waist"]
    ) > 0
    hand_clipped = action_report["raw_limit_violation_count"]["hand"] > 0
    status = "DRY_RUN_REJECTED" if rejected else ("DRY_RUN_OK_HAND_CLIPPED" if hand_clipped else "DRY_RUN_OK")
    action_count = action_report["actions"]
    raw_violations = action_report["raw_limit_violation_count"]
    limited_actions = action_report["delta_limited_action_count"]
    bounded_delta = action_report["bounded_max_delta"]
    print(
        f"{status} publishers_created=false actions={action_count} "
        f"raw_limit_violations={raw_violations} "
        f"delta_limited_actions={limited_actions} "
        f"bounded_max_delta={bounded_delta} init_steps={init_steps} report={report_path}",
        flush=True,
    )
    return 2 if rejected else 0


def run(args: argparse.Namespace) -> int:
    side_scope = A3SideScope(args.active_sides)
    logging.info(
        "active_sides=%s inactive_sides=%s",
        side_scope.active_sides,
        side_scope.inactive_sides,
    )
    logging.info(
        "observation_state_source=%s effective_step_limits arm=%.6f hand=%.1f waist=%.6f",
        args.observation_state_source,
        args.max_arm_step,
        args.max_hand_step,
        args.max_waist_step,
    )
    if args.max_arm_step < DEFAULT_POLICY_ARM_STEP:
        logging.warning(
            "configured --max-arm-step %.6f is below the default %.6f",
            args.max_arm_step,
            DEFAULT_POLICY_ARM_STEP,
        )
    if args.max_hand_step < DEFAULT_POLICY_HAND_STEP:
        logging.warning(
            "configured --max-hand-step %.1f is below the default %.1f",
            args.max_hand_step,
            DEFAULT_POLICY_HAND_STEP,
        )
    config = load_init_pose_config(args.init_pose_file)
    candidate = config.select(args.init_pose_id)
    logging.warning(
        "command execution remains open-loop within each action chunk; policy observations use %s state",
        args.observation_state_source,
    )
    logging.info(
        "selected init pose=%s source=%s sync_ms=%s",
        candidate.pose_id,
        candidate.source_bag,
        candidate.synchronization_ms,
    )
    lock_handle = args.lock_file.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another A3 deployment process holds {args.lock_file}") from error
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()

    rclpy.init()
    node = A3DeploymentNode(
        max_state_age_s=args.max_state_age,
        max_image_age_s=args.max_image_age,
    )
    command_log = None
    current_command = None
    publishers_enabled = False
    lower_lock = None
    node.start_background_spin()
    try:
        node.wait_until_ready(
            require_commands=False,
            require_lower=args.lock_lower,
            require_image=args.camera_source == CAMERA_SOURCE_ADU_D457,
            timeout_s=args.wait_ready_timeout,
        )
        startup_command, startup_source = capture_startup_command(node)
        intermediate_pose = build_intermediate_pose(startup_command)
        final_reset = final_reset_pose(startup_command, intermediate_pose)
        lower_lock = node.lower_lock() if args.lock_lower else None
        graph_deadline = time.monotonic() + 0.5
        while time.monotonic() < graph_deadline:
            node.assert_spin_healthy()
            time.sleep(0.05)
        conflicts = node.publisher_conflicts(include_lower=args.lock_lower)
        logging.warning("existing command publishers=%s", conflicts or "none")
        node.verify_consumers(include_lower=args.lock_lower)
        warm_actions, policy_metadata, warm_observation, camera_source = warm_policy(
            node, startup_command, args, side_scope
        )
        side_scope.validate_server_metadata(policy_metadata)
        feedback_barrier = time.monotonic()
        node.wait_for_feedback_after(feedback_barrier, args.wait_ready_timeout)
        startup_command, startup_source = capture_startup_command(node)
        live_pose = node.measured_pose()
        intermediate_pose = side_scope.hold_pose(build_intermediate_pose(startup_command), live_pose)
        scoped_candidate_pose = side_scope.hold_pose(candidate.pose, live_pose)
        candidate = replace(candidate, pose=scoped_candidate_pose)
        final_reset = side_scope.hold_pose(final_reset_pose(startup_command, intermediate_pose), live_pose)
        lower_lock = node.lower_lock() if args.lock_lower else None
        logging.info("post-warm startup capture source=%s", startup_source)
        if args.validate_only:
            print(
                f"VALIDATE_OK policy={args.policy_host}:{args.policy_port} "
                f"actions=(50,37) init_pose={candidate.pose_id} "
                f"startup_source={startup_source} observation_state_source={args.observation_state_source} "
                f"camera_source={camera_source} "
                f"command_state_tracking=disabled "
                f"conflicts={conflicts}",
                flush=True,
            )
            return 0

        if args.dry_run:
            return run_policy_dry_run(
                warm_actions,
                startup_command,
                startup_source=startup_source,
                candidate=candidate,
                conflicts=conflicts,
                metadata=policy_metadata,
                policy_observation=warm_observation,
                args=args,
                intermediate_pose=intermediate_pose,
                side_scope=side_scope,
            )

        if conflicts and not args.allow_command_conflicts:
            raise RuntimeError(
                "command publisher conflicts detected; stop conflicting control sources or pass "
                "--allow-command-conflicts after confirming the takeover procedure"
            )
        command_log = CommandLog(
            args.log_dir,
            run_config=build_run_config(args, policy_metadata),
        )
        node.create_command_publishers(include_lower=args.lock_lower)
        publishers_enabled = True
        current_command = startup_command
        logging.warning("REAL MOTION ENABLED logs=%s", command_log.run_dir)

        with KeyReader() as keys:
            init_start = current_command
            init_result = None
            if args.two_stage_reset:
                init_result = transition(
                    node,
                    current_command,
                    intermediate_pose,
                    candidate=candidate,
                    lower_lock=lower_lock,
                    duration_s=args.init_duration,
                    args=args,
                    keys=keys,
                    label="MOVE_TO_INIT_INTERMEDIATE",
                    side_scope=side_scope,
                )
                current_command = init_result.pose
                if init_result.fault is None and not init_result.operator_stopped:
                    wait_with_hold(
                        node,
                        current_command,
                        candidate=candidate,
                        lower_lock=lower_lock,
                        keys=keys,
                        args=args,
                        prompt="HOLD_INIT_INTERMEDIATE: press Enter to move to initialization target.",
                        accepted={"enter"},
                        side_scope=side_scope,
                    )
                    init_start = current_command
            if init_result is None or (init_result.fault is None and not init_result.operator_stopped):
                init_result = transition(
                    node,
                    init_start,
                    candidate.pose,
                    candidate=candidate,
                    lower_lock=lower_lock,
                    duration_s=args.init_duration,
                    args=args,
                    keys=keys,
                    label="MOVE_TO_INIT",
                    side_scope=side_scope,
                )
            current_command = init_result.pose
            fault = init_result.fault

            if fault is not None:
                key = wait_with_hold(
                    node,
                    current_command,
                    candidate=candidate,
                    lower_lock=lower_lock,
                    keys=keys,
                    args=args,
                    prompt=(f"FAULT_HOLD: {fault}. Press Enter to attempt bounded reset; q exits."),
                    accepted={"enter", "q"},
                    side_scope=side_scope,
                )
                if key == "q":
                    logging.error("operator exited from FAULT_HOLD without reset")
                    return 1
            else:
                if args.init_only:
                    wait_with_hold(
                        node,
                        current_command,
                        candidate=candidate,
                        lower_lock=lower_lock,
                        keys=keys,
                        args=args,
                        prompt="INIT_ONLY holding. Press Enter to reset.",
                        accepted={"enter"},
                        side_scope=side_scope,
                    )
                elif not init_result.operator_stopped:
                    key = wait_with_hold(
                        node,
                        current_command,
                        candidate=candidate,
                        lower_lock=lower_lock,
                        keys=keys,
                        args=args,
                        prompt="HOLD_INIT: press Enter to run policy; q skips to reset wait.",
                        accepted={"enter", "q"},
                        side_scope=side_scope,
                    )
                    if key == "enter":
                        execution_result = execute_policy(
                            node,
                            current_command,
                            candidate=candidate,
                            lower_lock=lower_lock,
                            keys=keys,
                            args=args,
                            command_log=command_log,
                            side_scope=side_scope,
                            camera_source=camera_source,
                        )
                        current_command = execution_result.pose
                        fault = execution_result.fault

                if fault is not None:
                    key = wait_with_hold(
                        node,
                        current_command,
                        candidate=candidate,
                        lower_lock=lower_lock,
                        keys=keys,
                        args=args,
                        prompt=(f"FAULT_HOLD: {fault}. Press Enter to attempt bounded reset; q exits."),
                        accepted={"enter", "q"},
                        side_scope=side_scope,
                    )
                    if key == "q":
                        logging.error("operator exited from FAULT_HOLD without reset")
                        return 1
                else:
                    wait_with_hold(
                        node,
                        current_command,
                        candidate=candidate,
                        lower_lock=lower_lock,
                        keys=keys,
                        args=args,
                        prompt="HOLD_FINAL: press Enter to reset to startup hand-down pose.",
                        accepted={"enter"},
                        side_scope=side_scope,
                    )

            while True:
                if args.two_stage_reset:
                    reset_result = transition(
                        node,
                        current_command,
                        intermediate_pose,
                        candidate=candidate,
                        lower_lock=lower_lock,
                        duration_s=args.reset_duration,
                        args=args,
                        keys=keys,
                        label="RESET_INTERMEDIATE",
                        side_scope=side_scope,
                    )
                    if reset_result.fault is None and not reset_result.operator_stopped:
                        current_command = reset_result.pose
                        wait_with_hold(
                            node,
                            current_command,
                            candidate=candidate,
                            lower_lock=lower_lock,
                            keys=keys,
                            args=args,
                            prompt="HOLD_RESET_INTERMEDIATE: press Enter to move to final reset pose.",
                            accepted={"enter"},
                            side_scope=side_scope,
                        )
                        reset_result = transition(
                            node,
                            current_command,
                            final_reset,
                            candidate=candidate,
                            lower_lock=lower_lock,
                            duration_s=args.reset_duration,
                            args=args,
                            keys=keys,
                            label="RESET_FINAL",
                            side_scope=side_scope,
                        )
                else:
                    reset_result = transition(
                        node,
                        current_command,
                        startup_command,
                        candidate=candidate,
                        lower_lock=lower_lock,
                        duration_s=args.reset_duration,
                        args=args,
                        keys=keys,
                        label="RESETTING",
                        side_scope=side_scope,
                    )
                current_command = reset_result.pose
                if reset_result.fault is None and not reset_result.operator_stopped:
                    break
                reason = reset_result.fault or "reset interrupted by operator"
                key = wait_with_hold(
                    node,
                    current_command,
                    candidate=candidate,
                    lower_lock=lower_lock,
                    keys=keys,
                    args=args,
                    prompt=(f"FAULT_HOLD: {reason}. Press Enter to retry bounded reset; q exits."),
                    accepted={"enter", "q"},
                    side_scope=side_scope,
                )
                if key == "q":
                    logging.error("operator exited before reset completed")
                    return 1
            hold_for(
                node,
                current_command,
                candidate=candidate,
                lower_lock=lower_lock,
                args=args,
                duration_s=0.5,
                side_scope=side_scope,
            )
            logging.warning("reset complete")
        return 0
    finally:
        if publishers_enabled and current_command is not None:
            try:
                hold_for(
                    node,
                    current_command,
                    candidate=candidate,
                    lower_lock=lower_lock,
                    args=args,
                    duration_s=0.3,
                    side_scope=side_scope,
                )
            except Exception:
                logging.exception("final measured hold failed")
        if command_log is not None:
            command_log.close()
        node.stop_background_spin()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop_requested = False

    def _signal_handler(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        return run(args)
    except KeyboardInterrupt:
        logging.warning("operator interrupt received")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
