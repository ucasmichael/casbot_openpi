"""Pure pose, interpolation, and safety logic for A3 deployment."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ARM_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

HAND_NAMES = (
    "left_thumb_rotation_0",
    "left_thumb_wiggles_1",
    "left_thumb_bent_2",
    "left_index_wiggles_0",
    "left_index_bent_1",
    "left_middle_bent",
    "left_ring_wiggles_0",
    "left_ring_bent_1",
    "left_pinky_wiggles_0",
    "left_pinky_bent_1",
    "right_thumb_rotation_0",
    "right_thumb_wiggles_1",
    "right_thumb_bent_2",
    "right_index_wiggles_0",
    "right_index_bent_1",
    "right_middle_bent",
    "right_ring_wiggles_0",
    "right_ring_bent_1",
    "right_pinky_wiggles_0",
    "right_pinky_bent_1",
)

HAND_STATE_ALIASES = {
    "left_thumb_rotation_0": "left_thumb_roration_0",
}

WAIST_NAMES = ("waist_roll", "waist_pitch", "waist_yaw")
WAIST_STATE_NAMES = ("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint")
MODEL_NAMES = ARM_NAMES + HAND_NAMES + WAIST_NAMES
MODEL_ACTION_DIM = len(MODEL_NAMES)
ACTION_HORIZON = 50

ARM_LOWER = np.deg2rad(
    np.asarray([-165, -5, -160, -55, -160, -93, -93, -165, -150, -160, -55, -160, -93, -93])
)
ARM_UPPER = np.deg2rad(
    np.asarray([165, 150, 160, 100, 160, 93, 93, 165, 5, 160, 100, 160, 93, 93])
)
HAND_LOWER = np.zeros(20, dtype=np.float64)
HAND_UPPER = np.full(20, 4095.0, dtype=np.float64)
WAIST_LOWER = np.deg2rad(np.asarray([-28, -20, -150], dtype=np.float64))
WAIST_UPPER = np.deg2rad(np.asarray([24, 20, 150], dtype=np.float64))
WAIST_HEIGHT_LOWER = -0.3
WAIST_HEIGHT_UPPER = 0.0
ARM_STIFFNESS_UPPER = 500.0
ARM_DAMPING_UPPER = 50.0

ARM_MARGIN = 0.02
WAIST_MARGIN = 0.02
HARD_MAX_ARM_STEP = 0.02
HARD_MAX_HAND_STEP = 200.0
HARD_MAX_WAIST_STEP = 0.003
DEFAULT_POLICY_ARM_STEP = HARD_MAX_ARM_STEP
DEFAULT_POLICY_HAND_STEP = HARD_MAX_HAND_STEP
DEFAULT_TRANSITION_ARM_STEP = math.radians(0.2)
DEFAULT_TRANSITION_HAND_STEP = 20.0
DEFAULT_TRANSITION_WAIST_STEP = HARD_MAX_WAIST_STEP


@dataclass(frozen=True)
class Pose:
    arm: np.ndarray
    hand: np.ndarray
    waist: np.ndarray

    @classmethod
    def from_state(cls, state: Sequence[float]) -> "Pose":
        values = np.asarray(state, dtype=np.float64)
        if values.shape != (MODEL_ACTION_DIM,):
            raise ValueError(f"qpose must have shape ({MODEL_ACTION_DIM},), got {values.shape}")
        return cls(values[:14].copy(), values[14:34].copy(), values[34:37].copy())

    def state(self) -> np.ndarray:
        return np.concatenate((self.arm, self.hand, self.waist)).astype(np.float32)


@dataclass(frozen=True)
class ArmTemplate:
    stiffness: np.ndarray
    damping: np.ndarray


@dataclass(frozen=True)
class HandTemplate:
    frame_id: str
    names: tuple[str, ...]
    velocity: np.ndarray
    effort: np.ndarray


@dataclass(frozen=True)
class InitPoseCandidate:
    pose_id: str
    pose: Pose
    arm_template: ArmTemplate
    hand_template: HandTemplate
    waist_height: float
    source_bag: str
    synchronization_ms: dict[str, float]


@dataclass(frozen=True)
class InitPoseConfig:
    path: Path
    default_pose_id: str
    candidates: dict[str, InitPoseCandidate]

    def select(self, pose_id: str | None = None) -> InitPoseCandidate:
        selected = pose_id or self.default_pose_id
        try:
            return self.candidates[selected]
        except KeyError as error:
            raise ValueError(
                f"unknown init pose {selected!r}; choices={sorted(self.candidates)}"
            ) from error


def _finite_array(values: Sequence[float], shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return array


def validate_pose(pose: Pose, *, margin: bool = True) -> None:
    arm = _finite_array(pose.arm, (14,), "arm")
    hand = _finite_array(pose.hand, (20,), "hand")
    waist = _finite_array(pose.waist, (3,), "waist")
    arm_margin = ARM_MARGIN if margin else 0.0
    waist_margin = WAIST_MARGIN if margin else 0.0
    if np.any(arm < ARM_LOWER + arm_margin) or np.any(arm > ARM_UPPER - arm_margin):
        indices = np.flatnonzero(
            (arm < ARM_LOWER + arm_margin) | (arm > ARM_UPPER - arm_margin)
        )
        raise ValueError(f"arm pose violates limits at indices {indices.tolist()}")
    if np.any(hand < HAND_LOWER) or np.any(hand > HAND_UPPER):
        indices = np.flatnonzero((hand < HAND_LOWER) | (hand > HAND_UPPER))
        raise ValueError(f"hand pose violates limits at indices {indices.tolist()}")
    if np.any(waist < WAIST_LOWER + waist_margin) or np.any(waist > WAIST_UPPER - waist_margin):
        indices = np.flatnonzero(
            (waist < WAIST_LOWER + waist_margin) | (waist > WAIST_UPPER - waist_margin)
        )
        raise ValueError(f"waist pose violates limits at indices {indices.tolist()}")


def validate_arm_template(template: ArmTemplate) -> None:
    stiffness = _finite_array(template.stiffness, (14,), "arm stiffness")
    damping = _finite_array(template.damping, (14,), "arm damping")
    if np.any(stiffness <= 0.0) or np.any(stiffness > ARM_STIFFNESS_UPPER):
        raise ValueError(f"arm stiffness must be within (0,{ARM_STIFFNESS_UPPER}]")
    if np.any(damping <= 0.0) or np.any(damping > ARM_DAMPING_UPPER):
        raise ValueError(f"arm damping must be within (0,{ARM_DAMPING_UPPER}]")


def validate_hand_template(template: HandTemplate) -> None:
    if not template.frame_id:
        raise ValueError("hand frame_id must not be empty")
    if template.names not in ((), HAND_NAMES):
        raise ValueError("hand template names must be empty or match model hand order")
    for label, values in (("velocity", template.velocity), ("effort", template.effort)):
        array = np.asarray(values, dtype=np.float64)
        if array.shape not in ((0,), (len(HAND_NAMES),)):
            raise ValueError(f"hand {label} must be empty or have {len(HAND_NAMES)} values")
        if not np.isfinite(array).all():
            raise ValueError(f"hand {label} contains NaN or Inf")


def validate_waist_height(value: float) -> None:
    if not math.isfinite(value) or not WAIST_HEIGHT_LOWER <= value <= WAIST_HEIGHT_UPPER:
        raise ValueError(
            f"waist height must be finite and within [{WAIST_HEIGHT_LOWER},{WAIST_HEIGHT_UPPER}]"
        )


def align_positions(
    names: Sequence[str],
    positions: Sequence[float],
    expected_names: Sequence[str],
    *,
    aliases: dict[str, str] | None = None,
) -> np.ndarray:
    if len(names) != len(positions):
        raise ValueError(f"joint name/value mismatch: {len(names)} != {len(positions)}")
    if len(set(names)) != len(names):
        raise ValueError("joint message contains duplicate names")
    by_name = {str(name): float(value) for name, value in zip(names, positions, strict=True)}
    aliases = aliases or {}
    output = []
    missing = []
    for name in expected_names:
        source = name if name in by_name else aliases.get(name, "")
        if source not in by_name:
            missing.append(name)
        else:
            output.append(by_name[source])
    if missing:
        raise ValueError(f"joint message is missing {missing}")
    return _finite_array(output, (len(expected_names),), "aligned positions")


def transition_step_count(
    start: Pose,
    target: Pose,
    *,
    rate_hz: float,
    min_duration_s: float,
    max_arm_step: float = DEFAULT_TRANSITION_ARM_STEP,
    max_hand_step: float = DEFAULT_TRANSITION_HAND_STEP,
    max_waist_step: float = DEFAULT_TRANSITION_WAIST_STEP,
) -> int:
    if min(rate_hz, max_arm_step, max_hand_step, max_waist_step) <= 0:
        raise ValueError("transition rates and step limits must be positive")
    validate_pose(start)
    validate_pose(target)
    counts = (
        int(np.ceil(np.max(np.abs(target.arm - start.arm)) / max_arm_step)),
        int(np.ceil(np.max(np.abs(target.hand - start.hand)) / max_hand_step)),
        int(np.ceil(np.max(np.abs(target.waist - start.waist)) / max_waist_step)),
        int(np.ceil(max(0.0, min_duration_s) * rate_hz)),
        1,
    )
    return max(counts)


def interpolate_poses(
    start: Pose,
    target: Pose,
    *,
    rate_hz: float,
    min_duration_s: float,
    max_arm_step: float = DEFAULT_TRANSITION_ARM_STEP,
    max_hand_step: float = DEFAULT_TRANSITION_HAND_STEP,
    max_waist_step: float = DEFAULT_TRANSITION_WAIST_STEP,
) -> Iterable[Pose]:
    steps = transition_step_count(
        start,
        target,
        rate_hz=rate_hz,
        min_duration_s=min_duration_s,
        max_arm_step=max_arm_step,
        max_hand_step=max_hand_step,
        max_waist_step=max_waist_step,
    )
    for index in range(1, steps + 1):
        alpha = index / steps
        pose = Pose(
            arm=start.arm + alpha * (target.arm - start.arm),
            hand=start.hand + alpha * (target.hand - start.hand),
            waist=start.waist + alpha * (target.waist - start.waist),
        )
        validate_pose(pose)
        yield pose


def bounded_action(
    previous: Pose,
    requested_state: Sequence[float],
    *,
    max_arm_step: float = DEFAULT_POLICY_ARM_STEP,
    max_hand_step: float = DEFAULT_POLICY_HAND_STEP,
    max_waist_step: float = HARD_MAX_WAIST_STEP,
) -> tuple[Pose, dict[str, float]]:
    if not 0 < max_arm_step <= HARD_MAX_ARM_STEP:
        raise ValueError(f"max_arm_step must be in (0, {HARD_MAX_ARM_STEP}]")
    if not 0 < max_hand_step <= HARD_MAX_HAND_STEP:
        raise ValueError(f"max_hand_step must be in (0, {HARD_MAX_HAND_STEP}]")
    if not 0 < max_waist_step <= HARD_MAX_WAIST_STEP:
        raise ValueError(f"max_waist_step must be in (0, {HARD_MAX_WAIST_STEP}]")
    validate_pose(previous)
    requested = Pose.from_state(requested_state)

    raw_arm = requested.arm - previous.arm
    raw_hand = requested.hand - previous.hand
    raw_waist = requested.waist - previous.waist
    arm = previous.arm + np.clip(raw_arm, -max_arm_step, max_arm_step)
    hand = previous.hand + np.clip(raw_hand, -max_hand_step, max_hand_step)
    waist = previous.waist + np.clip(raw_waist, -max_waist_step, max_waist_step)

    arm = np.clip(arm, ARM_LOWER + ARM_MARGIN, ARM_UPPER - ARM_MARGIN)
    hand = np.clip(hand, HAND_LOWER, HAND_UPPER)
    waist = np.clip(waist, WAIST_LOWER + WAIST_MARGIN, WAIST_UPPER - WAIST_MARGIN)
    command = Pose(arm=arm, hand=hand, waist=waist)
    validate_pose(command)
    stats = {
        "requested_arm_max_delta": float(np.max(np.abs(raw_arm))),
        "requested_hand_max_delta": float(np.max(np.abs(raw_hand))),
        "requested_waist_max_delta": float(np.max(np.abs(raw_waist))),
        "bounded_arm_max_delta": float(np.max(np.abs(command.arm - previous.arm))),
        "bounded_hand_max_delta": float(np.max(np.abs(command.hand - previous.hand))),
        "bounded_waist_max_delta": float(np.max(np.abs(command.waist - previous.waist))),
    }
    return command, stats


def analyze_action_chunk(
    initial: Pose,
    actions: Any,
    *,
    max_arm_step: float = DEFAULT_POLICY_ARM_STEP,
    max_hand_step: float = DEFAULT_POLICY_HAND_STEP,
    max_waist_step: float = HARD_MAX_WAIST_STEP,
) -> tuple[Pose, dict[str, Any]]:
    """Simulate one policy chunk and report raw-action safety properties."""
    validate_pose(initial)
    array = validate_action_chunk(actions)
    groups = {
        "arm": (ARM_NAMES, ARM_LOWER + ARM_MARGIN, ARM_UPPER - ARM_MARGIN, max_arm_step),
        "hand": (HAND_NAMES, HAND_LOWER, HAND_UPPER, max_hand_step),
        "waist": (WAIST_NAMES, WAIST_LOWER + WAIST_MARGIN, WAIST_UPPER - WAIST_MARGIN, max_waist_step),
    }
    report: dict[str, Any] = {
        "actions": int(array.shape[0]),
        "command_state_tracking": "disabled",
        "raw_limit_violation_count": {name: 0 for name in groups},
        "first_raw_limit_violation": {name: None for name in groups},
        "initial_requested_max_delta": {name: 0.0 for name in groups},
        "adjacent_requested_max_delta": {name: 0.0 for name in groups},
        "requested_from_simulated_max_delta": {name: 0.0 for name in groups},
        "bounded_max_delta": {name: 0.0 for name in groups},
        "delta_limited_action_count": {name: 0 for name in groups},
    }
    current = initial
    previous_requested = initial
    for step, requested_state in enumerate(array):
        requested = Pose.from_state(requested_state)
        for name, (joint_names, lower, upper, max_step) in groups.items():
            values = getattr(requested, name)
            out_of_range = (values < lower) | (values > upper)
            count = int(np.count_nonzero(out_of_range))
            report["raw_limit_violation_count"][name] += count
            if count and report["first_raw_limit_violation"][name] is None:
                index = int(np.flatnonzero(out_of_range)[0])
                report["first_raw_limit_violation"][name] = {
                    "step": step,
                    "joint": joint_names[index],
                    "value": float(values[index]),
                    "lower": float(lower[index]),
                    "upper": float(upper[index]),
                }
            reference = initial if step == 0 else previous_requested
            adjacent_delta = float(np.max(np.abs(values - getattr(reference, name))))
            key = "initial_requested_max_delta" if step == 0 else "adjacent_requested_max_delta"
            report[key][name] = max(report[key][name], adjacent_delta)
            requested_delta = float(np.max(np.abs(values - getattr(current, name))))
            report["requested_from_simulated_max_delta"][name] = max(
                report["requested_from_simulated_max_delta"][name], requested_delta
            )
            if requested_delta > max_step:
                report["delta_limited_action_count"][name] += 1

        next_command, stats = bounded_action(
            current,
            requested_state,
            max_arm_step=max_arm_step,
            max_hand_step=max_hand_step,
            max_waist_step=max_waist_step,
        )
        for name in groups:
            report["bounded_max_delta"][name] = max(
                report["bounded_max_delta"][name], stats[f"bounded_{name}_max_delta"]
            )
        current = next_command
        previous_requested = requested

    report["raw_limit_violation_total"] = int(
        sum(report["raw_limit_violation_count"].values())
    )
    report["simulated_final_qpose"] = current.state().tolist()
    return current, report


def validate_action_chunk(actions: Any) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float64)
    expected = (ACTION_HORIZON, MODEL_ACTION_DIM)
    if array.shape != expected:
        raise ValueError(f"policy action shape must be {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("policy actions contain NaN or Inf")
    return array


def select_medoid(states: dict[str, Sequence[float]]) -> str:
    if not states:
        raise ValueError("cannot select a medoid from no poses")
    names = sorted(states)
    matrix = np.stack([_finite_array(states[name], (MODEL_ACTION_DIM,), name) for name in names])
    scale = np.concatenate((np.ones(14), np.full(20, 1000.0), np.ones(3)))
    normalized = matrix / scale
    scores = np.zeros(len(names), dtype=np.float64)
    for index in range(len(names)):
        scores[index] = np.linalg.norm(normalized[index] - normalized, axis=1).sum()
    return names[int(np.argmin(scores))]


def load_init_pose_config(path: Path) -> InitPoseConfig:
    path = path.expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 2:
        raise ValueError(f"unsupported init pose schema: {raw.get('schema_version')!r}")
    model = raw.get("model", {})
    if model.get("action_dim") != MODEL_ACTION_DIM or model.get("action_horizon") != ACTION_HORIZON:
        raise ValueError("init pose model dimensions do not match pi05_a3_both")
    if tuple(model.get("names", [])) != MODEL_NAMES:
        raise ValueError("init pose joint order does not match deployment model order")
    candidates = {}
    for pose_id, item in raw.get("candidates", {}).items():
        pose = Pose.from_state(item["qpose"])
        validate_pose(pose)
        template = item.get("arm_template", {})
        hand_template = item.get("hand_template", {})
        candidate = InitPoseCandidate(
            pose_id=pose_id,
            pose=pose,
            arm_template=ArmTemplate(
                stiffness=_finite_array(template.get("stiffness", []), (14,), "arm stiffness"),
                damping=_finite_array(template.get("damping", []), (14,), "arm damping"),
            ),
            hand_template=HandTemplate(
                frame_id=str(hand_template.get("frame_id", "")),
                names=tuple(str(name) for name in hand_template.get("name", [])),
                velocity=np.asarray(hand_template.get("velocity", []), dtype=np.float64),
                effort=np.asarray(hand_template.get("effort", []), dtype=np.float64),
            ),
            waist_height=float(item.get("waist_height", 0.0)),
            source_bag=str(item.get("source_bag", "")),
            synchronization_ms={
                str(key): float(value) for key, value in item.get("synchronization_ms", {}).items()
            },
        )
        validate_arm_template(candidate.arm_template)
        validate_hand_template(candidate.hand_template)
        validate_waist_height(candidate.waist_height)
        candidates[pose_id] = candidate
    if not candidates:
        raise ValueError("init pose config has no candidates")
    default_pose_id = str(raw.get("default_pose_id", ""))
    if default_pose_id not in candidates:
        raise ValueError(f"default pose {default_pose_id!r} is not present")
    return InitPoseConfig(path=path, default_pose_id=default_pose_id, candidates=candidates)


def _read_varint(raw: bytes, index: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while index < len(raw) and shift < 64:
        byte = raw[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7
    raise ValueError("invalid protobuf varint")


def _encode_varint(value: int) -> bytes:
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            output.append(byte | 0x80)
        else:
            output.append(byte)
            return bytes(output)


def _skip_pb_value(raw: bytes, index: int, wire_type: int) -> int:
    if wire_type == 0:
        return _read_varint(raw, index)[1]
    if wire_type == 1:
        return index + 8
    if wire_type == 2:
        size, index = _read_varint(raw, index)
        return index + size
    if wire_type == 5:
        return index + 4
    raise ValueError(f"unsupported protobuf wire type {wire_type}")


def wrapper_bytes(message: Any) -> bytes:
    raw = getattr(message, "data", b"")
    if isinstance(raw, bytes | bytearray):
        return bytes(raw)
    output = bytearray()
    for item in raw:
        if isinstance(item, int):
            output.append(item & 0xFF)
        else:
            output.extend(bytes(item))
    return bytes(output)


def parse_waist_payload(raw: bytes) -> tuple[np.ndarray, float]:
    fields: dict[int, float] = {}
    index = 0
    while index < len(raw):
        tag, index = _read_varint(raw, index)
        field = tag >> 3
        wire_type = tag & 7
        if field in (2, 3, 4, 5) and wire_type == 1:
            if index + 8 > len(raw):
                raise ValueError("truncated waist protobuf")
            fields[field] = struct.unpack_from("<d", raw, index)[0]
            index += 8
        else:
            index = _skip_pb_value(raw, index, wire_type)
        if index > len(raw):
            raise ValueError("invalid waist protobuf length")
    if not any(field in fields for field in (2, 3, 4)):
        raise ValueError("waist protobuf contains no RPY fields")
    rpy = _finite_array(
        [fields.get(3, 0.0), fields.get(2, 0.0), fields.get(4, 0.0)],
        (3,),
        "waist RPY",
    )
    return rpy, float(fields.get(5, 0.0))


def encode_waist_payload(waist_rpy: Sequence[float], height: float) -> bytes:
    waist = _finite_array(waist_rpy, (3,), "waist RPY")
    height = float(height)
    if not math.isfinite(height) or not -0.3 <= height <= 0.0:
        raise ValueError("waist height must be finite and in [-0.3, 0.0]")
    pitch, roll, yaw = waist[1], waist[0], waist[2]
    return b"".join(
        _encode_varint((field << 3) | 1) + struct.pack("<d", float(value))
        for field, value in ((2, pitch), (3, roll), (4, yaw), (5, height))
    )
