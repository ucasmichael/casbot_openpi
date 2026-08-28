#!/usr/bin/env python3
"""Convert A3 MCAP episodes to LeRobot v2.1.

Training samples contain only:
- chest RGB image
- chest depth image
- arm command, hand command, and move-waist rpy state/action

The collector also records other qpose topics, but this script does not write
them into LeRobot training samples. At the end it prints per-dimension means for
the selected training qpose.

(base) lyx@ly-node-214-23-97-2:/data/lyx/casbot_openpi$ cd /data/lyx/casbot_openpi

export HF_DATASETS_CACHE=/data/lyx/hf_cache
export HF_LEROBOT_HOME=/data/lyx/dataset/lerobot

.venv/bin/python examples/a3_real/convert_a3_mcap_to_lerobot.py \
  --raw-dir /data/lyx/dataset/data0819_flat \
  --repo-id local_dataset/a3_0819_both_depth \
  --lerobot-root /data/lyx/dataset/lerobot \
  --joint-mode both \
  --mode image \
  --overwrite \
  --task "stack cups"
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import dataclasses
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import struct
import types
from functools import lru_cache
from typing import Any, Literal

from datasets import Dataset
from datasets import concatenate_datasets
from lerobot.common.constants import HF_LEROBOT_HOME
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
from mcap_ros2.decoder import Decoder
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as SciRot

RGB_TOPIC = "/hal/chest_front_d457_camera/rgb"
DEPTH_TOPIC = "/hal/chest_front_d457_camera/depth"
ARM_TOPIC = "/motion/control/pnc/arm_joint_command"
HAND_TOPIC = "/motion/control/hand_joint_command"
WAIST_MOVE_TOPIC = (
    "/motion/control/move_waist/"
    "pb_3Aaimdk_2Eprotocol_2EMotionControlMoveWaistChannel"
)
ARM_DIM = 14
HAND_DIM = 20
WAIST_RPY_NAMES = ["waist_roll", "waist_pitch", "waist_yaw"]
TRAINING_QPOSE_DIM = ARM_DIM + HAND_DIM + len(WAIST_RPY_NAMES)
DEFAULT_HAND_NAMES = [
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
]

CHEST_RGB_KEY = "observation.images.chest_front_rgb"
CHEST_DEPTH_KEY = "observation.images.chest_front_depth"
JointMode = Literal["left", "right", "both"]

RGB_INTRINSICS = {
    "width": 1280,
    "height": 720,
    "fx": 646.18811,
    "fy": 645.471252,
    "ppx": 640.269958,
    "ppy": 353.34375,
}

T_RGB_TORSO = np.array(
    [
        [-0.0000044428, -1.0000000000, -0.0000005113, 0.0150006000],
        [-0.6427891683, 0.0000032475, -0.7660431353, 0.2611690000],
        [0.7660431353, -0.0000030747, -0.6427891683, 0.0107409000],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

A3_CAMERA_ROOT_LINK = "torso_Link"
A3_PROJECTION_TARGET_LINKS = (
    "left_hand_palm_Link",
    "right_hand_palm_Link",
)
A3_LEFT_CHAIN = (
    ("left_shoulder_pitch_joint", "left_shoulder_pitch_Link"),
    ("left_shoulder_roll_joint", "left_shoulder_roll_Link"),
    ("left_shoulder_yaw_joint", "left_shoulder_yaw_Link"),
    ("left_elbow_joint", "left_elbow_Link"),
    ("left_wrist_roll_joint", "left_wrist_roll_Link"),
    ("left_wrist_pitch_joint", "left_wrist_pitch_Link"),
    ("left_wrist_yaw_joint", "left_wrist_yaw_Link"),
    ("left_hand_palm_joint", "left_hand_palm_Link"),
)
A3_RIGHT_CHAIN = (
    ("right_shoulder_pitch_joint", "right_shoulder_pitch_Link"),
    ("right_shoulder_roll_joint", "right_shoulder_roll_Link"),
    ("right_shoulder_yaw_joint", "right_shoulder_yaw_Link"),
    ("right_elbow_joint", "right_elbow_Link"),
    ("right_wrist_roll_joint", "right_wrist_roll_Link"),
    ("right_wrist_pitch_joint", "right_wrist_pitch_Link"),
    ("right_wrist_yaw_joint", "right_wrist_yaw_Link"),
    ("right_hand_palm_joint", "right_hand_palm_Link"),
)


@dataclasses.dataclass
class TopicSeries:
    times_ns: list[int] = dataclasses.field(default_factory=list)
    values: list[Any] = dataclasses.field(default_factory=list)
    names: list[str] | None = None

    def append(self, time_ns: int, value: Any) -> None:
        self.times_ns.append(time_ns)
        self.values.append(value)


@dataclasses.dataclass
class EpisodeData:
    frames: list[dict[str, Any]]
    joint_names: list[str]
    rgb_shape: tuple[int, int, int]
    depth_shape: tuple[int, int, int]
    selected_qpose: list[np.ndarray]
    excluded_qpose: dict[str, list[np.ndarray]]
    excluded_names: dict[str, list[str]]
    trim_info: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1e-4
    image_writer_processes: int = 0
    image_writer_threads: int = 4
    video_backend: str | None = None


@dataclasses.dataclass(frozen=True)
class ProjectionTrimConfig:
    enabled: bool = True
    urdf_path: Path = Path("/data/lyx/backup/a3_ultra_t2d0/urdf/model_planning.urdf")
    pad_frames: int = 0
    min_kept_frames: int = 30
    keep_full_when_missing: bool = True
    margin_px: int = 0


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def message_time_ns(ros_msg: Any, mcap_message: Any) -> int:
    header = getattr(ros_msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        return stamp_to_ns(stamp)
    return int(mcap_message.log_time)


def decode_rgb_image(msg: Any) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding not in {"bgr8", "rgb8"}:
        raise ValueError(f"Unsupported RGB image encoding: {msg.encoding}")

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    row_values = step // np.dtype(np.uint8).itemsize
    array = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, row_values)
    image = array[:, : width * 3].reshape(height, width, 3)
    if encoding == "bgr8":
        image = image[..., ::-1]
    return np.ascontiguousarray(image)


def decode_depth_image(msg: Any) -> Image.Image:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    is_bigendian = bool(getattr(msg, "is_bigendian", 0))

    if encoding in {"16uc1", "mono16"}:
        dtype = np.dtype(">u2" if is_bigendian else "<u2")
        row_values = step // dtype.itemsize
        array = np.frombuffer(msg.data, dtype=dtype).reshape(height, row_values)
        depth = array[:, :width].astype(np.uint16, copy=False)
        return Image.fromarray(np.ascontiguousarray(depth), mode="I;16")

    if encoding == "32fc1":
        dtype = np.dtype(">f4" if is_bigendian else "<f4")
        row_values = step // dtype.itemsize
        array = np.frombuffer(msg.data, dtype=dtype).reshape(height, row_values)
        depth_m = array[:, :width].astype(np.float32, copy=False)
        depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=65.535, neginf=0.0)
        depth_mm = np.clip(depth_mm * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
        return Image.fromarray(np.ascontiguousarray(depth_mm), mode="I;16")

    raise ValueError(f"Unsupported depth image encoding: {msg.encoding}")


@lru_cache(maxsize=1)
def _parse_float_triplet(text: str | None, *, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise ValueError(f"expected three floats, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def _matrix_from_xyz_rpy(xyz: str | None, rpy: str | None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = SciRot.from_euler("xyz", _parse_float_triplet(rpy, default=(0.0, 0.0, 0.0))).as_matrix()
    T[:3, 3] = _parse_float_triplet(xyz, default=(0.0, 0.0, 0.0))
    return T


def _matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        return T
    axis = axis / norm
    T[:3, :3] = SciRot.from_rotvec(axis * float(angle)).as_matrix()
    return T


def _scaled_rgb_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    sx = width / float(RGB_INTRINSICS["width"])
    sy = height / float(RGB_INTRINSICS["height"])
    return (
        RGB_INTRINSICS["fx"] * sx,
        RGB_INTRINSICS["fy"] * sy,
        RGB_INTRINSICS["ppx"] * sx,
        RGB_INTRINSICS["ppy"] * sy,
    )


@dataclasses.dataclass(frozen=True)
class UrdfJointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@lru_cache(maxsize=4)
def _load_urdf_joint_specs(urdf_path_s: str) -> dict[str, UrdfJointSpec]:
    urdf_path = Path(urdf_path_s)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    root = ET.parse(urdf_path).getroot()
    specs: dict[str, UrdfJointSpec] = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "fixed")
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin_elem = joint.find("origin")
        origin = _matrix_from_xyz_rpy(
            origin_elem.attrib.get("xyz") if origin_elem is not None else None,
            origin_elem.attrib.get("rpy") if origin_elem is not None else None,
        )
        axis_elem = joint.find("axis")
        axis = _parse_float_triplet(
            axis_elem.attrib.get("xyz") if axis_elem is not None else None,
            default=(1.0, 0.0, 0.0),
        )
        mimic_elem = joint.find("mimic")
        mimic_joint = mimic_elem.attrib.get("joint") if mimic_elem is not None else None
        mimic_multiplier = float(mimic_elem.attrib.get("multiplier", "1.0")) if mimic_elem is not None else 1.0
        mimic_offset = float(mimic_elem.attrib.get("offset", "0.0")) if mimic_elem is not None else 0.0
        specs[name] = UrdfJointSpec(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin=origin,
            axis=axis,
            mimic_joint=mimic_joint,
            mimic_multiplier=mimic_multiplier,
            mimic_offset=mimic_offset,
        )
    return specs


def _joint_value_for_spec(spec: UrdfJointSpec, qmap: dict[str, float]) -> float:
    if spec.mimic_joint is not None:
        return spec.mimic_multiplier * qmap.get(spec.mimic_joint, 0.0) + spec.mimic_offset
    return qmap.get(spec.name, 0.0)


def _link_positions_for_chain(
    joint_specs: dict[str, UrdfJointSpec],
    chain: tuple[tuple[str, str], ...],
    qmap: dict[str, float],
) -> dict[str, np.ndarray]:
    transform = np.eye(4, dtype=np.float64)
    positions: dict[str, np.ndarray] = {}
    for joint_name, link_name in chain:
        spec = joint_specs[joint_name]
        transform = transform @ spec.origin
        if spec.joint_type in {"revolute", "continuous"}:
            transform = transform @ _matrix_from_axis_angle(spec.axis, _joint_value_for_spec(spec, qmap))
        elif spec.joint_type == "prismatic":
            delta = np.eye(4, dtype=np.float64)
            axis = spec.axis
            norm = float(np.linalg.norm(axis))
            if norm > 0.0:
                delta[:3, 3] = axis / norm * _joint_value_for_spec(spec, qmap)
            transform = transform @ delta
        positions[link_name] = transform[:3, 3].copy()
    return positions


def _project_point_to_image(
    point_torso: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float] | None:
    point_rgb = T_RGB_TORSO[:3, :3] @ point_torso + T_RGB_TORSO[:3, 3]
    z = float(point_rgb[2])
    if z <= 1e-9:
        return None
    fx, fy, ppx, ppy = _scaled_rgb_intrinsics(width, height)
    u = fx * float(point_rgb[0]) / z + ppx
    v = fy * float(point_rgb[1]) / z + ppy
    return u, v, z


def _point_in_image(u: float, v: float, *, width: int, height: int, margin_px: int) -> bool:
    return (-margin_px) <= u < (width + margin_px) and (-margin_px) <= v < (height + margin_px)


def trim_window_from_projection(
    qpose_frames: list[np.ndarray],
    *,
    joint_names: list[str],
    image_shape: tuple[int, int, int],
    projection_trim: ProjectionTrimConfig,
    episode_name: str,
) -> tuple[int, int, dict[str, Any]]:
    frame_count = len(qpose_frames)
    if frame_count == 0:
        raise ValueError("cannot trim an empty frame sequence")

    if not projection_trim.enabled or frame_count < 2:
        return 0, frame_count - 1, {
            "enabled": projection_trim.enabled,
            "applied": False,
            "reason": "projection trimming disabled" if not projection_trim.enabled else "too_few_frames",
            "frame_count": frame_count,
        }

    height, width = image_shape[:2]
    joint_specs = _load_urdf_joint_specs(str(projection_trim.urdf_path))
    visible_mask = np.zeros(frame_count, dtype=bool)
    visible_counts = np.zeros(frame_count, dtype=np.int32)

    for frame_idx, qpose in enumerate(qpose_frames):
        qmap = {name: float(value) for name, value in zip(joint_names, qpose, strict=True)}
        left_positions = _link_positions_for_chain(joint_specs, A3_LEFT_CHAIN, qmap)
        right_positions = _link_positions_for_chain(joint_specs, A3_RIGHT_CHAIN, qmap)
        for link_name in A3_PROJECTION_TARGET_LINKS:
            point = left_positions.get(link_name)
            if point is None:
                point = right_positions.get(link_name)
            if point is None:
                continue
            projected = _project_point_to_image(point, width=width, height=height)
            if projected is None:
                continue
            u, v, _z = projected
            if _point_in_image(u, v, width=width, height=height, margin_px=projection_trim.margin_px):
                visible_mask[frame_idx] = True
                visible_counts[frame_idx] += 1

    tag1_frame = next((idx for idx, visible in enumerate(visible_mask) if visible), None)
    tag2_frame = next((idx for idx in range(frame_count - 1, -1, -1) if visible_mask[idx]), None)

    if tag1_frame is None or tag2_frame is None:
        info = {
            "enabled": projection_trim.enabled,
            "applied": False,
            "reason": "omnihand projection never entered RGB FOV; kept full synced episode",
            "frame_count": frame_count,
            "visible_frame_count": int(visible_mask.sum()),
            "target_links": list(A3_PROJECTION_TARGET_LINKS),
            "urdf_path": str(projection_trim.urdf_path),
            "image_shape": list(image_shape),
        }
        if not projection_trim.keep_full_when_missing:
            raise ValueError(
                "projection trim found no visible segment "
                f"(visible_frame_count={int(visible_mask.sum())}, urdf={projection_trim.urdf_path})"
            )
        return 0, frame_count - 1, info

    start_frame = max(0, int(tag1_frame) - projection_trim.pad_frames)
    end_frame = min(frame_count - 1, int(tag2_frame) + projection_trim.pad_frames)
    if end_frame < start_frame:
        start_frame, end_frame = 0, frame_count - 1

    applied = start_frame > 0 or end_frame < frame_count - 1
    kept_count = end_frame - start_frame + 1
    return start_frame, end_frame, {
        "enabled": projection_trim.enabled,
        "applied": applied,
        "frame_count": frame_count,
        "kept_frame_count": kept_count,
        "trim_start": start_frame,
        "trim_end": end_frame,
        "tag1_frame": int(tag1_frame),
        "tag2_frame": int(tag2_frame),
        "first_visible_frame": int(tag1_frame),
        "last_visible_frame": int(tag2_frame),
        "visible_frame_count": int(visible_mask.sum()),
        "visible_link_count_max": int(visible_counts.max()) if visible_counts.size else 0,
        "visible_link_count_mean": float(visible_counts.mean()) if visible_counts.size else 0.0,
        "target_links": list(A3_PROJECTION_TARGET_LINKS),
        "urdf_path": str(projection_trim.urdf_path),
        "image_shape": list(image_shape),
        "scan_mode": "forward_tag1_backward_tag2_ominihand_projection",
    }


def _read_varint(raw: bytes, index: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while index < len(raw):
        byte = raw[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7
        if shift >= 64:
            break
    raise ValueError("invalid protobuf varint")


def _skip_pb_value(raw: bytes, index: int, wire_type: int) -> int:
    if wire_type == 0:
        _, index = _read_varint(raw, index)
        return index
    if wire_type == 1:
        return index + 8
    if wire_type == 2:
        length, index = _read_varint(raw, index)
        return index + length
    if wire_type == 5:
        return index + 4
    raise ValueError(f"unsupported protobuf wire type: {wire_type}")


def wrapper_data_bytes(msg: Any) -> bytes:
    raw = getattr(msg, "data", b"")
    if isinstance(raw, bytes | bytearray):
        return bytes(raw)
    out = bytearray()
    for item in raw:
        if isinstance(item, int):
            out.append(item & 0xFF)
        elif isinstance(item, bytes | bytearray):
            out.extend(bytes(item))
        else:
            out.append(int(item) & 0xFF)
    return bytes(out)


def parse_move_waist_pb(raw: bytes) -> dict[str, float] | None:
    values: dict[str, float] = {}
    index = 0
    field_names = {
        2: "waist_pitch",
        3: "waist_roll",
        4: "waist_yaw",
    }
    try:
        while index < len(raw):
            tag, index = _read_varint(raw, index)
            field_number = tag >> 3
            wire_type = tag & 0x07
            if field_number in field_names and wire_type == 1:
                if index + 8 > len(raw):
                    return None
                values[field_names[field_number]] = struct.unpack_from("<d", raw, index)[0]
                index += 8
            else:
                index = _skip_pb_value(raw, index, wire_type)
            if index > len(raw):
                return None
    except Exception:
        return None
    if not any(name in values for name in ("waist_pitch", "waist_roll", "waist_yaw")):
        return None
    return {
        "waist_roll": float(values.get("waist_roll", 0.0)),
        "waist_pitch": float(values.get("waist_pitch", 0.0)),
        "waist_yaw": float(values.get("waist_yaw", 0.0)),
    }


def parse_move_waist_wrapper(msg: Any) -> np.ndarray:
    parsed = parse_move_waist_pb(wrapper_data_bytes(msg))
    if parsed is None:
        raise ValueError("failed to parse MotionControlMoveWaistChannel protobuf")
    rpy = np.asarray(
        [parsed["waist_roll"], parsed["waist_pitch"], parsed["waist_yaw"]],
        dtype=np.float32,
    )
    if not np.isfinite(rpy).all():
        raise ValueError(f"non-finite MotionControlMoveWaistChannel rpy: {rpy.tolist()}")
    return rpy


def parse_pnc_arm_joint_command(msg: Any) -> tuple[list[str], np.ndarray]:
    joints = list(getattr(msg, "joints", []))
    names = [str(getattr(joint, "name")) for joint in joints]
    positions = np.asarray([getattr(joint, "position") for joint in joints], dtype=np.float32)
    return names, positions


def parse_joint_state_positions(msg: Any, *, fallback_names: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    names = list(getattr(msg, "name", []))
    positions = np.asarray(getattr(msg, "position"), dtype=np.float32)
    if not names and fallback_names is not None and positions.shape[0] == len(fallback_names):
        names = fallback_names
    return names, positions


def read_mcap_files(mcap_files: list[Path]) -> dict[str, TopicSeries]:
    decoder = Decoder()
    topics: dict[str, TopicSeries] = {}

    for mcap_file in mcap_files:
        with mcap_file.open("rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                if topic not in {RGB_TOPIC, DEPTH_TOPIC, ARM_TOPIC, HAND_TOPIC, WAIST_MOVE_TOPIC}:
                    continue

                ros_msg = decoder.decode(schema, message)
                time_ns = message_time_ns(ros_msg, message)
                series = topics.setdefault(topic, TopicSeries())

                if topic == RGB_TOPIC:
                    series.append(time_ns, decode_rgb_image(ros_msg))
                    continue
                if topic == DEPTH_TOPIC:
                    series.append(time_ns, decode_depth_image(ros_msg))
                    continue
                if topic == WAIST_MOVE_TOPIC:
                    series.append(time_ns, parse_move_waist_wrapper(ros_msg))
                    continue

                if topic == ARM_TOPIC:
                    names, positions = parse_pnc_arm_joint_command(ros_msg)
                else:
                    names, positions = parse_joint_state_positions(ros_msg, fallback_names=DEFAULT_HAND_NAMES)
                if len(names) != positions.shape[0]:
                    raise ValueError(
                        f"{topic} has {len(names)} names but {positions.shape[0]} positions in {mcap_file}"
                    )
                if not np.isfinite(positions).all():
                    raise ValueError(f"{topic} has non-finite positions in {mcap_file}: {positions.tolist()}")
                if series.names is None:
                    series.names = names
                elif series.names != names:
                    raise ValueError(f"Joint names changed inside {mcap_file}: {topic}")
                series.append(time_ns, positions)

    for series in topics.values():
        order = np.argsort(series.times_ns)
        series.times_ns = [series.times_ns[i] for i in order]
        series.values = [series.values[i] for i in order]
    return topics


def nearest_value(series: TopicSeries, time_ns: int, tolerance_ns: int) -> Any | None:
    idx = bisect_left(series.times_ns, time_ns)
    candidates = []
    if idx < len(series.times_ns):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(series.times_ns[i] - time_ns))
    if abs(series.times_ns[best] - time_ns) > tolerance_ns:
        return None
    return series.values[best]


def make_qpose(
    arm: np.ndarray,
    hand: np.ndarray,
    waist_rpy: np.ndarray,
    arm_names: list[str],
    hand_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    if arm.shape != (ARM_DIM,):
        raise ValueError(f"{ARM_TOPIC} expected dim {ARM_DIM}, got {arm.shape[0]} names={arm_names}")
    if hand.shape != (HAND_DIM,):
        raise ValueError(f"{HAND_TOPIC} expected dim {HAND_DIM}, got {hand.shape[0]} names={hand_names}")
    if waist_rpy.shape != (len(WAIST_RPY_NAMES),):
        raise ValueError(f"{WAIST_MOVE_TOPIC} expected rpy dim {len(WAIST_RPY_NAMES)}, got {waist_rpy.shape[0]}")
    qpose = np.concatenate([arm, hand, waist_rpy]).astype(np.float32)
    names = [*arm_names, *hand_names, *WAIST_RPY_NAMES]
    if len(names) != TRAINING_QPOSE_DIM:
        raise ValueError(f"expected {TRAINING_QPOSE_DIM} qpose names, got {len(names)}: {names}")
    return qpose, names


def load_episode(
    episode_dir: Path,
    *,
    joint_mode: JointMode,
    sync_tolerance_ms: float,
    max_frames: int | None,
    projection_trim: ProjectionTrimConfig,
) -> EpisodeData:
    if joint_mode != "both":
        raise ValueError("A3 command+waist conversion only supports --joint-mode both")

    mcap_files = sorted(episode_dir.glob("**/*.mcap"))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found under {episode_dir}")

    topics = read_mcap_files(mcap_files)
    required = [RGB_TOPIC, DEPTH_TOPIC, ARM_TOPIC, HAND_TOPIC, WAIST_MOVE_TOPIC]
    missing = [topic for topic in required if topic not in topics or not topics[topic].values]
    if missing:
        raise ValueError(f"{episode_dir} is missing required topics: {missing}")

    arm_names = topics[ARM_TOPIC].names or []
    hand_names = topics[HAND_TOPIC].names or []
    tolerance_ns = int(sync_tolerance_ms * 1_000_000)

    synced_records: list[dict[str, Any]] = []
    selected_names: list[str] | None = None

    rgb_series = topics[RGB_TOPIC]
    for rgb_time_ns, rgb in zip(rgb_series.times_ns, rgb_series.values, strict=True):
        arm = nearest_value(topics[ARM_TOPIC], rgb_time_ns, tolerance_ns)
        hand = nearest_value(topics[HAND_TOPIC], rgb_time_ns, tolerance_ns)
        waist_rpy = nearest_value(topics[WAIST_MOVE_TOPIC], rgb_time_ns, tolerance_ns)
        depth = nearest_value(topics[DEPTH_TOPIC], rgb_time_ns, tolerance_ns)
        if arm is None or hand is None or waist_rpy is None or depth is None:
            continue

        state, names = make_qpose(arm, hand, waist_rpy, arm_names, hand_names)
        if selected_names is None:
            selected_names = names
        elif selected_names != names:
            raise ValueError(f"Selected joint names changed inside {episode_dir}")

        synced_records.append(
            {
                "rgb": rgb,
                "depth": depth,
                "state": state,
                "names": names,
            }
        )

    if not synced_records or selected_names is None:
        raise ValueError(f"No synchronized frames found for {episode_dir}")

    if len(selected_names) != TRAINING_QPOSE_DIM:
        raise ValueError(
            f"Unexpected selected qpose dim {len(selected_names)} for mode {joint_mode}: {selected_names}"
        )

    image_shape = synced_records[0]["rgb"].shape
    trim_start = 0
    trim_end = len(synced_records) - 1
    trim_info: dict[str, Any] = {
        "enabled": projection_trim.enabled,
        "applied": False,
        "frame_count": len(synced_records),
        "sync_tolerance_ms": sync_tolerance_ms,
    }
    if projection_trim.enabled:
        trim_start, trim_end, trim_info = trim_window_from_projection(
            [record["state"] for record in synced_records],
            joint_names=selected_names,
            image_shape=image_shape,
            projection_trim=projection_trim,
            episode_name=episode_dir.name,
        )
    kept_records = synced_records[trim_start : trim_end + 1]
    if projection_trim.enabled and len(kept_records) < projection_trim.min_kept_frames:
        raise ValueError(
            "projection trim kept too few frames "
            f"({len(kept_records)} < {projection_trim.min_kept_frames})"
        )
    if max_frames is not None:
        kept_records = kept_records[:max_frames]

    if not kept_records:
        raise ValueError(f"No synchronized frames found after trimming for {episode_dir}")

    frames = [
        {
            "observation.state": record["state"],
            "action": record["state"].copy(),
            CHEST_RGB_KEY: record["rgb"],
            CHEST_DEPTH_KEY: record["depth"],
        }
        for record in kept_records
    ]
    selected_qpose = [record["state"] for record in kept_records]

    trim_info = {
        **trim_info,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "kept_frame_count": len(kept_records),
        "source_frame_count": len(synced_records),
        "max_frames": max_frames,
    }

    return EpisodeData(
        frames=frames,
        joint_names=selected_names,
        rgb_shape=frames[0][CHEST_RGB_KEY].shape,
        depth_shape=(*frames[0][CHEST_DEPTH_KEY].size[::-1], 1),
        selected_qpose=selected_qpose,
        excluded_qpose={},
        excluded_names={},
        trim_info=trim_info,
    )


def read_task(episode_dir: Path, fallback: str) -> str:
    prompt_path = episode_dir / "prompt.txt"
    if prompt_path.exists():
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if prompt:
            return prompt

    metadata_path = episode_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prompt = str(metadata.get("prompt", "")).strip()
        if prompt:
            return prompt

    return fallback


def find_episode_dirs(raw_dir: Path) -> list[Path]:
    if list(raw_dir.glob("*.mcap")):
        return [raw_dir]
    episode_dirs = [path for path in sorted(raw_dir.glob("episode_*")) if path.is_dir()]
    if episode_dirs:
        return episode_dirs
    return sorted({path.parent for path in raw_dir.glob("**/*.mcap")})


def create_dataset(
    *,
    repo_id: str,
    root: Path | None,
    fps: int,
    mode: str,
    robot_type: str,
    joint_names: list[str],
    rgb_shape: tuple[int, int, int],
    depth_shape: tuple[int, int, int],
    overwrite: bool,
    dataset_config: DatasetConfig,
) -> LeRobotDataset:
    dataset_root = root / repo_id if root is not None else HF_LEROBOT_HOME / repo_id
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_root)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": [joint_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": [joint_names],
        },
        CHEST_RGB_KEY: {
            "dtype": mode,
            "shape": rgb_shape,
            "names": ["height", "width", "channels"],
        },
        CHEST_DEPTH_KEY: {
            "dtype": mode,
            "shape": depth_shape,
            "names": ["height", "width", "channels"],
        },
    }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def _save_episode_table_with_external_images(self: LeRobotDataset, episode_buffer: dict, episode_index: int) -> None:
    episode_dict = {key: episode_buffer[key] for key in self.hf_features}
    ep_dataset = Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
    self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
    self.hf_dataset.set_transform(lerobot_dataset_module.hf_transform_to_torch)
    ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
    ep_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(ep_data_path)


def save_episode_keep_image_files(dataset: LeRobotDataset) -> None:
    original_save_episode_table = dataset._save_episode_table  # noqa: SLF001
    original_rmtree = lerobot_dataset_module.shutil.rmtree

    def keep_images_rmtree(path: str | Path, *args, **kwargs) -> None:
        if Path(path) == dataset.root / "images":
            return None
        return original_rmtree(path, *args, **kwargs)

    dataset._save_episode_table = types.MethodType(_save_episode_table_with_external_images, dataset)  # noqa: SLF001
    lerobot_dataset_module.shutil.rmtree = keep_images_rmtree
    try:
        dataset.save_episode()
    finally:
        dataset._save_episode_table = original_save_episode_table  # noqa: SLF001
        lerobot_dataset_module.shutil.rmtree = original_rmtree


def qpose_mean_record(values: list[np.ndarray], joint_names: list[str] | None = None) -> dict[str, Any] | None:
    if not values:
        return None
    mean = np.stack(values, axis=0).mean(axis=0)
    names = []
    for idx in range(mean.shape[0]):
        joint_name = joint_names[idx] if joint_names is not None and idx < len(joint_names) else f"dim_{idx}"
        names.append(joint_name)
    return {
        "dim": int(mean.shape[0]),
        "frame_count": len(values),
        "joint_names": names,
        "mean": mean.astype(float).tolist(),
    }


def print_qpose_means(name: str, record: dict[str, Any] | None) -> None:
    if record is None:
        return
    print(f"\n{name} mean, dim={record['dim']}:")
    for idx, value in enumerate(record["mean"]):
        joint_name = record["joint_names"][idx]
        print(f"  [{idx:02d}] {joint_name}: {value:.6f}")


def write_qpose_mean_json(
    dataset_root: Path,
    *,
    joint_mode: JointMode,
    selected: list[np.ndarray],
    selected_names: list[str],
    excluded: dict[str, list[np.ndarray]],
    excluded_names: dict[str, list[str]],
) -> dict[str, Any]:
    stats = {
        "joint_mode": joint_mode,
        "training_qpose": qpose_mean_record(selected, selected_names),
        "excluded_qpose": {
            name: qpose_mean_record(values, excluded_names.get(name)) for name, values in sorted(excluded.items())
        },
    }
    stats_path = dataset_root / "meta" / "qpose_means.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def skip_record(raw_dir: Path, episode_dir: Path, reason: str) -> dict[str, Any]:
    return {
        "source": str(episode_dir),
        "source_relative": source_relative_to_raw(raw_dir, episode_dir),
        "reason": reason,
    }


def source_relative_to_raw(raw_dir: Path, episode_dir: Path) -> str:
    try:
        return str(episode_dir.resolve().relative_to(raw_dir.resolve()))
    except ValueError:
        return str(episode_dir)


def convert(
    raw_dir: Path,
    repo_id: str,
    lerobot_root: Path | None = None,
    *,
    task: str = "test",
    fps: int = 30,
    mode: str = "image",
    robot_type: str = "a3",
    joint_mode: JointMode = "both",
    sync_tolerance_ms: float = 80.0,
    projection_trim_enabled: bool = True,
    projection_urdf_path: Path = Path("/data/lyx/backup/a3_ultra_t2d0/urdf/model_planning.urdf"),
    projection_pad_frames: int = 0,
    projection_min_kept_frames: int = 30,
    projection_keep_full_when_missing: bool = True,
    projection_margin_px: int = 0,
    max_frames_per_episode: int | None = None,
    overwrite: bool = False,
) -> None:
    if mode not in {"image", "video"}:
        raise ValueError("--mode must be 'image' or 'video'")

    projection_trim = ProjectionTrimConfig(
        enabled=projection_trim_enabled,
        urdf_path=projection_urdf_path,
        pad_frames=projection_pad_frames,
        min_kept_frames=projection_min_kept_frames,
        keep_full_when_missing=projection_keep_full_when_missing,
        margin_px=projection_margin_px,
    )

    episode_dirs = find_episode_dirs(raw_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes found under {raw_dir}")

    skipped_episodes: list[dict[str, Any]] = []
    first: EpisodeData | None = None
    first_episode_dir: Path | None = None
    first_source_index: int | None = None
    for source_index, episode_dir in enumerate(episode_dirs):
        try:
            first = load_episode(
                episode_dir,
                joint_mode=joint_mode,
                sync_tolerance_ms=sync_tolerance_ms,
                max_frames=max_frames_per_episode,
                projection_trim=projection_trim,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[warn skip source {source_index:03d}] {episode_dir}: {reason}")
            skipped_episodes.append(skip_record(raw_dir, episode_dir, reason))
            continue
        first_episode_dir = episode_dir
        first_source_index = source_index
        break

    if first is None or first_episode_dir is None or first_source_index is None:
        raise RuntimeError(f"No valid episodes found under {raw_dir}")

    dataset = create_dataset(
        repo_id=repo_id,
        root=lerobot_root,
        fps=fps,
        mode=mode,
        robot_type=f"{robot_type}_{joint_mode}",
        joint_names=first.joint_names,
        rgb_shape=first.rgb_shape,
        depth_shape=first.depth_shape,
        overwrite=overwrite,
        dataset_config=DatasetConfig(use_videos=mode == "video"),
    )

    all_selected: list[np.ndarray] = []
    all_excluded: dict[str, list[np.ndarray]] = {}
    excluded_names: dict[str, list[str]] = {}
    trim_records: list[dict[str, Any]] = []
    kept_episode_count = 0

    try:
        for source_index, episode_dir in enumerate(episode_dirs):
            if source_index < first_source_index:
                continue
            if source_index == first_source_index:
                ep_data = first
            else:
                try:
                    ep_data = load_episode(
                        episode_dir,
                        joint_mode=joint_mode,
                        sync_tolerance_ms=sync_tolerance_ms,
                        max_frames=max_frames_per_episode,
                        projection_trim=projection_trim,
                    )
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    print(f"[warn skip source {source_index:03d}] {episode_dir}: {reason}")
                    skipped_episodes.append(skip_record(raw_dir, episode_dir, reason))
                    continue

            if ep_data.joint_names != first.joint_names:
                reason = f"Joint name mismatch. Expected {first.joint_names}, got {ep_data.joint_names}"
                print(f"[warn skip source {source_index:03d}] {episode_dir}: {reason}")
                skipped_episodes.append(skip_record(raw_dir, episode_dir, reason))
                continue

            try:
                episode_task = read_task(episode_dir, task)
            except Exception as exc:
                reason = f"failed to read task metadata: {type(exc).__name__}: {exc}"
                print(f"[warn skip source {source_index:03d}] {episode_dir}: {reason}")
                skipped_episodes.append(skip_record(raw_dir, episode_dir, reason))
                continue

            all_selected.extend(ep_data.selected_qpose)
            for key, values in ep_data.excluded_qpose.items():
                all_excluded.setdefault(key, []).extend(values)
                excluded_names.setdefault(key, ep_data.excluded_names[key])
            trim_records.append(
                {
                    "source": str(episode_dir),
                    "source_relative": source_relative_to_raw(raw_dir, episode_dir),
                    "trim_info": ep_data.trim_info,
                }
            )

            print(
                f"[episode {kept_episode_count:03d} source {source_index:03d}] "
                f"{episode_dir}: {len(ep_data.frames)} frames, task={episode_task!r}, "
                f"trim={ep_data.trim_info.get('trim_start')}:{ep_data.trim_info.get('trim_end')}, "
                f"applied={ep_data.trim_info.get('applied')}"
            )
            for frame in ep_data.frames:
                frame["task"] = episode_task
                dataset.add_frame(frame)
            if mode == "image":
                save_episode_keep_image_files(dataset)
            else:
                dataset.save_episode()
            kept_episode_count += 1
    finally:
        dataset.stop_image_writer()

    print(f"\nDone. LeRobot dataset root: {dataset.root}")
    print(f"Joint mode: {joint_mode}")
    print(f"Training state/action dim: {len(first.joint_names)}")
    print(f"RGB shape in LeRobot: {first.rgb_shape}")
    print(f"Depth shape in LeRobot: {first.depth_shape}")
    print(f"Kept episodes: {kept_episode_count}")
    if skipped_episodes:
        skipped_path = dataset.root / "meta" / "skipped_episodes.json"
        skipped_path.write_text(json.dumps(skipped_episodes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Skipped episodes: {len(skipped_episodes)}")
        print(f"Wrote skipped episode reasons: {skipped_path}")
    if trim_records:
        trim_path = dataset.root / "meta" / "projection_trim_stats.json"
        projection_trim_record = dataclasses.asdict(projection_trim)
        projection_trim_record["urdf_path"] = str(projection_trim.urdf_path)
        trim_path.write_text(
            json.dumps(
                {
                    "projection_trim": projection_trim_record,
                    "episodes": trim_records,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote projection trim stats: {trim_path}")
    qpose_stats = write_qpose_mean_json(
        dataset.root,
        joint_mode=joint_mode,
        selected=all_selected,
        selected_names=first.joint_names,
        excluded=all_excluded,
        excluded_names=excluded_names,
    )
    print_qpose_means("training qpose", qpose_stats["training_qpose"])
    for name, record in sorted(qpose_stats["excluded_qpose"].items()):
        print_qpose_means(f"excluded {name} qpose", record)
    print(f"\nWrote qpose means: {dataset.root / 'meta' / 'qpose_means.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="A3 collection/session directory or one episode dir.")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id, for example local_dataset/a3_0730_test_both.")
    parser.add_argument("--lerobot-root", type=Path, default=None, help="Parent directory for repo_id. Defaults to HF_LEROBOT_HOME.")
    parser.add_argument("--task", default="test", help="Fallback task if prompt.txt/metadata.json has no prompt.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--robot-type", default="a3")
    parser.add_argument("--joint-mode", choices=["left", "right", "both"], default="both", help="Select left arm+hand, right arm+hand, or both arms+hands.")
    parser.add_argument("--sync-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--no-projection-trim", dest="projection_trim_enabled", action="store_false", default=argparse.SUPPRESS, help="Disable projection-based head/tail trimming.")
    parser.add_argument("--projection-urdf-path", type=Path, default=Path("/data/lyx/backup/a3_ultra_t2d0/urdf/model_planning.urdf"), help="Path to the A3 URDF used for FK-based visibility trimming.")
    parser.add_argument("--projection-pad-frames", type=int, default=0, help="Extra frames to keep before tag1 and after tag2.")
    parser.add_argument("--projection-min-kept-frames", type=int, default=400, help="Skip an episode if projection trimming keeps fewer frames than this.")
    parser.add_argument("--keep-episodes-without-projection", dest="projection_keep_full_when_missing", action="store_false", default=argparse.SUPPRESS, help="Keep full episodes when no projected link enters the RGB FOV.")
    parser.add_argument("--projection-margin-px", type=int, default=0, help="Extra pixel margin around the RGB image bounds when judging visibility.")
    parser.add_argument("--max-frames-per-episode", type=int, default=None, help="Debug option for quick smoke tests.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(**vars(args))
