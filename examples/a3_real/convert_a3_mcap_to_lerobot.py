#!/usr/bin/env python3
"""Convert A3 MCAP episodes to LeRobot v2.1.

Training samples contain only:
- chest RGB image
- arm command, hand command, and move-waist rpy state/action

The collector also records depth and neck qpose, but this script does not write
them into LeRobot training samples. At the end it prints per-dimension means for
the selected training qpose and for the qpose dimensions left out of training.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import dataclasses
import json
from pathlib import Path
import shutil
import struct
from typing import Any, Literal

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
from mcap_ros2.decoder import Decoder
import numpy as np

RGB_TOPIC = "/hal/chest_front_d457_camera/rgb"
ARM_TOPIC = "/motion/control/arm_joint_command"
HAND_TOPIC = "/motion/control/hand_joint_command"
WAIST_MOVE_TOPIC = (
    "/motion/control/move_waist/"
    "pb_3Aaimdk_2Eprotocol_2EMotionControlMoveWaistChannel"
)
ARM_DIM = 14
HAND_DIM = 20
WAIST_RPY_NAMES = ["waist_roll", "waist_pitch", "waist_yaw"]
TRAINING_QPOSE_DIM = ARM_DIM + HAND_DIM + len(WAIST_RPY_NAMES)

CHEST_RGB_KEY = "observation.images.chest_front_rgb"
JointMode = Literal["left", "right", "both"]


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
    selected_qpose: list[np.ndarray]
    excluded_qpose: dict[str, list[np.ndarray]]
    excluded_names: dict[str, list[str]]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1e-4
    image_writer_processes: int = 0
    image_writer_threads: int = 4
    video_backend: str | None = None


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def message_time_ns(ros_msg: Any, mcap_message: Any) -> int:
    header = getattr(ros_msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        return stamp_to_ns(stamp)
    return int(getattr(mcap_message, "log_time"))


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
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    out = bytearray()
    for item in raw:
        if isinstance(item, int):
            out.append(item & 0xFF)
        elif isinstance(item, (bytes, bytearray)):
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


def read_mcap_files(mcap_files: list[Path]) -> dict[str, TopicSeries]:
    decoder = Decoder()
    topics: dict[str, TopicSeries] = {}

    for mcap_file in mcap_files:
        with mcap_file.open("rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                if topic not in {RGB_TOPIC, ARM_TOPIC, HAND_TOPIC, WAIST_MOVE_TOPIC}:
                    continue

                ros_msg = decoder.decode(schema, message)
                time_ns = message_time_ns(ros_msg, message)
                series = topics.setdefault(topic, TopicSeries())

                if topic == RGB_TOPIC:
                    series.append(time_ns, decode_rgb_image(ros_msg))
                    continue
                if topic == WAIST_MOVE_TOPIC:
                    series.append(time_ns, parse_move_waist_wrapper(ros_msg))
                    continue

                names = list(ros_msg.name)
                positions = np.asarray(ros_msg.position, dtype=np.float32)
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
) -> EpisodeData:
    if joint_mode != "both":
        raise ValueError("A3 command+waist conversion only supports --joint-mode both")

    mcap_files = sorted(episode_dir.glob("**/*.mcap"))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found under {episode_dir}")

    topics = read_mcap_files(mcap_files)
    required = [RGB_TOPIC, ARM_TOPIC, HAND_TOPIC, WAIST_MOVE_TOPIC]
    missing = [topic for topic in required if topic not in topics or not topics[topic].values]
    if missing:
        raise ValueError(f"{episode_dir} is missing required topics: {missing}")

    arm_names = topics[ARM_TOPIC].names or []
    hand_names = topics[HAND_TOPIC].names or []
    tolerance_ns = int(sync_tolerance_ms * 1_000_000)

    frames: list[dict[str, Any]] = []
    selected_qpose: list[np.ndarray] = []
    excluded_qpose: dict[str, list[np.ndarray]] = {}
    excluded_names: dict[str, list[str]] = {}
    selected_names: list[str] | None = None

    rgb_series = topics[RGB_TOPIC]
    for rgb_time_ns, rgb in zip(rgb_series.times_ns, rgb_series.values, strict=True):
        arm = nearest_value(topics[ARM_TOPIC], rgb_time_ns, tolerance_ns)
        hand = nearest_value(topics[HAND_TOPIC], rgb_time_ns, tolerance_ns)
        waist_rpy = nearest_value(topics[WAIST_MOVE_TOPIC], rgb_time_ns, tolerance_ns)
        if arm is None or hand is None or waist_rpy is None:
            continue

        state, names = make_qpose(arm, hand, waist_rpy, arm_names, hand_names)
        if selected_names is None:
            selected_names = names
        elif selected_names != names:
            raise ValueError(f"Selected joint names changed inside {episode_dir}")

        selected_qpose.append(state)
        frames.append(
            {
                "observation.state": state,
                "action": state.copy(),
                CHEST_RGB_KEY: rgb,
            }
        )

        if max_frames is not None and len(frames) >= max_frames:
            break

    if not frames or selected_names is None:
        raise ValueError(f"No synchronized frames found for {episode_dir}")

    if len(selected_names) != TRAINING_QPOSE_DIM:
        raise ValueError(
            f"Unexpected selected qpose dim {len(selected_names)} for mode {joint_mode}: {selected_names}"
        )

    return EpisodeData(
        frames=frames,
        joint_names=selected_names,
        rgb_shape=frames[0][CHEST_RGB_KEY].shape,
        selected_qpose=selected_qpose,
        excluded_qpose=excluded_qpose,
        excluded_names=excluded_names,
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
    try:
        source_relative = str(episode_dir.resolve().relative_to(raw_dir.resolve()))
    except ValueError:
        source_relative = str(episode_dir)
    return {
        "source": str(episode_dir),
        "source_relative": source_relative,
        "reason": reason,
    }


def convert(
    raw_dir: Path,
    repo_id: str,
    lerobot_root: Path | None = None,
    *,
    task: str = "test",
    fps: int = 30,
    mode: str = "video",
    robot_type: str = "a3",
    joint_mode: JointMode = "both",
    sync_tolerance_ms: float = 80.0,
    max_frames_per_episode: int | None = None,
    overwrite: bool = False,
) -> None:
    if mode not in {"image", "video"}:
        raise ValueError("--mode must be 'image' or 'video'")

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
        overwrite=overwrite,
        dataset_config=DatasetConfig(use_videos=mode == "video"),
    )

    all_selected: list[np.ndarray] = []
    all_excluded: dict[str, list[np.ndarray]] = {}
    excluded_names: dict[str, list[str]] = {}
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

            print(
                f"[episode {kept_episode_count:03d} source {source_index:03d}] "
                f"{episode_dir}: {len(ep_data.frames)} frames, task={episode_task!r}"
            )
            for frame in ep_data.frames:
                frame["task"] = episode_task
                dataset.add_frame(frame)
            dataset.save_episode()
            kept_episode_count += 1
    finally:
        dataset.stop_image_writer()

    print(f"\nDone. LeRobot dataset root: {dataset.root}")
    print(f"Joint mode: {joint_mode}")
    print(f"Training state/action dim: {len(first.joint_names)}")
    print(f"RGB shape in LeRobot: {first.rgb_shape}")
    print(f"Kept episodes: {kept_episode_count}")
    if skipped_episodes:
        skipped_path = dataset.root / "meta" / "skipped_episodes.json"
        skipped_path.write_text(json.dumps(skipped_episodes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Skipped episodes: {len(skipped_episodes)}")
        print(f"Wrote skipped episode reasons: {skipped_path}")
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
    parser.add_argument("--mode", choices=["image", "video"], default="video")
    parser.add_argument("--robot-type", default="a3")
    parser.add_argument("--joint-mode", choices=["left", "right", "both"], default="both", help="Select left arm+hand, right arm+hand, or both arms+hands.")
    parser.add_argument("--sync-tolerance-ms", type=float, default=80.0)
    parser.add_argument("--max-frames-per-episode", type=int, default=None, help="Debug option for quick smoke tests.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(**vars(args))
