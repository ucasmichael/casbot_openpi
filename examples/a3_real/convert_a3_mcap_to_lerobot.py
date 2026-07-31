#!/usr/bin/env python3
"""Convert A3 MCAP episodes to LeRobot v2.1.

Training samples contain only:
- chest RGB image
- qpose state/action selected by --joint-mode

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
from typing import Any, Literal

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
from mcap_ros2.decoder import Decoder
import numpy as np

RGB_TOPIC = "/hal/chest_front_d457_camera/rgb"
ARM_TOPIC = "/motion/control/arm_joint_state"
HAND_TOPIC = "/motion/control/hand_joint_state"
NECK_TOPIC = "/motion/control/neck_joint_state"

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


def read_mcap_files(mcap_files: list[Path]) -> dict[str, TopicSeries]:
    decoder = Decoder()
    topics: dict[str, TopicSeries] = {}

    for mcap_file in mcap_files:
        with mcap_file.open("rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                if topic not in {RGB_TOPIC, ARM_TOPIC, HAND_TOPIC, NECK_TOPIC}:
                    continue

                ros_msg = decoder.decode(schema, message)
                if not hasattr(ros_msg, "header"):
                    continue

                time_ns = stamp_to_ns(ros_msg.header.stamp)
                series = topics.setdefault(topic, TopicSeries())

                if topic == RGB_TOPIC:
                    series.append(time_ns, decode_rgb_image(ros_msg))
                    continue

                names = list(ros_msg.name)
                positions = np.asarray(ros_msg.position, dtype=np.float32)
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


def side_indices(names: list[str], side: str) -> list[int]:
    prefix = f"{side}_"
    return [idx for idx, name in enumerate(names) if name.startswith(prefix)]


def select_qpose(
    arm: np.ndarray,
    hand: np.ndarray,
    arm_names: list[str],
    hand_names: list[str],
    joint_mode: JointMode,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray], dict[str, list[str]]]:
    left_arm_idx = side_indices(arm_names, "left")
    right_arm_idx = side_indices(arm_names, "right")
    left_hand_idx = side_indices(hand_names, "left")
    right_hand_idx = side_indices(hand_names, "right")

    groups = {
        "left": (
            np.concatenate([arm[left_arm_idx], hand[left_hand_idx]]).astype(np.float32),
            [*(arm_names[i] for i in left_arm_idx), *(hand_names[i] for i in left_hand_idx)],
        ),
        "right": (
            np.concatenate([arm[right_arm_idx], hand[right_hand_idx]]).astype(np.float32),
            [*(arm_names[i] for i in right_arm_idx), *(hand_names[i] for i in right_hand_idx)],
        ),
    }

    if joint_mode == "left":
        selected_values, selected_names = groups["left"]
        excluded = {"right_arm_hand": groups["right"][0]}
        excluded_names = {"right_arm_hand": groups["right"][1]}
    elif joint_mode == "right":
        selected_values, selected_names = groups["right"]
        excluded = {"left_arm_hand": groups["left"][0]}
        excluded_names = {"left_arm_hand": groups["left"][1]}
    elif joint_mode == "both":
        selected_values = np.concatenate([groups["left"][0], groups["right"][0]]).astype(np.float32)
        selected_names = [*groups["left"][1], *groups["right"][1]]
        excluded = {}
        excluded_names = {}
    else:
        raise ValueError(f"Unknown joint mode: {joint_mode}")

    return selected_values, selected_names, excluded, excluded_names


def load_episode(
    episode_dir: Path,
    *,
    joint_mode: JointMode,
    sync_tolerance_ms: float,
    max_frames: int | None,
) -> EpisodeData:
    mcap_files = sorted(episode_dir.glob("**/*.mcap"))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found under {episode_dir}")

    topics = read_mcap_files(mcap_files)
    required = [RGB_TOPIC, ARM_TOPIC, HAND_TOPIC]
    missing = [topic for topic in required if topic not in topics or not topics[topic].values]
    if missing:
        raise ValueError(f"{episode_dir} is missing required topics: {missing}")

    arm_names = topics[ARM_TOPIC].names or []
    hand_names = topics[HAND_TOPIC].names or []
    neck_names = topics[NECK_TOPIC].names or []
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
        neck = nearest_value(topics[NECK_TOPIC], rgb_time_ns, tolerance_ns) if NECK_TOPIC in topics else None
        if arm is None or hand is None:
            continue

        state, names, excluded, names_excluded = select_qpose(arm, hand, arm_names, hand_names, joint_mode)
        if selected_names is None:
            selected_names = names
        elif selected_names != names:
            raise ValueError(f"Selected joint names changed inside {episode_dir}")

        selected_qpose.append(state)
        for key, value in excluded.items():
            excluded_qpose.setdefault(key, []).append(value)
            excluded_names.setdefault(key, names_excluded[key])
        if neck is not None:
            excluded_qpose.setdefault("neck", []).append(np.asarray(neck, dtype=np.float32))
            excluded_names.setdefault("neck", neck_names)

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

    if len(selected_names) not in {17, 34}:
        raise ValueError(f"Unexpected selected qpose dim {len(selected_names)} for mode {joint_mode}: {selected_names}")

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

    first = load_episode(
        episode_dirs[0],
        joint_mode=joint_mode,
        sync_tolerance_ms=sync_tolerance_ms,
        max_frames=max_frames_per_episode,
    )
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

    try:
        for ep_index, episode_dir in enumerate(episode_dirs):
            ep_data = first if ep_index == 0 else load_episode(
                episode_dir,
                joint_mode=joint_mode,
                sync_tolerance_ms=sync_tolerance_ms,
                max_frames=max_frames_per_episode,
            )
            if ep_data.joint_names != first.joint_names:
                raise ValueError(f"Joint name mismatch in {episode_dir}")

            all_selected.extend(ep_data.selected_qpose)
            for key, values in ep_data.excluded_qpose.items():
                all_excluded.setdefault(key, []).extend(values)
                excluded_names.setdefault(key, ep_data.excluded_names[key])

            episode_task = read_task(episode_dir, task)
            print(f"[episode {ep_index:03d}] {episode_dir}: {len(ep_data.frames)} frames, task={episode_task!r}")
            for frame in ep_data.frames:
                frame["task"] = episode_task
                dataset.add_frame(frame)
            dataset.save_episode()
    finally:
        dataset.stop_image_writer()

    print(f"\nDone. LeRobot dataset root: {dataset.root}")
    print(f"Joint mode: {joint_mode}")
    print(f"Training state/action dim: {len(first.joint_names)}")
    print(f"RGB shape in LeRobot: {first.rgb_shape}")
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
