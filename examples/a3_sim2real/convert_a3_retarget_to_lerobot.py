#!/usr/bin/env python3
"""Convert A3 retarget outputs into a local LeRobot v2.1 dataset.

The retarget pipeline stores executable commands in ``a3_qpose.json`` using
the A3 wire order (waist, arms, hands). OpenPI's existing A3 data path uses
arms, hands, then waist, so this converter validates and reorders every frame
before writing it as both the observation state and absolute action target.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import shutil
import types
from typing import Any

from datasets import Dataset
from datasets import concatenate_datasets
from lerobot.common.constants import HF_LEROBOT_HOME
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image

A3_EXECUTABLE_SCHEMA = "hand_obj_tracking.a3_executable_qpose37.v1"
RGB_KEY = "observation.images.chest_front_rgb"
DEPTH_KEY = "observation.images.chest_front_depth"

WIRE_WAIST_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
ARM_NAMES = tuple(
    f"{side}_{suffix}_joint"
    for side in ("left", "right")
    for suffix in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
)
HAND_NAMES = tuple(
    f"{side}_{suffix}"
    for side in ("left", "right")
    for suffix in (
        "thumb_rotation_0",
        "thumb_wiggles_1",
        "thumb_bent_2",
        "index_wiggles_0",
        "index_bent_1",
        "middle_bent",
        "ring_wiggles_0",
        "ring_bent_1",
        "pinky_wiggles_0",
        "pinky_bent_1",
    )
)
A3_WIRE_NAMES = WIRE_WAIST_NAMES + ARM_NAMES + HAND_NAMES
TRAINING_WAIST_NAMES = ("waist_roll", "waist_pitch", "waist_yaw")
TRAINING_NAMES = ARM_NAMES + HAND_NAMES + TRAINING_WAIST_NAMES
ACTION_DIM = len(TRAINING_NAMES)
VALID_SIDES = frozenset({"left", "right"})


@dataclasses.dataclass(frozen=True)
class FrameSpec:
    frame: str
    source_frame_index: int
    timestamp_ns: int
    action: np.ndarray
    rgb_path: Path
    depth_path: Path


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    path: Path
    active_sides: tuple[str, ...]
    frames: tuple[FrameSpec, ...]
    source_schema: str
    source_fps: int | None


@dataclasses.dataclass(frozen=True)
class ImageSpec:
    rgb_shape: tuple[int, int, int]
    depth_shape: tuple[int, int, int]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 1e-4
    image_writer_processes: int = 0
    image_writer_threads: int = 4
    video_backend: str | None = None


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def wire_to_training_qpose(values: Any, *, label: str = "a3_qpose_37d") -> np.ndarray:
    qpose = np.asarray(values, dtype=np.float64)
    if qpose.shape != (ACTION_DIM,):
        raise ValueError(f"{label} must have shape ({ACTION_DIM},), got {qpose.shape}")
    if not np.isfinite(qpose).all():
        raise ValueError(f"{label} contains NaN or Inf")

    hand = qpose[17:37]
    if np.any(hand < 0.0) or np.any(hand > 4095.0):
        raise ValueError(f"{label} hand actuator values must be within [0, 4095]")
    if not np.equal(hand, np.rint(hand)).all():
        raise ValueError(f"{label} hand actuator values must be integers")

    # Wire: yaw/roll/pitch + arms + hands. Training: arms + hands + roll/pitch/yaw.
    return np.concatenate((qpose[3:17], qpose[17:37], qpose[[1, 2, 0]])).astype(np.float32)


def discover_episode_dirs(raw_dir: Path) -> list[Path]:
    raw_dir = raw_dir.expanduser().resolve()
    if (raw_dir / "a3_qpose.json").is_file():
        return [raw_dir]
    return sorted(path for path in raw_dir.iterdir() if path.is_dir() and (path / "a3_qpose.json").is_file())


def read_source_fps(episode_dir: Path) -> int | None:
    metadata_path = episode_dir / "cfg" / "session_meta.json"
    if not metadata_path.is_file():
        return None
    metadata = load_json_object(metadata_path)
    raw_fps = metadata.get("source_metadata", {}).get("fps")
    if raw_fps is None:
        return None
    fps = int(raw_fps)
    if fps <= 0 or not math.isclose(float(raw_fps), fps):
        raise ValueError(f"Invalid source fps {raw_fps!r} in {metadata_path}")
    return fps


def load_episode_spec(
    episode_dir: Path,
    *,
    rgb_name: str,
    depth_name: str,
    expected_fps: int,
    max_frames: int | None = None,
) -> EpisodeSpec:
    action_path = episode_dir / "a3_qpose.json"
    payload = load_json_object(action_path)
    schema = str(payload.get("schema", ""))
    if schema != A3_EXECUTABLE_SCHEMA:
        raise ValueError(f"Unexpected schema {schema!r} in {action_path}; expected {A3_EXECUTABLE_SCHEMA!r}")

    names = tuple(payload.get("a3_qpose_37d_names", ()))
    if names != A3_WIRE_NAMES:
        raise ValueError(f"Unexpected A3 qpose order in {action_path}; regenerate it with the current exporter")

    raw_sides = payload.get("active_sides")
    if not isinstance(raw_sides, list) or not raw_sides:
        raise ValueError(f"Missing active_sides in {action_path}")
    unknown_sides = set(raw_sides) - VALID_SIDES
    if unknown_sides:
        raise ValueError(f"Unknown active_sides in {action_path}: {sorted(unknown_sides)}")
    active_sides = tuple(side for side in ("left", "right") if side in raw_sides)

    source_fps = read_source_fps(episode_dir)
    if source_fps is not None and source_fps != expected_fps:
        raise ValueError(f"Source fps {source_fps} in {episode_dir} does not match requested fps {expected_fps}")

    rows = payload.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No action frames in {action_path}")
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        rows = rows[:max_frames]

    frames: list[FrameSpec] = []
    seen_frames: set[str] = set()
    previous_source_frame_index: int | None = None
    previous_timestamp_ns: int | None = None
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{action_path}: frame row {row_index} is not an object")
        frame = str(row.get("frame", ""))
        if not frame or Path(frame).name != frame or frame in {".", ".."}:
            raise ValueError(f"{action_path}: invalid frame name at row {row_index}: {frame!r}")
        if frame in seen_frames:
            raise ValueError(f"{action_path}: duplicate frame {frame!r}")
        seen_frames.add(frame)

        source_frame_index = int(row.get("source_frame_index", -1))
        timestamp_ns = int(row.get("timestamp_ns", -1))
        if source_frame_index < 0 or (
            previous_source_frame_index is not None and source_frame_index <= previous_source_frame_index
        ):
            raise ValueError(f"{action_path}: source_frame_index is not strictly increasing at frame {frame}")
        # Source timestamps were serialized through floating point in some retarget
        # sessions, so adjacent 30 Hz frames can quantize to the same nanosecond.
        # Frame order is defined by source_frame_index; only an actual time reversal
        # indicates corrupt ordering.
        if timestamp_ns <= 0:
            raise ValueError(f"{action_path}: timestamp_ns must be positive at frame {frame}")
        if previous_timestamp_ns is not None and timestamp_ns < previous_timestamp_ns:
            raise ValueError(f"{action_path}: timestamp_ns decreases at frame {frame}")
        previous_source_frame_index = source_frame_index
        previous_timestamp_ns = timestamp_ns

        frame_dir = episode_dir / "all_data" / frame
        rgb_path = frame_dir / rgb_name
        depth_path = frame_dir / depth_name
        missing = [str(path) for path in (rgb_path, depth_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{action_path}: frame {frame} is missing files: {missing}")
        action = wire_to_training_qpose(
            row.get("a3_qpose_37d"),
            label=f"{action_path}: frame {frame} a3_qpose_37d",
        )
        frames.append(
            FrameSpec(
                frame=frame,
                source_frame_index=source_frame_index,
                timestamp_ns=timestamp_ns,
                action=action,
                rgb_path=rgb_path,
                depth_path=depth_path,
            )
        )

    return EpisodeSpec(
        path=episode_dir,
        active_sides=active_sides,
        frames=tuple(frames),
        source_schema=schema,
        source_fps=source_fps,
    )


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (height, width, 3), got {rgb.shape}: {path}")
    return np.ascontiguousarray(rgb)


def load_depth(path: Path) -> Image.Image:
    with Image.open(path) as image:
        depth = np.asarray(image)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"Depth image must be a uint16 single-channel PNG, got {depth.dtype} {depth.shape}: {path}")
    return Image.fromarray(np.ascontiguousarray(depth), mode="I;16")


def inspect_image_spec(frame: FrameSpec) -> ImageSpec:
    rgb = load_rgb(frame.rgb_path)
    depth = load_depth(frame.depth_path)
    if rgb.shape[:2] != depth.size[::-1]:
        raise ValueError(
            f"RGB/depth shape mismatch for frame {frame.frame}: RGB {rgb.shape[:2]}, depth {depth.size[::-1]}"
        )
    return ImageSpec(rgb_shape=rgb.shape, depth_shape=(*depth.size[::-1], 1))


def validate_repo_id(repo_id: str) -> None:
    path = Path(repo_id)
    if path.is_absolute() or len(path.parts) != 2 or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(
            "repo_id must have exactly two safe path components, for example local_dataset/a3_sim2real_move_cup"
        )


def create_dataset(
    *,
    repo_id: str,
    root: Path | None,
    fps: int,
    mode: str,
    robot_type: str,
    image_spec: ImageSpec,
    overwrite: bool,
    dataset_config: DatasetConfig,
) -> LeRobotDataset:
    validate_repo_id(repo_id)
    parent = (root if root is not None else HF_LEROBOT_HOME).expanduser().resolve()
    dataset_root = (parent / repo_id).resolve()
    if parent not in dataset_root.parents:
        raise ValueError(f"Dataset path escapes LeRobot root: {dataset_root}")
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_root)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [list(TRAINING_NAMES)],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [list(TRAINING_NAMES)],
        },
        RGB_KEY: {
            "dtype": mode,
            "shape": image_spec.rgb_shape,
            "names": ["height", "width", "channels"],
        },
        DEPTH_KEY: {
            "dtype": mode,
            "shape": image_spec.depth_shape,
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


def _save_episode_table_with_external_images(
    self: LeRobotDataset,
    episode_buffer: dict[str, Any],
    episode_index: int,
) -> None:
    episode_dict = {key: episode_buffer[key] for key in self.hf_features}
    episode_dataset = Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
    self.hf_dataset = concatenate_datasets([self.hf_dataset, episode_dataset])
    self.hf_dataset.set_transform(lerobot_dataset_module.hf_transform_to_torch)
    episode_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    episode_dataset.to_parquet(episode_path)


def save_episode_keep_image_files(dataset: LeRobotDataset) -> None:
    """Keep external PNGs for LeRobot image mode with the pinned LeRobot version."""
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


def episode_manifest_record(raw_dir: Path, episode: EpisodeSpec) -> dict[str, Any]:
    try:
        relative_path = str(episode.path.resolve().relative_to(raw_dir.resolve()))
    except ValueError:
        relative_path = str(episode.path)
    return {
        "source": str(episode.path),
        "source_relative": relative_path,
        "source_schema": episode.source_schema,
        "source_fps": episode.source_fps,
        "active_sides": list(episode.active_sides),
        "frame_count": len(episode.frames),
        "first_frame": episode.frames[0].frame,
        "last_frame": episode.frames[-1].frame,
    }


def preflight(
    raw_dir: Path,
    *,
    rgb_name: str,
    depth_name: str,
    fps: int,
    max_episodes: int | None,
    max_frames_per_episode: int | None,
    skip_invalid: bool,
) -> tuple[list[EpisodeSpec], list[dict[str, str]], ImageSpec]:
    episode_dirs = discover_episode_dirs(raw_dir)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        episode_dirs = episode_dirs[:max_episodes]
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories containing a3_qpose.json found under {raw_dir}")

    episodes: list[EpisodeSpec] = []
    skipped: list[dict[str, str]] = []
    image_spec: ImageSpec | None = None
    for source_index, episode_dir in enumerate(episode_dirs):
        try:
            episode = load_episode_spec(
                episode_dir,
                rgb_name=rgb_name,
                depth_name=depth_name,
                expected_fps=fps,
                max_frames=max_frames_per_episode,
            )
            current_image_spec = inspect_image_spec(episode.frames[0])
            if image_spec is None:
                image_spec = current_image_spec
            elif current_image_spec != image_spec:
                raise ValueError(
                    f"Image shape mismatch: expected {image_spec}, got {current_image_spec} in {episode_dir}"
                )
        except Exception as exc:
            if not skip_invalid:
                raise
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[warn skip source {source_index:03d}] {episode_dir}: {reason}")
            skipped.append({"source": str(episode_dir), "reason": reason})
            continue
        episodes.append(episode)
        print(
            f"[preflight source {source_index:03d}] {episode.path.name}: "
            f"{len(episode.frames)} frames, active_sides={episode.active_sides}"
        )

    if not episodes or image_spec is None:
        raise RuntimeError(f"No valid episodes found under {raw_dir}")
    return episodes, skipped, image_spec


def write_metadata(
    dataset_root: Path,
    *,
    raw_dir: Path,
    repo_id: str,
    task: str,
    fps: int,
    rgb_name: str,
    depth_name: str,
    episodes: list[EpisodeSpec],
    skipped: list[dict[str, str]],
    qpose_sum: np.ndarray,
    frame_count: int,
) -> None:
    manifest = {
        "schema": "openpi.a3_sim2real_conversion.v1",
        "source_root": str(raw_dir.resolve()),
        "repo_id": repo_id,
        "task": task,
        "fps": fps,
        "rgb_name": rgb_name,
        "depth_name": depth_name,
        "action_source": "a3_qpose.json:frames[].a3_qpose_37d",
        "source_order": list(A3_WIRE_NAMES),
        "training_order": list(TRAINING_NAMES),
        "mapping": "source[3:17] + source[17:37] + source[[1,2,0]]",
        "observation_action_policy": "observation.state and action contain the same absolute executable qpose",
        "episode_count": len(episodes),
        "frame_count": frame_count,
        "episodes": [episode_manifest_record(raw_dir, episode) for episode in episodes],
        "skipped_episodes": skipped,
    }
    meta_dir = dataset_root / "meta"
    (meta_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    qpose_means = {
        "joint_mode": "both",
        "training_qpose": {
            "dim": ACTION_DIM,
            "frame_count": frame_count,
            "joint_names": list(TRAINING_NAMES),
            "mean": (qpose_sum / frame_count).tolist(),
        },
        "excluded_qpose": {},
    }
    (meta_dir / "qpose_means.json").write_text(
        json.dumps(qpose_means, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def convert(
    raw_dir: Path,
    repo_id: str,
    lerobot_root: Path | None = None,
    *,
    task: str,
    fps: int = 30,
    mode: str = "image",
    robot_type: str = "a3_sim2real",
    rgb_name: str = "rgb_out.png",
    depth_name: str = "depth.png",
    max_episodes: int | None = None,
    max_frames_per_episode: int | None = None,
    skip_invalid: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if not task.strip():
        raise ValueError("task must not be empty")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if mode not in {"image", "video"}:
        raise ValueError("mode must be 'image' or 'video'")
    validate_repo_id(repo_id)

    episodes, skipped, image_spec = preflight(
        raw_dir,
        rgb_name=rgb_name,
        depth_name=depth_name,
        fps=fps,
        max_episodes=max_episodes,
        max_frames_per_episode=max_frames_per_episode,
        skip_invalid=skip_invalid,
    )
    total_frames = sum(len(episode.frames) for episode in episodes)
    print(
        f"Preflight complete: {len(episodes)} episodes, {total_frames} frames, "
        f"RGB={image_spec.rgb_shape}, depth={image_spec.depth_shape}"
    )
    if dry_run:
        print("Dry run complete; no LeRobot dataset was created.")
        return

    dataset = create_dataset(
        repo_id=repo_id,
        root=lerobot_root,
        fps=fps,
        mode=mode,
        robot_type=robot_type,
        image_spec=image_spec,
        overwrite=overwrite,
        dataset_config=DatasetConfig(use_videos=mode == "video"),
    )
    qpose_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    written_frames = 0
    try:
        for episode_index, episode in enumerate(episodes):
            for frame in episode.frames:
                rgb = load_rgb(frame.rgb_path)
                depth = load_depth(frame.depth_path)
                current_spec = ImageSpec(rgb_shape=rgb.shape, depth_shape=(*depth.size[::-1], 1))
                if current_spec != image_spec:
                    raise ValueError(
                        f"Image shape changed at {episode.path.name}/{frame.frame}: "
                        f"expected {image_spec}, got {current_spec}"
                    )
                dataset.add_frame(
                    {
                        "observation.state": frame.action.copy(),
                        "action": frame.action.copy(),
                        RGB_KEY: rgb,
                        DEPTH_KEY: depth,
                        "task": task,
                    }
                )
                qpose_sum += frame.action
                written_frames += 1
            if mode == "image":
                save_episode_keep_image_files(dataset)
            else:
                dataset.save_episode()
            print(
                f"[episode {episode_index:03d}] {episode.path.name}: "
                f"{len(episode.frames)} frames, active_sides={episode.active_sides}"
            )
    finally:
        dataset.stop_image_writer()

    if written_frames != total_frames:
        raise RuntimeError(f"Expected to write {total_frames} frames, wrote {written_frames}")
    write_metadata(
        dataset.root,
        raw_dir=raw_dir,
        repo_id=repo_id,
        task=task,
        fps=fps,
        rgb_name=rgb_name,
        depth_name=depth_name,
        episodes=episodes,
        skipped=skipped,
        qpose_sum=qpose_sum,
        frame_count=written_frames,
    )
    print(f"Done. LeRobot dataset root: {dataset.root}")
    print(f"Episodes: {len(episodes)}, frames: {written_frames}, action dim: {ACTION_DIM}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Retarget task root or one session directory.")
    parser.add_argument(
        "--repo-id", required=True, help="LeRobot repo id, for example local_dataset/a3_sim2real_move_cup_20260913."
    )
    parser.add_argument(
        "--lerobot-root", type=Path, default=None, help="Parent directory for repo-id; defaults to HF_LEROBOT_HOME."
    )
    parser.add_argument("--task", required=True, help="Language task stored for every episode.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--robot-type", default="a3_sim2real")
    parser.add_argument("--rgb-name", default="rgb_out.png", help="Per-frame synthesized RGB filename.")
    parser.add_argument("--depth-name", default="depth.png", help="Per-frame uint16 depth filename.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug option for smoke tests.")
    parser.add_argument("--max-frames-per-episode", type=int, default=None, help="Debug option for smoke tests.")
    parser.add_argument(
        "--skip-invalid", action="store_true", help="Skip invalid episodes instead of failing preflight."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without creating a dataset.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(**vars(parse_args()))
