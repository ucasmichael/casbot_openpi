#!/usr/bin/env python3
"""Convert egodex.ipc_collection.v1 SQLite/HDF5 episodes to A3 LeRobot v2.1.

Like convert_a3_mcap_to_lerobot.py, state defaults to measured feedback. Missing
feedback produces a warning and falls back to the latest command at or before
the observation, within the synchronization tolerance. See README.md for camera and timing
limitations. RGB and millimetre uint16 depth are saved as lossless PNG images.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from bisect import bisect_right
from contextlib import closing
import dataclasses
import itertools
import json
import logging
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
from PIL import Image

if __package__:
    from . import convert_a3_mcap_to_lerobot as mcap
else:
    import convert_a3_mcap_to_lerobot as mcap

ARM_NAMES = mcap.ARM_NAMES
HAND_NAMES = mcap.DEFAULT_HAND_NAMES
JOINT_NAMES = [*ARM_NAMES, *HAND_NAMES, *mcap.WAIST_RPY_NAMES]
COMMAND_KEYS = ("arm_command", "hand_command", "move_waist")
STATE_KEYS = ("arm_state", "hand_state", "waist_state")
TOPICS = {
    "arm_command": (mcap.ARM_TOPIC, "joint_msgs/msg/JointCommand"),
    "hand_command": (mcap.HAND_TOPIC, "sensor_msgs/msg/JointState"),
    "move_waist": (mcap.WAIST_MOVE_TOPIC, "ros2_plugin_proto/msg/RosMsgWrapper"),
    "arm_state": (mcap.ARM_STATE_TOPIC, "sensor_msgs/msg/JointState"),
    "hand_state": (mcap.HAND_STATE_TOPIC, "sensor_msgs/msg/JointState"),
    "waist_state": (mcap.WAIST_STATE_TOPIC, "sensor_msgs/msg/JointState"),
}
ordered_positions = mcap.ordered_positions


@dataclasses.dataclass(frozen=True)
class Config:
    fps: int = 30
    state_source: str = "feedback"
    command_alignment: str = "nearest"
    sync_tolerance_ms: float = 100.0
    max_clock_uncertainty_ms: float = 5.0
    image_tolerance_ms: float = 20.0
    max_rgb_depth_skew_ms: float = 10.0
    hand_range_tolerance: float = 1.0
    image_size: tuple[int, int] | None = None  # width, height; native by default
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if self.state_source not in {"command", "feedback"} or self.command_alignment not in {"nearest", "previous"}:
            raise ValueError("Invalid state source or command alignment")
        thresholds = (
            self.sync_tolerance_ms,
            self.max_clock_uncertainty_ms,
            self.image_tolerance_ms,
            self.max_rgb_depth_skew_ms,
            self.hand_range_tolerance,
        )
        if self.fps <= 0 or any(not np.isfinite(v) or v < 0 for v in thresholds):
            raise ValueError("fps must be positive and tolerances must be finite and nonnegative")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.image_size is not None and (len(self.image_size) != 2 or min(self.image_size) <= 0):
            raise ValueError("image_size must contain positive width and height")


@dataclasses.dataclass
class Sample:
    message_id: int
    time_ns: int
    uncertainty_ns: int
    values: np.ndarray | None
    error: str | None = None


@dataclasses.dataclass
class EpisodePlan:
    source: Path
    task: str
    camera: dict[str, Any]
    rgb_shape: tuple[int, int, int]
    states: list[np.ndarray]
    actions: list[np.ndarray]
    mapping: list[dict[str, Any]]
    report: dict[str, Any]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_episode_dirs(raw_dir: Path) -> list[Path]:
    # Discover by metadata too, so incomplete episodes are reported instead of hidden.
    candidates = {p.parent for p in raw_dir.rglob("robot.sqlite3")}
    candidates.update(p.parent for p in raw_dir.rglob("rgbd.h5"))
    candidates.update(p.parent for p in raw_dir.rglob("metadata.json") if p.parent.name.startswith("episode_"))
    if (raw_dir / "metadata.json").is_file():
        candidates.add(raw_dir)
    return sorted(candidates)


def parse_values(key: str, fields: dict[str, Any]) -> np.ndarray:
    if key == "move_waist":
        return mcap.parse_move_waist_wrapper(SimpleNamespace(data=fields["data"]))
    if key == "arm_command":
        joints = fields["joints"]
        return ordered_positions([j["name"] for j in joints], [j["position"] for j in joints], ARM_NAMES)
    names = ARM_NAMES if key == "arm_state" else mcap.WAIST_RPY_NAMES if key == "waist_state" else HAND_NAMES
    return ordered_positions(fields["name"], fields["position"], names, unnamed=key == "hand_command")


def read_series(episode: Path, config: Config) -> dict[str, mcap.TopicSeries]:
    required = (*COMMAND_KEYS, *STATE_KEYS) if config.state_source == "feedback" else COMMAND_KEYS
    series = {key: mcap.TopicSeries() for key in required}
    uri = (episode / "robot.sqlite3").resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("SQLite quick_check failed")
        topics = {key: (topic, typ) for key, topic, typ in connection.execute("SELECT key, topic, type FROM topics")}
        for key in required:
            if topics.get(key) != TOPICS[key]:
                if key in STATE_KEYS:
                    continue
                raise ValueError(f"Missing or unexpected topic for {key}: {topics.get(key)}")
            rows = connection.execute(
                "SELECT id, ipc_time_ns, clock_uncertainty_ns, fields_json FROM messages "
                "WHERE key=? ORDER BY ipc_time_ns, id",
                (key,),
            )
            for message_id, time_ns, uncertainty_ns, fields_json in rows:
                if not isinstance(time_ns, int) or time_ns <= 0:
                    if key in STATE_KEYS:
                        continue
                    raise ValueError(f"Missing/invalid calibrated IPC time: {key} message {message_id}")
                error = None
                try:
                    if uncertainty_ns is None or uncertainty_ns < 0:
                        raise ValueError("Missing/invalid clock uncertainty")
                    values = parse_values(key, json.loads(fields_json))
                except (ValueError, TypeError, KeyError) as exc:
                    if key not in STATE_KEYS:
                        raise
                    values, error = None, str(exc)
                series[key].append(time_ns, Sample(message_id, time_ns, uncertainty_ns, values, error))
            if not series[key].values and key in COMMAND_KEYS:
                raise ValueError(f"No messages for {key}")
    return series


def select_sample(series: mcap.TopicSeries, time_ns: int, tolerance_ns: int, *, previous: bool) -> Sample | None:
    if not previous:
        return mcap.nearest_value(series, time_ns, tolerance_ns)
    index = bisect_right(series.times_ns, time_ns) - 1
    if index < 0 or time_ns - series.times_ns[index] > tolerance_ns:
        return None
    return series.values[index]


def nearest_index(times: list[int], time_ns: int) -> int:
    index = bisect_left(times, time_ns)
    candidates = [i for i in (index, index - 1) if 0 <= i < len(times)]
    return min(candidates, key=lambda i: abs(times[i] - time_ns))


def valid_interval(valid: list[bool]) -> tuple[int, int]:
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("No synchronized valid frames")
    start, end = int(indices[0]), int(indices[-1]) + 1
    if not all(valid[start:end]):
        raise ValueError("Invalid interior frames; refusing to compress time across a gap")
    return start, end


def sample_error(sample: Sample | None, key: str, config: Config) -> str | None:
    if sample is None:
        return "missing topic/data or no sample within tolerance"
    if sample.error or sample.values is None:
        return sample.error or "invalid values"
    if sample.uncertainty_ns > config.max_clock_uncertainty_ms * 1e6:
        return "clock uncertainty exceeds tolerance"
    if key in {"hand_state", "hand_command"}:
        try:
            mcap.checked_hand(sample.values, config.hand_range_tolerance, feedback=key == "hand_state")
        except ValueError as exc:
            return str(exc)
    return None


def sample_record(sample: Sample, time_ns: int) -> dict:
    return {
        "message_id": sample.message_id,
        "ipc_time_ns": sample.time_ns,
        "skew_ms": (sample.time_ns - time_ns) / 1e6,
        "clock_uncertainty_ms": sample.uncertainty_ns / 1e6,
    }


def qpose(samples: dict[str, Sample], keys: tuple[str, ...], config: Config) -> tuple[np.ndarray, int]:
    hand, clipped = mcap.checked_hand(samples[keys[1]].values, config.hand_range_tolerance, feedback=keys == STATE_KEYS)
    values, names = mcap.make_qpose(samples[keys[0]].values, hand, samples[keys[2]].values, ARM_NAMES, HAND_NAMES)
    if names != JOINT_NAMES:
        raise ValueError("MCAP qpose contract changed")
    return values, clipped


def read_camera(episode: Path, frames: list[dict]) -> tuple[dict, tuple[int, int, int]]:
    camera = read_json(episode / "camera.json")
    if camera.get("rgb_format") != "rgb8" or camera.get("depth_format") != "z16":
        raise ValueError("Expected RGB8 color and Z16 depth")
    if camera.get("aligned_depth_intrinsics") != "color":
        raise ValueError("depth_aligned must use color intrinsics")
    scale = camera.get("depth_scale_m", 0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid depth_scale_m")
    with h5py.File(episode / "rgbd.h5", "r") as data:
        if json.loads(data.attrs["calibration_json"]) != camera:
            raise ValueError("HDF5 calibration does not match camera.json")
        height, width = camera["color"]["height"], camera["color"]["width"]
        for key, shape, dtype in (
            ("rgb", (len(frames), height, width, 3), np.uint8),
            ("depth_aligned", (len(frames), height, width), np.uint16),
        ):
            if data[key].shape != shape or data[key].dtype != dtype:
                raise ValueError(f"Unexpected HDF5 {key}: {data[key].shape}, {data[key].dtype}; expected {shape}")
    return camera, (height, width, 3)


def load_episode(episode: Path, config: Config, task: str | None = None) -> EpisodePlan:
    metadata = read_json(episode / "metadata.json")
    if metadata.get("format") != "egodex.ipc_collection.v1" or metadata.get("status") != "complete":
        raise ValueError("Expected a complete egodex.ipc_collection.v1 episode")
    if metadata.get("error"):
        raise ValueError("Collector reported an error")
    if metadata.get("quality_passed") is False:
        logging.warning("%s: collector quality_passed=False; validating each frame and feedback fallback", episode)
    task = (
        task.strip()
        if task is not None
        else mcap.read_task(episode, str(metadata.get("task_name", "")).replace("_", " "))
    )
    if not task:
        raise ValueError("Missing task text; supply --task")
    with (episode / "frames.jsonl").open(encoding="utf-8") as stream:
        frames = [json.loads(line) for line in stream if line.strip()]
    if not frames or [f["frame_index"] for f in frames] != list(range(len(frames))):
        raise ValueError("Frame indices must be contiguous, starting at zero")
    times = [f["ipc_time_ns"] for f in frames]
    if any(not isinstance(t, int) or t <= 0 for t in times) or any(b <= a for a, b in itertools.pairwise(times)):
        raise ValueError("RGB calibrated IPC timestamps must be positive and strictly increasing")
    if metadata.get("stored_frames") != len(frames):
        raise ValueError("Frame count differs from metadata.stored_frames")
    camera, rgb_shape = read_camera(episode, frames)
    series = read_series(episode, config)
    tolerance_ns = round(config.sync_tolerance_ms * 1e6)
    synced: list[tuple[np.ndarray, np.ndarray, dict] | None] = []
    rejected: list[dict] = []
    fallback_used = dict.fromkeys(mcap.STATE_COMPONENTS, 0)
    fallback_unavailable = dict.fromkeys(mcap.STATE_COMPONENTS, 0)
    for index, (frame, time_ns) in enumerate(zip(frames, times, strict=True)):
        samples = {}
        reason = None
        depth_time = frame["depth_ipc_time_ns"]
        if not isinstance(depth_time, int) or abs(depth_time - time_ns) > config.max_rgb_depth_skew_ms * 1e6:
            reason = "RGB/depth skew exceeds tolerance"
        for key in COMMAND_KEYS:
            sample = select_sample(series[key], time_ns, tolerance_ns, previous=config.command_alignment == "previous")
            error = sample_error(sample, key, config)
            if error:
                reason = f"{key}: {error}"
                break
            samples[key] = sample
        if reason is not None:
            rejected.append({"source_frame_index": index, "reason": reason})
            synced.append(None)
            continue
        action, _ = qpose(samples, COMMAND_KEYS, config)
        state_parts, state_messages = [], {}
        clipped = 0
        for component, feedback_key, command_key in zip(mcap.STATE_COMPONENTS, STATE_KEYS, COMMAND_KEYS, strict=True):
            key = feedback_key if config.state_source == "feedback" else command_key
            source = config.state_source
            sample = (
                select_sample(series[feedback_key], time_ns, tolerance_ns, previous=False)
                if source == "feedback"
                else samples[key]
            )
            error = sample_error(sample, key, config)
            feedback_error = error
            if error:
                key = command_key
                sample = select_sample(series[key], time_ns, tolerance_ns, previous=True)
                error = sample_error(sample, key, config)
                if error:
                    fallback_unavailable[component] += 1
                    reason = f"{feedback_key} unavailable; historical {command_key}: {error}"
                    continue
                fallback_used[component] += 1
                source = "historical_command"
            values = sample.values
            if component == "hand" and source == "feedback":
                values, clipped = mcap.checked_hand(values, config.hand_range_tolerance, feedback=True)
            state_parts.append(values)
            if source == "feedback":
                samples[key] = sample
            state_messages[component] = {
                "source": source,
                "key": key,
                "feedback_unavailable_reason": feedback_error,
                **sample_record(sample, time_ns),
            }
        if reason is not None:
            rejected.append({"source_frame_index": index, "reason": reason})
            synced.append(None)
            continue
        state, _ = mcap.make_qpose(*state_parts, ARM_NAMES, HAND_NAMES)
        mapping = {
            "source_frame_index": index,
            "rgb_ipc_time_ns": time_ns,
            "depth_ipc_time_ns": depth_time,
            "feedback_hand_clipped_values": clipped,
            "state_messages": state_messages,
            "messages": {key: sample_record(s, time_ns) for key, s in samples.items()},
        }
        synced.append((state, action, mapping))
    mcap.warn_feedback_fallback(episode, fallback_used, fallback_unavailable)
    start, end = valid_interval([s is not None for s in synced])
    # Keep real elapsed time on a regular LeRobot grid. Repeated source images are
    # recorded explicitly; an internal camera gap is never silently squeezed out.
    count = (times[end - 1] - times[start]) * config.fps // 1_000_000_000 + 1
    states, actions, mappings = [], [], []
    valid_times = times[start:end]
    for grid_index in range(count):
        target_ns = times[start] + round(grid_index * 1_000_000_000 / config.fps)
        source_index = start + nearest_index(valid_times, target_ns)
        if abs(times[source_index] - target_ns) > config.image_tolerance_ms * 1e6:
            raise ValueError(f"Camera gap near source frame {source_index}; no image within grid tolerance")
        state, action, mapping = synced[source_index]
        states.append(state)
        actions.append(action)
        mappings.append({**mapping, "frame_index": grid_index, "grid_ipc_time_ns": target_ns})
    if config.max_frames is not None:
        states, actions, mappings = (
            states[: config.max_frames],
            actions[: config.max_frames],
            mappings[: config.max_frames],
        )
    source_shape = rgb_shape
    if config.image_size is not None:
        width, height = config.image_size
        rgb_shape = (height, width, 3)
    report = {
        "source": str(episode),
        "task": task,
        "collector_quality_passed": metadata.get("quality_passed"),
        "source_frames": len(frames),
        "valid_source_frames": end - start,
        "output_frames": len(mappings),
        "trim_start": start,
        "trim_end_exclusive": end,
        "rejected_frames": rejected,
        "source_rgb_shape": source_shape,
        "output_rgb_shape": rgb_shape,
        "repeated_source_frames": len(mappings) - len({m["source_frame_index"] for m in mappings}),
        "feedback_hand_clipped_values": sum(m["feedback_hand_clipped_values"] for m in mappings),
        "state_fallback_counts": {
            c: sum(m["state_messages"][c]["source"] == "historical_command" for m in mappings)
            for c in mcap.STATE_COMPONENTS
        },
        "state_fallback_unavailable_counts": fallback_unavailable,
        "max_abs_skew_ms": {
            key: max((abs(m["messages"][key]["skew_ms"]) for m in mappings if key in m["messages"]), default=None)
            for key in series
        },
        "max_state_abs_skew_ms": {
            c: max(abs(m["state_messages"][c]["skew_ms"]) for m in mappings) for c in mcap.STATE_COMPONENTS
        },
        "camera": camera,
    }
    return EpisodePlan(episode, task, camera, rgb_shape, states, actions, mappings, report)


def read_images(
    data: h5py.File, source_index: int, camera: dict, size: tuple[int, int] | None
) -> tuple[Image.Image, Image.Image]:
    rgb = Image.fromarray(data["rgb"][source_index])
    depth = data["depth_aligned"][source_index]
    factor = float(camera["depth_scale_m"]) * 1000
    if not np.isclose(factor, 1.0, rtol=0, atol=1e-6):
        depth = np.clip(np.rint(depth.astype(np.float64) * factor), 0, 65535).astype(np.uint16)
    depth_image = Image.fromarray(depth)
    if size is not None:
        rgb = rgb.resize(size, Image.Resampling.BILINEAR)
        depth_image = depth_image.resize(size, Image.Resampling.NEAREST)
    return rgb, depth_image


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert(
    raw_dir: Path,
    repo_id: str | None = None,
    lerobot_root: Path | None = None,
    *,
    config: Config | None = None,
    task: str | None = None,
    max_episodes: int | None = None,
    dry_run: bool = False,
    report_path: Path | None = None,
    overwrite: bool = False,
) -> dict:
    config = config or Config()
    raw_dir = raw_dir.resolve()
    episodes = find_episode_dirs(raw_dir)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        episodes = episodes[:max_episodes]
    if not episodes:
        raise FileNotFoundError(f"No IPC episodes under {raw_dir}")
    dataset_root = None
    if not dry_run:
        if not repo_id or len(Path(repo_id).parts) != 2 or Path(repo_id).is_absolute() or ".." in Path(repo_id).parts:
            raise ValueError("--repo-id must be namespace/name")
        output_parent = (lerobot_root or mcap.HF_LEROBOT_HOME).resolve()
        dataset_root = (output_parent / repo_id).resolve()
        if (
            not dataset_root.is_relative_to(output_parent)
            or raw_dir.is_relative_to(dataset_root)
            or dataset_root.is_relative_to(raw_dir)
        ):
            raise ValueError("Output must be separate from the input tree and within lerobot-root")
        if dataset_root.exists() and not overwrite:
            raise FileExistsError(f"{dataset_root} already exists; pass --overwrite to replace it")
    plans, skipped = [], []
    for episode in episodes:
        try:
            plan = load_episode(episode, config, task)
            if plans and plan.rgb_shape != plans[0].rgb_shape:
                raise ValueError("Image size differs across episodes; supply --image-size WIDTH HEIGHT")
        except (ValueError, KeyError, OSError, sqlite3.Error) as exc:
            skipped.append({"source": str(episode), "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[skip] {episode}: {exc}", flush=True)
            continue
        plans.append(plan)
        print(f"[checked] {episode.parent.name}/{episode.name}: {len(plan.mapping)} frames", flush=True)
    report = {
        "source_format": "egodex.ipc_collection.v1",
        "config": dataclasses.asdict(config),
        "state_semantics": "command_target"
        if config.state_source == "command"
        else "feedback_with_historical_command_fallback",
        "action_semantics": "absolute_command_target",
        "joint_names": JOINT_NAMES,
        "units": {"arm": "radian", "hand": "encoder_count", "waist_rpy": "radian", "depth": "millimetre"},
        "camera_note": "Source camera mapped to chest_front feature keys; see calibration. Torso extrinsics are unknown.",
        "projection_trim": False,
        "output_mode": "image",
        "dry_run": dry_run,
        "kept_episodes": len(plans),
        "total_frames": sum(len(p.mapping) for p in plans),
        "episodes": [p.report for p in plans],
        "skipped_episodes": skipped,
    }
    if report_path is not None:
        write_json(report_path, report)
    if not plans:
        raise ValueError("No valid episodes; see reported skip reasons")
    if dry_run:
        print(
            f"Dry run: {len(plans)} episodes, {report['total_frames']} frames, {len(skipped)} skipped. No dataset written."
        )
        return report
    dataset = mcap.create_dataset(
        repo_id=repo_id,
        root=lerobot_root,
        fps=config.fps,
        mode="image",
        robot_type="a3_both",
        joint_names=JOINT_NAMES,
        rgb_shape=plans[0].rgb_shape,
        depth_shape=(*plans[0].rgb_shape[:2], 1),
        overwrite=overwrite,
        dataset_config=mcap.DatasetConfig(use_videos=False),
    )
    try:
        for episode_index, plan in enumerate(plans):
            with h5py.File(plan.source / "rgbd.h5", "r") as data:
                for state, action, mapping in zip(plan.states, plan.actions, plan.mapping, strict=True):
                    rgb, depth = read_images(data, mapping["source_frame_index"], plan.camera, config.image_size)
                    dataset.add_frame(
                        {
                            "observation.state": state,
                            "action": action,
                            mcap.CHEST_RGB_KEY: rgb,
                            mcap.CHEST_DEPTH_KEY: depth,
                            "task": plan.task,
                        }
                    )
            mcap.save_episode_keep_image_files(dataset)
            mapping_path = dataset.root / "meta" / "source_frames" / f"episode_{episode_index:06d}.jsonl"
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            with mapping_path.open("w", encoding="utf-8") as stream:
                for mapping in plan.mapping:
                    stream.write(json.dumps(mapping) + "\n")
            plan.report["episode_index"] = episode_index
            print(f"[saved {episode_index:03d}] {plan.source}: {len(plan.mapping)} frames", flush=True)
    finally:
        dataset.stop_image_writer()
    write_json(dataset.root / "meta" / "ipc_conversion.json", report)
    write_json(dataset.root / "meta" / "skipped_episodes.json", skipped)
    mcap.write_qpose_mean_json(
        dataset.root,
        joint_mode="both",
        selected=[s for p in plans for s in p.states],
        selected_names=JOINT_NAMES,
        excluded={},
        excluded_names={},
    )
    write_json(
        dataset.root / "meta" / "action_means.json",
        mcap.qpose_mean_record([a for p in plans for a in p.actions], JOINT_NAMES),
    )
    if report_path is not None:
        write_json(report_path, report)
    print(f"Done: {dataset.root}; {len(plans)} episodes, {report['total_frames']} frames, {len(skipped)} skipped.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", help="namespace/name; required unless --dry-run")
    parser.add_argument("--lerobot-root", type=Path)
    parser.add_argument("--task", help="Override all task text; otherwise prompt.txt, metadata.prompt, then task_name")
    parser.add_argument(
        "--state-source",
        choices=["feedback", "command"],
        default="feedback",
        help="Feedback state with warning and historical-command fallback if unavailable",
    )
    parser.add_argument("--command-alignment", choices=["nearest", "previous"], default="nearest")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sync-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--max-clock-uncertainty-ms", type=float, default=5.0)
    parser.add_argument("--image-tolerance-ms", type=float, default=20.0)
    parser.add_argument("--max-rgb-depth-skew-ms", type=float, default=10.0)
    parser.add_argument(
        "--hand-range-tolerance",
        type=float,
        default=1.0,
        help="Allowed encoder edge error; feedback is clipped and logged",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize RGB/depth; default keeps native size",
    )
    parser.add_argument("--max-frames-per-episode", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--mode",
        choices=["image"],
        default="image",
        help="Lossless PNG; uint16 depth must not use ordinary video encoding",
    )
    parser.add_argument("--joint-mode", choices=["both"], default="both")
    parser.add_argument(
        "--dry-run", action="store_true", help="Check all selected episodes and alignment without writing images"
    )
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = Config(
        fps=args.fps,
        state_source=args.state_source,
        command_alignment=args.command_alignment,
        sync_tolerance_ms=args.sync_tolerance_ms,
        max_clock_uncertainty_ms=args.max_clock_uncertainty_ms,
        image_tolerance_ms=args.image_tolerance_ms,
        max_rgb_depth_skew_ms=args.max_rgb_depth_skew_ms,
        hand_range_tolerance=args.hand_range_tolerance,
        image_size=tuple(args.image_size) if args.image_size else None,
        max_frames=args.max_frames_per_episode,
    )
    convert(
        args.raw_dir,
        args.repo_id,
        args.lerobot_root,
        config=config,
        task=args.task,
        max_episodes=args.max_episodes,
        dry_run=args.dry_run,
        report_path=args.report_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
