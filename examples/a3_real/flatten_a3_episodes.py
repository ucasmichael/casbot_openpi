#!/usr/bin/env python3
"""Flatten nested A3 collection/shard episodes into one numbered directory.

Example:
  python examples/a3_real/flatten_a3_episodes.py \
    --input-dir /data/lyx/dataset/data0813/data0813 \
    --output-dir /data/lyx/dataset/data0813/data0813_flat \

python examples/a3_real/flatten_a3_episodes.py \
  --input-dir /data/lyx/dataset/data0813/data0813 \
  --output-dir /data/lyx/dataset/data0813/data0813_flat \
  --method hardlink \
  --filter-zero-dims \
  --overwrite
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import shutil
from typing import Literal

import numpy as np


Method = Literal["copy", "hardlink", "symlink"]

ARM_TOPIC = "/motion/control/arm_joint_command"
HAND_TOPIC = "/motion/control/hand_joint_command"
ZERO_CHECK_TOPICS = (ARM_TOPIC, HAND_TOPIC)


@dataclasses.dataclass
class TopicZeroDimCheck:
    count: int = 0
    names: list[str] | None = None
    all_zero_dims: np.ndarray | None = None


@dataclasses.dataclass
class ZeroDimResult:
    skip: bool
    reasons: list[str]
    topic_counts: dict[str, int]


def episode_sort_key(input_dir: Path, path: Path) -> tuple[str, int, str]:
    suffix = path.name.removeprefix("episode_")
    try:
        episode_idx = int(suffix)
    except ValueError:
        episode_idx = 0
    return (str(path.parent.relative_to(input_dir)), episode_idx, path.name)


def find_episode_dirs(input_dir: Path) -> list[Path]:
    candidates = [path for path in input_dir.rglob("episode_*") if path.is_dir()]
    episodes = [
        path
        for path in candidates
        if (path / "metadata.json").exists()
        or (path / "prompt.txt").exists()
        or any(path.rglob("*.mcap"))
    ]
    return sorted(episodes, key=lambda path: episode_sort_key(input_dir, path))


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not overwrite:
        raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def copy_episode(src: Path, dst: Path, method: Method) -> None:
    if method == "copy":
        shutil.copytree(src, dst, copy_function=shutil.copy2)
    elif method == "hardlink":
        shutil.copytree(src, dst, copy_function=link_or_copy)
    elif method == "symlink":
        shutil.copytree(src, dst, symlinks=True, copy_function=symlink_file)
    else:
        raise ValueError(f"Unsupported method: {method}")


def link_or_copy(src: str, dst: str) -> None:
    try:
        Path(dst).hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def symlink_file(src: str, dst: str) -> None:
    Path(dst).symlink_to(Path(src).resolve())


def check_all_zero_dims(episode_dir: Path, *, zero_atol: float) -> ZeroDimResult:
    """Skip when any arm/hand joint dimension is zero for the whole episode."""
    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import Decoder
    except ImportError as exc:
        reason = f"zero-dim check unavailable: {type(exc).__name__}: {exc}"
        return ZeroDimResult(skip=False, reasons=[reason], topic_counts={})

    mcap_files = sorted(episode_dir.glob("**/*.mcap"))
    if not mcap_files:
        return ZeroDimResult(skip=False, reasons=[], topic_counts={})

    decoder = Decoder()
    topics = {topic: TopicZeroDimCheck() for topic in ZERO_CHECK_TOPICS}
    warnings: list[str] = []

    for mcap_file in mcap_files:
        try:
            with mcap_file.open("rb") as f:
                reader = make_reader(f)
                for schema, channel, message in reader.iter_messages():
                    topic = channel.topic
                    if topic not in topics:
                        continue

                    ros_msg = decoder.decode(schema, message)
                    names = list(getattr(ros_msg, "name", []))
                    positions = np.asarray(getattr(ros_msg, "position"), dtype=np.float32)
                    state = topics[topic]
                    state.count += 1
                    if state.all_zero_dims is None:
                        state.names = names
                        state.all_zero_dims = np.ones(positions.shape[0], dtype=bool)
                    elif state.all_zero_dims.shape[0] != positions.shape[0]:
                        warnings.append(
                            f"could not finish zero-dim check for {topic}: "
                            f"position dim changed from {state.all_zero_dims.shape[0]} to {positions.shape[0]}"
                        )
                        break
                    state.all_zero_dims &= np.isclose(positions, 0.0, atol=zero_atol)
        except Exception as exc:
            warnings.append(f"could not finish zero-dim check for {mcap_file.name}: {type(exc).__name__}: {exc}")
            break

    reasons: list[str] = []
    for topic, state in topics.items():
        if state.count == 0 or state.all_zero_dims is None:
            continue
        zero_indices = np.flatnonzero(state.all_zero_dims)
        if len(zero_indices) == 0:
            continue
        names = state.names or []
        zero_names = [names[idx] if idx < len(names) else f"dim_{idx}" for idx in zero_indices]
        reasons.append(f"all-zero joint dimensions in {topic}: {', '.join(zero_names)}")

    return ZeroDimResult(
        skip=bool(reasons),
        reasons=[*reasons, *warnings],
        topic_counts={topic: state.count for topic, state in topics.items() if state.count > 0},
    )


def flatten(
    input_dir: Path,
    output_dir: Path,
    *,
    method: Method,
    overwrite: bool,
    filter_zero_dims: bool,
    filter_zero_topics: bool,
    filter_invalid: bool,
    zero_atol: float,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if input_dir == output_dir or output_dir in input_dir.parents:
        raise ValueError("output-dir must not be the input-dir or one of its parents")

    episode_dirs = find_episode_dirs(input_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* directories found under {input_dir}")

    width = max(3, len(str(len(episode_dirs))))
    ensure_clean_output(output_dir, overwrite=overwrite)

    do_zero_filter = filter_zero_dims or filter_zero_topics or filter_invalid
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "method": method,
        "filter_zero_dims": do_zero_filter,
        "candidate_episode_count": len(episode_dirs),
        "episode_count": 0,
        "skipped_count": 0,
        "episodes": [],
        "skipped_episodes": [],
    }

    kept_idx = 0
    for src_idx, src in enumerate(episode_dirs, start=1):
        zero_check: ZeroDimResult | None = None
        if do_zero_filter:
            zero_check = check_all_zero_dims(src, zero_atol=zero_atol)
            if zero_check.skip:
                print(f"[skip {src_idx:0{width}d}/{len(episode_dirs)}] {src.relative_to(input_dir)}")
                for reason in zero_check.reasons:
                    print(f"  - {reason}")
                manifest["skipped_episodes"].append(
                    {
                        "source": str(src),
                        "source_relative": str(src.relative_to(input_dir)),
                        "reasons": zero_check.reasons,
                        "topic_counts": zero_check.topic_counts,
                    }
                )
                continue
            for reason in zero_check.reasons:
                print(f"[warn {src_idx:0{width}d}/{len(episode_dirs)}] {src.relative_to(input_dir)}: {reason}")

        kept_idx += 1
        dst_name = f"episode_{kept_idx:0{width}d}"
        dst = output_dir / dst_name
        print(f"[keep {src_idx:0{width}d}/{len(episode_dirs)}] {src.relative_to(input_dir)} -> {dst_name}")
        copy_episode(src, dst, method)
        record = {
            "episode": dst_name,
            "source": str(src),
            "source_relative": str(src.relative_to(input_dir)),
        }
        if zero_check is not None:
            record["topic_counts"] = zero_check.topic_counts
        manifest["episodes"].append(record)

    manifest["episode_count"] = kept_idx
    manifest["skipped_count"] = len(manifest["skipped_episodes"])

    manifest_path = output_dir / "episode_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nDone. Wrote {kept_idx} episodes to {output_dir}")
    if manifest["skipped_count"]:
        print(f"Skipped {manifest['skipped_count']} all-zero-dim episodes. Reasons are in the manifest.")
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Nested A3 dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Flat output directory.")
    parser.add_argument("--method", choices=["copy", "hardlink", "symlink"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true", help="Remove output-dir before writing.")
    parser.add_argument("--filter-zero-dims", action="store_true", help="Skip episodes where any arm/hand joint dimension is all zeros.")
    parser.add_argument("--filter-zero-topics", action="store_true", help="Alias for --filter-zero-dims kept for old commands.")
    parser.add_argument("--filter-invalid", action="store_true", help="Alias for --filter-zero-dims kept for old commands.")
    parser.add_argument("--zero-atol", type=float, default=0.0, help="Tolerance used for all-zero dimension checks.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    flatten(**vars(args))
