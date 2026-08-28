#!/usr/bin/env python3
"""Audit raw A3 MCAP files for topic gaps and all-zero topic payloads.

Example:
  .venv/bin/python examples/a3_real/audit_a3_mcap_quality.py \
    --raw-dir /data/lyx/dataset/data0819_flat \
    --out /tmp/a3_mcap_quality_report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from mcap_ros2.decoder import Decoder
import numpy as np


RGB_TOPIC = "/hal/chest_front_d457_camera/rgb"
DEPTH_TOPIC = "/hal/chest_front_d457_camera/depth"
ARM_COMMAND_TOPIC = "/motion/control/arm_joint_command"
HAND_COMMAND_TOPIC = "/motion/control/hand_joint_command"
ARM_STATE_TOPIC = "/motion/control/arm_joint_state"
HAND_STATE_TOPIC = "/motion/control/hand_joint_state"
NECK_STATE_TOPIC = "/motion/control/neck_joint_state"
WAIST_MOVE_TOPIC = (
    "/motion/control/move_waist/"
    "pb_3Aaimdk_2Eprotocol_2EMotionControlMoveWaistChannel"
)

DEFAULT_EXPECTED_TOPICS = (
    RGB_TOPIC,
    DEPTH_TOPIC,
    ARM_COMMAND_TOPIC,
    HAND_COMMAND_TOPIC,
    ARM_STATE_TOPIC,
    HAND_STATE_TOPIC,
    NECK_STATE_TOPIC,
    WAIST_MOVE_TOPIC,
)


@dataclasses.dataclass
class TopicAudit:
    count: int = 0
    schema_name: str | None = None
    first_time_ns: int | None = None
    last_time_ns: int | None = None
    times_ns: list[int] = dataclasses.field(default_factory=list)
    zero_check_count: int = 0
    nonzero_count: int = 0
    zero_check_error: str | None = None


def stamp_to_ns(stamp: Any) -> int | None:
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)


def message_time_ns(ros_msg: Any | None, mcap_message: Any) -> int:
    if ros_msg is not None:
        header = getattr(ros_msg, "header", None)
        stamp = getattr(header, "stamp", None)
        stamp_ns = stamp_to_ns(stamp) if stamp is not None else None
        if stamp_ns is not None:
            return stamp_ns
    return int(mcap_message.log_time)


def find_episode_dirs(raw_dir: Path) -> list[Path]:
    raw_dir = raw_dir.resolve()
    if list(raw_dir.glob("*.mcap")):
        return [raw_dir]
    episode_dirs = [path for path in sorted(raw_dir.glob("episode_*")) if path.is_dir()]
    if episode_dirs:
        return episode_dirs
    return sorted({path.parent for path in raw_dir.glob("**/*.mcap")})


def source_relative_to_raw(raw_dir: Path, episode_dir: Path) -> str:
    try:
        return str(episode_dir.resolve().relative_to(raw_dir.resolve()))
    except ValueError:
        return str(episode_dir)


def numeric_array_from_value(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return np.asarray([int(value)], dtype=np.float64)
    if isinstance(value, int | float):
        return np.asarray([value], dtype=np.float64)
    if isinstance(value, bytes | bytearray | memoryview):
        return np.frombuffer(bytes(value), dtype=np.uint8)
    if isinstance(value, str):
        return None
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"b", "i", "u", "f"}:
            return value.reshape(-1)
        return None
    if isinstance(value, list | tuple):
        if not value:
            return np.asarray([], dtype=np.float64)
        if all(isinstance(item, bool) for item in value):
            return np.asarray(value, dtype=np.bool_).reshape(-1)
        if all(isinstance(item, int | float) and not isinstance(item, bool) for item in value):
            return np.asarray(value, dtype=np.float64).reshape(-1)
        if all(isinstance(item, int) for item in value):
            return np.asarray(value, dtype=np.int64).reshape(-1)
    return None


def iter_public_fields(msg: Any) -> list[str]:
    slots = getattr(msg, "__slots__", None)
    if slots:
        return [field for field in slots if not field.startswith("_")]
    return [field for field in vars(msg) if not field.startswith("_")]


def numeric_payload_arrays(msg: Any, *, topic: str) -> list[np.ndarray]:
    """Return numeric payload arrays, excluding stamps and shape/metadata fields."""
    if hasattr(msg, "position"):
        arrays = []
        for field in ("position", "velocity", "effort"):
            array = numeric_array_from_value(getattr(msg, field, None))
            if array is not None and array.size:
                arrays.append(array)
        return arrays

    if hasattr(msg, "data"):
        array = numeric_array_from_value(getattr(msg, "data", None))
        return [array] if array is not None and array.size else []

    ignored_fields = {
        "header",
        "stamp",
        "frame_id",
        "name",
        "encoding",
        "height",
        "width",
        "step",
        "is_bigendian",
        "layout",
    }
    arrays = []
    for field in iter_public_fields(msg):
        if field in ignored_fields:
            continue
        array = numeric_array_from_value(getattr(msg, field, None))
        if array is not None and array.size:
            arrays.append(array)
    return arrays


def message_payload_is_zero(msg: Any, *, topic: str, zero_atol: float) -> bool | None:
    arrays = numeric_payload_arrays(msg, topic=topic)
    if not arrays:
        return None
    for array in arrays:
        if array.dtype.kind in {"f"}:
            if not np.all(np.isclose(array, 0.0, atol=zero_atol, equal_nan=False)):
                return False
        elif np.any(array != 0):
            return False
    return True


def summarize_dt_ms(times_ns: list[int]) -> dict[str, float | int | None]:
    if len(times_ns) < 2:
        return {"pair_count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    dt_ms = np.diff(np.asarray(sorted(times_ns), dtype=np.int64)) / 1e6
    return {
        "pair_count": int(dt_ms.size),
        "mean": float(dt_ms.mean()),
        "p50": float(np.percentile(dt_ms, 50)),
        "p95": float(np.percentile(dt_ms, 95)),
        "p99": float(np.percentile(dt_ms, 99)),
        "max": float(dt_ms.max()),
    }


def gap_records(topic: str, times_ns: list[int], *, threshold_ms: float, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(times_ns)
    if len(ordered) < 2:
        return []
    dt_ms = np.diff(np.asarray(ordered, dtype=np.int64)) / 1e6
    indices = np.flatnonzero(dt_ms > threshold_ms)
    records = [
        {
            "topic": topic,
            "pair": [int(idx), int(idx + 1)],
            "gap_ms": float(dt_ms[idx]),
            "start_time_ns": int(ordered[idx]),
            "end_time_ns": int(ordered[idx + 1]),
        }
        for idx in indices
    ]
    return sorted(records, key=lambda record: record["gap_ms"], reverse=True)[:limit]


def audit_episode(
    raw_dir: Path,
    episode_dir: Path,
    *,
    expected_topics: tuple[str, ...],
    gap_threshold_ms: float,
    max_gaps_per_topic: int,
    zero_atol: float,
) -> dict[str, Any]:
    decoder = Decoder()
    mcap_files = sorted(episode_dir.glob("**/*.mcap"))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found under {episode_dir}")

    topics: dict[str, TopicAudit] = {}
    decode_errors: list[dict[str, str]] = []

    for mcap_file in mcap_files:
        with mcap_file.open("rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                state = topics.setdefault(topic, TopicAudit())
                state.count += 1
                state.schema_name = schema.name

                ros_msg = None
                try:
                    ros_msg = decoder.decode(schema, message)
                except Exception as exc:  # noqa: BLE001
                    decode_errors.append(
                        {
                            "mcap_file": str(mcap_file),
                            "topic": topic,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

                time_ns = message_time_ns(ros_msg, message)
                state.times_ns.append(time_ns)
                if state.first_time_ns is None or time_ns < state.first_time_ns:
                    state.first_time_ns = time_ns
                if state.last_time_ns is None or time_ns > state.last_time_ns:
                    state.last_time_ns = time_ns

                if ros_msg is None or state.zero_check_error is not None or state.nonzero_count > 0:
                    continue
                try:
                    is_zero = message_payload_is_zero(ros_msg, topic=topic, zero_atol=zero_atol)
                except Exception as exc:  # noqa: BLE001
                    state.zero_check_error = f"{type(exc).__name__}: {exc}"
                    continue
                if is_zero is None:
                    continue
                state.zero_check_count += 1
                if not is_zero:
                    state.nonzero_count += 1

    count_zero_topics = [topic for topic in expected_topics if topics.get(topic, TopicAudit()).count == 0]
    all_zero_topics = [
        topic
        for topic, state in sorted(topics.items())
        if state.count > 0 and state.zero_check_count > 0 and state.nonzero_count == 0
    ]

    topic_records = {}
    gap_topics = {}
    for topic, state in sorted(topics.items()):
        gaps = gap_records(
            topic,
            state.times_ns,
            threshold_ms=gap_threshold_ms,
            limit=max_gaps_per_topic,
        )
        dt_summary = summarize_dt_ms(state.times_ns)
        record = {
            "count": state.count,
            "schema_name": state.schema_name,
            "first_time_ns": state.first_time_ns,
            "last_time_ns": state.last_time_ns,
            "dt_ms": dt_summary,
            "gap_count_gt_threshold": sum(
                1
                for gap_ms in (
                    np.diff(np.asarray(sorted(state.times_ns), dtype=np.int64)) / 1e6
                    if len(state.times_ns) >= 2
                    else []
                )
                if gap_ms > gap_threshold_ms
            ),
            "top_gaps": gaps,
            "zero_check_count": state.zero_check_count,
            "nonzero_count": state.nonzero_count,
            "all_zero_payload": state.zero_check_count > 0 and state.nonzero_count == 0,
            "zero_check_error": state.zero_check_error,
        }
        topic_records[topic] = record
        if record["gap_count_gt_threshold"]:
            gap_topics[topic] = {
                "gap_count": record["gap_count_gt_threshold"],
                "max_gap_ms": record["dt_ms"]["max"],
                "top_gaps": gaps,
            }

    return {
        "source": str(episode_dir),
        "source_relative": source_relative_to_raw(raw_dir, episode_dir),
        "mcap_files": [str(path) for path in mcap_files],
        "topic_count": len(topics),
        "message_count": sum(state.count for state in topics.values()),
        "count_zero_topics": count_zero_topics,
        "all_zero_topics": all_zero_topics,
        "gap_topics": gap_topics,
        "decode_errors": decode_errors[:50],
        "decode_error_count": len(decode_errors),
        "topics": topic_records,
    }


def print_episode_summary(index: int, total: int, episode_record: dict[str, Any], *, gap_threshold_ms: float) -> None:
    rel = episode_record["source_relative"]
    gap_topics = episode_record["gap_topics"]
    count_zero_topics = episode_record["count_zero_topics"]
    all_zero_topics = episode_record["all_zero_topics"]
    print(
        f"[{index:03d}/{total:03d}] {rel}: "
        f"topics={episode_record['topic_count']} messages={episode_record['message_count']} "
        f"gap_topics>{gap_threshold_ms:g}ms={len(gap_topics)} "
        f"count0={len(count_zero_topics)} all_zero={len(all_zero_topics)}",
        flush=True,
    )
    for topic, gap_info in sorted(gap_topics.items(), key=lambda item: item[1]["max_gap_ms"], reverse=True)[:8]:
        print(f"  gap {gap_info['max_gap_ms']:.3f} ms x{gap_info['gap_count']}: {topic}", flush=True)
    if count_zero_topics:
        print("  count=0:", flush=True)
        for topic in count_zero_topics:
            print(f"    {topic}", flush=True)
    if all_zero_topics:
        print("  all-zero payload:", flush=True)
        for topic in all_zero_topics:
            print(f"    {topic}", flush=True)


def audit(
    raw_dir: Path,
    out: Path,
    *,
    expected_topic: list[str] | None,
    gap_threshold_ms: float,
    max_gaps_per_topic: int,
    zero_atol: float,
    fail_on_gap: bool,
) -> None:
    raw_dir = raw_dir.resolve()
    expected_topics = tuple(expected_topic) if expected_topic else DEFAULT_EXPECTED_TOPICS
    episode_dirs = find_episode_dirs(raw_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No .mcap files or episode dirs found under {raw_dir}")

    report: dict[str, Any] = {
        "raw_dir": str(raw_dir),
        "gap_threshold_ms": gap_threshold_ms,
        "zero_atol": zero_atol,
        "expected_topics": list(expected_topics),
        "episode_count": len(episode_dirs),
        "episodes": [],
        "errors": [],
    }

    width = max(3, len(str(len(episode_dirs))))
    for index, episode_dir in enumerate(episode_dirs, start=1):
        try:
            record = audit_episode(
                raw_dir,
                episode_dir,
                expected_topics=expected_topics,
                gap_threshold_ms=gap_threshold_ms,
                max_gaps_per_topic=max_gaps_per_topic,
                zero_atol=zero_atol,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{index:0{width}d}/{len(episode_dirs):0{width}d}] {episode_dir}: {type(exc).__name__}: {exc}")
            report["errors"].append(
                {
                    "source": str(episode_dir),
                    "source_relative": source_relative_to_raw(raw_dir, episode_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        print_episode_summary(index, len(episode_dirs), record, gap_threshold_ms=gap_threshold_ms)
        report["episodes"].append(record)

    report["episodes_with_gap_count"] = sum(bool(ep["gap_topics"]) for ep in report["episodes"])
    report["episodes_with_count_zero_topics"] = sum(bool(ep["count_zero_topics"]) for ep in report["episodes"])
    report["episodes_with_all_zero_topics"] = sum(bool(ep["all_zero_topics"]) for ep in report["episodes"])
    report["error_count"] = len(report["errors"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nDone.")
    print(f"Episodes audited: {len(report['episodes'])}/{len(episode_dirs)}")
    print(f"Episodes with any topic gap > {gap_threshold_ms:g} ms: {report['episodes_with_gap_count']}")
    print(f"Episodes with expected topic count=0: {report['episodes_with_count_zero_topics']}")
    print(f"Episodes with all-zero payload topics: {report['episodes_with_all_zero_topics']}")
    if report["errors"]:
        print(f"Errors: {len(report['errors'])}")
    print(f"Report: {out}")

    if fail_on_gap and report["episodes_with_gap_count"]:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Raw A3 directory, an episode dir, or a dir with MCAP files.")
    parser.add_argument("--out", type=Path, default=Path("/tmp/a3_mcap_quality_report.json"), help="JSON report path.")
    parser.add_argument("--expected-topic", action="append", default=None, help="Expected topic. May be passed multiple times; defaults to common A3 topics.")
    parser.add_argument("--gap-threshold-ms", type=float, default=500.0, help="Flag adjacent messages in a topic when dt exceeds this threshold.")
    parser.add_argument("--max-gaps-per-topic", type=int, default=10, help="Number of largest gaps to keep per topic in the report.")
    parser.add_argument("--zero-atol", type=float, default=0.0, help="Tolerance for float all-zero payload checks.")
    parser.add_argument("--fail-on-gap", action="store_true", help="Exit with code 2 if any topic has a gap over the threshold.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audit(**vars(args))
