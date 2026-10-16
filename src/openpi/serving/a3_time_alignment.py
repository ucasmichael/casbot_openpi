"""Source-time alignment for A3 state and an external RealSense RGB stream."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import math
import time
from typing import Any

import numpy as np

PROTOCOL = "openpi.a3.external-camera.v1"
SOURCE_WINDOW_SCHEMA = "openpi.a3.source-state-window.v1"
CAMERA_SOURCE_ADU_D457 = "adu-d457"
CAMERA_SOURCE_EXTERNAL_D455 = "external-d455"
CAMERA_SOURCES = (CAMERA_SOURCE_ADU_D457, CAMERA_SOURCE_EXTERNAL_D455)

MAX_CLOCK_UNCERTAINTY_NS = 5_000_000
MAX_CLOCK_AGE_NS = 8_000_000_000
MAX_ALIGNMENT_ERROR_NS = 20_000_000
MAX_SOURCE_AGE_NS = 500_000_000
DRIFT_ALLOWANCE = 0.0001  # 100 ppm.


class SourceAlignmentError(RuntimeError):
    """Raised when an RGB/state pair cannot be proven sufficiently aligned."""


@dataclasses.dataclass(frozen=True)
class ClockSync:
    server_minus_actor_monotonic_ns: int
    monotonic_uncertainty_ns: int
    probe_count: int
    received_monotonic_ns: int
    sample_monotonic_ns: int


@dataclasses.dataclass(frozen=True)
class AlignedObservation:
    frame: Any
    state: np.ndarray
    metadata: dict[str, Any]


def _fail(message: str) -> None:
    raise SourceAlignmentError(message)


def _frame_source_monotonic_ns(frame: Any) -> int:
    if frame.timestamp_domain != "global_time":
        _fail(f"RealSense RGB timestamp domain must be global_time, got {frame.timestamp_domain!r}")
    timestamp_ms = float(frame.device_timestamp_ms)
    if not math.isfinite(timestamp_ms) or timestamp_ms <= 0:
        _fail("RealSense RGB source timestamp is invalid")
    wall_minus_monotonic = int(frame.host_wall_ns) - int(frame.host_monotonic_ns)
    source_monotonic_ns = round(timestamp_ms * 1_000_000.0) - wall_minus_monotonic
    delivery_ns = int(frame.host_monotonic_ns) - source_monotonic_ns
    if not 0 <= delivery_ns <= MAX_SOURCE_AGE_NS:
        _fail(
            f"RealSense global_time is inconsistent with the server clock: delivery_ms={delivery_ns / 1_000_000.0:.3f}"
        )
    return source_monotonic_ns


def _parse_window(
    window: Mapping[str, Any],
    clock: ClockSync,
    *,
    now_ns: int,
) -> tuple[dict[str, list[Mapping[str, Any]]], int, int]:
    if not isinstance(window, Mapping) or window.get("schema") != SOURCE_WINDOW_SCHEMA:
        _fail("ADU source-state window is missing or has an unsupported schema")
    errors = window.get("errors")
    if errors:
        _fail(f"ADU source timestamp error: {errors}")
    if clock.probe_count < 3:
        _fail("clock synchronization requires at least three probes")
    age_ns = now_ns - clock.sample_monotonic_ns
    if not 0 <= age_ns <= MAX_CLOCK_AGE_NS:
        _fail(f"clock calibration age is invalid: {age_ns / 1_000_000.0:.3f} ms")
    uncertainty_ns = clock.monotonic_uncertainty_ns + int(age_ns * DRIFT_ALLOWANCE)
    if not 0 <= uncertainty_ns <= MAX_CLOCK_UNCERTAINTY_NS:
        _fail(f"clock uncertainty exceeds 5 ms: {uncertainty_ns / 1_000_000.0:.3f} ms")

    raw_groups = window.get("groups")
    if not isinstance(raw_groups, Mapping):
        _fail("ADU source-state groups are missing")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for name, size in (("arm", 14), ("hand", 20), ("waist", 3)):
        rows = raw_groups.get(name)
        if not isinstance(rows, list) or not 2 <= len(rows) <= 64:
            _fail(f"{name}: expected 2..64 source samples")
        previous_source_ns: int | None = None
        for row in rows:
            if not isinstance(row, Mapping):
                _fail(f"{name}: source sample must be a mapping")
            source_ns = int(row.get("source_monotonic_ns", -1))
            header_ns = int(row.get("header_timestamp_ns", -1))
            receive_ns = int(row.get("receive_monotonic_ns", -1))
            position = row.get("position")
            if (
                not isinstance(position, Sequence)
                or isinstance(position, str | bytes)
                or len(position) != size
                or not all(math.isfinite(float(value)) for value in position)
            ):
                _fail(f"{name}: invalid state shape or non-finite values")
            if name == "hand" and any(not 0.0 <= float(value) <= 4095.0 for value in position):
                _fail("hand source state is outside encoder range")
            if header_ns <= 0 or receive_ns <= 0:
                _fail(f"{name}: timestamps must be positive")
            if previous_source_ns is not None and source_ns <= previous_source_ns:
                _fail(f"{name}: source timestamps are not strictly increasing")
            previous_source_ns = source_ns
        latest_server_ns = int(rows[-1]["source_monotonic_ns"]) + clock.server_minus_actor_monotonic_ns
        latest_age_ns = now_ns - latest_server_ns
        if not -20_000_000 <= latest_age_ns <= MAX_SOURCE_AGE_NS:
            _fail(f"{name}: latest source state age is {latest_age_ns / 1_000_000.0:.3f} ms")
        groups[name] = rows
    return groups, clock.server_minus_actor_monotonic_ns, uncertainty_ns


def select_aligned_frame(
    frames: Sequence[Any],
    window: Mapping[str, Any],
    clock: ClockSync,
    *,
    now_ns: int | None = None,
) -> AlignedObservation:
    """Select the newest RGB frame bracketed by all three real state streams."""
    current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
    groups, offset_ns, uncertainty_ns = _parse_window(window, clock, now_ns=current_ns)

    for frame in reversed(frames):
        frame_age_ns = current_ns - int(frame.host_monotonic_ns)
        if not 0 <= frame_age_ns <= MAX_SOURCE_AGE_NS:
            continue
        rgb_ns = _frame_source_monotonic_ns(frame)
        actor_target_ns = rgb_ns - offset_ns
        selected: dict[str, Mapping[str, Any]] = {}
        diagnostics: dict[str, dict[str, float | int]] = {}
        for name, rows in groups.items():
            first_ns = int(rows[0]["source_monotonic_ns"])
            last_ns = int(rows[-1]["source_monotonic_ns"])
            if not first_ns <= actor_target_ns <= last_ns:
                break
            row = min(rows, key=lambda item: abs(int(item["source_monotonic_ns"]) - actor_target_ns))
            skew_ns = int(row["source_monotonic_ns"]) - actor_target_ns
            if abs(skew_ns) + uncertainty_ns > MAX_ALIGNMENT_ERROR_NS:
                break
            selected[name] = row
            diagnostics[name] = {
                "header_timestamp_ns": int(row["header_timestamp_ns"]),
                "source_monotonic_ns": int(row["source_monotonic_ns"]),
                "server_source_monotonic_ns": int(row["source_monotonic_ns"]) + offset_ns,
                "state_minus_rgb_ms": skew_ns / 1_000_000.0,
                "source_to_receive_ms": (int(row["receive_monotonic_ns"]) - int(row["source_monotonic_ns"]))
                / 1_000_000.0,
                "clock_uncertainty_ms": uncertainty_ns / 1_000_000.0,
            }
        if len(selected) != 3:
            continue

        state = np.asarray(
            [
                *selected["arm"]["position"],
                *selected["hand"]["position"],
                *selected["waist"]["position"],
            ],
            dtype=np.float32,
        )
        if state.shape != (37,) or not np.isfinite(state).all():
            _fail(f"aligned A3 state must be finite [37], got {state.shape}")
        selected_server_times = [int(row["source_monotonic_ns"]) + offset_ns for row in selected.values()]
        metadata: dict[str, Any] = {
            "schema": PROTOCOL,
            "camera_source": CAMERA_SOURCE_EXTERNAL_D455,
            "time_alignment": "source",
            "rgb_frame_id": int(frame.frame_id),
            "rgb_source_monotonic_ns": rgb_ns,
            "device_rgb_timestamp_ms": float(frame.device_timestamp_ms),
            "rgb_timestamp_domain": frame.timestamp_domain,
            "host_frame_monotonic_ns": int(frame.host_monotonic_ns),
            "frame_age_ms": frame_age_ns / 1_000_000.0,
            "camera_delivery_ms": (int(frame.host_monotonic_ns) - rgb_ns) / 1_000_000.0,
            "clock_server_minus_actor_monotonic_ns": offset_ns,
            "clock_monotonic_uncertainty_ms": uncertainty_ns / 1_000_000.0,
            "clock_calibration_age_ms": (current_ns - clock.sample_monotonic_ns) / 1_000_000.0,
            "component_skew_ms": (max(selected_server_times) - min(selected_server_times)) / 1_000_000.0,
            "source_alignment": diagnostics,
            "source_max_abs_skew_ms": max(abs(value["state_minus_rgb_ms"]) for value in diagnostics.values()),
            "source_max_error_bound_ms": max(
                abs(value["state_minus_rgb_ms"]) + uncertainty_ns / 1_000_000.0 for value in diagnostics.values()
            ),
        }
        return AlignedObservation(frame=frame, state=state, metadata=metadata)

    _fail("no recent D455 RGB frame is bracketed by arm/hand/waist samples within the 20 ms error bound")
    raise AssertionError("unreachable")
