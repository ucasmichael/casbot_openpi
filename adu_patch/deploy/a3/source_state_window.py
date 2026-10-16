"""Bounded source-stamped A3 state history used for external-camera alignment."""

from __future__ import annotations

from collections import deque
import time
from typing import Any

SCHEMA = "openpi.a3.source-state-window.v1"


class SourceStateWindow:
    def __init__(self) -> None:
        self.groups: dict[str, deque[dict[str, Any]]] = {name: deque(maxlen=64) for name in ("arm", "hand", "waist")}
        self.errors: dict[str, str] = {}
        self.wall_offset_ns: int | None = None

    def add(
        self,
        group: str,
        message: Any,
        position: Any,
        *,
        monotonic_ns: int | None = None,
        wall_ns: int | None = None,
    ) -> None:
        if group not in self.groups:
            raise ValueError(f"unknown A3 state group {group!r}")
        receive_monotonic_ns = time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns)
        receive_wall_ns = time.time_ns() if wall_ns is None else int(wall_ns)
        wall_offset_ns = receive_wall_ns - receive_monotonic_ns
        if self.wall_offset_ns is not None and abs(wall_offset_ns - self.wall_offset_ns) > 5_000_000:
            self.errors["clock"] = "ADU wall clock stepped by more than 5 ms; restart the executor"
        self.wall_offset_ns = wall_offset_ns

        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        stamp_age_ns = receive_wall_ns - stamp_ns
        if stamp_ns <= 0 or not -20_000_000 <= stamp_age_ns <= 500_000_000:
            self.errors[group] = (
                f"missing, future, or stale ROS header timestamp: age_ms={stamp_age_ns / 1_000_000.0:.3f}"
            )
            self.groups[group].clear()
            return
        source_monotonic_ns = receive_monotonic_ns + stamp_ns - receive_wall_ns
        history = self.groups[group]
        if history and stamp_ns <= history[-1]["header_timestamp_ns"]:
            self.errors[group] = "ROS header timestamp did not advance"
            history.clear()
            return
        self.errors.pop(group, None)
        history.append(
            {
                "header_timestamp_ns": stamp_ns,
                "source_monotonic_ns": source_monotonic_ns,
                "receive_monotonic_ns": receive_monotonic_ns,
                "receive_wall_ns": receive_wall_ns,
                "position": [float(value) for value in position],
            }
        )
        while history and receive_monotonic_ns - history[0]["source_monotonic_ns"] > 300_000_000:
            history.popleft()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "snapshot_monotonic_ns": time.monotonic_ns(),
            "snapshot_wall_ns": time.time_ns(),
            "errors": dict(self.errors),
            "groups": {name: list(rows) for name, rows in self.groups.items()},
        }
