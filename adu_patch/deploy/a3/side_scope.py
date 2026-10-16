"""Side-scoped safety rules for A3 37D deployment poses and actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .core import MODEL_ACTION_DIM, Pose, validate_pose


def parse_active_sides(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated A3 side selector in deterministic order."""
    if value is None:
        return ("left", "right")
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not requested:
        raise ValueError("--active-sides cannot be empty")
    unknown = requested - {"left", "right"}
    if unknown:
        raise ValueError(f"unknown active sides: {sorted(unknown)}")
    return tuple(side for side in ("left", "right") if side in requested)


def _checked_state(values: Sequence[float], label: str) -> np.ndarray:
    state = np.asarray(values, dtype=np.float64)
    if state.shape != (MODEL_ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"{label} must be finite with shape ({MODEL_ACTION_DIM},), got {state.shape}")
    return state


@dataclass(frozen=True)
class A3SideScope:
    """Hold inactive arm and hand joints at the latest measured state."""

    active_sides: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = parse_active_sides(",".join(self.active_sides))
        object.__setattr__(self, "active_sides", normalized)

    @property
    def inactive_sides(self) -> tuple[str, ...]:
        return tuple(side for side in ("left", "right") if side not in self.active_sides)

    @property
    def is_scoped(self) -> bool:
        return bool(self.inactive_sides)

    def hold_state(self, target: Sequence[float], live_state: Sequence[float]) -> np.ndarray:
        """Replace inactive arm/hand dimensions in one 37D state."""
        target_array = _checked_state(target, "target state").copy()
        live_array = _checked_state(live_state, "live state")
        if "left" in self.inactive_sides:
            target_array[0:7] = live_array[0:7]
            target_array[14:24] = live_array[14:24]
        if "right" in self.inactive_sides:
            target_array[7:14] = live_array[7:14]
            target_array[24:34] = live_array[24:34]
        return target_array

    def hold_pose(self, target: Pose, live: Pose) -> Pose:
        """Return a pose whose inactive side is exactly the measured pose."""
        validate_pose(target)
        validate_pose(live)
        return Pose.from_state(self.hold_state(target.state(), live.state()))

    def hold_actions(self, actions: Any, live_state: Sequence[float]) -> np.ndarray:
        """Apply side holding to every action in a [T,37] chunk."""
        array = np.asarray(actions, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != MODEL_ACTION_DIM or not np.isfinite(array).all():
            raise ValueError(f"actions must be finite with shape [T,{MODEL_ACTION_DIM}], got {array.shape}")
        live = _checked_state(live_state, "live state")
        result = array.copy()
        for index in range(result.shape[0]):
            result[index] = self.hold_state(result[index], live)
        return result

    def validate_server_metadata(self, metadata: dict[str, Any]) -> None:
        """Require the policy server to apply the matching input/output scope."""
        if not self.is_scoped:
            return
        scope = metadata.get("a3_side_scope")
        if not isinstance(scope, dict):
            raise RuntimeError(
                "active-sides requires a scoped A3 policy server; "
                "restart serve_policy.py with --active-sides"
            )
        configured = tuple(scope.get("active_sides", ()))
        if configured != self.active_sides:
            raise RuntimeError(
                f"A3 side scope mismatch: ADU={self.active_sides} server={configured}"
            )
        if scope.get("inactive_input") != "checkpoint_state_mean":
            raise RuntimeError("policy server does not replace inactive inputs with checkpoint constants")
        if scope.get("inactive_output") != "live_request_state":
            raise RuntimeError("policy server does not hold inactive outputs at live request state")
