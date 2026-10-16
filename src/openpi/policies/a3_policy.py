import dataclasses
from typing import Any

import einops
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms
from openpi.models import model as _model

A3_ONE_SIDE_ACTION_DIM = 17
A3_BOTH_ACTION_DIM = 37
A3_ACTION_DIM = A3_BOTH_ACTION_DIM

A3_SIDE_INDICES = {
    "left": np.asarray([*range(7), *range(14, 24)], dtype=np.int64),
    "right": np.asarray([*range(7, 14), *range(24, 34)], dtype=np.int64),
}


def parse_active_sides(value: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated side selector in deterministic model order."""
    if value is None:
        return None
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    unknown = requested - A3_SIDE_INDICES.keys()
    if unknown:
        raise ValueError(f"unknown A3 active sides: {sorted(unknown)}")
    if not requested:
        raise ValueError("A3 active sides cannot be empty")
    return tuple(side for side in ("left", "right") if side in requested)


def validate_inactive_sides_frozen(
    active_sides: tuple[str, ...],
    norm_stats: dict[str, transforms.NormStats],
    *,
    use_quantiles: bool,
) -> tuple[str, ...]:
    """Refuse scoping unless every inactive input/output dimension was frozen in training."""
    inactive_sides = tuple(side for side in ("left", "right") if side not in active_sides)
    if not inactive_sides:
        return inactive_sides
    inactive_indices = np.concatenate([A3_SIDE_INDICES[side] for side in inactive_sides])
    for key in ("state", "actions"):
        stats = norm_stats.get(key)
        if stats is None:
            raise ValueError(f"A3 side scoping requires {key!r} normalization statistics")
        frozen = set(transforms.frozen_dim_indices(stats, use_quantiles=use_quantiles).tolist())
        not_frozen = sorted(set(inactive_indices.tolist()) - frozen)
        if not_frozen:
            raise ValueError(
                f"cannot deactivate A3 sides {list(inactive_sides)}: "
                f"checkpoint {key} dimensions are not frozen: {not_frozen}"
            )
    return inactive_sides


class A3ScopedPolicy(_base_policy.BasePolicy):
    """Mask inactive A3 inputs to training constants and hold live outputs."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        active_sides: tuple[str, ...],
        training_state: np.ndarray,
    ) -> None:
        unknown = set(active_sides) - A3_SIDE_INDICES.keys()
        if unknown or not active_sides:
            raise ValueError(f"invalid A3 active sides: {active_sides}")
        training_state = np.asarray(training_state, dtype=np.float32)
        if training_state.shape != (A3_ACTION_DIM,) or not np.isfinite(training_state).all():
            raise ValueError(f"A3 training state must be finite [{A3_ACTION_DIM}], got {training_state.shape}")

        self._policy = policy
        self._active_sides = tuple(side for side in ("left", "right") if side in active_sides)
        self._inactive_sides = tuple(side for side in ("left", "right") if side not in active_sides)
        self._training_state = training_state.copy()
        self._metadata = {
            **getattr(policy, "metadata", {}),
            "a3_side_scope": {
                "active_sides": list(self._active_sides),
                "inactive_sides": list(self._inactive_sides),
                "inactive_input": "checkpoint_state_mean",
                "inactive_output": "live_request_state",
            },
        }

    @override
    def infer(self, obs: dict, **kwargs: Any) -> dict:  # type: ignore[misc]
        requested_sides = obs.get("active_sides")
        if requested_sides is not None:
            requested = parse_active_sides(
                ",".join(requested_sides) if isinstance(requested_sides, list | tuple) else str(requested_sides)
            )
            if requested != self._active_sides:
                raise ValueError(f"A3 side scope mismatch: request={requested} server={self._active_sides}")
        live_state = np.asarray(obs.get("state"), dtype=np.float32)
        if live_state.shape != (A3_ACTION_DIM,) or not np.isfinite(live_state).all():
            raise ValueError(f"A3 live state must be finite [{A3_ACTION_DIM}], got {live_state.shape}")

        model_state = live_state.copy()
        for side in self._inactive_sides:
            indices = A3_SIDE_INDICES[side]
            model_state[indices] = self._training_state[indices]
        model_obs = {**obs, "state": model_state}

        result = dict(self._policy.infer(model_obs, **kwargs))
        actions = np.asarray(result.get("actions"))
        if actions.ndim != 2 or actions.shape[1] != A3_ACTION_DIM or not np.isfinite(actions).all():
            raise ValueError(f"A3 actions must be finite [T,{A3_ACTION_DIM}], got {actions.shape}")
        actions = actions.copy()
        for side in self._inactive_sides:
            indices = A3_SIDE_INDICES[side]
            actions[:, indices] = live_state[indices]
        result["actions"] = actions
        return result

    @property
    def metadata(self) -> dict:
        return self._metadata


def make_a3_example() -> dict:
    return {
        "state": np.zeros((A3_ACTION_DIM,), dtype=np.float32),
        "images": {
            "chest_rgb": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class A3Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        chest_image = _parse_image(data["images"]["chest_rgb"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                images = {
                    "base_0_rgb": chest_image,
                    "left_wrist_0_rgb": np.zeros_like(chest_image),
                    "right_wrist_0_rgb": np.zeros_like(chest_image),
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.False_,
                    "right_wrist_0_rgb": np.False_,
                }
            case _model.ModelType.PI0_FAST:
                images = {
                    "base_0_rgb": chest_image,
                    "base_1_rgb": np.zeros_like(chest_image),
                    "wrist_0_rgb": np.zeros_like(chest_image),
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "base_1_rgb": np.True_,
                    "wrist_0_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": images,
            "image_mask": image_masks,
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class A3Outputs(transforms.DataTransformFn):
    action_dim: int = A3_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
