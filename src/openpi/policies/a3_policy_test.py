import numpy as np
import pytest

from openpi import transforms
from openpi.policies import a3_policy


class CapturingPolicy:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.observation = None
        self.metadata = {"base": "preserved"}

    def infer(self, obs: dict, **kwargs) -> dict:
        self.observation = obs
        return {"actions": self.actions.copy(), "extra": kwargs}


def test_right_scope_masks_left_input_and_holds_left_output() -> None:
    live = np.arange(37, dtype=np.float32) + 100
    training = np.arange(37, dtype=np.float32) + 1000
    predicted = np.tile(np.arange(37, dtype=np.float32) + 2000, (3, 1))
    base = CapturingPolicy(predicted)
    policy = a3_policy.A3ScopedPolicy(base, active_sides=("right",), training_state=training)

    original = live.copy()
    result = policy.infer({"state": live, "images": {}, "active_sides": ["right"]}, request_id="test")

    left = a3_policy.A3_SIDE_INDICES["left"]
    right = a3_policy.A3_SIDE_INDICES["right"]
    np.testing.assert_array_equal(base.observation["state"][left], training[left])
    np.testing.assert_array_equal(base.observation["state"][right], live[right])
    np.testing.assert_array_equal(base.observation["state"][34:37], live[34:37])
    np.testing.assert_array_equal(result["actions"][:, left], np.tile(live[left], (3, 1)))
    np.testing.assert_array_equal(result["actions"][:, right], predicted[:, right])
    np.testing.assert_array_equal(result["actions"][:, 34:37], predicted[:, 34:37])
    np.testing.assert_array_equal(live, original)
    assert result["extra"] == {"request_id": "test"}
    assert policy.metadata["base"] == "preserved"
    assert policy.metadata["a3_side_scope"]["active_sides"] == ["right"]


def test_scope_rejects_mismatched_request() -> None:
    base = CapturingPolicy(np.zeros((3, 37), dtype=np.float32))
    policy = a3_policy.A3ScopedPolicy(
        base,
        active_sides=("right",),
        training_state=np.zeros(37, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="scope mismatch"):
        policy.infer({"state": np.zeros(37), "active_sides": ["left"]})


@pytest.mark.parametrize(
    ("value", "expected"),
    [("right", ("right",)), ("left,right", ("left", "right")), ("right,left", ("left", "right"))],
)
def test_parse_active_sides(value: str, expected: tuple[str, ...]) -> None:
    assert a3_policy.parse_active_sides(value) == expected


@pytest.mark.parametrize("value", ["", "head", "right,head"])
def test_parse_active_sides_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="active sides"):
        a3_policy.parse_active_sides(value)


def test_scope_rejects_non_frozen_inactive_side() -> None:
    mean = np.zeros(37, dtype=np.float32)
    q01 = np.zeros(37, dtype=np.float32)
    q99 = np.zeros(37, dtype=np.float32)
    q99[0] = 1.0
    stats = transforms.NormStats(mean=mean, std=q99, q01=q01, q99=q99)
    with pytest.raises(ValueError, match="dimensions are not frozen: \\[0\\]"):
        a3_policy.validate_inactive_sides_frozen(("right",), {"state": stats, "actions": stats}, use_quantiles=True)
