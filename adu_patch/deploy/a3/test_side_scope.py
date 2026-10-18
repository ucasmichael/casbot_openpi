from __future__ import annotations

import unittest

import numpy as np

from deploy.a3 import core
from deploy.a3.side_scope import A3SideScope, parse_active_sides


def pose(offset: float) -> core.Pose:
    return core.Pose(
        arm=(core.ARM_LOWER + core.ARM_UPPER) / 2.0 + offset,
        hand=np.full(20, 2000.0 + offset),
        waist=np.zeros(3),
    )


class SideScopeTest(unittest.TestCase):
    def test_right_only_holds_left_pose(self) -> None:
        target = pose(0.2)
        live = pose(0.0)
        result = A3SideScope(("right",)).hold_pose(target, live)

        np.testing.assert_allclose(result.arm[:7], live.arm[:7])
        np.testing.assert_allclose(result.hand[:10], live.hand[:10])
        np.testing.assert_allclose(result.arm[7:], target.arm[7:])
        np.testing.assert_allclose(result.hand[10:], target.hand[10:])
        np.testing.assert_allclose(result.waist, target.waist)

    def test_right_only_holds_every_action_frame(self) -> None:
        live = pose(0.0).state()
        actions = np.tile(pose(0.2).state(), (5, 1))
        result = A3SideScope(("right",)).hold_actions(actions, live)

        np.testing.assert_allclose(result[:, :7], np.tile(live[:7], (5, 1)))
        np.testing.assert_allclose(result[:, 14:24], np.tile(live[14:24], (5, 1)))
        np.testing.assert_allclose(result[:, 7:14], actions[:, 7:14])
        np.testing.assert_allclose(result[:, 24:34], actions[:, 24:34])
        np.testing.assert_allclose(result[:, 34:], actions[:, 34:])

    def test_parser_is_deterministic(self) -> None:
        self.assertEqual(parse_active_sides("right,left"), ("left", "right"))
        self.assertEqual(parse_active_sides(None), ("left", "right"))
        with self.assertRaisesRegex(ValueError, "unknown active sides"):
            parse_active_sides("right,head")

    def test_server_metadata_must_match(self) -> None:
        scope = A3SideScope(("right",))
        metadata = {
            "a3_side_scope": {
                "active_sides": ["right"],
                "inactive_input": "checkpoint_state_mean",
                "inactive_output": "live_request_state",
            }
        }
        scope.validate_server_metadata(metadata)
        with self.assertRaisesRegex(RuntimeError, "scope mismatch"):
            scope.validate_server_metadata({**metadata, "a3_side_scope": {**metadata["a3_side_scope"], "active_sides": ["left"]}})


if __name__ == "__main__":
    unittest.main()
