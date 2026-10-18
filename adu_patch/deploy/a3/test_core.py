from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from deploy.a3 import core


def safe_pose() -> core.Pose:
    return core.Pose(
        arm=(core.ARM_LOWER + core.ARM_UPPER) / 2,
        hand=np.full(20, 2000.0),
        waist=np.zeros(3),
    )


class CoreSafetyTest(unittest.TestCase):
    def test_model_contract(self) -> None:
        self.assertEqual(core.MODEL_ACTION_DIM, 37)
        self.assertEqual(core.ACTION_HORIZON, 50)
        self.assertEqual(core.MODEL_NAMES[-3:], core.WAIST_NAMES)

    def test_arm_limits_match_a3_urdf(self) -> None:
        np.testing.assert_allclose(
            np.rad2deg(core.ARM_LOWER),
            [-165, -5, -160, -55, -160, -93, -93, -165, -150, -160, -55, -160, -93, -93],
        )
        np.testing.assert_allclose(
            np.rad2deg(core.ARM_UPPER),
            [165, 150, 160, 100, 160, 93, 93, 165, 5, 160, 100, 160, 93, 93],
        )
        self.assertGreater(core.ARM_MARGIN, 0.0)

    def test_pose_limit_violation_is_rejected(self) -> None:
        pose = safe_pose()
        bad = core.Pose(pose.arm.copy(), pose.hand.copy(), pose.waist.copy())
        bad.arm[0] = core.ARM_UPPER[0]
        with self.assertRaisesRegex(ValueError, "arm pose violates"):
            core.validate_pose(bad)

    def test_policy_and_transition_hand_steps_are_separate(self) -> None:
        self.assertEqual(core.HARD_MAX_ARM_STEP, 0.02)
        self.assertEqual(core.DEFAULT_POLICY_ARM_STEP, 0.02)
        self.assertEqual(core.DEFAULT_TRANSITION_HAND_STEP, 20.0)
        self.assertEqual(core.DEFAULT_POLICY_HAND_STEP, 200.0)
        self.assertEqual(core.HARD_MAX_HAND_STEP, 200.0)
        self.assertLess(core.DEFAULT_TRANSITION_HAND_STEP, core.DEFAULT_POLICY_HAND_STEP)
        self.assertLessEqual(core.DEFAULT_POLICY_HAND_STEP, core.HARD_MAX_HAND_STEP)

    def test_executor_defaults_use_measured_state_and_full_policy_steps(self) -> None:
        try:
            from deploy.a3 import executor
        except ModuleNotFoundError as error:
            if error.name in {"joint_msgs", "ros2_plugin_proto", "rclpy"}:
                self.skipTest(f"ADU ROS message environment unavailable: {error.name}")
            raise
        argv = ["executor.py", "--dry-run"]
        with mock.patch.object(executor.sys, "argv", argv):
            args = executor.parse_args()
        self.assertEqual(args.observation_state_source, "measured")
        self.assertEqual(args.max_arm_step, 0.02)
        self.assertEqual(args.max_hand_step, 200.0)
        run_config = executor.build_run_config(args)
        self.assertFalse(run_config["uses_arm_state_to_command_projection"])

    def test_startup_and_intermediate_poses_do_not_use_arm_projection(self) -> None:
        try:
            from deploy.a3 import executor
        except ModuleNotFoundError as error:
            if error.name in {"joint_msgs", "ros2_plugin_proto", "rclpy"}:
                self.skipTest(f"ADU ROS message environment unavailable: {error.name}")
            raise
        measured = safe_pose()
        node = SimpleNamespace(measured_pose=lambda: measured)

        startup, source = executor.capture_startup_command(node)
        intermediate = executor.build_intermediate_pose(startup)

        np.testing.assert_allclose(startup.state(), measured.state())
        np.testing.assert_allclose(intermediate.arm, executor.INTERMEDIATE_ARM_STATE)
        self.assertEqual(source, "live_measured_feedback")

    def test_transition_respects_every_step_limit(self) -> None:
        start = safe_pose()
        target = core.Pose(
            arm=start.arm + 0.02,
            hand=start.hand + 100.0,
            waist=start.waist + 0.01,
        )
        frames = list(
            core.interpolate_poses(
                start,
                target,
                rate_hz=50.0,
                min_duration_s=0.1,
            )
        )
        previous = start
        for frame in frames:
            self.assertLessEqual(
                float(np.max(np.abs(frame.arm - previous.arm))),
                core.DEFAULT_TRANSITION_ARM_STEP + 1e-12,
            )
            self.assertLessEqual(
                float(np.max(np.abs(frame.hand - previous.hand))),
                core.DEFAULT_TRANSITION_HAND_STEP + 1e-12,
            )
            self.assertLessEqual(
                float(np.max(np.abs(frame.waist - previous.waist))),
                core.DEFAULT_TRANSITION_WAIST_STEP + 1e-12,
            )
            previous = frame
        np.testing.assert_allclose(frames[-1].state(), target.state())

    def test_bounded_action_respects_hard_deltas(self) -> None:
        previous = safe_pose()
        requested = previous.state().astype(np.float64)
        requested[:14] += 1.0
        requested[14:34] += 1000.0
        requested[34:] += 0.5
        command, stats = core.bounded_action(
            previous, requested, max_hand_step=core.HARD_MAX_HAND_STEP
        )
        self.assertLessEqual(stats["bounded_arm_max_delta"], core.HARD_MAX_ARM_STEP + 1e-12)
        self.assertLessEqual(stats["bounded_hand_max_delta"], core.HARD_MAX_HAND_STEP + 1e-12)
        self.assertLessEqual(stats["bounded_waist_max_delta"], core.HARD_MAX_WAIST_STEP + 1e-12)
        core.validate_pose(command)

    def test_bounded_action_is_open_loop(self) -> None:
        previous = safe_pose()
        requested = previous.state().astype(np.float64)
        requested[:14] += 1.0
        requested[14:34] += 1000.0
        requested[34:] += 0.5
        command, stats = core.bounded_action(previous, requested)
        self.assertAlmostEqual(stats["bounded_arm_max_delta"], core.HARD_MAX_ARM_STEP)
        self.assertAlmostEqual(
            stats["bounded_hand_max_delta"], core.DEFAULT_POLICY_HAND_STEP
        )
        self.assertAlmostEqual(stats["bounded_waist_max_delta"], core.HARD_MAX_WAIST_STEP)
        self.assertNotIn("command_arm_tracking_error", stats)
        core.validate_pose(command)

    def test_action_chunk_diagnostics_report_raw_violations_and_bound_deltas(self) -> None:
        initial = safe_pose()
        actions = np.tile(initial.state(), (core.ACTION_HORIZON, 1)).astype(np.float64)
        actions[0, 0] = core.ARM_UPPER[0]
        actions[1, 14] = -1.0
        final, report = core.analyze_action_chunk(initial, actions)
        self.assertEqual(report["actions"], core.ACTION_HORIZON)
        self.assertEqual(report["raw_limit_violation_count"]["arm"], 1)
        self.assertEqual(report["raw_limit_violation_count"]["hand"], 1)
        self.assertEqual(report["raw_limit_violation_total"], 2)
        self.assertGreater(report["delta_limited_action_count"]["arm"], 0)
        self.assertLessEqual(
            report["bounded_max_delta"]["arm"], core.HARD_MAX_ARM_STEP + 1e-12
        )
        self.assertLessEqual(
            report["bounded_max_delta"]["hand"], core.DEFAULT_POLICY_HAND_STEP + 1e-12
        )
        core.validate_pose(final)


    def test_transition_does_not_gate_on_feedback_tracking(self) -> None:
        try:
            from deploy.a3 import executor
        except ModuleNotFoundError as error:
            if error.name in {"joint_msgs", "ros2_plugin_proto", "rclpy"}:
                self.skipTest(f"ADU ROS message environment unavailable: {error.name}")
            raise
        start = safe_pose()
        target = core.Pose(start.arm.copy(), start.hand.copy(), start.waist.copy())
        target.hand[0] += 8.0

        class FakeNode:
            def __init__(self) -> None:
                self.measured_calls = 0
                self.published = []

            def measured_pose(self) -> core.Pose:
                self.measured_calls += 1
                return core.Pose(
                    arm=start.arm + 0.5,
                    hand=start.hand + 500.0,
                    waist=start.waist + 0.2,
                )

            def assert_spin_healthy(self) -> None:
                pass

            def publish_pose(self, pose: core.Pose, **_kwargs: object) -> core.Pose:
                self.published.append(pose)
                return pose

        node = FakeNode()
        args = SimpleNamespace(
            hold_rate=50.0,
            max_arm_step=core.HARD_MAX_ARM_STEP,
            max_hand_step=4.0,
            max_waist_step=core.HARD_MAX_WAIST_STEP,
        )
        keys = SimpleNamespace(poll=lambda: None)
        candidate = SimpleNamespace(arm_template=None, hand_template=None, waist_height=0.0)
        with mock.patch.object(executor, "_timed_tick", return_value=None):
            result = executor.transition(
                node,
                start,
                target,
                candidate=candidate,
                lower_lock=None,
                duration_s=0.0,
                args=args,
                keys=keys,
                label="TEST",
                side_scope=executor.A3SideScope(("left", "right")),
            )
        self.assertIsNone(result.fault)
        np.testing.assert_allclose(result.pose.state(), target.state())
        self.assertEqual(node.measured_calls, 2)
        self.assertEqual(len(node.published), 2)

    def test_command_log_records_feedback_and_tracking_error(self) -> None:
        try:
            from deploy.a3 import executor
        except ModuleNotFoundError as error:
            if error.name in {"joint_msgs", "ros2_plugin_proto", "rclpy"}:
                self.skipTest(f"ADU ROS message environment unavailable: {error.name}")
            raise
        command = safe_pose()
        feedback = core.Pose(
            arm=command.arm + 0.01,
            hand=command.hand + 2.0,
            waist=command.waist + 0.001,
        )
        with tempfile.TemporaryDirectory() as directory:
            log = executor.CommandLog(
                Path(directory), run_config={"effective_step_limits": {"arm": 0.02}}
            )
            observation = executor.PolicyObservation(
                state_pose=feedback,
                measured_pose=feedback,
                state_ages_ms=(4.0, 5.0, 6.0),
                source="measured",
            )
            log.write_action(
                chunk=1,
                step=2,
                round_trip_ms=3.0,
                stats={
                    "requested_arm_max_delta": 0.02,
                    "requested_hand_max_delta": 5.0,
                    "requested_waist_max_delta": 0.001,
                    "bounded_arm_max_delta": 0.02,
                    "bounded_hand_max_delta": 5.0,
                    "bounded_waist_max_delta": 0.001,
                },
                pose=command,
                feedback=feedback,
                tracking_reference=command,
                state_ages_ms=(1.0, 2.0, 3.0),
                configured_steps=(0.02, 200.0, 0.003),
                observation=observation,
                raw_action=command.state(),
                scoped_action=command.state(),
            )
            log.close()
            with (log.run_dir / "commands.csv").open() as file:
                row = next(csv.DictReader(file))
            with (log.run_dir / "run_config.json").open() as file:
                run_config = json.load(file)
        self.assertEqual(float(row["arm_state_age_ms"]), 1.0)
        self.assertAlmostEqual(
            float(row["tracking_error:left_shoulder_pitch_joint"]), 0.01
        )
        self.assertEqual(float(row["configured_arm_max_step"]), 0.02)
        self.assertEqual(float(row["configured_hand_max_step"]), 200.0)
        self.assertAlmostEqual(
            float(row["observation:left_shoulder_pitch_joint"]), feedback.arm[0]
        )
        self.assertEqual(run_config["effective_step_limits"]["arm"], 0.02)

    def test_policy_observation_uses_measured_feedback_by_default_mode(self) -> None:
        try:
            from deploy.a3 import executor
        except ModuleNotFoundError as error:
            if error.name in {"joint_msgs", "ros2_plugin_proto", "rclpy"}:
                self.skipTest(f"ADU ROS message environment unavailable: {error.name}")
            raise
        command = safe_pose()
        measured = core.Pose(
            arm=command.arm + 0.01,
            hand=command.hand + 2.0,
            waist=command.waist + 0.001,
        )
        timestamp = time.monotonic()
        node = SimpleNamespace(
            measured_sample=lambda: (measured, (timestamp, timestamp, timestamp))
        )

        sample = executor.capture_policy_observation(
            node, command, SimpleNamespace(observation_state_source="measured")
        )

        np.testing.assert_allclose(sample.state_pose.state(), measured.state())
        np.testing.assert_allclose(sample.measured_pose.state(), measured.state())
        self.assertEqual(sample.source, "measured")

    def test_legacy_policy_observation_mode_uses_last_command(self) -> None:
        try:
            from deploy.a3 import executor
        except ModuleNotFoundError as error:
            if error.name in {"joint_msgs", "ros2_plugin_proto", "rclpy"}:
                self.skipTest(f"ADU ROS message environment unavailable: {error.name}")
            raise
        command = safe_pose()
        measured = core.Pose(
            arm=command.arm + 0.01,
            hand=command.hand + 2.0,
            waist=command.waist + 0.001,
        )
        timestamp = time.monotonic()
        node = SimpleNamespace(
            measured_sample=lambda: (measured, (timestamp, timestamp, timestamp))
        )

        sample = executor.capture_policy_observation(
            node, command, SimpleNamespace(observation_state_source="command")
        )

        np.testing.assert_allclose(sample.state_pose.state(), command.state())
        np.testing.assert_allclose(sample.measured_pose.state(), measured.state())

    def test_invalid_action_chunk_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            core.validate_action_chunk(np.zeros((25, 37)))
        invalid = np.zeros((50, 37))
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            core.validate_action_chunk(invalid)

    def test_medoid_is_an_existing_pose(self) -> None:
        base = safe_pose().state()
        states = {
            "a": base - 0.1,
            "b": base,
            "c": base + 1.0,
        }
        self.assertEqual(core.select_medoid(states), "b")

    def test_load_init_pose_config_checks_joint_order(self) -> None:
        pose = safe_pose()
        raw = {
            "schema_version": 2,
            "model": {
                "action_dim": 37,
                "action_horizon": 50,
                "names": list(core.MODEL_NAMES),
            },
            "default_pose_id": "episode_001",
            "candidates": {
                "episode_001": {
                    "qpose": pose.state().tolist(),
                    "arm_template": {
                        "stiffness": [100.0] * 14,
                        "damping": [5.0] * 14,
                    },
                    "hand_template": {
                        "frame_id": "O10Hand",
                        "name": [],
                        "velocity": [],
                        "effort": [125.0] * 20,
                    },
                    "waist_height": 0.0,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = core.load_init_pose_config(path)
            self.assertEqual(config.select().pose_id, "episode_001")
            self.assertEqual(config.select().hand_template.names, ())
            self.assertEqual(config.select().hand_template.effort.tolist(), [125.0] * 20)
            raw["candidates"]["episode_001"]["hand_template"]["velocity"] = [0.0]
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hand velocity"):
                core.load_init_pose_config(path)
            raw["candidates"]["episode_001"]["hand_template"]["velocity"] = []
            raw["candidates"]["episode_001"]["arm_template"]["stiffness"][0] = 501.0
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stiffness"):
                core.load_init_pose_config(path)
            raw["candidates"]["episode_001"]["arm_template"]["stiffness"][0] = 100.0
            raw["candidates"]["episode_001"]["waist_height"] = 0.1
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "waist height"):
                core.load_init_pose_config(path)
            raw["candidates"]["episode_001"]["waist_height"] = 0.0
            raw["model"]["names"][0] = "wrong"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "joint order"):
                core.load_init_pose_config(path)


    def test_hand_command_preserves_recorded_message_envelope(self) -> None:
        try:
            from deploy.a3.ros_node import A3DeploymentNode
        except ModuleNotFoundError as error:
            if error.name in {"joint_msgs", "ros2_plugin_proto", "rclpy"}:
                self.skipTest(f"ADU ROS message environment unavailable: {error.name}")
            raise

        pose = safe_pose()
        pose.hand[0] = -1.0
        pose.hand[-1] = 5000.0
        template = core.HandTemplate(
            frame_id="O10Hand",
            names=(),
            velocity=np.asarray([], dtype=np.float64),
            effort=np.full(20, 125.0, dtype=np.float64),
        )
        stamp = SimpleNamespace(sec=1, nanosec=2)
        message = A3DeploymentNode._make_hand_command(pose, template, stamp)

        self.assertEqual(message.header.frame_id, "O10Hand")
        self.assertEqual(list(message.name), [])
        self.assertEqual(list(message.velocity), [])
        self.assertEqual(list(message.effort), [125.0] * 20)
        self.assertEqual(message.position[0], 0.0)
        self.assertEqual(message.position[-1], 4095.0)
        self.assertEqual(len(message.position), 20)


if __name__ == "__main__":
    unittest.main()
