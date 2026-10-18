"""ROS subscriptions and command publication for the A3 deployment state machine."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import threading
import time
from typing import Any

from joint_msgs.msg import Command
from joint_msgs.msg import JointCommand
import numpy as np
from PIL import Image as PilImage
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from ros2_plugin_proto.msg import RosMsgWrapper
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

from .core import ARM_NAMES
from .core import HAND_NAMES
from .core import HAND_STATE_ALIASES
from .core import WAIST_STATE_NAMES
from .core import ArmTemplate
from .core import HandTemplate
from .core import Pose
from .core import align_positions
from .core import encode_waist_payload
from .core import parse_waist_payload
from .core import validate_arm_template
from .core import validate_hand_template
from .core import validate_pose
from .core import validate_waist_height
from .core import wrapper_bytes
from .side_scope import A3SideScope
from .source_state_window import SourceStateWindow

ARM_STATE_TOPIC = "/motion/control/arm_joint_state"
HAND_STATE_TOPIC = "/motion/control/hand_joint_state"
WAIST_STATE_TOPIC = "/motion/control/waist_joint_state"
NECK_STATE_TOPIC = "/motion/control/neck_joint_state"
LEG_STATE_TOPIC = "/motion/control/leg_joint_state"
CAMERA_TOPIC = "/hal/chest_front_d457_camera/rgb"

ARM_COMMAND_TOPIC = "/motion/control/pnc/arm_joint_command"
HAND_COMMAND_TOPIC = "/motion/control/hand_joint_command"
WAIST_COMMAND_TOPIC = "/motion/control/move_waist/pb_3Aaimdk_2Eprotocol_2EMotionControlMoveWaistChannel"
NECK_COMMAND_TOPIC = "/motion/control/neck_joint_command"
LEG_COMMAND_TOPIC = "/motion/control/leg_joint_command"
PRIMARY_COMMAND_TOPICS = (ARM_COMMAND_TOPIC, HAND_COMMAND_TOPIC, WAIST_COMMAND_TOPIC)


@dataclass(frozen=True)
class LockedJoints:
    names: tuple[str, ...]
    positions: np.ndarray


@dataclass(frozen=True)
class LowerLock:
    neck: LockedJoints
    leg: LockedJoints


class A3DeploymentNode(Node):
    def __init__(self, *, max_state_age_s: float, max_image_age_s: float) -> None:
        super().__init__("openpi_a3_deployment")
        self.max_state_age_s = max_state_age_s
        self.max_image_age_s = max_image_age_s
        self._errors: dict[str, str] = {}
        self._snapshot_lock = threading.Lock()
        self._spin_executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._spin_error: BaseException | None = None

        self._arm_state: tuple[np.ndarray, np.ndarray, float] | None = None
        self._hand_state: tuple[np.ndarray, float] | None = None
        self._waist_state: tuple[np.ndarray, float] | None = None
        self._neck_state: tuple[tuple[str, ...], np.ndarray, float] | None = None
        self._leg_state: tuple[tuple[str, ...], np.ndarray, float] | None = None
        self._image: tuple[Image, float] | None = None
        self._source_state_window = SourceStateWindow()

        self._arm_command: tuple[np.ndarray, ArmTemplate, float] | None = None
        self._hand_command: tuple[np.ndarray, float] | None = None
        self._waist_command: tuple[np.ndarray, float, float] | None = None

        self._arm_publisher = None
        self._hand_publisher = None
        self._waist_publisher = None
        self._neck_publisher = None
        self._leg_publisher = None
        self._sequence = 0

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        command_monitor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.create_subscription(JointState, ARM_STATE_TOPIC, self._on_arm_state, state_qos)
        self.create_subscription(JointState, HAND_STATE_TOPIC, self._on_hand_state, state_qos)
        self.create_subscription(JointState, WAIST_STATE_TOPIC, self._on_waist_state, state_qos)
        self.create_subscription(JointState, NECK_STATE_TOPIC, self._on_neck_state, state_qos)
        self.create_subscription(JointState, LEG_STATE_TOPIC, self._on_leg_state, state_qos)
        self.create_subscription(Image, CAMERA_TOPIC, self._on_image, state_qos)
        self.create_subscription(
            JointCommand,
            ARM_COMMAND_TOPIC,
            self._on_arm_command,
            command_monitor_qos,
        )
        self.create_subscription(
            JointState,
            HAND_COMMAND_TOPIC,
            self._on_hand_command,
            command_monitor_qos,
        )
        self.create_subscription(
            RosMsgWrapper,
            WAIST_COMMAND_TOPIC,
            self._on_waist_command,
            command_monitor_qos,
        )

    def start_background_spin(self) -> None:
        if self._spin_thread is not None:
            raise RuntimeError("ROS background executor is already running")
        self._spin_executor = SingleThreadedExecutor(context=self.context)
        self._spin_executor.add_node(self)

        def spin() -> None:
            try:
                assert self._spin_executor is not None
                self._spin_executor.spin()
            except BaseException as error:
                self._spin_error = error

        self._spin_thread = threading.Thread(target=spin, name="a3-ros-executor", daemon=True)
        self._spin_thread.start()

    def assert_spin_healthy(self) -> None:
        if self._spin_error is not None:
            raise RuntimeError(f"ROS background executor failed: {self._spin_error}")
        if self._spin_thread is None or not self._spin_thread.is_alive():
            raise RuntimeError("ROS background executor is not running")

    def stop_background_spin(self) -> None:
        executor = self._spin_executor
        thread = self._spin_thread
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if thread is not None:
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError("ROS background executor did not stop")
        self._spin_executor = None
        self._spin_thread = None

    def _warn_changed(self, source: str, error: Exception) -> None:
        message = str(error)
        if self._errors.get(source) != message:
            self.get_logger().warning(f"rejected {source}: {message}")
            self._errors[source] = message

    def _on_arm_state(self, message: JointState) -> None:
        try:
            values = align_positions(message.name, message.position, ARM_NAMES)
            velocity = align_positions(message.name, message.velocity, ARM_NAMES)
            with self._snapshot_lock:
                self._arm_state = (values, velocity, time.monotonic())
                self._source_state_window.add("arm", message, values)
        except Exception as error:
            self._warn_changed("arm state", error)

    def _on_hand_state(self, message: JointState) -> None:
        try:
            values = align_positions(
                message.name,
                message.position,
                HAND_NAMES,
                aliases=HAND_STATE_ALIASES,
            )
            if np.any(values < -5.0) or np.any(values > 4100.0):
                raise ValueError(f"hand feedback is far outside encoder range: min={values.min()} max={values.max()}")
            with self._snapshot_lock:
                self._hand_state = (np.clip(values, 0.0, 4095.0), time.monotonic())
                self._source_state_window.add("hand", message, self._hand_state[0])
        except Exception as error:
            self._warn_changed("hand state", error)

    def _on_waist_state(self, message: JointState) -> None:
        try:
            values = align_positions(message.name, message.position, WAIST_STATE_NAMES)
            with self._snapshot_lock:
                self._waist_state = (values, time.monotonic())
                self._source_state_window.add("waist", message, values)
        except Exception as error:
            self._warn_changed("waist state", error)

    def _on_neck_state(self, message: JointState) -> None:
        self._on_locked_state("neck", message)

    def _on_leg_state(self, message: JointState) -> None:
        self._on_locked_state("leg", message)

    def _on_locked_state(self, group: str, message: JointState) -> None:
        try:
            names = tuple(str(name) for name in message.name)
            values = np.asarray(message.position, dtype=np.float64)
            if not names or len(names) != len(values) or not np.isfinite(values).all():
                raise ValueError("invalid names or positions")
            record = (names, values.copy(), time.monotonic())
            with self._snapshot_lock:
                if group == "neck":
                    self._neck_state = record
                else:
                    self._leg_state = record
        except Exception as error:
            self._warn_changed(f"{group} state", error)

    def _on_image(self, message: Image) -> None:
        with self._snapshot_lock:
            self._image = (message, time.monotonic())

    def _on_arm_command(self, message: JointCommand) -> None:
        try:
            joints = list(message.joints)
            names = [str(joint.name) for joint in joints]
            position = align_positions(names, [joint.position for joint in joints], ARM_NAMES)
            template = ArmTemplate(
                stiffness=align_positions(names, [joint.stiffness for joint in joints], ARM_NAMES),
                damping=align_positions(names, [joint.damping for joint in joints], ARM_NAMES),
            )
            with self._snapshot_lock:
                self._arm_command = (position, template, time.monotonic())
        except Exception as error:
            self._warn_changed("arm command", error)

    def _on_hand_command(self, message: JointState) -> None:
        try:
            names = list(message.name) or list(HAND_NAMES)
            position = align_positions(names, message.position, HAND_NAMES)
            with self._snapshot_lock:
                self._hand_command = (position, time.monotonic())
        except Exception as error:
            self._warn_changed("hand command", error)

    def _on_waist_command(self, message: RosMsgWrapper) -> None:
        try:
            waist, height = parse_waist_payload(wrapper_bytes(message))
            with self._snapshot_lock:
                self._waist_command = (waist, height, time.monotonic())
        except Exception as error:
            self._warn_changed("waist command", error)

    @staticmethod
    def _fresh(timestamp: float, max_age_s: float, label: str) -> None:
        age = time.monotonic() - timestamp
        if age > max_age_s:
            raise RuntimeError(f"stale {label}: {age:.3f}s > {max_age_s:.3f}s")

    def _measured_snapshot(
        self,
    ) -> tuple[Pose, np.ndarray, tuple[float, float, float]]:
        with self._snapshot_lock:
            arm_state = self._arm_state
            hand_state = self._hand_state
            waist_state = self._waist_state
            if arm_state is None or hand_state is None or waist_state is None:
                raise RuntimeError("arm/hand/waist feedback is incomplete")
            arm = arm_state[0].copy()
            velocity = arm_state[1].copy()
            hand = hand_state[0].copy()
            waist = waist_state[0].copy()
            timestamps = (arm_state[2], hand_state[1], waist_state[1])
        self._fresh(timestamps[0], self.max_state_age_s, "arm state")
        self._fresh(timestamps[1], self.max_state_age_s, "hand state")
        self._fresh(timestamps[2], self.max_state_age_s, "waist state")
        pose = Pose(arm=arm, hand=hand, waist=waist)
        validate_pose(pose)
        return pose, velocity, timestamps

    def measured_pose(self) -> Pose:
        return self._measured_snapshot()[0]

    def measured_sample(self) -> tuple[Pose, tuple[float, float, float]]:
        """Return the latest measured pose and monotonic receive timestamps."""
        pose, _velocity, timestamps = self._measured_snapshot()
        return pose, timestamps

    def source_state_window(self) -> dict[str, Any]:
        """Return source-stamped state histories while holding the ROS snapshot lock."""
        with self._snapshot_lock:
            return self._source_state_window.snapshot()

    def projected_command_pose(self, projection: Any) -> Pose:
        measured, velocity, _timestamps = self._measured_snapshot()
        pose = Pose(
            arm=projection.predict(measured.arm, velocity),
            hand=measured.hand.copy(),
            waist=measured.waist.copy(),
        )
        validate_pose(pose)
        return pose

    def command_pose(self) -> tuple[Pose, ArmTemplate, float]:
        with self._snapshot_lock:
            arm_command = self._arm_command
            hand_command = self._hand_command
            waist_command = self._waist_command
            if arm_command is None or hand_command is None or waist_command is None:
                raise RuntimeError("arm/hand/waist command-domain pose is incomplete")
            arm = arm_command[0].copy()
            template = ArmTemplate(
                stiffness=arm_command[1].stiffness.copy(),
                damping=arm_command[1].damping.copy(),
            )
            hand = hand_command[0].copy()
            waist = waist_command[0].copy()
            height = waist_command[1]
            timestamps = (arm_command[2], hand_command[1], waist_command[2])
        self._fresh(timestamps[0], self.max_state_age_s, "arm command")
        self._fresh(timestamps[1], self.max_state_age_s, "hand command")
        self._fresh(timestamps[2], self.max_state_age_s, "waist command")
        pose = Pose(arm=arm, hand=hand, waist=waist)
        validate_pose(pose)
        return pose, template, height

    def lower_lock(self) -> LowerLock:
        with self._snapshot_lock:
            neck_state = self._neck_state
            leg_state = self._leg_state
            if neck_state is None or leg_state is None:
                raise RuntimeError("neck/leg feedback is incomplete")
            neck = LockedJoints(neck_state[0], neck_state[1].copy())
            leg = LockedJoints(leg_state[0], leg_state[1].copy())
            timestamps = (neck_state[2], leg_state[2])
        self._fresh(timestamps[0], self.max_state_age_s, "neck state")
        self._fresh(timestamps[1], self.max_state_age_s, "leg state")
        return LowerLock(neck=neck, leg=leg)

    def wait_until_ready(
        self,
        *,
        require_commands: bool,
        require_lower: bool,
        timeout_s: float,
        require_image: bool = True,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while rclpy.ok() and time.monotonic() < deadline:
            self.assert_spin_healthy()
            try:
                self.measured_pose()
                if require_commands:
                    self.command_pose()
                if require_lower:
                    self.lower_lock()
                if require_image:
                    self.image_chw()
                return
            except (RuntimeError, ValueError) as error:
                last_error = error
                time.sleep(0.01)
        raise TimeoutError(f"A3 observations did not become ready: {last_error}")

    def wait_for_feedback_after(self, timestamp: float, timeout_s: float) -> Pose:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while rclpy.ok() and time.monotonic() < deadline:
            self.assert_spin_healthy()
            try:
                pose, _velocity, timestamps = self._measured_snapshot()
                if min(timestamps) > timestamp:
                    return pose
                last_error = RuntimeError("waiting for arm/hand/waist feedback newer than the warm-up barrier")
            except (RuntimeError, ValueError) as error:
                last_error = error
            time.sleep(0.005)
        raise TimeoutError(f"fresh post-warm-up feedback did not arrive: {last_error}")

    def image_chw(self) -> np.ndarray:
        with self._snapshot_lock:
            image = self._image
        if image is None:
            raise RuntimeError("camera image is unavailable")
        message, timestamp = image
        self._fresh(timestamp, self.max_image_age_s, "camera image")
        encoding = message.encoding.lower()
        if encoding not in {"rgb8", "bgr8"}:
            raise ValueError(f"unsupported RGB encoding {message.encoding!r}")
        if message.height <= 0 or message.width <= 0 or message.step < message.width * 3:
            raise ValueError("invalid RGB image dimensions")
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        image = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        source = PilImage.fromarray(np.ascontiguousarray(image))
        ratio = max(source.width / 224.0, source.height / 224.0)
        size = (max(1, int(source.width / ratio)), max(1, int(source.height / ratio)))
        resized = source.resize(size, resample=PilImage.Resampling.BILINEAR)
        padded = PilImage.new("RGB", (224, 224), 0)
        padded.paste(resized, ((224 - size[0]) // 2, (224 - size[1]) // 2))
        return np.ascontiguousarray(np.asarray(padded, dtype=np.uint8).transpose(2, 0, 1))

    def observation(
        self,
        state_pose: Pose,
        prompt: str,
        *,
        active_sides: tuple[str, ...] | None = None,
        include_image: bool = True,
    ) -> dict[str, Any]:
        validate_pose(state_pose)
        observation = {
            "state": state_pose.state(),
            "prompt": prompt,
        }
        if include_image:
            observation["images"] = {"chest_rgb": self.image_chw()}
        if active_sides is not None:
            observation["active_sides"] = list(active_sides)
        return observation

    def publisher_conflicts(self, *, include_lower: bool) -> dict[str, list[str]]:
        topics = list(PRIMARY_COMMAND_TOPICS)
        if include_lower:
            topics.extend((NECK_COMMAND_TOPIC, LEG_COMMAND_TOPIC))
        conflicts = {}
        for topic in topics:
            names = sorted(
                f"{item.node_namespace.rstrip('/')}/{item.node_name}".replace("//", "/")
                for item in self.get_publishers_info_by_topic(topic)
                if item.node_name != self.get_name()
            )
            if names:
                conflicts[topic] = names
        return conflicts

    def verify_consumers(self, *, include_lower: bool) -> None:
        topics = list(PRIMARY_COMMAND_TOPICS)
        if include_lower:
            topics.extend((NECK_COMMAND_TOPIC, LEG_COMMAND_TOPIC))
        missing = [topic for topic in topics if not self.get_subscriptions_info_by_topic(topic)]
        if missing:
            raise RuntimeError(f"command topics have no consumers: {missing}")

    def create_command_publishers(self, *, include_lower: bool) -> None:
        if self._arm_publisher is not None:
            raise RuntimeError("command publishers already exist")
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self._arm_publisher = self.create_publisher(JointCommand, ARM_COMMAND_TOPIC, command_qos)
        self._hand_publisher = self.create_publisher(JointState, HAND_COMMAND_TOPIC, command_qos)
        self._waist_publisher = self.create_publisher(RosMsgWrapper, WAIST_COMMAND_TOPIC, command_qos)
        if include_lower:
            self._neck_publisher = self.create_publisher(JointState, NECK_COMMAND_TOPIC, best_effort_qos)
            self._leg_publisher = self.create_publisher(JointState, LEG_COMMAND_TOPIC, best_effort_qos)

    def publish_pose(
        self,
        pose: Pose,
        *,
        arm_template: ArmTemplate,
        hand_template: HandTemplate,
        waist_height: float,
        lower_lock: LowerLock | None,
        side_scope: A3SideScope | None = None,
    ) -> Pose:
        if side_scope is not None:
            pose = side_scope.hold_pose(pose, self.measured_pose())
        validate_pose(pose)
        validate_arm_template(arm_template)
        validate_hand_template(hand_template)
        validate_waist_height(waist_height)
        if self._arm_publisher is None or self._hand_publisher is None or self._waist_publisher is None:
            raise RuntimeError("command publishers are not enabled")
        self._sequence += 1
        stamp = self.get_clock().now().to_msg()

        arm_message = JointCommand()
        arm_message.header.stamp = stamp
        arm_message.header.frame_id = "openpi_a3_deployment"
        arm_message.joints = []
        for index, (name, position) in enumerate(zip(ARM_NAMES, pose.arm, strict=True)):
            joint = Command()
            joint.name = name
            joint.sequence = self._sequence
            joint.position = float(position)
            joint.velocity = 0.0
            joint.effort = 0.0
            joint.stiffness = float(arm_template.stiffness[index])
            joint.damping = float(arm_template.damping[index])
            arm_message.joints.append(joint)
        self._arm_publisher.publish(arm_message)

        hand_message = self._make_hand_command(pose, hand_template, stamp)
        self._hand_publisher.publish(hand_message)

        waist_message = RosMsgWrapper()
        waist_message.serialization_type = "pb"
        waist_message.context = [
            "aimrt-pub_ts",
            str(time.time_ns()),
            "aimrt-pub_seq",
            str(self._sequence),
            "aimrt-serialization_type",
            "pb",
        ]
        payload = encode_waist_payload(pose.waist, waist_height)
        waist_message.data = [bytes([byte]) for byte in payload]
        self._waist_publisher.publish(waist_message)

        if lower_lock is not None:
            if self._neck_publisher is None or self._leg_publisher is None:
                raise RuntimeError("lower lock publishers are not enabled")
            self._neck_publisher.publish(self._make_joint_state(lower_lock.neck, stamp))
            self._leg_publisher.publish(self._make_joint_state(lower_lock.leg, stamp))
        return pose

    @staticmethod
    def _make_joint_state(locked: LockedJoints, stamp: Any) -> JointState:
        message = JointState()
        message.header.stamp = copy.deepcopy(stamp)
        message.name = list(locked.names)
        message.position = locked.positions.tolist()
        message.velocity = np.zeros(len(locked.names), dtype=np.float64).tolist()
        message.effort = np.zeros(len(locked.names), dtype=np.float64).tolist()
        return message

    @staticmethod
    def _make_hand_command(pose: Pose, template: HandTemplate, stamp: Any) -> JointState:
        validate_hand_template(template)
        message = JointState()
        message.header.stamp = copy.deepcopy(stamp)
        message.header.frame_id = template.frame_id
        message.name = list(template.names)
        message.position = np.clip(pose.hand, 0.0, 4095.0).tolist()
        message.velocity = template.velocity.tolist()
        message.effort = template.effort.tolist()
        return message
