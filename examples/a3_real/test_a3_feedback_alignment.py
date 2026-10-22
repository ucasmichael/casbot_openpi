"""Decode actual generated ROS2 MCAPs and compare their state semantics with IPC."""

import json
import sqlite3

import convert_a3_ipc_to_lerobot as ipc
import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest
import test_convert_a3_ipc_to_lerobot as ipc_tests


@pytest.fixture
def episode(tmp_path):
    return ipc_tests.episode.__wrapped__(tmp_path)


mcap = ipc.mcap
HEADER_DEF = """
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def write_mcap(episode, output):
    writer_module = pytest.importorskip("mcap_ros2.writer")
    output.mkdir()
    with (output / "recording.mcap").open("wb") as stream, writer_module.Writer(stream) as writer:
        joint_state = writer.register_msgdef(
            "sensor_msgs/msg/JointState",
            "std_msgs/Header header\nstring[] name\nfloat64[] position\n" + HEADER_DEF,
        )
        arm_command = writer.register_msgdef(
            "joint_msgs/msg/JointCommand",
            "std_msgs/Header header\njoint_msgs/Command[] joints\n"
            + HEADER_DEF
            + "\n"
            + "=" * 80
            + "\nMSG: joint_msgs/Command\nstring name\nfloat64 position\n",
        )
        wrapper = writer.register_msgdef("ros2_plugin_proto/msg/RosMsgWrapper", "uint8[] data\n")
        image = writer.register_msgdef(
            "sensor_msgs/msg/Image",
            "std_msgs/Header header\nuint32 height\nuint32 width\n"
            "string encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n" + HEADER_DEF,
        )
        with sqlite3.connect(episode / "robot.sqlite3") as db:
            available_topics = {r[0] for r in db.execute("SELECT key FROM topics")}
            for key, time_ns, fields_json in db.execute(
                "SELECT key, ipc_time_ns, fields_json FROM messages ORDER BY id"
            ):
                if key not in available_topics:
                    continue
                fields = json.loads(fields_json)
                fields["header"] = {"stamp": {"sec": time_ns // 10**9, "nanosec": time_ns % 10**9}, "frame_id": "test"}
                schema = arm_command if key == "arm_command" else wrapper if key == "move_waist" else joint_state
                writer.write_message(ipc.TOPICS[key][0], schema, fields, log_time=time_ns)
        frames = [json.loads(line) for line in (episode / "frames.jsonl").read_text().splitlines()]
        with h5py.File(episode / "rgbd.h5") as data:
            for index, frame in enumerate(frames):
                for topic, key, encoding, step in [
                    (mcap.RGB_TOPIC, "rgb", "rgb8", 24),
                    (mcap.DEPTH_TOPIC, "depth_aligned", "16UC1", 16),
                ]:
                    time_ns = frame["ipc_time_ns"]
                    writer.write_message(
                        topic,
                        image,
                        {
                            "header": {
                                "stamp": {"sec": time_ns // 10**9, "nanosec": time_ns % 10**9},
                                "frame_id": "test",
                            },
                            "height": 6,
                            "width": 8,
                            "encoding": encoding,
                            "is_bigendian": 0,
                            "step": step,
                            "data": data[key][index].tobytes(),
                        },
                        log_time=time_ns,
                    )
    return output


def load_mcap(path, **kwargs):
    return mcap.load_episode(
        path,
        joint_mode="both",
        sync_tolerance_ms=20,
        max_frames=None,
        projection_trim=mcap.ProjectionTrimConfig(enabled=False),
        **kwargs,
    )


@pytest.mark.parametrize("scenario", ["feedback", "no_feedback", "partial", "stale", "invalid", "future_only"])
def test_mcap_and_ipc_have_identical_feedback_and_fallback(episode, tmp_path, caplog, scenario):
    with sqlite3.connect(episode / "robot.sqlite3") as db:
        if scenario == "no_feedback":
            db.execute("DELETE FROM messages WHERE key IN ('arm_state','hand_state','waist_state')")
        elif scenario in {"partial", "future_only"}:
            db.execute("DELETE FROM messages WHERE key='arm_state'")
            if scenario == "future_only":
                db.execute("DELETE FROM messages WHERE key='arm_command' AND ipc_time_ns < 1790000000010000000")
        elif scenario == "stale":
            db.execute("UPDATE messages SET ipc_time_ns=ipc_time_ns-1000000000 WHERE key='arm_state'")
        elif scenario == "invalid":
            db.execute(
                "UPDATE messages SET fields_json=? WHERE key='arm_state'",
                (json.dumps({"name": ["unknown"] * 14, "position": [0] * 14}),),
            )
    plan = ipc.load_episode(episode, ipc.Config(fps=10, sync_tolerance_ms=20))
    converted = load_mcap(write_mcap(episode, tmp_path / "mcap"))
    np.testing.assert_array_equal([f["observation.state"] for f in converted.frames], plan.states)
    np.testing.assert_array_equal([f["action"] for f in converted.frames], plan.actions)
    assert converted.joint_names == ipc.JOINT_NAMES
    assert converted.state_info["fallback_counts"] == plan.report["state_fallback_counts"]
    if scenario != "feedback":
        assert "historical command" in caplog.text
    if scenario == "future_only":
        assert len(converted.frames) == 2
        assert converted.state_info["fallback_unavailable_counts"]["arm"] == 1


def test_mcap_command_mode_and_previous_action_alignment(episode, tmp_path):
    path = write_mcap(episode, tmp_path / "mcap")
    legacy = load_mcap(path, state_source="command")
    np.testing.assert_array_equal(legacy.frames[0]["observation.state"], legacy.frames[0]["action"])
    feedback = load_mcap(path, command_alignment="previous")
    np.testing.assert_array_equal(feedback.frames[0]["action"][:14], np.arange(14, dtype=np.float32))
    np.testing.assert_allclose(feedback.frames[0]["observation.state"][:14], np.arange(14) + 0.05)


def test_mcap_saves_independent_state_action_and_fallback_report(episode, tmp_path):
    with sqlite3.connect(episode / "robot.sqlite3") as db:
        db.execute("DELETE FROM messages WHERE key='arm_state'")
    path = write_mcap(episode, tmp_path / "mcap")
    mcap.convert(path, "test/mcap", tmp_path / "output", fps=10, projection_trim_enabled=False)
    root = tmp_path / "output/test/mcap"
    table = pq.read_table(root / "data/chunk-000/episode_000000.parquet").to_pydict()
    np.testing.assert_array_equal(table["observation.state"][0][:14], np.arange(14, dtype=np.float32))
    np.testing.assert_allclose(table["action"][0][:14], np.arange(14) + 0.1)
    report = ipc.read_json(root / "meta/state_sources.json")["episodes"][0]
    assert report["fallback_counts"] == {"arm": 3, "hand": 0, "waist": 0}
    assert report["frames"][0]["state_sources"]["arm"] == "historical_command"
    state_mean = ipc.read_json(root / "meta/qpose_means.json")["training_qpose"]["mean"]
    action_mean = ipc.read_json(root / "meta/action_means.json")["mean"]
    assert state_mean != action_mean
