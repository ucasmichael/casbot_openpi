"""Regression checks for IPC/MCAP semantics, calibrated timing and lossless output."""

import dataclasses
import json
import sqlite3
import struct

import convert_a3_ipc_to_lerobot as ipc
import h5py
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def episode(tmp_path):
    episode = tmp_path / "session" / "episode_001"
    episode.mkdir(parents=True)
    camera = {
        "name": "RealSense D455",
        "rgb_format": "rgb8",
        "depth_format": "z16",
        "depth_scale_m": 0.001,
        "aligned_depth_intrinsics": "color",
        "color": {"height": 6, "width": 8},
    }
    ipc.write_json(episode / "camera.json", camera)
    ipc.write_json(
        episode / "metadata.json",
        {
            "format": "egodex.ipc_collection.v1",
            "status": "complete",
            "quality_passed": True,
            "stored_frames": 3,
            "task_name": "move_cup",
            "prompt": "",
        },
    )
    origin = 1_790_000_000_000_000_000
    frames = [
        {"frame_index": i, "ipc_time_ns": origin + i * 100_000_000, "depth_ipc_time_ns": origin + i * 100_000_000}
        for i in range(3)
    ]
    (episode / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    rgb = np.zeros((3, 6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 203
    depth = np.broadcast_to(np.array([0, 1, 255, 256, 1000, 4096, 32768, 65535], dtype=np.uint16), (3, 6, 8))
    with h5py.File(episode / "rgbd.h5", "w") as data:
        data.attrs["calibration_json"] = json.dumps(camera)
        data.create_dataset("rgb", data=rgb)
        data.create_dataset("depth_aligned", data=depth)
    with sqlite3.connect(episode / "robot.sqlite3") as db:
        db.execute("CREATE TABLE topics (key TEXT, topic TEXT, type TEXT)")
        db.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, key TEXT, ipc_time_ns INTEGER, "
            "clock_uncertainty_ns INTEGER, fields_json TEXT)"
        )
        for key, (topic, typ) in ipc.TOPICS.items():
            db.execute("INSERT INTO topics VALUES (?, ?, ?)", (key, topic, typ))
            offsets = [-10, 10, 90, 110, 190, 210] if key in ipc.COMMAND_KEYS else [0, 100, 200]
            for index, offset in enumerate(offsets):
                if key == "arm_command":
                    fields = {
                        "joints": [{"name": name, "position": i + index / 10} for i, name in enumerate(ipc.ARM_NAMES)][
                            ::-1
                        ]
                    }
                elif key == "move_waist":
                    # Protobuf fields 2/3/4 are pitch/roll/yaw; output must be roll/pitch/yaw.
                    fields = {
                        "data": list(
                            b"\x11"
                            + struct.pack("<d", 0.2)
                            + b"\x19"
                            + struct.pack("<d", 0.1)
                            + b"\x21"
                            + struct.pack("<d", 0.3)
                        )
                    }
                elif key == "hand_command":
                    fields = {"name": [], "position": list(range(1000 + index, 1020 + index))}
                elif key == "arm_state":
                    fields = {"name": ipc.ARM_NAMES[::-1], "position": (np.arange(14) + 0.05).tolist()[::-1]}
                elif key == "hand_state":
                    names = ["left_thumb_roration_0", *ipc.HAND_NAMES[1:]]
                    fields = {"name": names[::-1], "position": [4096, *range(101, 120)][::-1]}
                else:
                    fields = {
                        "name": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                        "position": [0.33, 0.11, 0.22],
                    }
                # Raw ROS header is intentionally in another clock domain and must be ignored.
                fields["header"] = {"stamp": {"sec": 1, "nanosec": 0}}
                db.execute(
                    "INSERT INTO messages (key, ipc_time_ns, clock_uncertainty_ns, fields_json) VALUES (?, ?, ?, ?)",
                    (key, origin + offset * 1_000_000, 100_000, json.dumps(fields)),
                )
    return episode


def test_explicit_command_preserves_legacy_contract(episode):
    plan = ipc.load_episode(episode, ipc.Config(fps=10, state_source="command"))
    assert plan.task == "move cup"
    assert len(plan.mapping) == 3
    expected = np.array([*(np.arange(14) + 0.1), *range(1001, 1021), 0.1, 0.2, 0.3], dtype=np.float32)
    np.testing.assert_array_equal(plan.states[0], expected)
    np.testing.assert_array_equal(plan.actions[0], expected)
    assert plan.states[0] is not plan.actions[0]
    assert plan.mapping[0]["messages"]["arm_command"]["skew_ms"] == 10
    assert plan.rgb_shape == (6, 8, 3)


def test_default_feedback_is_independent_and_named_order_is_preserved(episode):
    plan = ipc.load_episode(episode, ipc.Config(fps=10, command_alignment="previous"))
    expected = np.array([*(np.arange(14) + 0.05), 4095, *range(101, 120), 0.11, 0.22, 0.33], dtype=np.float32)
    np.testing.assert_array_equal(plan.states[0], expected)
    np.testing.assert_array_equal(plan.actions[0][:14], np.arange(14, dtype=np.float32))
    assert plan.mapping[0]["messages"]["arm_command"]["skew_ms"] == -10
    assert plan.report["feedback_hand_clipped_values"] == 3


@pytest.mark.parametrize("missing", ["topic", "messages"])
def test_missing_feedback_warns_and_uses_history_per_component(episode, caplog, missing):
    with sqlite3.connect(episode / "robot.sqlite3") as db:
        db.execute(f"DELETE FROM {missing if missing == 'messages' else 'topics'} WHERE key='arm_state'")
    plan = ipc.load_episode(episode, ipc.Config(fps=10))
    np.testing.assert_array_equal(plan.states[0][:14], np.arange(14, dtype=np.float32))
    np.testing.assert_array_equal(plan.actions[0][:14], np.arange(14, dtype=np.float32) + np.float32(0.1))
    np.testing.assert_array_equal(plan.states[0][-3:], np.array([0.11, 0.22, 0.33], dtype=np.float32))
    assert "WARNING" in caplog.text
    assert "historical command" in caplog.text
    assert plan.report["state_fallback_counts"] == {"arm": 3, "hand": 0, "waist": 0}
    assert plan.mapping[0]["state_messages"]["arm"]["skew_ms"] == -10
    assert plan.mapping[0]["state_messages"]["waist"]["source"] == "feedback"


@pytest.mark.parametrize("bad", ["stale", "uncertain", "invalid"])
def test_unusable_feedback_falls_back(episode, caplog, bad):
    with sqlite3.connect(episode / "robot.sqlite3") as db:
        if bad == "stale":
            db.execute("UPDATE messages SET ipc_time_ns=ipc_time_ns-1000000000 WHERE key='arm_state'")
        elif bad == "uncertain":
            db.execute("UPDATE messages SET clock_uncertainty_ns=6000000 WHERE key='arm_state'")
        else:
            db.execute("UPDATE messages SET fields_json='{}' WHERE key='arm_state'")
    plan = ipc.load_episode(episode, ipc.Config(fps=10))
    assert plan.report["state_fallback_counts"]["arm"] == 3
    assert plan.mapping[0]["state_messages"]["arm"]["feedback_unavailable_reason"]
    assert "historical command" in caplog.text


def test_future_command_cannot_fill_missing_state(episode, caplog):
    with sqlite3.connect(episode / "robot.sqlite3") as db:
        db.execute("DELETE FROM messages WHERE key='arm_state'")
        db.execute("DELETE FROM messages WHERE key='arm_command' AND ipc_time_ns < 1790000000010000000")
    plan = ipc.load_episode(episode, ipc.Config(fps=10, sync_tolerance_ms=20))
    assert plan.report["trim_start"] == 1
    assert plan.report["state_fallback_unavailable_counts"]["arm"] == 1
    assert "no usable historical command" in caplog.text


def test_out_of_range_feedback_hand_falls_back_without_clipping_command(episode, caplog):
    with sqlite3.connect(episode / "robot.sqlite3") as db:
        fields = {"name": ipc.HAND_NAMES, "position": [65280] * 20}
        db.execute("UPDATE messages SET fields_json=? WHERE key='hand_state'", (json.dumps(fields),))
    plan = ipc.load_episode(episode, ipc.Config(fps=10))
    np.testing.assert_array_equal(plan.states[0][14:34], np.arange(1000, 1020, dtype=np.float32))
    assert plan.report["state_fallback_counts"]["hand"] == 3
    assert plan.report["feedback_hand_clipped_values"] == 0
    assert "hand feedback" in caplog.text


def test_stale_tail_trim_and_internal_gap_rejection():
    assert ipc.valid_interval([False, True, True, False, False]) == (1, 3)
    with pytest.raises(ValueError, match="interior"):
        ipc.valid_interval([True, False, True])
    with pytest.raises(ValueError, match="No synchronized"):
        ipc.valid_interval([False])


def test_regular_grid_preserves_elapsed_time(episode):
    plan = ipc.load_episode(episode, ipc.Config(fps=20, image_tolerance_ms=51))
    assert [m["source_frame_index"] for m in plan.mapping] == [0, 1, 1, 2, 2]
    assert plan.report["repeated_source_frames"] == 2
    assert np.diff([m["grid_ipc_time_ns"] for m in plan.mapping]).tolist() == [50_000_000] * 4
    with pytest.raises(ValueError, match="Camera gap"):
        ipc.load_episode(episode, ipc.Config(fps=20))


def test_clock_uncertainty_is_enforced(episode):
    with sqlite3.connect(episode / "robot.sqlite3") as db:
        db.execute("UPDATE messages SET clock_uncertainty_ns=6000000")
    with pytest.raises(ValueError, match="No synchronized"):
        ipc.load_episode(episode, ipc.Config(fps=10))


def test_unknown_duplicate_joints_and_nonfinite_rejected():
    with pytest.raises(ValueError, match="layout"):
        ipc.ordered_positions(["a", "a"], [1, 2], ["a", "b"])
    with pytest.raises(ValueError, match="Nonfinite"):
        ipc.ordered_positions(["a"], [np.nan], ["a"])


def test_depth_scaling_and_nearest_resize(episode):
    camera = ipc.read_json(episode / "camera.json")
    camera["depth_scale_m"] = 0.002
    with h5py.File(episode / "rgbd.h5") as data:
        rgb, depth = ipc.read_images(data, 0, camera, (16, 12))
    assert rgb.size == depth.size == (16, 12)
    assert np.asarray(depth).dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(depth)[0, ::2], [0, 2, 510, 512, 2000, 8192, 65535, 65535])
    assert np.asarray(rgb)[0, 0].tolist() == [203, 0, 0]


def test_lerobot_round_trip_and_source_provenance(episode, tmp_path):
    output = tmp_path / "output"
    report = ipc.convert(episode, "test/ipc", output, config=ipc.Config(fps=10), task="move the paper cup")
    root = output / "test/ipc"
    info = ipc.read_json(root / "meta/info.json")
    assert info["codebase_version"] == "v2.1"
    assert info["features"]["action"]["shape"] == [37]
    assert info["features"]["action"]["names"] == [ipc.JOINT_NAMES]
    assert info["features"][ipc.mcap.CHEST_DEPTH_KEY]["dtype"] == "image"
    table = pq.read_table(root / "data/chunk-000/episode_000000.parquet").to_pydict()
    assert not np.array_equal(table["observation.state"], table["action"])
    np.testing.assert_allclose(np.array(table["observation.state"])[:, -3:], [[0.11, 0.22, 0.33]] * 3)
    np.testing.assert_allclose(table["timestamp"], [0, 0.1, 0.2])
    with Image.open(table[ipc.mcap.CHEST_DEPTH_KEY][0]["path"]) as image:
        np.testing.assert_array_equal(np.asarray(image)[0], [0, 1, 255, 256, 1000, 4096, 32768, 65535])
    assert (root / "meta/source_frames/episode_000000.jsonl").is_file()
    assert report["state_semantics"] == "feedback_with_historical_command_fallback"
    assert ipc.read_json(root / "meta/ipc_conversion.json")["episodes"][0]["episode_index"] == 0


def test_output_cannot_overwrite_input(episode):
    with pytest.raises(ValueError, match="separate from the input"):
        ipc.convert(episode, "session/episode_001", episode.parent.parent, overwrite=True)


def test_invalid_config():
    with pytest.raises(ValueError, match="tolerances"):
        ipc.Config(sync_tolerance_ms=float("nan"))
    with pytest.raises(ValueError, match="max_frames"):
        dataclasses.replace(ipc.Config(), max_frames=0)


def test_collector_quality_flag_does_not_prevent_feedback_fallback(episode, caplog):
    metadata = ipc.read_json(episode / "metadata.json")
    metadata["quality_passed"] = False
    ipc.write_json(episode / "metadata.json", metadata)
    with sqlite3.connect(episode / "robot.sqlite3") as db:
        db.execute("DELETE FROM messages WHERE key='arm_state'")
    plan = ipc.load_episode(episode, ipc.Config(fps=10))
    assert plan.report["collector_quality_passed"] is False
    assert plan.report["state_fallback_counts"]["arm"] == 3
    assert "quality_passed=False" in caplog.text
