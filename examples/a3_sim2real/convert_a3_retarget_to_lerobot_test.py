from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

MODULE_PATH = Path(__file__).with_name("convert_a3_retarget_to_lerobot.py")
MODULE_SPEC = importlib.util.spec_from_file_location("a3_sim2real_converter", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
converter = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = converter
MODULE_SPEC.loader.exec_module(converter)


def write_test_episode(episode_dir: Path) -> None:
    frame_dir = episode_dir / "all_data" / "000000"
    frame_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(frame_dir / "rgb_out.png")
    Image.fromarray(np.zeros((4, 6), dtype=np.uint16), mode="I;16").save(frame_dir / "depth.png")
    (episode_dir / "cfg").mkdir()
    (episode_dir / "cfg" / "session_meta.json").write_text(
        json.dumps({"source_metadata": {"fps": 30}}),
        encoding="utf-8",
    )
    wire = np.zeros(converter.ACTION_DIM, dtype=np.float64)
    payload = {
        "schema": converter.A3_EXECUTABLE_SCHEMA,
        "active_sides": ["right"],
        "a3_qpose_37d_names": list(converter.A3_WIRE_NAMES),
        "frames": [
            {
                "frame": "000000",
                "source_frame_index": 0,
                "timestamp_ns": 1_000_000_000,
                "a3_qpose_37d": wire.tolist(),
            }
        ],
    }
    (episode_dir / "a3_qpose.json").write_text(json.dumps(payload), encoding="utf-8")


def test_wire_to_training_qpose_reorders_waist_after_arm_and_hand() -> None:
    wire = np.arange(converter.ACTION_DIM, dtype=np.float64)
    wire[17:37] = np.arange(20, dtype=np.float64) * 100

    actual = converter.wire_to_training_qpose(wire)

    expected = np.concatenate((wire[3:17], wire[17:37], wire[[1, 2, 0]])).astype(np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_wire_to_training_qpose_rejects_non_executable_hand_value() -> None:
    wire = np.zeros(converter.ACTION_DIM, dtype=np.float64)
    wire[17] = 12.5

    with pytest.raises(ValueError, match="must be integers"):
        converter.wire_to_training_qpose(wire)


def test_load_episode_spec_validates_and_aligns_frame_files(tmp_path) -> None:
    episode_dir = tmp_path / "episode_001"
    write_test_episode(episode_dir)

    episode = converter.load_episode_spec(
        episode_dir,
        rgb_name="rgb_out.png",
        depth_name="depth.png",
        expected_fps=30,
    )

    assert episode.active_sides == ("right",)
    assert episode.source_fps == 30
    assert len(episode.frames) == 1
    assert episode.frames[0].frame == "000000"
    assert episode.frames[0].action.shape == (converter.ACTION_DIM,)
    assert converter.inspect_image_spec(episode.frames[0]) == converter.ImageSpec(
        rgb_shape=(4, 6, 3),
        depth_shape=(4, 6, 1),
    )


@pytest.mark.parametrize(("timestamp_ns", "error"), [(1_000_000_000, None), (999_999_999, "decreases")])
def test_load_episode_spec_allows_equal_but_rejects_decreasing_timestamps(
    tmp_path, timestamp_ns: int, error: str | None
) -> None:
    episode_dir = tmp_path / "episode_001"
    write_test_episode(episode_dir)
    second_frame_dir = episode_dir / "all_data" / "000001"
    second_frame_dir.mkdir()
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(second_frame_dir / "rgb_out.png")
    Image.fromarray(np.zeros((4, 6), dtype=np.uint16), mode="I;16").save(second_frame_dir / "depth.png")

    action_path = episode_dir / "a3_qpose.json"
    payload = json.loads(action_path.read_text(encoding="utf-8"))
    payload["frames"].append(
        {
            "frame": "000001",
            "source_frame_index": 1,
            "timestamp_ns": timestamp_ns,
            "a3_qpose_37d": [0.0] * converter.ACTION_DIM,
        }
    )
    action_path.write_text(json.dumps(payload), encoding="utf-8")

    if error is None:
        episode = converter.load_episode_spec(
            episode_dir,
            rgb_name="rgb_out.png",
            depth_name="depth.png",
            expected_fps=30,
        )
        assert len(episode.frames) == 2
    else:
        with pytest.raises(ValueError, match=error):
            converter.load_episode_spec(
                episode_dir,
                rgb_name="rgb_out.png",
                depth_name="depth.png",
                expected_fps=30,
            )


def test_convert_writes_a_loadable_lerobot_dataset(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    write_test_episode(raw_dir / "episode_001")
    lerobot_root = tmp_path / "lerobot"

    converter.convert(
        raw_dir=raw_dir,
        repo_id="local_dataset/a3_sim2real_test",
        lerobot_root=lerobot_root,
        task="move the paper cup",
    )

    dataset_root = lerobot_root / "local_dataset" / "a3_sim2real_test"
    manifest = json.loads((dataset_root / "meta" / "conversion_manifest.json").read_text())
    qpose_means = json.loads((dataset_root / "meta" / "qpose_means.json").read_text())
    assert manifest["episode_count"] == 1
    assert manifest["frame_count"] == 1
    assert qpose_means["training_qpose"]["dim"] == converter.ACTION_DIM
    assert qpose_means["training_qpose"]["frame_count"] == 1
