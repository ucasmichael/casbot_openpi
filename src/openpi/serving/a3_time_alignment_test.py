import dataclasses

import numpy as np
import pytest

from openpi.serving import a3_rgb_source
from openpi.serving import a3_time_alignment


def alignment_fixture():
    now_ns = 10_000_000_000
    wall_ns = 100_000_000_000
    actor_offset_ns = 2_000_000_000
    rgb_source_ns = now_ns - 33_000_000
    frame = a3_rgb_source.RGBFrame(
        host_wall_ns=wall_ns,
        host_monotonic_ns=now_ns,
        frame_id=17,
        device_timestamp_ms=(wall_ns - 33_000_000) / 1_000_000.0,
        timestamp_domain="global_time",
        color=np.full((4, 6, 3), 23, dtype=np.uint8),
    )
    actor_target_ns = rgb_source_ns - actor_offset_ns
    groups = {}
    for name, size, marker in (("arm", 14, 1.0), ("hand", 20, 100.0), ("waist", 3, 0.1)):
        groups[name] = [
            {
                "header_timestamp_ns": wall_ns - actor_offset_ns - 33_000_000 + delta_ns,
                "source_monotonic_ns": actor_target_ns + delta_ns,
                "receive_monotonic_ns": actor_target_ns + delta_ns + 1_000_000,
                "position": [marker + index for index in range(size)],
            }
            for delta_ns in (-4_000_000, 6_000_000, 16_000_000)
        ]
    window = {
        "schema": a3_time_alignment.SOURCE_WINDOW_SCHEMA,
        "errors": {},
        "groups": groups,
    }
    clock = a3_time_alignment.ClockSync(
        server_minus_actor_monotonic_ns=actor_offset_ns,
        monotonic_uncertainty_ns=1_000_000,
        probe_count=7,
        received_monotonic_ns=now_ns,
        sample_monotonic_ns=now_ns,
    )
    return now_ns, frame, window, clock


def test_selects_source_aligned_rgb_and_preserves_pi05_state_order():
    now_ns, frame, window, clock = alignment_fixture()

    result = a3_time_alignment.select_aligned_frame([frame], window, clock, now_ns=now_ns)

    assert result.frame.frame_id == 17
    np.testing.assert_array_equal(result.state[:14], np.arange(1, 15, dtype=np.float32))
    np.testing.assert_array_equal(result.state[14:34], np.arange(100, 120, dtype=np.float32))
    np.testing.assert_allclose(result.state[34:37], np.asarray([0.1, 1.1, 2.1], dtype=np.float32))
    assert result.metadata["source_max_abs_skew_ms"] == pytest.approx(4.0)
    assert result.metadata["source_max_error_bound_ms"] == pytest.approx(5.0)
    assert result.metadata["camera_delivery_ms"] == pytest.approx(33.0)


def test_rejects_unknown_realsense_clock_domain():
    now_ns, frame, window, clock = alignment_fixture()

    with pytest.raises(a3_time_alignment.SourceAlignmentError, match="global_time"):
        a3_time_alignment.select_aligned_frame(
            [dataclasses.replace(frame, timestamp_domain="hardware_clock")],
            window,
            clock,
            now_ns=now_ns,
        )


def test_rejects_unbracketed_component_and_excess_clock_uncertainty():
    now_ns, frame, window, clock = alignment_fixture()
    window["groups"]["hand"] = window["groups"]["hand"][1:]
    with pytest.raises(a3_time_alignment.SourceAlignmentError, match="no recent D455"):
        a3_time_alignment.select_aligned_frame([frame], window, clock, now_ns=now_ns)

    _, frame, window, clock = alignment_fixture()
    with pytest.raises(a3_time_alignment.SourceAlignmentError, match="uncertainty"):
        a3_time_alignment.select_aligned_frame(
            [frame],
            window,
            dataclasses.replace(clock, monotonic_uncertainty_ns=6_000_000),
            now_ns=now_ns,
        )


def test_rejects_non_monotonic_and_wrong_shaped_state_samples():
    for failure in ("stamp", "shape"):
        now_ns, frame, window, clock = alignment_fixture()
        if failure == "stamp":
            window["groups"]["arm"][1]["source_monotonic_ns"] = window["groups"]["arm"][0]["source_monotonic_ns"]
        else:
            window["groups"]["arm"][0]["position"] = [0.0]
        with pytest.raises(a3_time_alignment.SourceAlignmentError):
            a3_time_alignment.select_aligned_frame([frame], window, clock, now_ns=now_ns)
