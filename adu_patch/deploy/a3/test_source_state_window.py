from types import SimpleNamespace

from deploy.a3.source_state_window import SCHEMA
from deploy.a3.source_state_window import SourceStateWindow


def message(timestamp_ns: int):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=timestamp_ns // 1_000_000_000, nanosec=timestamp_ns % 1_000_000_000)
        )
    )


def test_source_window_maps_header_to_monotonic_and_is_bounded():
    window = SourceStateWindow()
    for index in range(100):
        stamp_ns = 10_000_000_000 + index * 10_000_000
        window.add(
            "arm",
            message(stamp_ns),
            [0.0] * 14,
            monotonic_ns=stamp_ns - 2_000_000_000,
            wall_ns=stamp_ns + 1_000_000,
        )

    snapshot = window.snapshot()
    rows = snapshot["groups"]["arm"]
    assert snapshot["schema"] == SCHEMA
    assert len(rows) <= 31
    assert rows[-1]["source_monotonic_ns"] == stamp_ns - 2_001_000_000


def test_source_window_records_missing_header_and_clock_step():
    window = SourceStateWindow()
    zero = message(0)
    window.add("arm", zero, [0.0] * 14, monotonic_ns=1_000_000, wall_ns=2_000_000)
    assert "arm" in window.snapshot()["errors"]
    window.add("arm", zero, [0.0] * 14, monotonic_ns=2_000_000, wall_ns=10_000_000)
    assert "clock" in window.snapshot()["errors"]
