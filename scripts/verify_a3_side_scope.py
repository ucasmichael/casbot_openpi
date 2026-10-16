#!/usr/bin/env python3
"""Offline validation for A3 inactive-side input/output scoping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.policies import a3_policy
from openpi.shared import normalize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--active-sides", default="right")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    matches = sorted((checkpoint / "assets").glob("**/norm_stats.json"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one norm_stats.json under {checkpoint / 'assets'}, found {matches}")
    norm_stats = normalize.load(matches[0].parent)
    active_sides = a3_policy.parse_active_sides(args.active_sides)
    assert active_sides is not None
    inactive_sides = a3_policy.validate_inactive_sides_frozen(
        active_sides,
        norm_stats,
        use_quantiles=True,
    )
    state_mean = np.asarray(norm_stats["state"].mean, dtype=np.float32)
    result = {
        "result": "ok",
        "checkpoint": str(checkpoint),
        "norm_stats": str(matches[0]),
        "active_sides": list(active_sides),
        "inactive_sides": list(inactive_sides),
        "inactive_training_state": {
            side: state_mean[a3_policy.A3_SIDE_INDICES[side]].tolist() for side in inactive_sides
        },
        "input_rule": "inactive side uses checkpoint state mean",
        "output_rule": "inactive side uses live request state",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
