#!/usr/bin/env python3
"""Generate A3 deployment init-pose files from ModelScope qpose trajectories."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import requests
from modelscope.hub.api import HubApi, ModelScopeConfig


REPO_ID = "liyixuan2026/a3_dex_dataset_1010"
SOURCE_SCHEMA = "hand_obj_tracking.a3_executable_qpose37.v1"
TASKS = {
    "move_cup_20260913": "pi05_a3_sim2real_move_cup",
    "hang_blue_cup_20260914": "pi05_a3_sim2real_hang_blue_cup",
    "put_spoon_20260913": "pi05_a3_sim2real_put_spoon",
}
ARM_NAMES = tuple(
    f"{side}_{suffix}_joint"
    for side in ("left", "right")
    for suffix in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
)
HAND_NAMES = tuple(
    f"{side}_{suffix}"
    for side in ("left", "right")
    for suffix in (
        "thumb_rotation_0",
        "thumb_wiggles_1",
        "thumb_bent_2",
        "index_wiggles_0",
        "index_bent_1",
        "middle_bent",
        "ring_wiggles_0",
        "ring_bent_1",
        "pinky_wiggles_0",
        "pinky_bent_1",
    )
)
WIRE_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
) + ARM_NAMES + HAND_NAMES
MODEL_NAMES = ARM_NAMES + HAND_NAMES + ("waist_roll", "waist_pitch", "waist_yaw")
ARM_TEMPLATE = {
    "stiffness": [260.0, 260.0, 117.0, 143.0, 117.0, 52.0, 52.0] * 2,
    "damping": [7.0, 7.0, 5.0, 5.0, 3.5, 2.0, 2.0] * 2,
}
HAND_TEMPLATE = {
    "frame_id": "O10Hand",
    "name": [],
    "velocity": [],
    "effort": [125.0] * 20,
}


def download_json(
    api: HubApi, session: requests.Session, path: str
) -> tuple[dict[str, Any], str]:
    url = api.get_dataset_file_url(path, "a3_dex_dataset_1010", "liyixuan2026")
    response = session.get(url, timeout=60)
    response.raise_for_status()
    content = response.content
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, hashlib.sha256(content).hexdigest()


def wire_to_model(values: Any, source: str) -> list[float]:
    qpose = np.asarray(values, dtype=np.float64)
    if qpose.shape != (37,) or not np.isfinite(qpose).all():
        raise ValueError(f"invalid 37D qpose: {source}")
    hand = qpose[17:37]
    if np.any(hand < 0) or np.any(hand > 4095) or not np.equal(hand, np.rint(hand)).all():
        raise ValueError(f"invalid hand actuator values: {source}")
    # Wire is waist yaw/roll/pitch, arms, hands; deployment is arms, hands, roll/pitch/yaw.
    return np.concatenate((qpose[3:17], hand, qpose[[1, 2, 0]])).tolist()


def medoid(candidates: dict[str, dict[str, Any]]) -> str:
    names = sorted(candidates)
    matrix = np.asarray([candidates[name]["qpose"] for name in names], dtype=np.float64)
    scale = np.concatenate((np.ones(14), np.full(20, 1000.0), np.ones(3)))
    normalized = matrix / scale
    scores = np.asarray(
        [np.linalg.norm(row - normalized, axis=1).sum() for row in normalized]
    )
    return names[int(np.argmin(scores))]


def generate_task(api: HubApi, session: requests.Session, task: str, config: str) -> dict[str, Any]:
    entries = api.get_dataset_files(
        REPO_ID,
        root_path=task,
        recursive=False,
        page_number=1,
        page_size=1000,
    )
    episode_paths = sorted(item["Path"] for item in entries if item["Type"] == "tree")
    if not episode_paths:
        raise ValueError(f"no episodes found for {task}")

    candidates: dict[str, dict[str, Any]] = {}
    for episode_path in episode_paths:
        source_path = f"{episode_path}/a3_qpose.json"
        payload, source_sha256 = download_json(api, session, source_path)
        if payload.get("schema") != SOURCE_SCHEMA:
            raise ValueError(f"unexpected schema in {source_path}")
        if tuple(payload.get("a3_qpose_37d_names", ())) != WIRE_NAMES:
            raise ValueError(f"unexpected joint order in {source_path}")
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"no frames in {source_path}")
        first = frames[0]
        timestamp_ns = int(first.get("timestamp_ns", 0))
        if timestamp_ns <= 0 or not math.isfinite(timestamp_ns):
            raise ValueError(f"invalid first-frame timestamp in {source_path}")
        pose_id = Path(episode_path).name
        candidates[pose_id] = {
            "qpose": wire_to_model(first.get("a3_qpose_37d"), source_path),
            "arm_template": copy.deepcopy(ARM_TEMPLATE),
            "hand_template": copy.deepcopy(HAND_TEMPLATE),
            "waist_height": 0.0,
            "source_bag": f"modelscope://datasets/{REPO_ID}/{source_path}",
            "source_sha256": source_sha256,
            "source_frame": str(first.get("frame", "")),
            "source_frame_index": int(first.get("source_frame_index", -1)),
            "active_sides": list(payload.get("active_sides", [])),
            "rgb_time_ns": timestamp_ns,
            "synchronization_ms": {},
        }

    return {
        "schema_version": 2,
        "model": {
            "config": config,
            "action_dim": 37,
            "action_horizon": 50,
            "names": list(MODEL_NAMES),
        },
        "source": {
            "dataset": f"modelscope://datasets/{REPO_ID}",
            "task_directory": task,
            "selection": "medoid of the first executable qpose frame from every episode",
            "state_equals_action": True,
            "wire_to_model_order": "arms + hands + waist[roll,pitch,yaw]",
            "control_templates_from": "ADU deploy/a3/init_pose.json",
        },
        "default_pose_id": medoid(candidates),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    api = HubApi()
    session = requests.Session()
    session.headers.update(
        {
            "user-agent": ModelScopeConfig.get_user_agent(),
            "snapshot-identifier": "a3-init-pose-generation",
        }
    )
    session.cookies.update(ModelScopeConfig.get_cookies())
    for task, config in TASKS.items():
        output = args.output_root / task / "init_pose_v2.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = generate_task(api, session, task, config)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"wrote {output} candidates={len(result['candidates'])} "
            f"default={result['default_pose_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
