"""Replace command observations in an existing refined dataset with PhysX state.

Backs up the small Parquet tables and metadata next to the dataset; RGB files
are left in place. All episodes are validated before any table is replaced.
The backup is rollback-only. Training state is stored in the LeRobot Parquet
tables under ``data/`` as ``observation.state``.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from convert_a3_retarget_to_lerobot import load_json_object
from convert_a3_retarget_to_lerobot import load_physics_states
from convert_a3_retarget_to_lerobot import wire_to_training_qpose
from lerobot.common.datasets.compute_stats import get_feature_stats
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def repair(dataset_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "meta/conversion_manifest.json"
    manifest = load_json_object(manifest_path)
    info = load_json_object(dataset_root / "meta/info.json")
    stats_path = dataset_root / "meta/episodes_stats.jsonl"
    stats_rows = [json.loads(line) for line in stats_path.read_text().splitlines()]
    stats_by_episode = {row["episode_index"]: row for row in stats_rows}
    # Keep rollback material outside the LeRobot dataset. This prevents the
    # original command copy from being mistaken for a trainable state source.
    backup = dataset_root.parent / f"{dataset_root.name}.command_state_backup"
    if backup.exists():
        raise FileExistsError(f"Backup already exists; inspect the previous repair before retrying: {backup}")

    plans = []
    report_rows = []
    all_states = []
    for index, episode in enumerate(manifest["episodes"]):
        source = Path(episode["source"])
        commands = load_json_object(source / "a3_qpose.json")["frames"]
        states = load_physics_states(source, commands)
        relative = Path(info["data_path"].format(
            episode_chunk=index // info["chunks_size"], episode_index=index
        ))
        table = pq.read_table(dataset_root / relative)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        expected_actions = np.asarray([wire_to_training_qpose(row["a3_qpose_37d"]) for row in commands])
        np.testing.assert_array_equal(actions, expected_actions)
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(len(commands)))
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(len(commands), index))
        if len(states) != len(table) or len(table) != episode["frame_count"]:
            raise ValueError(f"Frame count mismatch for episode {index}")
        # Independently verify that render-state joints came from the saved
        # simulation trajectory at these observation timestamps.
        physics = load_json_object(source / "physics_robot_render_state.json")
        trajectory = json.loads(Path(physics["trajectory"]).read_text())
        names = physics["joint_names"]
        if any(row["joint_names"] != names for row in trajectory):
            raise ValueError(f"Physics trajectory joint order mismatch for episode {index}")
        times = np.asarray([row["time_s"] for row in trajectory])
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"Physics trajectory time reversal for episode {index}")
        query = np.asarray([(int(row["timestamp_ns"]) - int(commands[0]["timestamp_ns"])) * 1e-9
                            for row in commands])
        joints = np.asarray([row["joint_positions"] for row in trajectory])
        expected = np.column_stack([np.interp(query, times, joints[:, j]) for j in range(len(names))])
        measured = np.asarray([row["joint_positions"] for row in physics["frames"]])
        np.testing.assert_allclose(measured, expected, rtol=0, atol=5.1e-9)
        column = table.schema.get_field_index("observation.state")
        updated = table.set_column(column, table.schema.field(column),
                                   pa.array(states.tolist(), type=table.schema.field(column).type))
        state_stats = get_feature_stats(states, axis=0, keepdims=False)
        stats_by_episode[index]["stats"]["observation.state"] = {k: v.tolist() for k, v in state_stats.items()}
        report_rows.append({
            "episode_index": index, "source": str(source), "frames": len(states),
            "previous_state_equal_to_action": bool(np.array_equal(
                np.asarray(table["observation.state"].to_pylist(), dtype=np.float32), actions)),
            "measured_state_differs_from_action": bool(not np.array_equal(states, actions)),
            "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "physics_state_sha256": hashlib.sha256((source / "physics_robot_render_state.json").read_bytes()).hexdigest(),
            "max_render_vs_trajectory_error_rad": float(np.abs(measured - expected).max()),
        })
        episode["state_source"] = "physics_robot_render_state.json"
        plans.append((relative, updated))
        all_states.append(states)

    if sum(len(s) for s in all_states) != info["total_frames"]:
        raise ValueError("Dataset total frame count mismatch")
    # Keep a complete, inexpensive rollback of all files changed below.
    backup.mkdir()
    shutil.copytree(dataset_root / "meta", backup / "meta")
    for relative, _ in plans:
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset_root / relative, target)
    for relative, updated in plans:
        path = dataset_root / relative
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(updated, temporary)
        check = pq.read_table(temporary)
        np.testing.assert_array_equal(check["action"].to_pylist(), updated["action"].to_pylist())
        temporary.replace(path)
    temporary_stats = stats_path.with_suffix(".jsonl.tmp")
    temporary_stats.write_text("".join(json.dumps(row) + "\n" for row in stats_rows))
    temporary_stats.replace(stats_path)
    manifest["state_source"] = "physics_robot_render_state.json:frames[].joint_positions"
    manifest["state_storage"] = "data/chunk-*/episode_*.parquet:observation.state"
    manifest["state_hand_encoding"] = "official OmniHand SDK: measured active radians -> rounded/clipped actuator [0,4095]"
    manifest["observation_action_policy"] = "observation.state is synchronized actual PhysX state; action is absolute executable command"
    write_json(manifest_path, manifest)
    means_path = dataset_root / "meta/qpose_means.json"
    means = load_json_object(means_path)
    means["observation_state_mean"] = np.concatenate(all_states).mean(axis=0, dtype=np.float64).tolist()
    write_json(means_path, means)
    report = {
        "schema": "openpi.refined_physics_state_audit.v1", "dataset_root": str(dataset_root),
        "episodes": len(plans), "frames": info["total_frames"],
        "state_source": manifest["state_source"], "action_unchanged": True,
        "state_storage": manifest["state_storage"],
        "rgb_timestamp_alignment_verified": True, "raw_simulation_trajectory_verified": True,
        "backup": str(backup), "episode_audits": report_rows,
    }
    write_json(dataset_root / "meta/physics_state_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "episode_audits"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--omnihand-site-packages", type=Path)
    args = parser.parse_args()
    if args.omnihand_site_packages is not None:
        # Append so the OpenPI environment keeps its own NumPy/PyArrow versions.
        sys.path.append(str(args.omnihand_site_packages))
    repair(args.dataset_root)
