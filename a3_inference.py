#!/usr/bin/env python3
"""
cd /data/lyx/casbot_openpi

HF_HOME=/data/lyx/hf_cache \
HF_DATASETS_CACHE=/data/lyx/hf_cache/datasets \
HF_LEROBOT_HOME=/data/lyx/dataset/lerobot \
WANDB_MODE=disabled \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python a3_inference.py --num-steps 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from openpi.policies import policy_config
from openpi.training import config as training_config

DEFAULT_DATASET_ROOT = Path("/data/lyx/dataset/lerobot/local_dataset/a3_0819_both")
DEFAULT_CHECKPOINT = Path("/data/lyx/casbot_openpi/checkpoints/pi05_a3_both/pi05_a3_both_stack_cups_chunk50/29999")


def load_action_stats(checkpoint: Path, repo_id: str) -> dict[str, np.ndarray] | None:
    stats_path = checkpoint / "assets" / repo_id / "norm_stats.json"
    if not stats_path.is_file():
        return None

    raw_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    action_stats = raw_stats.get("norm_stats", raw_stats).get("actions")
    if action_stats is None:
        return None

    return {
        key: np.asarray(action_stats[key], dtype=np.float32)
        for key in ("mean", "std", "q01", "q99")
        if key in action_stats
    }


def load_joint_names(dataset_root: Path) -> list[str]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return list(info["features"]["action"]["names"][0])


def load_task(dataset_root: Path, task_index: int) -> str:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if int(record["task_index"]) == task_index:
                return str(record["task"])
    raise KeyError(f"task_index {task_index} not found in {tasks_path}")


def load_episode_table(dataset_root: Path, episode_index: int):
    parquet_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    return pq.read_table(parquet_path).to_pandas()


def read_video_frame(dataset_root: Path, episode_index: int, frame_index: int) -> np.ndarray:
    video_path = (
        dataset_root
        / "videos"
        / "chunk-000"
        / "observation.images.chest_front_rgb"
        / f"episode_{episode_index:06d}.mp4"
    )
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    try:
        return read_video_frame_pyav(video_path, frame_index)
    except Exception as pyav_error:
        try:
            return read_video_frame_opencv(video_path, frame_index)
        except Exception as opencv_error:
            raise RuntimeError(
                f"Failed to read frame {frame_index} from {video_path}. "
                f"PyAV error: {pyav_error}. OpenCV error: {opencv_error}"
            ) from opencv_error


def read_video_frame_pyav(video_path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(video_path)) as container:
        for current_index, frame in enumerate(container.decode(video=0)):
            if current_index < frame_index:
                continue
            return frame.to_ndarray(format="rgb24")

    raise RuntimeError(f"PyAV reached end of video before frame {frame_index}")


def read_video_frame_opencv(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
    finally:
        cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"OpenCV failed to read frame {frame_index}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def value_relative_error(abs_error: np.ndarray, target: np.ndarray, min_abs_target: float) -> np.ndarray:
    rel = np.full_like(abs_error, np.nan, dtype=np.float32)
    valid = np.abs(target) >= min_abs_target
    rel[valid] = abs_error[valid] / np.abs(target[valid])
    return rel


def range_relative_error(
    abs_error: np.ndarray,
    action_stats: dict[str, np.ndarray] | None,
    min_range: float,
) -> np.ndarray | None:
    if action_stats is None or "q01" not in action_stats or "q99" not in action_stats:
        return None
    scale = action_stats["q99"] - action_stats["q01"]
    rel = np.full_like(abs_error, np.nan, dtype=np.float32)
    valid = scale >= min_range
    rel[..., valid] = abs_error[..., valid] / scale[valid]
    return rel


def print_error_table(
    *,
    joint_names: list[str],
    gt: np.ndarray,
    pred: np.ndarray,
    abs_err: np.ndarray,
    value_rel_err: np.ndarray,
    range_rel_err: np.ndarray | None,
) -> None:
    print("\nPer-joint error:")
    if range_rel_err is None:
        print(f"{'idx':>3}  {'joint':<34} {'gt':>12} {'pred':>12} {'abs_err':>12} {'value_rel%':>11}")
    else:
        print(
            f"{'idx':>3}  {'joint':<34} {'gt':>12} {'pred':>12} {'abs_err':>12} {'value_rel%':>11} {'range_rel%':>11}"
        )
    for idx, name in enumerate(joint_names):
        value_rel = "n/a" if np.isnan(value_rel_err[idx]) else f"{value_rel_err[idx] * 100:>10.4f}"
        if range_rel_err is None:
            print(f"{idx:>3}  {name:<34} {gt[idx]:>12.6f} {pred[idx]:>12.6f} {abs_err[idx]:>12.6f} {value_rel:>11}")
        else:
            range_rel = "n/a" if np.isnan(range_rel_err[idx]) else f"{range_rel_err[idx] * 100:>10.4f}"
            print(
                f"{idx:>3}  {name:<34} "
                f"{gt[idx]:>12.6f} {pred[idx]:>12.6f} {abs_err[idx]:>12.6f} "
                f"{value_rel:>11} {range_rel:>11}"
            )


def summarize_errors(
    abs_errors: np.ndarray,
    value_rel_errors: np.ndarray,
    range_rel_errors: np.ndarray | None,
) -> None:
    print("\nSummary:")
    print(f"  mean_abs_error: {abs_errors.mean():.6f}")
    print(f"  max_abs_error:  {abs_errors.max():.6f}")
    print(f"  mean_value_rel_error: {np.nanmean(value_rel_errors) * 100:.4f}%")
    print(f"  max_value_rel_error:  {np.nanmax(value_rel_errors) * 100:.4f}%")
    if range_rel_errors is not None:
        print(f"  mean_range_rel_error: {np.nanmean(range_rel_errors) * 100:.4f}%")
        print(f"  max_range_rel_error:  {np.nanmax(range_rel_errors) * 100:.4f}%")

    if abs_errors.shape[-1] == 37:
        arm_indices = list(range(14))
        hand_indices = list(range(14, 34))
        waist_indices = list(range(34, 37))
        arm_abs = abs_errors[..., arm_indices]
        hand_abs = abs_errors[..., hand_indices]
        waist_abs = abs_errors[..., waist_indices]
        print(f"  arm_mean_abs_error_rad:       {arm_abs.mean():.6f}")
        print(f"  arm_mean_abs_error_deg:       {np.rad2deg(arm_abs).mean():.4f}")
        print(f"  arm_max_abs_error_deg:        {np.rad2deg(arm_abs).max():.4f}")
        print(f"  hand_mean_abs_error_raw:      {hand_abs.mean():.6f}")
        print(f"  hand_mean_abs_error_pct4095:  {hand_abs.mean() / 4095.0 * 100:.4f}%")
        print(f"  hand_max_abs_error_pct4095:   {hand_abs.max() / 4095.0 * 100:.4f}%")
        print(f"  waist_mean_abs_error_rad:     {waist_abs.mean():.6f}")
        print(f"  waist_mean_abs_error_deg:     {np.rad2deg(waist_abs).mean():.4f}")
        print(f"  waist_max_abs_error_deg:      {np.rad2deg(waist_abs).max():.4f}")
    else:
        print("  grouped_mean_abs_error: skipped; expected 37 action dims")


def run(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    joint_names = load_joint_names(dataset_root)
    table = load_episode_table(dataset_root, args.episode_index)
    if args.num_frames < 1:
        raise ValueError("--num-frames must be >= 1")

    frame_rows = table.iloc[args.start_frame : args.start_frame + args.num_frames]
    if len(frame_rows) < args.num_frames:
        raise ValueError(
            f"Episode {args.episode_index} has only {len(table)} frames; "
            f"cannot read {args.num_frames} frames from start frame {args.start_frame}."
        )

    train_config = training_config.get_config(args.config_name)
    repo_id = train_config.data.repo_id
    action_stats = load_action_stats(checkpoint, repo_id)
    if action_stats is None:
        print(f"Warning: could not load action q01/q99 from {checkpoint / 'assets' / repo_id / 'norm_stats.json'}")
    print(f"Loading policy config={args.config_name} checkpoint={checkpoint}", flush=True)
    first_row = frame_rows.iloc[0]
    first_task = args.prompt or load_task(dataset_root, int(first_row["task_index"]))
    print(
        "Evaluating dataset sample "
        f"episode={int(first_row['episode_index'])} frame={int(first_row['frame_index'])} "
        f"task={first_task!r} num_frames={args.num_frames}",
        flush=True,
    )
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        default_prompt=args.prompt,
        sample_kwargs={"num_steps": args.num_steps},
        normfix=args.normfix,
    )
    print("Policy loaded. Running inference...", flush=True)

    pred_actions: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []

    for _, row in frame_rows.iterrows():
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        task = args.prompt or load_task(dataset_root, int(row["task_index"]))
        image = read_video_frame(dataset_root, episode_index, frame_index)
        state = np.asarray(row["observation.state"], dtype=np.float32)
        gt_action = np.asarray(row["action"], dtype=np.float32)

        obs = {
            "images": {"chest_rgb": image},
            "state": state,
            "prompt": task,
        }
        print(f"Running frame={frame_index} task={task!r}", flush=True)
        pred_chunk = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
        pred_action = pred_chunk[0]

        pred_actions.append(pred_action)
        gt_actions.append(gt_action)
        print(f"frame={frame_index} task={task!r} infer_action_shape={pred_chunk.shape}")

    pred = np.stack(pred_actions)
    gt = np.stack(gt_actions)
    abs_err = np.abs(pred - gt)
    value_rel_err = value_relative_error(abs_err, gt, args.value_relative_min)
    range_rel_err = range_relative_error(abs_err, action_stats, args.range_relative_min)

    if args.num_frames == 1:
        print_error_table(
            joint_names=joint_names,
            gt=gt[0],
            pred=pred[0],
            abs_err=abs_err[0],
            value_rel_err=value_rel_err[0],
            range_rel_err=None if range_rel_err is None else range_rel_err[0],
        )
    else:
        print("\nMean per-joint error over selected frames:")
        print_error_table(
            joint_names=joint_names,
            gt=gt.mean(axis=0),
            pred=pred.mean(axis=0),
            abs_err=abs_err.mean(axis=0),
            value_rel_err=np.nanmean(value_rel_err, axis=0),
            range_rel_err=None if range_rel_err is None else range_rel_err.mean(axis=0),
        )
    summarize_errors(abs_err, value_rel_err, range_rel_err)

    if args.save_npz:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            pred_actions=pred,
            gt_actions=gt,
            abs_errors=abs_err,
            value_rel_errors=value_rel_err,
            range_rel_errors=np.asarray([]) if range_rel_err is None else range_rel_err,
            joint_names=np.asarray(joint_names),
        )
        print(f"\nSaved arrays: {args.save_npz}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default="pi05_a3_both")
    parser.add_argument("--normfix", action="store_true", help="Use the 0902 normfix training normalization rules.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--prompt", default=None, help="Override task prompt. Defaults to the dataset task.")
    parser.add_argument("--num-steps", type=int, default=10, help="Diffusion denoising steps for policy inference.")
    parser.add_argument(
        "--value-relative-min",
        type=float,
        default=1e-3,
        help="Hide value-relative error as n/a when abs(gt) is below this value.",
    )
    parser.add_argument(
        "--range-relative-min",
        type=float,
        default=1e-6,
        help="Show range-relative error as n/a when q99-q01 is below this value.",
    )
    parser.add_argument("--save-npz", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
