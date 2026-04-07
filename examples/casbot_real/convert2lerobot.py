"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

# example usage:
# export HF_LEROBOT_HOME=/opt/lyx/iros_datasets
# python examples/aloha_real/convert_aloha_data_to_lerobot.py --raw_dir /opt/lyx/xval_hdf5 --repo_id xvla

import os
import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", '/opt/lyx/iros_datasets')).expanduser()
print(HF_LEROBOT_HOME)

# ===== valid joints (24-dim) =====
ALL_JOINTS = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 
'left_elbow_pitch_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint',
 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_wrist_yaw_joint',
   'right_wrist_pitch_joint', 'right_wrist_roll_joint', 'head_yaw_joint',
    'head_pitch_joint', 'left_thumb_metacarpal_joint', 'left_thumb_proximal_joint',
     'left_thumb_distal_joint', 'left_index_proximal_joint', 'left_index_distal_joint',
      'left_middle_proximal_joint', 'left_middle_distal_joint', 
      'left_ring_proximal_joint', 'left_ring_distal_joint', 'left_pinky_proximal_joint', 
      'left_pinky_distal_joint', 'right_thumb_metacarpal_joint',
       'right_thumb_proximal_joint', 'right_thumb_distal_joint', 
       'right_index_proximal_joint', 'right_index_distal_joint',
        'right_middle_proximal_joint', 'right_middle_distal_joint', 
        'right_ring_proximal_joint', 'right_ring_distal_joint',
         'right_pinky_proximal_joint', 'right_pinky_distal_joint']

"""VALID_JOINTS = ['left_shoulder_pitch_joint', 
                'left_shoulder_roll_joint', 
                'left_shoulder_yaw_joint', 
                'left_elbow_pitch_joint', 
                'left_wrist_yaw_joint', 
                'left_wrist_pitch_joint',
                'left_wrist_roll_joint',
                'right_shoulder_pitch_joint', 
                'right_shoulder_roll_joint',
                'right_shoulder_yaw_joint', 
                'right_elbow_pitch_joint', 
                'right_wrist_yaw_joint',
                'right_wrist_pitch_joint', 
                'right_wrist_roll_joint', 
                'head_yaw_joint',
                'head_pitch_joint', 

                'left_thumb_metacarpal_joint', 
                'left_thumb_proximal_joint',
                'left_index_proximal_joint',
                'left_middle_proximal_joint',
                'left_ring_proximal_joint',
                'left_pinky_proximal_joint',
                
                'right_thumb_metacarpal_joint',
                'right_thumb_proximal_joint', 
                'right_index_proximal_joint',
                'right_middle_proximal_joint',
                'right_ring_proximal_joint',
                'right_pinky_proximal_joint',
                ]
"""

VALID_JOINTS = ['left_shoulder_pitch_joint', 
                'left_shoulder_roll_joint', 
                'left_shoulder_yaw_joint', 
                'left_elbow_pitch_joint', 
                'left_wrist_yaw_joint', 
                'left_wrist_pitch_joint',
                'left_wrist_roll_joint',
                'right_shoulder_pitch_joint', 
                'right_shoulder_roll_joint',
                'right_shoulder_yaw_joint', 
                'right_elbow_pitch_joint', 
                'right_wrist_yaw_joint',
                'right_wrist_pitch_joint', 
                'right_wrist_roll_joint', 
                'head_yaw_joint',
                'head_pitch_joint', 
                'left_thumb_metacarpal_joint', 
                'left_thumb_proximal_joint',
                # 'left_thumb_distal_joint', 
                'left_index_proximal_joint', 
                # 'left_index_distal_joint',
                
                'right_thumb_metacarpal_joint',
                'right_thumb_proximal_joint', 
                # 'right_thumb_distal_joint', 
                'right_index_proximal_joint', 
                # 'right_index_distal_joint',
                ]

ALL_STATES = ['LJ0', 'LJ1', 'LJ2', 'LJ3', 'LJUP', 'LJDOWN', 'LJPITCH', 'LJROLL', 
   'RJ6', 'RJ7', 'RJ8', 'RJ9', 'RJUP', 'RJDOWN', 'RJPITCH', 'RJROLL', 
   'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 
   'left_elbow_pitch', 'left_wrist_yaw', 'left_wrist_pitch', 'left_wrist_roll', 
   'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
    'right_elbow_pitch', 'right_wrist_yaw', 'right_wrist_pitch', 'right_wrist_roll', 
    'head_yaw', 'head_pitch', 'waist_yaw', 'left_thumb_metacarpal', 'left_thumb_proximal', 
    'left_index_proximal', 'left_index_distal', 'left_middle_proximal', 
    'left_middle_distal', 'left_ring_proximal', 'left_ring_distal', 'left_pinky_proximal',
    'left_pinky_distal', 'right_thumb_metacarpal', 'right_thumb_proximal', 
    'right_index_proximal', 'right_index_distal', 'right_middle_proximal', 
    'right_middle_distal', 'right_ring_proximal', 
    'right_ring_distal', 'right_pinky_proximal', 'right_pinky_distal']

VALID_STATES = [state[:-6] for state in VALID_JOINTS if state[:-6] in ALL_STATES]
ACTION_DIM = len(VALID_JOINTS)

import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [f"joint_{i}" for i in range(ACTION_DIM)]
    cameras = [
        "casbot_camera_chest_color",
        "casbot_camera_stereo_color",
        # "cam_low",
        # "cam_left_wrist",
        # "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (360, 640, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2
            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                buf = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"cv2.imdecode returned None for camera={camera}")
                imgs_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        raw_state = torch.from_numpy(ep["/observations/qpos"][:])
        raw_action = torch.from_numpy(ep["/actions"][:])

        # ===== slice valid 38-dim joint states =====
        # ===== hard slice according to confirmed indices =====
        def get_joint_indices(full_list, valid_list):
            """获取valid_list中每个关节在full_list中的索引"""
            
            # 创建名称到索引的映射
            name_to_index = {name: idx for idx, name in enumerate(full_list)}
            
            # 获取有效关节的索引
            indices = []
            
            for joint in valid_list:
                if joint in name_to_index:
                    indices.append(name_to_index[joint])
            
            return indices
        
        """state = torch.cat([
            raw_state[:, 16:16+16],
            raw_state[:, 16+17:16+21],
            raw_state[:, -10:-6],
        ], dim=1)

        action = torch.cat([
            raw_action[:, :18],
            raw_action[:, 19:21],
            raw_action[:, -11:-9],
            raw_action[:, -8:-6],
        ], dim=1)"""
        joint_indices = get_joint_indices(ALL_JOINTS, VALID_JOINTS)
        print(f"动作索引: {joint_indices}")
        action = raw_action[:, joint_indices]  # 直接根据关节索引选择
        state_indices = get_joint_indices(ALL_STATES, VALID_STATES)
        print(f"状态索引: {state_indices}")
        state = raw_state[:, state_indices]

        assert state.shape[1] == action.shape[1] == ACTION_DIM
        
        task = None
        if 'language_instruction' in ep:
            task = str(ep['language_instruction'].asstr()[()])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "casbot_camera_chest_color",
                "casbot_camera_stereo_color",
                # "cam_low",
                # "cam_left_wrist",
                # "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort, task


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    use_filter: bool = False
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort, task_origin = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        def compute_action_start_end(actions, threshold=0.02):
            base = actions[-1]
            for i in range(1, len(actions)):
                if np.linalg.norm(actions[-i] - base) > threshold:
                    return -i
            return -1
        
        def compute_action_start(actions, threshold=0.02):
            base = actions[0]
            for i in range(1, len(actions)):
                if np.linalg.norm(actions[i] - base) > threshold:
                    return i-1
            return 0
        
        if use_filter:
            end = num_frames + compute_action_start_end(action) + 1
            start = compute_action_start(action)
            print(f'episode_{ep_idx} num frames: {end - start}')
        else:
            start = 0
            end = num_frames

        for i in range(start, end):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            frame["task"] = task if task_origin is None else task_origin
            assert frame["task"] is not None

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "fold the cloth", # stack two insoles onto the third insole ; fold the towel
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    use_filter: bool = False,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        print('exist!')
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        return

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
        use_filter=use_filter,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
