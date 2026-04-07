import os
import cv2
import numpy as np
import dataclasses
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
from mcap_ros2.decoder import Decoder


# =========================
# Dataset config
# =========================

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


# =========================
# Create empty LeRobot dataset (ONLY ONCE)
# =========================

def create_empty_dataset(
    data_info: list,
    robot_type: str,
    save_dir: str,
    mode: str,
    dataset_config: DatasetConfig = DatasetConfig(),
) -> LeRobotDataset:

    image_info, arm_joint_names_state, arm_joint_names_action = data_info

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(arm_joint_names_state),),
            "names": [arm_joint_names_state],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(arm_joint_names_action),),
            "names": [arm_joint_names_action],
        },
    }

    for cam in image_info:
        cam_name, shape = image_info[cam]
        features[f"observation.images.{cam_name}"] = {
            "dtype": mode,
            "shape": shape,
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id="",
        root=save_dir,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


# =========================
# MCAP parsing
# =========================

def mcap2topic_(mcap_filename):
    decoder = Decoder()
    topic_info = {}
    color_image_topic_set = set()

    with open(mcap_filename, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            ros_msg = decoder.decode(schema, message)

            if topic not in topic_info:
                topic_info[topic] = {
                    "message_count": 0,
                    "sec": [],
                    "nanosec": [],
                    "find_sec": {},
                }

            ros_msg_type = ros_msg._type

            if ros_msg_type == "sensor_msgs/CompressedImage":
                fmt = ros_msg.format
                img = cv2.imdecode(np.frombuffer(ros_msg.data, np.uint8), -1)

                if img is not None:
                    if "bgr8" in fmt:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    topic_info[topic].setdefault("color", []).append(img)
                    color_image_topic_set.add(topic)

            elif ros_msg_type == "sensor_msgs/JointState":
                topic_info[topic]["name"] = ros_msg.name
                topic_info[topic].setdefault("position", []).append(ros_msg.position)

            else:
                continue

            sec = ros_msg.header.stamp.sec
            nanosec = ros_msg.header.stamp.nanosec

            topic_info[topic]["find_sec"].setdefault(sec, []).append(
                [nanosec, topic_info[topic]["message_count"]]
            )

            topic_info[topic]["sec"].append(sec)
            topic_info[topic]["nanosec"].append(nanosec)
            topic_info[topic]["message_count"] += 1

    return topic_info, list(color_image_topic_set)


# =========================
# Synchronization
# =========================

def synchronous_all_topic(all_topic_src, color_topic_list):
    min_len = min(all_topic_src[t]["message_count"] for t in color_topic_list)
    min_topic = min(color_topic_list, key=lambda t: all_topic_src[t]["message_count"])

    for t in all_topic_src:
        if all_topic_src[t]["message_count"] < min_len:
            return ["topic hz mismatch", False]

    sync_data = {}

    base = all_topic_src[min_topic]
    sync_data[min_topic] = {"color": [], "time": []}

    for sec, items in base["find_sec"].items():
        for nanosec, idx in items:
            sync_data[min_topic]["color"].append(base["color"][idx])
            sync_data[min_topic]["time"].append(sec * 1e3 + nanosec * 1e-6)

            for t in all_topic_src:
                if t == min_topic:
                    continue

                src = all_topic_src[t]
                if t not in sync_data:
                    sync_data[t] = {}

                if "position" in src:
                    sync_data[t].setdefault("position", []).append(
                        src["position"][idx]
                    )
                    sync_data[t]["name"] = src["name"]

    return [sync_data, min_topic, color_topic_list]


# =========================
# Add ONE episode (ONE mcap)
# =========================

def add_mcap_episode(dataset, mcap_file, task_name):
    all_topic_data, color_topics = mcap2topic_(mcap_file)
    rt = synchronous_all_topic(all_topic_data, color_topics)

    if len(rt) == 2:
        print(f"[Skip] {mcap_file}")
        return

    sync_data, min_cam, color_topics = rt

    joint_state = sync_data["/casbot/joint_states"]
    joint_action = sync_data["/casbot/joint_data"]

    T = len(joint_state["position"])

    for i in range(T):
        frame = {
            "observation.state": np.array(joint_state["position"][i], np.float32),
            "action": np.array(joint_action["position"][i], np.float32),
            "task": task_name,
        }

        for cam in color_topics:
            cam_name = "_".join(cam.split("/")[1:])
            frame[f"observation.images.{cam_name}"] = sync_data[cam]["color"][i]

        dataset.add_frame(frame)

    dataset.save_episode()


# =========================
# Folder-level conversion
# =========================

def convert_mcap_folder(
    mcap_dir,
    save_dir,
    task_name,
    robot_type="casbot",
    mode="image",
):
    mcap_files = sorted(Path(mcap_dir).glob("*.mcap"))
    assert len(mcap_files) > 0, "No mcap files found"

    # infer structure from first mcap
    topic_data, color_topics = mcap2topic_(str(mcap_files[0]))

    image_info = {}
    for cam in color_topics:
        img = topic_data[cam]["color"][0]
        image_info[cam] = (
            "_".join(cam.split("/")[1:]),
            (img.shape[2], img.shape[0], img.shape[1]),
        )

    joint_state = topic_data["/casbot/joint_states"]
    joint_action = topic_data["/casbot/joint_data"]

    dataset = create_empty_dataset(
        [image_info, joint_state["name"], joint_action["name"]],
        robot_type=robot_type,
        save_dir=save_dir,
        mode=mode,
    )

    for mcap in mcap_files:
        print(f"[Convert] {mcap.name}")
        add_mcap_episode(dataset, str(mcap), task_name)

    dataset.consolidate()
    print(f"[Done] Dataset saved to {save_dir}")


# =========================
# Main
# =========================

if __name__ == "__main__":
    convert_mcap_folder(
        mcap_dir="/opt/lyx/dataset/cmap_fold_cloth",
        save_dir="/opt/lyx/iros_datasets/local_dataset/casbot",
        task_name="fold the cloth",
        mode="image",  # or "video"
    )
