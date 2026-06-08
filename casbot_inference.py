import os
import pyarrow.parquet as pq
import numpy as np
import cv2
from openpi_client import image_tools, websocket_client_policy
from io import BytesIO
from PIL import Image
import einops
from tqdm import tqdm

# ==============================
# 基础配置
# ==============================
DATASET_ROOT = "/opt/lyx/iros_datasets/local_dataset/casbot/data/chunk-000/"
path_list = []
for i in range(60):
    PARQUET_PATH = os.path.join(DATASET_ROOT, f"episode_{i:06d}.parquet")
    path_list.append(PARQUET_PATH)

# 打开 policy client
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

# ==============================
# 读取 parquet 文件
# ==============================
print(f"Loading parquet data from: {DATASET_ROOT}")

actions = None
states = None
cam_high_meta = None
cam_right_meta = None
cam_left_meta = None

for path in tqdm(path_list):
    table = pq.read_table(path).to_pandas()

    # 各个字段
    for keys in ["action", "observation.state", 
                 "observation.images.casbot_camera_chest_color"]:
        if keys == "action":
            if actions is None:
                actions = np.stack(table[keys].to_numpy())
            else:
                actions = np.vstack((actions, np.stack(table[keys].to_numpy())))
        elif keys == "observation.state":
            if states is None:
                states = np.stack(table[keys].to_numpy())
            else:
                states = np.vstack((states, np.stack(table[keys].to_numpy())))
        elif keys == "observation.images.casbot_camera_chest_color":
            if cam_high_meta is None:
                cam_high_meta = table[keys].to_numpy()
            else:
                cam_high_meta = np.concatenate((cam_high_meta, table[keys].to_numpy()), axis=0)
    # actions = np.stack(table["action"].to_numpy()) if actions is not None else  np.stack([actions, np.stack(table["action"].to_numpy())])                   # (num_frames, 14), first dim stack
    # states = np.stack(table["observation.state"].to_numpy())          # (num_frames, 14)
    # cam_high_meta = table["observation.images.cam_high"].to_numpy()
    # cam_right_meta = table["observation.images.cam_right_wrist"].to_numpy()
    # cam_left_meta = table["observation.images.cam_left_wrist"].to_numpy()

print(f"✅ Loaded data: total frames = {len(actions)}")

def load_image_from_bytes(meta_dict):
    """从 parquet 中的 {'bytes': b'...'} 加载为 numpy RGB 图像"""
    if "bytes" in meta_dict and meta_dict["bytes"] is not None:
        img = Image.open(BytesIO(meta_dict["bytes"])).convert("RGB")
        return np.array(img)
    elif "path" in meta_dict:  # 回退到 path
        img_path = os.path.join(DATASET_ROOT, meta_dict["path"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return img
    else:
        raise ValueError("Invalid image metadata: neither 'bytes' nor 'path' present.")

num_steps = len(actions)
print(f"Loaded {num_steps} frames")

# ==============================
# 推理与对比
# ==============================
pred_actions = []

for step in tqdm(range(num_steps)):
    try:
        img = load_image_from_bytes(cam_high_meta[step])
        # left_img = load_image_from_bytes(cam_left_meta[step])
        # right_img = load_image_from_bytes(cam_right_meta[step])
    except Exception as e:
        print(f"[WARN] Failed to load images at step {step}: {e}")
        continue

    state = states[step]

    # ===== 构建 observation（resize 为 224×224 并转为 (3,224,224)）=====
    def preprocess_image(im):
        im_resized = image_tools.resize_with_pad(im, 224, 224)
        im_uint8 = image_tools.convert_to_uint8(im_resized)
        return einops.rearrange(im_uint8, "h w c -> c h w")

    observation = {
        "images": {
            "cam_low": preprocess_image(img),
            # "cam_left_wrist": preprocess_image(left_img),
            # "cam_right_wrist": preprocess_image(right_img),
        },
        "state": state,
        "prompt": "fold the towel",
    }

    # ===== 前向推理 =====
    pred_action_chunk = client.infer(observation)["actions"]
    pred_action = pred_action_chunk[0]
    pred_actions.append(pred_action)

    # ===== 对比 GT 与预测 =====
    gt_action = actions[step]
    diff = np.linalg.norm(pred_action - gt_action)

# ==============================
# 保存结果
# ==============================
pred_actions = np.array(pred_actions)
print("✅ Done. Saved pred_actions.npy & gt_actions.npy.")

import matplotlib.pyplot as plt
import numpy as np

# ==============================
# 计算 L1 误差并绘制
# ==============================
if len(pred_actions) == 0:
    print("⚠️ No predicted actions available — skipping plot.")
else:
    l1_errors = np.abs(pred_actions - actions[:len(pred_actions)])  # shape (N, 14)
    steps = np.arange(len(l1_errors))

    # ===== 打印每个维度的平均 L1 损失 =====
    mean_l1_per_dim = np.mean(l1_errors, axis=0)
    gt_mean_per_dim = np.mean(actions[:len(pred_actions)], axis=0)
    print("\n📊 平均 L1 误差（每个维度）:")
    for i in range(25):
        relative_error = (mean_l1_per_dim[i] / (np.abs(gt_mean_per_dim[i]) + 1e-8)) * 100  # 避免除零
        print(f"  joint_{i} : {mean_l1_per_dim[i]:.6f} , gt : {gt_mean_per_dim[i]:.6f} error rate: {relative_error:.4f}%")

    """# 右臂（前7维：joint1-6, gripper）
    plt.figure(figsize=(10, 5))
    for i in range(7):
        plt.plot(steps, l1_errors[:, i], label=f"RightArm_dim{i+1}")
    plt.title("Right Arm L1 Error per Step")
    plt.xlabel("Step")
    plt.ylabel("L1 Error")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig("docs/l1_error_right_arm.png", dpi=300)

    # 左臂（后7维：joint1-6, gripper）
    plt.figure(figsize=(10, 5))
    for i in range(7, 14):
        plt.plot(steps, l1_errors[:, i], label=f"LeftArm_dim{i-6}")
    plt.title("Left Arm L1 Error per Step")
    plt.xlabel("Step")
    plt.ylabel("L1 Error")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig("docs/l1_error_left_arm.png", dpi=300)

    print("✅ Saved L1 error plots: l1_error_right_arm.png & l1_error_left_arm.png")"""
