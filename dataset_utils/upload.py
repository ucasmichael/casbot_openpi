from modelscope.hub.api import HubApi
import os
# liyixuan2026/IROS2025_fold_towel_dataset
# ========== 用户配置部分 ==========
TOKEN = "ms-953ad102-4d0e-4ed6-9098-afcafc37457d"  # 从 ModelScope 用户中心获取
owner_name = "liyixuan2026"      # 你的 ModelScope 用户名
dataset_name = "a3_lerobot"  # 已在 ModelScope 网站上创建的数据集名称
path_in_repo = "stack_cups_teleop_0819_depth"
folder_path = "/data/lyx/dataset/lerobot/local_dataset/a3_0819_both_depth" # "/data/lhr/datasets/openpi_sft_data/agx_iros/fold_towel"  # 本地数据集路径
commit_message = "Upload dataset"
# =================================

def upload_dataset():
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ 找不到本地路径: {folder_path}")

    print("🔑 Logging in to ModelScope...")
    api = HubApi()
    api.login(TOKEN)

    repo_id = f"{owner_name}/{dataset_name}"
    print(f"🚀 准备上传到仓库: {repo_id}")
    print(f"📂 本地路径: {folder_path}")

    # 上传文件夹
    api.upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        path_in_repo = path_in_repo,
        commit_message=commit_message,
        repo_type='dataset'
    )

    print("✅ 数据集上传完成！请前往 ModelScope 网站检查。")

if __name__ == "__main__":
    upload_dataset()
