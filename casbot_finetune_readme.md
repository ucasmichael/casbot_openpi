# casbot02 数据微调pi05模型
## 环境配置
见`README.md`
## 数据处理

### 1. step1 cmap转hdf5
使用conda配置一个含有mcap，hdf5相关包的环境，然后执行下面脚本将mcap转为hdf5

eg: `conda activate mcap`

修改./examples/casbot_real/run.py中的路径
```python
root = "/opt/lyx/dataset/mcap_standard" # 文件夹下是采集好的mcap文件
save_root="/opt/lyx/iros_datasets/local_dataset/hdf5_dataset/casbot_test" 
```

```bash
python ./examples/casbot_real/run.py
```

### 2. step2 hdf5转lerobot
* 主要修改`examples/casbot_real/convert2lerobot.py`中的VALID_JOINTS,修改成需要使用的action相关话题：
```python
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
                'left_index_distal_joint',
                
                'right_thumb_metacarpal_joint',
                'right_thumb_proximal_joint', 
                # 'right_thumb_distal_joint', 
                'right_index_proximal_joint', 
                'right_index_distal_joint']
```

* 数据存储路径自定义：找到`./src/openpi/training/data_loader.py` 修改第129行开始的函数：
```python
def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root="/opt/lyx/iros_datasets/"+repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        root="/opt/lyx/iros_datasets/"+repo_id,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset
```

root改为hdf5数据存储的根目录。

开始转换：
```bash
source .venv/bin/activate
python examples/casbot_real/convert2lerobot_statecut.py --raw-dir /opt/lyx/iros_datasets/local_dataset/hdf5_dataset/casbot --repo-id local_dataset/casbot
```

### 3. step3 模型训练
* 修改`./src/openpi/policies/casbot_policy.py`

```python
@dataclasses.dataclass(frozen=True)
class CasbotOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 24 dims.
        return {"actions": np.asarray(data["actions"][:, :24])}
```
把`24`改成实际要预测action的维度（即之前定义的len(VALID_JOINTS)）

自定义hf_cache
```bash
export HF_DATASETS_CACHE="/path/to/hf_cache" # eg: /opt/lyx/hf_cache
export HF_LEROBOT_HOME="/path/to/lerobot" #eg: /opt/lyx/casbot_dataset
```

开始训练
```bash
uv run scripts/compute_norm_stats.py --config-name casbot_fold_cloth
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py casbot_fold_cloth --exp-name=casbot_fold_cloth --overwrite
```

最终模型会保存在`./checkpoints/casbot_fold_cloth/casbot_fold_cloth`下，加载调用即可