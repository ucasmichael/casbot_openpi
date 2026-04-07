# mcap2hdf5.py
import os
import numpy as np
import h5py

from mcap2lerobot import mcap2topic_, synchronous_all_topic


def save_data_hdf5(
    mcap_filename,
    save_path,
):
    assert save_path.endswith('.hdf5')

    all_topic_data, color_topics = mcap2topic_(mcap_filename)
    rt = synchronous_all_topic(all_topic_data, color_topics)

    if len(rt) == 2:
        return rt

    sync_data, min_cam, color_topics = rt
    # import pdb;pdb.set_trace()
    print(color_topics)

    joint_state = sync_data['/casbot/joint_states']['position'] # dim=38 有效 dim=26 (有效是指数据集里面max-min不为0的维度)
    joint_action = sync_data['/casbot/joint_data']['position'] # dim=53 有效 dim=24

    with h5py.File(save_path, 'w') as f:
        obs_grp = f.create_group('observations')
        img_grp = obs_grp.create_group('images')

        # ===== qpos =====
        obs_grp.create_dataset(
            'qpos',
            data=np.asarray(joint_state, dtype=np.float32)
        )

        # ===== actions =====
        f.create_dataset(
            'actions',
            data=np.asarray(joint_action, dtype=np.float32)
        )

        # ===== images =====
        for topic in color_topics:
            cam_name = '_'.join(topic.split('/')[1:])
            imgs = np.stack(sync_data[topic]['color'], axis=0)
            img_grp.create_dataset(
                cam_name,
                data=imgs,
                compression='gzip',
                compression_opts=4
            )

    return ['convert to hdf5 success', True]
