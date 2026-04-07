import cv2
import numpy as np
import dataclasses
# from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
from mcap_ros2.decoder import Decoder

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

"""def create_empty_dataset(
    data_info: list,
    robot_type: str,
    save_dir: str,
    mode:str,
    dataset_config: DatasetConfig = DatasetConfig(),
) -> LeRobotDataset:
    image_info, arm_joint_names_state, arm_joint_names_action = data_info

    cameras = list(image_info.keys())
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(arm_joint_names_state),),
            "names": [
                arm_joint_names_state,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(arm_joint_names_action),),
            "names": [
                arm_joint_names_action,
            ],
        },
    }
    for cam in cameras:
        features['observation.images.'+image_info[cam][0]] = {
            "dtype": mode,
            "shape": image_info[cam][1],
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
    return LeRobotDataset.create(
        repo_id='',
        fps=30,
        root=save_dir,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )"""

def mcap2topic_(_cur_mcap_filename):
    decoder = Decoder()
    topic_info = {}
    color_image_topic_set = set([])
    with open(_cur_mcap_filename, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            ros_msg = decoder.decode(schema, message)
            if topic not in topic_info:
                topic_info[topic] = {"message_count": 0, }
                topic_info[topic]['sec'] = []
                topic_info[topic]['nanosec'] = []
                topic_info[topic]['find_sec'] = {}
            ros_msg_type = ros_msg._type
            if ros_msg_type == 'sensor_msgs/CompressedImage':
                cur_format = ros_msg.format
                if 'bgr8' in cur_format:
                    bgr_image = cv2.imdecode(np.frombuffer(ros_msg.data, dtype=np.uint8), -1)
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    if 'color' not in topic_info[topic]:
                        topic_info[topic]['color'] = []
                    topic_info[topic]['color'].append(rgb_image)

                    color_image_topic_set.add(topic)

                elif '16UC1; png compressed' in cur_format:
                    depth_image = cv2.imdecode(np.frombuffer(ros_msg.data, dtype=np.uint8), -1)
                    if 'depth' not in topic_info[topic]:
                        topic_info[topic]['depth'] = []
                    topic_info[topic]['depth'].append(depth_image)
                elif 'jpeg' in cur_format:
                    rgb_image = cv2.imdecode(np.frombuffer(ros_msg.data, dtype=np.uint8), -1)
                    if 'color' not in topic_info[topic]:
                        topic_info[topic]['color'] = []
                    topic_info[topic]['color'].append(rgb_image)

                    color_image_topic_set.add(topic)
            elif ros_msg_type == 'sensor_msgs/Imu':
                if 'imu' not in topic_info[topic]:
                    topic_info[topic]['imu'] = []
                topic_info[topic]['imu'].append(ros_msg)
            elif ros_msg_type == 'sensor_msgs/JointState':
                topic_info[topic]['name'] = ros_msg.name
                if 'position' not in topic_info[topic]:
                    topic_info[topic]['position'] = []
                topic_info[topic]['position'].append(ros_msg.position)
            else:
                continue
                # topic_info[topic]['name'] = ros_msg
            if ros_msg.header.stamp.sec not in topic_info[topic]['find_sec']:
                topic_info[topic]['find_sec'][ros_msg.header.stamp.sec] = []
            topic_info[topic]['find_sec'][ros_msg.header.stamp.sec].append(
                [ros_msg.header.stamp.nanosec, topic_info[topic]["message_count"]])

            topic_info[topic]['sec'].append(ros_msg.header.stamp.sec)
            topic_info[topic]['nanosec'].append(ros_msg.header.stamp.nanosec)
            topic_info[topic]["message_count"] += 1
    return topic_info, list(color_image_topic_set)
def synchronous_all_topic(all_topic_src,color_topic_list):
    min_lengh = 9223372036854775807
    min_length_topic_name = ''
    for cur_topic_name in color_topic_list:
        cur_topic_value = all_topic_src[cur_topic_name]
        if cur_topic_value['message_count'] < min_lengh:
            min_lengh = cur_topic_value['message_count']
            min_length_topic_name = cur_topic_name
    cur_topic_value = all_topic_src[min_length_topic_name]
    all_topic_names = set(list(all_topic_src.keys()))
    all_topic_names.remove(min_length_topic_name)


    for cur_topic in all_topic_names:
        if all_topic_src[cur_topic]['message_count'] < all_topic_src[min_length_topic_name]['message_count']:
            return ['cur_topic hz %d' %(all_topic_src[cur_topic]['message_count']), False]


    dict_synchron_info = {}
    # dict_synchron_info[min_length_topic_name] = cur_topic_value
    dict_synchron_info[min_length_topic_name] = {}


    topic_min_time={}
    topic_max_time={}
    topic_offset_time ={}
    cur_topic_min_time = min(list(cur_topic_value['find_sec']))
    min_start_time = cur_topic_min_time
    max_end_time = max(list(cur_topic_value['find_sec']))
    for item in all_topic_src:
        topic_min_time[item] = min(list(all_topic_src[item]['find_sec']))
        topic_max_time[item] = max(list(all_topic_src[item]['find_sec']))
        topic_offset_time[item] = topic_min_time[item]-cur_topic_min_time
        min_start_time = max(topic_min_time[item],min_start_time)
        max_end_time = min(max_end_time,topic_max_time[item])

    for cur_sec in cur_topic_value['find_sec']:
        if cur_sec<min_start_time or cur_sec>=max_end_time:
            continue
        for cur_nanosec in cur_topic_value['find_sec'][cur_sec]:
            if 'color' not in dict_synchron_info[min_length_topic_name]:
                dict_synchron_info[min_length_topic_name]['color'] = []
                dict_synchron_info[min_length_topic_name]['time'] = []
            dict_synchron_info[min_length_topic_name]['color'].append(cur_topic_value['color'][cur_nanosec[1]])
            dict_synchron_info[min_length_topic_name]['time'].append(cur_sec * 1000 + cur_nanosec[0] * 0.000001)
            for other_topic in all_topic_names:
                if 'imu' in other_topic:
                    continue
                min_time = 9223372036854775807
                min_time_index = 0
                for find_cur_nanosec in all_topic_src[other_topic]['find_sec'][cur_sec]:
                    cur_time_delt = abs(find_cur_nanosec[0] - cur_nanosec[0])
                    if cur_time_delt < min_time:
                        min_time = cur_time_delt
                        min_time_index = find_cur_nanosec[1]
                        if find_cur_nanosec[0] > cur_nanosec[0]:
                            break
                    if find_cur_nanosec[0] > cur_nanosec[0]:
                        break
                if other_topic not in dict_synchron_info:
                    dict_synchron_info[other_topic] = {}
                if 'color' in all_topic_src[other_topic]:
                    if 'color' not in dict_synchron_info[other_topic]:
                        dict_synchron_info[other_topic]['color'] = []
                        dict_synchron_info[other_topic]['time'] = []
                    dict_synchron_info[other_topic]['color'].append(
                        all_topic_src[other_topic]['color'][min_time_index])
                    dict_synchron_info[other_topic]['time'].append(
                        all_topic_src[other_topic]['sec'][min_time_index] * 1000 +
                        all_topic_src[other_topic]['nanosec'][min_time_index] * 0.000001)
                if 'imu' in all_topic_src[other_topic]:
                    if 'imu' not in dict_synchron_info[other_topic]:
                        dict_synchron_info[other_topic]['imu'] = []
                        dict_synchron_info[other_topic]['time'] = []
                    dict_synchron_info[other_topic]['imu'].append(all_topic_src[other_topic]['imu'][min_time_index])
                    dict_synchron_info[other_topic]['time'].append(
                        all_topic_src[other_topic]['sec'][min_time_index] * 1000 +
                        all_topic_src[other_topic]['nanosec'][min_time_index] * 0.000001)
                if 'position' in all_topic_src[other_topic]:
                    if 'position' not in dict_synchron_info[other_topic]:
                        dict_synchron_info[other_topic]['position'] = []
                        dict_synchron_info[other_topic]['name'] = all_topic_src[other_topic]['name']
                        dict_synchron_info[other_topic]['time'] = []

                    dict_synchron_info[other_topic]['position'].append(
                        all_topic_src[other_topic]['position'][min_time_index])
                    dict_synchron_info[other_topic]['time'].append(
                        all_topic_src[other_topic]['sec'][min_time_index] * 1000 +
                        all_topic_src[other_topic]['nanosec'][min_time_index] * 0.000001)

                if 'depth' in all_topic_src[other_topic]:
                    if 'depth' not in dict_synchron_info[other_topic]:
                        dict_synchron_info[other_topic]['depth'] = []
                        dict_synchron_info[other_topic]['time'] = []

                    dict_synchron_info[other_topic]['depth'].append(
                        all_topic_src[other_topic]['depth'][min_time_index])
                    dict_synchron_info[other_topic]['time'].append(
                        all_topic_src[other_topic]['sec'][min_time_index] * 1000 +
                        all_topic_src[other_topic]['nanosec'][min_time_index] * 0.000001)
    return [dict_synchron_info, min_length_topic_name, color_topic_list]

def save_data2lerobot(dict_synchron_info,min_length_topic_name,color_topic_list,save_dir,valid_seg,model='image'):
    joint_state = dict_synchron_info['/casbot/joint_states']
    joint_action = dict_synchron_info['/casbot/joint_data']

    #from utils import save_result
    #save_result(joint_state,joint_action)

    arm_joint_names_state = joint_state['name']
    arm_joint_names_action = joint_action['name']

    color_topich_shape_dict = {}
    for cur_color_topic in color_topic_list:
        one_image_shape = dict_synchron_info[cur_color_topic]['color'][0].shape
        color_topich_shape_dict[cur_color_topic] = ['_'.join(cur_color_topic.split('/')[1:]),
                                                    (one_image_shape[2], one_image_shape[0], one_image_shape[1])]

    _robot_type = 'casbot_machine'
    dataset = create_empty_dataset(
        [color_topich_shape_dict, arm_joint_names_state,arm_joint_names_action],
        robot_type=_robot_type,
        save_dir=save_dir,
        mode=model,
        dataset_config=DatasetConfig(),
    )
    for cur_seg in valid_seg:
        cur_color_data = dict_synchron_info[min_length_topic_name]
        start_index = 0
        end_index = len(cur_color_data['time'])
        for index, cur_time in enumerate(cur_color_data['time']):
            if cur_time > cur_seg['start']:
                start_index = index
                break
        for index, cur_time in enumerate(cur_color_data['time']):
            if cur_time > cur_seg['end']:
                end_index = index
                break
        if end_index-start_index>5:
            for i in range(start_index, end_index, 1):
                frame = {
                    "observation.state": np.array(dict_synchron_info['/casbot/joint_states']['position'][i]).astype(np.float32),
                    "action": np.array(dict_synchron_info['/casbot/joint_data']['position'][i]).astype(np.float32),
                    "task": cur_seg['desc']
                }
                for cur_image_topic in color_topich_shape_dict:
                    frame['observation.images.'+color_topich_shape_dict[cur_image_topic][0]] = dict_synchron_info[cur_image_topic]['color'][i]
                dataset.add_frame(frame)
            dataset.save_episode()
    if len(dataset) == 0:
        return ['no valid seg in mcap', False]

    return ['convert is succ',True]
def save_data_lerobot(cur_mcap_filename,valid_segment,save_dir,model='image'):
     all_topic_data,color_topic_list = mcap2topic_(cur_mcap_filename)

     rt = synchronous_all_topic(all_topic_data,color_topic_list)
     if len(rt)==2:
         return rt
     synchron_topic_data, min_length_camera_topic_name, color_topic_list = rt
     rt = save_data2lerobot(synchron_topic_data, min_length_camera_topic_name, color_topic_list,save_dir, valid_segment,model)
     return rt
