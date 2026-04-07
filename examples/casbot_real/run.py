import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from mcap2hdf5 import save_data_hdf5


def convert_one(args):
    i, mcap_file_name, save_path = args

    if not os.path.exists(mcap_file_name):
        return i, mcap_file_name, "mcap not exists", False
    if os.path.exists(save_path):
        return i, mcap_file_name, "hdf5 exists", False

    try:
        rt = save_data_hdf5(mcap_file_name, save_path)
        return i, mcap_file_name, rt, True
    except Exception as e:
        return i, mcap_file_name, str(e), False


if __name__ == "__main__":
    root = "/opt/lyx/mcap_datasets/data0402"
    save_root = "/opt/lyx/casbot_dataset/local_dataset/hdf5_dataset/casbot"
    os.makedirs(save_root, exist_ok=True)

    files = sorted(os.listdir(root))

    tasks = []
    for i, file in enumerate(files):
        mcap_file_name = os.path.join(root, file)
        save_path = os.path.join(save_root, f"episode_{i}_4_2.hdf5")
        tasks.append((i, mcap_file_name, save_path))

    # 进程数
    """for i in range(5):
        convert_one(tasks[i])"""
    num_workers = min(8, os.cpu_count()//2)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_one, t) for t in tasks]

        for fut in as_completed(futures):
            i, fname, msg, ok = fut.result()
            status = "OK" if ok else "FAIL"
            print(f"[{status}] episode_{i} | {fname} | {msg}")
