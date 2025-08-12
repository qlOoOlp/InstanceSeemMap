import json
import os
import open3d as o3d
import numpy as np
import yaml
from utils.utils import load_config

def __main__():
    config = load_config("config/visualize_3d.yaml")
    map_path = os.path.join(config["data_path"],config["data_type"],config["dataset_type"],config["scene_id"],"map",f'{config["scene_id"]}_{config["version"]}',"viz","ours_map.ply")
    pcd = o3d.io.read_point_cloud(map_path)

    print(pcd)
    print(np.asarray(pcd.points))

    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    __main__()