import json
import os
import yaml
from utils.utils import load_config
from .visualizer import visualizer

def __main__():
    config = load_config("config/visualize.yaml")
    if config["save_image"]:
        output_path = os.path.join(config["data_path"],config["data_type"],config["dataset_type"],config["scene_id"],"map",f'{config["scene_id"]}_{config["version"]}')
        viz = visualizer(config) #, output_path)
    else:
        viz = visualizer(config)
    viz.visualize()

if __name__ == "__main__":
    __main__()