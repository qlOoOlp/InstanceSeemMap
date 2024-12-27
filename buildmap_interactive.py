#%%
#! gt맵 만드는 옵션 추가해야됨
import os
import sys
from mapbuilder.utils.mapbuilder import MapBuilder
from utils.parser import parse_args, save_args
from omegaconf import OmegaConf
import json
import torch

def load_config_from_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Invalid json path: {json_path}")
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    # args=parse_args()
    json_path = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data/habitat_sim/2t7WUuJeko7_2/map/hparam.json"
    config_dict = load_config_from_json(json_path)
    config = OmegaConf.create(config_dict)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print(config)
    mapbuilder=MapBuilder(config)
    mapbuilder.buildmap()

if __name__=="__main__":
    print("df")
    main()
# %%
