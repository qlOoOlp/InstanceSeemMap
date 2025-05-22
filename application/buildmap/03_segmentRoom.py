import os
import sys
from map.mapbuilder.utils.mapbuilder import MapBuilder
from utils.parser import parse_args_roomseg
from omegaconf import OmegaConf
from map.postprocessing.roomseg import RoomSegmentation, RoomSegProcessor


def check_dir(input_dir):
    parent_dir, _ = os.path.split(input_dir)
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"Parent directory {parent_dir} does not exist.")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

def main():
    args=parse_args_roomseg()
    config = OmegaConf.create(vars(args))

    save_dir = os.path.join(config['root_path'], config['data_type'],config['dataset_type'],config['scene_id'],'map',
                 f"{config['scene_id']}_{config['version']}","room_seg")
    check_dir(save_dir)
    RoomSegProcessor(config, save_dir).process()
    RoomSegmentation(config, save_dir).run()

if __name__=="__main__":
    main()