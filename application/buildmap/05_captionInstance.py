import os
from utils.parser import parse_args_extract_captions
from omegaconf import OmegaConf
from map.postprocessing.captionextract import caption_extractor
from map.postprocessing.captionregenerate import caption_regenerator

def check_dir(input_dir):
    parent_dir, _ = os.path.split(input_dir)
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"Parent directory {parent_dir} does not exist.")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

def main():
    args=parse_args_extract_captions()
    config = OmegaConf.create(vars(args))

    save_dir = os.path.join(config['root_path'], config['data_type'],config['dataset_type'],config['scene_id'],'map',
                 f"{config['scene_id']}_{config['version']}","05captionInstance")
    check_dir(save_dir)
    captioning = caption_extractor(config, save_dir)
    captioning.process()

    regenerator = caption_regenerator(config, save_dir, captioning)
    regenerator.process()

if __name__=="__main__":
    main()