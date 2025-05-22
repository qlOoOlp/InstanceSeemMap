import os
import sys
import json
import tqdm
from map.mapbuilder.utils.mapbuilder import MapBuilder
from utils.parser import parse_args_extract_captions
from omegaconf import OmegaConf
from map.postprocessing.extract_captions import extract_captions
from map.postprocessing.regenerate_caption import parse_object_goal_instruction
from map.postprocessing.gpt import GPTPrompt


def check_dir(input_dir):
    parent_dir, _ = os.path.split(input_dir)
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"Parent directory {parent_dir} does not exist.")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

def main():
    args=parse_args_extract_captions()
    config = OmegaConf.create(vars(args))

    captioning = extract_captions(config)
    captioning.run()

    caption_file = os.path.join(config['root_path'], config['data_type'], config['dataset_type'], config['scene_id'], 'map',
                                f"{config['scene_id']}_{config['version']}", 'caption', 'inst_data.json')
    regenerated_caption_file = os.path.join(config['root_path'], config['data_type'], config['dataset_type'], config['scene_id'], 'map',
                                f"{config['scene_id']}_{config['version']}", 'caption', 'regenerated_inst_data.json')
    prompt_obj = GPTPrompt()
    new_dict = {}
    with open(caption_file, "r") as st_json:
        inst_json = json.load(st_json)

    for inst_id, info in tqdm(inst_json.items(), desc="Processing instances"):
        info = json.dumps(info, indent=4)
        messages = prompt_obj.to_summarize_with_cate()[:]
        messages.append({"role": "user", "content": info})
        new_dict[inst_id] = parse_object_goal_instruction(messages=messages)
    with open(regenerated_caption_file, 'w') as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
    main()