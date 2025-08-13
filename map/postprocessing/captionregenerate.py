import os
import re, json
import copy
from tqdm import tqdm

from map.utils.gpt import GPTPrompt, parse_object_goal_instruction

class caption_regenerator:
    def __init__(self, config, save_dir, caption_extractor):
        self.config = config
        self.root_dir = config['root_path']
        self.scene_id = config['scene_id']
        self.version = config['version']
        self.save_dir = save_dir
        self.prev_info = caption_extractor.inst_dict
        self.new_info = copy.deepcopy(self.prev_info)
        self.data_dir = os.path.join(self.root_dir, config['data_type'],config['dataset_type'],config['scene_id'],
                                     'map', f"{config['scene_id']}_{config['version']}")
    def process(self):
        prompt_obj = GPTPrompt()
        for inst_id, inst_val in tqdm(self.prev_info.items(), desc="Regenerating Captions"):
            
            info = json.dumps(inst_val, indent=4)
            messages = prompt_obj.to_summarize_with_cate()[:]
            messages.append({"role": "user", "content": info})
            self.new_info[inst_id]["caption"] = parse_object_goal_instruction(messages=messages)
            del self.new_info[inst_id]["captions"]
        with open(os.path.join(self.save_dir, "final_inst_data.json"), "w") as f:
            json.dump(self.new_info, f, ensure_ascii=False, indent=4)
