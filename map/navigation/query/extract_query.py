import os
import argparse
import openai
import re, json
import copy
from tqdm import tqdm
import cv2


from utils.utils import load_config

from map.utils.gpt import GPTPrompt, parse_object_goal_instruction



def clean_json_string(s):
    return re.sub(r',\s*}', '}', s)



class queries_extractor:
    def __init__(self, config, save_dir):
        self.config = config
        self.targets = config["targets"]
        self.save_dir = save_dir
        self.data_dir = os.path.join(config["data_path"], config["data_type"], config["dataset_type"], config["scene_id"], "rgb")
    def process(self):

        # self.frame_path = os.path.join(self.data_dir, f"{int(self.config['frame_id']):06d}.png")
        prompt_obj = GPTPrompt()
        self.query_inst_dict = prompt_obj.make_queries(self.data_dir, self.targets)
        # print(self.query_inst_dict)
        for target_id, target_val in self.query_inst_dict.items():
            # print(target_val["msgs"]["object"].keys())
            # print(target_val["instance_category"], target_val["room_category"], target_val["frame_id"])
            queries = dict()
            for  query_type, msg in target_val["msgs"].items():
                queries[query_type] = parse_object_goal_instruction(messages=msg)
            self.query_inst_dict[target_id]["queries"] = queries
            del self.query_inst_dict[target_id]["msgs"]
            # for key, val in self.query_inst_dict[target_id].items():
            #     print(key)
            #     print(val)
        # raise Exception("sdf")
        self.save_result()
    def save_result(self):
        save_path = os.path.join(self.save_dir, "queries.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.query_inst_dict, f, indent=4, ensure_ascii=False) 