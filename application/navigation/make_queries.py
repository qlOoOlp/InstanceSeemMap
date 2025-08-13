import os
import argparse
import openai
import re, json
import copy
from tqdm import tqdm
from utils.utils import load_config
from map.navigation.query.extract_query import queries_extractor

from gpt import GPTPrompt
from cnt_dict import CNT_DICT

if __name__ == "__main__":
    config = load_config('config/navigation.yaml')
    save_dir = os.path.join(config["data_path"], config["data_type"], config["dataset_type"], config["scene_id"], "queries")
    os.mkdir(save_dir, exist_ok=True)
    queries = queries_extractor(config, save_dir)
    queries.process()
