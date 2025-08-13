from map.navigation.query.get_instance import query_process
from utils.utils import load_config
import os
import json

def main():
    config = load_config('config/navigation.yaml')
    inst_json_file = os.path.join(config["data_path"], config["data_type"], config["dataset_type"], config["scene_id"], "map", f"{config['scene_id']}_{config['version']}", "05captionInstance","final_inst_data.json")
    query_file = os.path.join(config["data_path"], config["data_type"], config["dataset_type"], config["scene_id"], "queries", "queries.json")
    with open(inst_json_file, "r") as st_json:
        inst_json = json.load(st_json)
    with open(query_file, "r") as st_json:
        query_json = json.load(st_json)
    for query in query_json["queries"]:
        ans = query_process(query, inst_json)
        print(f"{'='*20}\nQuery: {query},\n Instance ID: {ans}\n\n")

if __name__=="__main__":
   main()
