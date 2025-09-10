from map.navigation.query.get_instance import query_process
from utils.utils import load_config
import os
import json

def main():
    config = load_config('config/get_inst_from_query.yaml')
    inst_json_file = os.path.join(config["data_path"], config["data_type"], config["dataset_type"], config["scene_id"], "map", f"{config['scene_id']}_{config['version']}", "05captionInstance","final_inst_data.json")
    query_file = os.path.join(config["data_path"], config["data_type"], config["dataset_type"], config["scene_id"], "queries", "queries.json")
    matching_info = {}
    with open(inst_json_file, "r") as st_json:
        inst_json = json.load(st_json)
    with open(query_file, "r") as st_json:
        query_json = json.load(st_json)

    remove_list = []
    for inst_id, inst_val in inst_json.items():
        if inst_val["category"] in ["wall","floor","ceiling","undefined", "unknown"]:
            remove_list.append(inst_id)
        inst_val.pop("room_id")
        inst_val.pop("category_idx")
        inst_val.pop("top5_categories")
    for rem_id in remove_list:
        inst_json.pop(rem_id)
    for gt_id, query_info in query_json.items():
        matching_info[gt_id] = {}
        print(f"GT ID: {gt_id}\t|\tGT Info: {query_info['instance_category'],query_info['room_category'],query_info['frame_id']}")
        for query_type, query in query_info["queries"].items():
            ans = query_process(query, inst_json)
            print(f"{query_type}{'='*20}\nQuery: {query},\n Instance ID: {ans}")
            matching_info[gt_id][query_type] = int(ans)
    save_dir = os.path.join(config["data_path"], config["data_type"], config["dataset_type"], config["scene_id"], "map", f"{config['scene_id']}_{config['version']}", "nav")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "gt_to_inst_matching.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(matching_info, f, indent=4, ensure_ascii=False)

if __name__=="__main__":
   main()
