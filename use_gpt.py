import os
import argparse
import openai
import re, json
from tqdm import tqdm

from gpt import GPTPrompt
from cnt_dict import CNT_DICT

def clean_json_string(s):
    return re.sub(r',\s*}', '}', s)

def parse_object_goal_instruction(messages):
    """
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
 
    # openai_key = os.environ["OPENAI_KEY"]
    # openai.api_key = openai_key
    os.environ.get("OPENAI_API_KEY")

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
    )
 
    text = response.choices[0].message.content # .strip("\n")
    return text

def regenerate_caption(inst_json_file, new_inst_json_file):
    prompt_obj = GPTPrompt()

    new_dict = {}
    
    with open(inst_json_file, "r") as st_json:
        inst_json = json.load(st_json)

    for inst_id, info in tqdm(inst_json.items(), desc="Processing instances"):
        info = json.dumps(info, indent=4)
        messages = prompt_obj.to_summarize_with_cate()[:]
        messages.append({"role": "user", "content": info})

        new_dict[inst_id] = parse_object_goal_instruction(messages=messages)

    with open(new_inst_json_file,'w') as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=4)

def query_process(inst_json_file):
    prompt_obj = GPTPrompt()

    with open(inst_json_file, "r") as st_json:
        inst_json = json.load(st_json)

    messages = prompt_obj.query(query_type="object", json_data=inst_json)[:]

    print(parse_object_goal_instruction(messages=messages))

def make_queries_process(cnt_dict, img_dir, inst_json_file, p_file, f_path):
    with open(inst_json_file, "r") as st_json:
        inst_json = json.load(st_json)

    # 단순 전처리
    for key in inst_json:
        json_str = inst_json[key]
        cleaned_str = clean_json_string(json_str)
        inst_json[key] = json.loads(cleaned_str)

    prompt_obj = GPTPrompt()
    query_inst_dict, messages_list = prompt_obj.make_queries(img_dir=img_dir, inst_json=inst_json, pkl_file=p_file, cnt_dict=cnt_dict)

    for msgs in messages_list.values():
        for msg in msgs:
            query_inst_dict[int(f'{msg["inst_id"]}')]["query"] = parse_object_goal_instruction(messages=msg["prompt"])
            
    with open(f_path, 'w', encoding='utf-8') as f:
        json.dump(query_inst_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='/home/vlmap_RCI/Data/habitat_sim/mp3d',
                        help='Root directory for the dataset')
    parser.add_argument('--scene_id', type=str, default='2t7WUuJeko7_2')
    parser.add_argument('--version', type=str, default='room_seg1_floor_prior')
    parser.add_argument('--pkl-file', type=str, default='inst_category_room_seg2_floor.pkl')    # instance 정보 파일

    parser.add_argument('--gpt-usage', type=str, options=['regenerate_caption', 'query_process', 'make_queries_process'],
                        help='Usage of GPT: regenerate_caption, query_process, make_queries_process')
    parser.add_argument('--pre-json-file', type=str, default='inst_data_llama.json')
    parser.add_argument('--new-json-file', type=str, default='inst_data_llama_summary.json')

    # 생성된 query 저장 파일명
    parser.add_argument('--save-query-file', type=str, default='save_query_file.json')

    args = parser.parse_args()

    data_dir = os.path.join(args.root_dir, f"{args.scene_id}/map/{args.scene_id}_{args.version}")
    inst_caption_dir = os.path.join(data_dir, 'inst_caption')
    pre_json_file_path = os.path.join(inst_caption_dir, args.pre_json_file)
    new_json_file_path = os.path.join(inst_caption_dir, args.new_json_file)

    if args.gpt_usage == 'regenerate_caption':
        regenerate_caption(pre_json_file_path, new_json_file_path)
    elif args.gpt_usage == 'query_process':
        query_process(new_json_file_path)
    elif args.gpt_usage == 'make_queries_process':
        img_dir = os.path.join(args.root_dir, args.scene_id, 'rgb')
        pkl_file_path = os.path.join(inst_caption_dir, args.pkl_file)
        save_f_path = os.path.join(inst_caption_dir, args.save_query_file)

        make_queries_process(CNT_DICT, img_dir, new_json_file_path, pkl_file_path, save_f_path)
    