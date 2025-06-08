import os
import argparse
import openai
import json
from tqdm import tqdm
from gpt import GPTPrompt

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/vlmap_RCI/Data/habitat_sim/mp3d',
                        help='Root directory for the dataset')
    parser.add_argument('--scene_id', type=str, default='2t7WUuJeko7_2')
    parser.add_argument('--version', type=str, default='room_seg1_floor_prior')

    parser.add_argument('--gpt-usage', type=str, default='regenerate_caption')
    parser.add_argument('--pre-json-file', type=str, default='inst_data_llama.json')
    parser.add_argument('--new-json-file', type=str, default='inst_data_llama_summary.json',)

    args = parser.parse_args()

    data_dir = os.path.join(args.root_dir, f"{args.scene_id}/map/{args.scene_id}_{args.version}")
    inst_caption_dir = os.path.join(data_dir, 'inst_caption')
    pre_json_file_path = os.path.join(inst_caption_dir, args.pre_json_file)
    new_json_file_path = os.path.join(inst_caption_dir, args.new_json_file)

    if args.gpt_usage == 'regenerate_caption':
        regenerate_caption(pre_json_file_path, new_json_file_path)
    elif args.gpt_usage == 'query_process':
        query_process(new_json_file_path)
    