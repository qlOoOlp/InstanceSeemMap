import os
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

if __name__ == "__main__":
    prompt_obj = GPTPrompt()

    inst_json_file = 'inst_data_llama.json'
    new_inst_json_file = 'inst_data_llama_result.json'
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