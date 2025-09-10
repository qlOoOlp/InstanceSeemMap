import os
import argparse
import openai
import re, json
import copy
from tqdm import tqdm
import cv2


from utils.utils import load_config
from map.utils.gpt import GPTPrompt, parse_object_goal_instruction

import json, re

def parse_instance_id(ans_text: str) -> int:
    s = ans_text.strip()

    # 코드펜스 제거
    if s.startswith("```"):
        # ```json\n"99"\n``` 같은 경우 포함
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()

    # 그대로 정수 시도
    try:
        return int(s)
    except ValueError:
        pass

    # JSON literal일 수 있음: "99"
    try:
        v = json.loads(s)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and re.fullmatch(r"-?\d+", v):
            return int(v)
    except Exception:
        pass

    # 텍스트 중 첫 정수 하나만 집어오기 (비상구)
    m = re.search(r"-?\d+", s)
    if m:
        return int(m.group(0))

    # 정말 없으면 에러
    raise ValueError(f"Expected integer id, got: {ans_text!r}")



# def query_process(query, inst_json):
#     prompt_obj = GPTPrompt()
#     messages = prompt_obj.query(query, json_data=inst_json)[:]

#     ans = parse_object_goal_instruction(messages=messages)

#     print(ans)
#     return int(ans) #Todo0814: instance id로 나왔는지 확인할 것

def query_process(query, inst_json):
    prompt_obj = GPTPrompt()
    messages = prompt_obj.query(query, json_data=inst_json)
    ans = parse_object_goal_instruction(messages=messages)
    inst_id = parse_instance_id(ans)
    # print(inst_id)
    return inst_id
