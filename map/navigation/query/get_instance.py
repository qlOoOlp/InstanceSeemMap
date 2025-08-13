import os
import argparse
import openai
import re, json
import copy
from tqdm import tqdm
import cv2


from utils.utils import load_config
from map.utils.gpt import GPTPrompt, parse_object_goal_instruction




def query_process(query, inst_json):
    prompt_obj = GPTPrompt()
    messages = prompt_obj.query(query, json_data=inst_json)[:]

    ans = parse_object_goal_instruction(messages=messages)

    print(ans)
    return int(ans) #Todo0814: instance id로 나왔는지 확인할 것
