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