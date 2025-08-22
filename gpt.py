import os, base64
import random
import pickle
from mp3d_room import mp3d_room
from mp3d import mp3d

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class GPTPrompt:
    def __init__(self):
        self.system_prompt_info = f"""Identify and describe instances. Input and output must be in JSON format.
                                    The input field 'captions' contains a list of image captions aiming to identify the instance.
                                    Output 'color' as a visual color of the identified instance
                                    Output 'material' making up the identified instance
                                    Output 'caption' as a concise description of the identified instance."""

        self.example_1_info = """{
                                    "inst_category": "bed",
                                    "room": "bedroom",
                                    "captions": [
                                        "The bed in the scene is a twin bed, featuring a white and pink color scheme. The bed is made with a white sheet and a pink blanket, giving it a classic and elegant appearance. The bed is positioned in the corner of the room, with a window nearby, allowing for natural light to enter the space.",
                                        "The bed in the scene is a small, old-fashioned bed with a white and pink color scheme. It is covered with a white sheet, and there is a pink blanket on top of it. The bed appears to be made with a floral pattern, adding a touch of vintage charm to the room. The bed is situated in the corner of the room, surrounded by a window and a chair.",
                                        "The bed in the scene is a twin-sized bed with a floral bedspread. It is positioned in the corner of the room, close to the window. The bed appears to be old-fashioned, adding a vintage touch to the bedroom."
                                    ]
                                }"""
        
        self.response_1_info = """{
                                    "inst_category": "bed",
                                    "room": "bedroom",
                                    "color": "white, pink",
                                    "material": "sheet, blanket",
                                    "captions": "The twin-sized, old-fashioned bed with a white and pink floral bedspread sits in the corner near a window, adding vintage charm.",
                                }"""
        
        self.example_2_info = """{
                                    "inst_category": "sofa",
                                    "room": "living room",
                                    "captions": [
                                        "The sofa in the scene is a brown leather couch, which is placed in the living room. The couch is positioned in front of a window, and it appears to be well-maintained and comfortable.",
                                        "The sofa in the scene is an old-fashioned, antique-style couch. It is made of brown leather and has a vintage appearance. The couch is placed in the corner of the room, and it appears to be well-maintained and preserved.",
                                        "The sofa in the scene is a brown leather couch. It is placed in front of a window, and a doll is sitting on it. The couch appears to be of good quality and is well-maintained, adding a touch of elegance to the room."
                                    ]
                                }"""
        
        self.response_2_info = """{
                                    "inst_category": "sofa",
                                    "room": "living room",
                                    "color": "brown",
                                    "material": "leather",
                                    "captions": "The well-maintained brown leather couch, with a vintage charm, sits in front of a window, adding elegance to the room.",
                                }"""
        
        self.example_3_info = """{
                                "inst_category": "curtain",
                                "room": "bedroom",
                                "captions": [
                                    "The curtain in the scene is white and made of lace. It is hanging in front of a window, covering it partially. The lace pattern adds a delicate and elegant touch to the room.",
                                    "The curtain in the scene is white and has a lace pattern. It is hanging in front of a window, covering it partially. The curtain appears to be old and worn, suggesting that it has been in place for a long time.",
                                    "There is one curtain in the scene, and it is white. The curtain is hanging in front of a window, and it appears to be sheer, which suggests that it is made of a lightweight and delicate material. The sheer curtain allows some light to pass through while still providing privacy and a sense of coziness to the room."
                                ]
                            }"""
        
        self.response_3_info = """{
                                    "inst_category": "curtain",
                                    "room": "bedroom",
                                    "color": "white",
                                    "material": "lace",
                                    "captions": "The white lace curtain, sheer and slightly worn, hangs in front of a window, adding a delicate and cozy touch.",
                                }"""
        
        self.example_4_info = """{
                                    "inst_category": "chest_of_drawers",
                                    "room": "bedroom",
                                    "captions": [
                                        "The chest of drawers in the scene is made of wood and is placed in the corner of the room. It is a small wooden dresser with a vintage appearance, likely made of high-quality materials.",
                                        "The chest of drawers in the scene is made of wood and is placed in the corner of the room. It is a small, wooden piece of furniture that adds a touch of warmth and character to the space."
                                    ]
                                }"""
        
        self.response_4_info = """{
                                    "inst_category": "chest_of_drawers",
                                    "room": "bedroom",
                                    "color": "brown",
                                    "material": "wood",
                                    "captions": "The small wooden chest of drawers sits in the corner, adding warmth and vintage charm to the room.",
                                }"""
        
        self.example_5_info = """{
                                    "inst_category": "sink",
                                    "room": "kitchen",
                                    "captions": [
                                        "The sink in the scene is white and made of porcelain. It is located in the middle of the kitchen, under the window.",
                                        "The sink in the scene is made of wood and is located in the kitchen.",
                                        "The sink in the scene is white and made of porcelain. It is a single sink, located in a bathroom setting."
                                        ]
                                }"""
        
        self.response_5_info = """{
                                    "inst_category": "sink",
                                    "room": "kitchen",
                                    "color": "white",
                                    "material": "porcelain",
                                    "captions": "The white porcelain sink, located under a window, is placed in the kitchen, adding a clean and classic touch.",
                                }"""

    def to_summarize_with_cate(self):
        prompt_json = [
            {
                "role": "system",
                "content": self.system_prompt_info
            },
            {
                "role": "user",
                "content": self.example_1_info
            },
            {
                "role": "assistant",
                "content": self.response_1_info
            },
            {
                "role": "user",
                "content": self.example_2_info
            },
            {
                "role": "assistant",
                "content": self.response_2_info
            },
            {
                "role": "user",
                "content": self.example_3_info
            },
            {
                "role": "assistant",
                "content": self.response_3_info
            },
            {
                "role": "user",
                "content": self.example_4_info
            },
            {
                "role": "assistant",
                "content": self.response_4_info
            },
            {
                "role": "user",
                "content": self.example_5_info
            },
            {
                "role": "assistant",
                "content": self.response_5_info
            }
        ]
        return prompt_json
    
    def query(self, query, json_data):
        prompt_json = [
            {"role": "system", "content": "You are a helpful assistant that selects the most appropriate item from a list of JSON objects based on the user's query."},
            {"role": "user", "content": f"From the following JSON list, find the item that best matches the query and return only its instance_id. If there is no suitable match, do not return anything.: '{query}'\n\n{json_data}"}
        ]   
        return prompt_json
    
    def make_query(self, query_type, img_dir, inst_caption, inst_content):
        frame_num = f'{next(iter(inst_content["frames"].keys())):06}'
        base64_image = encode_image_to_base64(os.path.join(img_dir, f"{frame_num}.png"))
        
        prompt = ""
        reference_text = """
                        Ex1. The dark wood cabinet, likely walnut or mahogany, features a glossy finish, intricate carvings, and sturdy construction, suggesting high-quality craftsmanship from the late 19th or early 20th century.
                        Ex2. The large dark wood door, featuring an aged finish and glass panels, stands prominently in the hallway, adding character and elegance.
                        Ex3. The flat-screen TV monitor with a dark brown wooden frame is mounted on the wall, providing a modern touch to the hallway.
                        """
        
        if query_type == "object":
            prompt = f"""
                        You are given an image. Your task is to analyze the image and identify a visually prominent object.
                        Generate a short natural language query for a user to find this object.

                        The query can be:
                        - A specific target like "find a sofa"
                        - Or more abstract like "find something to sit on" or "find something to sleep"

                        Focus only on the object itself. Do NOT include room information or surrounding context.
                    """
        elif query_type == "room":
            prompt = """
                        You are given an image. Your task is to:
                        1. Identify the type of room shown in the image (e.g., bedroom, bathroom, kitchen, hallway).
                        2. Identify one prominent object in that room.
                        3. Generate a natural language search query.

                        Output must follow this exact format:
                        Find a [object] in the [room type].

                        Do NOT include:
                        - Bullet points
                        - Markdown formatting
                        - "Search query:" or any other prefixes
                        - Explanations or justification
                        - Descriptions beyond the target object and room

                        Good Examples:
                        Find a cozy bed in the bedroom.
                        Find a white sink in the bathroom.
                        Find a vintage chair in the living room.

                        Now generate only one sentence in that format.
                    """
        elif query_type == "caption":
            prompt = f"""
                        You are given an image and a list of object descriptions. Each description refers to a specific instance of an object in the scene.

                        Your task is to look at the image and, based on the visual elements you detect,
                        select the most visually relevant instance from the reference text,
                        and generate a highly detailed and specific search query describing that object.

                        Use natural language. Your goal is to make the query both *visually grounded* and *textually enriched*.

                        Reference Descriptions:
                        {reference_text}

                        Be specific, natural, and visually grounded. Do not describe the instance itself — focus on locating the target object using the instance as context.

                        - "Find a chair positioned next to the window covered with lace curtain."
                    """
        elif query_type == "mixed":
            prompt = f"""
                        You are given an image and a set of detailed descriptions of objects or furniture in the scene.

                        Your task:
                        1. Identify the most visually prominent object in the image.
                        2. Match it to the most relevant reference description.
                        3. Infer the type of room (e.g., bedroom, kitchen, hallway) from the image.
                        4. Generate a natural search query that includes:
                        - Object type
                        - Descriptive attributes (e.g., color, material, design)
                        - Room/location where the object is found

                        Important:
                        - Use adjectives to describe the object.
                        - **Do NOT use adjectives for the room type.** Keep it simple, like "in the bedroom" or "in the living room".

                        ### Example Output:
                        Find a vintage wooden bed with a colorful quilt in the bedroom.
                    """
            
        prompt_json = [
            {"role": "system", "content": "You are a helpful vision-language assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        return prompt_json, frame_num 
    
    # 이미지를 받아서 query_type에 따라 적절한 쿼리를 생성하는 함수
    def make_queries(self, img_dir, inst_json, pkl_file, cnt_dict): 
        with open(pkl_file, 'rb') as f:
            inst_dict = pickle.load(f)

        extracted_inst_list = []
        inst_id_list = list(inst_dict.keys())
        
        query_inst_dict = {}
        query_dict = {key: [] for key in cnt_dict}

        for query_type, cnt in cnt_dict.items():
            for _ in range(cnt):
                inst_info = {}
                # random inst_id 추출
                random_inst_id = random.choice(inst_id_list)
                inst_id_list.remove(random_inst_id)
                if random_inst_id in extracted_inst_list:
                    continue
                inst_caption = inst_json[f"{random_inst_id}"]["captions"]

                prompt_json, frame_num  = self.make_query(query_type, img_dir, inst_caption, inst_dict[random_inst_id])

                query_dict_ = dict()
                query_dict_["inst_id"] = random_inst_id
                query_dict_["prompt"] = prompt_json
                query_dict[query_type].append(query_dict_)
                
                inst_info["inst_id"] = random_inst_id
                inst_info["frame"] = frame_num
                inst_info["q_type"] = query_type
                query_inst_dict[random_inst_id] = inst_info

        return query_inst_dict, query_dict
    
# Usage example
if __name__ == "__main__":
    prompt_obj = GPTPrompt()
    json_data = prompt_obj.to_summarize_with_cate()
    print(json_data)