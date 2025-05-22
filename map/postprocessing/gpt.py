
from utils.matterport3d_room_categories import mp3d_room
from utils.matterport3d_categories import mp3dcat

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

# Usage example
if __name__ == "__main__":
    prompt_obj = GPTPrompt()
    json_data = prompt_obj.to_summarize_with_cate()
    print(json_data)