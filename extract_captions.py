import os
import json
import numpy as np
import pickle, re
from ollama import chat
from ollama import ChatResponse
from tqdm import tqdm

def clean_caption(text: str, max_sentences: int = 2, max_chars: int = 200) -> str:
    if not text:
        return ""

    # 1. 이상한 문자 제거 (non-breaking space, 유니코드 dash 등)
    text = text.encode("utf-8", "ignore").decode("utf-8")  # 깨진 바이트 제거
    text = re.sub(r'[\xa0–—\-]+', '-', text)  # 여러 개 dash/nbsp -> 일반 "-"
    text = re.sub(r'\s+', ' ', text).strip()  # 불필요한 공백 정리

    # 2. 문장 단위로 자르기
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 3. 문장 개수 제한
    sentences = sentences[:max_sentences]

    # 4. 전체 길이 제한
    result = " ".join(sentences)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(' ', 1)[0] + "..."

    return result

if __name__ == '__main__':
    pkl_file = '/nvme0n1/vlmaps/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_room_seg2_floor/inst_category_room_seg2_floor.pkl'
    obs_file = '/nvme0n1/vlmaps/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_room_seg2_floor/obstacles_room_seg2_floor.npy'
    room_file = '/nvme0n1/vlmaps/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_room_seg2_floor/room_seg/room_seg.npy'
    image_root_file = '/nvme0n1/vlmaps/Data/habitat_sim/mp3d/2t7WUuJeko7_2/rgb'
    
    room_cate = [
        "void",             # 0
        "living room",      # 6
        "kitchen",
        "bathroom",
        "bedroom",          #
        "hallway",
    ]

    with open(pkl_file, 'rb') as f:
        instance_dict = pickle.load(f)
        obstacles = np.load(obs_file)
        x_indices, y_indices = np.where(obstacles == 1)

        xmin = np.min(x_indices)
        xmax = np.max(x_indices)
        ymin = np.min(y_indices)
        ymax = np.max(y_indices)

        rooms = np.load(room_file)
        inst_dict = {}

    for inst_id, info in tqdm(instance_dict.items()):
        captions = []
        inst_info = {} # inst_id, category, room, caption
        
        class_name = info["category"]
        print(f"inst_did: {inst_id}")
        if class_name in ['floor', 'wall']: continue

        inst_info["category"] = class_name
        rotated_mask = np.rot90(info['mask'][xmin:xmax+1, ymin:ymax+1], k=1)
        x_indices, y_indices = np.where(rotated_mask == 1)
        mean_x = np.mean(x_indices)
        mean_y = np.mean(y_indices)

        inst_info["room"] = room_cate[rooms[int(mean_x), int(mean_y)]] 
        cnt = 3 if len(list(info['frames'].keys())) > 3 else len(list(info['frames'].keys()))

        for i in range(cnt):
            frame_name = list(info['frames'].keys())[i]
            image_file = os.path.join(image_root_file, f"{int(frame_name):06d}.png")

            qs = f"There is one {class_name} in the scene. Describe its color, material, and one special feature. Don’t repeat any word or phrase. Limit your response to 2 sentences maximum. Finish as soon as main traits are mentioned. Avoid repeating words, phrases, or patterns."
            
            response: ChatResponse = chat(model='llama3.2-vision:latest', messages=[
                {
                    'role': 'user',
                    'content': qs,
                    'images': [image_file]
                },
            ])
            caption = clean_caption(response['message']['content'], max_sentences=2, max_chars=200)
            captions.append(caption)

        inst_info["captions"] = captions
        inst_dict[inst_id] = inst_info

    with open('./inst_data_llama.json','w') as f:
        json.dump(inst_dict, f, ensure_ascii=False, indent=4)
