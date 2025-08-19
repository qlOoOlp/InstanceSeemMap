import os
import json
import numpy as np
import pickle
from ollama import chat
from ollama import ChatResponse
from tqdm import tqdm
import matplotlib.pyplot as plt
from map.utils.matterport3d_room_categories import mp3d_room
import subprocess
import time


class caption_extractor():
    def __init__(self, config, save_dir):
        self.config = config
        self.root_dir = config['root_path']
        self.scene_id = config['scene_id']
        self.version = config['version']
        self.save_dir = save_dir
        self.data_dir = os.path.join(self.root_dir, config['data_type'],config['dataset_type'],config['scene_id'],
                                     'map', f"{config['scene_id']}_{config['version']}")
        self.room_cat = ["void"] + mp3d_room
        self.restart_interval = 5

    def restart_ollama(self):
        print("Ollama 서버를 재시작합니다...")
        try:
            pids = subprocess.check_output(['pidof', 'ollama']).split()
            for pid in pids:
                subprocess.run(['kill', pid], check=True)
            time.sleep(5)
            subprocess.Popen(['ollama', 'serve'])
            time.sleep(10)
            print("Ollama 서버 재시작 완료.")
        except Exception as e:
            print(f"Ollama 재시작 실패: {e}")
    
    def load_data(self):
        instance_dict_path = os.path.join(self.data_dir, "02buildCatMap", f"categorized_instance_dict_{self.version}.pkl")
        with open(instance_dict_path, 'rb') as f:
            self.instance_dict = pickle.load(f)
        self.obs = np.load(os.path.join(self.data_dir, f"01buildFeatMap", f"obstacles_{self.version}.npy"))
        self.check_boundary()
        self.room = np.load(os.path.join(self.data_dir, "04classificateRoom", "room_map.npy"))
        room_info_path = os.path.join(self.data_dir, "04classificateRoom", "room_info.json")
        with open(room_info_path, "r", encoding="utf-8") as f:
            room_dict = json.load(f)
        self.room_dict = {int(k): v for k, v in room_dict.items()}
        self.image_root_path = os.path.join(self.root_dir, self.config["data_type"], self.config["dataset_type"], self.config["scene_id"], "rgb")
        self.inst_dict = {}

    def process(self):
        self.load_data()
        for i, (inst_id, inst_val) in enumerate(tqdm(self.instance_dict.items(), desc="Extract Instance Captions")):
            # if i > 0 and i % self.restart_interval == 0:
            #     self.restart_ollama()
            print(inst_id)
            captions = []
            inst_info = {}
            class_name = inst_val["category"] #! 이거 top n categories로 수정해야되려나
            if class_name in ["floor", "wall"]: continue
            inst_info["category"] = class_name
            # inst_info["mask"] = inst_val["mask"]
            mask = inst_val["mask"][self.xmin:self.xmax+1, self.ymin:self.ymax+1]
            inst_room_mask = self.room[mask==1]
            nonzero_values = inst_room_mask[inst_room_mask != 0]
            if nonzero_values.size > 0:
                inst_room_id = np.bincount(nonzero_values).argmax()
            else:
                inst_room_id = None
            inst_info["room_cat"] = self.room_dict.get(inst_room_id, None)
            inst_info["room_id"] = inst_room_id
            frame_keys = list(inst_val.get('frames', {}).keys())  # >>> CHANGED
            cnt = 3 if len(frame_keys) > 3 else len(frame_keys)
            for i in range(cnt):
                print(i,cnt)
                frame_name = frame_keys[i] 
                image_file = os.path.join(self.image_root_path, f"{int(frame_name):06d}.png")

                qs = f"There is one {class_name} in the scene.\
                    Describe and identify the instance including the color and quality of the material."
                try: 
                    print("captioning")
                    response: ChatResponse = chat(model='llama3.2-vision', messages=[
                        {
                            'role': 'user',
                            'content': qs,
                            'images': [image_file]
                        }],
                            keep_alive="10m")
                    captions.append(response['message']['content'])
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    captions.append(f"[error] caption generation failed: {e}")  # >>> CHANGED
            inst_info["captions"] = captions
            inst_info["category_idx"] =inst_val["category_idx"]
            inst_info["top5_categories"] = inst_val["categories"][:5]
            self.inst_dict[inst_id] = inst_info
        self.save_result()


    def check_boundary(self):
        x_indices, y_indices = np.where(self.obs == 1)
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        if xmin == 0 and ymin == 0:
            x_indices, y_indices = np.where(self.obs == 0)
            xmin, xmax = np.min(x_indices), np.max(x_indices)
            ymin, ymax = np.min(y_indices), np.max(y_indices)
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax

    def plot_array(self, arr):
        plt.imshow(arr, cmap='tab20')  # 범주형이면 tab20, tab10 같은 colormap 권장
        plt.colorbar(label='Value')
        plt.title("Integer-valued Numpy Array")
        plt.show()

    def save_result(self):
        def to_native(obj):
            # dict: 키도 문자열로 강제(안전)
            if isinstance(obj, dict):
                return {str(k): to_native(v) for k, v in obj.items()}
            # list/tuple
            if isinstance(obj, (list, tuple)):
                return [to_native(x) for x in obj]
            # numpy 배열
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # numpy 스칼라들
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        save_path = os.path.join(self.save_dir, "inst_data.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(to_native(self.inst_dict), f, ensure_ascii=False, indent=2)