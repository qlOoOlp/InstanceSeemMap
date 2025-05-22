import os
import json
import numpy as np
import pickle
from ollama import chat
from ollama import ChatResponse
from tqdm import tqdm
from map.utils.matterport3d_room_categories import mp3d_room


class extract_captions():
    def __init__(self, config):
        self.room_cat = ['void'] + mp3d_room
        self.data_dir = os.path.join(config['root_path'], config['data_type'], config['dataset_type'], config['scene_id'], 'map',
                                     f"{config['scene_id']}_{config['version']}")
        self.save_dir = os.path.join(self.data_dir, 'caption', 'inst_data.json')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.instance_dict_path = os.path.join(self.data_dir, f"categorized_instance_dict_{config['version']}.pkl") 
        self.obs_path = os.path.join(self.data_dir, f"obstacles_{config['version']}.npy")
        self.room_path = os.path.join(self.data_dir,"room_seg", f"room_seg_{config['version']}.npy")
        self.image_root_path = os.path.join(config['root_path'], config['data_type'], config['dataset_type'], config['scene_id'],'rgb')

        self.load_data()

    def load_data(self):
        with open(self.instance_dict_path, 'rb') as f:
            self.instance_dict = pickle.load(f)
        self.obs = np.load(self.obs_path)
        x_indices, y_indices = np.where(self.obs == 1)
        self.xmin = np.min(x_indices)
        self.xmax = np.max(x_indices)
        self.ymin = np.min(y_indices)
        self.ymax = np.max(y_indices)
        self.room = np.load(self.room_path)
        self.inst_dict = {}

    def run(self):
        for inst_id, info in tqdm(self.instance_dict.items()):
            captions = []
            inst_info = {}
            class_name = info['category']
            print(f"inst_id: {inst_id}")
            if class_name in ['floor', 'wall']: continue
            inst_info['category'] = class_name
            mask = info['mask'][self.xmin:self.xmax+1, self.ymin:self.ymax+1]
            x_indices, y_indices = np.where(mask == 1)
            mean_x = np.mean(x_indices)
            mean_y = np.mean(y_indices)
            inst_info["room"] = self.room_cat[self.room[int(mean_x), int(mean_y)]] 
            cnt = 3 if len(list(info['frames'].keys())) > 3 else len(list(info['frames'].keys()))
            for i in range(cnt):
                frame_name = list(info['frames'].keys())[i]
                image_file = os.path.join(self.image_root_path, f"{int(frame_name):06d}.png")

                qs = f"There is one {class_name} in the scene.\
                    Describe and identify the instance including the color and quality of the material."
                
                response: ChatResponse = chat(model='llama3.2-vision', messages=[
                    {
                        'role': 'user',
                        'content': qs,
                        'images': [image_file]
                    },
                ])
                captions.append(response['message']['content'])

            inst_info["captions"] = captions
            self.inst_dict[inst_id] = inst_info

        with open(self.save_dir,'w') as f:
            json.dump(self.inst_dict, f, ensure_ascii=False, indent=4)

