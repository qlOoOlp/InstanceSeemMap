import os
import argparse
import clip
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from roomseg_utils import *
from lang import *

class RoomSegProcessor:
    def __init__(self, root_dir, scene_id, version, k_size):
        self.root_dir = root_dir
        self.scene_id = scene_id
        self.version = version
        self.k_size = k_size

        self.data_dir = os.path.join(self.root_dir, f"{self.scene_id}/map/{self.scene_id}_{self.version}")
        self.room_seg_dir = os.path.join(self.data_dir, 'room_seg')

        obstacles_save_path = os.path.join(self.data_dir, f'obstacles_{self.version}.npy') 
        obstacles = load_map(obstacles_save_path)
        x_indices, y_indices = np.where(obstacles == 1)
        xmin, xmax, ymin, ymax = np.min(x_indices), np.max(x_indices), np.min(y_indices), np.max(y_indices)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_version = "ViT-B/32"
        clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                        'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
        clip_model, _ = clip.load(clip_version)  # clip.available_models()
        clip_model.to(device).eval()

        if self.scene_id == '2t7WUuJeko7_2':
            self.lang = lang_2t7WUuJeko7_2
        
        # clip result
        clip_grid_save_path = os.path.join(self.data_dir, f'clip_grid_{self.version}.npy')
        clip_grid = load_map(clip_grid_save_path)[xmin:xmax+1, ymin:ymax+1]
        map_feats = clip_grid.reshape((-1, clip_grid.shape[-1]))
        text_feats = get_text_feats(self.lang, clip_model, clip_feat_dim)
        scores_list = map_feats @ text_feats.T
        self.predicts = np.argmax(scores_list, axis=1).reshape((xmax - xmin + 1, ymax - ymin + 1))

        self.pallete = get_new_pallete(len(self.lang))
        self.pallette_list = np.array(self.pallete).reshape(-1, 3)

        self.room_seg = cv2.imread(os.path.join(self.room_seg_dir, 'room_seg.png'), 0)

        # room_seg result before dillation
        self.new_room_seg_color = np.zeros((self.predicts.shape[0], self.predicts.shape[1], 3), dtype=np.uint8)
        self.new_room_seg = np.zeros((self.predicts.shape[0], self.predicts.shape[1]), dtype=np.uint8)

        # room_seg result after dillation
        self.new_room_seg_color2 = np.zeros((self.predicts.shape[0], self.predicts.shape[1], 3), dtype=np.uint8)
        self.new_room_seg2 = np.zeros((self.predicts.shape[0], self.predicts.shape[1]), dtype=np.uint8)

    def extract_dominant_colors(self, target, thres=300):
        dom_arr = []
        idx_arr = []
        for idx in np.unique(target):
            if idx==0: continue
            extracted_boolean = target==idx
            if np.sum(extracted_boolean) < thres: continue
            extracted_region_color = self.clip_color * extracted_boolean[:, :, None]

            # black color 제외 후, dominant color 추출
            non_black_pixels = extracted_region_color[~np.all(extracted_region_color == [0, 0, 0], axis=2)]
            unique_colors, counts = np.unique(non_black_pixels, axis=0, return_counts=True)
            dominant_color = unique_colors[np.argmax(counts)]
            
            dom_arr.append(dominant_color)
            idx_arr.append(idx)
        
        return dom_arr, idx_arr
    
    def color_room_seg(self, dom_arr, idx_arr, void_room_seg, room_seg, room_seg_color):
        for i, idx in enumerate(idx_arr):
            extracted_boolean = void_room_seg==idx
            room_seg_color[extracted_boolean] = dom_arr[i]
            for j, pal in enumerate(self.pallette_list):
                if not (all(dom_arr[i] == pal)): continue
                room_seg[extracted_boolean==True] = j

    def process(self):
        # clip result
        mask, _ = get_new_mask_pallete(self.predicts, self.pallete, out_label_flag=True, labels=self.lang)
        self.clip_color = np.array(mask.convert("RGB"))

        # remove part of void
        union_array = np.logical_or(self.predicts, self.room_seg==0)
        self.room_seg_ = self.room_seg * union_array

        # extract_dominant_colors
        dom_arr1, idx_arr1 = self.extract_dominant_colors(target=self.room_seg_, thres=300)
        # room_seg result before dillation
        self.color_room_seg(dom_arr1, idx_arr1, self.room_seg_, self.new_room_seg, self.new_room_seg_color)

        # room_seg result after dillation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.k_size, self.k_size))
        dst = cv2.dilate(self.new_room_seg, kernel)
        np.save(os.path.join(self.room_seg_dir, 'room_seg_result.npy'), dst)

        # dom_arr2, idx_arr2 = self.extract_dominant_colors(target=dst, thres=300)
        # self.color_room_seg(dom_arr2, idx_arr2, dst, self.new_room_seg2, self.new_room_seg_color2)
        # np.save(os.path.join('room_seg_result_color.npy'), self.new_room_seg_color2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/vlmap_RCI/Data/habitat_sim/mp3d',
                        help='Root directory for the dataset')
    parser.add_argument('--scene_id', type=str, default='2t7WUuJeko7_2')
    parser.add_argument('--version', type=str, default='room_seg1_floor_prior')
    parser.add_argument('--k_size', type=int, default=11)

    args = parser.parse_args()

    processor = RoomSegProcessor(args.root_dir, args.scene_id, args.version, args.k_size)
    processor.process()