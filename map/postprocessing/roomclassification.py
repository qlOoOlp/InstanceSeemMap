import os
import cv2
import json
import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from map.utils.clip_utils import get_text_feats
from map.utils.mapping_utils import get_new_pallete, get_new_mask_pallete, load_map

def load_clip_model(version, device):
    model, _ = clip.load(version)
    model.to(device).eval()
    return model, _

def compute_clip(clip_grid, labels, model, feat_dim):
    map_feats = clip_grid.reshape((-1, clip_grid.shape[-1]))
    text_feats = get_text_feats(labels, model, feat_dim)
    scores_list = map_feats @ text_feats.T
    predicts = np.argmax(scores_list, axis=1)
    return predicts.reshape(clip_grid.shape[:2])

def get_index_and_color_maps(predicts, labels):
    new_palette = get_new_pallete(len(labels))
    mask, patches = get_new_mask_pallete(predicts, new_palette, out_label_flag=True, labels=labels)
    index_map = predicts
    index_map_color = np.array(mask.convert("RGB"))
    return index_map, index_map_color, new_palette, patches

def remove_void(index_map, room_seg):
    # union_array = np.logical_and(index_map ==0, room_seg != 0) #!#!#!#!#!#!#!#!?????????????????
    union_array = np.logical_or(index_map, room_seg == 0) #!#!#!#!#!#!#!#!?????????????????
    return room_seg * union_array

def get_dominant_colors(target_seg, index_map, index_map_color, thres):
    dom_colors, idx_list = [], []
    for idx in np.unique(target_seg):
        if idx == 0: 
            continue
        extracted_boolean = (target_seg == idx)
        if np.sum(extracted_boolean) < thres: 
            continue
        extracted_color = index_map_color * extracted_boolean[:, :, None]
        non_black_pixels = extracted_color[~np.all(extracted_color == [0, 0, 0], axis=2)]
        if non_black_pixels.size == 0: 
            continue
        unique_colors, counts = np.unique(non_black_pixels, axis=0, return_counts=True)
        dom_color = unique_colors[np.argmax(counts)]
        dom_colors.append(dom_color)
        idx_list.append(idx)
    return dom_colors, idx_list

def create_room_segmentation(idx_list, dom_colors, target_seg, palette):
    room_seg_color = np.zeros((target_seg.shape[0], target_seg.shape[1], 3), dtype=np.uint8)
    room_seg_index = np.zeros(target_seg.shape[:2], dtype=np.uint8)
    palette_list = np.array(palette).reshape(-1, 3)
    for i, idx in enumerate(idx_list):
        mask = (target_seg == idx)
        room_seg_color[mask] = dom_colors[i]
        for j, pal in enumerate(palette_list):
            if np.all(dom_colors[i] == pal):
                room_seg_index[mask] = j
    return room_seg_color, room_seg_index

def dilate_segmentation(seg_index, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(seg_index, kernel)

# --------- [추가] room_seg_index2(후처리 결과) -> "룸 고유 id 맵" 생성 ---------
def build_room_id_map(room_seg_index2: np.ndarray) -> np.ndarray:
    """
    room_seg_index2: 후처리된 '클래스 라벨 id' 맵 (0 = void)
    반환: 같은 클래스 내 연결성분마다 1,2,3,...의 '룸 고유 id'를 부여한 맵 (int32)
    """
    h, w = room_seg_index2.shape
    room_id_map = np.zeros((h, w), dtype=np.int32)
    current_id = 0

    unique_classes = np.unique(room_seg_index2)
    for cls_id in unique_classes:
        if cls_id == 0:
            continue
        # 해당 클래스 영역만 추출
        cls_mask = (room_seg_index2 == cls_id).astype(np.uint8)
        # 연결성분 레이블링 (connectivity=4로 유지)
        num, comp = cv2.connectedComponents(cls_mask, connectivity=4)
        # comp: 0은 배경, 1..num-1이 성분
        for cid in range(1, num):
            current_id += 1
            room_id_map[comp == cid] = current_id

    return room_id_map
# ---------------------------------------------------------------------------

class RoomClsProcessor:
    def __init__(self, config, save_dir):
        self.config = config
        self.root_dir = config['root_path']
        self.scene_id = config['scene_id']
        self.version = config['version']
        self.save_dir = save_dir
        self.data_dir = os.path.join(self.root_dir, config['data_type'],config['dataset_type'],config['scene_id'],
                                     'map', f"{config['scene_id']}_{config['version']}")
        self.room_seg_dir = os.path.join(self.data_dir, '03segmentRoom')
        self.feat_dim = {
                            'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                            'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768
                        }[config["clip_version"]]
    def load_data(self):
        self.obs = load_map(os.path.join(self.data_dir, "01buildFeatMap",f"obstacles_{self.version}.npy"))
        self.check_boundary()
        self.seg_result = cv2.imread(os.path.join(self.room_seg_dir, '7-restore_output.png'),0) #!#!#!#!#!#!#!#!어떤 것을 불러와야되지
        self.clip_grid = load_map(os.path.join(self.data_dir, "01buildFeatMap", f"clip_grid_{self.version}.npy"))
        self.clip_grid = np.rot90(self.clip_grid) #!#!#!#!#mp3d
        self.clip_grid = self.clip_grid[self.xmin:self.xmax+1, self.ymin:self.ymax+1]
        self.obs = self.obs[self.xmin:self.xmax+1, self.ymin:self.ymax+1]

    def load_clip(self):
        self.clip_model, _ = load_clip_model(self.config["clip_version"], self.config["device"])

    def process(self):
        self.load_data()
        self.load_clip()
        # Step1. CLIP predict
        index_map, index_map_color, palette, patches = self.predict()
        # Step2. void 제거
        # self.seg_result = remove_void(self.obs, self.seg_result)
        self.seg_result = remove_void(index_map, self.seg_result)
        self.plot_array(self.obs)
        self.plot_array(index_map)
        self.plot_array(self.seg_result)
        # Step3. dominant color 기반 room segmentation
        dom_colors, idx_list = get_dominant_colors(self.seg_result, index_map, index_map_color, self.config["thres_area"])
        room_seg_color, room_seg_index = create_room_segmentation(idx_list, dom_colors, self.seg_result, palette)
        # Step4. morphological dilation
        dilated_seg = dilate_segmentation(room_seg_index, self.config["kernel_size"])
        # Step5. dilation 이후 다시 dominant color 계산
        dom_colors2, idx_list2 = get_dominant_colors(dilated_seg, index_map, index_map_color, self.config["thres_area"])
        room_seg_color2, room_seg_index2 = create_room_segmentation(idx_list2, dom_colors2, dilated_seg, palette)

        self.plot_array(room_seg_index2)

        room_id_map = build_room_id_map(room_seg_index2)
        self.plot_array(room_id_map)
        room_id_map_path = os.path.join(self.save_dir, "room_map.npy")
        os.makedirs(os.path.dirname(room_id_map_path), exist_ok=True)
        np.save(room_id_map_path, room_id_map.astype(np.int32))

        labels = list(self.config["lang_labels"]) 
        id2name = {}
        room_ids = [int(rid) for rid in np.unique(room_id_map) if rid != 0]

        for rid in room_ids:
            mask = (room_id_map == rid)
            cls_vals = room_seg_index2[mask]
            cls_vals = cls_vals[cls_vals != 0]
            if cls_vals.size == 0:
                cls_id = 0
            else:
                vals, cnts = np.unique(cls_vals, return_counts=True)
                cls_id = int(vals[np.argmax(cnts)])
            class_name = labels[cls_id] if 0 <= cls_id < len(labels) else "void"
            id2name[str(rid)] = class_name

        json_path = os.path.join(self.save_dir, "room_info.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(id2name, f, ensure_ascii=False, indent=2)
        # --------------------------------------------------------

    def predict(self):
        predicts = compute_clip(self.clip_grid, self.config["lang_labels"], self.clip_model, self.feat_dim)
        index_map, index_map_color, palette, patches = get_index_and_color_maps(predicts, list(self.config["lang_labels"]))
        return index_map, index_map_color, palette, patches

    def save_result(self, name):
        path = os.path.join(self.save_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.result)

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
