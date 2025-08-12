import os
import cv2
import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from roomseg_utils import *

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
    union_array = np.logical_or(index_map, room_seg == 0)
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

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dim = {
        'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
        'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768
    }[args.clip_version]

    clip_model, _ = load_clip_model(args.clip_version, device)

    # path
    data_dir = os.path.join(args.root_dir, f"{args.scene_id}/map/{args.scene_id}_{args.version}")
    room_seg_dir = os.path.join(data_dir, 'room_seg')

    obs_npy = load_map(os.path.join(data_dir, f'obstacles_{args.version}.npy'))
    x_indices, y_indices = np.where(obs_npy == 1)
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    ymin, ymax = np.min(y_indices), np.max(y_indices)

    # load images
    _room_seg = cv2.imread(os.path.join(room_seg_dir, 'room_seg.png'), 0)
    clip_grid = load_map(os.path.join(data_dir, f'clip_grid_{args.version}.npy'))
    clip_grid = clip_grid[xmin:xmax+1, ymin:ymax+1]

    # Step1. CLIP
    predicts = compute_clip(clip_grid, args.lang_labels, clip_model, feat_dim)
    index_map, index_map_color, palette, patches = get_index_and_color_maps(predicts, args.lang_labels)

    # Step2. void 제거
    room_seg = remove_void(index_map, _room_seg)

    # Step3. dominant color 기반 room segmentation
    dom_colors, idx_list = get_dominant_colors(room_seg, index_map, index_map_color, args.thres_area)
    room_seg_color, room_seg_index = create_room_segmentation(idx_list, dom_colors, room_seg, palette)

    # Step4. morphological dilation
    dilated_seg = dilate_segmentation(room_seg_index, args.kernel_size)

    # Step5. dilation 이후 다시 dominant color 계산
    dom_colors2, idx_list2 = get_dominant_colors(dilated_seg, index_map, index_map_color, args.thres_area)
    room_seg_color2, room_seg_index2 = create_room_segmentation(idx_list2, dom_colors2, dilated_seg, palette)
    
    np.save(os.path.join(room_seg_dir, 'room_seg_class.npy'), room_seg_index2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/home/vlmap_RCI/Data/habitat_sim/mp3d")
    parser.add_argument("--scene_id", type=str, default="2t7WUuJeko7_2")
    parser.add_argument("--version", type=str, default="room_seg1_floor_prior")
    parser.add_argument("--clip_version", type=str, default="ViT-B/32")
    parser.add_argument("--lang_labels", default=["void", "living room", "kitchen", "bathroom", "bedroom", "hallway"], help="Language labels for CLIP classification")
    parser.add_argument("--thres_area", type=int, default=300, help="Minimum pixel count to consider a region")
    parser.add_argument("--kernel_size", type=int, default=11, help="Kernel size for morphological dilation")
    args = parser.parse_args()

    main(args)