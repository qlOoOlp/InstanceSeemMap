import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.patches as mpatches

from map.utils.clip_utils import get_text_feats
from map.utils.mapping_utils import load_map, svae_map, get_new_pallete, get_new_mask_pallete

def get_map_bbox(obstacles, obstacle_index=0):
    x_indices, y_indices = np.where(obstacles == obstacle_index)
    xmin = np.min(x_indices)
    xmax = np.max(x_indices)
    ymin = np.min(y_indices)
    ymax = np.max(y_indices)
    return xmin, xmax, ymin, ymax

def viz(map, title=None, is_show=True, save_path=None, save_name=None):
        map_pil = Image.fromarray(map)
        plt.figure(figsize=(8,6),dpi=120)
        plt.imshow(map_pil, cmap="gray")
        if title:
            plt.title(title)
        if save_path:
            plt.savefig(os.path.join(save_path, f"{save_name}.png"), bbox_inches='tight', pad_inches=0.1)
        if is_show:
            plt.show(block=False)

def viz_map(predicts, instance, floor_mask=False, title=None, is_show=True, save_path=None, save_name=None, bool_block=False):
    new_pallete = get_new_pallete(len(instance))
    mask, patches = get_new_mask_pallete(predicts, new_pallete, out_label_flag=True, labels=instance)
    seg = mask.convert('RGBA')
    seg = np.array(seg)
    if floor_mask:
        floor_mask = predicts == instance.index("floor")
        void_mask = predicts == 0
        seg[floor_mask] = [225, 225, 225, 255]
        seg[void_mask] = [225,225,225,255]
    seg = Image.fromarray(seg)
    plt.figure(figsize=(10,6), dpi=120)
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(seg)
    if save_path:
        plt.savefig(os.path.join(save_path, f"{save_name}.png"), bbox_inches='tight', pad_inches=0.1)
    if is_show:
        plt.show(block=bool_block)

    return predicts, instance





def visualize_obstacles(obstacles, bbox=None, slicing=False, is_show=True, save_path=None, save_name=None):
    if bbox == None:
        if slicing:
            bbox = get_map_bbox(obstacles)
            obstacles = slicing_maps(obstacles, bbox)    
    else:
        obstacles = slicing_maps(obstacles, bbox)
    viz(obstacles, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)
    return obstacles

def visualize_rgb(rgb, bbox=None, is_show=True, save_path=None, save_name=None):
    if bbox:
        obstacles = slicing_maps(obstacles, bbox)
    viz(rgb, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)
    return rgb

def get_2Dmap(categories, map_data, vlm_type=None, vlm=None, bbox=None, is_categorized=False, floor_mask=True, is_show=False, save_path=None, save_name=None):
    if not is_categorized:
        if not vlm:
            raise ValueError("VLM must be provided for map visualization.")
        if vlm_type == "lseg":
            text_feats = get_text_feats(categories, vlm, 512)
        elif vlm_type == "seem":
            text_feats = vlm.encode_prompt(categories, task='default')
            text_feats = text_feats.cpu().numpy()
        else:
            raise ValueError(f"Unsupported VLM type: {vlm_type}")
        map_feats = map_data.reshape((-1, map_data.shape[-1]))
        scores_list = map_feats @ text_feats.T
        predicts = np.argmax(scores_list, axis=1)
        if bbox:
            predicts = predicts.reshape((bbox[1]-bbox[0]+1, bbox[3]-bbox[2]+1))
        else:
            predicts = predicts.reshape(map_data.shape)
        return viz_map(predicts, categories, floor_mask = floor_mask, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)
    else:
        if bbox:
            map_data = map_data.reshape((bbox[1]-bbox[0]+1, bbox[3]-bbox[2]+1))
        return viz_map(map_data, categories, floor_mask=floor_mask, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)
            
def get_2Dmap_ours(categories, map_data, vlm=None, bbox=None, use_avg_height=False, is_categorized=False, floor_mask=True, is_show=False, save_path=None, save_name=None):
    grid, instances = map_data
    threshold_l = -50000
    threshold_h = 50000
    if bbox:
        grid = grid[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
    if not is_categorized:
        text_feats = vlm.encode_prompt(categories, task='default')
        text_feats = text_feats.cpu().numpy()
        id_list = []
        instance_feat = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for arg in grid[i,j].keys():
                    if arg not in id_list:
                        id_list.append(arg)
        for id in id_list:
            instance_feat.append(instances[id]["embedding"])
        instance_feat = np.array(instance_feat)
        scores_list = instance_feat @ text_feats.T
        predicts = np.argmax(scores_list, axis=1)
        grid_2d = np.zeros(grid.shape, dtype=int)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not grid[i,j]: continue
                max_height= -50000
                candidate_val = 0
                for key, val in grid[i,j].items():
                    if key == 2: continue
                    candidate = predicts[id_list.index(key)]
                    if use_avg_height:
                        candidate_height = instances[key]["avg_height"]
                    else: candidate_height = val[1]
                    if threshold_l < candidate_height < threshold_h and candidate_height < max_height:
                        max_height = candidate_height
                        candidate_val = candidate
                grid_2d[i,j] = candidate_val
        return viz_map(grid_2d, categories, floor_mask=floor_mask, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)
    else:
        id_list = []
        instance_feat = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for arg in grid[i,j].keys():
                    if arg not in id_list:
                        id_list.append(arg)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not grid[i,j]: continue
                max_height= -50000
                candidate_val = 0
                for key, val in grid[i,j].items():
                    if key == 2: continue
                    candidate = instances[key]['category_idx']
                    if use_avg_height:
                        candidate_height = instances[key]["avg_height"]
                    else: candidate_height = val[1]
                    if threshold_l < candidate_height < threshold_h and candidate_height < max_height:
                        max_height = candidate_height
                        candidate_val = candidate
                grid_2d[i,j] = candidate_val
        return viz_map(grid_2d, categories, floor_mask=floor_mask, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)

def visualize_2Dmap_cat(vlm_type, vlm, categories, map_data, bbox=None, is_show=True, use_avg_height = False, save_path=None, save_name=None):
    # Function for visualize 2D map
    if vlm_type == "ours":
        map, categories = get_2Dmap_ours(categories=categories, map_data=map_data, vlm=vlm, bbox=bbox, is_show=is_show, use_avg_height=use_avg_height, save_path=save_path, save_name=save_name)
    else:
        map, categoriesq = get_2Dmap(categories=categories, map_data=map_data, vlm_type=vlm_type, vlm=vlm, bbox=bbox, is_show=is_show, save_path=save_path, save_name=save_name)
    return map, categories

def visualize_2Dmap_categorized(vlm_type, map_data, bbox=None, is_show=True, use_avg_height = False, save_path=None, save_name=None):
    # Function for visualize 2D map
    if vlm_type == "ours":
        map, categories = get_2Dmap_ours(categories=categories, map_data=map_data, bbox=bbox, is_categorized=True, use_avg_height=use_avg_height, is_show=is_show, save_path=save_path, save_name=save_name)
    else:
        map, categoriesq = get_2Dmap(categories=categories, map_data=map_data, bbox=bbox, is_categorized=True, is_show=is_show, save_path=save_path, save_name=save_name)
    return map, categories

def visualize_2Dmap_inst_in_cat(categories, map_data, bbox=None, is_show=True, save_path=None, save_name=None):
    # Function for visualize instances in each category
    grid, instances = map_data
    cat_maps = []
    if bbox:
        grid = grid[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]

    for i, cat in enumerate(categories):
        if cat in ["void", "floor"]:
            continue
        instance_map = np.zeros(grid.shape, dtype=int)
        instance_keys = []
        for inst_key, inst_item in instances.items():
            if inst_item['category'] == cat:
                inst_mask = inst_item['mask']
                if bbox:
                    inst_mask = inst_mask[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
                instance_map[inst_mask==1] = inst_key
                instance_keys.append(inst_key)
        if len(instance_keys) == 0: continue
        cat_map = viz_map(instance_map, instance_keys, title=f"{cat} instances", is_show=is_show, save_path=save_path, save_name=save_name+f"_{cat}_instances", bool_block=True)
        cat_maps.append(cat_map)
    return cat_maps, categories

def visualize_map(map_data, bbox=None, is_show=True, save_path=None, save_name=None):
    # Function for visualize our map as 2.5D type
    pass

def visualize_instances(rgb, instances, map_data, bbox=None, is_show=True, save_path=None, save_name=None):
    # Function for visualize specific instances on the map
    grid, instances = map_data
    if bbox:
        grid = grid[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
    instance_map = np.zeros(grid.shape, dtype=int)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not grid[i,j]: continue
            for key, val in grid[i,j].items():
                if key not in instances: continue
                instance_map[i,j] = key
    # ----- Overlay 준비 -----
    rgb_overlay = rgb.copy().astype(float) / 255.0  # [0,1]로 정규화
    # 색상 지정 (instance별 고정 색상)
    unique_instances = np.unique(instance_map)
    unique_instances = unique_instances[unique_instances != 0]  # 0 제외
    rng = np.random.default_rng(seed=42)
    color_map = {inst: rng.uniform(0, 1, size=3) for inst in unique_instances}
    alpha = 0.5
    for inst in unique_instances:
        mask = (instance_map == inst)
        color = color_map[inst]
        rgb_overlay[mask] = (1 - alpha) * rgb_overlay[mask] + alpha * color
    # ----- 시각화 -----
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_overlay)
    plt.axis('off')
    # ----- Legend 추가 -----
    legend_patches = []
    for inst in unique_instances:
        color = color_map[inst]
        patch = mpatches.Patch(color=color, label=f'Instance {inst}')
        legend_patches.append(patch)
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    if save_name:
        plt.title(save_name)
    if save_path:
        plt.savefig(os.path.join(save_path, f"{save_name}.png"), bbox_inches='tight', pad_inches=0.1)
    if is_show:
        plt.show(block=False)
                

def visualize_heatmap(rgb, map_data, heatmap_type="simscore", bbox=None, is_show=True, save_path=None, save_name=None):
    # Function for visualize heatmap on the map
    pass

def visualize_room(room, bbox=None, is_show=True, save_path=None, save_name=None):
    pass

def slicing_maps(maps, bbox):
    if isinstance(maps, list):
        return list(map[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1] for map in maps)
    else:
        return maps[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]

def load_maps(map_paths):
    result = []
    for path in map_paths:
        result.append(load_map(path))
    return result

def save_maps(path, map):
    for p, m in zip(path, map):
        svae_map(p, m)
