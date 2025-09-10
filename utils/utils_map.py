import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.patches as mpatches
import cv2
import open3d as o3d
import matplotlib.cm as cm

from map.utils.clip_utils import get_text_feats
from map.utils.mapping_utils import load_map, save_map, get_new_pallete, get_new_mask_pallete

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

def viz_map(predicts, instance, floor_mask=False, title=None, is_show=True, save_path=None, save_name=None, bool_block=False, bbox=None):
    if bbox:
        predicts = predicts[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
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
        rgb = slicing_maps(rgb, bbox)
    viz(rgb, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)
    return rgb

def get_2Dmap(categories, map_data, vlm_type=None, vlm=None, bbox=None, is_categorized=False, floor_mask=True, is_show=False, save_path=None, save_name=None):
    if bbox:
        map_data = map_data[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
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
                    if threshold_l < candidate_height < threshold_h and candidate_height > max_height:
                        max_height = candidate_height
                        candidate_val = candidate
                grid_2d[i,j] = candidate_val
        return viz_map(grid_2d, categories, floor_mask=floor_mask, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)
    else:
        grid_2d = np.zeros(grid.shape, dtype=int)
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
                    if threshold_l < candidate_height < threshold_h and candidate_height > max_height:
                        max_height = candidate_height
                        candidate_val = candidate
                grid_2d[i,j] = candidate_val
        return viz_map(grid_2d, categories, floor_mask=floor_mask, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)

def visualize_2Dmap_cat(vlm_type, vlm, categories, map_data, bbox=None, is_show=True, use_avg_height = False, save_path=None, save_name=None, floor_mask=True):
    # Function for visualize 2D map
    if vlm_type == "ours":
        map, categories = get_2Dmap_ours(categories=categories, map_data=map_data, vlm=vlm, bbox=bbox, is_show=is_show, use_avg_height=use_avg_height, save_path=save_path, save_name=save_name, floor_mask=floor_mask)
    else:
        map, categories = get_2Dmap(categories=categories, map_data=map_data, vlm_type=vlm_type, vlm=vlm, bbox=bbox, is_show=is_show, save_path=save_path, save_name=save_name, floor_mask=floor_mask)
    return map, categories

def visualize_2Dmap_categorized(vlm_type, categories, map_data, bbox=None, is_show=True, use_avg_height = False, save_path=None, save_name=None, floor_mask=True):
    # Function for visualize 2D map
    if vlm_type == "ours":
        map, categories = get_2Dmap_ours(categories=categories, map_data=map_data, bbox=bbox, is_categorized=True, use_avg_height=use_avg_height, is_show=is_show, save_path=save_path, save_name=save_name, floor_mask=floor_mask)
    else:
        map, categories = get_2Dmap(categories=categories, map_data=map_data, bbox=bbox, is_categorized=True, is_show=is_show, save_path=save_path, save_name=save_name, floor_mask=floor_mask)
    return map, categories

def visualize_2Dmap_inst_in_cat(categories, map_data, bbox=None, is_show=True, save_path=None, save_name=None):
    # Function for visualize instances in each category
    grid, instances = map_data
    cat_maps = []
    if bbox:
        grid = grid[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
    for i, cat in enumerate(categories):
        # if cat in ["void", "floor"]:
        #     continue
        instance_map = np.zeros(grid.shape, dtype=int)
        instance_keys = ["void"]
        inst_id = 1
        for inst_key, inst_item in instances.items():
            if inst_item['category'] == cat:
                inst_mask = inst_item['mask']
                if bbox:
                    inst_mask = inst_mask[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
                if np.sum(inst_mask) == 0: continue
                instance_map[inst_mask==1] = inst_id
                instance_keys.append(inst_key)
                inst_id += 1
        if len(instance_keys) == 0: continue
        cat_map = viz_map(instance_map, instance_keys, title=f"{cat} instances", is_show=is_show, save_path=save_path, save_name=save_name+f"_{cat}_instances", bool_block=True)
        cat_maps.append(cat_map)
    return cat_maps, categories

# def visualize_map(map_data, bbox=None, is_show=True, save_path=None, save_name=None):
#     # Function for visualize our map as 2.5D type
#     grid, instances = map_data
#     if bbox:
#         grid = grid[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
#     print(grid.shape)
#     all_points = []
#     all_colors = []
#     # kyaaa = []
#     # for obj_id, obj_info in instances.items():
#     #     mask = obj_info['mask']  # (H, W), bool
#     #     avg_height = obj_info['avg_height']  # float
#     #     kyaaa.append(avg_height)

#     # print(max(kyaaa), min(kyaaa))
#     # raise Exception("stop")
#     z_scale= 20
#     rng = np.random.default_rng(seed=42)  # 색상 고정 시드

#     for obj_id, obj_info in instances.items():
#         mask = obj_info['mask']  # (H, W), bool
#         avg_height = obj_info['avg_height'] * z_scale # float
#         if bbox:
#             mask = mask[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
#         # mask가 True인 위치의 (y, x) 좌표 가져오기
#         ys, xs = np.nonzero(mask)
#         # if xs.shape[0] == 0: continue
#         # if ys.shape[0] == 0: continue

#         # 포인트 (x, y, z) 만들기
#         zs = np.full_like(xs, avg_height+5.0, dtype=float)

#         points = np.stack([xs, ys, zs], axis=1)  # shape (N, 3)
#         all_points.append(points)

#         # 색상 랜덤 부여 (고정된 색상 사용)
#         color = rng.uniform(0, 1, size=3)
#         colors = np.tile(color, (points.shape[0], 1))
#         all_colors.append(colors)

#     # 모든 객체 포인트 concat
#     all_points = np.concatenate(all_points, axis=0)
#     all_colors = np.concatenate(all_colors, axis=0)

#     # Open3D PointCloud 만들기
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(all_points)
#     pcd.colors = o3d.utility.Vector3dVector(all_colors)

#     # 시각화
#     o3d.visualization.draw_geometries([pcd],
#                                       window_name='Object 3D Map',
#                                       width=800,
#                                       height=600,
#                                       point_show_normal=False)
    






# def visualize_map(
#     map_data,
#     bbox=None,
#     is_show=True,
#     save_path=None,
#     save_name=None,
#     out_range=50.0,
#     offset=5.0,
#     use_bars=False
# ):
#     grid, instances = map_data
#     if bbox is not None:
#         grid = grid[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]

#     # ---------------- 높이 정규화용 배열 ----------------
#     heights = np.array(
#         [float(np.squeeze(info['avg_height']))  # ★ 리스트·배열 → float
#          for info in instances.values()],
#         dtype=float
#     )
#     h_min, h_max = heights.min(), heights.max()
#     if h_max == h_min:
#         h_max += 1e-6

#     def scaled_height(h):
#         h = float(np.squeeze(h))               # ★ 추가 캐스팅
#         return ((h - h_min) / (h_max - h_min)) * out_range + offset

#     cmap = cm.get_cmap('viridis')

#     all_points, all_colors = [], []
#     extra_geoms = []

#     for obj_id, obj_info in instances.items():
#         mask = obj_info['mask']
#         if bbox is not None:
#             mask = mask[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]

#         ys, xs = np.nonzero(mask)
#         if xs.size == 0:
#             continue

#         z_val = scaled_height(obj_info['avg_height'])
#         zs = np.full_like(xs, z_val, dtype=float)

#         all_points.append(np.stack([xs, ys, zs], axis=1))

#         norm_h = (float(np.squeeze(obj_info['avg_height'])) - h_min) / (h_max - h_min)
#         color = cmap(norm_h)[:3]
#         all_colors.append(np.tile(color, (xs.size, 1)))

#         if use_bars:
#             x_c, y_c = xs.mean(), ys.mean()
#             bar_pts = o3d.utility.Vector3dVector([[x_c, y_c, 0], [x_c, y_c, z_val]])
#             line = o3d.geometry.LineSet(
#                 points=bar_pts,
#                 lines=o3d.utility.Vector2iVector([[0, 1]])
#             )
#             line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
#             extra_geoms.append(line)

#     if not all_points:
#         print("포인트가 없습니다.")
#         return

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0).astype(np.float32))
#     pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0).astype(np.float32))

#     if is_show:
#         o3d.visualization.draw_geometries([pcd, *extra_geoms],
#             window_name='Object 3D Map',
#             width=900, height=700,
#             point_show_normal=False
#         )

#     if save_path is not None and save_name is not None:
#         o3d.io.write_point_cloud(f"{save_path}/{save_name}.ply", pcd)


def visualize_map(map_data, bbox=None, is_show=True, save_path=None, save_name=None, out_range=50.0, offset=5.0, use_bars=False, color_by_height=False):
    grid, instances = map_data
    if bbox is not None:
        grid = grid[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]

    # ----- 높이 정규화 -----
    heights = np.array(
        [float(np.squeeze(i["avg_height"])) for i in instances.values()], dtype=float
    )
    h_min, h_max = heights.min(), heights.max()
    if h_max == h_min:
        h_max += 1e-6

    def scaled_height(h):
        h = float(np.squeeze(h))
        return ((h - h_min) / (h_max - h_min)) * out_range + offset

    cmap = cm.get_cmap("viridis")
    rng = np.random.default_rng(seed=42)

    all_pts, all_cols, extra_geoms = [], [], []

    for obj_id, info in instances.items():
        mask = info["mask"]
        if bbox is not None:
            mask = mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]

        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue

        z_val = scaled_height(info["avg_height"])
        zs = np.full_like(xs, z_val, dtype=float)
        all_pts.append(np.stack([xs, ys, zs], axis=1))

        # ----- 색상 결정 -----
        if color_by_height:
            norm_h = (float(np.squeeze(info["avg_height"])) - h_min) / (h_max - h_min)
            base_color = cmap(norm_h)[:3]
        else:
            base_color = rng.uniform(0, 1, size=3)
        all_cols.append(np.tile(base_color, (xs.size, 1)))

        # ----- 기둥(LineSet) -----
        if use_bars:
            x_c, y_c = xs.mean(), ys.mean()
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([[x_c, y_c, 0], [x_c, y_c, z_val]]),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            extra_geoms.append(line)

    if not all_pts:
        print("포인트가 없습니다.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.concatenate(all_pts, axis=0).astype(np.float32)
    )
    pcd.colors = o3d.utility.Vector3dVector(
        np.concatenate(all_cols, axis=0).astype(np.float32)
    )

    # ----- 시각화 (창 확실히 닫기) -----
    if is_show:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Object 3D Map", width=900, height=700, visible=True
        )
        vis.add_geometry(pcd)
        for g in extra_geoms:
            vis.add_geometry(g)

        vis.run()            # 사용자가 q 누르거나 창 닫을 때까지
        vis.destroy_window() # ← 리소스 확실히 해제

    # ----- 저장 -----
    if save_path and save_name:
        o3d.io.write_point_cloud(f"{save_path}/{save_name}.ply", pcd)





def visualize_instances(rgb, inst, map_data, bbox=None, is_show=True, save_path=None, save_name=None):
    # Function for visualize specific instances on the map
    grid, instances = map_data
    print(f"Instance ids: {instances.keys()}")
    if bbox:
        grid = grid[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
        rgb = rgb[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
    instance_map = np.zeros(grid.shape, dtype=int)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not grid[i,j]: continue
            for key, val in grid[i,j].items():
                if key not in inst: continue
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
                

def visualize_heatmap(vlm_type, vlm, target, rgb, map_data, heatmap_type="simscore", bbox=None, is_show=True, save_path=None, save_name=None, categories=None):
    # Function for visualize heatmap on the map
    grid, instances = map_data
    if bbox:
        grid = grid[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
        rgb = rgb[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
    if heatmap_type == 'simscore':
        inst_key_dict = {}
        vec_list = []
        if not vlm:
            raise ValueError("VLM must be provided for map visualization.")
        if vlm_type == "lseg":
            text_feats = get_text_feats(target, vlm, 512)
        elif vlm_type in  ["seem", "ours"]:
            text_feats = vlm.encode_prompt(target, task='default')
            text_feats = text_feats.cpu().numpy()
        norms = np.linalg.norm(text_feats, axis=1, keepdims=True)
        # text_feats = text_feats / norms
        for i, (key, item) in enumerate(instances.items()):
            feat = item["embedding"]
            feat = feat / np.linalg.norm(feat, axis=0, keepdims=True)
            inst_key_dict[key] = i
            vec_list.append(feat)
        feats = np.stack(vec_list, axis=1)
        # feats = feats/ np.linalg.norm(feats, axis=0, keepdims=True)
        scores = text_feats @ feats
        heatmap = np.zeros(grid.shape, dtype=np.float32)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not grid[i,j]: continue
                max_score = 0.0
                for key, val in grid[i,j].items():
                    if key in [0,1,2]: continue
                    candidate = inst_key_dict[key]
                    candidate_val = np.max(scores[:, candidate])
                    if candidate_val > max_score:
                        max_score = candidate_val
                heatmap[i,j] = max_score
        visualize_heatmap_2d(rgb, heatmap, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)
    else:
        targets = list(categories.index(t) for t in target)
        n_cat = len(instances[1]["categories"])
        target_scores = {}
        for tar in targets:
            score_dict = {}
            for i, (key, item) in enumerate(instances.items()):
                vall = np.where(item['categories']==tar)[0]
                if vall < n_cat / 2:
                    score_dict[key] = (n_cat/2-vall) / n_cat *2
                else:
                    score_dict[key] = 0
            target_scores[tar] = score_dict
        heatmap = np.zeros(grid.shape, dtype=np.float32)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not grid[i,j]: continue
                max_score = 0.0
                for key, val in grid[i,j].items():
                    if key in [0,1,2]: continue
                    candidate_val = max(target_scores[tar][key] for tar in targets)
                    if candidate_val > max_score:
                        max_score = candidate_val
                heatmap[i,j] = max_score
        visualize_heatmap_2d(rgb, heatmap, title=save_name, is_show=is_show, save_path=save_path, save_name=save_name)
        

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
        save_map(p, m)


def visualize_heatmap_2d(rgb: np.ndarray, heatmap: np.ndarray, transparency: float = 0.5, title=None, is_show=True, save_path=None, save_name=None):
    min_val = 0
    max_val = 1
    heatmap = (heatmap - min_val) / (max_val - min_val + 1e-8)
    heatmap[heatmap < 0.4] = 0
    sim_new = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    heat = heat[:, :, ::-1].astype(np.float32)  # convert to RGB
    heat_rgb = heat * transparency + rgb * (1 - transparency)
    rgb = heat_rgb.astype(np.uint8)
    plt.title("heatmap")
    plt.imshow(rgb)
    plt.axis('off')  # 축 제거 (옵션)
    ax = plt.gca()
    ax.legend_ = None

    if title:
        plt.title(title)
    else:
        plt.title("heatmap")

    if save_path:
        plt.savefig(os.path.join(save_path, f"{save_name}.png"), bbox_inches='tight', pad_inches=0.1)
    if is_show:
        plt.show(block=False)
