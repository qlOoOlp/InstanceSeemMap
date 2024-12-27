import numpy as np
import torch
from omegaconf import DictConfig
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
import os
from scipy.ndimage import label
import tempfile
from collections import Counter
from skimage.transform import resize
import matplotlib.pyplot as plt

from seem.utils.get_feat import get_SEEM_feat
from seem.base_model import build_vl_model
from utils.mapping_utils import project_point, pos2grid_id, transform_pc, depth2pc, depth2pc4Real, get_sim_cam_mat, get_sim_cam_mat4Real
from utils.get_transform import get_transform
from mapbuilder.map.seemmap import SeemMap
from mapbuilder.utils.datamanager import DataManager, DataManager4Real
from mapbuilder.utils.preprocessing import IQR, depth_filtering

class SeemMap_bbox(SeemMap):
    def __init__(self, config:DictConfig):
        super().__init__(config)
        self.bool_submap = self.config["no_submap"]
        self.bool_seemID = self.config["using_seemID"]
        self.bool_upsample = self.config["upsample"]
        self.bool_IQR = self.config["no_IQR"]
        self.bool_postprocess = self.config["no_postprocessing"]
        self.bool_size = self.config["using_size"]

        self.min_size_denoising_after_projection = self.config["min_size_denoising_after_projection"]
        self.threshold_pixelSize = self.config["threshold_pixelSize"]
        self.threshold_semSim = self.config["threshold_semSim"]
        self.threshold_geoSim = self.config["threshold_geoSim"]
        self.threshold_bbox = self.config["threshold_bbox"]
        self.threshold_semSim_post = self.config["threshold_semSim_post"]
        self.threshold_geoSim_post = self.config["threshold_geoSim_post"]
        self.threshold_pixelSize_post = self.config["threshold_pixelSize_post"]

        self.max_height = self.config["max_height"]

    def processing(self):
        print("start")
        # with tempfile.TemporaryDirectory(dir=self.map_path) as temp_save_dir:
        tf_list = []
        pre_matching_id = {}
        new_instance_id = 3
        pbar = tqdm(range(self.datamanager.numData))
        while self.datamanager.count < self.datamanager.numData-1: # Because count is increased when data_getter is called
            # print("start")
            rgb, depth, (pos,rot) = self.datamanager.data_getter()
            rot = rot @ self.datamanager.rectification_matrix
            pos[1] += self.camera_height
            # if pos[1] < 0:
            #     print(f"Height is negative: {pos[1]}")
            #     print(pos[1])
            #     print(self.camera_height)
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = pos.reshape(-1)
            tf_list.append(pose)
            if len(tf_list) == 1:
                init_tf_inv = np.linalg.inv(tf_list[0])
            tf = init_tf_inv @ pose

            map_idx, map_conf, embeddings, category_dict = get_SEEM_feat(self.model, rgb, self.threshold_confidence)

            if map_idx is None:
                pbar.update(1)
                continue

            if self.bool_upsample:
                upsampling_resolution = (depth.shape[0], depth.shape[1])
                combined = np.stack((map_idx, map_conf), axis=-1)
                upsampled_combined = resize(combined, upsampling_resolution, order=0, preserve_range=True, anti_aliasing=False)
                map_idx = upsampled_combined[:, :, 0].astype(np.uint16)
                map_conf = upsampled_combined[:, :, 1]


            if self.data_type == "rtabmap":
                pc, mask = depth2pc4Real(depth, self.datamanager.projection_matrix, rgb.shape[:2], min_depth=0.5, max_depth=self.max_depth)
                # shuffle_mask = np.arange(pc.shape[1])
                # np.random.shuffle(shuffle_mask)
                # shuffle_mask = shuffle_mask[::self.config['depth_sample_rate']]
                # mask = mask[shuffle_mask]
                # pc = pc[:, shuffle_mask]
                # pc = pc[:, mask]
                pc_global = transform_pc(pc, tf)
                # rgb_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], rgb.shape[:2])
                # feat_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], map_idx.shape)
            else:
                pc, mask = depth2pc(depth, max_depth=self.max_depth, min_depth=0.5)
                # shuffle_mask = np.arange(pc.shape[1])
                # np.random.shuffle(shuffle_mask)
                # shuffle_mask = shuffle_mask[::self.config['depth_sample_rate']]
                # mask = mask[shuffle_mask]
                # pc = pc[:, shuffle_mask]
                # pc = pc[:, mask]
                pc_global = transform_pc(pc, tf)
                # rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
                # feat_cam_mat = get_sim_cam_mat(map_idx.shape[0], map_idx.shape[1])
            frame_mask = np.zeros_like(map_idx)
            depth_shape = depth.shape
            h_ratio = depth.shape[0] / map_idx.shape[0]
            w_ratio = depth.shape[1] / map_idx.shape[1]


            # print("depth process")
            # depth vaule filtering
            # if not self.bool_upsample and (h_ratio != 4 or w_ratio != 4):
            #     raise ValueError(f"Ratio between depth and SEEM Feature map is not 4: h_ratio = {h_ratio}, w_ratio = {w_ratio}")
            if self.bool_IQR:  map_idx = IQR(map_idx, pc, depth_shape, h_ratio, w_ratio, self.max_depth, self.min_depth)
            else: map_idx = depth_filtering(map_idx, pc, depth_shape, h_ratio, w_ratio, self.max_depth, self.min_depth)

            # rgb map processing
            # print("submap process")
            if self.bool_submap: self.submap_processing(map_idx, depth, rgb, pc, pc_global, h_ratio, w_ratio)

            # Projecting SEEM feature map to the grid map
            # print("projection process")
            feat_dict = {}
            for seem_id in np.unique(map_idx):
                if seem_id == 0 :  continue
                feat_map_inst_mask = (map_idx == seem_id).astype(np.uint8)
                feat_map = np.zeros_like(self.grid, dtype=np.float32)
                feat_map_bool = np.zeros_like(self.grid, dtype=np.bool)
                for i,j in np.argwhere(feat_map_inst_mask ==1):
                    new_i = int(i * h_ratio)
                    new_j = int(j * w_ratio)
                    if depth[new_i, new_j] < self.min_depth or depth[new_i,new_j]> self.max_depth:
                        raise Exception("Depth filtering is failed")
                    pp = pc_global[:,new_i*depth_shape[1]+new_j]
                    x,y = pos2grid_id(self.gs,self.cs,pp[0],pp[2])
                    feat_map[y,x] = pp[1]
                    feat_map_bool[y,x]=True
                feat_map_bool = self.denoising(feat_map_bool, self.min_size_denoising_after_projection)
                if np.sum(feat_map_bool) < self.threshold_pixelSize: 
                    map_idx[map_idx == seem_id] = 0
                    continue
                feat_map[feat_map_bool == False] = 0
                feat_dict[seem_id] = feat_map


            # main processing
            # print("process start")
            matching_id = {}
            new_pre_matching_id = {}
            for seem_id in np.unique(map_idx):
                if seem_id == 0 : continue
                feat_map_mask = feat_dict[seem_id]
                feat_map_mask_int = (feat_map_mask != 0).astype(int)
                feat_size = np.sum(feat_map_mask_int)
                avg_height = np.sum(feat_map_mask) / np.sum(feat_map_mask_int)
                if np.sum(feat_map_mask_int) == 0:
                    raise ValueError("Feature map is empty")
                if seem_id in [1,2]:
                    max_id = seem_id #- 1
                    matching_id[seem_id] = max_id
                    candidate_mask = (map_idx == seem_id).astype(np.uint8)
                    pixels = np.sum(candidate_mask)
                    frame_mask[candidate_mask == seem_id] = max_id
                else:
                    if avg_height < -self.max_height: continue #! 천장 조명 필터링
                    candidate_emb = embeddings[seem_id]
                    candidate_emb_normalized = candidate_emb / np.linalg.norm(candidate_emb)
                    candidate_category_id = category_dict[seem_id]
                    max_id = -1
                    candidate_mask = (map_idx == seem_id).astype(np.uint8)
                    pixels = np.sum(candidate_mask)
                    candidate_bbox = self.calculate_bbox(feat_map_mask_int)
                    # else:
                    # print("step2")
                    max_semSim = self.threshold_semSim
                    max_geoSim = self.threshold_bbox
                    for instance_id, instance_val in self.instance_dict.items():
                        try: instance_emb = instance_val["embedding"]
                        except:
                            raise ValueError(f"Instance {instance_id} does not have embedding")
                        instance_emb_normalized = instance_emb / np.linalg.norm(instance_emb)
                        instance_category_id = instance_val["category_id"]
                        instance_bbox = instance_val["bbox"]
                        geoSim = self.calculate_geoSim(candidate_bbox, instance_bbox)
                        # print(geoSim)
                        if self.bool_seemID:
                            if candidate_category_id == instance_category_id:
                                if geoSim > max_geoSim:
                                    max_geoSim = geoSim
                                    max_id = instance_id
                        else:
                            semSim = candidate_emb_normalized @ instance_emb_normalized.T
                            if semSim > max_semSim:
                                if geoSim > max_geoSim:
                                    max_geoSim = geoSim
                                    max_id = instance_id
                    if max_id != -1:
                        matching_id[seem_id] = max_id
                        # new_pre_matching_id[max_id] = {"embedding":candidate_emb, "mask": candidate_mask, "category_id": candidate_category_id}
                        instance_emb = self.instance_dict[max_id]["embedding"]
                        instance_count = self.instance_dict[max_id]["count"]
                        instance_size = self.instance_dict[max_id]["size"]
                        instance_bbox = self.instance_dict[max_id]["bbox"]
                        f_ratio = feat_size / (instance_size + feat_size)
                        i_ratio = instance_size / (instance_size + feat_size)
                        if not self.bool_size:
                            self.instance_dict[max_id]["embedding"] = (instance_emb * instance_count + candidate_emb) / (instance_count + 1)
                        else: self.instance_dict[max_id]["embedding"] = instance_emb * i_ratio + candidate_emb * f_ratio
                        self.instance_dict[max_id]["count"] = instance_count + 1
                        self.instance_dict[max_id]["size"] = instance_size + feat_size
                        self.instance_dict[max_id]["frames"][self.datamanager.count]=pixels
                        self.instance_dict[max_id]["bbox"] = (min(candidate_bbox[0], instance_bbox[0]), min(candidate_bbox[1], instance_bbox[1]), max(candidate_bbox[2], instance_bbox[2]), max(candidate_bbox[3], instance_bbox[3]))
                        frame_mask[candidate_mask == 1] = max_id
                        # self.visualization(np.logical_or(instance_mask, feat_map_mask_int).astype(int)) #!##!#!#!#
                    else:
                        new_id = new_instance_id
                        matching_id[seem_id] = new_id
                        self.instance_dict[new_id] = {"embedding":candidate_emb, "count":1, "size":feat_size, "frames":{self.datamanager.count:pixels}, "category_id":candidate_category_id, "avg_height":avg_height, "bbox":candidate_bbox}
                        frame_mask[candidate_mask == 1] = new_id
                        max_id = new_id
                        new_instance_id += 1
                        # self.visualization(feat_map_mask_int)
                # print("grid saving")
                for coord in np.argwhere(feat_map_mask_int != 0):
                    i,j = coord
                    self.grid[i,j].setdefault(max_id, [0, feat_map_mask[i,j],1])[2] +=1
            self.frame_mask_dict[self.datamanager.count] = frame_mask
            pbar.update(1)
        pbar.close()
        if self.bool_postprocess: self.postprocessing2()


                                

            
    def submap_processing(self, map_idx:NDArray, depth:NDArray, rgb:NDArray, pc_local: NDArray, pc_global: NDArray, h_ratio:float, w_ratio:float):
        for i in range(0,depth.shape[0],4):
            for j in range(0,depth.shape[1],4):
                if depth[i,j] < self.min_depth or depth[i,j] > self.max_depth: continue
                pp = pc_global[:,i*depth.shape[1]+j]
                h = pc_local[:,i*depth.shape[1]+j][1]
                # print(pp,pc_local[:,i*depth.shape[1]+j])
                # raise Exception("sdf")
                x,y = pos2grid_id(self.gs,self.cs,pp[0],pp[2])
                if h < -self.max_height: continue
                if h < self.color_top_down_height[y,x]:
                    # print(h)
                    self.color_top_down[y,x] = rgb[i,j,:]
                    self.color_top_down_height[y,x] = h
                if h > self.camera_height:continue
                self.obstacles[y,x]=0

    def denoising(self, mask:NDArray, min_size:int =5) -> NDArray:
        # type1. biggest one return / type2. removing small noise
        labeled_mask, num_features = label(mask)

        largest_region_label = None
        largest_region_size = 0
        for region_label in range(1, num_features + 1):
            region_size = np.sum(labeled_mask == region_label)
            if region_size > largest_region_size:
                largest_region_size = region_size
                largest_region_label = region_label
        mask[:] = False
        if largest_region_label is not None and largest_region_size >= min_size:
            mask[labeled_mask == largest_region_label] = True
        return mask

        # for region_label in range(1, num_features + 1):
        #     region_size = np.sum(labeled_mask == region_label)
        #     if region_size < min_size:
        #         mask[labeled_mask == region_label] = False
        return mask

    def visualization(self, mask):
        print("visualization")
        coords = np.argwhere(mask==1)
        if coords.size ==0:
            print("Empty mask")
            return
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        print(y_min, x_min, y_max, x_max)
        cropped_mask = mask[850:1100, 850:1100]
        plt.figure(figsize=(6,6))
        plt.imshow(cropped_mask, cmap="gray")
        plt.title("Mask")
        plt.axis("off")
        plt.show()

    def calculate_bbox(self, mask):
        coords = np.argwhere(mask==1)
        if coords.size ==0:
            print("Empty mask")
            return
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return y_min, x_min, y_max, x_max
    

    def calculate_geoSim(self, bbox1, bbox2):
        inter_y_min = max(bbox1[0], bbox2[0])
        inter_x_min = max(bbox1[1], bbox2[1])
        inter_y_max = min(bbox1[2], bbox2[2])
        inter_x_max = min(bbox1[3], bbox2[3])
        # print(inter_y_min, inter_x_min, inter_y_max, inter_x_max)
        inter_height = max(0,inter_y_max - inter_y_min)
        inter_width = max(0,inter_x_max - inter_x_min)
        inter_area = inter_height * inter_width
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        # print("val: ",inter_area, area1, area2)
        val = max(inter_area/area1, inter_area/area2)
        return val


    def calculate_geoSim2(self, bbox1, bbox2):
        inter_y_min = max(bbox1[0], bbox2[0])
        inter_x_min = max(bbox1[1], bbox2[1])
        inter_y_max = min(bbox1[2], bbox2[2])
        inter_x_max = min(bbox1[3], bbox2[3])
        union_y_min = min(bbox1[0], bbox2[0])
        union_x_min = min(bbox1[1], bbox2[1])
        union_y_max = max(bbox1[2], bbox2[2])
        union_x_max = max(bbox1[3], bbox2[3])
        # print(inter_y_min, inter_x_min, inter_y_max, inter_x_max)
        inter_height = max(0,inter_y_max - inter_y_min)
        inter_width = max(0,inter_x_max - inter_x_min)
        inter_area = inter_height * inter_width
        union_height = max(0,union_y_max - union_y_min)
        union_width = max(0,union_x_max - union_x_min)
        union_area = union_height * union_width
        # print("val: ",inter_area, area1, area2)
        return inter_area/union_area




    def postprocessing2(self):
        init = True
        while True:
            if init : tsp = 0.9
            else: tsp = self.threshold_semSim_post
            grid_map = {}
            for y in range(self.gs):
                for x in range(self.gs):
                    for key in self.grid[y,x].keys():
                        if key not in grid_map:
                            grid_map[key] = set()
                        grid_map[key].add((y,x))

            new_instance_dict = {}
            matching_dict = {}
            new_grid = np.empty((self.gs, self.gs), dtype=object)
            count0 = 0
            count1 = 0
            for i in range(self.gs):
                for j in range(self.gs):
                    new_grid[i, j] = {}
                    if 1 in self.grid[i, j].keys(): count0 += 1
                    if 2 in self.grid[i, j].keys(): count1 += 1
            print(f"Size of wall and floor: {count0}, {count1}")

            pbar2 = tqdm(total=len(self.instance_dict.items()), leave=True)
            updated = False  # 변경 여부를 추적하기 위한 플래그
            for instance_id, instance_val in self.instance_dict.items():
                tf = True
                try:
                    instance_y, instance_x = zip(*grid_map[instance_id])
                except:
                    pbar2.update(1)
                    continue
                instance_y = np.array(instance_y)
                instance_x = np.array(instance_x)
                instance_mask = np.zeros((self.gs, self.gs), dtype=np.uint8)
                instance_mask[instance_y, instance_x] = 1
                bbox1 = self.calculate_bbox(instance_mask)
                # if np.sum(instance_mask) < self.threshold_pixelSize_post:
                #     pbar2.update(1)
                #     continue
                instance_emb = instance_val["embedding"]
                instance_size = np.sum(instance_mask)
                for new_id, new_val in new_instance_dict.items():
                    # if new_id in [1,2]: continue
                    new_mask = new_val["mask"]
                    new_emb = new_val["embedding"]
                    new_count = new_val["count"]
                    new_avg_height = new_val["avg_height"]


                    bbox2 = self.calculate_bbox(new_mask)
                    iou1 = self.calculate_geoSim(bbox1, bbox2)
                    iou2 = self.calculate_geoSim(bbox2, bbox1)

                    # intersection = np.logical_and(instance_mask, new_mask).astype(int)
                    # iou1 = np.sum(intersection) / np.sum(instance_mask)
                    # iou2 = np.sum(intersection) / np.sum(new_mask)
                    instance_emb_normalized = instance_emb / np.linalg.norm(instance_emb)
                    new_emb_normalized = new_emb / np.linalg.norm(new_emb)
                    semSim = instance_emb_normalized @ new_emb_normalized.T
                    if max(iou1,iou2) > self.threshold_geoSim_post and semSim > tsp:
                        # self.visualization(instance_mask)
                        # self.visualization(new_mask)
                        new_size = np.sum(new_mask)
                        i_ratio = instance_size / (instance_size + new_size)
                        n_ratio = new_size / (instance_size + new_size)
                        if not self.bool_size:
                            new_instance_dict[new_id]["embedding"] = (new_emb * new_count + instance_emb) / (new_count + 1)
                        else:
                            new_instance_dict[new_id]["embedding"] = instance_emb * i_ratio + new_emb * n_ratio
                        new_instance_dict[new_id]["avg_height"] = (new_avg_height * new_count + instance_val["avg_height"]) / (new_count + 1)
                        new_instance_dict[new_id]["count"] = new_count + 1

                        new_instance_dict[new_id]["size"] = new_size + instance_size

                        new_instance_dict[new_id]["mask"] = np.logical_or(new_mask, instance_mask).astype(np.uint8)
                        new_instance_dict[new_id]["frames"] = dict(Counter(new_instance_dict[new_id]["frames"]) + Counter(instance_val["frames"]))
                        tf = False
                        matching_dict[instance_id] = new_id
                        for frame_key in instance_val["frames"].keys():
                            frame_mask = self.frame_mask_dict[frame_key]
                            self.frame_mask_dict[frame_key][frame_mask == instance_id] = new_id
                        updated = True  # 변경이 발생했음을 표시
                        break
                if tf:
                    if np.sum(instance_mask) < self.threshold_pixelSize_post:
                        pbar2.update(1)
                        continue
                    new_instance_dict[instance_id] = {"mask": instance_mask, "embedding": instance_emb, "count": 1, "size":instance_size, "frames": instance_val["frames"], "avg_height": instance_val["avg_height"]}
                    matching_dict[instance_id] = instance_id
                pbar2.update(1)

            for instance_id in new_instance_dict.keys():
                frames = new_instance_dict[instance_id]["frames"]
                new_instance_dict[instance_id]["frames"] = dict(sorted(frames.items(), key=lambda x: x[1], reverse=True))
            print(new_instance_dict.keys())

            for y in range(self.gs):
                for x in range(self.gs):
                    for key, val in self.grid[y, x].items():
                        if key in [1, 2]:
                            new_grid[y, x][key] = val
                            continue
                        if key not in matching_dict.keys(): continue
                        new_id = matching_dict[key]
                        if new_id not in new_grid[y, x].keys():
                            new_grid[y, x][new_id] = val
            self.grid = new_grid.copy()
            self.instance_dict = new_instance_dict.copy()
            init = False
            if not updated:
                break




    def postprocessing(self):
        while True:
            grid_map = {}
            for y in range(self.gs):
                for x in range(self.gs):
                    for key in self.grid[y,x].keys():
                        if key not in grid_map:
                            grid_map[key] = set()
                        grid_map[key].add((y,x))

            new_instance_dict = {}
            matching_dict = {}
            new_grid = np.empty((self.gs, self.gs), dtype=object)
            count0 = 0
            count1 = 0
            for i in range(self.gs):
                for j in range(self.gs):
                    new_grid[i, j] = {}
                    if 1 in self.grid[i, j].keys(): count0 += 1
                    if 2 in self.grid[i, j].keys(): count1 += 1
            print(f"Size of wall and floor: {count0}, {count1}")

            pbar2 = tqdm(total=len(self.instance_dict.items()), leave=True)
            updated = False  # 변경 여부를 추적하기 위한 플래그
            for instance_id, instance_val in self.instance_dict.items():
                tf = True
                try:
                    instance_y, instance_x = zip(*grid_map[instance_id])
                except:
                    pbar2.update(1)
                    continue
                instance_y = np.array(instance_y)
                instance_x = np.array(instance_x)
                instance_mask = np.zeros((self.gs, self.gs), dtype=np.uint8)
                instance_mask[instance_y, instance_x] = 1
                if np.sum(instance_mask) < 10:
                    pbar2.update(1)
                    continue
                instance_emb = instance_val["embedding"]
                for new_id, new_val in new_instance_dict.items():
                    # if new_id in [1,2]: continue
                    new_mask = new_val["mask"]
                    new_emb = new_val["embedding"]
                    new_count = new_val["count"]
                    new_avg_height = new_val["avg_height"]
                    intersection = np.logical_and(instance_mask, new_mask).astype(int)
                    iou1 = np.sum(intersection) / np.sum(instance_mask)
                    iou2 = np.sum(intersection) / np.sum(new_mask)
                    instance_emb_normalized = instance_emb / np.linalg.norm(instance_emb)
                    new_emb_normalized = new_emb / np.linalg.norm(new_emb)
                    semSim = instance_emb_normalized @ new_emb_normalized.T
                    if max(iou1,iou2) > self.threshold_geoSim_post and semSim > self.threshold_semSim_post:
                        new_instance_dict[new_id]["embedding"] = (new_emb * new_count + instance_emb) / (new_count + 1)
                        new_instance_dict[new_id]["avg_height"] = (new_avg_height * new_count + instance_val["avg_height"]) / (new_count + 1)
                        new_instance_dict[new_id]["count"] = new_count + 1
                        new_instance_dict[new_id]["mask"] = np.logical_or(new_mask, instance_mask).astype(np.uint8)
                        new_instance_dict[new_id]["frames"] = dict(Counter(new_instance_dict[new_id]["frames"]) + Counter(instance_val["frames"]))
                        tf = False
                        matching_dict[instance_id] = new_id
                        for frame_key in instance_val["frames"].keys():
                            frame_mask = self.frame_mask_dict[frame_key]
                            self.frame_mask_dict[frame_key][frame_mask == instance_id] = new_id
                        updated = True  # 변경이 발생했음을 표시
                        break
                if tf:
                    new_instance_dict[instance_id] = {"mask": instance_mask, "embedding": instance_emb, "count": 1, "frames": instance_val["frames"], "avg_height": instance_val["avg_height"]}
                    matching_dict[instance_id] = instance_id
                pbar2.update(1)

            for instance_id in new_instance_dict.keys():
                frames = new_instance_dict[instance_id]["frames"]
                new_instance_dict[instance_id]["frames"] = dict(sorted(frames.items(), key=lambda x: x[1], reverse=True))
            print(new_instance_dict.keys())

            for y in range(self.gs):
                for x in range(self.gs):
                    for key, val in self.grid[y, x].items():
                        if key in [1, 2]:
                            new_grid[y, x][key] = val
                            continue
                        if key not in matching_dict.keys(): continue
                        new_id = matching_dict[key]
                        if new_id not in new_grid[y, x].keys():
                            new_grid[y, x][new_id] = val
            self.grid = new_grid.copy()
            self.instance_dict = new_instance_dict.copy()

            if not updated:
                break
    
    def preprocessing(self):
        raise NotImplementedError
    
    def _init_map(self):
        if self.bool_submap:
            self.color_top_down_height = (self.camera_height + 1) * np.ones((self.gs, self.gs), dtype=np.float32)
            self.color_top_down = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
            self.obstacles = np.ones((self.gs, self.gs), dtype=np.uint8)
            self.weight = np.zeros((self.gs, self.gs), dtype=np.float32)
        self.grid = np.empty((self.gs,self.gs),dtype=object)
        for i in range(self.gs):
            for j in range(self.gs):
                self.grid[i,j] = {}
        background_emb = self.model.encode_prompt(["wall","floor"], task = "default")
        background_emb = background_emb.cpu().numpy()
        self.instance_dict = {}
        self.instance_dict[1] = {"embedding":background_emb[0,:], "count":0,"size":0, "frames":{}, "category_id":1, "avg_height":5000, "bbox":(0,0,self.gs,self.gs)}
        self.instance_dict[2] = {"embedding":background_emb[1,:], "count":0,"size":0, "frames":{}, "category_id":2, "avg_height":0, "bbox": (0,0,self.gs,self.gs)}
        self.frame_mask_dict = {}
    
    def save_map(self):
        if self.bool_submap:
            self.datamanager.save_map(color_top_down=self.color_top_down,
                                    grid=self.grid,
                                    obstacles=self.obstacles,
                                    weight=self.weight,
                                    instance_dict=self.instance_dict,
                                    frame_mask_dict=self.frame_mask_dict)
        else:
            self.datamanager.save_map(grid=self.grid,
                                    instance_dict=self.instance_dict,
                                    frame_mask_dict=self.frame_mask_dict)