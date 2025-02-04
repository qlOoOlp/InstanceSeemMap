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
from scipy.ndimage import label
from sklearn.cluster import DBSCAN

from map.seem.utils.get_feat import get_SEEM_feat
from map.seem.base_model import build_vl_model
from map.utils.mapping_utils import project_point, pos2grid_id, transform_pc, depth2pc, depth2pc4Real, get_sim_cam_mat, get_sim_cam_mat4Real
from map.utils.get_transform import get_transform
from map.mapbuilder.map.seemmap import SeemMap
from map.mapbuilder.utils.datamanager import DataManager, DataManager4Real
from map.mapbuilder.utils.preprocessing import IQR, depth_filtering


class SeemMap_dbscan(SeemMap):
    def __init__(self, config:DictConfig):
        super().__init__(config)
        self.bool_submap = self.config["no_submap"]
        self.bool_seemID = self.config["using_seemID"]
        self.bool_upsample = self.config["upsample"]
        self.bool_IQR = self.config["no_IQR"]
        self.bool_postprocess = self.config["no_postprocessing"]

    def processing(self):
        tf_list = []
        new_instance_id = 2
        pbar = tqdm(range(self.datamanager.numData))
        while self.datamanager.count < self.datamanager.numData-1: # Because count is increased when data_getter is called
            rgb, depth, (pos,rot) = self.datamanager.data_getter()
            rot = rot @ self.datamanager.rectification_matrix
            pos[1] += self.config["camera_height"]
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = pos.reshape(-1)
            tf_list.append(pose)
            if len(tf_list) == 1:
                init_tf_inv = np.linalg.inv(tf_list[0])
            tf = init_tf_inv @ pose

            map_idx, map_conf, embeddings, category_dict = get_SEEM_feat(self.model, rgb, self.config["threshold_confidence"])
            if map_idx == None:
                pbar.update(1)
                continue


            if self.bool_upsample:
                upsampling_resolution = (depth.shape[0], depth.shape[1])
                combined = np.stack((map_idx, map_conf), axis=-1)
                upsampled_combined = resize(combined, upsampling_resolution, order=0, preserve_range=True, anti_aliasing=False)
                map_idx = upsampled_combined[:, :, 0].astype(np.uint8)
                map_conf = upsampled_combined[:, :, 1]


            if self.data_type == "rtabmap":
                pc, mask = depth2pc4Real(depth, self.datamanager.projection_matrix, rgb.shape[:2], min_depth=0.5, max_depth=self.config["max_depth"])
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
                pc, mask = depth2pc(depth, max_depth=self.config["max_depth"])
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



            # depth vaule filtering
            if not self.bool_upsample and (h_ratio != 4 or w_ratio != 4):
                raise ValueError(f"Ratio between depth and SEEM Feature map is not 4: h_ratio = {h_ratio}, w_ratio = {w_ratio}")
            if self.bool_IQR:  map_idx = IQR(map_idx, pc, depth_shape, h_ratio, w_ratio, self.config["max_depth"], self.config["min_depth"])
            else: map_idx = depth_filtering(map_idx, pc, depth_shape, h_ratio, w_ratio, self.config["max_depth"], self.config["min_depth"])

            # rgb map processing
            if self.bool_submap: self.submap_processing(map_idx, depth, rgb, pc_global, h_ratio, w_ratio)

            # Projecting SEEM feature map to the grid map
            feat_dict = {}
            for seem_id in np.unique(map_idx):
                if seem_id == 0 :  continue
                feat_map_inst_mask = (map_idx == seem_id).astype(np.uint8)
                feat_map = np.zeros_like(self.grid, dtype=np.float32)
                feat_map_bool = np.zeros_like(self.grid, dtype=np.bool)
                for i,j in np.argwhere(feat_map_inst_mask ==1):
                    new_i = int(i * h_ratio)
                    new_j = int(j * w_ratio)
                    if depth[new_i, new_j] < 0.1 or depth[new_i,new_j]>3:
                        raise Exception("Depth filtering is failed")
                    pp = pc_global[:,new_i*depth_shape[1]+new_j]
                    x,y = pos2grid_id(self.config["gs"],self.config["cs"],pp[0],pp[2])
                    feat_map[y,x] = pp[1]
                    feat_map_bool[y,x]=True
                feat_map_bool = self.denoising(feat_map_bool, self.config["min_size_denoising_after_projection"])
                if np.sum(feat_map_bool) < self.config["threshold_pixelSize"]: 
                    map_idx[map_idx == seem_id] = 0
                    continue
                feat_map[feat_map_bool == False] = 0
                feat_dict[seem_id] = feat_map

            # main processing
            matching_id = {}
            for seem_id in np.unique(map_idx):
                if seem_id == 0 : continue
                feat_map_mask = feat_dict[seem_id]
                feat_map_mask_int = (feat_map_mask != 0).astype(int)
                if np.sum(feat_map_mask_int) == 0:
                    raise ValueError("Feature map is empty")
                if seem_id in [1,2]:
                    max_id = seem_id - 1
                    matching_id[seem_id] = max_id
                    candidate_mask = (map_idx == seem_id).astype(np.uint8)
                    pixels = np.sum(candidate_mask)
                    frame_mask[candidate_mask == seem_id] = max_id
                else:
                    candidate_emb = embeddings[seem_id]
                    candidate_emb_normalized = candidate_emb / np.linalg.norm(candidate_emb)
                    candidate_category_id = category_dict[seem_id]
                    max_id = -1
                    candidate_mask = (map_idx == seem_id).astype(np.uint8)
                    pixels = np.sum(candidate_mask)

                    max_semSim = self.config["threshold_semSim"]
                    max_geoSim = self.config["threshold_geoSim"]
                    for instance_id, instance_val in self.instance_dict.items():
                        try: instance_emb = instance_val["embedding"]
                        except:
                            raise ValueError(f"Instance {instance_id} does not have embedding")
                        instance_emb_normalized = instance_emb / np.linalg.norm(instance_emb)
                        instance_category_id = instance_val["category_id"]
                        semSim = candidate_emb_normalized @ instance_emb_normalized.T
                        if self.bool_seemID:
                            if candidate_category_id == instance_category_id:
                                max_id = instance_id
                        else:
                            if semSim > max_semSim:
                                max_semSim = semSim
                                max_id = instance_id
                    if max_id != -1:
                        matching_id[seem_id] = max_id
                        instance_emb = self.instance_dict[max_id]["embedding"]
                        instance_count = self.instance_dict[max_id]["count"]
                        self.instance_dict[max_id]["embedding"] = (instance_emb * instance_count + candidate_emb) / (instance_count + 1)
                        self.instance_dict[max_id]["count"] = instance_count + 1
                        self.instance_dict[max_id]["frames"][self.datamanager.count]=pixels
                        frame_mask[candidate_mask == 1] = max_id
                    else:
                        new_id = new_instance_id
                        matching_id[seem_id] = new_id
                        self.instance_dict[new_id] = {"embedding":candidate_emb, "count":1, "frames":{self.datamanager.count:pixels}, "category_id":candidate_category_id}
                        frame_mask[candidate_mask == 1] = new_id
                        new_instance_id += 1
                        max_id = new_id
                for coord in np.argwhere(feat_map_mask_int != 0):
                    i,j = coord
                    self.grid[i,j].setdefault(max_id, [0, feat_map_mask[i,j],1])[2] +=1
            self.frame_mask_dict[self.datamanager.count] = frame_mask
            pbar.update(1)
        pbar.close()
        self.dbscanning()
        if self.bool_postprocess: self.postprocessing()

    def dbscanning(self):
        num_instances = 2
        new_instance_dict = {}
        new_instance_dict[0] = self.instance_dict[0]
        new_instance_dict[1] = self.instance_dict[1]
        new_grid = np.empty((self.config["gs"],self.config["gs"]),dtype=object)
        for i in range(self.config["gs"]):
            for j in range(self.config["gs"]):
                new_grid[i,j] = {}
        pbar2 = tqdm(total=len(self.instance_dict.keys()), leave=True)
        for id, val in self.instance_dict.items():
            id_mask = np.array([[id in d for d in row] for row in self.grid]).astype(np.uint8)
            if np.sum(id_mask) < self.config["threshold_pixelSize"]: continue
            coords = np.column_stack(np.where(id_mask == 1))
            if id in [0,1]:
                for coord in coords:
                    if id in new_grid[coord[0], coord[1]]:
                        new_grid[coord[0], coord[1]][id][2] += self.grid[coord[0], coord[1]][id][2]
                    else:
                        grid_val = self.grid[coord[0], coord[1]][id]
                        new_grid[coord[0], coord[1]][id] = [grid_val[0], grid_val[1], grid_val[2]]
                pbar2.update(1)
                continue
            db = DBSCAN(eps=2, min_samples=1).fit(coords)
            labels = db.labels_
            labeled_mask = np.zeros_like(id_mask, dtype=int)
            for label, (x,y) in zip(labels, coords):
                labeled_mask[x,y] = label + 1
            num_features = len(np.unique(labeled_mask))
            for key in range(1, num_features+1):
                instance_mask = (labeled_mask == key).astype(np.uint8)
                new_instance_dict[num_instances] = self.instance_dict[id]
                if np.sum(instance_mask) < self.config["threshold_pixelSize"]: continue
                for i,j in np.argwhere(instance_mask==1):
                    if num_instances in new_grid[i,j]: raise Exception("Wrong instance ID")
                    new_grid[i,j][num_instances] = [self.grid[i,j][id][0], self.grid[i,j][id][1], self.grid[i,j][id][2]]
                num_instances += 1
            pbar2.update(1)
        pbar2.close()
        self.instance_dict = new_instance_dict.copy()
        self.grid = new_grid.copy()
            
    def submap_processing(self, map_idx:NDArray, depth:NDArray, rgb:NDArray, pc_global:NDArray, h_ratio:float, w_ratio:float):
        for i in range(map_idx.shape[0]):
            for j in range(map_idx.shape[1]):
                new_i = int(i * h_ratio)
                new_j = int(j * w_ratio)
                rgb_seem_id = map_idx[i,j]
                if (depth[new_i, new_j] < 0.1 or depth[new_i, new_j] > 3) and rgb_seem_id != 0:
                    # print(map_idx[new_i, new_j], pc[:,new_i*depthshape[1]+new_j][2])
                    print(self.config["max_depth"], self.config["min_depth"])
                    print(self.bool_IQR)
                    raise Exception(f"Depth filtering is failed: {depth[new_i, new_j]}, {rgb_seem_id}")
                if depth[new_i, new_j] < 0.1 or depth[new_i, new_j] > 3: continue
                pp = pc_global[:,new_i*depth.shape[1]+new_j]
                x,y = pos2grid_id(self.config["gs"],self.config["cs"],pp[0],pp[2])
                h=pp[1]
                if h > -self.config["camera_height"]:self.obstacles[y,x]=0
                if h > self.color_top_down_height[y,x]:
                    self.color_top_down[y,x] = rgb[new_i,new_j,:]
                    self.color_top_down_height[y,x] = h

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
    
    def postprocessing(self):
        grid_map = {}
        for y in range(self.config["gs"]):
            for x in range(self.config["gs"]):
                for key in self.grid[y,x].keys():
                    if key not in grid_map:
                        grid_map[key] = set()
                    grid_map[key].add((y,x))
        
        new_instance_dict = {}
        matching_dict = {}
        new_grid = np.empty((self.config["gs"],self.config["gs"]),dtype=object)
        count0=0
        count1=0
        for i in range(self.config["gs"]):
            for j in range(self.config["gs"]):
                new_grid[i,j] = {}
                if 0 in self.grid[i,j].keys():count0+=1
                if 1 in self.grid[i,j].keys():count1+=1
        print(f"Size of wall and floor: {count0}, {count1}")
        pbar2 = tqdm(total=len(self.instance_dict.items()), leave=True)
        for instance_id, instance_val in self.instance_dict.items():
            tf = True
            try: instance_y, instance_x = zip(*grid_map[instance_id])
            except: 
                pbar2.update(1)
                continue
            instance_y = np.array(instance_y)
            instance_x = np.array(instance_x)
            instance_mask = np.zeros((self.config["gs"],self.config["gs"]),dtype=np.uint8)
            instance_mask[instance_y, instance_x] = 1

            instance_emb = instance_val["embedding"]
            for new_id, new_val in new_instance_dict.items():
                if new_id in [0,1]: continue
                new_mask = new_val["mask"]
                new_emb = new_val["embedding"]
                new_count = new_val["count"]
                intersection = np.logical_and(instance_mask, new_mask).astype(int)
                iou1 = np.sum(intersection) / np.sum(instance_mask)
                instance_emb_normalized = instance_emb / np.linalg.norm(instance_emb)
                new_emb_normalized = new_emb / np.linalg.norm(new_emb)
                semSim = instance_emb_normalized @ new_emb_normalized.T
                if iou1 > self.config["threshold_geoSim"] and semSim > self.config["threshold_semSim"]:

                    new_instance_dict[new_id]["embedding"] = (new_emb * new_count + instance_emb) / (new_count + 1)
                    new_instance_dict[new_id]["count"] = new_count + 1
                    new_instance_dict[new_id]["mask"] = np.logical_or(new_mask, instance_mask).astype(np.uint8)
                    new_instance_dict[new_id]["frames"] = dict(Counter(new_instance_dict[new_id]["frames"]) + Counter(instance_val["frames"]))
                    tf = False
                    matching_dict[instance_id] = new_id
                    for frame_key in instance_val["frames"].keys():
                        frame_mask = self.frame_mask_dict[frame_key]
                        self.frame_mask_dict[frame_key][frame_mask == instance_id] = new_id
                    break
            if tf:
                new_instance_dict[instance_id] = {"mask":instance_mask, "embedding":instance_emb, "count":1, "frames":instance_val["frames"]}
                matching_dict[instance_id] = instance_id
            pbar2.update(1)
        for instance_id in new_instance_dict.keys():
            frames = new_instance_dict[instance_id]["frames"]
            new_instance_dict[instance_id]["frames"] = dict(sorted(frames.items(), key=lambda x:x[1], reverse=True))
        print(new_instance_dict.keys())
        for y in range(self.config["gs"]):
            for x in range(self.config["gs"]):
                for key, val in self.grid[y,x].items():
                    if key in [1,2]:
                        new_grid[y,x][key] = val
                        continue
                    if key not in matching_dict.keys(): continue
                    new_id = matching_dict[key]
                    if new_id not in new_grid[y,x].keys():
                        new_grid[y,x][new_id] = val
        self.grid = new_grid.copy()
        self.instance_dict = new_instance_dict.copy()
    
    def preprocessing(self):
        raise NotImplementedError
    
    def _init_map(self):
        if self.bool_submap:
            self.color_top_down_height = np.zeros((self.config["gs"], self.config["gs"]), dtype=np.float32)
            self.color_top_down = np.zeros((self.config["gs"], self.config["gs"], 3), dtype=np.uint8)
            self.obstacles = np.ones((self.config["gs"], self.config["gs"]), dtype=np.uint8)
            self.weight = np.zeros((self.config["gs"], self.config["gs"]), dtype=np.float32)
        self.grid = np.empty((self.config["gs"],self.config["gs"]),dtype=object)
        for i in range(self.config["gs"]):
            for j in range(self.config["gs"]):
                self.grid[i,j] = {}
        background_emb = self.model.encode_prompt(["wall","floor"], task = "default")
        background_emb = background_emb.cpu().numpy()
        self.instance_dict = {}
        self.instance_dict[0] = {"embedding":background_emb[0,:], "count":0, "frames":{}, "category_id":0}
        self.instance_dict[1] = {"embedding":background_emb[1,:], "count":0, "frames":{}, "category_id":1}
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