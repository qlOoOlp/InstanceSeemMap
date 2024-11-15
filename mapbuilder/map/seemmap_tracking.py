import numpy as np
import torch
from omegaconf import DictConfig
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
import os
from scipy.ndimage import label
import tempfile

from seem.utils.get_feat import get_SEEM_feat
from seem.base_model import build_vl_model
from utils.mapping_utils import project_point, pos2grid_id, transform_pc, depth2pc, depth2pc4Real, get_sim_cam_mat, get_sim_cam_mat4Real
from utils.get_transform import get_transform
from mapbuilder.map.seemmap import SeemMap
from mapbuilder.utils.datamanager import DataManager, DataManager4Real
from mapbuilder.utils.preprocessing import IQR

class SeemMap_tracking(SeemMap):
    def __init__(self, config:DictConfig):
        super().__init__(config)
        self.bool_submap = self.config["no_submap"]
        self.bool_seemID = self.config["using_seemID"]
        self.bool_upsample = self.config["upsample"]
        self.bool_IQR = self.config["preprocess_IQR"]

    def processing(self):
        tf_list = []
        pre_matching_id = {}
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
            if self.data_type == "rtabmap":
                pc, mask = depth2pc4Real(depth, self.datamanager.projection_matrix, rgb.shape[:2], min_depth=0.5, max_depth=self.config["max_depth"])
                # shuffle_mask = np.arange(pc.shape[1])
                # np.random.shuffle(shuffle_mask)
                # shuffle_mask = shuffle_mask[::self.config['depth_sample_rate']]
                # mask = mask[shuffle_mask]
                # pc = pc[:, shuffle_mask]
                pc = pc[:, mask]
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
            if h_ratio != 4 or w_ratio != 4:
                raise ValueError(f"Ratio between depth and SEEM Feature map is not 4: h_ratio = {h_ratio}, w_ratio = {w_ratio}")
            
            map_idx = IQR(map_idx, pc, depth_shape, h_ratio, w_ratio)

            if self.bool_submap: self.submap_processing(map_idx, depth, rgb, pc_global, h_ratio, w_ratio)
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
                feat_map_bool = removing_noise(feat_map_bool)
                    
            
    def submap_processing(self, map_idx:NDArray, depth:NDArray, rgb:NDArray, pc_global:NDArray, h_ratio:float, w_ratio:float):
        for i in range(map_idx.shape[0]):
            for j in range(map_idx.shape[1]):
                new_i = int(i * h_ratio)
                new_j = int(j * w_ratio)
                rgb_seem_id = map_idx[i,j]
                if (depth[new_i, new_j] < 0.1 or depth[new_i, new_j] > 3) and rgb_seem_id != 0:
                    # print(map_idx[new_i, new_j], pc[:,new_i*depthshape[1]+new_j][2])
                    raise Exception(f"Depth filtering is failed: {depth[new_i, new_j]}")
                if depth[new_i, new_j] < 0.1 or depth[new_i, new_j] > 3: continue
                pp = pc_global[:,new_i*depth.shape[1]+new_j]
                x,y = pos2grid_id(self.config["gs"],self.config["cs"],pp[0],pp[2])
                h=pp[1]
                if h > -self.config["camera_height"]:self.obstacles[y,x]=0
                if h > self.color_top_down_height[y,x]:
                    self.color_top_down[y,x] = rgb[new_i,new_j,:]
                    self.color_top_down_height[y,x] = h

    
    def postprocessing(self):
        raise NotImplementedError
    
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
        self.background_grid = np.zeros((self.config["gs"], self.config["gs"], self.feat_dim), dtype=np.float32)
        background_emb = self.model.encode_prompt(["wall","floor"], task = "default")
        background_emb = background_emb.cpu().numpy()
        self.instance_dict = {}
        self.instance_dict[0] = {"embedding":self.pano_emb[0,:], "count":0, "frames":{}, "category_id":0}
        self.instance_dict[1] = {"embedding":self.pano_emb[1,:], "count":0, "frames":{}, "category_id":1}
        self.frame_mask_dict = {}
    
    def save_map(self):
        if self.bool_submap:
            self.datamanager.save_map(color_top_down=self.color_top_down,
                                    grid=self.grid,
                                    obstacles=self.obstacles,
                                    weight=self.weight,
                                    instance_dict=self.instance_dict,
                                    frame_mask_dict=self.frame_mask_dict,
                                    background_grid=self.background_grid)
        else:
            self.datamanager.save_map(grid=self.grid,
                                    instance_dict=self.instance_dict,
                                    frame_mask_dict=self.frame_mask_dict,
                                    background_grid=self.background_grid)