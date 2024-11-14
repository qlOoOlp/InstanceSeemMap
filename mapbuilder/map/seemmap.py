import numpy as np
import torch
from omegaconf import DictConfig
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
import os

from seem.utils.get_feat import get_SEEM_feat
from seem.base_model import build_vl_model
from utils.mapping_utils import project_point, pos2grid_id, transform_pc, depth2pc, depth2pc4Real, get_sim_cam_mat, get_sim_cam_mat4Real
from utils.get_transform import get_transform
from mapbuilder.map.map import Map
from mapbuilder.utils.datamanager import DataManager, DataManager4Real

class SeemMap(Map):
    def __init__(self, config:DictConfig):
        self.config = config
        self.feat_dim = self.config["feat_dim"]
        self.device = self.config["device"]
        self.data_type = self.config["data_type"]
        self.root_path = self.config["root_path"]
        self.data_path = os.path.join(self.root_path, f"{self.data_type}/{self.config['scene_id']}")
        self.map_path = os.path.join(self.data_path, f"map/{self.config['scene_id']}_{self.config['version']}")
        if self.data_type == "rtabmap":
            self.datamanager = DataManager4Real(version=self.config["version"], data_path=self.data_path, map_path=self.map_path)
        else: self.datamanager = DataManager(version=self.config["version"], data_path=self.data_path, map_path=self.map_path)        
        self._setup_SEEM()
        self.start_map()


    def _setup_SEEM(self):
        rgb_shape, _ = self.datamanager.get_data_shape()
        self.model = build_vl_model("seem", input_size = rgb_shape[0])


    def processing(self):
        tf_list = []
        print("Processing data...")
        pbar = tqdm(range(self.datamanager.numData))
        while self.datamanager.count < self.datamanager.numData:
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
            map_emb = self.ra2pa(map_idx, embeddings)

            if self.data_type == "rtabmap":
                pc, mask = depth2pc4Real(depth, self.datamanager.projection_matrix, rgb.shape[:2], min_depth=0.5, max_depth=self.config["max_depth"])
                shuffle_mask = np.arange(pc.shape[1])
                np.random.shuffle(shuffle_mask)
                shuffle_mask = shuffle_mask[::self.config['depth_sample_rate']]
                mask = mask[shuffle_mask]
                pc = pc[:, shuffle_mask]
                pc = pc[:, mask]
                pc_global = transform_pc(pc, tf)
                rgb_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], rgb.shape[:2])
                feat_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], map_idx.shape)
            else:
                pc, mask = depth2pc(depth, max_depth=self.config["max_depth"])
                shuffle_mask = np.arange(pc.shape[1])
                np.random.shuffle(shuffle_mask)
                shuffle_mask = shuffle_mask[::self.config['depth_sample_rate']]
                mask = mask[shuffle_mask]
                pc = pc[:, shuffle_mask]
                pc = pc[:, mask]
                pc_global = transform_pc(pc, tf)
                rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
                feat_cam_mat = get_sim_cam_mat(map_idx.shape[0], map_idx.shape[1])
            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                x,y = pos2grid_id(self.config['gs'], self.config['cs'], p[0], p[2])
                if x>= self.obstacles.shape[0] or y>= self.obstacles.shape[1] or x<0 or y<0 or p_local[1] < -0.5: continue
                rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
                rgb_v = rgb[rgb_py, rgb_px, :]
                if p_local[1] > self.color_top_down_height[y,x]:
                    self.color_top_down[y,x] = rgb_v
                    self.color_top_down_height[y,x] = p_local[1]
                px, py, pz = project_point(feat_cam_mat, p_local)
                if not (px < 0 or py < 0 or px >= map_emb.shape[1] or py >= map_emb.shape[0]):
                    feat = map_emb[py, px, :]
                    self.grid[y,x]=(self.grid[y,x]*self.weight[y,x]+feat)/(self.weight[y,x]+1)
                    self.weight[y,x]+=1
                if p_local[1] > self.config["camera_height"]:
                    continue
                self.obstacles[y,x]=0
            pbar.update(1)

            
    def ra2pa(self, map_idx, embeddings):
        map_emb = np.zeros((map_idx.shape[0], map_idx.shape[1], self.feat_dim))
        for row in range(map_idx.shape[0]):
            for col in range(map_idx.shape[1]):
                idx = map_idx[row, col]
                if idx != 0:
                    map_emb[row, col] = embeddings[idx]
        return map_emb

    
    def postprocessing(self):
        raise NotImplementedError
    
    def preprocessing(self):
        raise NotImplementedError
    
    def _load_map(self):
        raise NotImplementedError
    
    def _init_map(self):
        self.color_top_down_height = np.zeros((self.config["gs"], self.config["gs"]), dtype=np.float32)
        self.color_top_down = np.zeros((self.config["gs"], self.config["gs"], 3), dtype=np.uint8)
        self.grid = np.zeros((self.config["gs"], self.config["gs"], self.feat_dim), dtype=np.float32)
        self.obstacles = np.ones((self.config["gs"], self.config["gs"]), dtype=np.uint8)
        self.weight = np.zeros((self.config["gs"], self.config["gs"]), dtype=np.float32)
    
    def start_map(self):
        self._init_map()
        self.save_map()
    
    def save_map(self):
        self.datamanager.save_map(color_top_down=self.color_top_down,
                                  grid=self.grid,
                                  obstacles=self.obstacles,
                                  weight=self.weight)