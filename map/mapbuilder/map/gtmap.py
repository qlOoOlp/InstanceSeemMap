import numpy as np
import torch
from omegaconf import DictConfig
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
from map.lseg.modules.models.lseg_net import LSegEncNet


from map.utils.mapping_utils import project_point, pos2grid_id, transform_pc, depth2pc, depth2pc4Real, get_sim_cam_mat, get_sim_cam_mat4Real
from map.utils.lseg_utils import get_lseg_feats
from map.utils.get_transform import get_transform
from map.mapbuilder.map.map import Map
from map.mapbuilder.utils.datamanager import DataManager, DataManager4Real


class gtMap(Map):
    def __init__(self, config:DictConfig):
        super().__init__(config)
        self.camera_height = self.config["camera_height"]
        self.gs = self.config["gs"]
        self.cs = self.config["cs"]
        self.depth_sample_rate = self.config["depth_sample_rate"]
        self.min_depth = self.config["min_depth"]
        self.max_depth = self.config["max_depth"]

    def processing(self):
        print("Processing data...")
        b_pos, b_rot = self.datamanager.get_init_pose()
        base_pose = np.eye(4)
        base_pose[:3, :3] = b_rot
        base_pose[:3, 3] = b_pos.reshape(-1)
        # print(base_pose)
        self.init_base_tf = base_pose
        self.base_transform = np.array([[0,0,-1,0],[-1,0,0,0],[0,1,0,0],[0,0,0,1]])
        # print(self.init_base_tf)
        self.init_base_tf = self.base_transform @ self.init_base_tf @ np.linalg.inv(self.base_transform)
        # print(self.init_base_tf)
        self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        self.base2cam_tf = np.array([[1,0,0,0],[0,-1,0,1.5],[0,0,-1,0],[0,0,0,1]])
        self.init_cam_tf = self.init_base_tf @ self.base2cam_tf
        self.inv_init_cam_tf = np.linalg.inv(self.init_cam_tf)
        pbar = tqdm(range(self.datamanager.numData))
        while self.datamanager.count < self.datamanager.numData-1:
            rgb, depth, (pos,rot), semantic = self.datamanager.data_getter()
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = pos.reshape(-1)

            base_pose = self.base_transform @ pose @ np.linalg.inv(self.base_transform)
            tf = self.inv_init_base_tf @ base_pose
            pc, mask = depth2pc(depth, min_depth=self.min_depth, max_depth=self.max_depth)
            shuffle_mask = np.arange(pc.shape[1])
            np.random.shuffle(shuffle_mask)
            shuffle_mask = shuffle_mask[::self.depth_sample_rate]
            mask = mask[shuffle_mask]
            pc = pc[:, shuffle_mask]
            # print(pc.shape)
            pc =pc[:, mask]
            pc_transform = tf @ self.base_transform @ self.base2cam_tf
            pc_global = transform_pc(pc, pc_transform)
            rgb_cam_mat = get_sim_cam_mat(rgb.shape[0],rgb.shape[1])
            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                # if p_local[2] == 0: continue
                x, y = pos2grid_id(self.gs, self.cs, p[0], p[1])
                if x >= self.obstacles.shape[0] or y >= self.obstacles.shape[1] or \
                    x < 0 or y < 0 or p[2] > 2: #2
                    continue
                rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
                rgb_v = rgb[rgb_py, rgb_px, :]
                semantic_v = semantic[rgb_py, rgb_px]
                if semantic_v %1 !=0:
                    raise ValueError("semantic value is not integer")
                if p[2] > self.color_top_down_height[y,x]:
                    self.color_top_down[y,x] = rgb_v
                    self.color_top_down_height[y,x] = p[2]
                    self.gt[y,x] = semantic_v
                if p[2] < 1e-4:#self.camera_height:
                    continue
                self.obstacles[y,x] = 1
            pbar.update(1)
        
    def postprocessing(self):
        raise NotImplementedError
    
    def preprocessing(self):
        raise NotImplementedError
    
    def _load_map(self):
        raise NotImplementedError
    
    def _init_map(self):
        self.color_top_down_height = np.zeros((self.gs, self.gs), dtype=np.float32) #(self.camera_height + 1) * np.ones((self.gs, self.gs), dtype=np.float32)
        self.color_top_down = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.gt = np.zeros((self.gs, self.gs), dtype=int)
        self.obstacles = np.zeros((self.gs, self.gs), dtype=np.uint8)
    
    def start_map(self):
        self._init_map()
        self.save_map()
    
    def save_map(self):
        self.datamanager.save_map(color_top_down=self.color_top_down,
                                  grid=self.gt,
                                  obstacles=self.obstacles)
