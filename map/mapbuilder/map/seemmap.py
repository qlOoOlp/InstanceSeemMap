import numpy as np
import torch
from omegaconf import DictConfig
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
import os

from map.seem.utils.get_feat import get_SEEM_feat
from map.seem.base_model import build_vl_model
from map.utils.mapping_utils import project_point, pos2grid_id, transform_pc, depth2pc, depth2pc4Real, get_sim_cam_mat, get_sim_cam_mat4Real
# from utils.get_transform import get_transform
from map.mapbuilder.map.map import Map
# from mapbuilder.utils.datamanager import DataManager, DataManager4Real
from abc import abstractmethod

class SeemMap(Map):
    def __init__(self, config:DictConfig):
        super().__init__(config)
        self.feat_dim = self.config["feat_dim"]

        self.camera_height = self.config["camera_height"]
        self.gs = self.config["gs"]
        self.cs = self.config["cs"]
        self.depth_sample_rate = self.config["depth_sample_rate"]
        self.downsampling_ratio = self.config["downsampling_ratio"]
        self.threshold_confidence = self.config["threshold_confidence"]
        self.max_depth = self.config["max_depth"]
        self.min_depth = self.config["min_depth"]
        self.pose_type = self.config["pose_type"]
        self.rot_map = self.config["no_rot_map"]
        dataset_key = str(self.config.get("dataset_type_key", self.config.get("dataset_type", ""))).strip().lower()
        self.hm3dsem_mat_mode = self.pose_type == "mat" and dataset_key == "hm3dsem"
        # Keep bbox-aligned behavior: hm3dsem(mat) always applies rectification in pose rotation.
        self.use_mat_rectification = self.hm3dsem_mat_mode or bool(self.config.get("use_mat_rectification", False))

        self.quat_base_transform = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )

        self._setup_SEEM()

    def _setup_SEEM(self):
        rgb_shape, _ = self.datamanager.get_data_shape()
        print(f"SEEM input size : {rgb_shape[0]*self.downsampling_ratio}")
        self.model = build_vl_model("seem", input_size=int(rgb_shape[0] * self.downsampling_ratio), device=self.device,)

    def _pose_to_tf(self, pose):
        if isinstance(pose, np.ndarray):
            if pose.shape == (4, 4):
                return pose.copy()
            raise ValueError(f"Invalid pose matrix shape: {pose.shape}")
        if isinstance(pose, (list, tuple)) and len(pose) == 2:
            pose_tf = np.eye(4, dtype=float)
            pose_tf[:3, :3] = pose[1].copy()
            pose_tf[:3, 3] = pose[0].copy().reshape(-1)
            return pose_tf
        raise TypeError(f"Unsupported pose type for SeemMap: {type(pose)}")

    def _build_tf(self, pose):
        if self.pose_type == "mat":
            tf = self._pose_to_tf(pose)
            if self.use_mat_rectification:
                tf[:3, :3] = tf[:3, :3] @ self.datamanager.rectification_matrix
            return tf

        pose_tf = self._pose_to_tf(pose)
        base_pose = self.base_transform @ pose_tf @ np.linalg.inv(self.base_transform)
        return self.inv_init_base_tf @ base_pose

    def _point_to_map(self, p):
        if self.hm3dsem_mat_mode:
            # Match seemmap_bbox4hm3d22 convention used in seemmap_bbox.
            x_m = float(p[0])
            y_m = float(p[2])
            h = float(p[1]) + self.camera_height
        else:
            x_m = float(p[0])
            y_m = float(p[1])
            h = float(p[2])
            if self.pose_type == "mat":
                h += self.camera_height
        x, y = pos2grid_id(self.gs, self.cs, x_m, y_m)
        return x, y, h

    def _in_grid(self, x, y):
        return 0 <= x < self.gs and 0 <= y < self.gs

    # @abstractmethod
    def processing(self):
        print("Processing data...")

        if self.pose_type == "mat":
            init_pose_mat = self._pose_to_tf(self.datamanager.get_init_pose())
            # hm3dsem(mat) keeps DataManager default rectification to mirror seemmap_bbox.
            if self.use_mat_rectification and not self.hm3dsem_mat_mode:
                self.datamanager.rectification_matrix = np.linalg.inv(init_pose_mat[:3, :3])
        else:
            b_pos, b_rot = self.datamanager.get_init_pose()
            base_pose = np.eye(4)
            base_pose[:3, :3] = b_rot
            base_pose[:3, 3] = b_pos.reshape(-1)
            self.init_base_tf = base_pose
            self.base_transform = self.quat_base_transform.copy()
            self.init_base_tf = self.base_transform @ self.init_base_tf @ np.linalg.inv(self.base_transform)
            self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
            self.base2cam_tf = np.eye(4)
            self.base2cam_tf[:3,:3] = self.datamanager.rectification_matrix
            self.base2cam_tf[1,3] = self.camera_height
            self.init_cam_tf = self.init_base_tf @ self.base2cam_tf
            self.inv_init_cam_tf = np.linalg.inv(self.init_cam_tf)

        pbar = tqdm(range(self.datamanager.numData))
        while self.datamanager.count < self.datamanager.numData-1:
            rgb, depth, pose = self.datamanager.data_getter()
            tf = self._build_tf(pose)

            map_idx, map_conf, embeddings, category_dict = get_SEEM_feat(self.model, rgb, self.threshold_confidence)
            if map_idx is None:
                pbar.update(1)
                continue

            map_emb = self.ra2pa(map_idx, embeddings)

            if self.data_type == "rtabmap":
                pc, mask = depth2pc4Real(
                    depth,
                    self.datamanager.projection_matrix,
                    rgb.shape[:2],
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )
                shuffle_mask = np.arange(pc.shape[1])
                np.random.shuffle(shuffle_mask)
                shuffle_mask = shuffle_mask[::self.depth_sample_rate]
                mask = mask[shuffle_mask]
                pc = pc[:, shuffle_mask]
                pc = pc[:, mask]
                if self.pose_type == "mat":
                    pc_global = transform_pc(pc, tf)
                else:
                    pc_transform = tf @ self.base_transform @ self.base2cam_tf
                    pc_global = transform_pc(pc, pc_transform)
                rgb_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], rgb.shape[:2])
                feat_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], map_idx.shape)
            else:
                if self.pose_type == "mat":
                    if self.hm3dsem_mat_mode:
                        pc, mask = depth2pc(
                            depth,
                            max_depth=self.max_depth,
                            min_depth=self.min_depth,
                            depth_scale=1000,
                        )
                    else:
                        pc, mask = depth2pc(
                            depth,
                            intr_mat=np.array([[600, 0, 599.5], [0, 600, 339.5], [0, 0, 1]]),
                            max_depth=self.max_depth,
                            min_depth=self.min_depth,
                            depth_scale=6553.5,
                        )

                else:
                    pc, mask = depth2pc(depth, max_depth=self.max_depth, min_depth=self.min_depth)
                shuffle_mask = np.arange(pc.shape[1])
                np.random.shuffle(shuffle_mask)
                shuffle_mask = shuffle_mask[::self.depth_sample_rate]
                mask = mask[shuffle_mask]
                pc = pc[:, shuffle_mask]
                pc = pc[:, mask]
                if self.pose_type == "mat":
                    pc_global = transform_pc(pc, tf)
                else:
                    pc_transform = tf @ self.base_transform @ self.base2cam_tf
                    pc_global = transform_pc(pc, pc_transform)
                rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
                feat_cam_mat = get_sim_cam_mat(map_idx.shape[0], map_idx.shape[1])
            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                x, y, h = self._point_to_map(p)
                if not self._in_grid(x, y):
                    continue
                rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
                rgb_v = rgb[rgb_py, rgb_px, :]
                if h > self.color_top_down_height[y,x]:
                    self.color_top_down[y,x] = rgb_v
                    self.color_top_down_height[y,x] = h
                px, py, pz = project_point(feat_cam_mat, p_local)
                if not (px < 0 or py < 0 or px >= map_emb.shape[1] or py >= map_emb.shape[0]):
                    feat = map_emb[py, px, :]
                    self.grid[y,x]=(self.grid[y,x]*self.weight[y,x]+feat)/(self.weight[y,x]+1)
                    self.weight[y,x]+=1
                if h < 1e-4:
                    continue
                self.obstacles[y,x]=0
            pbar.update(1)

    # Convert region aligned feature to pixel aligned feature
    def ra2pa(self, map_idx, embeddings):
        map_emb = np.zeros((map_idx.shape[0], map_idx.shape[1], self.feat_dim))
        for row in range(map_idx.shape[0]):
            for col in range(map_idx.shape[1]):
                idx = map_idx[row, col]
                if idx != 0:
                    map_emb[row, col] = embeddings[idx]
        return map_emb

    
    # @abstractmethod
    def postprocessing(self):
        raise NotImplementedError
    
    # @abstractmethod
    def preprocessing(self):
        raise NotImplementedError
    
    def _load_map(self):
        raise NotImplementedError
    
    # @abstractmethod
    def _init_map(self):
        if self.hm3dsem_mat_mode:
            self.color_top_down_height = -10 * np.ones((self.gs, self.gs), dtype=np.float32)
        else:
            self.color_top_down_height = np.zeros((self.gs, self.gs), dtype=np.float32)
        self.color_top_down = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.grid = np.zeros((self.gs, self.gs, self.feat_dim), dtype=np.float32)
        self.obstacles = np.ones((self.gs, self.gs), dtype=np.uint8)
        self.weight = np.zeros((self.gs, self.gs), dtype=np.float32)
    
    def start_map(self):
        self._init_map()
        self.save_map()
    
    # @abstractmethod
    def save_map(self):
        if self.rot_map:
            if self.hm3dsem_mat_mode:
                self.datamanager.save_map(
                    color_top_down=np.rot90(self.color_top_down, k=1),
                    grid=np.rot90(self.grid, k=1),
                    obstacles=np.rot90(self.obstacles, k=1),
                    weight=np.rot90(self.weight, k=1),
                )
            else:
                self.datamanager.save_map(
                    color_top_down=np.transpose(self.color_top_down, (1, 0, 2)),
                    grid=np.transpose(self.grid, (1, 0, 2)),
                    obstacles=self.obstacles.T,
                    weight=self.weight.T,
                )
        else:
            self.datamanager.save_map(color_top_down=self.color_top_down,
                                    grid=self.grid,
                                    obstacles=self.obstacles,
                                    weight=self.weight)
