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

from map.seem.utils.get_feat import get_SEEM_feat
from map.seem.base_model import build_vl_model
from map.utils.mapping_utils import project_point, pos2grid_id, transform_pc, depth2pc, depth2pc4Real, get_sim_cam_mat, get_sim_cam_mat4Real
from map.utils.get_transform import get_transform
from map.mapbuilder.map.seemmap import SeemMap
from map.mapbuilder.utils.datamanager import DataManager, DataManager4Real
from map.mapbuilder.utils.preprocessing import IQR, depth_filtering

class ObstacleMap(SeemMap):
    def __init__(self, config:DictConfig):
        super().__init__(config)
        self.bool_submap = self.config["no_submap"]
        self.bool_seemID = self.config["using_seemID"]
        self.bool_upsample = self.config["upsample"]
        self.bool_IQR = self.config["no_IQR"]
        self.bool_postprocess = self.config["no_postprocessing"]

    def processing(self):
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
            self.base2cam_tf[:3, :3] = self.datamanager.rectification_matrix
            self.base2cam_tf[1, 3] = self.camera_height

        pbar = tqdm(range(self.datamanager.numData))
        while self.datamanager.count < self.datamanager.numData-1:
            rgb, depth, pose = self.datamanager.data_getter()
            tf = self._build_tf(pose)

            map_idx, map_conf, embeddings, category_dict = get_SEEM_feat(self.model, rgb, self.threshold_confidence)

            if map_idx is None:
                pbar.update(1)
                continue
            
            map_idx[map_idx != 1] = 0

            if self.data_type == "rtabmap":
                pc, mask = depth2pc4Real(
                    depth,
                    self.datamanager.projection_matrix.copy(),
                    rgb.shape[:2],
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )
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
                    pc, mask = depth2pc(depth, min_depth=self.min_depth, max_depth=self.max_depth)

            if self.pose_type == "mat":
                pc_global = transform_pc(pc, tf)
            else:
                pc_transform = tf @ self.base_transform @ self.base2cam_tf
                pc_global = transform_pc(pc, pc_transform)

            depth_shape = depth.shape
            h_ratio = depth.shape[0] / map_idx.shape[0]
            w_ratio = depth.shape[1] / map_idx.shape[1]
            if self.bool_IQR:
                map_idx = IQR(map_idx, pc, depth_shape, h_ratio, w_ratio, self.max_depth, self.min_depth)
            else:
                map_idx = depth_filtering(map_idx, pc, depth_shape, h_ratio, w_ratio, self.max_depth, self.min_depth)

            if self.bool_submap:
                self.submap_processing(map_idx, depth, rgb, pc_global, h_ratio, w_ratio)
            feat_map_inst_mask = (map_idx == 1).astype(np.uint8)
            feat_map = np.zeros_like(self.grid, dtype=np.float32)
            feat_map_bool = np.zeros_like(self.grid, dtype=bool)
            depth_w = depth_shape[1]
            for i,j in np.argwhere(feat_map_inst_mask == 1):
                new_i = int(i * h_ratio)
                new_j = int(j * w_ratio)
                new_i = min(max(new_i, 0), depth_shape[0] - 1)
                new_j = min(max(new_j, 0), depth_w - 1)
                if depth[new_i,new_j] < self.min_depth or depth[new_i, new_j] > self.max_depth:
                    raise Exception("Depth filtering is failed")
                pp = pc_global[:, new_i * depth_w + new_j]
                x, y, h = self._point_to_map(pp)
                if not self._in_grid(x, y):
                    continue
                feat_map[y, x] = h
                feat_map_bool[y, x] = True
            feat_map_bool = self.denoising(feat_map_bool, self.config["min_size_denoising_after_projection"])
            if np.sum(feat_map_bool) < self.config["threshold_pixelSize"]:
                map_idx[map_idx == 1] = 0
                pbar.update(1)
                continue
            feat_map[feat_map_bool == False] = 0
            self.grid[feat_map_bool] = 1
            pbar.update(1)
        pbar.close()

    def preprocessing(self):
        raise NotImplementedError

            
    def submap_processing(self, map_idx:NDArray, depth:NDArray, rgb:NDArray, pc_global:NDArray, h_ratio:float, w_ratio:float):
        depth_h, depth_w = depth.shape[:2]
        for i in range(map_idx.shape[0]):
            for j in range(map_idx.shape[1]):
                new_i = int(i * h_ratio)
                new_j = int(j * w_ratio)
                new_i = min(max(new_i, 0), depth_h - 1)
                new_j = min(max(new_j, 0), depth_w - 1)
                rgb_seem_id = map_idx[i,j]
                if (depth[new_i, new_j] < self.min_depth or depth[new_i, new_j] > self.max_depth) and rgb_seem_id != 0:
                    print(self.max_depth, self.min_depth)
                    print(self.bool_IQR)
                    raise Exception(f"Depth filtering is failed: {depth[new_i, new_j]}, {rgb_seem_id}")
                if depth[new_i, new_j] < self.min_depth or depth[new_i, new_j] > self.max_depth:
                    continue
                pp = pc_global[:, new_i * depth_w + new_j]
                x, y, h = self._point_to_map(pp)
                if not self._in_grid(x, y):
                    continue
                if h < self.camera_height:
                    self.obstacles[y, x] = 0
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



    
    def _init_map(self):
        if self.bool_submap:
            if self.hm3dsem_mat_mode:
                self.color_top_down_height = -10 * np.ones((self.gs, self.gs), dtype=np.float32)
            else:
                self.color_top_down_height = np.zeros((self.gs, self.gs), dtype=np.float32)
            self.color_top_down = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
            self.obstacles = np.ones((self.gs, self.gs), dtype=np.uint8)
            self.weight = np.zeros((self.gs, self.gs), dtype=np.float32)
        self.grid = np.zeros((self.gs, self.gs), dtype=np.int32)
        background_emb = self.model.encode_prompt(["wall","floor"], task = "default")
        background_emb = background_emb.cpu().numpy()
        self.instance_dict = {}
        self.instance_dict[1] = {"embedding":background_emb[0,:], "count":0, "frames":{}, "category_id":1, "avg_height":5000}
        self.instance_dict[2] = {"embedding":background_emb[1,:], "count":0, "frames":{}, "category_id":2, "avg_height":0}
        self.frame_mask_dict = {}
    
    def save_map(self):
        if self.bool_submap:
            if self.rot_map:
                if self.hm3dsem_mat_mode:
                    self.datamanager.save_map(
                        color_top_down=np.rot90(self.color_top_down, k=1),
                        wall=np.rot90(self.grid, k=1),
                        obstacles=np.rot90(self.obstacles, k=1),
                    )
                else:
                    self.datamanager.save_map(
                        color_top_down=np.transpose(self.color_top_down, (1, 0, 2)),
                        wall=self.grid.T,
                        obstacles=self.obstacles.T,
                    )
            else:
                self.datamanager.save_map(color_top_down=self.color_top_down,
                                        wall=self.grid,
                                        obstacles=self.obstacles)
        else:
            if self.rot_map:
                if self.hm3dsem_mat_mode:
                    self.datamanager.save_map(wall=np.rot90(self.grid, k=1),
                                            obstacles=np.rot90(self.obstacles, k=1))
                else:
                    self.datamanager.save_map(wall=self.grid.T,
                                            obstacles=self.obstacles.T)
            else:
                self.datamanager.save_map(wall=self.grid,
                                        obstacles=self.obstacles)
