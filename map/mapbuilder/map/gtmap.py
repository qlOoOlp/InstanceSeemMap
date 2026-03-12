import numpy as np
import torch
from omegaconf import DictConfig
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
from map.lseg.modules.models.lseg_net import LSegEncNet
import clip
from PIL import Image

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
        self.max_height = float(self.config.get("max_height", 3.0))
        self.rot_map = self.config["no_rot_map"]
        self.pose_type = self.config["pose_type"]
        dataset_key = str(self.config.get("dataset_type_key", self.config.get("dataset_type", ""))).strip().lower()
        self.hm3dsem_mat_mode = self.pose_type == "mat" and dataset_key == "hm3dsem"
        # Keep bbox-aligned behavior: hm3dsem(mat) always applies rectification in pose rotation.
        self.use_mat_rectification = self.hm3dsem_mat_mode or bool(self.config.get("use_mat_rectification", False))
        self.quat_base_transform = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )

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
        raise TypeError(f"Unsupported pose type for gtMap: {type(pose)}")

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

    def processing(self):

        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
        clip_model.eval()
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
            self.base2cam_tf[:3, :3] = self.datamanager.rectification_matrix
            self.base2cam_tf[1, 3] = self.camera_height

        pbar = tqdm(range(self.datamanager.numData))
        while self.datamanager.count < self.datamanager.numData-1:
            rgb, depth, pose, semantic = self.datamanager.data_getter()
            tf = self._build_tf(pose)

            if self.data_type == "rtabmap":
                pc, mask = depth2pc4Real(
                    depth,
                    self.datamanager.projection_matrix.copy(),
                    rgb.shape[:2],
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )
                rgb_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], rgb.shape[:2])
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
                rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])

            shuffle_mask = np.arange(pc.shape[1], dtype=np.int64)
            np.random.shuffle(shuffle_mask)
            shuffle_mask = shuffle_mask[::self.depth_sample_rate]
            mask = mask[shuffle_mask]
            pc = pc[:, shuffle_mask]
            pc = pc[:, mask]
            if pc.shape[1] == 0:
                pbar.update(1)
                continue

            if self.pose_type == "mat":
                pc_global = transform_pc(pc, tf)
            else:
                pc_transform = tf @ self.base_transform @ self.base2cam_tf
                pc_global = transform_pc(pc, pc_transform)

            
            image = clip_preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                clip_features = clip_model.encode_image(image).cpu().numpy().flatten()


            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                x, y, h = self._point_to_map(p)
                if not self._in_grid(x, y) or h > self.max_height:
                    continue
                rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
                rgb_v = rgb[rgb_py, rgb_px, :]
                semantic_v = semantic[rgb_py, rgb_px]
                if semantic_v %1 !=0:
                    raise ValueError("semantic value is not integer")
                if h > self.color_top_down_height[y,x]:
                    self.color_top_down[y,x] = rgb_v
                    self.color_top_down_height[y,x] = h
                    self.gt[y,x] = semantic_v
                self.clip_grid[y, x] = (self.clip_grid[y, x] * self.weight[y, x] + clip_features) / (self.weight[y, x] + 1)
                self.weight[y,x] += 1
                if h < 1e-4:
                    continue
                self.obstacles[y,x] = 0
            pbar.update(1)
        
    def postprocessing(self):
        raise NotImplementedError
    
    def preprocessing(self):
        raise NotImplementedError
    
    def _load_map(self):
        raise NotImplementedError
    
    def _init_map(self):
        if self.hm3dsem_mat_mode:
            self.color_top_down_height = -10 * np.ones((self.gs, self.gs), dtype=np.float32)
        else:
            self.color_top_down_height = np.zeros((self.gs, self.gs), dtype=np.float32)
        self.color_top_down = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.gt = np.zeros((self.gs, self.gs), dtype=int)
        self.obstacles = np.ones((self.gs, self.gs), dtype=np.uint8)
        self.clip_grid = np.zeros((self.gs, self.gs, 512), dtype=float)
        self.weight = np.zeros((self.gs, self.gs), dtype=np.float32)
    
    def start_map(self):
        self._init_map()
        self.save_map()
    
    def save_map(self):
        if self.rot_map:
            if self.hm3dsem_mat_mode:
                self.datamanager.save_map(
                    color_top_down=np.rot90(self.color_top_down, k=1),
                    grid=np.rot90(self.gt, k=1),
                    weight=np.rot90(self.weight, k=1),
                    clip_grid=np.rot90(self.clip_grid, k=1),
                    obstacles=np.rot90(self.obstacles, k=1),
                )
            else:
                self.datamanager.save_map(
                    color_top_down=np.transpose(self.color_top_down, (1, 0, 2)),
                    grid=self.gt.T,
                    weight=self.weight.T,
                    clip_grid=np.transpose(self.clip_grid, (1, 0, 2)),
                    obstacles=self.obstacles.T,
                )
        else:
            self.datamanager.save_map(color_top_down=self.color_top_down,
                                    grid=self.gt,
                                    weight = self.weight,
                                    clip_grid = self.clip_grid,
                                    obstacles=self.obstacles)
