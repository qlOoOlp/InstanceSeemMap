import numpy as np
import torch
from omegaconf import DictConfig
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
from map.lseg.modules.models.lseg_net import LSegEncNet
import clip
import os


from map.utils.mapping_utils import project_point, pos2grid_id, transform_pc, depth2pc, depth2pc4Real, get_sim_cam_mat, get_sim_cam_mat4Real
from map.utils.lseg_utils import get_lseg_feats
from map.utils.get_transform import get_transform
from map.lseg.additional_utils.models import resize_image, pad_image, crop_image
from map.mapbuilder.map.map import Map
from map.mapbuilder.utils.datamanager import DataManager, DataManager4Real


CLIP_FEAT_DIM_DICT = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}


class LsegMap(Map):
    def __init__(self, config:DictConfig):
        super().__init__(config)
        self.feat_dim = CLIP_FEAT_DIM_DICT[self.config["clip_version"]]

        self.camera_height = self.config["camera_height"]
        self.gs = self.config["gs"]
        self.cs = self.config["cs"]
        self.crop_size = self.config["crop_size"]
        self.base_size = self.config["base_size"]
        self.depth_sample_rate = self.config["depth_sample_rate"]
        self.min_depth = self.config["min_depth"]
        self.max_depth = self.config["max_depth"]
        self.max_height = float(self.config.get("max_height", 3.0))
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


        self._setup_CLIP()


    def _setup_CLIP(self):
        self.lang = self.config["lang"]
        self.labels = self.lang.split(",")
        self.clip_version = self.config["clip_version"]
        self.clip_feat_dim = CLIP_FEAT_DIM_DICT[self.clip_version]
        self.ckpt = self.config["lseg_ckpt"]
        print(f"Loading LSeg model...")
        model = LSegEncNet(self.lang, arch_option=0,
                           block_depth=0,
                           activation='lrelu',
                           crop_size=self.crop_size)
        model_state_dict = model.state_dict()
        lseg_pretrained_state_dict = torch.load(self.ckpt)
        lseg_pretrained_state_dict = {k.lstrip('net.'): v for k, v in lseg_pretrained_state_dict['state_dict'].items()}
        model_state_dict.update(lseg_pretrained_state_dict)
        model.load_state_dict(model_state_dict)
        model.eval()
        self.model = model.cuda()

        self.transform, self._MEAN, self._STD = get_transform()

        

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
        raise TypeError(f"Unsupported pose type for LsegMap: {type(pose)}")

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
        print("Processing data...")

        # print(self.datamanager.get_init_pose())
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
        while self.datamanager.count < self.datamanager.numData -1:
            rgb, depth, pose = self.datamanager.data_getter()
            tf = self._build_tf(pose)

            # print("step2. get lseg features")
            _, features, _ = get_lseg_feats(self.model, rgb, self.labels, self.crop_size, self.base_size, self.transform, self._MEAN, self._STD)
            # print("step3. processing pc")
            if self.data_type == "rtabmap":
                pc, mask = depth2pc4Real(depth, self.datamanager.projection_matrix, rgb.shape[:2], min_depth=self.min_depth, max_depth=self.max_depth)
                shuffle_mask = np.arange(pc.shape[1])
                np.random.shuffle(shuffle_mask)
                shuffle_mask = shuffle_mask[::self.depth_sample_rate]
                mask = mask[shuffle_mask]
                pc = pc[:, shuffle_mask]
                pc =pc[:, mask]
                if self.pose_type == "mat":
                    pc_global = transform_pc(pc, tf)
                else:
                    pc_transform = tf @ self.base_transform @ self.base2cam_tf
                    pc_global = transform_pc(pc, pc_transform)
                rgb_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2],rgb.shape[:2])
                feat_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], features.shape[2:])
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
                # print(pc.shape)
                shuffle_mask = np.arange(pc.shape[1])
                np.random.shuffle(shuffle_mask)
                shuffle_mask = shuffle_mask[::self.depth_sample_rate]
                mask = mask[shuffle_mask]
                pc = pc[:, shuffle_mask]
                # print(pc.shape)
                pc =pc[:, mask]
                if self.pose_type == "mat":
                    pc_global = transform_pc(pc, tf)
                else:
                    pc_transform = tf @ self.base_transform @ self.base2cam_tf
                    pc_global = transform_pc(pc, pc_transform)
                rgb_cam_mat = get_sim_cam_mat(rgb.shape[0],rgb.shape[1])
                feat_cam_mat = get_sim_cam_mat(features.shape[2], features.shape[3])
            # print("step4. projection")
            # print(pc.shape, pc_global.shape)
            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                x, y, h = self._point_to_map(p)

                # ignore points projected to outside of the map and points that are 0.5 higher than the camera (could be from the ceiling)
                if not self._in_grid(x, y):
                    continue

                # Step4. rgb embedding vector
                rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
                rgb_v = rgb[rgb_py, rgb_px, :]
                # semantic_v = semantic[rgb_py, rgb_px]
                # if semantic_v == 40:
                #     semantic_v = -1

                # when the projected location is already assigned a color value before, overwrite if the current point has larger height
                if h > self.max_height:
                    continue
                if h > self.color_top_down_height[y,x]:
                    self.color_top_down[y,x] = rgb_v
                    self.color_top_down_height[y,x] = h
                px, py, pz = project_point(feat_cam_mat, p_local)
                if not (px < 0 or py < 0 or px >= features.shape[3] or py >= features.shape[2]):
                    feat = features[0, :, py, px]
                    # feat = np.array(feat)
                    self.grid[y, x] = (self.grid[y, x] * self.weight[y, x] + feat) / (self.weight[y, x] + 1)
                    self.weight[y, x] += 1

                # build an obstacle map ignoring points on the floor (0 means occupied, 1 means free)
                if h < 1e-4:
                    continue
                self.obstacles[y,x]=0
            pbar.update(1)
        # self.save_map()
    
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
        self.grid = np.zeros((self.gs, self.gs, self.feat_dim), dtype=np.float32)
        self.obstacles = np.ones((self.gs, self.gs), dtype=np.uint8)
        self.weight = np.zeros((self.gs, self.gs), dtype=np.float32)
    
    def start_map(self):
        self._init_map()
        self.save_map()
    
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
