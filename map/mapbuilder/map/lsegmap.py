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

        


    def processing(self):
        tf_list = []
        print("Processing data...")
        pbar = tqdm(range(self.datamanager.numData))
        while self.datamanager.count < self.datamanager.numData -1:
            # print("step1. load data")
            rgb, depth, (pos,rot) = self.datamanager.data_getter()
            rot = rot @ self.datamanager.rectification_matrix
            pos[1] += self.camera_height
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = pos.reshape(-1)

            tf_list.append(pose)
            if len(tf_list) == 1:
                init_tf_inv = np.linalg.inv(tf_list[0])
            tf = init_tf_inv @ pose

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
                pc_global = transform_pc(pc, tf)
                rgb_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2],rgb.shape[:2])
                feat_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], features.shape[2:])
            else:
                pc, mask = depth2pc(depth, min_depth=self.min_depth, max_depth=self.max_depth)
                # print(pc.shape)
                shuffle_mask = np.arange(pc.shape[1])
                np.random.shuffle(shuffle_mask)
                shuffle_mask = shuffle_mask[::self.depth_sample_rate]
                mask = mask[shuffle_mask]
                pc = pc[:, shuffle_mask]
                # print(pc.shape)
                pc =pc[:, mask]
                pc_global = transform_pc(pc, tf)
                rgb_cam_mat = get_sim_cam_mat(rgb.shape[0],rgb.shape[1])
                feat_cam_mat = get_sim_cam_mat(features.shape[2], features.shape[3])
            # print("step4. projection")
            # print(pc.shape, pc_global.shape)
            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                x, y = pos2grid_id(self.gs, self.cs, p[0], p[2])

                # ignore points projected to outside of the map and points that are 0.5 higher than the camera (could be from the ceiling)
                if x >= self.obstacles.shape[0] or y >= self.obstacles.shape[1] or \
                    x < 0 or y < 0 or p_local[1] < -0.5:
                    continue

                # Step4. rgb embedding vector
                rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
                rgb_v = rgb[rgb_py, rgb_px, :]
                # semantic_v = semantic[rgb_py, rgb_px]
                # if semantic_v == 40:
                #     semantic_v = -1

                # when the projected location is already assigned a color value before, overwrite if the current point has larger height
                if p_local[1] < self.color_top_down_height[y,x]:
                    self.color_top_down[y,x] = rgb_v
                    self.color_top_down_height[y,x] = p_local[1]
                px, py, pz = project_point(feat_cam_mat, p_local)
                if not (px < 0 or py < 0 or px >= features.shape[3] or py >= features.shape[2]):
                    feat = features[0, :, py, px]
                    # feat = np.array(feat)
                    self.grid[y, x] = (self.grid[y, x] * self.weight[y, x] + feat) / (self.weight[y, x] + 1)
                    self.weight[y, x] += 1

                # build an obstacle map ignoring points on the floor (0 means occupied, 1 means free)
                if p_local[1] > self.camera_height:
                    continue
                self.obstacles[y,x]=1
            pbar.update(1)
        # self.save_map()
    
    def postprocessing(self):
        raise NotImplementedError
    
    def preprocessing(self):
        raise NotImplementedError
    
    def _load_map(self):
        raise NotImplementedError
    
    def _init_map(self):
        self.color_top_down_height = (self.camera_height + 1) * np.ones((self.gs, self.gs), dtype=np.float32)
        self.color_top_down = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.grid = np.zeros((self.gs, self.gs, self.feat_dim), dtype=np.float32)
        self.obstacles = np.zeros((self.gs, self.gs), dtype=np.uint8)
        self.weight = np.zeros((self.gs, self.gs), dtype=np.float32)
    
    def start_map(self):
        self._init_map()
        self.save_map()
    
    def save_map(self):
        self.datamanager.save_map(color_top_down=self.color_top_down,
                                  grid=self.grid,
                                  obstacles=self.obstacles,
                                  weight=self.weight)