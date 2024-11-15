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
from mapbuilder.map.seemmap import SeemMap
from mapbuilder.utils.datamanager import DataManager, DataManager4Real

class SeemMap_floodfill(SeemMap):
    def __init__(self, config:DictConfig):
        super().__init__(config)


    def processing(self):
        raise NotImplementedError
    
    def postprocessing(self):
        raise NotImplementedError
    
    def preprocessing(self):
        raise NotImplementedError
    
    def _init_map(self):
        raise NotImplementedError
    
    def save_map(self):
        raise NotImplementedError