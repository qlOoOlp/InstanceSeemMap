import numpy as np
import torch
import os
from abc import ABC, abstractmethod
from map.mapbuilder.utils.datamanager import DataManager, DataManager4Real, DataManager4gt
from omegaconf import DictConfig


class Map(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = self.config["device"]
        self.data_type = self.config["data_type"]
        self.root_path = self.config["root_path"]
        self.start_frame = self.config["start_frame"]
        self.end_frame = self.config["end_frame"]
        if self.data_type == "habitat_sim":# and self.config["dataset_type"] != "mp3d":
            self.data_path = os.path.join(self.root_path, f"{self.data_type}/{self.config['dataset_type']}/{self.config['scene_id']}")
        else: self.data_path = os.path.join(self.root_path, f"{self.data_type}/{self.config['scene_id']}") 
        self.map_path = os.path.join(self.data_path, f"map/{self.config['scene_id']}_{self.config['version']}/01buildFeatMap/")
        if self.config["only_gt"]:
            self.datamanager = DataManager4gt(version=self.config["version"], data_path=self.data_path, map_path=self.map_path,start_frame=self.start_frame,end_frame=self.end_frame)
        else:
            if self.data_type == "rtabmap":
                self.datamanager = DataManager4Real(version=self.config["version"], data_path=self.data_path, map_path=self.map_path,start_frame=self.start_frame,end_frame=self.end_frame)
            else: self.datamanager = DataManager(version=self.config["version"], data_path=self.data_path, map_path=self.map_path,start_frame=self.start_frame,end_frame=self.end_frame, skip_frames=self.config["skip_frames"], pose_type=self.config["pose_type"])


    @abstractmethod
    def processing(self):
        pass

    @abstractmethod
    def preprocessing(self):
        pass

    @abstractmethod
    def postprocessing(self):
        pass

    @abstractmethod
    def _load_map(self):
        pass

    @abstractmethod
    def _init_map(self):
        pass

    @abstractmethod
    def start_map(self):
        pass

    @abstractmethod
    def save_map(self):
        pass