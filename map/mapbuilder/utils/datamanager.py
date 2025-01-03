import numpy as np
import torch
import yaml
from typing import Dict, Union, Tuple
from numpy.typing import NDArray
import os
import pickle
import json
from omegaconf import OmegaConf, DictConfig

from map.mapbuilder.utils.utils import rgbLoader, depthLoader, poseLoader
# from typing import

class DataManager():
    def __init__(self, version:str, data_path:str, map_path:str, start_frame:int=0, end_frame:int=-1):
        self.__start_frame = start_frame
        self.__end_frame = end_frame
        self.__data_path = data_path
        self.__rgb_path = self.__data_path + '/rgb'
        self.__depth_path = self.__data_path + '/depth'
        self.__pose_path = self.__data_path + '/pose'
        self.__map_path = map_path
        self.__version = version
        self._load_data()
        # create map path if not exists (if exists, do nothing)
        os.makedirs(self.__map_path, exist_ok=True)

        self.__rectification_matrix = np.eye(3)
        self.__rectification_matrix[1,1] = -1
        self.__rectification_matrix[2,2] = -1

    def _load_data(self)->None:
        self.check_path(self.__data_path, self.__rgb_path, self.__depth_path, self.__pose_path)

        self.__rgblist = sorted(os.listdir(self.__rgb_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))

        self.__depthlist = sorted(os.listdir(self.__depth_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self.__depthlist = [os.path.join(self.__depth_path, x) for x in self.__depthlist]

        self.__poselist = sorted(os.listdir(self.__pose_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self.__poselist = [os.path.join(self.__pose_path, x) for x in self.__poselist]
        # check data length mismatch
        if self.__end_frame == -1:
            self.__rgblist = [os.path.join(self.__rgb_path, x) for x in self.__rgblist][self.__start_frame:]
            self.__depthlist = [os.path.join(self.__depth_path, x) for x in self.__depthlist][self.__start_frame:]
            self.__poselist = [os.path.join(self.__pose_path, x) for x in self.__poselist][self.__start_frame:]
        else:
            self.__rgblist = [os.path.join(self.__rgb_path, x) for x in self.__rgblist][self.__start_frame:self.__end_frame+1]
            self.__depthlist = [os.path.join(self.__depth_path, x) for x in self.__depthlist][self.__start_frame:self.__end_frame+1]
            self.__poselist = [os.path.join(self.__pose_path, x) for x in self.__poselist][self.__start_frame:self.__end_frame+1]
        if not len(self.__rgblist) == len(self.__depthlist) == len(self.__poselist):
            raise ValueError("Data length mismatch")

        self.__numData = len(self.__rgblist)
        self.__count = -1
        print(f"Data loaded: {self.__numData} data")

    def load_map(self, *args)->Dict[str, Union[dict,NDArray]]:
        result = []
        for path in args:
            try:
                with open(path, 'rb') as f:
                    temp = np.load(f, allow_pickle=True)
                result.append[temp]
            except:
                try:
                    with open(path, 'rb') as f:
                        temp = pickle.load(f)
                    result.append[temp]
                except: raise ValueError("Invalid map path")
        return result
    
    
    def save_map(self, **kwargs)->None:
        for key, value in kwargs.items():
            if type(value) == type({1:1}):
                map_save_path = os.path.join(self.__map_path, f"{key}_{self.__version}.pkl")
                with open(map_save_path, 'wb') as f:
                    pickle.dump(value, f)
                continue
            map_save_path = os.path.join(self.__map_path, f"{key}_{self.__version}.npy")
            np.save(map_save_path, value, allow_pickle=True)

    def data_getter(self)->Tuple[NDArray,NDArray,Tuple[NDArray,NDArray]]:
        self.__count += 1
        if self.__count >= self.__numData:
            raise ValueError("Data out of range")
        rgb_dir = self.__rgblist[self.__count]
        depth_dir = self.__depthlist[self.__count]
        pose_dir = self.__poselist[self.__count]
        rgb = rgbLoader(rgb_dir)
        depth = depthLoader(depth_dir)
        pose = poseLoader(pose_dir)
        return rgb, depth, pose

    def check_path(self, *args)->None:
        for path in args:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Invalid path: {path}")
            
    def get_data_shape(self):
        rgb = rgbLoader(self.__rgblist[0])
        depth = depthLoader(self.__depthlist[0])
        return rgb.shape, depth.shape

    def managing_temp(self, type: int, **kwargs):
        if type == 0: # set temp path
            return os.path.join(kwargs["temp_dir"], kwargs["temp_name"])
        elif type == 1 : # save temp
            np.save(kwargs["instance_save_path"], kwargs["instance"])
        else: # type == 2, load temp
            return np.load(kwargs["instance_save_path"])

    @property
    def numData(self):
        return self.__numData
    @property
    def count(self):
        return self.__count
    @property
    def rectification_matrix(self):
        return self.__rectification_matrix
    @rectification_matrix.setter
    def rectification_matrix(self, value:NDArray):
        if value.shape != (3, 3):
            raise ValueError("Invalid rectification matrix")
        self.__rectification_matrix = value



class DataManager4Real(DataManager):
    def __init__(self, data_path:str, map_path:str):
        super().__init__(data_path, map_path)
        calib_path = self.data_path + '/calibration.yaml'
        self.check_path(calib_path)
        with open(calib_path, 'r') as f:
            calib = yaml.safe_load(f)
            rectification_matrix = calib.get('rectification_matrix', {}).get('data', [])
            projection_matrix = calib.get('projection_matrix', {}).get('data', [])
            try:
                projection_matrix = np.array(projection_matrix).reshape(3, 4)
                projection_matrix = projection_matrix[:,:3]
            except:
                raise ValueError("Invalid projection matrix")
            try:
                rectification_matrix = np.array(rectification_matrix).reshape(3, 3)
            except:
                raise ValueError("Invalid rectification matrix")
        self.__projection_matrix = projection_matrix
        self.__rectification_matrix = rectification_matrix

    @property
    def projection_matrix(self):
        return self.__projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value:NDArray):
        if value.shape != (3, 3):
            raise ValueError("Invalid projection matrix")
        self.__projection_matrix = value
    
    # @property
    # def rectification_matrix(self):
    #     return self.__rectification_matrix
    
    # @rectification_matrix.setter
    # def rectification_matrix(self, value:NDArray):
    #     if value.shape != (3, 3):
    #         raise ValueError("Invalid rectification matrix")
    #     self.__rectification_matrix = value



class DataLoader():
    def __init__(self, config:DictConfig):
        self.config = config
        self.data_path = os.path.join(self.config["root_path"], f"{self.config['data_type']}/{self.config['scene_id']}")
        self.__map_path = os.path.join(self.data_path, f"map/{self.config['scene_id']}_{self.config['version']}")
        self.load_hparams()
        self.getmap()
        
        
    def load_hparams(self):
        hparam_path = os.path.join(self.__map_path, "hparam.json")
        with open(hparam_path, 'r') as f:
            data = json.load(f)
        self.hparams = OmegaConf.create(data)

    def getmap(self):
        if self.hparams["vlm"] == "seem" and self.hparams["seem_type"] != "base":
            results = self.load_map(os.path.join(self.__map_path, f"color_top_down_{self.config['version']}.npy"),
                                    os.path.join(self.__map_path, f"obstacles_{self.config['version']}.npy"),
                                    os.path.join(self.__map_path, f"weight_{self.config['version']}.npy"),
                                    os.path.join(self.__map_path, f"grid_{self.config['version']}.npy"),
                                    os.path.join(self.__map_path, f"background_grid_{self.config['version']}.npy"),
                                    os.path.join(self.__map_path, f"frame_mask_dict_{self.config['version']}.pkl"),
                                    os.path.join(self.__map_path, f"instance_dict_{self.config['version']}.pkl"))
            self.color_map, self.obstacle_map, self.weight_map, self.grid_map, self.background_grid, self.frame_mask_dict, self.instance_dict = results
        else:
            self.color_map, self.obstacle_map, self.weight_map, self.grid_map = self.load_map(os.path.join(self.__map_path, f"color_top_down_{self.config['version']}.npy"),
                                                                                            os.path.join(self.__map_path, f"obstacles_{self.config['version']}.npy"),
                                                                                            os.path.join(self.__map_path, f"weight_{self.config['version']}.npy"),
                                                                                            os.path.join(self.__map_path, f"grid_{self.config['version']}.npy"))

    def load_map(self, *args)->Dict[str, Union[dict,NDArray]]:
        result = []
        for path in args:
            try:
                with open(path, 'rb') as f:
                    temp = np.load(f, allow_pickle=True)
                result.append(temp)
            except:
                try:
                    with open(path, 'rb') as f:
                        temp = pickle.load(f)
                    result.append(temp)
                except:
                    raise ValueError(f"Invalid map path: {path}")
        return result
    
    def save_map(self, **kwargs)->None:
        for key, value in kwargs.items():
            if type(value) == type({1:1}):
                map_save_path = os.path.join(self.__map_path, f"{key}_{self.config['version']}.pkl")
                with open(map_save_path, 'wb') as f:
                    pickle.dump(value, f)
                continue
            map_save_path = os.path.join(self.__map_path, f"{key}_{self.config['version']}.npy")
            np.save(map_save_path, value, allow_pickle=True)
    