import numpy as np
import torch
import yaml
from typing import Dict, Union, Tuple
from numpy.typing import NDArray
import os
import pickle
import json
from omegaconf import OmegaConf, DictConfig

from map.mapbuilder.utils.utils import rgbLoader, depthLoader, poseLoader, depthLoader2, poseLoader2
# from typing import

class DataManager():
    def __init__(self, version:str, data_path:str, map_path:str, start_frame:int=0, end_frame:int=-1):
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._data_path = data_path
        self._rgb_path = self._data_path + '/rgb'
        self._depth_path = self._data_path + '/depth'
        self._pose_path = self._data_path + '/pose'
        self._map_path = map_path
        self._version = version
        if not isinstance(self, DataManager4gt):
            self._load_data()
        # create map path if not exists (if exists, do nothing)
        os.makedirs(self._map_path, exist_ok=True)

        self._rectification_matrix = np.eye(3)
        self._rectification_matrix[1,1] = -1
        self._rectification_matrix[2,2] = -1

    def _load_data(self)->None:
        self.check_path(self._data_path, self._rgb_path, self._depth_path, self._pose_path)

        self._rgblist = sorted(os.listdir(self._rgb_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))

        self._depthlist = sorted(os.listdir(self._depth_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self._depthlist = [os.path.join(self._depth_path, x) for x in self._depthlist]

        self._poselist = sorted(os.listdir(self._pose_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self._poselist = [os.path.join(self._pose_path, x) for x in self._poselist]
        # check data length mismatch
        if self._end_frame == -1:
            self._rgblist = [os.path.join(self._rgb_path, x) for x in self._rgblist][self._start_frame:]
            self._depthlist = [os.path.join(self._depth_path, x) for x in self._depthlist][self._start_frame:]
            self._poselist = [os.path.join(self._pose_path, x) for x in self._poselist][self._start_frame:]
        else:
            self._rgblist = [os.path.join(self._rgb_path, x) for x in self._rgblist][self._start_frame:self._end_frame+1]
            self._depthlist = [os.path.join(self._depth_path, x) for x in self._depthlist][self._start_frame:self._end_frame+1]
            self._poselist = [os.path.join(self._pose_path, x) for x in self._poselist][self._start_frame:self._end_frame+1]
        
        if not len(self._rgblist) == len(self._depthlist) == len(self._poselist):
            print(len(self._rgblist), len(self._depthlist), len(self._poselist))
            raise ValueError("Data length mismatch")

        self._numData = len(self._rgblist)
        self._count = -1
        print(f"Data loaded: {self._numData} data")

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
                map_save_path = os.path.join(self._map_path, f"{key}_{self._version}.pkl")
                with open(map_save_path, 'wb') as f:
                    pickle.dump(value, f)
                continue
            map_save_path = os.path.join(self._map_path, f"{key}_{self._version}.npy")
            np.save(map_save_path, value, allow_pickle=True)

    def data_getter(self)->Tuple[NDArray,NDArray,Tuple[NDArray,NDArray]]:
        self._count += 1
        if self._count >= self._numData:
            raise ValueError("Data out of range")
        rgb_dir = self._rgblist[self._count]
        depth_dir = self._depthlist[self._count]
        pose_dir = self._poselist[self._count]
        rgb = rgbLoader(rgb_dir)
        depth = depthLoader(depth_dir)
        pose = poseLoader(pose_dir)
        return rgb, depth, pose

    def check_path(self, *args)->None:
        for path in args:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Invalid path: {path}")
            
    def get_data_shape(self):
        rgb = rgbLoader(self._rgblist[0])
        depth = depthLoader(self._depthlist[0])
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
        return self._numData

    @property
    def count(self):
        return self._count

    @property
    def rectification_matrix(self):
        return self._rectification_matrix

class DataManager4Real(DataManager):
    def __init__(self, version:str, data_path:str, map_path:str, start_frame:int=0, end_frame:int=-1):
        super().__init__(version, data_path, map_path, start_frame, end_frame)
        calib_path = self._data_path + '/calibration.yaml'
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
        self._projection_matrix = projection_matrix
        self._rectification_matrix = rectification_matrix

    @property
    def projection_matrix(self):
        return self._projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value:NDArray):
        if value.shape != (3, 3):
            raise ValueError("Invalid projection matrix")
        self._projection_matrix = value
    
    # @property
    # def rectification_matrix(self):
    #     return self.__rectification_matrix
    
    # @rectification_matrix.setter
    # def rectification_matrix(self, value:NDArray):
    #     if value.shape != (3, 3):
    #         raise ValueError("Invalid rectification matrix")
    #     self.__rectification_matrix = value


class DataManager4gt(DataManager):
    def __init__(self, version:str, data_path:str, map_path:str, start_frame:int=0, end_frame:int=-1):
        super().__init__(version, data_path, map_path, start_frame, end_frame)
        self._semantic_path = self._data_path + '/semantic'
        self._load_data()
    def _load_data(self)->None:
        self.check_path(self._data_path, self._rgb_path, self._depth_path, self._pose_path, self._semantic_path)

        self._rgblist = sorted(os.listdir(self._rgb_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))

        self._depthlist = sorted(os.listdir(self._depth_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self._depthlist = [os.path.join(self._depth_path, x) for x in self._depthlist]

        self._poselist = sorted(os.listdir(self._pose_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self._poselist = [os.path.join(self._pose_path, x) for x in self._poselist]

        self._semanticlist = sorted(os.listdir(self._semantic_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self._semanticlist = [os.path.join(self._semantic_path, x) for x in self._semanticlist]
    
        if self._end_frame == -1:
            self._rgblist = [os.path.join(self._rgb_path, x) for x in self._rgblist][self._start_frame:]
            self._depthlist = [os.path.join(self._depth_path, x) for x in self._depthlist][self._start_frame:]
            self._poselist = [os.path.join(self._pose_path, x) for x in self._poselist][self._start_frame:]
            self._semanticlist = [os.path.join(self._semantic_path, x) for x in self._semanticlist][self._start_frame:]
        else:
            self._rgblist = [os.path.join(self._rgb_path, x) for x in self._rgblist][self._start_frame:self._end_frame+1]
            self._depthlist = [os.path.join(self._depth_path, x) for x in self._depthlist][self._start_frame:self._end_frame+1]
            self._poselist = [os.path.join(self._pose_path, x) for x in self._poselist][self._start_frame:self._end_frame+1]
            self._semanticlist = [os.path.join(self._semantic_path, x) for x in self._semanticlist][self._start_frame:self._end_frame+1]
        if not len(self._rgblist) == len(self._depthlist) == len(self._poselist) == len(self._semanticlist):
            raise ValueError("Data length mismatch")

        self._numData = len(self._rgblist)
        self._count = -1
        print(f"Data loaded: {self._numData} data")
        
    def data_getter(self)->Tuple[NDArray,NDArray,Tuple[NDArray,NDArray],NDArray]:
        self._count += 1
        if self._count >= self._numData:
            raise ValueError("Data out of range")
        rgb_dir = self._rgblist[self._count]
        depth_dir = self._depthlist[self._count]
        pose_dir = self._poselist[self._count]
        semantic_dir = self._semanticlist[self._count]
        rgb = rgbLoader(rgb_dir)
        depth = depthLoader(depth_dir)
        pose = poseLoader(pose_dir)
        semantic = depthLoader(semantic_dir)
        return rgb, depth, pose, semantic
    

class DataManagerRoom(DataManager):
    def __init__(self, version:str, data_path:str, map_path:str, scene_id:str):
        super().__init__(version, data_path, map_path)
        self.label_path = os.path.join(self._data_path, f'{scene_id}_single_label.txt')
        self.check_path(self.label_path)
        # self.__rgb_path = self.data_path + '/rgb'
        # self.__depth_path = self.data_path + '/depth'
        # self.check_path(self.__depth_path)
        # self.check_path(self.__pose_path)
        self.load_data()
 
        # calib_path = self.data_path + '/calibration.yaml'
        # self.check_path(calib_path)
 
        self.rectification_matrix = np.eye(3)
        self.rectification_matrix[1,1] = -1
        self.rectification_matrix[2,2] = -1
 
    def load_data(self)->None:
        print(self._rgb_path)
        self.rgblist = sorted(os.listdir(self._rgb_path))
        self.rgblist = [os.path.join(self._rgb_path, x) for x in self.rgblist]
 
        print(self._depth_path)
        self.depthlist = sorted(os.listdir(self._depth_path))
        self.depthlist = [os.path.join(self._depth_path, x) for x in self.depthlist]
 
        print(self.pose_path)
        self.poselist = sorted(os.listdir(self._pose_path))
        self.poselist = [os.path.join(self._pose_path, x) for x in self.poselist]
 
        print(self.label_path)
        with open(self.label_path, "r") as label_txt:
            self.labellist = [int(line_content.split(" ")[1]) for line_content in label_txt]
 
        # check data length mismatch
        if not len(self.rgblist) == len(self.depthlist) == len(self.poselist):
            raise ValueError("Data length mismatch")
        self.numData = len(self.rgblist)
        self.count = -1
        print(f"Data loaded: {self.numData} data")   
 
    def data_getter(self)->Tuple[NDArray,NDArray,Tuple[NDArray,NDArray]]:
        self.count += 1
        if self.count >= self.numData:
            raise ValueError("Data out of range")
        rgb_dir = self.rgblist[self.count]
        depth_dir = self.depthlist[self.count]
        pose_dir = self.poselist[self.count]
        rgb = rgbLoader(rgb_dir)
        depth = depthLoader2(depth_dir)
        pose = poseLoader2(pose_dir)
        gt_label = self.labellist[self.count]
        return rgb, depth, pose, gt_label, rgb_dir




class DataLoader():
    def __init__(self, config:DictConfig):
        self.config = config
        self.data_path = os.path.join(self.config["root_path"], f"{self.config['data_type']}/{self.config['dataset_type']}/{self.config['scene_id']}")
        self._map_path = os.path.join(self.data_path, f"map/{self.config['scene_id']}_{self.config['version']}")
        self.load_hparams()
        self.getmap()
        
    def load_hparams(self):
        hparam_path = os.path.join(self._map_path, "hparam.json")
        with open(hparam_path, 'r') as f:
            data = json.load(f)
        self.hparams = OmegaConf.create(data)

    def getmap(self):
        if self.hparams["vlm"] == "seem" and self.hparams["seem_type"] != "base":
            results = self.load_map(os.path.join(self._map_path, f"color_top_down_{self.config['version']}.npy"),
                                    os.path.join(self._map_path, f"obstacles_{self.config['version']}.npy"),
                                    os.path.join(self._map_path, f"weight_{self.config['version']}.npy"),
                                    os.path.join(self._map_path, f"grid_{self.config['version']}.npy"),
                                    os.path.join(self._map_path, f"frame_mask_dict_{self.config['version']}.pkl"),
                                    os.path.join(self._map_path, f"instance_dict_{self.config['version']}.pkl"))
            self.color_map, self.obstacle_map, self.weight_map, self.grid_map, self.frame_mask_dict, self.instance_dict = results
        else:
            self.color_map, self.obstacle_map, self.weight_map, self.grid_map = self.load_map(os.path.join(self._map_path, f"color_top_down_{self.config['version']}.npy"),
                                                                                            os.path.join(self._map_path, f"obstacles_{self.config['version']}.npy"),
                                                                                            os.path.join(self._map_path, f"weight_{self.config['version']}.npy"),
                                                                                            os.path.join(self._map_path, f"grid_{self.config['version']}.npy"))

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
                map_save_path = os.path.join(self._map_path, f"{key}_{self.config['version']}.pkl")
                with open(map_save_path, 'wb') as f:
                    pickle.dump(value, f)
                continue
            map_save_path = os.path.join(self._map_path, f"{key}_{self.config['version']}.npy")
            np.save(map_save_path, value, allow_pickle=True)