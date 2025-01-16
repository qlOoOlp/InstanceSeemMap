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
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.data_path = data_path
        self.rgb_path = self.data_path + '/rgb'
        self.depth_path = self.data_path + '/depth'
        self.pose_path = self.data_path + '/pose'
        self.map_path = map_path
        self.version = version
        if not isinstance(self, DataManager4gt) and not isinstance(self, DataManagerRoom):
            self.load_data()
        # create map path if not exists (if exists, do nothing)
        os.makedirs(self.map_path, exist_ok=True)

        self.rectification_matrix = np.eye(3)
        self.rectification_matrix[1,1] = -1
        self.rectification_matrix[2,2] = -1

    def _load_data(self)->None:
        self.check_path(self.data_path, self.rgb_path, self.depth_path, self.pose_path)

        self.rgblist = sorted(os.listdir(self.rgb_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))

        self.depthlist = sorted(os.listdir(self.depth_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self.depthlist = [os.path.join(self.depth_path, x) for x in self.depthlist]

        self.poselist = sorted(os.listdir(self.pose_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self.poselist = [os.path.join(self.pose_path, x) for x in self.poselist]
        # check data length mismatch
        if self.end_frame == -1:
            self.rgblist = [os.path.join(self.rgb_path, x) for x in self.rgblist][self.start_frame:]
            self.depthlist = [os.path.join(self.depth_path, x) for x in self.depthlist][self.start_frame:]
            self.poselist = [os.path.join(self.pose_path, x) for x in self.poselist][self.start_frame:]
        else:
            self.rgblist = [os.path.join(self.rgb_path, x) for x in self.rgblist][self.start_frame:self.end_frame+1]
            self.depthlist = [os.path.join(self.depth_path, x) for x in self.depthlist][self.start_frame:self.end_frame+1]
            self.poselist = [os.path.join(self.pose_path, x) for x in self.poselist][self.start_frame:self.end_frame+1]
        
        if not len(self.rgblist) == len(self.depthlist) == len(self.poselist):
            print(len(self.rgblist), len(self.depthlist), len(self.poselist))
            raise ValueError("Data length mismatch")

        self.numData = len(self.rgblist)
        self.count = -1
        print(f"Data loaded: {self.numData} data")

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
                map_save_path = os.path.join(self.map_path, f"{key}_{self.version}.pkl")
                with open(map_save_path, 'wb') as f:
                    pickle.dump(value, f)
                continue
            map_save_path = os.path.join(self.map_path, f"{key}_{self.version}.npy")
            np.save(map_save_path, value, allow_pickle=True)

    def data_getter(self)->Tuple[NDArray,NDArray,Tuple[NDArray,NDArray]]:
        self.count += 1
        if self.count >= self.numData:
            raise ValueError("Data out of range")
        rgb_dir = self.rgblist[self.count]
        depth_dir = self.depthlist[self.count]
        pose_dir = self.poselist[self.count]
        rgb = rgbLoader(rgb_dir)
        depth = depthLoader(depth_dir)
        pose = poseLoader(pose_dir)
        return rgb, depth, pose

    def check_path(self, *args)->None:
        for path in args:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Invalid path: {path}")
            
    def get_data_shape(self):
        rgb = rgbLoader(self.rgblist[0])
        depth = depthLoader(self.depthlist[0])
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
        return self.numData

    @property
    def count(self):
        return self.count

    @property
    def rectification_matrix(self):
        return self.rectification_matrix

class DataManager4Real(DataManager):
    def __init__(self, version:str, data_path:str, map_path:str, start_frame:int=0, end_frame:int=-1):
        super().__init__(version, data_path, map_path, start_frame, end_frame)
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
        self.projection_matrix = projection_matrix
        self.rectification_matrix = rectification_matrix

    @property
    def projection_matrix(self):
        return self.projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value:NDArray):
        if value.shape != (3, 3):
            raise ValueError("Invalid projection matrix")
        self.projection_matrix = value
    
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
        self.semantic_path = self.data_path + '/semantic'
        self.load_data()
    def _load_data(self)->None:
        self.check_path(self.data_path, self.rgb_path, self.depth_path, self.pose_path, self.semantic_path)

        self.rgblist = sorted(os.listdir(self.rgb_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))

        self.depthlist = sorted(os.listdir(self.depth_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self.depthlist = [os.path.join(self.depth_path, x) for x in self.depthlist]

        self.poselist = sorted(os.listdir(self.pose_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self.poselist = [os.path.join(self.pose_path, x) for x in self.poselist]

        self.semanticlist = sorted(os.listdir(self.semantic_path), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        self.semanticlist = [os.path.join(self.semantic_path, x) for x in self.semanticlist]
    
        if self.end_frame == -1:
            self.rgblist = [os.path.join(self.rgb_path, x) for x in self.rgblist][self.start_frame:]
            self.depthlist = [os.path.join(self.depth_path, x) for x in self.depthlist][self.start_frame:]
            self.poselist = [os.path.join(self.pose_path, x) for x in self.poselist][self.start_frame:]
            self.semanticlist = [os.path.join(self.semantic_path, x) for x in self.semanticlist][self.start_frame:]
        else:
            self.rgblist = [os.path.join(self.rgb_path, x) for x in self.rgblist][self.start_frame:self.end_frame+1]
            self.depthlist = [os.path.join(self.depth_path, x) for x in self.depthlist][self.start_frame:self.end_frame+1]
            self.poselist = [os.path.join(self.pose_path, x) for x in self.poselist][self.start_frame:self.end_frame+1]
            self.semanticlist = [os.path.join(self.semantic_path, x) for x in self.semanticlist][self.start_frame:self.end_frame+1]
        if not len(self.rgblist) == len(self.depthlist) == len(self.poselist) == len(self.semanticlist):
            raise ValueError("Data length mismatch")

        self.numData = len(self.rgblist)
        self.count = -1
        print(f"Data loaded: {self.numData} data")
        
    def data_getter(self)->Tuple[NDArray,NDArray,Tuple[NDArray,NDArray],NDArray]:
        self.count += 1
        if self.count >= self.numData:
            raise ValueError("Data out of range")
        rgb_dir = self.rgblist[self.count]
        depth_dir = self.depthlist[self.count]
        pose_dir = self.poselist[self.count]
        semantic_dir = self.semanticlist[self.count]
        rgb = rgbLoader(rgb_dir)
        depth = depthLoader(depth_dir)
        pose = poseLoader(pose_dir)
        semantic = depthLoader(semantic_dir)
        return rgb, depth, pose, semantic
    

class DataManagerRoom(DataManager):
    def __init__(self, version:str, data_path:str, map_path:str, scene_id:str, start_frame:int=0, end_frame:int=-1):
        super().__init__(version, data_path, map_path, start_frame, end_frame)
        self.label_path = os.path.join(self.data_path, f'{scene_id}_single_label.txt')
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
    
    def get_data_shape(self):
        rgb = rgbLoader(self.rgblist[0])
        depth = depthLoader2(self.depthlist[0])
        return rgb.shape, depth.shape
 
    def load_data(self)->None:
        print(self.rgb_path)
        self.rgblist = sorted(os.listdir(self.rgb_path))
        self.rgblist = [os.path.join(self.rgb_path, x) for x in self.rgblist]
 
        print(self.depth_path)
        self.depthlist = sorted(os.listdir(self.depth_path))
        self.depthlist = [os.path.join(self.depth_path, x) for x in self.depthlist]
 
        print(self.pose_path)
        self.poselist = sorted(os.listdir(self.pose_path))
        self.poselist = [os.path.join(self.pose_path, x) for x in self.poselist]
 
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