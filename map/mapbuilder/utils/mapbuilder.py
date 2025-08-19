import numpy as np
import torch
from omegaconf import DictConfig
import os
import clip
from sklearn.cluster import DBSCAN

from map.utils.matterport3d_categories import mp3dcat
from map.utils.replica_categories import replica_cat
from map.utils.clip_utils import get_text_feats
from map.seem.base_model import build_vl_model
from map.mapbuilder.utils.datamanager import DataManager, DataManager4Real, DataLoader
from map.mapbuilder.map.lsegmap import LsegMap
from map.mapbuilder.map.seemmap import SeemMap
from map.mapbuilder.map.seemmap_tracking import SeemMap_tracking
from map.mapbuilder.map.seemmap_bbox import SeemMap_bbox
from map.mapbuilder.map.seemmap_dbscan import SeemMap_dbscan
from map.mapbuilder.map.seemmap_floodfill import SeemMap_floodfill
from map.mapbuilder.map.obstaclemap import ObstacleMap
from map.mapbuilder.map.seemmap_bbox4hovsg import SeemMap_bbox4hovsg
from map.mapbuilder.map.seemmap_bbox4hm3d import SeemMap_bbox4hm3d
from map.mapbuilder.map.seemmap_bbox4hm3d22 import SeemMap_bbox4hm3d22
from map.mapbuilder.map.gtmap import gtMap
from PIL import Image
import matplotlib.pyplot as plt

CLIP_FEAT_DIM_DICT = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}

class MapBuilder():
    def __init__(self, conf:DictConfig):
        self.conf = conf
        self.vlm = self.conf["vlm"]
        if self.conf["only_gt"]:
            self.map = gtMap(self.conf)
        elif self.vlm == "lseg":
            self.map = LsegMap(self.conf)
        elif self.vlm == "seem":
            if self.conf["seem_type"]=="base": self.map = SeemMap(self.conf)
            elif self.conf["seem_type"]=="tracking" : self.map = SeemMap_tracking(self.conf)
            elif self.conf["seem_type"]=="bbox" : self.map = SeemMap_bbox(self.conf)
            elif self.conf["seem_type"]=="bbox4hovsg": self.map = SeemMap_bbox4hovsg(self.conf)
            elif self.conf["seem_type"]=="bbox4hm3d": self.map = SeemMap_bbox4hm3d(self.conf)
            elif self.conf["seem_type"]=="bbox4hm3d22": self.map = SeemMap_bbox4hm3d22(self.conf)
            elif self.conf["seem_type"]=="dbscan" : self.map = SeemMap_dbscan(self.conf)
            elif self.conf["seem_type"]=="floodfill" : self.map = SeemMap_floodfill(self.conf)
            elif self.conf["seem_type"]=="obstacle": self.map = ObstacleMap(self.conf)
    def buildmap(self):
        print("#"*100)
        self.map.start_map()
        self.map.processing()
        self.map.save_map()
        print("="*(len("Map building done")+10),"Map building done","#"*100,sep='\n')





class CategorizedMapBuilder():
    def __init__(self, conf:DictConfig):
        self.conf = conf
        self.dataloader = DataLoader(self.conf)
        x_indices, y_indices = np.where(self.dataloader.obstacle_map == 0)
        self.xmin, self.xmax, self.ymin, self.ymax = np.min(x_indices), np.max(x_indices), np.min(y_indices), np.max(y_indices)
        self.others = self.conf["room_items"]
        if self.xmin == 0 and self.ymin == 0:
            x_indices, y_indices = np.where(self.dataloader.obstacle_map == 1)
            self.xmin, self.xmax, self.ymin, self.ymax = np.min(x_indices), np.max(x_indices), np.min(y_indices), np.max(y_indices)
        self.obstacles = self.conf['obstacle_items']
        self.max_obs_height = self.conf['max_obs_height']

    def processing(self):
        self.wall_mask = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=bool)
        self.door_mask = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=bool)
        self.window_mask = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=bool)
        self.others_mask = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=bool)
        self.new_mask = np.ones((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=bool)
        self.new_mask[self.dataloader.weight_map==0] = 0 #!
        for i in range(self.dataloader.grid_map.shape[0]):
            for j in range(self.dataloader.grid_map.shape[1]): 
                if 1 in self.dataloader.grid_map[i,j]:
                    self.wall_mask[i,j] = 1
        self.model = build_vl_model("seem", input_size=360)
        if self.conf["dataset_type"]== "mp3d":
            self.categories = mp3dcat
        elif self.conf["dataset_type"] in ["Replica", "replica"]:
            self.categories = replica_cat
        else:
            self.categories = replica_cat #! hm3dsem cat으로!
            # raise ValueError(f"dataset_type {self.conf['dataset_type']} not supported")
        text_feats = self.model.encode_prompt(self.categories, task = "default")
        text_feats = text_feats.cpu().numpy()
        instance_feat = []

        windows = []
        doors = []
        others=[]

        id2idx={}
        import copy
        new_instance_dict = copy.deepcopy(self.dataloader.instance_dict)
        for idx, (id, val) in enumerate(self.dataloader.instance_dict.items()):
            id2idx[id]=idx
            instance_feat.append(val["embedding"])
        instance_feat = np.array(instance_feat)
        self.matching_cos = instance_feat @ text_feats.T
        for id in self.dataloader.instance_dict.keys():
            cos_list = self.matching_cos[list(self.dataloader.instance_dict.keys()).index(id)]
            cos_list2 = np.argsort(cos_list)[::-1]
            if np.max(cos_list) == 0: # ours는 instance마다 진행하니 아무것도 할당 안돼 0벡터 갖는 경우가 없어 이 처리 과정이 불필요하긴함
                swit = np.where(cos_list2 == 0)[0][0]
                cos_list2[swit] = cos_list2[0]
                cos_list2[0]=0
            if "window" in self.categories[cos_list2[0]]:
                windows.append(id)
            if "door" in self.categories[cos_list2[0]]:
                doors.append(id)
            new_instance_dict[id]["categories"]=cos_list2
            new_instance_dict[id]['category_idx']=cos_list2[0]
            new_instance_dict[id]["category"]=self.categories[cos_list2[0]]
            for cat in self.others: 
                if cat in self.categories[cos_list2[0]]:
                    others.append(id)
            if id == 2:
                continue
            else:
                # if new_instance_dict[id]['category'] in self.obstacles:
                for cat in new_instance_dict[id]["categories"][:3]:
                    if self.categories[cat] in self.obstacles:
                        # self.new_mask[new_instance_dict[id]["mask"]==1] = 0 #!
                        for coord in np.argwhere(new_instance_dict[id]["mask"]==1):
                            i,j = coord
                            if self.dataloader.grid_map[i,j][id][1] < self.max_obs_height: 
                                self.new_mask[i,j] = 0
        for i in range(self.dataloader.grid_map.shape[0]):
            for j in range(self.dataloader.grid_map.shape[1]):
                if len(self.dataloader.grid_map[i,j].keys()) == 0 : continue
                for key in self.dataloader.grid_map[i,j].keys():
                    if key in windows:
                        self.window_mask[i,j] = 1
                    if key in doors:
                        self.door_mask[i,j] = 1
                    if key in others:
                        self.others_mask[i,j] = 1
        self.dataloader.save_map(is_walls= False, categorized_instance_dict=new_instance_dict)
        self.dataloader.save_map(is_walls= False, semantic_obstacles=self.new_mask)
        obstacles_pil = Image.fromarray(self.new_mask)#[self.xmin:self.xmax+1, self.ymin:self.ymax+1])

        plt.figure(figsize=(8,6), dpi=120)
        plt.imshow(obstacles_pil, cmap='gray')
        plt.show()
        self.dataloader.save_map(is_walls=True, wall_mask=self.wall_mask)
        self.dataloader.save_map(is_walls=True,window_mask=self.window_mask)
        self.dataloader.save_map(is_walls=True,door_mask=self.door_mask)
        self.dataloader.save_map(is_walls=True,others_mask=self.others_mask)

