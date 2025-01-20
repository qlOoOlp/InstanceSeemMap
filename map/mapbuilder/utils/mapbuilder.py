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
from map.mapbuilder.map.seemmap_bbox_roomseg import SeemMap_roomseg
from map.mapbuilder.map.seemmap_dbscan import SeemMap_dbscan
from map.mapbuilder.map.seemmap_floodfill import SeemMap_floodfill
from map.mapbuilder.map.obstaclemap import ObstacleMap
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
            elif self.conf["seem_type"]=="dbscan" : self.map = SeemMap_dbscan(self.conf)
            elif self.conf["seem_type"]=="floodfill" : self.map = SeemMap_floodfill(self.conf)
            elif self.conf["seem_type"]=="obstacle": self.map = ObstacleMap(self.conf)
            elif self.conf["seem_type"]=="room_seg": self.map = SeemMap_roomseg(self.conf)
    def buildmap(self):
        print("#"*100)
        self.map.start_map()
        self.map.processing()
        self.map.save_map()
        print("="*(len("Map building done")+10),"Map building done","#"*100,sep='\n')





class IndexMapBuilder():
    def __init__(self, conf:DictConfig):
        self.conf = conf
        self.dataloader = DataLoader(self.conf)
        x_indices, y_indices = np.where(self.dataloader.obstacle_map == 0)
        self.xmin, self.xmax, self.ymin, self.ymax = np.min(x_indices), np.max(x_indices), np.min(y_indices), np.max(y_indices)
    def processing(self):
        self.wall_mask = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=bool)
        self.window_mask = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=bool)
        self.door_mask = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=bool)
        for i in range(self.dataloader.grid_map.shape[0]):
            for j in range(self.dataloader.grid_map.shape[1]): 
                if 1 in self.dataloader.grid_map[i,j]:
                    self.wall_mask[i,j] = 1
        self.model = build_vl_model("seem", input_size=360)
        if self.conf["dataset_type"]== "mp3d":
            self.categories = mp3dcat
        elif self.conf["dataset_type"]== "replica":
            self.categories = replica_cat
        else:
            raise ValueError(f"dataset_type {self.conf['dataset_type']} not supported")
        text_feats = self.model.encode_prompt(self.categories, task = "default")
        text_feats = text_feats.cpu().numpy()
        instance_feat = []

        windows = []
        doors = []

        for id, val in self.embeddings.items():
            instance_feat.append(val["embedding"])
        instance_feat = np.array(instance_feat)
        self.matching_cos = instance_feat @ text_feats.T
        for id in self.embeddings.keys():
            cos_list = self.matching_cos[list(self.embeddings.keys()).index(id)]
            cos_list2 = np.argsort(cos_list)[::-1]
            if np.max(cos_list) == 0: # ours는 instance마다 진행하니 아무것도 할당 안돼 0벡터 갖는 경우가 없어 이 처리 과정이 불필요하긴함
                swit = np.where(cos_list2 == 0)[0][0]
                cos_list2[swit] = cos_list2[0]
                cos_list2[0]=0
            if "window" in cos_list2[0]:
                windows.append(id)
            if "door" in cos_list2[0]:
                doors.append(id)
        for i in range(self.dataloader.grid_map.shape[0]):
            for j in range(self.dataloader.grid_map.shape[1]):
                if len(self.dataloader.grid_map[i,j].keys()) == 0 : continue
                for key in self.dataloader.grid_map[i,j].keys():
                    if key in windows:
                        self.window_mask[i,j] = 1
                    if key in doors:
                        self.door_mask[i,j] = 1

        obstacles_pil = Image.fromarray(self.wall_mask[self.xmin:self.xmax+1, self.ymin:self.ymax+1])
        plt.figure(figsize=(8,6), dpi=120)
        plt.imshow(obstacles_pil, cmap='gray')
        plt.show()
        self.dataloader.save_map(wall_mask=self.wall_mask)
        self.dataloader.save_map(window_mask=self.window_mask)
        self.dataloader.save_map(door_mask=self.door_mask)





# class IndexMapBuilder():
#     def __init__(self, conf:DictConfig):
#         self.conf = conf
#         self.dataloader = DataLoader(self.conf)
#         x_indices, y_indices = np.where(self.dataloader.obstacle_map == 0)
#         self.xmin, self.xmax, self.ymin, self.ymax = np.min(x_indices), np.max(x_indices), np.min(y_indices), np.max(y_indices)

#         self.query = list(self.conf["query"])
#         self.build_category_instance_map()
#         if self.conf["save_category_map"]: self.dataloader.save_map(category_map=self.category_map)
#         if self.conf["save_instance_map"]: 
#             if self.conf["seem_instance_method"]=="dbscan": self.dataloader.save_map(instance_dict=self.instance_dict)
#             self.dataloader.save_map(instance_map=self.instance_map)
#         if self.conf["visualize"]: self.visualizer()
        
#     def build_category_instance_map(self):
#         if self.dataloader.hparams["vlm"] == "lseg":
#             clip_model, preprocess = clip.load(self.dataloader.hparams["clip_version"])  # clip.available_models()
#             clip_model.to(self.conf["device"]).eval()
#             clip_feat_dim = CLIP_FEAT_DIM_DICT[self.dataloader.hparams["clip_version"]]
#             text_feats = get_text_feats(self.query, clip_model, clip_feat_dim)
#             map_feats = self.dataloader.grid_map.reshape((-1, self.dataloader.grid_map.shape[-1]))
#             scores_list = map_feats @ text_feats.T
#             predicts = np.argmax(scores_list, axis=1)
#             predicts = predicts.reshape((self.dataloader.grid_map.shape[0], self.dataloader.grid_map.shape[1]))
#             predicts = predicts[self.xmin:self.xmax+1, self.ymin:self.ymax+1]
#             self.category_map = predicts.copy()
#             if self.conf["seem_instance_method"]=="floodfill":
#                 self.similarity_instance_map = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=int)
#                 self.visited = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=bool)
#                 current_instance=3
#                 wall_semantic = get_text_feats(["wall"], clip_model, clip_feat_dim)
#                 wall_semantic = wall_semantic
#                 wall_semantic_normalized = wall_semantic/np.linalg.norm(wall_semantic)
#                 self.wall_semantic_normalized = wall_semantic_normalized
#                 floor_semantic = get_text_feats(["floor"], clip_model, clip_feat_dim)
#                 floor_semantic = floor_semantic
#                 floor_semantic_normalized = floor_semantic/np.linalg.norm(floor_semantic)
#                 self.floor_semantic_normalized = floor_semantic_normalized
#                 for i in range(self.dataloader.grid_map.shape[0]):
#                     for j in range(self.dataloader.grid_map.shape[1]):
#                         if not self.visited[i,j]:
#                             if np.sum(self.dataloader.grid_map[i,j])==0:continue
#                             self.flood_fill(i,j,current_instance)
#                             current_instance+=1
#                 for item in np.unique(self.similarity_instance_map):
#                     if np.sum(self.similarity_instance_map==item)<100:
#                         self.similarity_instance_map[self.similarity_instance_map==item]=0
#                 self.similarity_instance_map = self.similarity_instance_map[self.xmin:self.xmax+1, self.ymin:self.ymax+1]
#                 self.instance_map = self.similarity_instance_map.copy()
#             elif self.conf["seem_instance_method"]=="dbscan":
#                 nnew_dict = {}
#                 instance_map, instance_dict, num_instances = self.instance_segmentation_with_dbscan(nnew_dict, eps=2, min_samples=5)
#                 class_map = self.create_class_map(instance_map, instance_dict)
#                 self.instance_map = instance_map.copy()
#                 self.instance_dict = instance_dict.copy()

#         elif self.dataloader.hparams["vlm"] == "seem" and self.dataloader.hparams["seem_type"] == "base":
#             model = build_vl_model("seem", input_size=360)
#             text_feats = model.encode_prompt(self.query, task = "default")
#             text_feats = text_feats.cpu().numpy()
#             map_feats = self.dataloader.grid_map.reshape((-1, self.dataloader.grid_map.shape[-1]))
#             scores_list = map_feats @ text_feats.T
#             predicts = np.argmax(scores_list, axis=1)
#             predicts = predicts.reshape((self.dataloader.grid_map.shape[0], self.dataloader.grid_map.shape[1]))
#             predicts = predicts[self.xmin:self.xmax+1, self.ymin:self.ymax+1]
#             self.category_map = predicts.copy()
#             if self.conf["seem_instance_method"]=="floodfill":
#                 self.similarity_instance_map = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=int)
#                 self.visited = np.zeros((self.dataloader.grid_map.shape[0],self.dataloader.grid_map.shape[1]), dtype=bool)
#                 current_instance=3
#                 wall_semantic = model.encode_prompt("wall", task="default")
#                 wall_semantic = wall_semantic.cpu().numpy()
#                 wall_semantic_normalized = wall_semantic/np.linalg.norm(wall_semantic)
#                 self.wall_semantic_normalized = wall_semantic_normalized
#                 floor_semantic = model.encode_prompt("floor", task="default")
#                 floor_semantic = floor_semantic.cpu().numpy()
#                 floor_semantic_normalized = floor_semantic/np.linalg.norm(floor_semantic)
#                 self.floor_semantic_normalized = floor_semantic_normalized
#                 for i in range(self.dataloader.grid_map.shape[0]):
#                     for j in range(self.dataloader.grid_map.shape[1]):
#                         if not self.visited[i,j]:
#                             if np.sum(self.dataloader.grid_map[i,j])==0:continue
#                             self.flood_fill(i,j,current_instance)
#                             current_instance+=1

#                 for item in np.unique(self.similarity_instance_map):
#                     if np.sum(self.similarity_instance_map==item)<100:
#                         self.similarity_instance_map[self.similarity_instance_map==item]=0

#                 self.similarity_instance_map = self.similarity_instance_map[self.xmin:self.xmax+1, self.ymin:self.ymax+1]
#                 self.instance_map = self.similarity_instance_map.copy()
#             elif self.conf["seem_instance_method"]=="dbscan":
#                 nnew_dict = {}
#                 instance_map, instance_dict, num_instances = self.instance_segmentation_with_dbscan(nnew_dict, eps=2, min_samples=5)
#                 class_map = self.create_class_map(instance_map, instance_dict)
#                 self.instance_map = instance_map.copy()
#                 self.instance_dict = instance_dict.copy()
                
#         else:
#             model = build_vl_model("seem", input_size=360)
#             text_feats = model.encode_prompt(self.query, task = "default")
#             text_feats = text_feats.cpu().numpy()
#             grid_map_cropped = self.dataloader.grid_map[self.xmin:self.xmax+1, self.ymin:self.ymax+1]
#             # background_grid_cropped = self.dataloader.background_grid[self.xmin:self.xmax+1, self.ymin:self.ymax+1]
#             if self.conf["indexing_method"] in ["height", "count"]:
#                 id_list = []
#                 instance_feat = []
#                 ids = []
#                 for i in range(self.dataloader.grid_map.shape[0]):
#                     for j in range(self.dataloader.grid_map.shape[1]):
#                         for arg in self.dataloader.grid_map[i,j].keys():
#                             if arg not in ids:
#                                 ids.append(arg)
#                 for id, val in self.dataloader.instance_dict.items():
#                     id_list.append(id)
#                     instance_feat.append(val["embedding"])
#                 instance_feat = np.array(instance_feat)
#                 scores_list = instance_feat @ text_feats.T
#                 predicts = np.argmax(scores_list, axis=1)
#                 category_map = np.zeros((grid_map_cropped.shape[0], grid_map_cropped.shape[1]),dtype=int)
#                 instance_map = np.zeros((grid_map_cropped.shape[0], grid_map_cropped.shape[1]),dtype=int)
#                 for i in range(grid_map_cropped.shape[0]):
#                     for j in range(grid_map_cropped.shape[1]):
#                         if len(grid_map_cropped[i,j].keys())==0: continue
#                         if len(grid_map_cropped[i,j].keys())==1:
#                             for key in grid_map_cropped[i,j].keys():
#                                 category_map[i,j] = predicts[list(self.dataloader.instance_dict.keys()).index(key)]
#                                 instance_map[i,j] = key+1
#                         else:
#                             max_conf=0
#                             max_height=50000
#                             max_observed = 0
#                             for key,val in grid_map_cropped[i,j].items():
#                                 candidate = predicts[list(self.dataloader.instance_dict.keys()).index(key)]
#                                 candidate_conf = val[0]
#                                 candidate_height = val[1]
#                                 candidate_observed = val[2]
#                                 if self.conf["indexing_method"]=="height":
#                                     if candidate_height < max_height:
#                                         max_height = candidate_height
#                                         candidate_val = candidate
#                                         candidate_key = key
#                                 elif self.conf["indexing_method"]=="count":
#                                     if candidate_observed > max_observed:
#                                         max_observed = candidate_observed
#                                         candidate_val = candidate
#                                         candidate_key = key
#                             category_map[i,j] = candidate_val
#                             instance_map[i,j] = candidate_key+1
#                 self.category_map = category_map.copy()
#                 self.instance_map = instance_map.copy()

#             elif self.conf["indexing_method"]=="mode":
#                 id_list = []
#                 instance_feat = []
#                 ids = []
#                 for i in range(self.dataloader.grid_map.shape[0]):
#                     for j in range(self.dataloader.grid_map.shape[1]):
#                         for arg in self.dataloader.grid_map[i,j].keys():
#                             if arg not in ids:
#                                 ids.append(arg)
#                 for id, val in self.dataloader.instance_dict.items():
#                     id_list.append(id)
#                     instance_feat.append(val["embedding"])
#                 instance_feat = np.array(instance_feat)
#                 scores_list = instance_feat @ text_feats.T
#                 predicts = np.argmax(scores_list, axis=1)
#                 center_weight = 3
#                 category_map = np.zeros_like(grid_map_cropped, dtype=np.uint16)
#                 instance_map = np.zeros_like(grid_map_cropped, dtype=np.uint16)
#                 grid_upper = np.empty((grid_map_cropped.shape[0]+1, grid_map_cropped.shape[1]+1), dtype=object)
#                 for i in range(grid_map_cropped.shape[0]+1):
#                     for j in range(grid_map_cropped.shape[1]+1):
#                         grid_upper[i,j] = {}
#                 grid_upper[1:,1:] = grid_map_cropped
#                 for i in range(1, grid_map_cropped.shape[0]+1):
#                     for j in range(1, grid_map_cropped.shape[1]+1):
#                         candidate = grid_upper[i-1:i+2,j-1:j+2]
#                         item_dict={}
#                         for candidate_i in range(candidate.shape[0]):
#                             for candidate_j in range(candidate.shape[1]):
#                                 if candidate_i == i and candidate_j == j:
#                                     for key, val in candidate[candidate_i,candidate_j].items():
#                                         if key in item_dict.keys():
#                                             item_dict[key] += center_weight
#                                         else: item_dict[key] = center_weight
#                                 else:
#                                     for key, val in candidate[candidate_i,candidate_j].items():
#                                         if key in item_dict.keys():
#                                             item_dict[key] += 1
#                                         else: item_dict[key] = 1
#                         if len(item_dict) == 0:
#                             category_map[i-1,j-1] = 0
#                             continue
#                         max_key = max(item_dict, key=item_dict.get)
#                         instance_map[i-1,j-1] = max_key+1
#                         category_map[i-1,j-1] = predicts[list(self.dataloader.instance_dict.keys()).index(max_key)]
#                 self.category_map = category_map.copy()
#                 self.instance_map = instance_map.copy()

#             else:
#                 raise ValueError(f"Invalid indexing method: {self.conf['indexing_method']}")
            
    
#     def build_instance_map(self):
#         raise NotImplementedError

#     def visualizer(self):
#         raise NotImplementedError
    

#     def flood_fill(self,x,y,current_instance):
#         stack = [(x, y)]
#         self.similarity_instance_map[x, y] = current_instance
#         self.visited[x, y] = True
#         while stack:
#             cx, cy = stack.pop()
#             # 현재 픽셀의 임베딩 벡터
#             current_embedding = self.dataloader.grid_map[cx, cy]
#             # for dx in range(-2, 3):  # -2, -1, 0, 1, 2
#                 # for dy in range(-2, 3):
#             for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
#                     if dx == 0 and dy == 0:  # 중심 픽셀 제외
#                         continue
#                     nx, ny = cx + dx, cy + dy

#                     # 경계 체크
#                     if nx < 0 or ny < 0 or nx >= self.dataloader.grid_map.shape[0] or ny >= self.dataloader.grid_map.shape[1]:
#                         continue
#                     if self.visited[nx, ny]:
#                         continue

#                     # 이웃 픽셀과의 유사도 계산
#                     neighbor_embedding = self.dataloader.grid_map[nx, ny]
#                     if np.sum(neighbor_embedding)==0:continue
#                     current_embedding_normalized = current_embedding/np.linalg.norm(current_embedding)
#                     neighbor_embedding_normalized = neighbor_embedding/np.linalg.norm(neighbor_embedding)
#                     # current_embedding_norm = normalize(current_embedding.reshape(1, -1))
#                     # neighbor_embedding_norm = normalize(neighbor_embedding.reshape(1, -1))
#                     wall_similarity = np.dot(neighbor_embedding_normalized, self.wall_semantic_normalized.T)
#                     if wall_similarity > 0.3:
#                         self.visited[nx, ny] = True
#                         self.similarity_instance_map[nx, ny] = 1
#                         continue
#                     floor_similarity = np.dot(neighbor_embedding_normalized, self.floor_semantic_normalized.T)
#                     if floor_similarity > 0.3:
#                         self.visited[nx, ny] = True
#                         self.similarity_instance_map[nx, ny] = 2
#                         continue
#                     similarity = np.dot(current_embedding_normalized, neighbor_embedding_normalized)

#                     # 유사도가 임계값 이상일 경우 같은 인스턴스로 묶기
#                     if similarity >= self.conf["threshold_semSim"]:
#                         # print(current_instance)
#                         self.similarity_instance_map[nx, ny] = current_instance
#                         self.visited[nx, ny] = True
#                         stack.append((nx, ny))

#     def instance_segmentation_with_dbscan(self, nnew_dict, eps=2, min_samples=5):
#         num_class = len(np.unique(self.category_map))
#         new_grid = np.zeros_like(self.category_map)
#         inst= 1
#         instance_dict ={}
#         for class_id in np.unique(self.category_map):
#             if class_id ==0: continue
#             if class_id ==1 or class_id ==2:
#                 new_grid[self.category_map==class_id] = inst
#                 instance_dict[inst]={"class_id":class_id, "class":self.query[class_id]}
#                 inst+=1
#                 continue
#             id_mask = np.where(self.category_map == class_id, 1, 0)
#             if np.sum(id_mask) < 50 : continue
#             coords = np.column_stack(np.where(id_mask==1))
#             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
#             db = DBSCAN(eps=2, min_samples=5).fit(coords)
#             labels = db.labels_
#             labeled_mask = np.zeros_like(id_mask, dtype=int)
#             for label, (x,y) in zip(labels, coords):
#                 labeled_mask[x,y] = label + 1
#             num_features = len(np.unique(labeled_mask))
#             for key in range(1,num_features+1):
#                 instance_mask = (labeled_mask == key).astype(np.uint8)
#                 if np.sum(instance_mask) < 50 : continue
#                 new_grid[labeled_mask==key] = inst
#                 if class_id not in nnew_dict.keys():
#                     nnew_dict[class_id] = [inst]
#                 else:
#                     nnew_dict[class_id].append(inst)
#                 instance_dict[inst]={"class_id":class_id, "class":self.query[class_id]}
#                 inst+=1
#         return new_grid, instance_dict, inst
#     def create_class_map(self, instance_map, instance_dict):
#         class_map = np.zeros_like(instance_map)
#         for inst_id, info in instance_dict.items():
#             class_id = info["class_id"]
#             class_map[instance_map == inst_id] = class_id
#         return class_map
