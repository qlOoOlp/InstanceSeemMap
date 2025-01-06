import numpy as np
import torch
import os
import pickle
import clip

from map.utils.mapping_utils import load_map
from map.utils.clip_utils import get_text_feats
from map.seem.base_model import build_vl_model





def gt_idx_change(gt, ori_categories, new_categories):
    new_gt = np.zeros(gt.shape)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            try:
                cat = ori_categories[gt[i,j]]
            except:
                print(gt[i,j])
                raise Exception("Error")
            new_gt[i,j] = new_categories.index(cat)
    return new_gt



class idxMap():
    def __init__(self, type, model, categories, grid, version, grid_path):
        self.type = type
        self.categories = categories
        self.num_classes = len(categories)
        self.sorted_idx_dict = {}
        self.model = model
        if self.type == "lseg":
            # self.grid_path = os.path.join(path,"map",f"{scene_id}_{version}",f"grid_{version}.npy")
            # self.grid = load_map(self.grid_path)
            # self.grid = self.grid[xmin:xmax+1, ymin:ymax+1]
            self.grid = grid
            self.idx_map = np.zeros(self.grid.shape[:2])
            self.lsegmap()

        elif self.type == "seem":
            # self.grid_path = os.path.join(path,"map",f"{scene_id}_{version}",f"grid_{version}.npy")
            # self.grid = load_map(self.grid_path)
            # self.grid = self.grid[xmin:xmax+1, ymin:ymax+1]
            self.grid = grid
            self.idx_map = np.zeros(self.grid.shape[:2])
            self.seemmap()
        else:
            # self.grid_path = os.path.join(path,"map",f"{scene_id}_{version}",f"grid_{version}.npy")
            # self.grid1 = load_map(self.grid_path)
            # self.grid1 = self.grid1[xmin:xmax+1, ymin:ymax+1]
            self.grid1 = grid
            self.embeddings_path = os.path.join('/'.join(grid_path.split('/')[:-1]), f"instance_dict_{version}.pkl")
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            self.idx_map = np.zeros(self.grid1.shape)
            self.ourmap()
    
    def lsegmap(self):
        text_feats = get_text_feats(self.categories, self.model, self.clip_feat_dim)
        map_feats = self.grid.reshape(-1, self.grid.shape[-1])
        self.matching_cos = map_feats @ text_feats.T
        height, width = self.grid.shape[:2]
        for i in range(height):
            for j in range(width):
                val = self.matching_cos[i*width+j]
                val2 = np.argsort(val)[::-1]
                if np.max(val) == 0: #해당 grid 지점에 아무것도 할당 안돼 0벡터 갖는 경우(맵 외부)에 대한 처리
                    swit = np.where(val2 == 0)[0][0]
                    val2[swit] = val2[0]
                    val2[0]=0
                best_val = val2[0]
                self.sorted_idx_dict[(i,j)] = val2.tolist()
                self.idx_map[i,j] = best_val
        self.idx_map = self.idx_map.astype(np.int32)
    
    def seemmap(self):
        # print(self.categories)
        text_feats = self.model.encode_prompt(self.categories, task = "default")
        text_feats = text_feats.cpu().numpy()
        map_feats = self.grid.reshape(-1, self.grid.shape[-1])
        self.matching_cos = map_feats @ text_feats.T
        height, width = self.grid.shape[:2]
        for i in range(height):
            for j in range(width):
                val = self.matching_cos[i*width+j]
                val2 = np.argsort(val)[::-1]
                if np.max(val) == 0: #해당 grid 지점에 아무것도 할당 안돼 0벡터 갖는 경우(맵 외부)에 대한 처리
                    swit = np.where(val2 == 0)[0][0]
                    val2[swit] = val2[0]
                    val2[0]=0
                best_val = val2[0]
                # print(val)
                self.sorted_idx_dict[(i,j)] = val2.tolist()
                self.idx_map[i,j] = best_val
        self.idx_map = self.idx_map.astype(np.int32)

    def ourmap(self):
        text_feats = self.model.encode_prompt(self.categories, task = "default")
        text_feats = text_feats.cpu().numpy()
        instance_feat = []
        self.embeddings[1]["avg_height"] = 2
        self.embeddings[2]["avg_height"] = 1.5
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
            self.sorted_idx_dict[id] = cos_list2.tolist()

        self.topdown_instance_map = np.zeros(self.grid1.shape[:2])
        for i in range(self.grid1.shape[0]):
            for j in range(self.grid1.shape[1]):
                if len(self.grid1[i,j].keys()) == 0 : continue
                if len(self.grid1[i,j].keys()) == 1:
                    for key in self.grid1[i,j].keys():
                        self.topdown_instance_map[i,j] = key
                        self.idx_map[i,j] = self.sorted_idx_dict[key][0]
                else:
                    max_height = 50000
                    for key, val in self.grid1[i,j].items():
                        # if key == 2: continue
                        candidate_height = self.embeddings[key]["avg_height"] #^ using instance average height value
                        # candidate_height = self.grid1[i,j][key][1] #^ using pixel level height value
                        if max_height > candidate_height:
                            max_height = candidate_height
                            candidate_val = key
                    self.topdown_instance_map[i,j] = candidate_val
                    self.idx_map[i,j] = self.sorted_idx_dict[candidate_val][0]
        self.topdown_instance_map = self.topdown_instance_map.astype(np.int32)
        self.idx_map = self.idx_map.astype(np.int32)

    # def setup_vlm(self,type):
    #     if type == "lseg":
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         clip_version = "ViT-B/32"
    #         self.clip_feat_dim = CLIP_FEAT_DIM_DICT[clip_version]
    #         self.model, preprocess = clip.load(clip_version)  # clip.available_models()
    #         self.model.to(device).eval()
    #     elif type == "seem":
    #         self.model = build_vl_model("seem", input_size = 360)