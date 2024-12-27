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
from mapbuilder.map.map import Map
from mapbuilder.utils.datamanager import DataManager, DataManager4Real

class SeemMap_floodfill(Map):
    def __init__(self, config:DictConfig):
        super().__init__(config)
        self.feat_dim = self.config["feat_dim"]

        self.camera_height = self.config["camera_height"]
        self.gs = self.config["gs"]
        self.cs = self.config["cs"]
        self.depth_sample_rate = self.config["depth_sample_rate"]
        self.downsampling_ratio = self.config["downsampling_ratio"]
        self.threshold_confidence = self.config["threshold_confidence"]
        self.max_depth = self.config["max_depth"]
        self.min_depth = self.config["min_depth"]

        self._setup_SEEM()

    def _setup_SEEM(self):
        rgb_shape, _ = self.datamanager.get_data_shape()
        print(f"SEEM input size : {rgb_shape[0]*self.downsampling_ratio}")
        self.model = build_vl_model("seem", input_size = int(rgb_shape[0]*self.downsampling_ratio))

    # @abstractmethod
    def processing(self):
        tf_list = []
        print("Processing data...")
        pbar = tqdm(range(self.datamanager.numData))
        while self.datamanager.count < self.datamanager.numData-1:
            rgb, depth, (pos,rot) = self.datamanager.data_getter()
            rot = rot @ self.datamanager.rectification_matrix
            # print(self.datamanager.rectification_matrix)
            pos[1] += self.camera_height
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = pos.reshape(-1)
            # print(pose)

            tf_list.append(pose)
            if len(tf_list) == 1:
                init_tf_inv = np.linalg.inv(tf_list[0])
            # print(init_tf_inv)
            tf = init_tf_inv @ pose
            # print(tf)

            map_idx, map_conf, embeddings, category_dict = get_SEEM_feat(self.model, rgb, self.threshold_confidence)
            if map_idx is None:
                pbar.update(1)
                continue

            map_emb = self.ra2pa(map_idx, embeddings)

            if self.data_type == "rtabmap":
                pc, mask = depth2pc4Real(depth, self.datamanager.projection_matrix, rgb.shape[:2], min_depth=0.5, max_depth=self.max_depth)
                # pc[1,:] += self.camera_height
                shuffle_mask = np.arange(pc.shape[1])
                np.random.shuffle(shuffle_mask)
                shuffle_mask = shuffle_mask[::self.depth_sample_rate]
                mask = mask[shuffle_mask]
                pc = pc[:, shuffle_mask]
                pc = pc[:, mask]
                pc_global = transform_pc(pc, tf)
                rgb_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], rgb.shape[:2])
                feat_cam_mat = get_sim_cam_mat4Real(self.datamanager.projection_matrix, rgb.shape[:2], map_idx.shape)
            else:
                pc, mask = depth2pc(depth, max_depth=self.max_depth)
                # pc[1,:] += self.camera_height
                shuffle_mask = np.arange(pc.shape[1])
                np.random.shuffle(shuffle_mask)
                shuffle_mask = shuffle_mask[::self.depth_sample_rate]
                mask = mask[shuffle_mask]
                pc = pc[:, shuffle_mask]
                pc = pc[:, mask]
                pc_global = transform_pc(pc, tf)
                rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
                feat_cam_mat = get_sim_cam_mat(map_idx.shape[0], map_idx.shape[1])
            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                x,y = pos2grid_id(self.gs, self.cs, p[0], p[2])
                if x>= self.obstacles.shape[0] or y>= self.obstacles.shape[1] or x<0 or y<0 or p_local[1] < -0.5: continue
                rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
                rgb_v = rgb[rgb_py, rgb_px, :]
                # print(p_local[1])

                if p_local[1] < self.color_top_down_height[y,x]:
                    self.color_top_down[y,x] = rgb_v
                    self.color_top_down_height[y,x] = p_local[1]
                px, py, pz = project_point(feat_cam_mat, p_local)
                if not (px < 0 or py < 0 or px >= map_emb.shape[1] or py >= map_emb.shape[0]):
                    feat = map_emb[py, px, :]
                    self.grid[y,x]=(self.grid[y,x]*self.weight[y,x]+feat)/(self.weight[y,x]+1)
                    self.weight[y,x]+=1
                if p_local[1] > self.camera_height:
                    continue
                self.obstacles[y,x]=1
            pbar.update(1)
        self.postprocessing()

    # Convert region aligned feature to pixel aligned feature
    def ra2pa(self, map_idx, embeddings):
        map_emb = np.zeros((map_idx.shape[0], map_idx.shape[1], self.feat_dim))
        for row in range(map_idx.shape[0]):
            for col in range(map_idx.shape[1]):
                idx = map_idx[row, col]
                if idx != 0:
                    map_emb[row, col] = embeddings[idx]
        return map_emb

    
    # @abstractmethod
    def postprocessing(self):
        # self.instance_dict = {}
        current_instance = 3
        self.height, self.width = self.grid.shape[0], self.grid.shape[1]
        self.similarity_threshold = 0.99

        pbar = tqdm(total=self.height * self.width,leave=True)
        for i in range(self.height):
            for j in range(self.width):
                if not self.visited[i,j]:
                    if np.sum(self.grid[i,j])==0:
                        pbar.update(1)
                        continue
                    # print(self.grid[i,j].shape)
                    self.instance_dict[current_instance] = {"embedding":self.grid[i,j], "count":1}
                    self.flood_fill(i, j, current_instance)
                    current_instance += 1
                pbar.update(1)
        pbar.close()
        for item in np.unique(self.similarity_instance_map):
            if np.sum(self.similarity_instance_map==item)<50:
                self.similarity_instance_map[self.similarity_instance_map==item]=0
                self.instance_dict.pop(item)
        # self.postprocessing2()

    def postprocessing2(self):
        pbar = tqdm(total=len(self.instance_dict.keys()),leave=True)
        new_instance_dict = {}
        new_grid = np.zeros((self.gs, self.gs), dtype = int)
        for key, val in self.instance_dict.items():
            matching_val = (0,0.85)
            candidate_emb = val["embedding"]
            candidate_emb_normalized = candidate_emb/np.linalg.norm(candidate_emb)
            candidate_count = val["count"]
            coordinates = np.argwhere(self.similarity_instance_map==key)
            for coord in coordinates:
                cx,cy = coord
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nx, ny = cx +dx, cy+dy
                    if nx<0 or ny<0 or nx> self.height or ny> self.width:
                        continue
                    comp_idx = self.similarity_instance_map[nx,ny]

                    if comp_idx not in new_instance_dict: continue
                    comp_emb = new_instance_dict[comp_idx]["embedding"]
                    comp_count = new_instance_dict[comp_idx]["count"]
                    comp_emb_normalized = comp_emb/np.linalg.norm(comp_emb)
                    similarity = np.dot(candidate_emb_normalized, comp_emb_normalized)
                    if similarity > matching_val[1]:
                        matching_val = (comp_idx, similarity)
            if matching_val[0] !=0:
                matching_id = matching_val[0]
                # print(matching_id, "1")
                matching_emb = new_instance_dict[matching_id]["embedding"]
                matching_count = new_instance_dict[matching_id]["count"]
                new_instance_dict[matching_id] = {"embedding":(matching_emb*matching_count+candidate_emb*candidate_count)/(matching_count+candidate_count), "count":matching_count+candidate_count}
                new_grid[self.similarity_instance_map==key] = matching_id
                for id in np.unique(new_grid):
                    if id < 0:
                        print(1)
                        print(id)
                        raise Exception("sdfsf")
            else:
                # print(key, "2")
                new_instance_dict[key] = val
                new_grid[self.similarity_instance_map==key] = key
                for id in np.unique(new_grid):
                    if id < 0:
                        print(2)
                        print(id)
                        raise Exception("sdfsf")
            pbar.update(1)
        pbar.close()
        # print("new_grid",np.unique(new_grid))
        self.instance_dict = new_instance_dict
        self.similarity_instance_map = new_grid
        # print("simil",np.unique(self.similarity_instance_map))
        # print("dict",self.instance_dict.keys())





    # @abstractmethod
    def preprocessing(self):
        raise NotImplementedError
    
    def _load_map(self):
        raise NotImplementedError
    



    def flood_fill(self, x, y, current_instance):
        # 스택을 사용해 Flood Fill (재귀 깊이 문제를 피하기 위해 스택 방식 사용)
        stack = [(x, y)]
        self.similarity_instance_map[x, y] = current_instance
        self.visited[x, y] = True

        while stack:
            cx, cy = stack.pop()
            # 현재 픽셀의 임베딩 벡터
            current_embedding = self.grid[cx, cy]




            # # 8방향 이웃 탐색 (필터 순회)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = cx + dx, cy + dy

                # 경계 체크
                if nx < 0 or ny < 0 or nx >= self.height or ny >= self.width:
                    continue
                if self.visited[nx, ny]:
                    continue
                # 이웃 픽셀과의 유사도 계산
                neighbor_embedding = self.grid[nx, ny]
                if np.sum(neighbor_embedding)==0:continue
                # similarity = np.dot(current_embedding, neighbor_embedding)
                current_embedding_normalized = current_embedding/np.linalg.norm(current_embedding)
                neighbor_embedding_normalized = neighbor_embedding/np.linalg.norm(neighbor_embedding)
                # current_embedding_norm = normalize(current_embedding.reshape(1, -1))
                # neighbor_embedding_norm = normalize(neighbor_embedding.reshape(1, -1))
                wall_similarity = np.dot(neighbor_embedding_normalized, self.wall_semantic_normalized.T)
                if wall_similarity > 0.3:
                    self.visited[nx, ny] = True
                    self.similarity_instance_map[nx, ny] = 1
                    self.instance_dict[1]["count"]+=1
                    continue
                floor_similarity = np.dot(neighbor_embedding_normalized, self.floor_semantic_normalized.T)
                if floor_similarity > 0.3:
                    self.visited[nx, ny] = True
                    self.similarity_instance_map[nx, ny] = 2
                    self.instance_dict[2]["count"]+=1
                    continue
                similarity = np.dot(current_embedding_normalized, neighbor_embedding_normalized)
                # print(similarity)
                # similarity = cosine_similarity(current_embedding_normalized.reshape(1, -1), neighbor_embedding_normalized.reshape(1, -1))[0, 0]
                # print(similarity)
                # 유사도가 임계값 이상일 경우 같은 인스턴스로 묶기
                if similarity >= self.similarity_threshold:
                    # print(similarity)
                    # print(current_instance)
                    # print("hi")
                    self.similarity_instance_map[nx, ny] = current_instance
                    self.visited[nx, ny] = True
                    instance_embedding = self.instance_dict[current_instance]["embedding"]
                    instance_count = self.instance_dict[current_instance]["count"]
                    self.instance_dict[current_instance] = {"embedding":(instance_embedding*instance_count+current_embedding)/(instance_count+1), "count":instance_count+1}
                    stack.append((nx, ny))
                # else:
                #     print(similarity)
                #     print("nooo")



            # for dx in range(-2, 3):  # -2, -1, 0, 1, 2
            #     for dy in range(-2, 3):
            #         if dx == 0 and dy == 0:  # 중심 픽셀 제외
            #             continue
            #         nx, ny = cx + dx, cy + dy

            #         # 경계 체크
            #         if nx < 0 or ny < 0 or nx >= self.height or ny >= self.width:
            #             continue
            #         if self.visited[nx, ny]:
            #             continue

            #         # 이웃 픽셀과의 유사도 계산
            #         neighbor_embedding = self.grid[nx, ny]
            #         if np.sum(neighbor_embedding)==0:continue
            #         current_embedding_normalized = current_embedding/np.linalg.norm(current_embedding)
            #         neighbor_embedding_normalized = neighbor_embedding/np.linalg.norm(neighbor_embedding)
            #         # current_embedding_norm = normalize(current_embedding.reshape(1, -1))
            #         # neighbor_embedding_norm = normalize(neighbor_embedding.reshape(1, -1))
            #         similarity = np.dot(current_embedding_normalized, neighbor_embedding_normalized)
            #         # rug_similarity = np.dot(neighbor_embedding_normalized, rug_semantic_normalized.T)
            #         # if rug_similarity > 0.3:
            #         #     visited[nx, ny] = True
            #         #     similarity_instance_map[nx, ny] = 3
            #         #     continue
            #         wall_similarity = np.dot(neighbor_embedding_normalized, self.wall_semantic_normalized.T)
            #         if wall_similarity > 0.3:
            #             self.visited[nx, ny] = True
            #             self.similarity_instance_map[nx, ny] = 1
            #             self.instance_dict[1]["count"]+=1
            #             continue
            #         floor_similarity = np.dot(neighbor_embedding_normalized, self.floor_semantic_normalized.T)
            #         if floor_similarity > 0.3:
            #             self.visited[nx, ny] = True
            #             self.similarity_instance_map[nx, ny] = 2
            #             self.instance_dict[2]["count"]+=1
            #             continue
            #         similarity = np.dot(current_embedding_normalized, neighbor_embedding_normalized)

            #         # 유사도가 임계값 이상일 경우 같은 인스턴스로 묶기
            #         if similarity >= self.similarity_threshold:
            #             # print(current_instance)
            #             self.similarity_instance_map[nx, ny] = current_instance
            #             self.visited[nx, ny] = True
            #             instance_embedding = self.instance_dict[current_instance]["embedding"]
            #             instance_count = self.instance_dict[current_instance]["count"]
            #             self.instance_dict[current_instance] = {"embedding":(instance_embedding*instance_count+current_embedding)/(instance_count+1), "count":instance_count+1}
            #             stack.append((nx, ny))



    # @abstractmethod
    def _init_map(self):
        # self.color_top_down_height = np.zeros((self.gs, self.gs), dtype=np.float32)
        self.instance_dict = {}
        background_emb = self.model.encode_prompt(["wall","floor"], task = "default")
        background_emb = background_emb.cpu().numpy()
        self.instance_dict[1] = {"embedding":background_emb[0,:], "count":1}
        self.instance_dict[2] = {"embedding":background_emb[1,:], "count":1}
        self.wall_semantic_normalized = background_emb[0,:]/np.linalg.norm(background_emb[0,:])
        self.floor_semantic_normalized = background_emb[1,:]/np.linalg.norm(background_emb[1,:])
        self.color_top_down_height = (self.camera_height + 1) * np.ones((self.gs, self.gs), dtype=np.float32)#np.zeros((self.gs, self.gs), dtype=np.float32)
        self.color_top_down = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.grid = np.zeros((self.gs, self.gs, self.feat_dim), dtype=np.float32)
        self.obstacles = np.zeros((self.gs, self.gs), dtype=np.uint8)
        self.weight = np.zeros((self.gs, self.gs), dtype=np.float32)
        self.similarity_instance_map = np.zeros((self.gs, self.gs), dtype = int)
        self.visited = np.zeros((self.gs, self.gs), dtype = bool)
    
    def start_map(self):
        self._init_map()
        self.save_map()
    
    # @abstractmethod
    def save_map(self):
        print("simil",np.unique(self.similarity_instance_map))
        print("hello")
        self.datamanager.save_map(color_top_down=self.color_top_down,
                                  grid=self.similarity_instance_map,
                                  obstacles=self.obstacles,
                                  weight=self.weight,
                                  instance_dict = self.instance_dict)