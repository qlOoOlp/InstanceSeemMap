import pickle
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import os
import numpy as np
from map.seem.base_model import build_vl_model
from map.utils.matterport3d_categories import mp3dcat
from map.utils.replica_categories import replica_cat
from map.utils.mapping_utils import load_map
from PIL import Image



class floodfill_process():
    def __init__(self, grid, category, embs, model):
        self.model = model
        self.grid = grid
        self.category = category
        self.embs = embs
        self.height, self.width = grid.shape[0], grid.shape[1]
        self.visited = np.zeros((grid.shape[0], grid.shape[1]), dtype=bool)
        self.similarity_instance_map = np.zeros((grid.shape[0], grid.shape[1]), dtype=int)
        self.similarity_threshold = 0.99
        self.instance_dict = {}
        background_emb = self.model.encode_prompt(["wall","floor"], task = "default")
        background_emb = background_emb.cpu().numpy()
        self.wall_semantic_normalized = background_emb[0,:]/np.linalg.norm(background_emb[0,:])
        self.floor_semantic_normalized = background_emb[1,:]/np.linalg.norm(background_emb[1,:])
        self.instance_dict[1] = {"embedding":background_emb[0], "count":0}
        self.instance_dict[2] = {"embedding":background_emb[1], "count":0}

    def process(self):
        current_instance = 3
        pbar = tqdm(total=self.height * self.width, leave=True)
        for i in range(self.height):
            for j in range(self.width):
                if not self.visited[i,j]:
                    if np.sum(self.grid[i,j])==0:
                        pbar.update(1)
                        continue
                    self.instance_dict[current_instance] = {"embedding":self.grid[i,j], "count":1}
                    self.flood_fill(i, j, current_instance)
                    current_instance += 1
                pbar.update(1)
        pbar.close()
        for item in np.unique(self.similarity_instance_map):
            if np.sum(self.similarity_instance_map==item)<50:
                self.similarity_instance_map[self.similarity_instance_map==item]=0
                self.instance_dict.pop(item)
        sem_features = [val["embedding"] for val in self.instance_dict.values()]
        sem_feat = np.vstack(sem_features)
        scores_list = sem_feat @ self.embs.T
        predicts = np.argmax(scores_list, axis=1)
        for i, key in enumerate(self.instance_dict.keys()):
            self.instance_dict[key]["category_id"] = predicts[i]
            self.instance_dict[key]["category"] = self.category[predicts[i]]
        return self.similarity_instance_map, self.instance_dict


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
                if similarity >= self.similarity_threshold:
                    self.similarity_instance_map[nx, ny] = current_instance
                    self.visited[nx, ny] = True
                    instance_embedding = self.instance_dict[current_instance]["embedding"]
                    instance_count = self.instance_dict[current_instance]["count"]
                    self.instance_dict[current_instance] = {"embedding":(instance_embedding*instance_count+current_embedding)/(instance_count+1), "count":instance_count+1}
                    stack.append((nx, ny))

class dbscan_process():
    def __init__(self, grid, category, embs, model, minmax):
        self.model = model
        self.grid = grid
        self.category = category
        self.embs = embs
        self.xmin, self.xmax, self.ymin, self.ymax = minmax
    
    def process(self):
        map_feats = self.grid.reshape(-1, self.grid.shape[-1])
        scores_list = map_feats @ self.embs.T
        predicts = np.argmax(scores_list, axis=1)
        predicts = predicts.reshape((self.xmax - self.xmin + 1, self.ymax - self.ymin + 1))
        instance_map, instance_dict, num_instances = self.instance_segmentation_with_dbscan(predicts, eps=2, min_samples=5)
        # class_map = self.create_class_map(instance_map, instance_dict)
        return instance_map, instance_dict


    def instance_segmentation_with_dbscan(self,map_array, eps=2, min_samples=5):
        num_class = len(np.unique(map_array))
        new_grid = np.zeros_like(map_array)
        inst= 1
        instance_dict ={}
        pbar = tqdm(total=len(np.unique(map_array)), leave=True)
        for category_id in np.unique(map_array):
            if category_id ==0:
                pbar.update(1)
                continue
            if category_id ==1 or category_id ==2:
                new_grid[map_array==category_id] = inst
                instance_dict[inst]={"category_id":category_id, "category":self.category[category_id]}
                inst+=1
                pbar.update(1)
                continue
            id_mask = np.where(map_array == category_id, 1, 0)
            if np.sum(id_mask) < 50 :
                pbar.update(1)
                continue
            coords = np.column_stack(np.where(id_mask==1))
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            db = DBSCAN(eps=2, min_samples=5).fit(coords)
            labels = db.labels_
            labeled_mask = np.zeros_like(id_mask, dtype=int)
            for label, (x,y) in zip(labels, coords):
                labeled_mask[x,y] = label + 1
            num_features = len(np.unique(labeled_mask))
            for key in range(1,num_features+1):
                instance_mask = (labeled_mask == key).astype(np.uint8)
                if np.sum(instance_mask) < 50 :continue
                new_grid[labeled_mask==key] = inst
                instance_dict[inst]={"category_id":category_id, "category":self.category[category_id]}
                inst+=1
            pbar.update(1)
        return new_grid, instance_dict, inst
    def create_class_map(self,instance_map, instance_dict):
        class_map = np.zeros_like(instance_map)
        for inst_id, info in instance_dict.items():
            category_id = info["category_id"]
            class_map[instance_map == inst_id] = category_id
        return class_map
    


def __main__():

    model = build_vl_model("seem", input_size=360)
    root_dir = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data/habitat_sim"
    dataset_type = "replica"
    scenes = ["apartment_0_1","apartment_1_4", "apartment_2_2", "frl_apartment_1_1" ,"office_2_1" ,"office_3_1", "office_4_1", "room_0_1", "room_1_1"]#["1LXtFkjw3qL_1_none","2t7WUuJeko7_2","5LpN3gDmAk7_4","UwV83HsGsw3_1"]
    version = "seem_new"
    targets = ["floodfill","dbscan"]
    if dataset_type == "mp3d":
        categories = mp3dcat
    elif dataset_type == "replica":
        categories = replica_cat
    else:
        raise ValueError(f"dataset_type {dataset_type} not supported")


    for scene in range(len(scenes)):
        scene_id = scenes[scene]
        print(f"{scene_id} : {scene+1}/{len(scenes)}")
        path = os.path.join(root_dir, dataset_type, scene_id, "map", f"{scene_id}_{version}")
        obstacle_path = os.path.join(path, f"obstacles_{version}.npy")
        grid_path = os.path.join(path, f"grid_{version}.npy")
        obstacles = load_map(obstacle_path)
        grid = load_map(grid_path)
        x_indices, y_indices = np.where(obstacles == 1)
        xmin = np.min(x_indices)
        xmax = np.max(x_indices)
        ymin = np.min(y_indices)
        ymax = np.max(y_indices)
        if xmin==0 and ymin==0:
            x_indices, y_indices = np.where(obstacles == 0)
            xmin = np.min(x_indices)
            xmax = np.max(x_indices)
            ymin = np.min(y_indices)
            ymax = np.max(y_indices)
        grid = grid[xmin:xmax+1, ymin:ymax+1]
        embs = model.encode_prompt(categories, task="default")
        embs = embs.cpu().numpy()
        if "floodfill" in targets:
            flmap_path = os.path.join(path, f"floodfill_{version}.npy")
            flinstance_dict_path = os.path.join(path, f"floodfill_instance_dict_{version}.pkl")
            flmap = floodfill_process(grid, categories, embs, model)
            floodfill_map, floodfill_instance_dict = flmap.process()
            print(np.unique(floodfill_map))
            print(floodfill_instance_dict.keys())
            print(len(np.unique(floodfill_map)), len(floodfill_instance_dict))
            np.save(flmap_path, floodfill_map)
            with open(flinstance_dict_path, 'wb') as f:
                pickle.dump(floodfill_instance_dict, f)
        if "dbscan" in targets:
            dbmap_path = os.path.join(path, f"dbscan_{version}.npy")
            dbinstance_dict_path = os.path.join(path, f"dbscan_instance_dict_{version}.pkl")
            dbmap = dbscan_process(grid, categories, embs, model, (xmin, xmax, ymin, ymax))
            dbscan_map, dbscan_instance_dict = dbmap.process()
            print(np.unique(dbscan_map))
            print(len(np.unique(dbscan_map)), len(dbscan_instance_dict))
            np.save(dbmap_path, dbscan_map)
            with open(dbinstance_dict_path, 'wb') as f:
                pickle.dump(dbscan_instance_dict, f)


if __name__ == "__main__":
    __main__()