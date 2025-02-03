import numpy as np
from map.utils.mapping_utils import load_map, save_map
import json
from map.utils.replica_categories import replica_cat, cat2id#, new_cat

# cat2id = {value:key for key, value in replica_cat.items()}


# path = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data/replica/room_1/habitat/info_semantic.json"
semantic_path = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data/habitat_sim/replica/apartment_0_1/map/apartment_0_1_gt/semantic_info_gt.json"
map_path = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data/habitat_sim/replica/apartment_0_1/map/apartment_0_1_gt2/grid_gt2_ori.npy"
new_map_path = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data/habitat_sim/replica/apartment_0_1/map/apartment_0_1_gt2/grid_gt2.npy"

map = load_map(map_path)
new_map = np.zeros_like(map)

with open(semantic_path, "r",encoding="utf-8") as file:
    ori_sem = json.load(file)

# with open(path, "r",encoding="utf-8") as file:
#     data = json.load(file)

# p = 0
categories = []

for i in range(map.shape[0]):
    for j in range(map.shape[1]):
        if map[i,j] == -1:
            new_map[i,j] = 102
        else: new_map[i,j] = map[i,j]
        # try:category = ori_sem[map[i,j]]
        # except:
        #     print(map[i,j])
        #     raise Exception("sdf")
        # try:
        #     new_id = cat2id[category]
        # except:
        #     if int(map[i,j])==0:
        #         new_map[i,j] = 0
        #         continue
        #     else:
        #         raise ValueError(f"{map[i,j]} : category {category} not found in new_cat")
        # new_map[i,j] = new_id
        # if category not in categories:
        #     categories.append(category)
        # print(map[i,j], ori_sem[map[i,j]], new_id, p)

print(categories)
print(np.unique(new_map))
save_map(new_map_path,new_map)