import numpy as np
import torch
from numpy.typing import NDArray
from seem.xdecoder_seem.utils.constants import COCO_PANOPTIC_CLASSES

def get_SEEM_feat(model : torch.nn.Module, image: NDArray, threshold_confidence = 0.5):
    features = model.encode_image([image], mode="default")[0]
    map_idx = features["conf_idx"] 
    map_idx_np = map_idx.cpu().numpy()
    if len(np.unique(map_idx_np)) == 1 : return [None, None, None, None]
    
    num_categories = len(features["category_id"])
    category_id = features["category_id"]
    map_conf = features["conf_score"]
    embeddings = features["caption"]


    COCO_PANOPTIC_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
    # pano_category = ['ceiling-merged','floor-wood','floor-other-merged','pavement-merged','platform','playingfield','road','sand','snow','dirt-merged','gravel','sky-other-merged','mountain-merged','river','sea','tree-merged','fence-merged','grass-merged','rock-merged','building-other-merged','wall-brick','wall-stone','wall-tile','wall-wood','wall-other-merged']
    pano_category = ['ceiling-merged','pavement-merged','platform','playingfield','road','sand','snow','dirt-merged','gravel','sky-other-merged','mountain-merged','river','sea','tree-merged','fence-merged','grass-merged','rock-merged','building-other-merged']
    floor_category = ['floor-wood','floor-other-merged',]
    wall_category = ['wall-brick','wall-stone','wall-tile','wall-wood','wall-other-merged']


    category_id = category_id.cpu().numpy()
    map_conf = map_conf.cpu().numpy()
    embeddings = embeddings.cpu().numpy()
    embedding_dict = {}
    category_dict = {}
    new_map_idx_np = np.zeros_like(map_idx_np)

    map_idx_np[map_conf<threshold_confidence]=0




    # unique_values = np.unique(map_idx_np)  # 고유 인덱스 값 추출# 고유 인덱스 값의 개수에 따라 컬러맵 생성
    # colors = plt.cm.get_cmap("tab20", len(unique_values))
    # color_list = [colors(i) for i in range(len(unique_values))]
    # cmap = mcolors.ListedColormap(color_list)

    # # 고유 인덱스 값을 0부터 시작하는 인덱스로 매핑하는 사전 생성
    # norm = mcolors.BoundaryNorm(unique_values, cmap.N)

    # # 배열 시각화
    # fig, ax = plt.subplots()
    # cax = ax.imshow(map_idx_np, cmap=cmap, norm=norm)
    # plt.title("Category Map with Clear Color Matching")

    # # 인덱스 값과 색상을 매칭하는 범례 생성
    # patches = [mpatches.Patch(color=color_list[i], label=f"Category {int(unique_values[i])}")
    #         for i in range(len(unique_values))]
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # plt.show()

    # for id in category_id:
    #     print(f" [ {id} : {COCO_PANOPTIC_CLASSES[id]} ],", end=" ")
    # print()
    # print("############111111#############")
    # for id in np.unique(map_idx_np):
    #     if id == 0 :
    #         print(id, category_id[id-1], COCO_PANOPTIC_CLASSES[category_id[id-1]])
    #     elif COCO_PANOPTIC_CLASSES[category_id[id-1]] in floor_category:
    #         print(id, category_id[id-1], COCO_PANOPTIC_CLASSES[category_id[id-1]])
    #     elif COCO_PANOPTIC_CLASSES[category_id[id-1]] in wall_category:
    #         print(id, category_id[id-1], COCO_PANOPTIC_CLASSES[category_id[id-1]])
    #     else:
    #         print(id, category_id[id-1], COCO_PANOPTIC_CLASSES[category_id[id-1]])

    inst_id = 3
    for id in np.unique(map_idx_np):
        if id ==0: continue
        if COCO_PANOPTIC_CLASSES[category_id[id-1]] in wall_category:
            new_map_idx_np[map_idx_np == id] = 1
            embedding_dict[1] = embeddings[id-1]
        elif COCO_PANOPTIC_CLASSES[category_id[id-1]] in floor_category:
            new_map_idx_np[map_idx_np == id] = 2
            embedding_dict[2] = embeddings[id-1]
        elif COCO_PANOPTIC_CLASSES[category_id[id-1]] in pano_category: continue
        else:
            new_map_idx_np[map_idx_np == id] = inst_id
            embedding_dict[inst_id] = embeddings[id-1]
            category_dict[inst_id] = category_id[id-1]+2
            inst_id += 1
    # print("############222222#############")
    # print(category_dict.keys(), embedding_dict.keys(), np.unique(new_map_idx_np))
    # for id in np.unique(new_map_idx_np):
    #     if id in [0,1,2]:
    #         print(id)
    #     else:
    #         print(id, category_dict[id], COCO_PANOPTIC_CLASSES[category_dict[id]-2])
    # print("#########################")

    return [new_map_idx_np, map_conf, embedding_dict, category_dict]