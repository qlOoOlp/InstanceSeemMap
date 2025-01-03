import numpy as np
from map.utils.clip_utils import get_text_feats
from map.seem.base_model import build_vl_model
 
def get_indMap(gridMap, vlnType, lang, xymaxmin):
    if "lseg" in vlnType:
        import torch
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_version = "ViT-B/32"
        clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                        'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
        clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
        clip_model.to(device).eval()

        text_feats = get_text_feats(lang, clip_model, clip_feat_dim)
        map_feats = gridMap.reshape((-1, gridMap.shape[-1]))
        scores_list = map_feats @ text_feats.T

        predicts = np.argmax(scores_list, axis=1)
        predicts = predicts.reshape((xymaxmin[0] - xymaxmin[1] + 1, xymaxmin[2] - xymaxmin[3] + 1))
    elif "seem" in vlnType:
        print("Original SEEM based")
        model = build_vl_model("seem", input_size = 360)
        t_emb=model.encode_prompt(lang,task="default")
        t_npy = t_emb.cpu().numpy()
        map_feats = gridMap.reshape((-1, gridMap.shape[-1]))
        scores_list = map_feats @ t_npy.T
        predicts = np.argmax(scores_list, axis=1)
        predicts = predicts.reshape((xymaxmin[0] - xymaxmin[1] + 1, xymaxmin[2] - xymaxmin[3] + 1))
    elif "sam" in vlnType:
        NotImplementedError
    elif vlnType == "dummy_geo_dist4_reverse3_4llava_wall4":
        from map.utils.mapping_utils import load_map
        ggrid_map = load_map("/home/hong/VLMAPS/vlseem/Data/habitat/vlmaps_dataset/UwV83HsGsw3_1/map/gggrid_dummy_geo_dist4_reverse3_4llava_wall4.npy")
        return ggrid_map



    elif vlnType == "hello":
        model = build_vl_model("seem", input_size = 360)
        import pickle
        embeddings_path = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data/habitat_sim/2t7WUuJeko7_2_mini/map/2t7WUuJeko7_2_mini_tracking_quality/instance_dict_tracking_quality.pkl"

        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        grid = gridMap
        print(grid.shape)
        id_list = []
        instance_feat = []
        print("embeddings:",len(embeddings.keys()))
        t_emb=model.encode_prompt(lang,task="default")
        t_npy = t_emb.cpu().numpy()

        ids=[]
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for arg in grid[i,j].keys():
                    if arg not in ids:
                        ids.append(arg)
        embeddings[1]["avg_height"] = 0
        embeddings[2]["avg_height"] = 0

        # np.set_printoptions(threshold=np.inf)
        # print(ids)
        # print("grid ids:",len(ids))
        # print(grid)
        # print(embeddings)
        for id, val in embeddings.items():
            id_list.append(id)
            instance_feat.append(val["embedding"])
        instance_feat = np.array(instance_feat)
        # print("instance_feat:",instance_feat.shape)
        scores_list = instance_feat @ t_npy.T
        # print("scores_list:",scores_list)
        # key_list = list(embeddings.keys())
        # print(key_list)
        # print(scores_list.shape)
        predicts = np.argmax(scores_list, axis=1)
        ggrid_map = np.zeros((grid.shape[0], grid.shape[1]), dtype=int)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if len(grid[i,j].keys()) == 0:
                    continue
                if len(grid[i,j].keys()) == 1:
                    # print(grid[i,j][0])
                    for key in grid[i,j].keys():
                        # if key in [1,0]:
                        #     ggrid_map[i,j]=key
                        #     continue

                        if lang[predicts[list(embeddings.keys()).index(key)]] == "lighting": continue
                        ggrid_map[i,j] = predicts[list(embeddings.keys()).index(key)]
                else:
                    max_conf = 0
                    max_height = 50000
                    max_observed = 0
                    for key, val in grid[i,j].items():
                        if key == 2 :continue
                        # print(arg)
                        # print(predicts.shape)
                        # if key in [1,0]: continue
                        candidate = predicts[list(embeddings.keys()).index(key)]
                        # if lang[candidate] == "lighting": continue
                        candidate_conf = val[0]
                        candidate_height = embeddings[key]["avg_height"] #val[1]
                        candidate_observed = val[2]
                        # if max_conf < candidate_conf:
                        #     max_conf = candidate_conf
                        #     candidate_val = candidate
                        if max_height > candidate_height:
                            max_height = candidate_height
                            candidate_val = candidate
                        # if candidate_observed > max_observed:
                        #     max_observed = candidate_observed
                        #     candidate_val = candidate
                    ggrid_map[i,j] = candidate_val
        predicts = ggrid_map.reshape((-1, ggrid_map.shape[-1]))



    else: #"dummy_instance_height" in vlnType:
        model = build_vl_model("seem", input_size = 360)
        import pickle
        embeddings_path = "/home/hong/VLMAPS/vlseem/Data/habitat/vlmaps_dataset/1LXtFkjw3qL_1/map/instance_dict_dummy_geo_dist4_reverse3_4llava_wall4.pkl"

        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        grid = gridMap
        print(grid.shape)
        id_list = []
        instance_feat = []
        print("embeddings:",len(embeddings.keys()))
        t_emb=model.encode_prompt(lang,task="default")
        t_npy = t_emb.cpu().numpy()
        # print(embeddings.keys())
        # print("grid:",grid.shape)



        # ids=[]
        # for i in range(grid.shape[0]):
        #     for j in range(grid.shape[1]):
        #         for arg in grid[i,j].keys():
        #             if arg not in ids:
        #                 ids.append(arg)

        # # np.set_printoptions(threshold=np.inf)
        # # print(ids)
        # # print("grid ids:",len(ids))
        # # print(grid)
        # # print(embeddings)
        # # print(type(embeddings))
        # # print(embeddings.keys())
        # # print(embeddings[0].keys())
        # for id, val in embeddings.items():
        #     id_list.append(id)
        #     instance_feat.append(val["embedding"])
        # instance_feat = np.array(instance_feat)
        # # print("instance_feat:",instance_feat.shape)
        # scores_list = instance_feat @ t_npy.T
        # # print("scores_list:",scores_list)
        # # key_list = list(embeddings.keys())
        # # print(key_list)
        # # print(scores_list.shape)
        # predicts = np.argmax(scores_list, axis=1)
        # ggrid_map = np.zeros((grid.shape[0], grid.shape[1]), dtype=int)
        # for i in range(grid.shape[0]):
        #     for j in range(grid.shape[1]):
        #         if len(grid[i,j].keys()) == 0:
        #             continue
        #         if len(grid[i,j].keys()) == 1:
        #             # print(grid[i,j][0])
        #             for key in grid[i,j].keys():
        #                 ggrid_map[i,j] = predicts[list(embeddings.keys()).index(key)]
        #         else:
        #             max_conf = 0
        #             max_height = 50000
        #             # candidate_val = 0
        #             for key, val in grid[i,j].items():
        #                 # print(arg)
        #                 # print(predicts.shape)
        #                 candidate = predicts[list(embeddings.keys()).index(key)]
        #                 candidate_conf = val#[0]
        #                 candidate_height = val#[1] #! here
        #                 # if max_conf < candidate_conf:
        #                 #     max_conf = candidate_conf
        #                 #     candidate_val = candidate
        #                 if max_height > candidate_height:
        #                     max_height = candidate_height
        #                     candidate_val = candidate
        #             ggrid_map[i,j] = candidate_val
        # predicts = ggrid_map.reshape((-1, ggrid_map.shape[-1]))











        # #################################################################################################
        # ids=[]
        # for i in range(grid.shape[0]):
        #     for j in range(grid.shape[1]):
        #         for arg in grid[i,j].keys():
        #             if arg not in ids:
        #                 ids.append(arg)

        # # np.set_printoptions(threshold=np.inf)
        # # print(ids)
        # # print("grid ids:",len(ids))
        # # print(grid)
        # # print(embeddings)
        # for id, val in embeddings.items():
        #     id_list.append(id)
        #     instance_feat.append(val["embedding"])
        # instance_feat = np.array(instance_feat)
        # # print("instance_feat:",instance_feat.shape)
        # scores_list = instance_feat @ t_npy.T
        # # print("scores_list:",scores_list)
        # # key_list = list(embeddings.keys())
        # # print(key_list)
        # # print(scores_list.shape)
        # predicts = np.argmax(scores_list, axis=1)
        # ggrid_map = np.zeros((grid.shape[0], grid.shape[1]), dtype=int)
        # for i in range(grid.shape[0]):
        #     for j in range(grid.shape[1]):
        #         if len(grid[i,j].keys()) == 0:
        #             continue
        #         if len(grid[i,j].keys()) == 1:
        #             # print(grid[i,j][0])
        #             for key in grid[i,j].keys():
        #                 ggrid_map[i,j] = predicts[list(embeddings.keys()).index(key)]
        #         else:
        #             max_conf = 0
        #             max_height = 0
        #             # candidate_val = 0
        #             for key, val in grid[i,j].items():
        #                 # print(arg)
        #                 # print(predicts.shape)
        #                 candidate = predicts[list(embeddings.keys()).index(key)]
        #                 candidate_conf = val[0]
        #                 candidate_height = val[2]
        #                 # if max_conf < candidate_conf:
        #                 #     max_conf = candidate_conf
        #                 #     candidate_val = candidate
        #                 if max_height < candidate_height:
        #                     max_height = candidate_height
        #                     candidate_val = candidate
        #             ggrid_map[i,j] = candidate_val
        # predicts = ggrid_map.reshape((-1, ggrid_map.shape[-1]))











        #################################################################################################
        ids=[]
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for arg in grid[i,j].keys():
                    if arg not in ids:
                        ids.append(arg)

        # np.set_printoptions(threshold=np.inf)
        # print(ids)
        # print("grid ids:",len(ids))
        # print(grid)
        # print(embeddings)
        for id, val in embeddings.items():
            id_list.append(id)
            instance_feat.append(val["embedding"])
        instance_feat = np.array(instance_feat)
        # print("instance_feat:",instance_feat.shape)
        scores_list = instance_feat @ t_npy.T
        # print("scores_list:",scores_list)
        # key_list = list(embeddings.keys())
        # print(key_list)
        # print(scores_list.shape)
        predicts = np.argmax(scores_list, axis=1)
        ggrid_map = np.empty_like(grid, dtype=dict)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ggrid_map[i,j] = {}
                
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if len(grid[i,j].keys()) == 0:
                    continue
                # if len(grid[i,j].keys()) == 1:
                #     # print(grid[i,j][0])
                for key in grid[i,j].keys():
                    new_key = predicts[list(embeddings.keys()).index(key)]
                    ggrid_map[i,j][new_key] = grid[i,j][key][2]
                        # ggrid_map[i,j] = predicts[list(embeddings.keys()).index(key)]
                # else:
                #     max_conf = 0
                #     max_height = 0
                #     # candidate_val = 0
                #     for key, val in grid[i,j].items():
                #         # print(arg)
                #         # print(predicts.shape)
                #         candidate = predicts[list(embeddings.keys()).index(key)]
                #         candidate_conf = val[0]
                #         candidate_height = val[2]
                #         # if max_conf < candidate_conf:
                #         #     max_conf = candidate_conf
                #         #     candidate_val = candidate
                #         if max_height < candidate_height:
                #             max_height = candidate_height
                #             candidate_val = candidate
                #     ggrid_map[i,j] = candidate_val

        center_weight = 3
        gggrid_map = np.zeros_like(ggrid_map, dtype=np.uint16)
        grid_upper = np.empty((ggrid_map.shape[0]+1, ggrid_map.shape[1]+1), dtype=dict)
        for i in range(ggrid_map.shape[0]+1):
            for j in range(ggrid_map.shape[1]+1):
                grid_upper[i,j] = {}
        grid_upper[1:,1:] = ggrid_map
        for i in range(1,gggrid_map.shape[0]+1):
            for j in range(1,gggrid_map.shape[1]+1):
                candidate = grid_upper[i-1:i+2,j-1:j+2]
                item_dict = {}
                for candidate_i in range(candidate.shape[0]):
                    for candidate_j in range(candidate.shape[1]):
                        if candidate_i == i and candidate_j == j:
                            # if len(candidate[candidate_i,candidate_j]) == 0: continue
                            for key, val in candidate[candidate_i,candidate_j].items():
                                if key in item_dict.keys():
                                    item_dict[key] += center_weight * val
                                else: item_dict[key] = center_weight * val
                        else:
                            # if len(candidate[candidate_i,candidate_j]) == 0: continue
                            for key, val in candidate[candidate_i,candidate_j].items():
                                if key in item_dict.keys():
                                    item_dict[key] += 1 * val
                                else: item_dict[key] = 1 *  val
                if len(item_dict) ==0:
                    gggrid_map[i-1,j-1] = 0
                    continue
                max_key = max(item_dict, key=item_dict.get)
                gggrid_map[i-1,j-1] = max_key
        predicts = gggrid_map.reshape((-1, ggrid_map.shape[-1]))





    # else:
    #     raise Exception("Wrong vln type")
    return predicts