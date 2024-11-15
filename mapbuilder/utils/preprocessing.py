import numpy as np
from typing import Tuple
from numpy.typing import NDArray




def IQR(map_idx:NDArray, pc:NDArray, depth_shape:Tuple[int,int], h_ratio:float, w_ratio:float)-> NDArray:
    for id in np.unique(map_idx):
        if id == 0 : continue
        feat_mask = (map_idx == id).astype(np.uint8)
        depths = {}
        for i,j in np.argwhere(feat_mask==1):
            new_i = int(i * h_ratio)
            new_j = int(j * w_ratio)
            depth_val = pc[:,new_i*depth_shape[1]+new_j][2]
            if depth_val < 0.1 or depth_val > 3:
                map_idx[i,j] = 0
                continue
            depths[(i,j)] = depth_val
        if len(depths) == 0: continue
        depths_val = list(depths.values())
        depths_key = list(depths.keys())
        Q1 = np.percentile(depths_val, 25)
        Q3 = np.percentile(depths_val, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        for i in range(len(depths_val)):
            depth_val = depths_val[i]
            x,y = depths_key[i]
            if depth_val < lower_bound or depth_val > upper_bound:
                map_idx[x,y] = 0
                depths.pop((x,y))
    return map_idx


