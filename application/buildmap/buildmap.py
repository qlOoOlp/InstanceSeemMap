#%%
#! gt맵 만드는 옵션 추가해야됨
import os
import sys
from map.mapbuilder.utils.mapbuilder import MapBuilder
from utils.parser import parse_args, save_args
from omegaconf import OmegaConf

def main():
    args=parse_args()
    config = OmegaConf.create(vars(args))
    mapbuilder=MapBuilder(config)
    mapbuilder.buildmap()

if __name__=="__main__":
    main()
# %%
