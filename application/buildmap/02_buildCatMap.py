
#! gt맵 만드는 옵션 추가해야됨
import os
import sys
from map.mapbuilder.utils.mapbuilder import CategorizedMapBuilder
from utils.parser import parse_args_indexing_map, save_args
from omegaconf import OmegaConf

def main():
    args=parse_args_indexing_map()
    config = OmegaConf.create(vars(args))
    mapbuilder=CategorizedMapBuilder(config)
    mapbuilder.processing()

if __name__=="__main__":
    main()