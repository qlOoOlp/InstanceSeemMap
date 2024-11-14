import numpy as np
import torch
from omegaconf import DictConfig


from mapbuilder.map.lsegmap import LsegMap
from mapbuilder.map.seemmap import SeemMap


class MapBuilder():
    def __init__(self, conf:DictConfig):
        self.conf = conf
        self.vlm = self.conf["vlm"]
        if self.vlm == "lseg":
            self.map = LsegMap(self.conf)
        elif self.vlm == "seem":
            self.map = SeemMap(self.conf)
    def buildmap(self):
        self.map.start_map()
        self.map.processing()
        self.map.save_map()
        print("Map building done")