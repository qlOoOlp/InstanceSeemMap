import numpy as np
import torch
from omegaconf import DictConfig


from mapbuilder.map.lsegmap import LsegMap
from mapbuilder.map.seemmap import SeemMap
from mapbuilder.map.seemmap_tracking import SeemMap_tracking
from mapbuilder.map.seemmap_dbscan import SeemMap_dbscan
from mapbuilder.map.seemmap_floodfill import SeemMap_floodfill


class MapBuilder():
    def __init__(self, conf:DictConfig):
        self.conf = conf
        self.vlm = self.conf["vlm"]
        if self.vlm == "lseg":
            self.map = LsegMap(self.conf)
        elif self.vlm == "seem":
            if self.conf["seem_type"]=="base": self.map = SeemMap(self.conf)
            elif self.conf["seem_type"]=="tracking" : self.map = SeemMap_tracking(self.conf)
            elif self.conf["seem_type"]=="dbscan" : self.map = SeemMap_dbscan(self.conf)
            elif self.conf["seem_type"]=="floodfill" : self.map = SeemMap_floodfill(self.conf)
    def buildmap(self):
        self.map.start_map()
        self.map.processing()
        self.map.save_map()
        print("Map building done")