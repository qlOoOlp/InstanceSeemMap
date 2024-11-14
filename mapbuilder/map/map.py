import numpy as np
import torch
from abc import ABC, abstractmethod
from mapbuilder.utils.datamanager import DataManager


class Map(ABC):

    @abstractmethod
    def processing(self):
        pass

    @abstractmethod
    def preprocessing(self):
        pass

    @abstractmethod
    def postprocessing(self):
        pass

    @abstractmethod
    def _load_map(self):
        pass

    @abstractmethod
    def _init_map(self):
        pass

    @abstractmethod
    def start_map(self):
        pass

    @abstractmethod
    def save_map(self):
        pass