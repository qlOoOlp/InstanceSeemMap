import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os

from application.evaluation.metrics.iou import IoU
from application.evaluation.metrics.metrics import gt_idx_change, idxMap, SegmentationMetric
from map.utils.matterport3d_categories import mp3dcat
from map.utils.mapping_utils import load_map


def metrics():
    def __init__(self,):
        raise NotImplementedError
    
    def visualize_map(self, ):
        raise NotImplementedError
    
    def get_metrics(self, ):
        raise NotImplementedError