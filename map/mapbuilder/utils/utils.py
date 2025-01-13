import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def rgbLoader(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def depthLoader(path):
    with open(path, 'rb') as f:
        depth = np.load(f)
    return depth

def poseLoader(path):
    with open(path, "r") as f:
        line = f.readline()
        row = [float(x) for x in line.split()]
        pos = np.array(row[:3], dtype=float).reshape((3, 1))
        quat = row[3:]
        r = R.from_quat(quat)
        rot = r.as_matrix()
        return pos, rot
    
def depthLoader2(path):
    raise NotImplementedError

def poseLoader2(path):
    raise NotImplementedError