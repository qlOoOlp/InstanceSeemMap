import numpy as np
import cv2
from PIL import Image
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
    
'''
img to npy
'''
def depthLoader2(path):
    with open(path, 'rb') as f:
        depth = np.array(Image.open(f))
        # print(depth)
    return depth/4000 #/5000

def poseLoader2(path):
    with open(path, "r") as f:
        lines = f.readlines()
        inst_mat = ' '.join(lines[0].split())
        inst_mat = np.array(inst_mat.split(), dtype=float).reshape((3, 3))
        trans_mat = ' '.join(lines[1].split())
        trans_mat = np.array(trans_mat.split(), dtype=float).reshape((4, 4))

        rot = trans_mat[:3, :3]
        pos = trans_mat[:3, 3]
        # row = [float(x) for x in line.split()]
        # pos = np.array(trans_mat.split()[:3], dtype=float).reshape((3, 1))
        # quat = row[3:]
        # r = R.from_quat(quat)
        # rot = r.as_matrix()

    return inst_mat, pos, rot