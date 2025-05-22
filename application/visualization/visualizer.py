import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from map.utils.mapping_utils import load_map

class visualizer():
    def __init__(self, config, output_path=None):
        self.config = config
        self.bool_save = config["save_image"]
        if self.bool_save:
            self.output_path = output_path
        else:
            self.output_path = None
        self.target_map = config["visualize"]
        self.data_dir = os.path.join(config["data_path"], config["data_type"], config["dataset_type"], config["scene_id"], "map")
        self.target_dir = os.path.join(self.data_dir, f"{config['scene_id']}_{config['version']}")
        self.gt_dir = os.path.join(self.data_dir,f"{config['scene_id']}_{config['gt_version']}")
        self.scene_id = config["scene_id"]
        self.version = config["version"]
        self.dataset_type = config["dataset_type"]

    def visualize(self):
        self.visualize_obstacles()
        for target in self.target_map:
            if target == "obstacles":
                self.visualize_obstacles()
                continue
            if target == "rgb":
                self.visualize_rgb()
                continue
            if target == "map":
                self.visualize_map()
                continue
            if target == "heatmap":
                self.visualize_heatmap()
                continue

    def visualize_obstacles(self):
        obstacles_path = os.path.join(self.target_dir, f"obstacles_{self.version}.npy")
        obstacles = load_map(obstacles_path)
        if self.vlm == "ours":
            x_indices, y_indices = np.where(obstacles == 0)
        else:
            x_indices, y_indices = np.where(obstacles == 1)
        xmin = np.min(x_indices)
        xmax = np.max(x_indices)
        ymin = np.min(y_indices)
        ymax = np.max(y_indices)
        self.target_bbox = (xmin, xmax, ymin, ymax)
        obstacles_pil = Image.fromarray(obstacles[xmin:xmax+1, ymin:ymax+1])
        plt.figure(figsize=(8,6),dpi=120)
        plt.imshow(obstacles_pil, cmap="gray")
        plt.title(f"{self.vlm}_obstacles")
        if self.bool_save:
            plt.savefig(os.path.join(self.data_dir, f"{self.vlm}_obstacles.png"), bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)

        if "gt" in self.target_map:
            gt_obstacles_path = os.path.join(self.gt_dir, f"obstacles_{self.config['gt_version']}.npy")
            gt_obstacles = load_map(gt_obstacles_path)
            x_indices, y_indices = np.where(gt_obstacles == 0)
            xmin = np.min(x_indices)
            xmax = np.max(x_indices)
            ymin = np.min(y_indices)
            ymax = np.max(y_indices)
            self.gt_bbox = (xmin, xmax, ymin, ymax)
            gt_obstacles_pil = Image.fromarray(gt_obstacles[xmin:xmax+1, ymin:ymax+1])
            plt.figure(figsize=(8,6),dpi=120)
            plt.imshow(gt_obstacles_pil, cmap="gray")
            plt.title(f"gt_obstacles")
            if self.bool_save:
                plt.savefig(os.path.join(self.data_dir, f"{self.config['gt_version']}_obstacles.png"), bbox_inches='tight', pad_inches=0.1)
            plt.show(block=False)


    def visualize_rgb(self):
        rgb_path = os.path.join(self.target_dir, f"color_top_down_{self.version}")
        rgb = load_map(rgb_path)
        rgb =rgb[self.target_bbox[0]:self.target_bbox[1]+1, self.target_bbox[2]:self.target_bbox[3]+1]
        rgb_pil = Image.fromarray(rgb)
        plt.figure(figsize=(8,6),dpi=120)
        plt.imshow(rgb_pil)
        plt.title(f"{self.vlm}_rgb")
        if self.bool_save:
            plt.savefig(os.path.join(self.data_dir, f"{self.vlm}_rgb.png"), bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)

        if "gt" in self.target_map:
            gt_rgb_path = os.path.join(self.gt_dir, f"color_top_down_{self.config['gt_version']}.npy")
            gt_rgb = load_map(gt_rgb_path)
            gt_rgb =gt_rgb[self.gt_bbox[0]:self.gt_bbox[1]+1, self.gt_bbox[2]:self.gt_bbox[3]+1]
            gt_rgb_pil = Image.fromarray(gt_rgb)
            plt.figure(figsize=(8,6),dpi=120)
            plt.imshow(gt_rgb_pil)
            plt.title(f"gt_rgb")
            if self.bool_save:
                plt.savefig(os.path.join(self.data_dir, f"{self.config['gt_version']}_rgb.png"), bbox_inches='tight', pad_inches=0.1)
            plt.show(block=False)
    
    def visualize_map(self):
        map_path = os.path.join(self.target_dir, f"grid_{self.version}.npy")
        #! 그 3D 형태로 2d mask들이 쌓인 느낌? 이거 그 인스턴스마다 높이값 일정하게 저장되어있으니 활용하면 될듯?
        raise NotImplementedError
    
    def visualize_instance(self):
        raise NotImplementedError
    

    def visualize_heatmap(self):
        raise NotImplementedError