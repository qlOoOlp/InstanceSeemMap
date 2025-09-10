import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import clip
import pickle as pkl

from utils.utils_map import get_map_bbox, visualize_obstacles, visualize_rgb, visualize_map, visualize_2Dmap_cat, visualize_2Dmap_categorized, visualize_2Dmap_inst_in_cat, visualize_instances, visualize_heatmap, visualize_room, viz_map
from map.utils.mapping_utils import load_map
from map.seem.base_model import build_vl_model
from map.utils.matterport3d_categories import mp3dcat
from map.utils.replica_categories import replica_cat
from map.utils.hm3dsem_categories import hm3dsem_cat

CLIP_FEAT_DIM_DICT = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}
class visualizer():
    def __init__(self, config, output_path=None):
        self.config = config
        self.bool_save = config["save_image"]
        self.target_map = config["visualize"]
        self.data_dir = os.path.join(config["data_path"], config["data_type"], config["dataset_type"], config["scene_id"], "map")
        self.target_dir = os.path.join(self.data_dir, f"{config['scene_id']}_{config['version']}")
        self.bool_use_avg_height = config["using_avg_height"]
        if output_path:
            self.output_path = output_path
        else:
            self.output_path = os.path.join(self.target_dir, 'viz/')
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'categories'), exist_ok=True)
        self.gt_dir = os.path.join(self.data_dir,f"{config['scene_id']}_{config['gt_version']}/01buildFeatMap/")
        self.scene_id = config["scene_id"]
        self.version = config["version"]
        self.dataset_type = config["dataset_type"]
        self.vlm = config["vlm"]
        self.using_categorized_map = config["using_categorized_map"]
        self.using_postprocessed_obstacles = config["using_postprocessed_obstacles"]
        self.load_vlm()
        self.grid_flag = False
        if self.dataset_type == "mp3d":
            self.categories = mp3dcat
        elif self.dataset_type in ["Replica","replica", "hm3dsem"]:
            self.categories = replica_cat
        elif self.dataset_type == "hm3dsem":
            self.categories = hm3dsem_cat
        self.floor_mask = config["filtering_floor"]

    def visualize(self):
        if self.using_postprocessed_obstacles:
            obstacles_path = os.path.join(self.target_dir,"02buildCatMap", f"semantic_obstacles_{self.version}.npy")
        else:
            obstacles_path = os.path.join(self.target_dir,"01buildFeatMap", f"obstacles_{self.version}.npy")
        rgb_path = os.path.join(self.target_dir,"01buildFeatMap", f"color_top_down_{self.version}.npy")
        
        self.obstacles = load_map(obstacles_path)
        if self.using_postprocessed_obstacles:
            self.bbox = get_map_bbox(self.obstacles, obstacle_index=1)
        else: self.bbox = get_map_bbox(self.obstacles)#, obstacle_index=1)#)
        print(f"bbox: {self.bbox}")
        self.rgb = load_map(rgb_path)

        if "gt" in self.target_map:
            gt_obstacles_path = os.path.join(self.gt_dir, f"obstacles_{self.config['gt_version']}.npy")
            self.gt_obstacles = load_map(gt_obstacles_path)
            self.gt_bbox = get_map_bbox(self.gt_obstacles)
            self.gt_rgb = load_map(os.path.join(self.gt_dir, f"color_top_down_{self.config['gt_version']}.npy"))
            self.gt_grid = load_map(os.path.join(self.gt_dir, f"grid_{self.config['gt_version']}.npy"))
        for target in self.target_map:
            if target == "obstacles":
                self.visualize_obstacles()
                continue
            if target == "rgb":
                self.visualize_rgb()
                continue
            if target == "2Dmap":
                self.visualize_2Dmap()
                continue
            if target == "map":
                self.visualize_map()
                continue
            if target == "instance":
                self.visualize_instance()
                continue
            if target == "instances_in_cat":
                self.visualize_instances_in_cat()
                continue
            if target == "heatmap":
                self.visualize_heatmap()
                continue
            if target == "room":
                self.visualize_room()
                continue

    def visualize_obstacles(self):
        if self.bool_save:
            visualize_obstacles(self.obstacles, bbox=self.bbox, is_show=False, save_path = self.output_path, save_name=f"{self.vlm}_obstacles")
        else:
            visualize_obstacles(self.obstacles, bbox=self.bbox, is_show = True)
        if "gt" in self.target_map:
            if self.bool_save:
                visualize_obstacles(self.gt_obstacles, bbox=self.gt_bbox, is_show=False, save_path = self.output_path, save_name=f"gt_{self.config['gt_version']}_obstacles")
            else:
                visualize_obstacles(self.gt_obstacles, bbox=self.gt_bbox, is_show = True)

    def visualize_rgb(self):
        if self.bool_save:
            visualize_rgb(self.rgb, bbox=self.bbox, is_show=False, save_path = self.output_path, save_name=f"{self.vlm}_rgb")
        else:
            visualize_rgb(self.rgb, bbox=self.bbox, is_show = True)
        if "gt" in self.target_map:
            if self.bool_save:
                visualize_rgb(self.gt_rgb, bbox=self.gt_bbox, is_show=False, save_path = self.output_path, save_name=f"gt_{self.config['gt_version']}_rgb")
            else:
                visualize_rgb(self.gt_rgb, bbox=self.gt_bbox, is_show = True)

    def visualize_2Dmap(self):
        self.get_grid()
        if self.vlm == "ours":
            if self.bool_save:
                visualize_2Dmap_categorized(self.vlm, self.categories, (self.grid,self.instance_dict), self.bbox, is_show=False, use_avg_height=self.bool_use_avg_height, save_path=self.output_path, save_name=f"{self.vlm}_2Dmap_categorized", floor_mask=self.floor_mask)
            else:
                visualize_2Dmap_categorized(self.vlm, self.categories, (self.grid,self.instance_dict), self.bbox, is_show=True, use_avg_height=self.bool_use_avg_height, floor_mask=self.floor_mask)
        else:
            if self.bool_save:
                visualize_2Dmap_cat(self.vlm, self.model, self.categories, self.grid, use_avg_height=self.bool_use_avg_height, bbox=self.bbox, is_show=False, save_path=self.output_path, save_name=f"{self.vlm}_2Dmap", floor_mask=self.floor_mask)
            else:
                visualize_2Dmap_cat(self.vlm, self.model, self.categories, self.grid, use_avg_height=self.bool_use_avg_height, bbox=self.bbox, is_show=True, floor_mask=self.floor_mask)
        if "gt" in self.target_map:
            if self.bool_save:
                viz_map(self.gt_grid, self.categories, floor_mask=True, title=f"gt_{self.config['gt_version']}_2Dmap", is_show=False, save_path=self.output_path, save_name=f"gt_{self.config['gt_version']}_2Dmap", bbox=self.gt_bbox)
            else:
                viz_map(self.gt_grid, self.categories, floor_mask=True, title=f"gt_{self.config['gt_version']}_2Dmap", is_show=True, bbox=self.gt_bbox)

    def visualize_map(self):
        self.get_grid()
        map_path = os.path.join(self.target_dir, f"grid_{self.version}.npy")
        if self.bool_save:
            visualize_map((self.grid, self.instance_dict), bbox=self.bbox, is_show=False, save_path=self.output_path, save_name=f"{self.vlm}_map")
        else:
            visualize_map((self.grid, self.instance_dict), bbox=self.bbox, is_show=True)

    def visualize_instance(self):
        self.get_grid()
        instances = self.config["instances_to_visualize"]
        if not instances:
            instances = self.instance_dict.keys() 
        if self.bool_save:
            visualize_instances(self.rgb, instances, (self.grid, self.instance_dict), bbox=self.bbox, is_show=False, save_path=self.output_path, save_name=f"{self.vlm}_instances")
        else:
            visualize_instances(self.rgb, instances, (self.grid, self.instance_dict), bbox=self.bbox, is_show=True)
    
    def visualize_instances_in_cat(self):
        self.get_grid()
        categories = self.config["categories_to_visualize"]
        if not categories: categories = self.categories
        if self.bool_save:
            visualize_2Dmap_inst_in_cat(categories, (self.grid, self.instance_dict), bbox=self.bbox, is_show=False, save_path=os.path.join(self.output_path,'categories'), save_name=f"{self.vlm}_categories")
        else:
            visualize_2Dmap_inst_in_cat(categories, (self.grid, self.instance_dict), bbox=self.bbox, is_show=True)

    def visualize_heatmap(self):
        self.get_grid()
        target = self.config["heatmap_target"]
        heatmap_type = self.config["heatmap_type"]
        if not target: raise ValueError("Please specify the target for heatmap visualization.")
        if self.bool_save:
            visualize_heatmap(self.vlm, self.model, target, self.rgb, (self.grid, self.instance_dict), heatmap_type=heatmap_type, bbox=self.bbox, is_show=False, save_path=self.output_path, save_name=f"{self.vlm}_heatmap_{target}",categories=self.categories)
        else:
            visualize_heatmap(self.vlm, self.model, target, self.rgb, (self.grid, self.instance_dict), heatmap_type=heatmap_type, bbox=self.bbox, is_show=True, categories=self.categories)

    def visualize_room(self):
        raise NotImplementedError

    def load_vlm(self):
        if self.vlm == "lseg":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_version = "ViT-B/32"
            self.clip_feat_dim = CLIP_FEAT_DIM_DICT[clip_version]
            self.model, preprocess = clip.load(clip_version)  # clip.available_models()
            self.model.to(device).eval()

        elif self.vlm in ["seem", "ours", "floodfill", "dbscan"]:
            self.model = build_vl_model("seem", input_size = 360)

    def get_grid(self):
        if not self.grid_flag:
            grid_path = os.path.join(self.target_dir,"01buildFeatMap", f"grid_{self.version}.npy")
            self.grid = load_map(grid_path)
            if self.vlm == "ours":
                if self.using_categorized_map:
                    instance_dict_path = os.path.join(self.target_dir,"02buildCatMap", f"categorized_instance_dict_{self.version}.pkl")
                    with open(instance_dict_path, 'rb') as f:
                        self.instance_dict = pkl.load(f)
                else:
                    instance_dict_path = os.path.join(self.target_dir,"01buildFeatMap", f"instance_dict_{self.version}.pkl")
                    with open(instance_dict_path, 'rb') as f:
                        self.instance_dict = pkl.load(f)
            self.grid_flag = True
        return