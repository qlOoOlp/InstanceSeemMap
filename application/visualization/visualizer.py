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
from map.utils.dataset_categories import resolve_dataset_categories

CLIP_FEAT_DIM_DICT = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}
class visualizer():
    def __init__(self, config, output_path=None):
        self.config = config
        self.bool_save = config.get("save_image", True)
        self.bool_show = config.get("show_image", (not self.bool_save))
        self.target_map = config.get("visualize", [])
        self.data_dir = os.path.join(config["data_path"], config["data_type"], config["dataset_type"], config["scene_id"], "map")
        self.target_dir = os.path.join(self.data_dir, f"{config['scene_id']}_{config['version']}")
        self.bool_use_avg_height = config.get("using_avg_height", True)
        if output_path:
            self.output_path = output_path
        else:
            self.output_path = os.path.join(self.target_dir, 'viz/')
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'categories'), exist_ok=True)
        self.gt_dir = os.path.join(self.data_dir,f"{config['scene_id']}_{config['gt_version']}/01buildFeatMap/")
        self.gt_version = config["gt_version"]
        self.scene_id = config["scene_id"]
        self.version = config["version"]
        self.dataset_type = config["dataset_type"]
        self.vlm = config["vlm"]
        self.using_categorized_map = config.get("using_categorized_map", False)
        self.using_postprocessed_obstacles = config.get("using_postprocessed_obstacles", False)
        self.load_vlm()
        self.grid_flag = False
        dataset_type_for_logic = config.get("dataset_type_key", self.dataset_type)
        self.dataset_type_key, self.categories, self.category_source = resolve_dataset_categories(dataset_type_for_logic)
        print(
            f"[visualizer] dataset_type={self.dataset_type} (canonical={self.dataset_type_key}), "
            f"categories={len(self.categories)}, source={self.category_source}"
        )
        self.floor_mask = config.get("filtering_floor", False)
        self._gt_obs_loaded = False
        self._gt_rgb_loaded = False
        self._gt_grid_loaded = False

    def _resolve_obstacles_path(self):
        base_obstacles_path = os.path.join(self.target_dir, "01buildFeatMap", f"obstacles_{self.version}.npy")
        if str(self.vlm).lower() in ("lseg", "seem"):
            return base_obstacles_path, 0
        if not self.using_postprocessed_obstacles:
            return base_obstacles_path, 0

        sem_obstacles_path = os.path.join(
            self.target_dir, "02buildCatMap", f"semantic_obstacles_{self.version}.npy"
        )
        if os.path.exists(sem_obstacles_path):
            return sem_obstacles_path, 1

        print(
            f"[visualizer] semantic obstacles not found: {sem_obstacles_path}. "
            f"Fallback to base obstacles: {base_obstacles_path}"
        )
        return base_obstacles_path, 0

    def _load_gt_obstacles_and_bbox(self) -> bool:
        if self._gt_obs_loaded:
            return True
        gt_obstacles_path = os.path.join(self.gt_dir, f"obstacles_{self.gt_version}.npy")
        if not os.path.exists(gt_obstacles_path):
            print(f"[visualizer] GT obstacles not found: {gt_obstacles_path}")
            return False
        self.gt_obstacles = load_map(gt_obstacles_path)
        self.gt_bbox = get_map_bbox(self.gt_obstacles)
        self._gt_obs_loaded = True
        return True

    def _load_gt_rgb(self) -> bool:
        if self._gt_rgb_loaded:
            return True
        if not self._load_gt_obstacles_and_bbox():
            return False
        gt_rgb_path = os.path.join(self.gt_dir, f"color_top_down_{self.gt_version}.npy")
        if not os.path.exists(gt_rgb_path):
            print(f"[visualizer] GT rgb not found: {gt_rgb_path}")
            return False
        self.gt_rgb = load_map(gt_rgb_path)
        self._gt_rgb_loaded = True
        return True

    def _load_gt_grid(self) -> bool:
        if self._gt_grid_loaded:
            return True
        if not self._load_gt_obstacles_and_bbox():
            return False
        gt_grid_path = os.path.join(self.gt_dir, f"grid_{self.gt_version}.npy")
        if not os.path.exists(gt_grid_path):
            print(f"[visualizer] GT grid not found: {gt_grid_path}")
            return False
        self.gt_grid = load_map(gt_grid_path)
        self._gt_grid_loaded = True
        return True

    def _build_grid_valid_mask(self):
        """
        Build a robust valid-cell mask for grid-wise maps.
        Priority:
        1) weight > 0 (same rule used during map accumulation)
        2) fallback to non-zero feature norm when weight is unavailable
        Then intersect with obstacle free-space mask when available.
        """
        valid_mask = None
        if hasattr(self, "weight") and self.weight is not None:
            valid_mask = self.weight > 0
        elif self.grid.ndim == 3:
            valid_mask = np.linalg.norm(self.grid, axis=-1) > 1e-6

        if (
            valid_mask is not None
            and hasattr(self, "obstacles")
            and self.obstacles is not None
            and hasattr(self, "obstacle_index")
            and self.obstacles.shape == valid_mask.shape
        ):
            valid_mask = np.logical_and(valid_mask, self.obstacles == self.obstacle_index)
        return valid_mask

    def visualize(self):
        obstacles_path, obstacle_index = self._resolve_obstacles_path()
        self.obstacle_index = obstacle_index
        rgb_path = os.path.join(self.target_dir,"01buildFeatMap", f"color_top_down_{self.version}.npy")
        
        self.obstacles = load_map(obstacles_path)
        self.bbox = get_map_bbox(self.obstacles, obstacle_index=obstacle_index)
        print(f"bbox: {self.bbox}")
        self.rgb = load_map(rgb_path)

        for target in self.target_map:
            if target == "gt":
                continue
            if target == "obstacles":
                self.visualize_obstacles()
                continue
            if target == "rgb":
                self.visualize_rgb()
                continue
            if target in ("2Dmap", "2DMap"):
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
        visualize_obstacles(
            self.obstacles,
            bbox=self.bbox,
            is_show=self.bool_show,
            save_path=self.output_path if self.bool_save else None,
            save_name=f"{self.vlm}_obstacles" if self.bool_save else None,
        )
        if "gt" in self.target_map:
            if self._load_gt_obstacles_and_bbox():
                visualize_obstacles(
                    self.gt_obstacles,
                    bbox=self.gt_bbox,
                    is_show=self.bool_show,
                    save_path=self.output_path if self.bool_save else None,
                    save_name=f"gt_{self.gt_version}_obstacles" if self.bool_save else None,
                )

    def visualize_rgb(self):
        visualize_rgb(
            self.rgb,
            bbox=self.bbox,
            is_show=self.bool_show,
            save_path=self.output_path if self.bool_save else None,
            save_name=f"{self.vlm}_rgb" if self.bool_save else None,
        )
        if "gt" in self.target_map:
            if self._load_gt_rgb():
                visualize_rgb(
                    self.gt_rgb,
                    bbox=self.gt_bbox,
                    is_show=self.bool_show,
                    save_path=self.output_path if self.bool_save else None,
                    save_name=f"gt_{self.gt_version}_rgb" if self.bool_save else None,
                )

    def visualize_2Dmap(self):
        self.get_grid()
        if self.vlm == "ours":
            visualize_2Dmap_categorized(
                self.vlm,
                self.categories,
                (self.grid, self.instance_dict),
                self.bbox,
                is_show=self.bool_show,
                use_avg_height=self.bool_use_avg_height,
                save_path=self.output_path if self.bool_save else None,
                save_name=f"{self.vlm}_2Dmap_categorized" if self.bool_save else None,
                floor_mask=self.floor_mask,
            )
        else:
            valid_mask = self._build_grid_valid_mask()
            visualize_2Dmap_cat(
                self.vlm,
                self.model,
                self.categories,
                self.grid,
                use_avg_height=self.bool_use_avg_height,
                bbox=self.bbox,
                is_show=self.bool_show,
                save_path=self.output_path if self.bool_save else None,
                save_name=f"{self.vlm}_2Dmap" if self.bool_save else None,
                floor_mask=self.floor_mask,
                valid_mask=valid_mask,
            )
        if "gt" in self.target_map and self._load_gt_grid():
            viz_map(
                self.gt_grid,
                self.categories,
                floor_mask=True,
                title=f"gt_{self.gt_version}_2Dmap" if self.bool_save else None,
                is_show=self.bool_show,
                save_path=self.output_path if self.bool_save else None,
                save_name=f"gt_{self.gt_version}_2Dmap" if self.bool_save else None,
                bbox=self.gt_bbox,
            )

    def visualize_map(self):
        self.get_grid()
        visualize_map(
            (self.grid, self.instance_dict),
            bbox=self.bbox,
            is_show=self.bool_show,
            save_path=self.output_path if self.bool_save else None,
            save_name=f"{self.vlm}_map" if self.bool_save else None,
        )

    def visualize_instance(self):
        self.get_grid()
        instances = self.config.get("instances_to_visualize", [])
        if not instances:
            instances = self.instance_dict.keys() 
        visualize_instances(
            self.rgb,
            instances,
            (self.grid, self.instance_dict),
            bbox=self.bbox,
            is_show=self.bool_show,
            save_path=self.output_path if self.bool_save else None,
            save_name=f"{self.vlm}_instances" if self.bool_save else None,
        )
    
    def visualize_instances_in_cat(self):
        self.get_grid()
        categories = self.config.get("categories_to_visualize", [])
        if not categories: categories = self.categories
        visualize_2Dmap_inst_in_cat(
            categories,
            (self.grid, self.instance_dict),
            bbox=self.bbox,
            is_show=self.bool_show,
            save_path=os.path.join(self.output_path, "categories") if self.bool_save else None,
            save_name=f"{self.vlm}_categories" if self.bool_save else None,
        )

    def visualize_heatmap(self):
        self.get_grid()
        target = self.config.get("heatmap_target", [])
        heatmap_type = self.config.get("heatmap_type", "simscore")
        if not target: raise ValueError("Please specify the target for heatmap visualization.")
        visualize_heatmap(
            self.vlm,
            self.model,
            target,
            self.rgb,
            (self.grid, self.instance_dict),
            heatmap_type=heatmap_type,
            bbox=self.bbox,
            is_show=self.bool_show,
            save_path=self.output_path if self.bool_save else None,
            save_name=f"{self.vlm}_heatmap_{target}" if self.bool_save else None,
            categories=self.categories,
        )

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
            weight_path = os.path.join(self.target_dir, "01buildFeatMap", f"weight_{self.version}.npy")
            if os.path.exists(weight_path):
                self.weight = load_map(weight_path)
            else:
                self.weight = None
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
