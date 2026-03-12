import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import tqdm
import torch
import clip

from map.utils.dataset_categories import normalize_dataset_type, resolve_dataset_categories
from map.utils.mapping_utils import load_map
from map.seem.base_model import build_vl_model
from map.utils.mapping_utils import get_new_mask_pallete, get_new_pallete
from .utils import gt_idx_change, idxMap
from .metrics import SegmentationMetric

CLIP_FEAT_DIM_DICT = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}

class evaluation():
    def __init__(self, config, output_path = None):
        self.output_path = output_path
        self.config = config
        self.bool_save = config["bool_save"]
        self.data_dir = os.path.join(config["data_path"], config["data_type"], config["dataset_type"])
        self.scenes = config["scene_ids"]
        self.vlm = config["vlm"]
        self.version = config["version"]
        self.gt_version = config["gt_version"]
        self.bool_visualize = config["visualize"]
        dataset_type_for_logic = config.get("dataset_type_key", config["dataset_type"])
        self.dataset_type_key = normalize_dataset_type(dataset_type_for_logic)
        self.load_vlm()
        self.load_cat()
        self._setup_ignore_index()
        self.set_path()

    def _setup_ignore_index(self):
        if self.dataset_type_key == "mp3d":
            base_ignore = [0, 2, 17, 39, 40]
        elif self.dataset_type_key in ["replica", "hm3dsem"]:
            base_ignore = [0, 40]
        else:
            raise ValueError(f"dataset_type {self.config['dataset_type']} not supported")

        self.unknown_index = len(self.categories) - 1
        ignore_values = set(base_ignore + [self.unknown_index])
        self.ignore_index = sorted([idx for idx in ignore_values if 0 <= idx < len(self.categories)])

    @staticmethod
    def _bbox_from_value(obstacles, value):
        x_indices, y_indices = np.where(obstacles == value)
        if x_indices.size == 0:
            return None
        return int(np.min(x_indices)), int(np.max(x_indices)), int(np.min(y_indices)), int(np.max(y_indices))

    def _compute_crop_bounds(self, obstacles, scene_id):
        bbox_one = self._bbox_from_value(obstacles, 1)
        bbox_zero = self._bbox_from_value(obstacles, 0)

        if bbox_one is None and bbox_zero is None:
            h, w = obstacles.shape[:2]
            print(f"[evaluation][{scene_id}] warning: obstacles map has no 0/1 values; using full map bounds")
            return 0, h - 1, 0, w - 1
        if bbox_one is None:
            return bbox_zero
        if bbox_one[0] == 0 and bbox_one[2] == 0 and bbox_zero is not None:
            return bbox_zero
        return bbox_one

    def _sanitize_gt_labels(self, gt, scene_id):
        gt = np.asarray(gt).astype(np.int32)
        invalid_mask = (gt < 0) | (gt >= len(self.categories))
        invalid_count = int(np.sum(invalid_mask))
        if invalid_count > 0:
            print(
                f"[evaluation][{scene_id}] remapped {invalid_count} invalid GT labels "
                f"to unknown index {self.unknown_index}"
            )
            gt = gt.copy()
            gt[invalid_mask] = self.unknown_index
        return gt

    def set_path(self):
        self.gt_paths = []
        self.obstacle_paths = []
        self.color_paths = []
        self.grid_paths = []
        if self.vlm == "ours": self.embedding_paths = []
        for scene_id in self.scenes:
            self.gt_paths.append(os.path.join(self.data_dir, scene_id, "map", f"{scene_id}_{self.gt_version}", "01buildFeatMap", f"grid_{self.gt_version}.npy"))
            self.obstacle_paths.append(os.path.join(self.data_dir, scene_id, "map", f"{scene_id}_{self.gt_version}", "01buildFeatMap", f"obstacles_{self.gt_version}.npy"))
            self.color_paths.append(os.path.join(self.data_dir, scene_id, "map", f"{scene_id}_{self.version}", "01buildFeatMap", f"color_top_down_{self.version}.npy"))
            if self.vlm == "floodfill" or self.vlm == "dbscan":
                self.grid_paths.append(os.path.join(self.data_dir, scene_id, "map", f"{scene_id}_{self.version}", "01buildFeatMap", f"{self.vlm}_{self.version}.npy"))
            else:
                self.grid_paths.append(os.path.join(self.data_dir, scene_id, "map", f"{scene_id}_{self.version}", "01buildFeatMap", f"grid_{self.version}.npy"))
        print(f"Loaded paths for {len(self.scenes)} scenes")

    def load_cat(self):
        dataset_type_for_logic = self.config.get("dataset_type_key", self.config["dataset_type"])
        self.dataset_type_key, self.categories, self.category_source = resolve_dataset_categories(dataset_type_for_logic)
        print(
            f"[evaluation] dataset_type={self.config['dataset_type']} (canonical={self.dataset_type_key}) "
            f"categories={len(self.categories)} source={self.category_source}"
        )

    def evaluate(self):
        result_data = []
        pbar = tqdm.tqdm(total=len(self.scenes))
        for scene_id, gt_path, obstacle_path, color_path, grid_path in zip(self.scenes, self.gt_paths, self.obstacle_paths, self.color_paths, self.grid_paths):
            gt = load_map(gt_path)
            # gt[gt==-1]=40  #^ here
            obstacles = load_map(obstacle_path)
            color = load_map(color_path)
            # if self.vlm == "ours":
            print(f"Size of gt: {gt.shape} \nSize of obstacles: {obstacles.shape}\nSize of color: {color.shape}\nSize of grid: {load_map(grid_path).shape}")
            xmin, xmax, ymin, ymax = self._compute_crop_bounds(obstacles, scene_id)
            print(xmin, ymin, xmax, ymax)
            if self.bool_visualize:
                obstacles_pil = Image.fromarray(obstacles[xmin:xmax+1, ymin:ymax+1])
                plt.figure(figsize=(8,6), dpi=120)
                plt.imshow(obstacles_pil, cmap="gray")
                plt.show(block=False)
                color_pil = Image.fromarray(color[xmin:xmax+1, ymin:ymax+1])
                plt.figure(figsize=(8,6), dpi=120)
                plt.imshow(color_pil, cmap="gray")
                plt.show(block=False)
            gt = gt[xmin:xmax+1, ymin:ymax+1]
            gt = self._sanitize_gt_labels(gt, scene_id)
            grid = load_map(grid_path)
            if self.vlm != "floodfill" and self.vlm != "dbscan":
                grid = grid[xmin:xmax+1, ymin:ymax+1]
            
            index_map = idxMap(self.vlm, self.model, self.categories, grid, self.version, grid_path)

            if self.bool_visualize:
                self.visualize_rgb(index_map, gt, gt_path)
            segmet = SegmentationMetric(index_map, gt, self.categories, ignore_list=self.ignore_index)
            if self.vlm != "floodfill" and self.vlm != "dbscan":
                top_k_auc, top_k_auc_mpacc, top_k_auc_fwmpacc, top_k_acc, top_k_mpacc, top_k_fwmpacc, k_spec_normalized, k_spec = segmet.cal_auc()
                pacc, mpacc, miou, fwmiou, hovsg_results = segmet.cal_ori()
                    
                result_data.append({
                    "scene_id": scene_id,
                    "pacc": float(pacc),
                    "mpacc": float(mpacc),
                    "miou": float(miou),
                    "fwmiou": float(fwmiou),
                    "top_k_auc": float(top_k_auc),
                    "top_k_auc_mpacc": float(top_k_auc_mpacc),
                    "top_k_auc_fwmpacc": float(top_k_auc_fwmpacc),
                    "hovsg_pacc": float(hovsg_results[0]),
                    "hovsg_mpacc": float(hovsg_results[1]),
                    "hovsg_miou": float(hovsg_results[3]),
                    "hovsg_fwmiou": float(hovsg_results[4])
                })
                if self.vlm == "ours":
                    result_data[-1]["num_embeddings"] = len(index_map.embeddings.keys())
                    print(f"# of embeddings: {len(index_map.embeddings.keys())}")
                if not self.bool_save:
                    print(scene_id, pacc, mpacc, miou, fwmiou, top_k_auc, top_k_auc_mpacc, top_k_auc_fwmpacc, sep="//////")
                    print(hovsg_results[0], hovsg_results[1], hovsg_results[3], hovsg_results[4], sep="//////")
            else:
                pacc, mpacc, miou, fwmiou, hovsg_results = segmet.cal_ori()
                result_data.append({
                    "scene_id": scene_id,
                    "pacc": float(pacc),
                    "mpacc": float(mpacc),
                    "miou": float(miou),
                    "fwmiou": float(fwmiou)})
                result_data[-1]["num_embeddings"] = len(index_map.embeddings.keys())
                if not self.bool_save:
                    print(scene_id, pacc, mpacc, miou, fwmiou, sep="//////")
                    print(hovsg_results[0], hovsg_results[1], hovsg_results[3], hovsg_results[4], sep="//////")
                    print(f"# of embeddings: {len(index_map.embeddings.keys())}")
                
            pbar.update(1)
        return result_data



    def visualize_rgb(self, index_map, gt, gt_path):
        new_pallete = get_new_pallete(len(self.categories))
        mask, patches = get_new_mask_pallete(index_map.idx_map, new_pallete, out_label_flag=True, labels = self.categories)
        seg = mask.convert("RGBA")
        seg = np.array(seg)
        seg = Image.fromarray(seg)
        plt.figure(figsize=(10,6), dpi=120)
        plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
        plt.axis('off')
        plt.title(f"{self.vlm} map")
        plt.imshow(seg)
        plt.savefig(os.path.join('/'.join(gt_path.split('/')[:-2]), f"{self.vlm}_map.png"), bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)
        mask, patches = get_new_mask_pallete(gt, new_pallete, out_label_flag=True, labels = self.categories)
        seg = mask.convert("RGBA")
        seg = np.array(seg)
        seg = Image.fromarray(seg)
        plt.figure(figsize=(10,6), dpi=120)
        plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
        plt.axis('off')
        plt.title("gt map")
        plt.imshow(seg)
        plt.savefig(os.path.join('/'.join(gt_path.split('/')[:-2]), "gt_map.png"), bbox_inches='tight', pad_inches=0.1)
        plt.show(block=True)

    def load_vlm(self):
        if self.vlm == "lseg":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_version = "ViT-B/32"
            self.clip_feat_dim = CLIP_FEAT_DIM_DICT[clip_version]
            self.model, preprocess = clip.load(clip_version)  # clip.available_models()
            self.model.to(device).eval()

        elif self.vlm in ["seem", "ours", "floodfill", "dbscan"]:
            self.model = build_vl_model("seem", input_size = 360)
