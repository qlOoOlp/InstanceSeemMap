import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import tqdm
import torch
import clip

from map.utils.matterport3d_categories import mp3dcat
from map.utils.replica_categories import replica_cat
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
        if config["dataset_type"] == "mp3d":
            self.ignore_index = [0,2,17,39,40,-1]
        elif config["dataset_type"] == "replica":
            self.ignore_index = [0,40]#,31,102]#[0,31,37,40,93,94,95,97,102] #[0,40,31,102]
        self.load_vlm()
        self.load_cat()
        self.set_path()


    def set_path(self):
        self.gt_paths = []
        self.obstacle_paths = []
        self.color_paths = []
        self.grid_paths = []
        if self.vlm == "ours": self.embedding_paths = []
        for scene_id in self.scenes:
            self.gt_paths.append(os.path.join(self.data_dir, scene_id, "map", f"{scene_id}_{self.gt_version}", f"grid_{self.gt_version}.npy"))
            self.obstacle_paths.append(os.path.join(self.data_dir, scene_id, "map", f"{scene_id}_{self.gt_version}", f"obstacles_{self.gt_version}.npy"))
            self.color_paths.append(os.path.join(self.data_dir, scene_id, "map", f"{scene_id}_{self.version}", f"color_top_down_{self.version}.npy"))
            if self.vlm == "floodfill" or self.vlm == "dbscan":
                self.grid_paths.append(os.path.join(self.data_dir, scene_id, "map", f"{scene_id}_{self.version}", f"{self.vlm}_{self.version}.npy"))
            else:
                self.grid_paths.append(os.path.join(self.data_dir, scene_id, "map", f"{scene_id}_{self.version}", f"grid_{self.version}.npy"))
        print(f"Loaded paths for {len(self.scenes)} scenes")

    def load_cat(self):
        if self.config["dataset_type"]=="mp3d":
            self.categories = mp3dcat
        elif self.config["dataset_type"]=="replica":
            self.categories = replica_cat#[replica_cat[i] for i in sorted(replica_cat.keys())]
        else:
            raise ValueError(f"dataset_type {self.config['dataset_type']} not supported")
        print(f"Loaded categories: {self.config['dataset_type']} - {len(self.categories)} classes")

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
            x_indices, y_indices = np.where(obstacles == 1)
            # else:
            #     x_indices, y_indices = np.where(obstacles == 0)
            xmin = np.min(x_indices)
            xmax = np.max(x_indices)
            ymin = np.min(y_indices)
            ymax = np.max(y_indices)
            if xmin == 0 and ymin ==0 :
                x_indices, y_indices = np.where(obstacles == 0)
                xmin = np.min(x_indices)
                xmax = np.max(x_indices)
                ymin = np.min(y_indices)
                ymax = np.max(y_indices)
                # if xmin == 0 and ymin == 0 :
                #     raise ValueError("No valid area in the map")
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
            # if self.config["dataset_type"] == "mp3d":
            gt[gt==-1] = len(self.categories)-1
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
                if not self.bool_save:
                    print(scene_id, pacc, mpacc, miou, fwmiou, top_k_auc, top_k_auc_mpacc, top_k_auc_fwmpacc, sep="//////")
                    print(hovsg_results[0], hovsg_results[1], hovsg_results[3], hovsg_results[4], sep="//////")
                    print(f"# of embeddings: {len(index_map.embeddings.keys())}")
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