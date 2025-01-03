import numpy as np
from lseg.modules.models.lseg_net import LSegEncNet
import torch
import os
import pickle
from utils.mapping_utils import load_map
import clip

from utils.lseg_utils import get_lseg_feats
from utils.clip_utils import get_text_feats
from utils.get_transform import get_transform
from seem.utils.get_feat import get_SEEM_feat
from seem.base_model import build_vl_model



CLIP_FEAT_DIM_DICT = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}


def gt_idx_change(gt, ori_categories, new_categories):
    new_gt = np.zeros(gt.shape)
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            try:
                cat = ori_categories[gt[i,j]]
            except:
                print(gt[i,j])
                raise Exception("Error")
            new_gt[i,j] = new_categories.index(cat)
    return new_gt



class idxMap():
    def __init__(self, type, categories, xymaxmin, scene_id, version, path):
        xmin, xmax, ymin, ymax = xymaxmin
        self.type = type
        self.categories = categories
        self.num_classes = len(categories)
        self.sorted_idx_dict = {}
        if self.type == "lseg":
            self.grid_path = os.path.join(path,"map",f"{scene_id}_{version}",f"grid_{version}.npy")
            self.grid = load_map(self.grid_path)
            self.grid = self.grid[xmin:xmax+1, ymin:ymax+1]
            self.idx_map = np.zeros(self.grid.shape[:2])
            self.lsegmap()

        elif self.type == "seem":
            self.grid_path = os.path.join(path,"map",f"{scene_id}_{version}",f"grid_{version}.npy")
            self.grid = load_map(self.grid_path)
            self.grid = self.grid[xmin:xmax+1, ymin:ymax+1]
            self.idx_map = np.zeros(self.grid.shape[:2])
            self.seemmap()
        else:
            self.grid_path = os.path.join(path,"map",f"{scene_id}_{version}",f"grid_{version}.npy")
            self.grid1 = load_map(self.grid_path)
            self.grid1 = self.grid1[xmin:xmax+1, ymin:ymax+1]
            self.embeddings_path = os.path.join(path,"map",f"{scene_id}_{version}",f"instance_dict_{version}.pkl")
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            self.idx_map = np.zeros(self.grid1.shape)
            self.ourmap()
    
    def lsegmap(self):
        self.setup_vlm("lseg")
        text_feats = get_text_feats(self.categories, self.model, self.clip_feat_dim)
        map_feats = self.grid.reshape(-1, self.grid.shape[-1])
        self.matching_cos = map_feats @ text_feats.T
        height, width = self.grid.shape[:2]
        for i in range(height):
            for j in range(width):
                val = self.matching_cos[i*width+j]
                val2 = np.argsort(val)[::-1]
                if np.max(val) == 0: #해당 grid 지점에 아무것도 할당 안돼 0벡터 갖는 경우(맵 외부)에 대한 처리
                    swit = np.where(val2 == 0)[0][0]
                    val2[swit] = val2[0]
                    val2[0]=0
                best_val = val2[0]
                self.sorted_idx_dict[(i,j)] = val2.tolist()
                self.idx_map[i,j] = best_val
        self.idx_map = self.idx_map.astype(np.int32)
    
    def seemmap(self):
        self.setup_vlm("seem")
        print(self.categories)
        text_feats = self.model.encode_prompt(self.categories, task = "default")
        text_feats = text_feats.cpu().numpy()
        map_feats = self.grid.reshape(-1, self.grid.shape[-1])
        self.matching_cos = map_feats @ text_feats.T
        height, width = self.grid.shape[:2]
        for i in range(height):
            for j in range(width):
                val = self.matching_cos[i*width+j]
                val2 = np.argsort(val)[::-1]
                if np.max(val) == 0: #해당 grid 지점에 아무것도 할당 안돼 0벡터 갖는 경우(맵 외부)에 대한 처리
                    swit = np.where(val2 == 0)[0][0]
                    val2[swit] = val2[0]
                    val2[0]=0
                best_val = val2[0]
                # print(val)
                self.sorted_idx_dict[(i,j)] = val2.tolist()
                self.idx_map[i,j] = best_val
        self.idx_map = self.idx_map.astype(np.int32)

    def ourmap(self):
        self.setup_vlm("seem")
        text_feats = self.model.encode_prompt(self.categories, task = "default")
        text_feats = text_feats.cpu().numpy()
        instance_feat = []
        self.embeddings[1]["avg_height"] = 2
        self.embeddings[2]["avg_height"] = 1.5
        for id, val in self.embeddings.items():
            instance_feat.append(val["embedding"])
        instance_feat = np.array(instance_feat)
        self.matching_cos = instance_feat @ text_feats.T
        for id in self.embeddings.keys():
            cos_list = self.matching_cos[list(self.embeddings.keys()).index(id)]
            cos_list2 = np.argsort(cos_list)[::-1]
            if np.max(cos_list) == 0: # ours는 instance마다 진행하니 아무것도 할당 안돼 0벡터 갖는 경우가 없어 이 처리 과정이 불필요하긴함
                swit = np.where(cos_list2 == 0)[0][0]
                cos_list2[swit] = cos_list2[0]
                cos_list2[0]=0
            self.sorted_idx_dict[id] = cos_list2.tolist()

        self.topdown_instance_map = np.zeros(self.grid1.shape[:2])
        for i in range(self.grid1.shape[0]):
            for j in range(self.grid1.shape[1]):
                if len(self.grid1[i,j].keys()) == 0 : continue
                if len(self.grid1[i,j].keys()) == 1:
                    for key in self.grid1[i,j].keys():
                        self.topdown_instance_map[i,j] = key
                        self.idx_map[i,j] = self.sorted_idx_dict[key][0]
                else:
                    max_height = 50000
                    for key, val in self.grid1[i,j].items():
                        # if key == 2: continue
                        # candidate_height = self.embeddings[key]["avg_height"] #^ using instance aveerage
                        candidate_height = self.grid1[i,j][key][1] #^ using pixel level height value
                        if max_height > candidate_height:
                            max_height = candidate_height
                            candidate_val = key
                    self.topdown_instance_map[i,j] = candidate_val
                    self.idx_map[i,j] = self.sorted_idx_dict[candidate_val][0]
        self.topdown_instance_map = self.topdown_instance_map.astype(np.int32)
        self.idx_map = self.idx_map.astype(np.int32)

    def setup_vlm(self,type):
        if type == "lseg":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_version = "ViT-B/32"
            self.clip_feat_dim = CLIP_FEAT_DIM_DICT[clip_version]
            self.model, preprocess = clip.load(clip_version)  # clip.available_models()
            self.model.to(device).eval()
        elif type == "seem":
            self.model = build_vl_model("seem", input_size = 360)


class SegmentationMetric():
    def __init__(self,idx_map:idxMap, gt, categories, ignore_list=[]):
        self.idx_map = idx_map
        self.categories = categories
        self.num_classes = len(categories)
        self.ignore_list = ignore_list
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.gt = gt
        self.map_size = gt.shape[0]*gt.shape[1]
        self.num_gt_classes = len(np.unique(gt))

    def cal_auc(self, sampling_rate = 1):
        target = self.gt
        if self.idx_map.type != "lseg" and self.idx_map.type != "seem":
            pred = self.idx_map.topdown_instance_map.copy()
            matching_cos = self.idx_map.sorted_idx_dict.copy()
            assert pred.shape == target.shape
            auc_map = np.full(target.shape, -1)
            k_spec = []
            top_k_acc = []
            top_k_mpacc = []
            top_k_fwmpacc = []
            for i in range(target.shape[0]):
                for j in range(target.shape[1]):
                    if target[i][j] in self.ignore_list:
                        continue
                    id = pred[i][j]
                    if id == 0 and target[i][j] == 0:
                        auc_map[i][j] = 1
                        continue
                    elif id == 0 and target[i][j] != 0:
                        auc_map[i][j] = self.num_classes-1 # 이렇게 아무 인스턴스 할당 안된 부분은 무조건 최하위로 점수가 매겨지니 hovsg에서는 비고려대상이었던 부분
                        continue
                    position = matching_cos[id].index(target[i][j])+1
                    auc_map[i][j] = position
            labeled = np.sum(np.isin(target,self.ignore_list,invert=True))
            for i in range(0,self.num_classes, sampling_rate):
                correct = np.sum((auc_map <=i)&(auc_map>0))
                # labeled = np.sum(auc_map > 0)
                acc = correct / labeled if labeled != 0 else 0
                top_k_acc.append(acc)
                k_spec.append(i)

                mpacc = 0
                fwmpacc = 0
                for id in np.unique(target):
                    if id in self.ignore_list:
                        continue
                    num_id = np.sum(target == id)
                    if num_id == 0:
                        raise ValueError("Wrong id") # Because only id in gt map is considered
                    correct_id = np.sum((auc_map <= i)&(auc_map>0)&(target == id))
                    mpacc+= correct_id / num_id
                    fwmpacc += correct_id
                mpacc = mpacc / self.num_gt_classes if self.num_gt_classes != 0 else 0
                fwmpacc = fwmpacc / labeled if labeled != 0 else 0
                top_k_mpacc.append(mpacc)
                top_k_fwmpacc.append(fwmpacc)


            k_spec_normalized = [k/self.num_classes for k in k_spec]
            top_k_auc_pacc = np.trapz(top_k_acc, k_spec_normalized)
            top_k_auc_mpacc = np.trapz(top_k_mpacc, k_spec_normalized)
            top_k_auc_fwmpacc = np.trapz(top_k_fwmpacc, k_spec_normalized)
            return top_k_auc_pacc, top_k_auc_mpacc, top_k_auc_fwmpacc, top_k_acc, top_k_mpacc, top_k_fwmpacc, k_spec_normalized, k_spec
        else:
            pred = self.idx_map.idx_map.copy()
            matching_cos = self.idx_map.sorted_idx_dict.copy()
            assert pred.shape == target.shape
            auc_map = np.full(target.shape, -1)
            k_spec = []
            top_k_acc = []
            top_k_mpacc = []
            top_k_fwmpacc = []
            for i in range(target.shape[0]):
                for j in range(target.shape[1]):
                    if target[i][j] in self.ignore_list:
                        continue
                    if matching_cos[(i,j)][0] == 0 and target[i][j] == 0:
                        auc_map[i][j] = 1
                        continue
                    elif matching_cos[(i,j)][0] == 0 and target[i][j] != 0:
                        auc_map[i][j] = self.num_classes-1
                        continue
                    position = matching_cos[(i,j)].index(target[i][j])+1
                    auc_map[i][j] = position
            labeled = np.sum(np.isin(target,self.ignore_list,invert=True))
            for i in range(0,self.num_classes, sampling_rate):
                correct = np.sum((auc_map <=i)&(auc_map>=0))
                # labeled = np.sum(auc_map >= 0)
                acc = correct / labeled if labeled != 0 else 0
                top_k_acc.append(acc)
                k_spec.append(i)

                mpacc = 0
                fwmpacc = 0

                for id in np.unique(target):
                    if id in self.ignore_list:
                        continue
                    num_id = np.sum(target == id)
                    if num_id == 0:
                        raise ValueError("Wrong id") # Because only id in gt map is considered
                    correct_id = np.sum((auc_map <= i)&(auc_map>0)&(target == id))
                    mpacc+= correct_id / num_id
                    fwmpacc += correct_id
                mpacc = mpacc / self.num_gt_classes if self.num_gt_classes != 0 else 0
                fwmpacc = fwmpacc / labeled if labeled != 0 else 0
                top_k_mpacc.append(mpacc)
                top_k_fwmpacc.append(fwmpacc)

            k_spec_normalized = [k/self.num_classes for k in k_spec]
            top_k_auc_pacc = np.trapz(top_k_acc, k_spec_normalized)
            top_k_auc_mpacc = np.trapz(top_k_mpacc, k_spec_normalized)
            top_k_auc_fwmpacc = np.trapz(top_k_fwmpacc, k_spec_normalized)
            return top_k_auc_pacc, top_k_auc_mpacc, top_k_auc_fwmpacc, top_k_acc, top_k_mpacc, top_k_fwmpacc, k_spec_normalized, k_spec



    def cal_ori(self):
        pred = self.idx_map.idx_map.copy()
        target = self.gt.copy()
        target = target.astype(np.int32)
        assert pred.shape == target.shape

        self.im_pred = pred.copy()
        self.im_lab = target.copy()
        self.pacc = self.calculate_pacc()
        self.mpacc, self.class_pacc_list = self.calculate_mpacc()
        self.miou, self.fwmiou, self.class_iou_list, self.area_size = self.calculate_miou_fwmiou()
        return self.pacc, self.mpacc, self.miou, self.fwmiou

    def calculate_pacc(self):
        im_pred = np.asarray(self.im_pred)
        im_lab = np.asarray(self.im_lab)
        mask = np.isin(im_lab, self.ignore_list, invert=True) 
        pixel_labeled = np.sum(mask)
        # pixel_labeled = np.sum(im_lab >= 0)
        pixel_correct = np.sum((im_lab == im_pred) & mask)
        pacc = pixel_correct / pixel_labeled if pixel_labeled !=0 else 0
        return pacc


    @staticmethod
    def pixel_correct_labeled_cal(im_pred, im_lab, num_class, ignore_list):
        im_pred = np.asarray(im_pred)
        im_lab = np.asarray(im_lab)

        # Remove classes from unlabeled pixels in gt image.
        pixel_correct = np.zeros(num_class, dtype=np.float64)
        pixel_labeled = np.zeros(num_class, dtype=np.float64)

        for cls in range(num_class):  # 클래스 1부터 num_class-1까지
            if cls in ignore_list:
                continue
            pixel_correct[cls] = np.sum((im_pred == cls) & (im_lab == cls))
            pixel_labeled[cls] = np.sum(im_lab == cls)

        return pixel_correct, pixel_labeled
    
    def calculate_mpacc(self):
        pixel_correct, pixel_labeled = self.pixel_correct_labeled_cal(self.im_pred, self.im_lab, self.num_classes, self.ignore_list)
        # 클래스별 픽셀 정확도 계산

        class_pixel_accuracy = np.divide(
            pixel_correct, pixel_labeled, 
            out=np.full_like(pixel_correct, np.nan, dtype=np.float64),  # 초기값을 np.nan으로 설정
            where=pixel_labeled > 0  # labeled이 0인 경우 np.nan 유지
        )
        # np.nanmean을 사용하여 pixel_labeled가 0인 클래스는 평균에 포함되지 않음
        mean_acc = np.nanmean(class_pixel_accuracy)
        return mean_acc, class_pixel_accuracy
    

    @staticmethod
    def intersection_union_cal(im_pred, im_lab, num_class, ignore_list):
        im_pred = np.asarray(im_pred)
        im_lab = np.asarray(im_lab)
        # Remove classes from unlabeled pixels in gt image. 
        # im_pred = im_pred * (im_lab > 0)
        # Compute area intersection:
        intersection = np.where(im_pred == im_lab, im_pred, -1)
        area_inter, _ = np.histogram(intersection, bins=num_class,
                                            range=(0,num_class))
        # Compute area union: 
        area_pred, _ = np.histogram(im_pred, bins=num_class,
                                    range=(0,num_class))
        area_lab, _ = np.histogram(im_lab, bins=num_class,
                                range=(0,num_class))
        area_union = area_pred + area_lab - area_inter
        for cls in ignore_list:
            area_inter[cls] = 0
            area_union[cls] = 0
        return area_inter, area_union


    def calculate_miou_fwmiou(self):
        area_inter, area_union = self.intersection_union_cal(self.im_pred, self.im_lab, self.num_classes, self.ignore_list)
        area_size = np.bincount(self.im_lab.flatten(), minlength=self.num_classes)

        # 영역 크기가 0인 클래스는 계산에서 제외
        valid_mask = area_union > 0  # 유효한 클래스만 필터링
        for cls in self.ignore_list:
            valid_mask[cls] = False  # 무시 리스트에 포함된 클래스도 제외
            area_size[cls] = 0      # area_size도 0으로 설정

        class_iou_list = np.zeros(self.num_classes, dtype=np.float32)  # 초기화
        class_iou_list[valid_mask] = area_inter[valid_mask] / area_union[valid_mask]  # 유효한 클래스에 대해서만 계산

        print(class_iou_list[valid_mask])
        miou = np.nanmean(class_iou_list[valid_mask])  # 유효한 클래스만 포함하여 mIoU 계산
        fwmiou = np.sum(class_iou_list[valid_mask] * area_size[valid_mask] / np.sum(area_size[valid_mask]))  # FWmIoU 계산

        return miou, fwmiou, class_iou_list, area_size