import numpy as np
import torch
import os
import pickle
import clip

from map.utils.mapping_utils import load_map
from map.utils.clip_utils import get_text_feats
from map.seem.base_model import build_vl_model
from .utils import idxMap
from application.evaluation.metrics.metrics_hovsg import pixel_accuracy, mean_accuracy, per_class_iou, mean_iou, frequency_weighted_iou


class SegmentationMetric():
    def __init__(self,idx_map:idxMap, gt, categories, ignore_list=[]):
        self.idx_map = idx_map
        self.categories = categories
        self.num_classes = len(categories)
        self.ignore_list = ignore_list
        # self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
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
        print(pred.shape, target.shape)
        if pred.shape != target.shape:
            target = target[:pred.shape[0], :pred.shape[1]]
        assert pred.shape == target.shape

        self.im_pred = pred.copy()
        self.im_lab = target.copy()
        # print(np.unique(self.im_lab))
        self.pacc = self.calculate_pacc()
        self.mpacc, self.class_pacc_list = self.calculate_mpacc()
        self.miou, self.fwmiou = self.calculate_miou_fwmiou()
        hovsg_results = self.calculate_hovsg()
        return self.pacc, self.mpacc, self.miou, self.fwmiou, hovsg_results


    def calculate_pacc(self):
        im_pred = np.asarray(self.im_pred)
        im_lab = np.asarray(self.im_lab)
        mask = np.isin(im_lab, self.ignore_list, invert=True) 
        pixel_labeled = np.sum(mask)
        # pixel_labeled = np.sum(im_lab >= 0)
        pixel_correct = np.sum((im_lab == im_pred) & mask)
        pacc = pixel_correct / pixel_labeled if pixel_labeled !=0 else 0
        return pacc


    
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
    
    def calculate_miou_fwmiou(self):
        area_inter, area_union, area_size = self.intersection_union_cal(self.im_pred, self.im_lab, self.num_classes)
        # area_size = np.bincount(self.im_lab.flatten(), minlength=self.num_classes)

        # 영역 크기가 0인 클래스는 계산에서 제외
        valid_mask = area_size > 0  # 유효한 클래스만 필터링
        for cls in self.ignore_list:
            valid_mask[cls] = False  # 무시 리스트에 포함된 클래스도 제외
            area_size[cls] = 0      # area_size도 0으로 설정

        # class_iou_list = np.zeros(self.num_classes, dtype=np.float32)  # 초기화
        class_iou_list = area_inter[valid_mask] / area_union[valid_mask]  # 유효한 클래스에 대해서만 계산

        try:
            miou = np.mean(class_iou_list)
        except:
            raise ValueError("NaN must not be in class_iou_list, but it is.")
        total_size = np.sum(area_size[valid_mask])
        freq = area_size[valid_mask] / total_size
        try:
            fwmiou = np.sum(class_iou_list * freq)  # FWmIoU 계산
        except:
            raise ValueError("NaN must not be in class_iou_list, but it is.")
        # print(class_iou_list[valid_mask])
        # miou = np.nanmean(class_iou_list[valid_mask])  # 유효한 클래스만 포함하여 mIoU 계산
        # fwmiou = np.sum(class_iou_list[valid_mask] * area_size[valid_mask] / np.sum(area_size[valid_mask]))  # FWmIoU 계산

        return miou, fwmiou #! metrics.py line 143에서 output받는 내용도 수정해줘야됨 (원래 4개 반환했는데 2개만 받는 것으로)

    @staticmethod
    def intersection_union_cal(im_pred, im_lab, num_class):
        # print(im_pred, im_lab)
        im_pred = np.asarray(im_pred)
        im_lab = np.asarray(im_lab)
        # Remove classes from unlabeled pixels in gt image. 
        # im_pred = im_pred * (im_lab > 0)
        # Compute area intersection:
        intersection = np.where(im_pred == im_lab, im_pred, -1)
        # print(intersection)
        area_inter, _ = np.histogram(intersection, bins=num_class,
                                            range=(0,num_class))
        # Compute area union: 
        area_pred = np.bincount(im_pred.flatten(), minlength=num_class)
        area_lab = np.bincount(im_lab.flatten(), minlength=num_class)
        area_union = area_pred + area_lab - area_inter
        return area_inter, area_union, area_lab
    
    def calculate_hovsg(self):
        pa = pixel_accuracy(self.im_pred, self.im_lab, self.ignore_list)
        mpa = mean_accuracy(self.im_pred, self.im_lab, self.ignore_list)
        iou = per_class_iou(self.im_pred, self.im_lab, self.ignore_list)
        miou = mean_iou(self.im_pred, self.im_lab, self.ignore_list)
        fwmiou = frequency_weighted_iou(self.im_pred, self.im_lab, self.ignore_list)
        return(pa, mpa, iou, miou, fwmiou)