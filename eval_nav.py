import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import cv2


iou_threshold = 0.3

scene_id = "00824-Dd4bFSTQ8gi"
pred_version = "test0811mh3"

pred_path = f"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/{scene_id}/map/{scene_id}_{pred_version}/02buildCatMap/categorized_instance_dict_{pred_version}.pkl"
pred_obs_path = f"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/{scene_id}/map/{scene_id}_{pred_version}/02buildCatMap/semantic_obstacles_{pred_version}.npy"
gt_path = f"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/val/{scene_id}/topdown_gt/instance_masks.pkl"
gt_meta_path = f"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/val/{scene_id}/topdown_gt/floors_meta.json"
hovsg_path = f"/home/hong_RCI/vlmap/HOVSG/data/hm3dsem/{scene_id}/dict/instance_masks.pkl"
hovsg_meta_path = f"/home/hong_RCI/vlmap/HOVSG/data/hm3dsem/{scene_id}/dict/floors_meta.json"


pred_result_path = f"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/{scene_id}/map/{scene_id}_{pred_version}/nav/gt_to_inst_matching.json"
hovsg_result_path = f"/home/hong_RCI/vlmap/HOVSG/data/hm3dsem/{scene_id}/nav/gt_to_inst_matching.json"



def restore_single_instance(inst_dict, floors_meta, target_oid, out_path=None):
    """
    특정 instance id의 mask를 복원하여 2D top-down map으로 저장.
    
    Args:
        masks_dir (str): instance_masks.pkl, floors_meta.json이 있는 디렉토리
        target_oid (int): 복원할 instance id (obj_id)
        out_path (str): 저장할 .npy 경로 (None이면 저장하지 않고 반환만 함)
    
    Returns:
        mask (np.ndarray): (H, W) 이진 마스크 (0=배경, 1=해당 instance)
    """
    # 파일 경로
    # entry 불러오기
    # print(inst_dict.keys())
    entry = inst_dict.get(str(target_oid), inst_dict.get(int(target_oid)))
    if entry is None:
        raise KeyError(f"Instance id {target_oid} not found in {inst_dict}")

    fid = int(entry["floor_id"])
    fmeta = floors_meta.get(str(fid), floors_meta.get(fid))
    H, W = int(fmeta["H"]), int(fmeta["W"])

    # 빈 mask 생성
    mask = np.zeros((H, W), dtype=np.uint8)

    # 마스크 복원
    if "mask_lin" in entry:
        lin = np.asarray(entry["mask_lin"], dtype=np.int64)
        rr = (lin // W).astype(np.int32)
        cc = (lin %  W).astype(np.int32)
    elif "mask_rc" in entry:
        rr = np.asarray(entry["mask_rc"]["rows"], dtype=np.int32)
        cc = np.asarray(entry["mask_rc"]["cols"], dtype=np.int32)
    else:
        raise ValueError(f"No mask found for instance {target_oid}")

    mask[rr, cc] = 1

    # 저장 옵션
    if out_path is not None:
        np.save(out_path, mask)
        print(f"[instance {target_oid}] mask saved to {out_path}")

    return mask

def restore_single_instance2(inst_dict, floors_meta, target_oid, out_path=None):
    H, W = int(floors_meta["0"]["H"]), int(floors_meta["0"]["W"])
    mask2d = np.zeros((H, W), dtype=np.uint8)

    # entry 찾기 (key가 str/int 둘 다 가능)
    entry = inst_dict.get(target_oid, inst_dict.get(str(target_oid)))
    if entry is None:
        raise KeyError(f"Instance {target_oid} not found.")

    lin = np.asarray(entry["mask"], dtype=np.int64)
    if lin.size > 0:
        rr = (lin // W).astype(np.int32)
        cc = (lin %  W).astype(np.int32)
        mask2d[rr, cc] = 1

    return mask2d

def mask_to_bbox(mask):
    """이진 마스크에서 [xmin, ymin, xmax, ymax] bbox 반환"""
    # print(np.unique(mask))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None  # 빈 마스크
    return [xs.min(), ys.min(), xs.max(), ys.max()]

def bbox_iou(bbox1, bbox2):
    """두 bbox ([xmin, ymin, xmax, ymax])의 IoU 계산"""
    if bbox1 is None or bbox2 is None:
        return 0.0

    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    inter_w = max(0, xB - xA + 1)
    inter_h = max(0, yB - yA + 1)
    inter_area = inter_w * inter_h

    area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union = area1 + area2 - inter_area

    # print(bbox1, bbox2)
    # print(inter_area / union if union > 0 else 0.0)

    return inter_area / union if union > 0 else 0.0

def check_ok(pred_bbox, gt_dict, gt_meta):
    target_id = 0
    max_iou = 0.1
    target_iou = 0
    for check_id, check_val in gt_dict.items():
        for back in ["wall", "floor", "ceiling", "unknown"]:
            if back in check_val["category"]: continue
        cand_mask = restore_single_instance(gt_dict, gt_meta, check_id)
        gt_bbox = mask_to_bbox(cand_mask)
        target_iou = bbox_iou(pred_bbox, gt_bbox)
        if target_iou > max_iou:
            max_iou = target_iou
            target_id = check_id
    return target_id, max_iou


with open(pred_path, "rb") as f:
    pred_dict = pickle.load(f)
with open(gt_path, "rb") as f:
    gt_dict = pickle.load(f)
with open(gt_meta_path, "rb") as f:
    gt_meta = json.load(f)
with open(hovsg_path, "rb") as f:
    hovsg_dict = pickle.load(f)
with open(hovsg_meta_path, "rb") as f:
    hovsg_meta = json.load(f)
with open(pred_result_path, "r") as f:
    pred_result = json.load(f)
with open(hovsg_result_path, "r") as f:
    hovsg_result = json.load(f)


pred_obs = np.load(pred_obs_path)
x_indices, y_indices = np.where(pred_obs == 1)
xmin = np.min(x_indices)
xmax = np.max(x_indices)
ymin = np.min(y_indices)
ymax = np.max(y_indices)
if xmin == 0 and ymin == 0:
    x_indices, y_indices = np.where(pred_obs == 0)
    xmin = np.min(x_indices)
    xmax = np.max(x_indices)
    ymin = np.min(y_indices)
    ymax = np.max(y_indices)


pred_return = {"object":0,"room":0,"caption":0,"mixed":0,"mixed_a":0}
hovsg_return = {"object":0,"room":0,"caption":0,"mixed":0,"mixed_a":0}
for gt_id in pred_result.keys():
    print(gt_id)
    preds = pred_result[gt_id]
    hovsgs = hovsg_result[gt_id]
    gt_id = int(gt_id)
    print(preds.keys())
    for cat in preds.keys():
        pred_iou = 0
        hovsg_iou = 0
        pred_s = False
        hovsg_s = False
        pred_id = preds[cat]
        hovsg_id = hovsgs[cat]
        gt_mask = restore_single_instance(gt_dict, gt_meta, gt_id)
        gt_mask = np.flip(gt_mask.T, axis=(0, 1))
        hovsg_mask = restore_single_instance2(hovsg_dict, hovsg_meta, hovsg_id)
        hovsg_mask = np.flip(hovsg_mask.T, axis=(0, 1))
        h, w = gt_mask.shape[:2]  # B의 resolution
        hovsg_mask_resized = cv2.resize(hovsg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # print(gt_mask.shape, hovsg_mask_resized.shape)
        # plt.imshow(gt_mask, cmap="gray")  # 흑백으로 시각화
        # plt.axis("off")
        # plt.show()
        # plt.imshow(hovsg_mask_resized, cmap="gray")  # 흑백으로 시각화
        # plt.axis("off")
        # plt.show()
        gt_bbox = mask_to_bbox(gt_mask)
        hovsg_bbox = mask_to_bbox(hovsg_mask_resized)
        hovsg_iou = bbox_iou(hovsg_bbox, gt_bbox)

        if pred_id!=-1 and pred_id != 0:
            pred_mask=pred_dict[pred_id]["mask"]

            pred_mask_crop = pred_mask[xmin:xmax+1,ymin:ymax+1]
            pred_mask_resized = cv2.resize(pred_mask_crop.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            pred_bbox = mask_to_bbox(pred_mask_resized)
            pred_iou = bbox_iou(pred_bbox, gt_bbox)
            c = np.logical_or(hovsg_mask_resized, gt_mask)


            gt = gt_mask.astype(bool)
            pred = hovsg_mask_resized.astype(bool)
            assert gt.shape == pred.shape
            # RGB 오버레이 생성
            overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
            overlay[..., 2] = gt * 255    # Blue 채널: GT
            overlay[..., 0] = pred * 255  # Red  채널: HOV-SG
            # plt.figure(figsize=(6, 6))
            # plt.imshow(overlay)
            # plt.axis("off")


            # # plt.imshow(c, cmap="gray")  # 흑백으로 시각화
            # # plt.axis("off")
            # plt.show()
            if cat in ["object","room"]:
                if pred_iou < iou_threshold:
                    match_id, match_iou = check_ok(pred_bbox, gt_dict, gt_meta)
                    if match_id != 0 :
                        if gt_dict[gt_id]["category"] == gt_dict[match_id]["category"]: 
                            if cat == "room":
                                if gt_dict[gt_id]["room_category"] == gt_dict[match_id]["room_category"]: pred_iou = match_iou
                            else: pred_iou = match_iou
                if hovsg_iou < iou_threshold:
                    match_id, match_iou = check_ok(hovsg_bbox, gt_dict, gt_meta)
                    if match_id != 0 :
                        if gt_dict[gt_id]["category"] == gt_dict[match_id]["category"]: 
                            if cat == "room":
                                if gt_dict[gt_id]["room_category"] == gt_dict[match_id]["room_category"]: hovsg_iou = match_iou
                            else: hovsg_iou = match_iou
            if pred_iou > iou_threshold: pred_s = True
            if hovsg_iou > iou_threshold: hovsg_s = True
            pred_return[cat] += int(pred_s)
            hovsg_return[cat] += int(hovsg_s)
        print(pred_iou, hovsg_iou)
print(pred_return, hovsg_return)
for cat in pred_return.keys():
    pred_return[cat] = pred_return[cat]/len(pred_result)
    hovsg_return[cat] = hovsg_return[cat]/len(hovsg_result)
print(pred_return, hovsg_return)
        # c = np.logical_or(hovsg_mask_resized, gt_mask)
        # plt.imshow(c, cmap="gray")  # 흑백으로 시각화
        # plt.axis("off")
        # plt.show()

        # c = np.logical_or(pred_mask_resized, gt_mask)
        # plt.imshow(c, cmap="gray")  # 흑백으로 시각화
        # plt.axis("off")
        # plt.show()
        # c = np.logical_or(hovsg_mask_resized, gt_mask)
        # plt.imshow(c, cmap="gray")  # 흑백으로 시각화
        # plt.axis("off")
        # plt.show()



        # print(pred_mask_crop.shape)
        # print(gt_mask.shape, gt_mask.sum(), "pixels covered")
        # print(hovsg_mask.shape, hovsg_mask.sum(), "pixels covered")
        # print(pred_mask_resized.shape)
        # print(gt_mask.shape)
