# visualize_rgb_masks_with_room_overlay.py
# ------------------------------------------------------------
# 인스턴스별 RGB 마스크 2.5D 포인트클라우드 + 룸 레이블 바닥 오버레이
# - crop_bbox: (ymin,xmin,ymax,xmax)로 인스턴스 포인트만 크롭
# - room_map이 crop된 배열이면 origin_yx=(ymin,xmin)만큼 전역 오프셋
# - 시작 뷰: 탑뷰, 배경: 흰색
# - 룸 레이어가 항상 인스턴스보다 아래로 보이도록 자동 높이 보정
# ------------------------------------------------------------

import os
import pickle
import numpy as np
import open3d as o3d
import colorsys

# ====== 사용자 조절 파라미터 ======
ROOM_PUSH_DIR = "up"   # "down"이면 인스턴스보다 아래(더 멀리), "up"이면 위(더 가까이)
ROOM_MARGIN   = 0.50     # 인스턴스와 룸 사이 여유(m). 필요시 0.02~2.0 조정
# ==============================


# -------------------- 로더 & 유틸 --------------------
def _unwrap_np_object(obj):
    if isinstance(obj, np.ndarray):
        try:
            return obj.item()
        except Exception:
            pass
    return obj

def load_instance_dict(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj = _unwrap_np_object(obj)
        if isinstance(obj, dict):
            if "instance_dict" in obj:
                cand = _unwrap_np_object(obj["instance_dict"])
                if isinstance(cand, dict):
                    return cand
            return obj
        raise ValueError("PKL에서 instance_dict 형태를 찾지 못했습니다.")
    elif ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if "instance_dict" in data:
            cand = _unwrap_np_object(data["instance_dict"])
            if isinstance(cand, dict):
                return cand
        if len(data.files) == 1:
            sole = _unwrap_np_object(data[data.files[0]])
            if isinstance(sole, dict):
                return sole
        raise ValueError("NPZ에서 instance_dict를 찾지 못했습니다.")
    else:
        raise ValueError("지원하지 않는 포맷입니다. (.pkl 또는 .npz)")

def estimate_global_grid_size(instance_dict: dict) -> int:
    max_y1 = 0
    max_x1 = 0
    for v in instance_dict.values():
        bbox = v.get("rgb_bbox") or v.get("bbox")
        if bbox is None:
            continue
        y0, x0, y1, x1 = bbox
        max_y1 = max(max_y1, int(y1))
        max_x1 = max(max_x1, int(x1))
    return int(max(max_y1, max_x1))

def load_binary_mask_and_bbox(path: str):
    """
    0/1 바이너리 배열 로드 후 '0' 영역 bbox 반환.
    반환: (ymin, xmin, ymax, xmax), (1-mask)  ← mask는 사용 안 해도 됨
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        m = np.load(path)
    elif ext == ".npz":
        data = np.load(path, allow_pickle=True)
        m = data["mask"] if "mask" in data else data[list(data.files)[0]]
    else:
        raise ValueError("지원 포맷 아님(.npy/.npz)")

    m = (m != 0).astype(np.uint8)
    ys, xs = np.where(m == 0)            # 0 기준 bbox (요청사항 반영)
    if ys.size == 0:
        raise RuntimeError("바이너리 마스크 내 0이 없습니다.")
    ymin, ymax = int(ys.min()), int(ys.max()) + 1
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    return (ymin, xmin, ymax, xmax), (1 - m)


# -------------------- 포인트클라우드 빌더 --------------------
def build_point_cloud_from_instance_dict(
    instance_dict: dict,
    cell_size: float = 0.05,
    center_origin: bool = True,
    include_bg: bool = False,
    stride: int = 1,
    height_exaggeration: float = 1.0,
    height_baseline="min",
    layer_gap: float = 0.0,
    crop_bbox: tuple = None,           # (ymin,xmin,ymax,xmax) — 인스턴스만 크롭
    half_extent_override: float = None,
    cat_dict: dict = None,
) -> o3d.geometry.PointCloud:
    gs_est = estimate_global_grid_size(instance_dict)
    half_extent = (gs_est * cell_size * 0.5) if half_extent_override is None else float(half_extent_override)

    # 높이 기준 계산
    inst_meta = []
    for inst_id, inst in instance_dict.items():
        if not include_bg and inst_id in (1, 2):
            continue
        rgb_mask = inst.get("rgb_mask")
        bbox = inst.get("rgb_bbox") or inst.get("bbox")
        if rgb_mask is None or bbox is None:
            continue
        inst_meta.append((inst_id, float(inst.get("avg_height", 0.0))))
    if not inst_meta:
        raise RuntimeError("유효한 인스턴스가 없습니다. rgb_mask/bbox 확인하세요.")

    heights = np.array([h for _, h in inst_meta], dtype=np.float32)
    if isinstance(height_baseline, (int, float)):
        base_h = float(height_baseline)
    elif str(height_baseline).lower() == "mean":
        base_h = float(np.mean(heights))
    else:
        base_h = float(np.min(heights))

    order = np.argsort(heights)
    rank_map = {inst_meta[idx][0]: int(rk) for rk, idx in enumerate(order)}

    all_pts, all_cols = [], []

    for inst_id, inst in instance_dict.items():
        if inst_id ==1 or inst_id ==2 : continue
        if cat_dict[inst_id]["category"] in ["rug", "mat", "floor", "wall", "ceiling"]: continue
        if not include_bg and inst_id in (1, 2):
            continue

        rgb_mask = inst.get("rgb_mask")
        bbox = inst.get("rgb_bbox") or inst.get("bbox")
        avg_h = float(inst.get("avg_height", 0.0))
        if rgb_mask is None or bbox is None:
            continue

        scaled_h = base_h + (avg_h - base_h) * float(height_exaggeration) + float(layer_gap) * rank_map.get(inst_id, 0)

        y0, x0, y1, x1 = bbox
        valid = np.any(rgb_mask != 0, axis=2)
        ys, xs = np.where(valid)
        if ys.size == 0:
            continue
        if stride > 1:
            idx = np.arange(ys.size)[::stride]
            ys, xs = ys[idx], xs[idx]

        gy = y0 + ys
        gx = x0 + xs

        if crop_bbox is not None:
            cy0, cx0, cy1, cx1 = crop_bbox
            keep = (gx >= cx0) & (gx < cx1) & (gy >= cy0) & (gy < cy1)
            if not np.any(keep):
                continue
            gx, gy = gx[keep], gy[keep]
            xs, ys = xs[keep], ys[keep]

        world_x = gx.astype(np.float32) * cell_size
        world_z = gy.astype(np.float32) * cell_size
        if center_origin:
            world_x -= half_extent
            world_z -= half_extent
        world_y = np.full_like(world_x, scaled_h, dtype=np.float32)

        pts = np.stack([world_x, world_y, world_z], axis=1)
        cols = rgb_mask[ys, xs, :].astype(np.float32) / 255.0

        all_pts.append(pts)
        all_cols.append(cols)

    if not all_pts:
        raise RuntimeError("유효한 포인트가 없습니다. (크롭 범위 밖이거나 모두 배경)")

    all_pts = np.concatenate(all_pts, axis=0)
    all_cols = np.concatenate(all_cols, axis=0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_pts)
    pc.colors = o3d.utility.Vector3dVector(all_cols)
    return pc

def _room_id_to_pastel(room_id: int):
    g = 0.61803398875
    h = (room_id * g) % 1.0
    s = 0.25
    v = 0.95
    r, g_, b = colorsys.hsv_to_rgb(h, s, v)
    w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    c = np.array([r, g_, b], dtype=np.float32)
    return (0.65 * w + 0.35 * c).astype(np.float32)

def build_room_floor_pointcloud(
    room_map: np.ndarray,
    cell_size: float,
    center_origin: bool,
    half_extent: float,
    room_height: float = 0.0,
    stride: int = 2,
    origin_yx: tuple = (0, 0),    # (ymin,xmin) 오프셋 → 전역 좌표계 보정
    swap_xz: bool = False,        # 필요 시 좌표 스왑
    flip_x: bool = False,         # x축 반전
    flip_z: bool = False,         # z축 반전
) -> o3d.geometry.PointCloud:
    H, W = room_map.shape[:2]
    ys, xs = np.where(room_map != 0)
    if ys.size == 0:
        return o3d.geometry.PointCloud()

    if stride > 1:
        idx = np.arange(ys.size)[::stride]
        ys, xs = ys[idx], xs[idx]

    rooms = room_map[ys, xs].astype(int)

    # 전역 오프셋 보정
    oy, ox = origin_yx
    ys_g = ys + int(oy)
    xs_g = xs + int(ox)

    # 옵션 보정
    if swap_xz:
        xs_use, ys_use = ys_g, xs_g
        H_use, W_use = W, H
    else:
        xs_use, ys_use = xs_g, ys_g
        H_use, W_use = H, W

    if flip_x:
        xs_use = (W_use - 1) - xs_use
    if flip_z:
        ys_use = (H_use - 1) - ys_use

    world_x = xs_use.astype(np.float32) * cell_size
    world_z = ys_use.astype(np.float32) * cell_size
    if center_origin:
        world_x -= half_extent
        world_z -= half_extent
    world_y = np.full_like(world_x, float(room_height), dtype=np.float32)

    pts = np.stack([world_x, world_y, world_z], axis=1)

    uniq = np.unique(rooms)
    lut = {rid: _room_id_to_pastel(int(rid)) for rid in uniq}
    cols = np.stack([lut[int(r)] for r in rooms], axis=0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(cols)
    return pc

def merge_point_clouds(pcs):
    valid = [pc for pc in pcs if pc is not None and len(pc.points) > 0]
    if not valid:
        return o3d.geometry.PointCloud()
    if len(valid) == 1:
        return valid[0]
    pts = np.concatenate([np.asarray(pc.points) for pc in valid], axis=0)
    cols = np.concatenate([np.asarray(pc.colors) for pc in valid], axis=0)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    out.colors = o3d.utility.Vector3dVector(cols)
    return out


# -------------------- 시각화 --------------------
def visualize_point_cloud(pc: o3d.geometry.PointCloud, point_size: float = 2.2):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="2.5D RGB Masks + Room Overlay", width=1280, height=800)
    vis.add_geometry(pc)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    bbox = pc.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([0.0, -1.0, 0.0])   # 탑뷰
    ctr.set_up([0.0, 0.0, -1.0])
    ctr.set_zoom(0.7)

    vis.run()
    vis.destroy_window()


# -------------------- 실행 예시 --------------------
if __name__ == "__main__":
    # 경로들
    instance_dict_path = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_rgbmask/01buildFeatMap/instance_dict_rgbmask.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/01buildFeatMap/instance_dict_test0811mh3.pkl"#"/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_rgbmask/01buildFeatMap/instance_dict_rgbmask.pkl"
    cat_dict_path = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_rgbmask/02buildCatMap/categorized_instance_dict_rgbmask.pkl"
    crop_mask_path     = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_rgbmask/01buildFeatMap/obstacles_rgbmask.npy"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/01buildFeatMap/obstacles_test0811mh3.npy"#"/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_rgbmask/01buildFeatMap/obstacles_rgbmask.npy"
    room_map_path      = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/04classificateRoom/room_map.npy"#"/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_final_test2/04classificateRoom/room_map.npy"

    # 파라미터
    cell_size = 0.05
    include_bg = False
    stride_inst = 1
    stride_room = 2
    height_exaggeration = 1.0
    height_baseline = "min"
    layer_gap = 0.01

    # 1) 데이터 로드
    instance_dict = load_instance_dict(instance_dict_path)
    cat_dict = load_instance_dict(cat_dict_path)
    crop_bbox, _ = load_binary_mask_and_bbox(crop_mask_path)   # (ymin,xmin,ymax,xmax)
    room_map = np.load(room_map_path)

    # 사용자 환경에서 정합이 맞던 회전/전치 보정(필요 없으면 주석 처리)
    # room_map = np.rot90(np.rot90(room_map).T, -1)

    # 2) 센터 정렬 기준(half_extent) 통일
    H_room, W_room = room_map.shape[:2]
    half_extent = max(H_room, W_room) * cell_size * 0.5

    # 3) 인스턴스 RGB 포인트 (크롭 적용)
    pc_inst = build_point_cloud_from_instance_dict(
        instance_dict,
        cell_size=cell_size,
        center_origin=True,
        include_bg=include_bg,
        stride=stride_inst,
        height_exaggeration=height_exaggeration,
        height_baseline=height_baseline,
        layer_gap=layer_gap,
        crop_bbox=crop_bbox,
        half_extent_override=half_extent,
        cat_dict=cat_dict
    )

    # ---- 인스턴스 Y범위를 읽어 룸 높이 자동 산정(깊이 버퍼 확실히 분리) ----
    inst_y = np.asarray(pc_inst.points)[:, 1]
    y_min, y_max = float(inst_y.min()), float(inst_y.max())
    if ROOM_PUSH_DIR.lower() == "down":
        room_height = y_min - ROOM_MARGIN   # 더 아래(멀리)
    else:
        room_height = y_max + ROOM_MARGIN   # 더 위(가깝게)

    # 4) 룸 바닥 레이어 (room_map은 이미 crop되어 있으면 origin_yx=crop_bbox[:2])
    pc_room = build_room_floor_pointcloud(
        room_map=room_map,
        cell_size=cell_size,
        center_origin=True,
        half_extent=half_extent,
        room_height=room_height,
        stride=stride_room,
        origin_yx=crop_bbox[:2],
        swap_xz=False,
        flip_x=False,
        flip_z=False
    )

    # 5) 합쳐서 렌더 (깊이 버퍼로 실제 아래/위가 결정되므로 순서는 무관)
    pc_all = merge_point_clouds([pc_room, pc_inst])
    visualize_point_cloud(pc_all, point_size=2.4)
