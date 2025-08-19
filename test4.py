# visualize_rgb_masks_with_room_overlay.py
# ------------------------------------------------------------
# 인스턴스별 RGB 마스크 2.5D 포인트클라우드 + 룸 레이블 바닥 오버레이
# - crop_bbox: (ymin,xmin,ymax,xmax)로 인스턴스 포인트만 크롭
# - room_map이 crop된 배열이면 origin_yx=(ymin,xmin)만큼 전역 오프셋
# - 시작 뷰: 탑뷰, 배경: 흰색
# - 룸 레이어가 항상 인스턴스보다 아래로 보이도록 자동 높이 보정
# - ★ 추가 기능 ★: ROOM_ID만 표시 + 그 방 내부 인스턴스만 표시
# ------------------------------------------------------------

import os
import pickle
import numpy as np
import open3d as o3d
import colorsys

# ====== 사용자 조절 파라미터 (기존 유지) ======
ROOM_PUSH_DIR = "down"   # "down"이면 인스턴스보다 아래(더 멀리), "up"이면 위(더 가까이)
ROOM_MARGIN   = 0.50     # 인스턴스와 룸 사이 여유(m)
# ====== 신규: 표시할 방 ID 하드코딩 ======
ROOM_ID       = 6
# ========================================


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
    ys, xs = np.where(m == 0)            # 0 기준 bbox
    if ys.size == 0:
        raise RuntimeError("바이너리 마스크 내 0이 없습니다.")
    ymin, ymax = int(ys.min()), int(ys.max()) + 1
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    return (ymin, xmin, ymax, xmax), (1 - m)


# -------------------- 포인트클라우드 빌더 (변경 없음) --------------------
def build_point_cloud_from_instance_dict(
    instance_dict: dict,
    cell_size: float = 0.05,
    center_origin: bool = True,
    include_bg: bool = False,
    stride: int = 1,
    height_exaggeration: float = 1.0,
    height_baseline="min",
    layer_gap: float = 0.0,
    crop_bbox: tuple = None,
    half_extent_override: float = None
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
        # world_z는 아래에서 공통처리
        world_y = np.full_like(world_x, scaled_h, dtype=np.float32)
        world_z = gy.astype(np.float32) * cell_size
        if center_origin:
            world_z -= half_extent

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


# -------------------- 시각화 (변경 없음) --------------------
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


# -------------------- ★ 추가: 인스턴스 PC를 ROOM_ID 내부로 필터 ★ --------------------
def _invert_room_mapping_to_local(gx, gy, room_shape, origin_yx, swap_xz, flip_x, flip_z):
    """전역 grid(gx,gy) -> room_map 로컬 좌표(xs,ys) 역변환."""
    H, W = room_shape[:2]
    oy, ox = origin_yx

    if not swap_xz:
        xs_use, ys_use = gx.copy(), gy.copy()
        H_use, W_use = H, W
    else:
        xs_use, ys_use = gy.copy(), gx.copy()
        H_use, W_use = W, H

    if flip_x:
        xs_use = (W_use - 1) - xs_use
    if flip_z:
        ys_use = (H_use - 1) - ys_use

    if not swap_xz:
        xs_g, ys_g = xs_use, ys_use
    else:
        xs_g, ys_g = ys_use, xs_use

    xs_local = xs_g - int(ox)
    ys_local = ys_g - int(oy)
    return xs_local, ys_local

def filter_pc_by_room_id(
    pc: o3d.geometry.PointCloud,
    room_map: np.ndarray,
    room_id: int,
    cell_size: float,
    center_origin: bool,
    half_extent: float,
    origin_yx: tuple,
    swap_xz: bool = False,
    flip_x: bool = False,
    flip_z: bool = False,
) -> o3d.geometry.PointCloud:
    """기존 인스턴스 PC에서 ROOM_ID 영역에 속하는 포인트만 남김."""
    if len(pc.points) == 0:
        return pc

    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors)

    # world -> grid index (gx, gy)
    if center_origin:
        gx = np.rint((pts[:, 0] + half_extent) / cell_size).astype(np.int64)
        gy = np.rint((pts[:, 2] + half_extent) / cell_size).astype(np.int64)
    else:
        gx = np.rint(pts[:, 0] / cell_size).astype(np.int64)
        gy = np.rint(pts[:, 2] / cell_size).astype(np.int64)

    rx, ry = _invert_room_mapping_to_local(
        gx, gy,
        room_shape=room_map.shape,
        origin_yx=origin_yx,
        swap_xz=swap_xz,
        flip_x=flip_x,
        flip_z=flip_z
    )

    inside = (rx >= 0) & (rx < room_map.shape[1]) & (ry >= 0) & (ry < room_map.shape[0])
    if not np.any(inside):
        return o3d.geometry.PointCloud()
    idx_inside = np.where(inside)[0]
    keep_room = room_map[ry[idx_inside], rx[idx_inside]].astype(int) == int(room_id)
    if not np.any(keep_room):
        return o3d.geometry.PointCloud()

    idx_final = idx_inside[keep_room]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[idx_final])
    out.colors = o3d.utility.Vector3dVector(cols[idx_final])
    return out


# -------------------- 실행 예시 --------------------
if __name__ == "__main__":
    # 경로들
    instance_dict_path = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_rgbmask/01buildFeatMap/instance_dict_rgbmask.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/01buildFeatMap/instance_dict_test0811mh3.pkl"#"/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_rgbmask/01buildFeatMap/instance_dict_rgbmask.pkl"
    crop_mask_path     = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_rgbmask/01buildFeatMap/obstacles_rgbmask.npy"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/01buildFeatMap/obstacles_test0811mh3.npy"#"/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_rgbmask/01buildFeatMap/obstacles_rgbmask.npy"
    room_map_path      = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/04classificateRoom/room_map.npy"#"/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_final_test2/04classificateRoom/room_map.npy"

    # 파라미터 (기존 유지)
    cell_size = 0.05
    include_bg = False
    stride_inst = 1
    stride_room = 2
    height_exaggeration = 1.0
    height_baseline = "min"
    layer_gap = 0.02

    # 1) 데이터 로드
    instance_dict = load_instance_dict(instance_dict_path)
    crop_bbox, _  = load_binary_mask_and_bbox(crop_mask_path)   # (ymin,xmin,ymax,xmax)
    room_map      = np.load(room_map_path)

    # 사용자 환경에서 정합이 맞던 회전/전치 보정(필요 없으면 주석 처리)
    room_map = np.rot90(np.rot90(room_map).T, -1)

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
        half_extent_override=half_extent
    )

    # ★ ROOM_ID 영역으로 인스턴스 포인트 필터링
    pc_inst_room = filter_pc_by_room_id(
        pc=pc_inst,
        room_map=room_map,
        room_id=ROOM_ID,
        cell_size=cell_size,
        center_origin=True,
        half_extent=half_extent,
        origin_yx=crop_bbox[:2],  # room_map이 전역에서 (ymin,xmin)만큼 오프셋되어 있다고 가정
        swap_xz=False,
        flip_x=False,
        flip_z=False
    )

    # ---- 인스턴스 Y범위를 읽어 룸 높이 자동 산정 (기존 로직 유지) ----
    if len(pc_inst_room.points) > 0:
        inst_y = np.asarray(pc_inst_room.points)[:, 1]
    else:
        inst_y = np.asarray(pc_inst.points)[:, 1]
    y_min, y_max = float(inst_y.min()), float(inst_y.max())
    if ROOM_PUSH_DIR.lower() == "up":
        room_height = y_min - ROOM_MARGIN
    else:
        room_height = y_max + ROOM_MARGIN

    # 4) 룸 바닥 레이어 — ROOM_ID만 남긴 mask로 생성 (기존 함수 사용)
    room_map_masked = np.where(room_map.astype(int) == int(ROOM_ID), room_map, 0)
    pc_room = build_room_floor_pointcloud(
        room_map=room_map_masked,
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

    # 5) 합쳐서 렌더
    pc_all = merge_point_clouds([pc_room, pc_inst_room])
    visualize_point_cloud(pc_all, point_size=2.4)
