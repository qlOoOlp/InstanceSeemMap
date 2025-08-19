# visualize_instance_masks_as_pointcloud.py
# ------------------------------------------------------------
# 인스턴스별 "mask"(이진)로 2.5D 포인트클라우드 시각화
# - 인스턴스마다 구분되는 색 자동 할당 (골든 레이쇼 기반)
# - 높이 과장/레이어 간격 옵션
# - 시작 뷰: 탑뷰, 배경: 흰색
# - ★추가★: ROOM_IDS(리스트)만 표시 + 해당 방 내부 인스턴스만 표시
# - ★추가★: crop_npy(0/1)로 bbox 크롭, room_npy(정수맵) 바닥 오버레이
# - ★추가★: ROOM_PUSH_DIR/ROOM_MARGIN으로 바닥 레이어를 항상 아래/위로 분리
# ------------------------------------------------------------
# 필요 패키지: open3d
#   pip install open3d
# ------------------------------------------------------------

import os
import pickle
import numpy as np
import colorsys
import open3d as o3d

# ====== 사용자 조절 파라미터 ======
ROOM_ID       = 5          # 단일 방 ID (ROOM_IDS가 비어있을 때 fallback)
ROOM_IDS      = [1,2,3,4,5,6,7,8]#[1,2,3,4,5,6]        # ▶▶ 리스트로 여러 방 표시
ROOM_PUSH_DIR = "down"     # "down": 인스턴스보다 아래(더 멀리), "up": 위(더 가까이)
ROOM_MARGIN   = 0.50       # 인스턴스와 룸 사이 여유(m)
# ================================

# -------------------- 로딩 도우미 --------------------

def _unwrap_np_object(obj):
    """np.ndarray(object) 래핑을 풀어 dict를 꺼냄 (가능한 경우)."""
    if isinstance(obj, np.ndarray):
        try:
            return obj.item()
        except Exception:
            pass
    return obj

def _looks_like_instance_dict(d):
    """
    인스턴스 dict 추정: 값 중 하나라도 dict이고,
    그 안에 'mask'와 'avg_height' 또는 'rgb_mask' 등이 있으면 True.
    """
    if not isinstance(d, dict):
        return False
    for v in d.values():
        if isinstance(v, dict):
            k = v.keys()
            if ("mask" in k and "avg_height" in k) or ("rgb_mask" in k and "avg_height" in k):
                return True
    return False

def load_instance_dict(path: str) -> dict:
    """
    instance_dict를 로드:
    - .pkl: pickle.load → dict 또는 ndarray(object) → dict
    - .npz: 'instance_dict' 키 또는 단일 object 항목 탐색
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj = _unwrap_np_object(obj)

        # 1) 이미 올바른 instance_dict
        if _looks_like_instance_dict(obj):
            return obj

        # 2) {'instance_dict': ...}
        if isinstance(obj, dict) and "instance_dict" in obj:
            cand = _unwrap_np_object(obj["instance_dict"])
            if _looks_like_instance_dict(cand):
                return cand

        # 3) 리스트/튜플 케이스
        if isinstance(obj, (list, tuple)) and obj:
            cand = _unwrap_np_object(obj[0])
            if _looks_like_instance_dict(cand):
                return cand

        raise ValueError("주어진 PKL은 instance_dict 구조가 아닙니다. 올바른 파일을 지정하세요.")

    elif ext == ".npz":
        data = np.load(path, allow_pickle=True)
        # 1) 명시 키
        if "instance_dict" in data:
            cand = _unwrap_np_object(data["instance_dict"])
            if _looks_like_instance_dict(cand):
                return cand
        # 2) 단일 object 항목 탐색
        if len(data.files) == 1:
            sole = _unwrap_np_object(data[data.files[0]])
            if _looks_like_instance_dict(sole):
                return sole

        raise ValueError("NPZ 내부에서 instance_dict를 찾지 못했습니다.")

    else:
        raise ValueError("지원 포맷 아님: .pkl 또는 .npz 만 지원")


# -------------------- 유틸 --------------------

def estimate_global_grid_size(instance_dict: dict) -> int:
    """
    저장된 bbox / mask를 기반으로 전역 그리드 크기(gs)를 추정.
    """
    max_y1 = 0
    max_x1 = 0
    for v in instance_dict.values():
        bbox = v.get("rgb_bbox") or v.get("bbox")
        m = v.get("mask", None)
        if bbox is not None:
            y0, x0, y1, x1 = bbox
            max_y1 = max(max_y1, int(y1))
            max_x1 = max(max_x1, int(x1))
        elif m is not None and isinstance(m, np.ndarray):
            ys, xs = np.where(m > 0)
            if ys.size:
                max_y1 = max(max_y1, int(ys.max()) + 1)
                max_x1 = max(max_x1, int(xs.max()) + 1)
    return int(max(max_y1, max_x1))

def _id_to_rgb(inst_id: int, s: float = 0.70, v: float = 0.95):
    """인스턴스 id를 안정적으로 구분되는 RGB로 변환 (골든 레이쇼)."""
    g = 0.61803398875
    h = (inst_id * g) % 1.0
    r, g_, b = colorsys.hsv_to_rgb(h, s, v)
    return np.array([r, g_, b], dtype=np.float32)


# -------------------- 포인트클라우드 빌더 (mask 전용) --------------------

def build_point_cloud_from_instance_masks(
    instance_dict: dict,
    cell_size: float = 0.05,
    center_origin: bool = True,
    include_bg: bool = False,      # True면 id 1(벽), 2(바닥) 포함
    stride: int = 1,               # 픽셀 다운샘플 간격

    # ── 높이 매핑 옵션 ──
    mapping: str = "linear",       # 'linear' | 'clamp' | 'percentile' | 'tanh' | 'flat'
    height_exaggeration: float = 1.0,  # linear/clamp에서만 사용
    height_baseline = "min",            # 'min' | 'mean' | 'floor' | float
    layer_gap: float = 0.0,             # 인스턴스 계층 간격(낮은→높은 순)

    # ── 추가 파라미터 (필요한 매핑에서만 사용) ──
    max_height_offset: float = None,    # 'clamp'에서 평균 대비 ±캡(m)
    percentile_clip=(5.0, 95.0),        # 'percentile'에서 퍼센타일 클립
    target_range_m: float = 0.6,        # 'percentile'/'tanh'에서 최종 전체 범위(m)
    compress_scale: float = 0.25,       # 'tanh'에서 눌러주는 스케일(m)

    # ── ★추가: 크롭/정렬 일치 옵션 ──
    crop_bbox: tuple = None,            # (ymin,xmin,ymax,xmax) 내부만 사용
    half_extent_override: float = None  # 룸과 한 기준으로 정렬하려면 동일 값 전달
) -> o3d.geometry.PointCloud:
    """
    instance_dict의 이진 mask로 포인트클라우드 생성.
    세로축은 선택한 매핑 방식으로 '왜곡을 억제'해 표현합니다.
    """
    # 전역 그리드 크기 추정(센터 정렬용)
    gs_est = estimate_global_grid_size(instance_dict)
    half_extent = (gs_est * cell_size * 0.5) if half_extent_override is None else float(half_extent_override)

    # 사용할 인스턴스와 avg_height 수집
    meta = []
    for inst_id, inst in instance_dict.items():
        if inst_id ==1 or inst_id ==2 : continue
        if inst["category"] in ["rug", "mat", "floor", "wall", "ceiling"]: continue
        if not include_bg and inst_id in (1, 2):
            continue
        m = inst.get("mask", None)
        if m is None or np.sum(m) == 0:
            continue
        meta.append((inst_id, float(inst.get("avg_height", 0.0))))
    if not meta:
        raise RuntimeError("유효한 마스크가 없습니다.")

    # 높이 배열 및 베이스라인
    heights = np.array([h for _, h in meta], dtype=np.float32)

    # 'floor' 기준: id==2가 있으면 그 평균 높이를 기준으로 사용
    if (isinstance(height_baseline, str) and height_baseline.lower() == "floor"
        and 2 in instance_dict and "avg_height" in instance_dict[2]):
        base_h = float(instance_dict[2]["avg_height"])
    elif isinstance(height_baseline, (int, float)):
        base_h = float(height_baseline)
    elif str(height_baseline).lower() == "mean":
        base_h = float(np.mean(heights))
    else:
        base_h = float(np.min(heights))

    # 낮은→높은 순으로 랭크 (layer_gap용)
    order = np.argsort(heights)
    rank_map = {meta[idx][0]: int(rk) for rk, idx in enumerate(order)}

    # percentile 매핑 대비 사전 계산
    lo_p, hi_p = percentile_clip
    h_lo = np.percentile(heights, lo_p) if mapping == "percentile" else None
    h_hi = np.percentile(heights, hi_p) if mapping == "percentile" else None
    eps = 1e-6

    def scaled_height(avg_h: float, inst_id: int) -> float:
        """선택한 매핑 방식으로 높이 변환."""
        if mapping == "flat":
            h = base_h

        elif mapping == "linear":
            delta = (avg_h - base_h) * float(height_exaggeration)
            h = base_h + delta

        elif mapping == "clamp":
            delta = (avg_h - base_h) * float(height_exaggeration)
            if max_height_offset is not None:
                delta = np.clip(delta, -float(max_height_offset), float(max_height_offset))
            h = base_h + delta

        elif mapping == "percentile":
            # 퍼센타일로 클립 후, [h_lo, h_hi] → 길이 target_range_m로 선형 재배치
            clamped = np.clip(avg_h, h_lo, h_hi)
            t = (clamped - h_lo) / max(h_hi - h_lo, eps)  # 0~1
            h = base_h + (t - 0.5) * float(target_range_m)

        elif mapping == "tanh":
            # 부드럽게 눌러주기: delta→tanh(delta/scale) * (target_range/2)
            delta = avg_h - base_h
            comp = np.tanh(delta / float(compress_scale))
            h = base_h + comp * (float(target_range_m) * 0.5)

        else:
            # 알 수 없는 매핑이면 linear 취급
            delta = (avg_h - base_h) * float(height_exaggeration)
            h = base_h + delta

        # 인스턴스 랭크에 따른 계층 간격 추가
        return h + float(layer_gap) * rank_map.get(inst_id, 0)

    all_pts = []
    all_cols = []

    for inst_id, inst in instance_dict.items():
        if inst_id ==1 or inst_id ==2 : continue
        if inst["category"] in ["rug", "mat", "floor", "wall", "ceiling"]: continue
        if not include_bg and inst_id in (1, 2):
            continue
        m = inst.get("mask", None)
        if m is None or np.sum(m) == 0:
            continue

        avg_h = float(inst.get("avg_height", 0.0))
        h_vis = scaled_height(avg_h, inst_id)

        ys, xs = np.where(m > 0)
        if ys.size == 0:
            continue
        if stride > 1:
            idx = np.arange(ys.size)[::stride]
            ys = ys[idx]; xs = xs[idx]

        # ★ 크롭 적용 (전역좌표 기준)
        if crop_bbox is not None:
            y0, x0, y1, x1 = crop_bbox
            keep = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            ys, xs = ys[keep], xs[keep]
            if ys.size == 0:
                continue

        # 그리드 인덱스 -> 월드 좌표
        world_x = xs.astype(np.float32) * cell_size
        world_z = ys.astype(np.float32) * cell_size
        if center_origin:
            world_x -= half_extent
            world_z -= half_extent
        world_y = np.full_like(world_x, h_vis, dtype=np.float32)

        pts = np.stack([world_x, world_y, world_z], axis=1)

        # 인스턴스 고유 색
        if inst_id == 1:
            color = np.array([0.6, 0.6, 0.6], dtype=np.float32)  # wall: 회색
        elif inst_id == 2:
            color = np.array([0.8, 0.8, 0.8], dtype=np.float32)  # floor: 밝은 회색
        else:
            color = _id_to_rgb(inst_id, s=0.70, v=0.95)
        cols = np.tile(color, (pts.shape[0], 1))

        all_pts.append(pts)
        all_cols.append(cols)

    if not all_pts:
        raise RuntimeError("유효한 마스크 포인트가 없습니다. (크롭/필터 결과)")

    all_pts = np.concatenate(all_pts, axis=0)
    all_cols = np.concatenate(all_cols, axis=0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_pts)
    pc.colors = o3d.utility.Vector3dVector(all_cols)
    return pc


# -------------------- 룸 바닥 오버레이(파스텔) --------------------

def _room_id_to_pastel(room_id: int):
    """골든 레이쇼 기반 hue + 낮은 S로 파스텔 색상 생성 후 흰색과 블렌딩."""
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
    origin_yx: tuple = (0, 0),
    swap_xz: bool = False,
    flip_x: bool = False,
    flip_z: bool = False,
) -> o3d.geometry.PointCloud:
    """
    room_map(2D int)을 height=room_height에 옅은 색으로 표시.
    room_map이 이미 bbox로 '크롭된 배열'이면 origin_yx=(ymin,xmin)을 전달.
    """
    H, W = room_map.shape[:2]
    ys, xs = np.where(room_map != 0)
    if ys.size == 0:
        return o3d.geometry.PointCloud()

    if stride > 1:
        idx = np.arange(ys.size)[::stride]
        ys, xs = ys[idx], xs[idx]

    rooms = room_map[ys, xs].astype(int)

    oy, ox = origin_yx
    ys_g = ys + int(oy)
    xs_g = xs + int(ox)

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
    # (중요) world_z는 아래 줄에서 같이 빼줘야 함
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


# -------------------- 인스턴스 PC를 ROOM_ID(S) 영역으로 필터 --------------------

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
    """기존 인스턴스 PC에서 특정 ROOM_ID 영역에 속하는 포인트만 남김 (단일)."""
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

def filter_pc_by_room_ids(
    pc: o3d.geometry.PointCloud,
    room_map: np.ndarray,
    room_ids,
    cell_size: float,
    center_origin: bool,
    half_extent: float,
    origin_yx: tuple,
    swap_xz: bool = False,
    flip_x: bool = False,
    flip_z: bool = False,
) -> o3d.geometry.PointCloud:
    """기존 인스턴스 PC에서 ROOM_IDS(여러 방) 영역에 속하는 포인트만 남김."""
    if len(pc.points) == 0:
        return pc
    room_ids = list(room_ids)
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
    vals = room_map[ry[idx_inside], rx[idx_inside]].astype(int)
    keep_room = np.isin(vals, np.array(room_ids, dtype=int))
    if not np.any(keep_room):
        return o3d.geometry.PointCloud()

    idx_final = idx_inside[keep_room]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[idx_final])
    out.colors = o3d.utility.Vector3dVector(cols[idx_final])
    return out


# -------------------- 시각화(탑뷰 + 흰 배경) --------------------

def visualize_point_cloud(pc: o3d.geometry.PointCloud, point_size: float = 2.5):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="2.5D Instance Masks PointCloud", width=1280, height=800)
    vis.add_geometry(pc)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # 흰색 배경

    # 탑뷰 세팅
    bbox = pc.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    ctr = vis.get_view_control()
    ctr.set_lookat(center)              # 씬 중심
    ctr.set_front([0.0, -1.0, 0.0])     # +Y 위에서 -Y 방향으로 내려봄
    ctr.set_up([0.0, 0.0, -1.0])        # 화면 상단이 -Z 방향
    ctr.set_zoom(0.7)                   # 초기 줌

    vis.run()
    vis.destroy_window()


# -------------------- 바이너리 크롭 마스크 로더 --------------------

def load_binary_mask_and_bbox(path: str):
    """
    0/1 바이너리 넘파이(.npy/.npz) 로드 후
    '0'의 bbox(ymin,xmin,ymax,xmax)와 (1-mask) 반환.
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
    ys, xs = np.where(m == 0)  # 0 기준 bbox
    if ys.size == 0:
        raise RuntimeError("바이너리 마스크 내 0이 없습니다.")
    ymin, ymax = int(ys.min()), int(ys.max()) + 1
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    return (ymin, xmin, ymax, xmax), (1 - m)


# -------------------- 실행 예시 --------------------

if __name__ == "__main__":
    # 1) 저장된 instance_dict(.pkl/.npz) 경로
    instance_dict_path = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00873-bxsVRursffK/map/00873-bxsVRursffK_test0811mh3/02buildCatMap/categorized_instance_dict_test0811mh3.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00862-LT9Jq6dN3Ea/map/00862-LT9Jq6dN3Ea_test0811mh6/02buildCatMap/categorized_instance_dict_test0811mh6.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00824-Dd4bFSTQ8gi/map/00824-Dd4bFSTQ8gi_test0811mh3/02buildCatMap/categorized_instance_dict_test0811mh3.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/02buildCatMap/categorized_instance_dict_test0811mh3.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/02buildCatMap/categorized_instance_dict_test0811mh3.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00862-LT9Jq6dN3Ea/map/00862-LT9Jq6dN3Ea_test0811mh6/02buildCatMap/categorized_instance_dict_test0811mh6.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/01buildFeatMap/instance_dict_test0811mh3.pkl" #"/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_rgbmask/01buildFeatMap/instance_dict_rgbmask.pkl"

    # 2) 크롭/룸 경로 (이전과 동일한 예시)
    crop_mask_path = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00873-bxsVRursffK/map/00873-bxsVRursffK_test0811mh3/01buildFeatMap/obstacles_test0811mh3.npy"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00862-LT9Jq6dN3Ea/map/00862-LT9Jq6dN3Ea_test0811mh6/01buildFeatMap/obstacles_test0811mh6.npy"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00824-Dd4bFSTQ8gi/map/00824-Dd4bFSTQ8gi_test0811mh3/01buildFeatMap/obstacles_test0811mh3.npy"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/01buildFeatMap/obstacles_test0811mh3.npy"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/01buildFeatMap/obstacles_test0811mh3.npy" #"/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_rgbmask/01buildFeatMap/obstacles_rgbmask.npy"  # 0/1
    room_map_path  = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00873-bxsVRursffK/map/00873-bxsVRursffK_test0811mh3/04classificateRoom/room_map.npy"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00862-LT9Jq6dN3Ea/map/00862-LT9Jq6dN3Ea_test0811mh6/04classificateRoom/room_map.npy"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/04classificateRoom/room_map.npy"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/04classificateRoom/room_map.npy" #"/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_final_test2/04classificateRoom/room_map.npy"     # 2D int

    # ---- 로드 ----
    instance_dict = load_instance_dict(instance_dict_path)
    crop_bbox, _   = load_binary_mask_and_bbox(crop_mask_path)   # (ymin,xmin,ymax,xmax)
    room_map       = np.load(room_map_path)

    # 필요 시 사용자 환경 정합 보정 (그대로 사용하던 변환이 있으면 유지)
    # room_map = np.rot90(np.rot90(room_map).T, -1)

    # ---- half_extent 통일 (룸 기준 권장) ----
    H_room, W_room = room_map.shape[:2]
    cell_size = 0.05           # 저장 시 cs와 동일 권장
    half_extent = max(H_room, W_room) * cell_size * 0.5

    # ---- 인스턴스 포인트 (mask 기반 + 크롭/정렬 동기화) ----
    pc_inst = build_point_cloud_from_instance_masks(
        instance_dict,
        cell_size=cell_size,
        center_origin=True,
        include_bg=False,        # 1/2(벽/바닥) 포함하려면 True
        stride=1,                # 큰 씬이면 2~4 권장
        height_exaggeration=0.3, # 기존 기본값 그대로
        height_baseline="mean",  # 'min'|'mean'|'floor'|float
        layer_gap=0.02,          # 인스턴스 높이 순서대로 간격
        crop_bbox=crop_bbox,
        half_extent_override=half_extent
    )

    # ---- ROOM_IDS 설정 (리스트 없으면 단일 ROOM_ID로 대체) ----
    room_ids = ROOM_IDS if isinstance(ROOM_IDS, (list, tuple, set)) and len(ROOM_IDS) > 0 else [ROOM_ID]

    # ---- ROOM_IDS 영역으로 인스턴스 포인트 필터링 ----
    pc_inst_rooms = filter_pc_by_room_ids(
        pc=pc_inst,
        room_map=room_map,
        room_ids=room_ids,
        cell_size=cell_size,
        center_origin=True,
        half_extent=half_extent,
        origin_yx=crop_bbox[:2],  # room_map이 전역에서 (ymin,xmin)만큼 오프셋되었다고 가정
        swap_xz=False,
        flip_x=False,
        flip_z=False
    )

    # ---- 인스턴스 Y범위를 읽어 룸 높이 자동 산정 ----
    if len(pc_inst_rooms.points) > 0:
        inst_y = np.asarray(pc_inst_rooms.points)[:, 1]
    else:
        inst_y = np.asarray(pc_inst.points)[:, 1]
    y_min, y_max = float(inst_y.min()), float(inst_y.max())
    if ROOM_PUSH_DIR.lower() == "up":
        room_height = y_min - ROOM_MARGIN
    else:
        room_height = y_max + ROOM_MARGIN

    # ---- 룸 바닥 레이어: ROOM_IDS 합집합만 남기고 시각화 ----
    room_map_masked = np.where(np.isin(room_map.astype(int), np.array(room_ids, dtype=int)), room_map, 0)
    pc_room = build_room_floor_pointcloud(
        room_map=room_map_masked,
        cell_size=cell_size,
        center_origin=True,
        half_extent=half_extent,
        room_height=room_height,
        stride=2,
        origin_yx=crop_bbox[:2],
        swap_xz=False,
        flip_x=False,
        flip_z=False
    )

    # ---- 합쳐서 시각화 ----
    pc_all = o3d.geometry.PointCloud()
    if len(pc_room.points) > 0 and len(pc_inst_rooms.points) > 0:
        pts = np.concatenate([np.asarray(pc_room.points), np.asarray(pc_inst_rooms.points)], axis=0)
        cols = np.concatenate([np.asarray(pc_room.colors), np.asarray(pc_inst_rooms.colors)], axis=0)
        pc_all.points = o3d.utility.Vector3dVector(pts)
        pc_all.colors = o3d.utility.Vector3dVector(cols)
    elif len(pc_room.points) > 0:
        pc_all = pc_room
    else:
        pc_all = pc_inst_rooms

    visualize_point_cloud(pc_all, point_size=3.0)
