# visualize_instance_masks_as_pointcloud.py
# ------------------------------------------------------------
# 인스턴스별 "mask"(이진)로 2.5D 포인트클라우드 시각화
# - 인스턴스마다 구분되는 색 자동 할당 (골든 레이쇼 기반)
# - 높이 과장/레이어 간격 옵션
# - 시작 뷰: 탑뷰, 배경: 흰색
# ------------------------------------------------------------
# 필요 패키지: open3d
#   pip install open3d
# ------------------------------------------------------------

import os
import pickle
import numpy as np
import colorsys
import open3d as o3d
from map.utils.replica_categories import replica_cat

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
    compress_scale: float = 0.25        # 'tanh'에서 눌러주는 스케일(m)
) -> o3d.geometry.PointCloud:
    """
    instance_dict의 이진 mask로 포인트클라우드 생성.
    세로축은 선택한 매핑 방식으로 '왜곡을 억제'해 표현합니다.
    """
    # 전역 그리드 크기 추정(센터 정렬용)
    gs_est = estimate_global_grid_size(instance_dict)
    half_extent = gs_est * cell_size * 0.5

    # 사용할 인스턴스와 avg_height 수집
    meta = []
    for inst_id, inst in instance_dict.items():
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
        if inst["category"] in ["rug", "mat", "floor","ceiling","lamp"]:
            continue
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

    all_pts = np.concatenate(all_pts, axis=0)
    all_cols = np.concatenate(all_cols, axis=0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_pts)
    pc.colors = o3d.utility.Vector3dVector(all_cols)
    return pc


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
    ctr.set_up([0.0, 0.0, -1.0])        # 화면 상단이 -Z 방향(원하면 변경)
    ctr.set_zoom(0.7)                   # 초기 줌

    vis.run()
    vis.destroy_window()


# -------------------- 실행 예시 --------------------

if __name__ == "__main__":
    # 1) 저장된 instance_dict(.pkl/.npz) 경로
    # 예) instance_dict_path = "/path/to/instance_dict_rgbmask.pkl"  # mask/avg_height 포함 파일
    # instance_dict_path = "/path/to/your/instance_dict.pkl"
    instance_dict_path = "/home/vlmap_RCI/Data/habitat_sim/Replica/office3/map/office3_final_test/02buildCatMap/categorized_instance_dict_final_test.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/02buildCatMap/categorized_instance_dict_test0811mh3.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/01buildFeatMap/instance_dict_test0811mh3.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00862-LT9Jq6dN3Ea/map/00862-LT9Jq6dN3Ea_test0811mh6/02buildCatMap/categorized_instance_dict_test0811mh6.pkl" #"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/02buildCatMap/categorized_instance_dict_test0811mh3.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/01buildFeatMap/instance_dict_test0811mh3.pkl" #"/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_rgbmask/01buildFeatMap/instance_dict_rgbmask.pkl"


    # 2) 로드
    instance_dict = load_instance_dict(instance_dict_path)

    # 3) 포인트클라우드 생성 (마스크 기반)
    pc = build_point_cloud_from_instance_masks(
        instance_dict,
        cell_size=0.05,          # 저장 시 cs와 동일 권장
        center_origin=True,
        include_bg=False,        # 1/2(벽/바닥) 포함하려면 True
        stride=1,                # 큰 씬이면 2~4 권장
        height_exaggeration=0.3, # 높이 차이 과장
        height_baseline="mean",   # 'min'|'mean'|float
        layer_gap=0.02           # 인스턴스 높이 순서대로 0.15m 간격
    )

    # 4) 시각화 (탑뷰 + 흰 배경)
    visualize_point_cloud(pc, point_size=3.0)
