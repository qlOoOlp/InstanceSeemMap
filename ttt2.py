# visualize_instance_masks_as_pointcloud.py
# ------------------------------------------------------------
# 인스턴스 mask로 2.5D 포인트클라우드 시각화
# - category_id(정수) 또는 category/category_idx로 카테고리별 고정 색상
# - 범례: 콘솔 출력 + category_legend.png 저장
# - 탑뷰/흰 배경 기본
# - 회전(턴테이블) mp4 녹화 지원
# ------------------------------------------------------------
# 필요 패키지:
#   pip install open3d opencv-python
#   (선택) pip install matplotlib
# ------------------------------------------------------------

import os
import math
import pickle
import numpy as np
import colorsys
import collections
import open3d as o3d
from typing import Tuple, Set, Dict

# (선택) 범례 이미지 저장용
try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# (선택) mp4 저장용
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# replica 카테고리명 (index -> name)
try:
    from map.utils.replica_categories import replica_cat
except Exception:
    replica_cat = None  # 없으면 fallback 이름 사용

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
    instance_dict 로드:
    - .pkl: pickle.load → dict 또는 ndarray(object) → dict
    - .npz: 'instance_dict' 키 또는 단일 object 항목 탐색
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj = _unwrap_np_object(obj)

        if _looks_like_instance_dict(obj):
            return obj

        if isinstance(obj, dict) and "instance_dict" in obj:
            cand = _unwrap_np_object(obj["instance_dict"])
            if _looks_like_instance_dict(cand):
                return cand

        if isinstance(obj, (list, tuple)) and obj:
            cand = _unwrap_np_object(obj[0])
            if _looks_like_instance_dict(cand):
                return cand

        raise ValueError("주어진 PKL은 instance_dict 구조가 아닙니다.")

    elif ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if "instance_dict" in data:
            cand = _unwrap_np_object(data["instance_dict"])
            if _looks_like_instance_dict(cand):
                return cand
        if len(data.files) == 1:
            sole = _unwrap_np_object(data[data.files[0]])
            if _looks_like_instance_dict(sole):
                return sole

        raise ValueError("NPZ 내부에서 instance_dict를 찾지 못했습니다.")

    else:
        raise ValueError("지원 포맷 아님: .pkl 또는 .npz 만 지원")

# -------------------- 유틸 --------------------

def estimate_global_grid_size(instance_dict: dict) -> int:
    """저장된 bbox / mask를 기반으로 전역 그리드 크기(gs)를 추정."""
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

def _get_cat_name(cat_id: int) -> str:
    if replica_cat is not None:
        try:
            if 0 <= int(cat_id) < len(replica_cat):
                return str(replica_cat[int(cat_id)])
        except Exception:
            pass
    return f"cat_{int(cat_id)}"

def _strict_int_scalar(x) -> int:
    """스칼라를 엄격히 int로 변환. 실패 시 ValueError."""
    a = np.asarray(x)
    if a.size != 1:
        raise ValueError(f"정수 스칼라가 아님: shape={a.shape}, value={x}")
    return int(a.item())

def _build_palette(category_ids):
    """
    tab20 기반 팔레트 → 부족하면 golden-ratio로 보강.
    같은 cat_id는 같은 색, 다른 cat_id는 분명히 다른 색.
    """
    cat_ids = list(sorted(set(category_ids)))
    lut = {}
    if _HAS_MPL:
        cmap = cm.get_cmap('tab20', 20)
        for i, cid in enumerate(cat_ids):
            if i < 20:
                lut[cid] = np.array(cmap(i)[:3], dtype=np.float32)
            else:
                h = ((i - 20) * 0.61803398875) % 1.0
                r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.95)
                lut[cid] = np.array([r, g, b], dtype=np.float32)
    else:
        for i, cid in enumerate(cat_ids):
            h = (i * 0.61803398875) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.95)
            lut[cid] = np.array([r, g, b], dtype=np.float32)
    return lut

def _print_and_save_legend(category_ids_present: Set[int], cat2color: dict, out_png="category_legend.png"):
    items = [(cid, _get_cat_name(cid)) for cid in sorted(category_ids_present, key=lambda x: (x < 0, x))]
    print("\n[Category Legend]")
    for cid, name in items:
        col = (cat2color.get(cid, np.array([0.5, 0.5, 0.5])) * 255).astype(int)
        print(f"  id={cid:>5} | {name} | color RGB={tuple(col.tolist())}")

    if _HAS_MPL:
        fig_h = max(2, int(len(items) * 0.35))
        fig, ax = plt.subplots(figsize=(8, fig_h))
        ax.axis('off')
        y = 0.95
        dy = 0.9 / max(1, len(items))
        for cid, name in items:
            col = cat2color.get(cid, np.array([0.5, 0.5, 0.5]))
            ax.add_patch(plt.Rectangle((0.05, y - 0.03), 0.05, 0.03, transform=ax.transAxes, color=col))
            ax.text(0.12, y - 0.02, f"[{cid}] {name}", transform=ax.transAxes, va='center', fontsize=10)
            y -= dy
        plt.tight_layout()
        try:
            plt.savefig(out_png, dpi=150)
            print(f"[Legend saved] {os.path.abspath(out_png)}")
        except Exception as e:
            print(f"[Legend save failed] {e}")
        plt.close(fig)
    else:
        print("(matplotlib 미설치로 PNG 저장은 생략됩니다.)")

# -------------------- 카테고리 추출(폴백 포함) --------------------

def _build_name_index(replica_cat_list) -> Dict[str, int]:
    """replica_cat 이름 → 인덱스 맵 (소문자/공백 제거)."""
    if replica_cat_list is None:
        return {}
    name2idx = {}
    for i, n in enumerate(replica_cat_list):
        key = str(n).strip().lower().replace(" ", "")
        name2idx[key] = i
    return name2idx

def _extract_category_id(inst: dict,
                         name2idx: Dict[str, int],
                         unknown_name2negid: Dict[str, int]) -> int:
    """
    우선순위:
      1) category_id
      2) category_idx
      3) category(문자열) -> replica_cat 매칭 (없으면 이름별 음수 id)
      4) 완전 미지정 -> -9999
    """
    # 1) category_id
    if "category_id" in inst:
        try:
            return _strict_int_scalar(inst["category_id"])
        except Exception:
            pass
    # 2) category_idx
    if "category_idx" in inst:
        try:
            return _strict_int_scalar(inst["category_idx"])
        except Exception:
            pass
    # 3) category (string)
    cat_name = None
    for key in ("category", "class", "label"):
        if key in inst and isinstance(inst[key], (str, np.str_, np.string_)):
            cat_name = str(inst[key]).strip()
            break
    if cat_name:
        k = cat_name.lower().replace(" ", "")
        if k in name2idx:
            return int(name2idx[k])
        if k not in unknown_name2negid:
            unknown_name2negid[k] = -1000 - len(unknown_name2negid)
        return unknown_name2negid[k]
    # 4) 완전 미지정
    return -9999

# -------------------- 포인트클라우드 빌더 (mask 전용) --------------------

def build_point_cloud_from_instance_masks(
    instance_dict: dict,
    cell_size: float = 0.05,
    center_origin: bool = True,
    include_bg: bool = False,      # True면 id 1(벽), 2(바닥) 포함 (인스턴스 id 기준)
    stride: int = 1,               # 픽셀 다운샘플 간격
    mapping: str = "linear",       # 'linear' | 'clamp' | 'percentile' | 'tanh' | 'flat'
    height_exaggeration: float = 1.0,
    height_baseline = "min",       # 'min' | 'mean' | 'floor' | float
    layer_gap: float = 0.0,
    max_height_offset: float = None,
    percentile_clip=(5.0, 95.0),
    target_range_m: float = 0.6,
    compress_scale: float = 0.25
) -> Tuple[o3d.geometry.PointCloud, Set[int], dict]:
    """
    instance_dict의 이진 mask로 포인트클라우드 생성.
    반환: (point_cloud, 사용된 category_id 집합, cat_id->color LUT)
    """
    # 전역 그리드 크기 추정(센터 정렬용)
    gs_est = estimate_global_grid_size(instance_dict)
    half_extent = gs_est * cell_size * 0.5

    # replica 이름 인덱스/미지정 이름 음수 id 맵
    name2idx = _build_name_index(replica_cat)
    unknown_name2negid: Dict[str, int] = {}

    # 1) 수집 + category_id(폴백 포함) 추출
    entries = []
    for inst_id, inst in instance_dict.items():
        if not include_bg and inst_id in (1, 2):
            continue
        m = inst.get("mask", None)
        if m is None or np.sum(m) == 0:
            continue

        cat_id = _extract_category_id(inst, name2idx, unknown_name2negid)

        entries.append({
            "inst_id": inst_id,
            "cat_id":  int(cat_id),
            "avg_h":   float(inst.get("avg_height", 0.0)),
            "mask":    m
        })
    if not entries:
        raise RuntimeError("유효한 마스크가 없습니다.")

    # 2) 진단 로그: 카테고리 분포
    cat_counts = collections.Counter([e["cat_id"] for e in entries])
    uniq_cats = sorted(cat_counts.keys())
    print(f"[DEBUG] unique category_ids (count={len(uniq_cats)}): {uniq_cats}")
    print("[DEBUG] category frequency (top 20):",
          ", ".join([f"{cid}:{cat_counts[cid]}" for cid in uniq_cats[:20]]))

    # 3) 팔레트 구성
    cat2color = _build_palette(uniq_cats)

    # 4) 높이 스케일링 준비
    heights = np.array([e["avg_h"] for e in entries], dtype=np.float32)
    if (isinstance(height_baseline, str) and height_baseline.lower() == "floor"
        and 2 in instance_dict and "avg_height" in instance_dict[2]):
        base_h = float(instance_dict[2]["avg_height"])
    elif isinstance(height_baseline, (int, float)):
        base_h = float(height_baseline)
    elif str(height_baseline).lower() == "mean":
        base_h = float(np.mean(heights))
    else:
        base_h = float(np.min(heights))

    order = np.argsort(heights)
    rank_map = {entries[idx]["inst_id"]: int(rk) for rk, idx in enumerate(order)}

    lo_p, hi_p = percentile_clip
    h_lo = np.percentile(heights, lo_p) if mapping == "percentile" else None
    h_hi = np.percentile(heights, hi_p) if mapping == "percentile" else None
    eps = 1e-6

    def scaled_height(avg_h: float, inst_id: int) -> float:
        if mapping == "flat":
            h = base_h
        elif mapping == "linear":
            h = base_h + (avg_h - base_h) * float(height_exaggeration)
        elif mapping == "clamp":
            d = (avg_h - base_h) * float(height_exaggeration)
            if max_height_offset is not None:
                d = np.clip(d, -float(max_height_offset), float(max_height_offset))
            h = base_h + d
        elif mapping == "percentile":
            c = np.clip(avg_h, h_lo, h_hi)
            t = (c - h_lo) / max(h_hi - h_lo, eps)
            h = base_h + (t - 0.5) * float(target_range_m)
        elif mapping == "tanh":
            d = avg_h - base_h
            h = base_h + np.tanh(d / float(compress_scale)) * (float(target_range_m) * 0.5)
        else:
            h = base_h + (avg_h - base_h) * float(height_exaggeration)
        return h + float(layer_gap) * rank_map.get(inst_id, 0)

    # 5) 포인트/색 생성
    all_pts, all_cols = [], []
    for e in entries:
        inst_id, cat_id, avg_h, m = e["inst_id"], e["cat_id"], e["avg_h"], e["mask"]
        h_vis = scaled_height(avg_h, inst_id)

        ys, xs = np.where(m > 0)
        if ys.size == 0:
            continue
        if stride > 1:
            idx = np.arange(ys.size)[::stride]
            ys = ys[idx]; xs = xs[idx]

        world_x = xs.astype(np.float32) * cell_size
        world_z = ys.astype(np.float32) * cell_size
        if center_origin:
            world_x -= half_extent
            world_z -= half_extent
        world_y = np.full_like(world_x, h_vis, dtype=np.float32)

        pts = np.stack([world_x, world_y, world_z], axis=1)
        color = cat2color.get(cat_id, np.array([0.5, 0.5, 0.5], dtype=np.float32))
        cols  = np.tile(color, (pts.shape[0], 1))

        all_pts.append(pts)
        all_cols.append(cols)

    all_pts = np.concatenate(all_pts, axis=0)
    all_cols = np.concatenate(all_cols, axis=0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_pts)
    pc.colors = o3d.utility.Vector3dVector(all_cols)

    # 범례 출력/저장
    _print_and_save_legend(set(uniq_cats), cat2color, out_png="category_legend.png")

    return pc, set(uniq_cats), cat2color

# -------------------- 시각화(탑뷰 + 흰 배경) --------------------

def visualize_point_cloud(pc: o3d.geometry.PointCloud, point_size: float = 2.5):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="2.5D Category-colored PointCloud", width=1280, height=800)
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

# -------------------- 회전 영상 저장 (턴테이블) --------------------

def _look_at_extrinsic(eye, center, up=(0, 1, 0)):
    """eye/center/up -> Open3D extrinsic(4x4, world->camera)"""
    eye    = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up     = np.asarray(up, dtype=np.float32)

    z = eye - center                 # camera forward
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)

    extr = np.eye(4, dtype=np.float32)
    extr[:3, :3] = np.stack([x, y, z], axis=0)
    extr[:3, 3]  = -extr[:3, :3] @ eye
    return extr

def record_turntable_mp4(
    pc,
    out_path="turntable.mp4",
    seconds=8,
    fps=30,
    width=1280,
    height=800,
    point_size=3.0,
    bg=(1.0, 1.0, 1.0),
    elevation_deg=25.0,     # 위/아래 각도(고정)
    radius_scale=2.2        # 모델 크기에 대한 카메라 반경 스케일
):
    """Open3D로 한 바퀴(360°) 회전 영상을 MP4로 저장"""
    if not _HAS_CV2:
        raise RuntimeError("opencv-python이 필요합니다: pip install opencv-python")

    vis = o3d.visualization.Visualizer()
    # 일부 Open3D는 visible 인자를 지원(무헤드 녹화). 미지원이면 창이 뜰 수 있음.
    try:
        vis.create_window(window_name="rec", width=width, height=height, visible=False)
    except TypeError:
        vis.create_window(window_name="rec", width=width, height=height)
    vis.add_geometry(pc)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array(bg, dtype=np.float32)

    # 중심/반경 계산
    bbox   = pc.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = float(np.linalg.norm(extent)) * radius_scale

    # 초기 카메라 파라미터 준비
    ctr   = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()  # intrinsics 유지
    elev  = math.radians(elevation_deg)

    # 비디오 라이터
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    total = int(seconds * fps)
    for i in range(total):
        azim = 2.0 * math.pi * (i / total)   # 0 → 2π
        # 구면 좌표계에서 eye 계산 (Y축을 수직축으로 가정)
        eye = np.array([
            center[0] + radius * math.cos(elev) * math.cos(azim),
            center[1] + radius * math.sin(elev),
            center[2] + radius * math.cos(elev) * math.sin(azim)
        ], dtype=np.float32)

        param.extrinsic = _look_at_extrinsic(eye, center, up=(0, 1, 0))
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events(); vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=False))  # HxWx3 float32 [0..1]
        frame = (np.clip(img, 0, 1) * 255).astype(np.uint8)                 # RGB
        frame = frame[:, :, ::-1]  # RGB->BGR (OpenCV)
        vw.write(frame)

    vw.release()
    vis.destroy_window()
    print(f"[Saved] {os.path.abspath(out_path)}")

# -------------------- 실행 예시 --------------------

if __name__ == "__main__":
    # 1) 저장된 instance_dict(.pkl/.npz) 경로
    instance_dict_path = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/02buildCatMap/categorized_instance_dict_test0811mh3.pkl"

    # 2) 로드
    instance_dict = load_instance_dict(instance_dict_path)

    # 3) 포인트클라우드 생성 (마스크 기반, 카테고리 색)
    pc, cats, cat2color = build_point_cloud_from_instance_masks(
        instance_dict,
        cell_size=0.05,
        center_origin=True,
        include_bg=False,
        stride=1,
        height_exaggeration=0.3,
        height_baseline="mean",
        layer_gap=0.02
    )

    # 4-a) 인터랙티브 창으로 보기
    # visualize_point_cloud(pc, point_size=3.0)

    # 4-b) 회전 mp4로 저장
    record_turntable_mp4(
        pc,
        out_path="turntable.mp4",
        seconds=8,
        fps=30,
        width=1280,
        height=800,
        point_size=3.0,
        bg=(1.0, 1.0, 1.0),
        elevation_deg=25.0,
        radius_scale=2.2
    )
