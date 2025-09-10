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
import hashlib  # 이름/ID 해시용

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

# ============================================================
# [변경] 고분별 팔레트: Lab 공간 maximin 기반의 "서로 확실히 다른" 색
# ============================================================

def _srgb_to_xyz(rgb):
    """rgb: (...,3) in [0,1] -> XYZ (D65)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    a = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
    xyz = a @ M.T
    return xyz

def _xyz_to_lab(xyz):
    """XYZ -> CIE Lab (D65, 2°)."""
    # D65 white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn
    eps = 216/24389
    kappa = 24389/27

    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)

def _delta_e76(lab1, lab2):
    d = lab1 - lab2
    return np.sqrt(np.sum(d * d, axis=-1))

def _hsv_grid_candidates():
    """
    HSV 후보색 그리드(결정적 순서).
    - H: 0..357 step 3 (120 값)
    - S: 0.60, 0.72, 0.85 (3)
    - V: 0.70, 0.82, 0.94 (3)
    총 120*3*3 = 1080 후보
    """
    hs = np.arange(0, 360, 3, dtype=np.float64)
    Ss = np.array([0.60, 0.72, 0.85], dtype=np.float64)
    Vs = np.array([0.70, 0.82, 0.94], dtype=np.float64)
    rgbs = []
    for v in Vs:
        for s in Ss:
            for h in hs:
                r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
                rgbs.append([r, g, b])
    return np.array(rgbs, dtype=np.float64)  # (1080,3)

def _generate_distinct_palette(K: int) -> np.ndarray:
    """
    Lab 공간 maximin으로 K개 색을 고르는 결정적 팔레트.
    반환: (K,3) in [0,1]
    """
    # 후보 생성
    cand_rgb = _hsv_grid_candidates()              # (Nc,3)
    cand_lab = _xyz_to_lab(_srgb_to_xyz(cand_rgb)) # (Nc,3)

    # 시드: 대비 높은 몇 개를 먼저 고정
    seed_hsv = [
        (0.00, 0.80, 0.92),   # vivid red
        (0.12, 0.80, 0.92),   # orange
        (0.58, 0.65, 0.92),   # blue-cyan
        (0.33, 0.80, 0.88),   # green
        (0.75, 0.55, 0.92),   # purple
    ]
    seed_rgb = np.array([colorsys.hsv_to_rgb(*hsv) for hsv in seed_hsv], dtype=np.float64)
    seed_lab = _xyz_to_lab(_srgb_to_xyz(seed_rgb))

    sel_rgb = [seed_rgb[0]]
    sel_lab = [seed_lab[0]]

    # 초기 min-dist
    min_d = _delta_e76(cand_lab, sel_lab[0][None, :])

    # 나머지 시드 반영
    for i in range(1, min(len(seed_lab), K)):
        d = _delta_e76(cand_lab, seed_lab[i][None, :])
        min_d = np.minimum(min_d, d)
        sel_rgb.append(seed_rgb[i])
        sel_lab.append(seed_lab[i])

    # greedy maximin
    while len(sel_rgb) < K:
        idx = int(np.argmax(min_d))
        sel_rgb.append(cand_rgb[idx])
        sel_lab.append(cand_lab[idx])
        d = _delta_e76(cand_lab, cand_lab[idx][None, :])
        min_d = np.minimum(min_d, d)

    return np.clip(np.array(sel_rgb, dtype=np.float64), 0.0, 1.0)

def _stable_hash_index(x: int, mod: int) -> int:
    """정수 x를 0..mod-1 범위의 결정적 인덱스로 매핑."""
    h = hashlib.sha1(str(int(x)).encode('utf-8')).hexdigest()
    return int(h[:8], 16) % max(1, mod)

def _build_global_cat_lut(replica_cat_list, base_size: int = 256):
    """
    전역 팔레트 및 cid->color LUT.
    - known_size = len(replica_cat) 또는 base_size 중 큰 값으로 팔레트 길이 결정
    - 0..known_size-1 은 바로 인덱스로 매핑 (직관성 보존, 고정)
    - 그 외(큰 양수/음수)는 해시로 팔레트 인덱스에 매핑
    """
    known_size = len(replica_cat_list) if replica_cat_list is not None else 0
    K = max(known_size, base_size)
    PALETTE = _generate_distinct_palette(K)  # (K,3)

    def _get_color(cid: int) -> np.ndarray:
        if cid >= 0 and cid < K:
            return PALETTE[cid]
        # 음수 또는 큰 양수는 해시 분산으로 안정 매핑
        idx = _stable_hash_index(cid, K)
        return PALETTE[idx]

    return _get_color, K

_GET_COLOR_FOR_CID, _PALETTE_SIZE = _build_global_cat_lut(replica_cat, base_size=384)

def _stable_neg_id_from_name(name_norm: str) -> int:
    """
    이름 문자열을 안정적인 음수 ID로 변환 (실행/등장순서와 무관).
    """
    h = hashlib.sha1(name_norm.encode('utf-8')).hexdigest()
    val = int(h[:8], 16)  # 32-bit
    return -(1000 + (val % 10_000_000))

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

# (이름은 유지하되 내부 구현은 전역 규칙 사용)
def _build_palette(category_ids):
    lut = {}
    for cid in sorted(set(category_ids)):
        lut[cid] = _GET_COLOR_FOR_CID(cid)
    return lut

def _print_and_save_legend(category_ids_present: Set[int], cat2color: dict, out_png="category_legend.png"):
    items = [(cid, _get_cat_name(cid)) for cid in sorted(category_ids_present, key=lambda x: (x < 0, x))]
    print("\n[Category Legend]")
    for cid, name in items:
        col_arr = cat2color.get(cid, _GET_COLOR_FOR_CID(cid))
        col = (np.asarray(col_arr) * 255).astype(int)
        print(f"  id={cid:>5} | {name} | color RGB={tuple(col.tolist())}")

    if _HAS_MPL:
        fig_h = max(2, int(len(items) * 0.35))
        fig, ax = plt.subplots(figsize=(8, fig_h))
        ax.axis('off')
        y = 0.95
        dy = 0.9 / max(1, len(items))
        for cid, name in items:
            col_arr = cat2color.get(cid, _GET_COLOR_FOR_CID(cid))
            ax.add_patch(plt.Rectangle((0.05, y - 0.03), 0.05, 0.03, transform=ax.transAxes, color=col_arr))
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
                         name2idx: Dict[str, int]) -> int:
    """
    우선순위:
      1) category_id
      2) category_idx
      3) category(문자열) -> replica_cat 매칭 (없으면 이름 해시 기반 음수 id)
      4) 완전 미지정 -> -9999
    """
    if "category_id" in inst:
        try:
            return _strict_int_scalar(inst["category_id"])
        except Exception:
            pass
    if "category_idx" in inst:
        try:
            return _strict_int_scalar(inst["category_idx"])
        except Exception:
            pass
    cat_name = None
    for key in ("category", "class", "label"):
        if key in inst and isinstance(inst[key], (str, np.str_, np.string_)):
            cat_name = str(inst[key]).strip()
            break
    if cat_name:
        k = cat_name.lower().replace(" ", "")
        if k in name2idx:
            return int(name2idx[k])
        return _stable_neg_id_from_name(k)
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

    # replica 이름 인덱스
    name2idx = _build_name_index(replica_cat)

    # 1) 수집 + category_id(폴백 포함) 추출
    entries = []
    for inst_id, inst in instance_dict.items():
        if not include_bg and inst_id in (1, 2):
            continue
        if inst["category"] in ["rug", "mat", "floor", "wall", "ceiling", "lamp"]:
            continue
        if inst["category"] == "chair":
            inst["avg_height"] -= 0.2
        m = inst.get("mask", None)
        if m is None or np.sum(m) == 0:
            continue

        cat_id = _extract_category_id(inst, name2idx)

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

    # 3) 팔레트 구성 - 전역 규칙 기반(cat_id 고정)
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
        color = cat2color.get(cat_id, _GET_COLOR_FOR_CID(cat_id))  # 고정 + 고분별 팔레트
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
    try:
        vis.create_window(window_name="rec", width=width, height=height, visible=False)
    except TypeError:
        vis.create_window(window_name="rec", width=width, height=height)
    vis.add_geometry(pc)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array(bg, dtype=np.float32)

    bbox   = pc.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = float(np.linalg.norm(extent)) * radius_scale

    ctr   = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    elev  = math.radians(elevation_deg)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    total = int(seconds * fps)
    for i in range(total):
        azim = 2.0 * math.pi * (i / total)
        eye = np.array([
            center[0] + radius * math.cos(elev) * math.cos(azim),
            center[1] + radius * math.sin(elev),
            center[2] + radius * math.cos(elev) * math.sin(azim)
        ], dtype=np.float32)

        param.extrinsic = _look_at_extrinsic(eye, center, up=(0, 1, 0))
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events(); vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=False))
        frame = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frame = frame[:, :, ::-1]  # RGB->BGR
        vw.write(frame)

    vw.release()
    vis.destroy_window()
    print(f"[Saved] {os.path.abspath(out_path)}")

# -------------------- 실행 예시 --------------------

if __name__ == "__main__":
    # 1) 저장된 instance_dict(.pkl/.npz) 경로
    instance_dict_path = "/home/vlmap_RCI/Data/habitat_sim/Replica/room1/map/room1_final_test/02buildCatMap/categorized_instance_dict_final_test.pkl"

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
        layer_gap=0.05
    )

    # 4-a) 인터랙티브 창으로 보기
    visualize_point_cloud(pc, point_size=3.0)

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
