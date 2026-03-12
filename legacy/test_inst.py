# visualize_instance_masks_as_pointcloud.py
# ------------------------------------------------------------
# 인스턴스별 mask로 2.5D 포인트클라우드 시각화
# - category_id(0~100 정수) 또는 category/category_idx 기반으로 카테고리별 동일 색상
# - 범례: 콘솔 출력 + category_legend.png 저장
# - 시작 뷰: 탑뷰, 배경: 흰색
# - ★안정 팔레트(Stable Palette): replica_cat 전역 색상 매핑을 파일로 저장/로드
# ------------------------------------------------------------
# 필요 패키지: open3d, (선택) matplotlib
#   pip install open3d matplotlib
# ------------------------------------------------------------

import os
import json
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

# replica 카테고리명 (index -> name)
try:
    from map.utils.replica_categories import replica_cat
except Exception:
    replica_cat = None  # 없으면 fallback 이름 사용

# ---------- 안정 팔레트(파일 캐시) 설정 ----------
PALETTE_JSON = "replica_palette.json"  # 필요 시 절대경로로 변경

# -------------------- 로딩 도우미 --------------------

def _unwrap_np_object(obj):
    if isinstance(obj, np.ndarray):
        try:
            return obj.item()
        except Exception:
            pass
    return obj

def _looks_like_instance_dict(d):
    if not isinstance(d, dict):
        return False
    for v in d.values():
        if isinstance(v, dict):
            k = v.keys()
            if ("mask" in k and "avg_height" in k) or ("rgb_mask" in k and "avg_height" in k):
                return True
    return False

def load_instance_dict(path: str) -> dict:
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
    a = np.asarray(x)
    if a.size != 1:
        raise ValueError(f"정수 스칼라가 아님: shape={a.shape}, value={x}")
    return int(a.item())

# ---------- 안정 팔레트: 로드/생성/업데이트 ----------

def _hsv_color(h: float, s: float = 0.80, v: float = 0.95):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return [float(r), float(g), float(b)]

def _det_color_by_index(idx: int) -> list:
    """cat 인덱스 기반 결정론적 색상 (golden-ratio)."""
    phi = 0.61803398875
    return _hsv_color((idx * phi) % 1.0)

def _det_color_by_string(key: str) -> list:
    """문자열 해시 기반 결정론적 색상(unknown name 등)."""
    h = (hash(key) & 0xFFFFFFFF) / 0xFFFFFFFF  # [0,1)
    return _hsv_color(h)

def _load_palette(path: str) -> Dict[str, list]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # sanity: 값은 [r,g,b] floats
            return {str(k): list(map(float, v)) for k, v in data.items()}
        except Exception:
            pass
    return {}

def _save_palette(path: str, palette: Dict[str, list]):
    tmp = {str(k): [float(c) for c in v] for k, v in palette.items()}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tmp, f, ensure_ascii=False, indent=2)
        print(f"[StablePalette] saved: {os.path.abspath(path)}")
    except Exception as e:
        print(f"[StablePalette] save failed: {e}")

def _ensure_replica_palette(replica_cat_list, palette_path: str) -> Dict[str, list]:
    """
    파일에서 팔레트를 로드하고, replica_cat 인덱스의 색이 빠져 있으면 채워 넣어 저장.
    키는 문자열화된 정수(cat_id).
    """
    palette = _load_palette(palette_path)

    if replica_cat_list is not None:
        # 우선 tab20(있으면) → 나머지 golden-ratio로 생성해 넣기
        if _HAS_MPL:
            cmap = cm.get_cmap("tab20", 20)
        for i in range(len(replica_cat_list)):
            k = str(i)
            if k not in palette:
                if _HAS_MPL and i < 20:
                    rgb = list(map(float, cmap(i)[:3]))
                else:
                    rgb = _det_color_by_index(i)
                palette[k] = rgb

    # 최소한 기본 unknown id(-9999)는 지정
    if str(-9999) not in palette:
        palette[str(-9999)] = _hsv_color(0.0, 0.0, 0.75)  # 회색ish

    # 저장(없거나 보강된 경우)
    _save_palette(palette_path, palette)
    return palette

def _get_color_from_palette(cat_id: int, cat_name_key: str, palette: Dict[str, list]) -> np.ndarray:
    """
    cat_id가 팔레트에 있으면 그 색을, 없으면 문자열/인덱스 기반 결정론적 색을 생성해
    팔레트에 추가(디스크 반영) 후 반환.
    """
    k = str(int(cat_id))
    if k in palette:
        return np.array(palette[k], dtype=np.float32)

    # 팔레트에 없는 새 id: 이름 키가 있으면 문자열 해시, 없으면 인덱스 기반
    if cat_name_key:
        rgb = _det_color_by_string(cat_name_key)
    else:
        rgb = _det_color_by_index(int(cat_id))

    palette[k] = rgb
    _save_palette(PALETTE_JSON, palette)
    return np.array(rgb, dtype=np.float32)

# -------------------- 카테고리 추출(폴백 포함) --------------------

def _build_name_index(replica_cat_list) -> Dict[str, int]:
    if replica_cat_list is None:
        return {}
    name2idx = {}
    for i, n in enumerate(replica_cat_list):
        key = str(n).strip().lower().replace(" ", "")
        name2idx[key] = i
    return name2idx

def _extract_category_id(inst: dict,
                         name2idx: Dict[str, int],
                         unknown_name2negid: Dict[str, int]) -> Tuple[int, str]:
    """
    반환: (cat_id, name_key)
      - name_key는 팔레트 미스 시 색 안정화를 위한 문자열 키(가능하면 사용)
    """
    # 1) category_id
    if "category_id" in inst:
        try:
            cid = _strict_int_scalar(inst["category_id"])
            return cid, str(cid)
        except Exception:
            pass
    # 2) category_idx
    if "category_idx" in inst:
        try:
            cid = _strict_int_scalar(inst["category_idx"])
            return cid, str(cid)
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
            idx = int(name2idx[k])
            return idx, k  # 안정 팔레트: replica 인덱스 사용
        if k not in unknown_name2negid:
            unknown_name2negid[k] = -1000 - len(unknown_name2negid)
        return unknown_name2negid[k], k

    # 4) 완전 미지정
    return -9999, "unknown"

# -------------------- 포인트클라우드 빌더 (mask 전용) --------------------

def build_point_cloud_from_instance_masks(
    instance_dict: dict,
    cell_size: float = 0.05,
    center_origin: bool = True,
    include_bg: bool = False,
    stride: int = 1,
    mapping: str = "linear",
    height_exaggeration: float = 1.0,
    height_baseline = "min",
    layer_gap: float = 0.0,
    max_height_offset: float = None,
    percentile_clip=(5.0, 95.0),
    target_range_m: float = 0.6,
    compress_scale: float = 0.25
) -> Tuple[o3d.geometry.PointCloud, Set[int], dict]:
    # 전역 그리드 크기 추정
    gs_est = estimate_global_grid_size(instance_dict)
    half_extent = gs_est * cell_size * 0.5

    # 안정 팔레트 준비
    stable_palette = _ensure_replica_palette(replica_cat, PALETTE_JSON)

    # replica 이름 인덱스/미지정 이름 음수 id 맵
    name2idx = _build_name_index(replica_cat)
    unknown_name2negid: Dict[str, int] = {}

    # 1) 수집 + category_id 추출
    entries = []
    for inst_id, inst in instance_dict.items():
        if inst_id == 1 or inst_id == 2:
            if not include_bg:
                continue
        # 선택적 필터링(원문 유지)
        if inst.get("category") in ["rug", "mat", "floor", "wall", "ceiling"]:
            continue

        m = inst.get("mask", None)
        if m is None or np.sum(m) == 0:
            continue

        cat_id, name_key = _extract_category_id(inst, name2idx, unknown_name2negid)

        entries.append({
            "inst_id": inst_id,
            "cat_id":  int(cat_id),
            "name_key": name_key,
            "avg_h":   float(inst.get("avg_height", 0.0)),
            "mask":    m
        })
    if not entries:
        raise RuntimeError("유효한 마스크가 없습니다.")

    # 2) 진단 로그
    cat_counts = collections.Counter([e["cat_id"] for e in entries])
    uniq_cats = sorted(cat_counts.keys())
    print(f"[DEBUG] unique category_ids (count={len(uniq_cats)}): {uniq_cats}")
    print("[DEBUG] category frequency (top 20):",
          ", ".join([f"{cid}:{cat_counts[cid]}" for cid in uniq_cats[:20]]))

    # 3) 높이 스케일링 준비
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

    # 4) 포인트/색 생성 (안정 팔레트 사용)
    all_pts, all_cols = [], []
    for e in entries:
        inst_id, cat_id, name_key, avg_h, m = e["inst_id"], e["cat_id"], e["name_key"], e["avg_h"], e["mask"]
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

        # 핵심: 안정 팔레트에서 색상 획득(없으면 생성+저장)
        color = _get_color_from_palette(cat_id, name_key, stable_palette)
        cols  = np.tile(color, (pts.shape[0], 1))

        all_pts.append(pts)
        all_cols.append(cols)

    all_pts = np.concatenate(all_pts, axis=0)
    all_cols = np.concatenate(all_cols, axis=0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_pts)
    pc.colors = o3d.utility.Vector3dVector(all_cols)

    # 사용된 카테고리 id→색 LUT(실행 시점 기준) 반환용
    cat2color_now = {int(k): np.array(v, dtype=np.float32) for k, v in _load_palette(PALETTE_JSON).items()}

    _print_and_save_legend(set(uniq_cats), {cid: cat2color_now.get(cid, np.array([0.5,0.5,0.5])) for cid in uniq_cats},
                           out_png="category_legend.png")

    return pc, set(uniq_cats), cat2color_now

# -------------------- 시각화(탑뷰 + 흰 배경) --------------------

def visualize_point_cloud(pc: o3d.geometry.PointCloud, point_size: float = 2.5):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="2.5D Category-colored PointCloud", width=1280, height=800)
    vis.add_geometry(pc)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    bbox = pc.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([0.0, -1.0, 0.0])
    ctr.set_up([0.0, 0.0, -1.0])
    ctr.set_zoom(0.7)

    vis.run()
    vis.destroy_window()


def _print_and_save_legend(category_ids_present, cat2color, out_png="category_legend.png"):
    """콘솔에 범례를 출력하고, matplotlib이 있으면 PNG로도 저장."""
    items = [(cid, _get_cat_name(cid)) for cid in sorted(category_ids_present, key=lambda x: (x < 0, x))]
    print("\n[Category Legend]")
    for cid, name in items:
        col = (cat2color.get(cid, np.array([0.5, 0.5, 0.5])) * 255).astype(int)
        print(f"  id={cid:>5} | {name} | color RGB={tuple(col.tolist())}")

    if not _HAS_MPL:
        print("(matplotlib 미설치로 PNG 저장은 생략됩니다.)")
        return

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

# -------------------- 실행 예시 --------------------

if __name__ == "__main__":
    instance_dict_path = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_test0811mh3/02buildCatMap/categorized_instance_dict_test0811mh3.pkl"#"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00873-bxsVRursffK/map/00873-bxsVRursffK_test0811mh3/02buildCatMap/categorized_instance_dict_test0811mh3.pkl"

    instance_dict = load_instance_dict(instance_dict_path)

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

    visualize_point_cloud(pc, point_size=3.0)
