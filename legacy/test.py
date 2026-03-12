# visualize_rgb_masks_as_pointcloud.py
import os
import pickle
import numpy as np

# 시각화: Open3D (설치 필요: pip install open3d)
import open3d as o3d


def load_instance_dict(path: str):
    """
    path가 .pkl 이면 pickle 로드, .npz면 np.load에서 'instance_dict' 키를 찾음.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if "instance_dict" in data:
            return data["instance_dict"].item()
        raise ValueError("NPZ 내부에 'instance_dict' 키가 없습니다.")
    else:
        raise ValueError("지원하지 않는 포맷입니다. (.pkl 또는 .npz)")


def estimate_global_grid_size(instance_dict: dict) -> int:
    """
    저장된 bbox들 최대치를 기준으로 전역 그리드 크기(gs)를 추정.
    (정확한 gs를 알고 있다면 이 추정은 건너뛰고 직접 넣어도 됨)
    """
    max_y1 = 0
    max_x1 = 0
    for v in instance_dict.values():
        bbox = v.get("rgb_bbox") or v.get("bbox")
        if bbox is None:
            continue
        y0, x0, y1, x1 = bbox
        max_y1 = max(max_y1, y1)
        max_x1 = max(max_x1, x1)
    return int(max(max_y1, max_x1))


def build_point_cloud_from_instance_dict(
    instance_dict: dict,
    cell_size: float = 0.05,    # 그리드 해상도(cs). 저장 당시 값 사용 권장.
    center_origin: bool = True, # True이면 (0,0)을 그리드 중앙으로 맞춤
    include_bg: bool = False,   # True이면 1(wall),2(floor) 포함
    stride: int = 1,            # 다운샘플링 간격(픽셀 단위)  ← 기존 파라미터 유지
    height_exaggeration: float = 1.0,  # ★ 추가: 높이 과장 배율
    height_baseline="min",              # ★ 추가: 'min'|'mean'|float
    layer_gap: float = 0.0              # ★ 추가: 인스턴스 계층 간격(미터)
) -> o3d.geometry.PointCloud:
    """
    instance_dict에서 인스턴스별 rgb_mask를 모아 하나의 포인트클라우드(PointCloud) 생성.
    각 픽셀은 (X,Z) 격자 좌표를 world로 매핑하고, 높이는 (과장/오프셋 적용된) avg_height로 설정.
    """
    # 전역 그리드 크기 추정 (중심 보정용)
    gs_est = estimate_global_grid_size(instance_dict)
    half_extent = gs_est * cell_size * 0.5  # 중심 정렬 시 사용

    # ── 사전 패스: 사용할 인스턴스와 평균 높이 수집 ──
    inst_meta = []  # (inst_id, avg_h)
    for inst_id, inst in instance_dict.items():
        if not include_bg and inst_id in (1, 2):
            continue
        rgb_mask = inst.get("rgb_mask", None)
        bbox = inst.get("rgb_bbox") or inst.get("bbox")
        if rgb_mask is None or bbox is None:
            continue
        inst_meta.append((inst_id, float(inst.get("avg_height", 0.0))))

    if not inst_meta:
        raise RuntimeError("유효한 인스턴스가 없습니다. rgb_mask/bbox가 비었을 수 있습니다.")

    # ── 기준 높이(baseline) 결정 ──
    heights = np.array([h for _, h in inst_meta], dtype=np.float32)
    if isinstance(height_baseline, (int, float)):
        base_h = float(height_baseline)
    elif str(height_baseline).lower() == "mean":
        base_h = float(np.mean(heights))
    else:  # 'min' (기본값)
        base_h = float(np.min(heights))

    # ── layer_gap용 순위 매기기(낮은 높이→높은 높이) ──
    order = np.argsort(heights)
    rank_map = {inst_meta[idx][0]: int(rk) for rk, idx in enumerate(order)}

    all_pts = []
    all_cols = []

    # ── 실제 포인트 생성 ──
    for inst_id, inst in instance_dict.items():
        if not include_bg and inst_id in (1, 2):
            continue

        rgb_mask = inst.get("rgb_mask", None)
        bbox = inst.get("rgb_bbox") or inst.get("bbox")
        avg_h = float(inst.get("avg_height", 0.0))

        # rgb_mask와 bbox가 모두 있어야 복원 가능
        if rgb_mask is None or bbox is None:
            continue

        # 과장/오프셋 적용된 높이 계산
        # y' = base_h + (avg_h - base_h)*height_exaggeration + layer_gap*rank
        scaled_h = base_h + (avg_h - base_h) * float(height_exaggeration) + float(layer_gap) * rank_map.get(inst_id, 0)

        y0, x0, y1, x1 = bbox
        H, W, _ = rgb_mask.shape

        # 유효 픽셀(== rgb가 0이 아닌 픽셀) 인덱스
        valid = np.any(rgb_mask != 0, axis=2)
        # (주의) 기존 코드 유지: stride 블록은 no-op 성격이므로 그대로 둠
        if stride > 1:
            valid[::stride, :] = valid[::stride, :]
            valid[:, ::stride] = valid[:, ::stride]
        ys, xs = np.where(valid)
        if ys.size == 0:
            continue

        # 전역 그리드 인덱스로 복원
        gy = y0 + ys
        gx = x0 + xs

        # 그리드 인덱스 -> 월드 XZ
        world_x = gx.astype(np.float32) * cell_size
        world_z = gy.astype(np.float32) * cell_size

        if center_origin:
            world_x -= half_extent
            world_z -= half_extent

        # 높이 적용(2.5D): 모든 유효 픽셀을 동일 scaled_h로
        world_y = np.full_like(world_x, scaled_h, dtype=np.float32)

        pts = np.stack([world_x, world_y, world_z], axis=1)

        # 색상 0~1 정규화
        cols = rgb_mask[ys, xs, :].astype(np.float32) / 255.0

        all_pts.append(pts)
        all_cols.append(cols)

    if not all_pts:
        raise RuntimeError("유효한 포인트가 없습니다. rgb_mask가 없거나 모두 배경(0)일 수 있습니다.")

    all_pts = np.concatenate(all_pts, axis=0)
    all_cols = np.concatenate(all_cols, axis=0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_pts)
    pc.colors = o3d.utility.Vector3dVector(all_cols)
    return pc

def visualize_point_cloud(pc: o3d.geometry.PointCloud, point_size: float = 2.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="2.5D RGB Masks PointCloud", width=1280, height=800)
    vis.add_geometry(pc)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # ← 여기만 흰색으로!

    # ── 탑뷰 세팅 ──
    bbox = pc.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([0.0, -1.0, 0.0])
    ctr.set_up([0.0, 0.0, -1.0])
    ctr.set_zoom(0.7)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # 사용 예시
    # 1) 저장된 pkl/npz 경로
    # instance_dict_path = "/home/vlmap_RCI/Data/habitat_sim/hm3dsem/00829-QaLdnwvtxbs/map/00829-QaLdnwvtxbs_rgbmask/01buildFeatMap/instance_dict_rgbmask.pkl"  # 또는 .npz
    instance_dict_path = "/home/vlmap_RCI/Data/habitat_sim/mp3d/2t7WUuJeko7_2/map/2t7WUuJeko7_2_rgbmask/01buildFeatMap/instance_dict_rgbmask.pkl"
    # 2) 로드
    instance_dict = load_instance_dict(instance_dict_path)

    # 3) 포인트 클라우드 생성
    #    cell_size(cs)는 저장 시 사용한 값과 동일하게 맞추는 것을 권장(예: 0.05)
    pc = build_point_cloud_from_instance_dict(
        instance_dict,
        cell_size=0.05,
        center_origin=True,
        include_bg=False,  # 벽/바닥(1,2)을 포함하려면 True
        stride=1,           # 큰 씬이면 2~4로 올려 다운샘플 가능
        layer_gap=0.05
    )

    # 4) 시각화
    visualize_point_cloud(pc, point_size=2.0)
