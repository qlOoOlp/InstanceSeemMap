# color_by_object_hardcoded_robust.py
# ------------------------------------------------------------
# GLB를 geometry(=drawcall) 단위로 서로 다른 색으로 시각화 (강건 버전)
# - dump()로 월드좌표 메쉬 획득
# - 발광(emissive) 머티리얼로 조명 이슈 제거
# - 카메라 축(Z/Y/X) 3종 시도 후 보이는 뷰 선택
# - double sided로 백페이스 컬링 회피
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
import trimesh
import pyrender
import imageio
import colorsys

# ===== 사용자 설정 (하드코딩) =====
GLB_PATH = Path("/home/vlmap_RCI/Data/habitat_sim/hm3dsem/val/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.semantic.glb")
OFFSCREEN = False  # True면 PNG 저장 후 종료, False면 뷰어 실행
SCREENSHOT_PATH = Path("colored_objects.png")
VIEWPORT_W, VIEWPORT_H = 1600, 1200
FOV_DEG = 60.0

def golden_ratio_color(i: int):
    phi = (1 + 5 ** 0.5) / 2
    h = (i * (1 / phi)) % 1.0
    s = 0.65; v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return np.array([r, g, b, 1.0], dtype=np.float32)

def make_emissive_material(rgba):
    # 발광으로 만들어 조명/노멀 없이도 확실히 보이게
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=rgba.tolist(),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        emissiveFactor=rgba[:3].tolist(),  # <- 여기서 발광
        alphaMode="OPAQUE"
    )

def look_at(eye, target, up):
    f = (target - eye).astype(np.float64)
    f /= (np.linalg.norm(f) + 1e-9)
    u = up.astype(np.float64); u /= (np.linalg.norm(u) + 1e-9)
    s = np.cross(f, u); s /= (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
    m[:3, 3] = eye
    return m

def load_world_meshes(glb_path: Path):
    loaded = trimesh.load(glb_path, force='scene')
    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    # 노드 트랜스폼이 적용된 Trimesh 목록
    meshes = [m for m in loaded.dump() if isinstance(m, trimesh.Trimesh)]
    return meshes

def compute_bounds(meshes):
    mins, maxs = [], []
    for m in meshes:
        if m.vertices.size == 0:
            continue
        bmin, bmax = m.bounds
        mins.append(bmin); maxs.append(bmax)
    if not mins:
        raise RuntimeError("유효한 삼각형 메쉬가 없습니다.")
    mins = np.vstack(mins).min(axis=0)
    maxs = np.vstack(maxs).max(axis=0)
    return mins, maxs

def add_meshes_to_scene(pr_scene, meshes):
    for i, g in enumerate(meshes):
        if not isinstance(g, trimesh.Trimesh) or g.vertices.size == 0 or g.faces.size == 0:
            continue
        color = golden_ratio_color(i)
        mat = make_emissive_material(color)
        # double sided: 백페이스 컬링 회피
        mesh = pyrender.Mesh.from_trimesh(g, material=mat, smooth=False)
        for prim in mesh.primitives:
            if prim.material is not None:
                prim.material.doubleSided = True
        pr_scene.add(mesh)

def build_scene(glb_path: Path):
    meshes = load_world_meshes(glb_path)
    if len(meshes) == 0:
        raise RuntimeError("메쉬를 찾지 못했습니다.")
    mn, mx = compute_bounds(meshes)
    center = (mn + mx) / 2.0
    size = float(np.max(mx - mn) + 1e-9)

    # 장면 생성: ambient_light 올려서 항상 밝게
    pr_scene = pyrender.Scene(bg_color=[0.03, 0.03, 0.03, 1.0],
                              ambient_light=[0.25, 0.25, 0.25])
    add_meshes_to_scene(pr_scene, meshes)

    # 카메라: 여러 축 시도(Z-up, Y-up, X-up 비스듬) → 첫 렌더에서 실제 픽셀 차지 큰 뷰 선택
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(FOV_DEG), znear=0.01, zfar=1e6)
    cam_node = pr_scene.add(cam, pose=np.eye(4))
    candidate_views = []
    # 뷰 포인트 후보들
    views = [
        (np.array([0.0, -1.8, 1.2]), np.array([0.0, 0.0, 1.0])),  # Z-up 사선
        (np.array([0.0,  1.8, 1.2]), np.array([0.0, 1.0, 0.0])),  # Y-up 가정(뒤에서)
        (np.array([ 1.8, 0.0, 1.2]), np.array([0.0, 0.0, 1.0])),  # X→Z 사선
    ]
    # 장면 크기에 맞춰 스케일
    views = [ (center + v[0]*size, v[1]) for v in views ]

    r = pyrender.OffscreenRenderer(VIEWPORT_W, VIEWPORT_H)
    best_score, best_pose, best_img = -1, None, None
    for eye, up in views:
        pose = look_at(eye, center, up)
        pr_scene.set_pose(cam_node, pose=pose)
        color, _ = r.render(pr_scene)
        # "보임" 점수: 이미지의 비검정 픽셀 비율로 평가
        gray = (0.2126*color[...,0] + 0.7152*color[...,1] + 0.0722*color[...,2]).astype(np.uint8)
        score = (gray > 5).mean()  # 0~1
        if score > best_score:
            best_score, best_pose, best_img = score, pose, color
    pr_scene.set_pose(cam_node, pose=best_pose)
    # 디렉셔널 라이트 추가(발광이라 필요 없지만 약간의 입체감용)
    pr_scene.add(pyrender.DirectionalLight(intensity=3000.0), pose=best_pose)

    # 첫 시점이 진짜 보이는지 확인(디버깅용)
    print(f"[Info] initial visible ratio: {best_score:.3f}")

    r.delete()
    return pr_scene

def main():
    assert GLB_PATH.exists(), f"GLB not found: {GLB_PATH}"
    scene = build_scene(GLB_PATH)

    if OFFSCREEN:
        r = pyrender.OffscreenRenderer(viewport_width=VIEWPORT_W, viewport_height=VIEWPORT_H)
        color, _ = r.render(scene)
        imageio.imwrite(SCREENSHOT_PATH, color)
        r.delete()
        print(f"[Saved] {SCREENSHOT_PATH.resolve()}")
    else:
        # Raymond 조명 켜기(발광이라 상관없지만 보조적으로)
        pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False, point_size=2)

if __name__ == "__main__":
    main()
