import numpy as np
import cv2
from scipy.spatial.distance import cdist
import pyvisgraph as vg
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Dict
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.ndimage import binary_opening, binary_closing
from skimage.graph import route_through_array

def get_segment_islands_pos(segment_map, label_id, detect_internal_contours=False):
    mask = segment_map == label_id
    mask = mask.astype(np.uint8)
    detect_type = cv2.RETR_EXTERNAL
    if detect_internal_contours:
        detect_type = cv2.RETR_TREE

    contours, hierarchy = cv2.findContours(mask, detect_type, cv2.CHAIN_APPROX_SIMPLE)
    # convert contours back to numpy index order
    contours_list = []
    for contour in contours:
        tmp = contour.reshape((-1, 2))
        tmp_1 = np.stack([tmp[:, 1], tmp[:, 0]], axis=1)
        contours_list.append(tmp_1)

    centers_list = []
    bbox_list = []
    for c in contours_list:
        xmin = np.min(c[:, 0])
        xmax = np.max(c[:, 0])
        ymin = np.min(c[:, 1])
        ymax = np.max(c[:, 1])
        bbox_list.append([xmin, xmax, ymin, ymax])

        centers_list.append([(xmin + xmax) / 2, (ymin + ymax) / 2])

    return contours_list, centers_list, bbox_list, hierarchy


def find_closest_points_between_two_contours(obs_map, contour_a, contour_b):
    a = np.zeros_like(obs_map, dtype=np.uint8)
    b = np.zeros_like(obs_map, dtype=np.uint8)
    cv2.drawContours(a, [contour_a[:, [1, 0]]], 0, 255, 1)
    cv2.drawContours(b, [contour_b[:, [1, 0]]], 0, 255, 1)
    rows_a, cols_a = np.where(a == 255)
    rows_b, cols_b = np.where(b == 255)
    pts_a = np.concatenate([rows_a.reshape((-1, 1)), cols_a.reshape((-1, 1))], axis=1)
    pts_b = np.concatenate([rows_b.reshape((-1, 1)), cols_b.reshape((-1, 1))], axis=1)
    dists = cdist(pts_a, pts_b)
    id = np.argmin(dists)
    ida, idb = np.unravel_index(id, dists.shape)
    return [rows_a[ida], cols_a[ida]], [rows_b[idb], cols_b[idb]]


def point_in_contours(obs_map, contours_list, point):
    """
    obs_map: np.ndarray, 1 free, 0 occupied
    contours_list: a list of cv2 contours [[(col1, row1), (col2, row2), ...], ...]
    point: (row, col)
    """
    row, col = int(point[0]), int(point[1])
    ids = []
    print("contours num: ", len(contours_list))
    for con_i, contour in enumerate(contours_list):
        contour_cv2 = contour[:, [1, 0]]
        con_mask = np.zeros_like(obs_map, dtype=np.uint8)
        cv2.drawContours(con_mask, [contour_cv2], 0, 255, -1)
        # con_mask_copy = con_mask.copy()
        # cv2.circle(con_mask_copy, (col, row), 10, 0, 3)
        # cv2.imshow("contour_mask", con_mask_copy)
        # cv2.waitKey()
        if con_mask[row, col] == 255:
            ids.append(con_i)
    return ids

# def visualize_visgraph(obs_map, graph, show=True, out_path=None):
#     import matplotlib.pyplot as plt

#     obs_map_vis = (obs_map[:, :, None] * 255).astype(np.uint8)
#     obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])

#     plt.figure(figsize=(10, 10))
#     plt.imshow(obs_map_vis)

#     # 올바른 nodes 가져오기
#     nodes = graph.graph.get_points()

#     for node in nodes:
#         plt.scatter(node.y, node.x, color='red', s=5)

#         # 올바른 neighbor 가져오기
#         for neighbor in graph.graph.get_adjacent_points(node):
#             plt.plot([node.y, neighbor.y], [node.x, neighbor.x], color='blue', linewidth=1)

#     plt.title("VISGRAPH")
#     if out_path:
#         plt.savefig(out_path)
#     if show:
#         plt.show()
#     plt.close()



# def build_visgraph_with_obs_map(obs_map, use_internal_contour=False, internal_point=None, vis=False):
#     obs_map_vis = (obs_map[:, :, None] * 255).astype(np.uint8)
#     obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
#     if vis:
#         cv2.imshow("obs", obs_map_vis)
#         cv2.waitKey()

#     contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
#         obs_map, 0, detect_internal_contours=use_internal_contour
#     )

#     if use_internal_contour:
#         ids = point_in_contours(obs_map, contours_list, internal_point)
#         assert len(ids) == 2, f"The internal point is not in 2 contours, but {len(ids)}"
#         point_a, point_b = find_closest_points_between_two_contours(
#             obs_map, contours_list[ids[0]], contours_list[ids[1]]
#         )
#         obs_map = cv2.line((obs_map * 255).astype(np.uint8), (point_a[1], point_a[0]), (point_b[1], point_b[0]), 255, 5)
#         obs_map = obs_map == 255
#         contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
#             obs_map, 0, detect_internal_contours=False
#         )

#     poly_list = []

#     for contour in contours_list:
#         if vis:
#             contour_cv2 = contour[:, [1, 0]]
#             cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)
#             cv2.imshow("obs", obs_map_vis)
#         contour_pos = []
#         for [row, col] in contour:
#             contour_pos.append(vg.Point(row, col))
#         poly_list.append(contour_pos)
#         xlist = [x.x for x in contour_pos]
#         zlist = [x.y for x in contour_pos]
#         if vis:
#             # plt.plot(xlist, zlist)

#             cv2.waitKey()
#     for i, poly in enumerate(poly_list):
#         print(f"Polygon {i}: length = {len(poly)}")
#         if len(poly) < 3:
#             print(f"[WARNING] Polygon {i} too small!")




#     g = vg.VisGraph()
#     # g.build(poly_list, workers=4)
#     try:
#         g.build(poly_list, workers=4)
#         print("[DEBUG] build() completed")
#     except Exception as e:
#         print(f"[ERROR] build() failed: {e}")

#     visualize_visgraph(obs_map, g, show=True, out_path="visgraph.png")
#     return g








from scipy.spatial.distance import cdist

# ─────────────────────────────────────────
# 1) 보조 함수들
# ─────────────────────────────────────────
def signed_area(poly):
    """shoelace 공식 – 음수면 CW, 양수면 CCW"""
    s = 0.0
    for i in range(len(poly) - 1):
        s += (poly[i + 1].x - poly[i].x) * (poly[i + 1].y + poly[i].y)
    return s

def ensure_cw(poly):
    return poly if signed_area(poly) < 0 else list(reversed(poly))

def contour_to_poly(contour_rc, eps=1.5):
    """
    contour_rc : (N,2)  (row, col) 즉 (y, x)
    반환       : CW·닫힌 vg.Point 리스트
    """
    # ── ① 꼭짓점 줄이기
    cnt_cv = contour_rc[:, [1, 0]].astype(np.int32)           # (x,y)
    cnt_cv = cv2.approxPolyDP(cnt_cv, eps, closed=True)

    # ── ② (col,row) → vg.Point(x=col, y=row)
    poly = [vg.Point(int(pt[0][0]), int(pt[0][1])) for pt in cnt_cv]

    # ── ③ 닫힌 다각형 보장
    if poly[0] != poly[-1]:
        poly.append(poly[0])

    # ── ④ 시계방향 강제
    return ensure_cw(poly)

# ─────────────────────────────────────────
# 2) 메인 루틴
# ─────────────────────────────────────────
def build_visgraph_with_obs_map(
        obs_map: np.ndarray,
        use_internal_contour=False,
        internal_point=None,
        vis=False,
        eps=1.5,
):
    """
    obs_map : 1 = free, 0 = obstacle
    """

    contours_list, _, _, _ = get_segment_islands_pos(
        obs_map, 0, detect_internal_contours=use_internal_contour
    )

    # 내부-통로를 뚫는 옵션은 그대로 두고, 이후 contour_list 최신화만 사용
    # …

    # ── 컨투어 → 다각형 변환
    poly_list = [contour_to_poly(c, eps) for c in contours_list
                 if len(c) >= 3]

    # ── visgraph 생성
    g = vg.VisGraph()
    g.build(poly_list, workers=4)

    # 시각화 함수는 node.x, node.y 순서로 바꾸면 끝
    # visualize_visgraph(obs_map, g)

    return g

# ─────────────────────────────────────────
# 3) 시각화 (scatter 만 교정)
# ─────────────────────────────────────────
def visualize_visgraph(obs_map, graph, show=True, out_path=None):
    import matplotlib.pyplot as plt

    vis = np.tile((obs_map[..., None] * 255).astype(np.uint8), [1, 1, 3])

    plt.figure(figsize=(8, 8))
    plt.imshow(vis)

    for p in graph.graph.get_points():
        plt.scatter(p.x, p.y, s=5, c='r')          # ← x, y 순서 교정
        for q in graph.graph.get_adjacent_points(p):
            plt.plot([p.x, q.x], [p.y, q.y], c='b', lw=1)

    if out_path:
        plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()













# def build_visgraph_with_obs_map(obs_map, use_internal_contour=False, internal_point=None, vis=False):
#     obs_map_vis = (obs_map[:, :, None] * 255).astype(np.uint8)
#     obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
#     if vis:
#         cv2.imshow("obs", obs_map_vis)
#         cv2.waitKey()

#     contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
#         obs_map, 0, detect_internal_contours=use_internal_contour
#     )

#     if use_internal_contour:
#         ids = point_in_contours(obs_map, contours_list, internal_point)
#         assert len(ids) == 2, f"The internal point is not in 2 contours, but {len(ids)}"
#         point_a, point_b = find_closest_points_between_two_contours(
#             obs_map, contours_list[ids[0]], contours_list[ids[1]]
#         )
#         obs_map = cv2.line((obs_map * 255).astype(np.uint8), (point_a[1], point_a[0]), (point_b[1], point_b[0]), 255, 5)
#         obs_map = obs_map == 255
#         contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
#             obs_map, 0, detect_internal_contours=False
#         )


#     poly_list = []

#     for contour in contours_list:

#         if len(contour) < 20:  # polygon으로 불가능
#             continue
#         if vis:
#             contour_cv2 = contour[:, [1, 0]]
#             cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)
#             cv2.imshow("obs", obs_map_vis)
#         contour_pos = []
#         for [row, col] in contour:
#             contour_pos.append(vg.Point(row, col))
#         poly_list.append(contour_pos)
#         xlist = [x.x for x in contour_pos]
#         zlist = [x.y for x in contour_pos]
#         if vis:
#             # plt.plot(xlist, zlist)
#             cv2.waitKey()
#     g = vg.VisGraph()
#     g.build(poly_list, workers=1)
#     return g





# import cv2, numpy as np, pyvisgraph as vg
# from shapely.geometry import Polygon

# def build_visgraph_with_obs_map(obs_map,
#                                 use_internal_contour=False,
#                                 internal_point=None,
#                                 vis=False):
#     """
#     obs_map : 2-D numpy, 1=FREE, 0=OBSTACLE(＋바깥 void)
#     반환     : pyvisgraph.VisGraph
#     """
#     # ---------- 디버그용 시각화 베이스 ----------
#     if vis:
#         vis_img = np.dstack([obs_map*255]*3).astype(np.uint8)

#     # ---------- (선택) 내부 contour 연결선 ----------
#     if use_internal_contour and internal_point is not None:
#         ids = point_in_contours(obs_map, *get_segment_islands_pos(obs_map, 0)[:3], internal_point)
#         if len(ids) == 2:
#             pa, pb = find_closest_points_between_two_contours(obs_map, ids[0], ids[1])
#             obs_map = cv2.line(obs_map.copy().astype(np.uint8)*255,
#                                (pa[1], pa[0]), (pb[1], pb[0]), 255, 5) == 255

#     # ---------- 0 픽셀 컨투어 모두 추출 ----------
#     mode = cv2.RETR_TREE if use_internal_contour else cv2.RETR_EXTERNAL
#     contours, _ = cv2.findContours((obs_map == 0).astype(np.uint8),
#                                    mode, cv2.CHAIN_APPROX_NONE)

#     if len(contours) == 0:
#         raise RuntimeError("No obstacle contours found!")

#     # ---------- 바깥 경계 = 면적 최대 컨투어 ----------
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     outer = contours[0].squeeze()         # (N,2) col,row

#     # CW로 뒤집기 (pyvisgraph 요구)
#     outer_poly = Polygon([(c, r) for c, r in outer])
#     if outer_poly.exterior.is_ccw:
#         outer = outer[::-1]

#     poly_list = [[vg.Point(r, c) for c, r in outer]]  # 첫 원소 = 외벽

#     # ---------- 내부 장애물 컨투어 ----------
#     for cnt in contours[1:]:
#         if len(cnt) < 6 or cv2.contourArea(cnt) < 5:
#             continue
#         pts = cnt.squeeze()
#         poly_list.append([vg.Point(r, c) for c, r in pts])  # CCW 그대로

#         if vis:
#             cv2.drawContours(vis_img, [cnt], -1, (255, 0, 0), 1)

#     if vis:
#         cv2.drawContours(vis_img, [outer], -1, (0, 0, 255), 2)
#         cv2.imshow("outer(red) inner(blue)", vis_img); cv2.waitKey()

#     # ---------- VisGraph 생성 ----------
#     G = vg.VisGraph()
#     G.build(poly_list, workers=1)
#     return G







# def build_visgraph_with_obs_map(obs_map, use_internal_contour=True, internal_point=None, vis=False):
#     obs_map_vis = (obs_map[:, :, None] * 255).astype(np.uint8)
#     obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
#     if vis:
#         cv2.imshow("obs", obs_map_vis)
#         cv2.waitKey()

#     # 첫 번째 contour 추출
#     contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
#         obs_map, 0, detect_internal_contours=use_internal_contour
#     )
#     # 내부 contour 연결선 추가
#     if use_internal_contour:
#         ids = point_in_contours(obs_map, contours_list, internal_point)
#         assert len(ids) == 2, f"The internal point is not in 2 contours, but {len(ids)}"
#         point_a, point_b = find_closest_points_between_two_contours(
#             obs_map, contours_list[ids[0]], contours_list[ids[1]]
#         )
#         obs_map = cv2.line((obs_map * 255).astype(np.uint8), (point_a[1], point_a[0]), (point_b[1], point_b[0]), 255, 5)
#         obs_map = obs_map == 255
#         contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
#             obs_map, 0, detect_internal_contours=False
#         )

#     # Shapely로 병합
#     shapely_polygons = []
#     for c in contours_list:
#         if len(c) >= 3:
#             poly = Polygon(c)
#             if poly.is_valid and poly.area > 5:#10:  # 작은 self-intersecting 폴리곤 제거
#                 shapely_polygons.append(poly)

#     merged = unary_union(shapely_polygons)
#     # Ensure merged is iterable
#     if isinstance(merged, Polygon):
#         merged = [merged]
#     elif isinstance(merged, MultiPolygon):
#         merged = list(merged.geoms)  # <-- 여기 수정!
#     elif isinstance(merged, GeometryCollection):          # ★ 추가
#         # 컬렉션 안에 있는 유효한 Polygon만 추출
#         merged = [g for g in merged.geoms if isinstance(g, Polygon) and g.is_valid]
#     else:
#         raise ValueError(f"Unexpected geometry type: {type(merged)}")
    
#     poly_list = []
#     for poly in merged:
#         if poly.area < 2:#10:
#             continue
#         coords = np.array(poly.exterior.coords).astype(int)
#         if len(coords) < 5:#20:
#             continue
#         contour_pos = [vg.Point(p[0], p[1]) for p in coords]  # row=y, col=x
#         poly_list.append(contour_pos)

#         if vis:
#             for pt in coords:
#                 cv2.circle(obs_map_vis, (pt[1], pt[0]), 1, (0, 255, 0), -1)
#             cv2.drawContours(obs_map_vis, [coords[:, [1, 0]]], -1, (255, 0, 0), 2)
#             cv2.imshow("merged", obs_map_vis)
#             cv2.waitKey()

#     # Visibility Graph 생성
#     g = vg.VisGraph()
#     g.build(poly_list, workers=1)
#     return g




import pyvisgraph as vg
import pyvisgraph.visible_vertices as vv
from math import sqrt, acos, pi
import numpy as np
import cv2
# 🧠 방어용 monkey patch (ZeroDivision 방지)
def _angle2_safe(p1, p2, p3):
    a = (p2.x - p1.x)**2 + (p2.y - p1.y)**2
    b = (p2.x - p3.x)**2 + (p2.y - p3.y)**2
    c = (p3.x - p1.x)**2 + (p3.y - p1.y)**2
    if a == 0 or c == 0:
        return pi  # 180도로 간주
    cos_value = (a + c - b) / (2 * sqrt(a) * sqrt(c))
    cos_value = min(1, max(-1, cos_value))  # 수치 오차 보정
    return acos(cos_value)
vv.angle2 = _angle2_safe


# 👇 중복 점 제거 함수
def clean_coords(coords, eps=1e-3):
    uniq = []
    for pt in coords:
        if not uniq or np.linalg.norm(np.array(pt) - np.array(uniq[-1])) > eps:
            uniq.append(tuple(pt))
    if len(uniq) > 2 and np.linalg.norm(np.array(uniq[0]) - np.array(uniq[-1])) < eps:
        uniq.pop()
    return uniq


# 👇 메인 함수
def build_visgraph_with_obs_map33(obs_map, use_internal_contour=True, internal_point=None, vis=False):
    obs_map_vis = (obs_map[:, :, None] * 255).astype(np.uint8)
    obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
    print(f"ouse_internal_contour: {use_internal_contour}")
    if vis:
        cv2.imshow("obs", obs_map_vis)
        cv2.waitKey()

    contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
        obs_map, 0, detect_internal_contours=use_internal_contour
    )

    if use_internal_contour:
        ids = point_in_contours(obs_map, contours_list, internal_point)
        assert len(ids) == 2, f"The internal point is not in 2 contours, but {len(ids)}"
        point_a, point_b = find_closest_points_between_two_contours(
            obs_map, contours_list[ids[0]], contours_list[ids[1]]
        )
        obs_map = cv2.line((obs_map * 255).astype(np.uint8), (point_a[1], point_a[0]), (point_b[1], point_b[0]), 255, 5)
        obs_map = obs_map == 255
        contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
            obs_map, 0, detect_internal_contours=False
        )

    poly_list = []
    for contour in contours_list:
        if len(contour) < 20:
            continue

        # ✅ 좌표 정제 (중복 점 제거)
        cleaned = clean_coords([[row, col] for [row, col] in contour])

        contour_pos = [vg.Point(row, col) for [row, col] in cleaned]
        poly_list.append(contour_pos)

        if vis:
            contour_cv2 = np.array([[c[1], c[0]] for c in cleaned], dtype=np.int32)
            cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)
            cv2.imshow("obs", obs_map_vis)
            cv2.waitKey()

    g = vg.VisGraph()
    g.build(poly_list, workers=1)
    return g





def get_nearby_position(goal: Tuple[float, float], G: vg.VisGraph) -> Tuple[float, float]:
    for dr, dc in zip([-1, 1, -1, 1], [-1, -1, 1, 1]):
        goalvg_new = vg.Point(goal[0] + dr, goal[1] + dc)
        poly_id_new = G.point_in_polygon(goalvg_new)
        if poly_id_new == -1:
            return (goal[0] + dr, goal[1] + dc)







import numpy as np, cv2
from skimage.graph import route_through_array
from skimage.draw import line         # Bresenham
from scipy.ndimage import distance_transform_edt, label


# ─────────────────────────────────────────
# 0. 유틸
# ─────────────────────────────────────────
def snap_to_free(mask, p):
    r, c = map(int, p)
    if mask[r, c]:
        return (r, c)
    dist, (rr, cc) = distance_transform_edt(mask == 0, return_indices=True)
    return (int(rr[r, c]), int(cc[r, c]))

def line_of_sight(mask, a, b):
    rr, cc = line(a[0], a[1], b[0], b[1])
    return np.all(mask[rr, cc])

def inflate_obstacles(free_mask: np.ndarray, radius_px: int):
    """free_mask: 1=free,0=obs → 로봇 반경만큼 장애물 팽창 후 free=1·obs=0 반환"""
    if radius_px <= 0:
        return free_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (radius_px*2+1, radius_px*2+1))
    obs = 1 - free_mask                # 1=obs,0=free 로 뒤집기
    obs_dil = cv2.dilate(obs, kernel)
    return 1 - obs_dil                 # 다시 free=1, obs=0

# ─────────────────────────────────────────
# 1. 메인
# ─────────────────────────────────────────
def plan_to_pos_v3(start, goal, obstacles, vis=False,
                     diag=True, simplify=True, robot_radius_px=10): #!#!#!#!#!#!#!#!#!
    """
    obstacles        : 0=obs, 1=free  (H,W) np.ndarray
    robot_radius_px  : 픽셀 단위 로봇 반경 (안전거리 포함)
    반환             : 코너 포인트 리스트 [(row,col), …]
    """
    # ① 입력 정제 + 로봇 반경 반영
    base_free = obstacles.astype(np.uint8)
    free = inflate_obstacles(base_free, robot_radius_px)

    # ② 시작·목표 스냅
    start = snap_to_free(free, start)
    goal  = snap_to_free(free, goal)

    # ③ 연결 성분 검사
    conn = np.ones((3,3), np.uint8) if diag else np.array([[0,1,0],
                                                           [1,1,1],
                                                           [0,1,0]],np.uint8)
    lbl,_ = label(free, structure=conn)
    if lbl[start] != lbl[goal]:
        raise ValueError("로봇 반경을 고려하면 start와 goal이 연결돼 있지 않습니다.")

    # ④ A* (skimage)
    cost = np.where(free==1, 1.0, np.inf)
    idx,_ = route_through_array(cost, start, goal,
                                fully_connected=diag, geometric=diag)
    path = [(int(r),int(c)) for r,c in idx]

    # ⑤ 직선 구간 압축
    if simplify and len(path) >= 2:
        simp = [path[0]]
        for j in range(2, len(path)):
            if not line_of_sight(free, simp[-1], path[j]):
                simp.append(path[j-1])
        simp.append(path[-1])
        path = simp

    # ⑥ 시각화(옵션)
    if vis:
        vis_img = np.dstack([base_free*255]*3).astype(np.uint8)      # 원본(팽창 전) 표시
        # 경로
        for k,(r,c) in enumerate(path):
            pt = (c,r)
            cv2.circle(vis_img, pt, 5, (255,0,0), -1)
            if k: cv2.line(vis_img, last_pt, pt, (255,0,0), 2)
            last_pt = pt
        cv2.circle(vis_img, (start[1],start[0]), 5, (0,255,0), -1)
        cv2.circle(vis_img, (goal[1], goal[0]),  5, (0,0,255), -1)
        cv2.imshow(f"planned path (r={robot_radius_px}px)", vis_img)
        cv2.waitKey(0)#; cv2.destroyWindow(f"planned path (r={robot_radius_px}px)")

    return path
    # """
    # 반환: [(row,col), …] ─ 연속 직선으로 이동 가능한 코너 포인트 리스트
    # """
    # # ① 입력 정제
    # free = obstacles.astype(np.uint8)
    # start = snap_to_free(free, start)
    # goal  = snap_to_free(free, goal)

    # conn = np.ones((3,3), np.uint8) if diag else np.array([[0,1,0],
    #                                                        [1,1,1],
    #                                                        [0,1,0]],np.uint8)
    # lbl,_ = label(free, structure=conn)
    # if lbl[start] != lbl[goal]:
    #     raise ValueError("start와 goal이 같은 자유 공간에 연결돼 있지 않습니다.")

    # cost = np.where(free==1, 1.0, np.inf)
    # idx,_ = route_through_array(cost, start, goal,
    #                             fully_connected=diag, geometric=diag)
    # path = [(int(r),int(c)) for r,c in idx]          # 전체 격자 경로

    # # ②  직선 가능 구간으로 압축
    # if simplify and len(path) >= 2:
    #     simp = [path[0]]
    #     i = 0
    #     for j in range(2, len(path)):
    #         if not line_of_sight(free, simp[-1], path[j]):   # 시야 끊기면
    #             simp.append(path[j-1])                       # 직전 점이 코너
    #     simp.append(path[-1])
    #     path = simp

    # # ③ 시각화 (옵션)
    # if vis:
    #     vis_img = np.dstack([free*255]*3).astype(np.uint8)
    #     for k,(r,c) in enumerate(path):
    #         pt = (c,r)
    #         cv2.circle(vis_img, pt, 5, (255,0,0), -1)
    #         if k: cv2.line(vis_img, last_pt, pt, (255,0,0), 2)
    #         last_pt = pt
    #     cv2.circle(vis_img, (start[1],start[0]), 5, (0,255,0), -1)
    #     cv2.circle(vis_img, (goal[1], goal[0]),  5, (0,0,255), -1)
    #     cv2.imshow("planned path", vis_img)#; cv2.waitKey(0); cv2.destroyAllWindows()
    #     cv2.waitKey()

    # return path





def plan_to_pos_v2(start, goal, obstacles, G: vg.VisGraph = None, vis=False):
    """
    plan a path on a cropped obstacles map represented by a graph.
    Start and goal are tuples of (row, col) in the map.
    """

    print("start: ", start)
    print("goal: ", goal)
    print(obstacles.shape)
    if vis:
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 3, (255, 0, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 3, (0, 0, 255), -1)
        cv2.imshow("planned path", obs_map_vis)
        cv2.waitKey()

    path = []
    startvg = vg.Point(start[0], start[1])
    if obstacles[int(start[0]), int(start[1])] == 0:
        print("start in obstacles")
        rows, cols = np.where(obstacles == 1)
        dist_sq = (rows - start[0]) ** 2 + (cols - start[1]) ** 2
        id = np.argmin(dist_sq)
        new_start = [rows[id], cols[id]]
        path.append(new_start)
        startvg = vg.Point(new_start[0], new_start[1])

    goalvg = vg.Point(goal[0], goal[1])
    poly_id = G.point_in_polygon(goalvg)
    if obstacles[int(goal[0]), int(goal[1])] == 0:
        print("goal in obstacles")
        try:
            goalvg = G.closest_point(goalvg, poly_id, length=1)
        except:
            goal_new = get_nearby_position(goal, G)
            goalvg = vg.Point(goal_new[0], goal_new[1])

        print("goalvg: ", goalvg)
    path_vg = G.shortest_path(startvg, goalvg)

    for point in path_vg:
        subgoal = [point.x, point.y]
        path.append(subgoal)
    print(path)

    # check the final goal is not in obstacles
    # if obstacles[int(goal[0]), int(goal[1])] == 0:
    #     path = path[:-1]

    if vis:
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])

        for i, point in enumerate(path):
            subgoal = (int(point[1]), int(point[0]))
            print(i, subgoal)
            obs_map_vis = cv2.circle(obs_map_vis, subgoal, 5, (255, 0, 0), -1)
            if i > 0:
                cv2.line(obs_map_vis, last_subgoal, subgoal, (255, 0, 0), 2)
            last_subgoal = subgoal
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 5, (0, 255, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 5, (0, 0, 255), -1)

        seg = Image.fromarray(obs_map_vis)
        cv2.imshow("planned path", obs_map_vis)
        cv2.waitKey()

    return path



def get_bbox(center, size):
    """
    Return min corner and max corner coordinate
    """
    min_corner = center - size / 2
    max_corner = center + size / 2
    return min_corner, max_corner


def get_dist_to_bbox_2d(center, size, pos):
    min_corner_2d, max_corner_2d = get_bbox(center, size)

    dx = pos[0] - center[0]
    dy = pos[1] - center[1]

    if pos[0] < min_corner_2d[0] or pos[0] > max_corner_2d[0]:
        if pos[1] < min_corner_2d[1] or pos[1] > max_corner_2d[1]:
            """
            star region
            *  |  |  *
            ___|__|___
               |  |
            ___|__|___
               |  |
            *  |  |  *
            """

            dx_c = np.abs(dx) - size[0] / 2
            dy_c = np.abs(dy) - size[1] / 2
            dist = np.sqrt(dx_c * dx_c + dy_c * dy_c)
            return dist
        else:
            """
            star region
               |  |
            ___|__|___
            *  |  |  *
            ___|__|___
               |  |
               |  |
            """
            dx_b = np.abs(dx) - size[0] / 2
            return dx_b
    else:
        if pos[1] < min_corner_2d[1] or pos[1] > max_corner_2d[1]:
            """
            star region
               |* |
            ___|__|___
               |  |
            ___|__|___
               |* |
               |  |
            """
            dy_b = np.abs(dy) - size[1] / 2
            return dy_b

        """
        star region
           |  |  
        ___|__|___
           |* |   
        ___|__|___
           |  |   
           |  |  
        """
        return 0
