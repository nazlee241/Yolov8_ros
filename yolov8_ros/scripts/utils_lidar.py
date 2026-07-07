"""
utils_lidar.py  ─  LiDAR utilities (ROS1 Noetic compatible)

KEY CHANGE FROM ROS2:
  sensor_msgs_py.point_cloud2  →  sensor_msgs.point_cloud2  (ROS1 built-in)
  The API is identical: pc2.read_points(msg, field_names, skip_nans)
"""

import numpy as np
import cv2

# ROS1 point_cloud2 helper (ships with sensor_msgs in Noetic)
import sensor_msgs.point_cloud2 as pc2


# ── 1. PointCloud2 → numpy XYZ ──────────────────────────────────────────────
def pointcloud2_to_xyz(msg):
    """
    Convert a ROS1 PointCloud2 message to an (N,3) float64 NumPy array.
    Returns an empty (0,3) array when the cloud is empty.
    """
    gen  = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pts  = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float64)
    return pts if pts.size else np.empty((0, 3), dtype=np.float64)


# ── 2. ROI filter (Livox frame) ─────────────────────────────────────────────
def roi_filter(points, roi):
    if len(points) == 0:
        return points
    mask = (
        (points[:, 0] > roi["x"][0]) & (points[:, 0] < roi["x"][1]) &
        (points[:, 1] > roi["y"][0]) & (points[:, 1] < roi["y"][1]) &
        (points[:, 2] > roi["z"][0]) & (points[:, 2] < roi["z"][1])
    )
    return points[mask]


# ── 3. Livox → Camera frame ─────────────────────────────────────────────────
def livox_to_camera(points, T_cam_lidar):
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    N = points.shape[0]
    h = np.hstack([points, np.ones((N, 1), dtype=np.float64)])
    return (T_cam_lidar @ h.T).T[:, :3]


# ── 4. Z-filter ─────────────────────────────────────────────────────────────
def z_filter(points, min_z=0.1):
    if len(points) == 0:
        return points
    return points[points[:, 2] > min_z]


# ── 5. Project camera-frame points → pixel (u, v) ───────────────────────────
def project_points(pts_cam, K):
    """(N,3) → (N,2) float pixel coords."""
    p = (K @ pts_cam.T).T
    z = np.where(np.abs(p[:, 2]) < 1e-6, 1e-6, p[:, 2])
    return np.column_stack([p[:, 0] / z, p[:, 1] / z])


# ══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH — Raw LiDAR-only distance (NO YOLO guidance, NO filtering)
# ══════════════════════════════════════════════════════════════════════════════

def get_raw_lidar_distance(pts_cam, K, yolo_box, img_w, img_h,
                            expand_px=5,
                            max_dist=30.0):
    """
    Compute RAW LiDAR-only distance for a YOLO box region.

    Ground Truth baseline:
      • Projects all LiDAR points → image
      • Keeps points landing inside the YOLO box (same frustum as fusion)
      • Takes the MEAN depth of ALL points — no IQR, no std check, no
        centroid check, no validation whatsoever

    Returns (float, None) on success or (None, str) on failure.
    """
    if len(pts_cam) == 0:
        return None, "no pts_cam"

    x1, y1, x2, y2 = yolo_box[:4]
    uv = project_points(pts_cam, K)

    x1e = max(0,     x1 - expand_px)
    y1e = max(0,     y1 - expand_px)
    x2e = min(img_w, x2 + expand_px)
    y2e = min(img_h, y2 + expand_px)

    inside = (
        (uv[:, 0] >= x1e) & (uv[:, 0] <= x2e) &
        (uv[:, 1] >= y1e) & (uv[:, 1] <= y2e) &
        (pts_cam[:, 2] > 0.1) &
        (pts_cam[:, 2] <= max_dist)
    )
    pts_in = pts_cam[inside]

    if len(pts_in) == 0:
        return None, "no pts in frustum region"

    raw_dist = float(np.mean(pts_in[:, 2]))
    return raw_dist, None


# ══════════════════════════════════════════════════════════════════════════════
# FUSION ENTRY POINT — get validated depth for one YOLO box
# ══════════════════════════════════════════════════════════════════════════════

def get_distance_for_yolo_box(pts_cam, K, yolo_box, img_w, img_h,
                               expand_px=5,
                               min_raw_pts=6,
                               min_clean_pts=4,
                               iqr_factor=1.5,
                               max_depth_std=1.8,
                               max_dist=30.0,
                               centroid_check=True):
    """
    Compute validated median LiDAR depth for one YOLO detection.

    Returns (float, None)  — success
            (None,  str)   — failure reason
    """
    if len(pts_cam) == 0:
        return None, "no pts_cam"

    x1, y1, x2, y2 = yolo_box[:4]
    uv = project_points(pts_cam, K)

    x1e = max(0,     x1 - expand_px)
    y1e = max(0,     y1 - expand_px)
    x2e = min(img_w, x2 + expand_px)
    y2e = min(img_h, y2 + expand_px)

    inside = (
        (uv[:, 0] >= x1e) & (uv[:, 0] <= x2e) &
        (uv[:, 1] >= y1e) & (uv[:, 1] <= y2e)
    )
    pts_in = pts_cam[inside]
    uv_in  = uv[inside]

    if len(pts_in) < min_raw_pts:
        return None, f"only {len(pts_in)} pts in frustum (need {min_raw_pts})"

    # Max-distance filter
    mask_md = pts_in[:, 2] <= max_dist
    pts_md  = pts_in[mask_md]
    uv_md   = uv_in[mask_md]

    if len(pts_md) < min_raw_pts:
        return None, "too few pts after max_dist filter"

    # IQR outlier removal on depth
    d = pts_md[:, 2]
    q1, q3 = np.percentile(d, [25, 75])
    iqr    = q3 - q1
    mask_iq  = (d >= q1 - iqr_factor * iqr) & (d <= q3 + iqr_factor * iqr)
    pts_clean = pts_md[mask_iq]
    uv_clean  = uv_md[mask_iq]

    if len(pts_clean) < min_clean_pts:
        return None, f"after IQR: {len(pts_clean)} pts (need {min_clean_pts})"

    d_std = float(np.std(pts_clean[:, 2]))
    if d_std > max_depth_std:
        return None, f"depth std {d_std:.2f} m > {max_depth_std} m"

    if centroid_check:
        u_c = float(np.mean(uv_clean[:, 0]))
        v_c = float(np.mean(uv_clean[:, 1]))
        if not (x1 <= u_c <= x2 and y1 <= v_c <= y2):
            return None, f"centroid ({u_c:.0f},{v_c:.0f}) outside box"

    dist = float(np.median(pts_clean[:, 2]))
    return dist, None


# ══════════════════════════════════════════════════════════════════════════════
# 3D BOX VIA BACK-PROJECTION
# ══════════════════════════════════════════════════════════════════════════════

def build_3d_box_from_yolo(yolo_box, depth_m, K,
                           box_depth=1.0,
                           shrink_x=0.07,
                           shrink_y_top=0.02,
                           shrink_y_bot=0.10):
    """
    Build a tight 3D cuboid by back-projecting the shrunken YOLO 2D box
    corners at the measured LiDAR depth.

    Returns corners : (8,3) camera frame — front face 0-3, rear face 4-7
    """
    x1, y1, x2, y2 = yolo_box[:4]
    bw = x2 - x1
    bh = y2 - y1

    sx1 = x1 + shrink_x     * bw
    sx2 = x2 - shrink_x     * bw
    sy1 = y1 + shrink_y_top * bh
    sy2 = y2 - shrink_y_bot * bh

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    D      = depth_m

    def bp(u, v):
        return np.array([
            (u - cx) * D / fx,
            (v - cy) * D / fy,
            D
        ], dtype=np.float64)

    f_tl = bp(sx1, sy1)
    f_tr = bp(sx2, sy1)
    f_br = bp(sx2, sy2)
    f_bl = bp(sx1, sy2)

    shift = np.array([0.0, 0.0, box_depth], dtype=np.float64)
    r_tl  = f_tl + shift
    r_tr  = f_tr + shift
    r_br  = f_br + shift
    r_bl  = f_bl + shift

    corners = np.array([
        f_tl, f_tr, f_br, f_bl,
        r_tl, r_tr, r_br, r_bl,
    ], dtype=np.float64)

    return corners


# ── Project 3-D corners → image pixels ──────────────────────────────────────
def project_to_image(points_3d, K):
    return project_points(points_3d, K)


# ── Draw 12-edge cuboid ──────────────────────────────────────────────────────
_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # front face
    (4, 5), (5, 6), (6, 7), (7, 4),   # rear  face
    (0, 4), (1, 5), (2, 6), (3, 7),   # pillars
]


def draw_cuboid(img, pts2d, color=(255, 0, 0), thickness=2):
    pts = pts2d.astype(int)
    for a, b in _EDGES:
        pa = (int(np.clip(pts[a, 0], -10000, 10000)),
              int(np.clip(pts[a, 1], -10000, 10000)))
        pb = (int(np.clip(pts[b, 0], -10000, 10000)),
              int(np.clip(pts[b, 1], -10000, 10000)))
        cv2.line(img, pa, pb, color, thickness, lineType=cv2.LINE_AA)
    return img
