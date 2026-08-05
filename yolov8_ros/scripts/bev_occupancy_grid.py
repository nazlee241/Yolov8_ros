#!/usr/bin/env python3
"""
bev_occupancy_grid.py

Bird's-Eye View (BEV) occupancy grid generation for the LiDAR trunk
detector (see new_lidar_distance.py).

Every frame this builds two views of the same square, sensor-centered
top-down grid:

  1. nav_msgs/OccupancyGrid - the standard ROS map message. Drop it
     straight into RViz's "Map" display (or any nav-stack/costmap tool)
     with no extra glue code.
  2. A colorized sensor_msgs/Image (BGR) - a friendlier bird's-eye render
     with range rings, a sensor marker, and the tracked trunks drawn as
     colored circles with ID/distance labels. Meant for viewing outside
     RViz -- rqt_image_view, Foxglove Studio, a browser via web_video_server,
     or anything else that can display a plain image topic.

Both are single-frame snapshots (not accumulated maps): cells with no
LiDAR return this frame are left "free" (0) rather than "unknown" (-1),
since there is no persistent map state being built up over time.

Note the two outputs intentionally use different pixel conventions:
  - The OccupancyGrid follows the standard ROS map layout (row-major,
    x -> column, y -> row, origin at the grid's bottom-left corner) so any
    OccupancyGrid-aware tool interprets it correctly.
  - The rendered image instead puts the sensor at the image center with
    +x (forward) pointing up and +y (left) pointing left, i.e. a
    conventional "looking down on a vehicle driving up the page" layout,
    which is far more intuitive for a human glancing at a snapshot.
"""
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid


# OccupancyGrid cell values (0-100 per the message spec).
OCC_GENERIC = 60   # non-trunk obstacle return
OCC_TRUNK = 100    # cell belonging to a tracked trunk cluster

# BGR colors for the rendered image.
_BGR_BG = (24, 24, 24)
_BGR_GENERIC_PT = (90, 90, 90)
_BGR_RING = (70, 70, 70)
_BGR_SENSOR = (255, 255, 255)
_BGR_TEXT = (0, 255, 255)
_BGR_CONFIRMED = (60, 220, 60)
_BGR_UNCONFIRMED = (0, 165, 255)


class BevGrid:
    """Geometry shared by both BEV outputs: a square grid of `dim` x `dim`
    cells, `resolution` meters/cell, centered on the sensor origin."""

    def __init__(self, size_m, resolution):
        self.size_m = float(size_m)
        self.resolution = float(max(resolution, 1e-3))
        self.dim = max(1, int(round(self.size_m / self.resolution)))
        self.half = self.size_m / 2.0

    def world_to_map_cell(self, x, y):
        """ROS map convention: x -> column, y -> row, origin bottom-left.
        Vectorized; x/y may be scalars or arrays."""
        col = np.floor((np.asarray(x) + self.half) / self.resolution).astype(np.int64)
        row = np.floor((np.asarray(y) + self.half) / self.resolution).astype(np.int64)
        return row, col

    def in_bounds(self, row, col):
        return (row >= 0) & (row < self.dim) & (col >= 0) & (col < self.dim)


def build_occupancy_grid_msg(bev, points_xy, trunk_points_list, frame_id, stamp):
    """
    points_xy: (N,2) generic (already ground-filtered) obstacle points.
    trunk_points_list: list of (M,2) point arrays, one per tracked trunk
        cluster; stamped at a higher value so trunks stand out from
        background clutter in any OccupancyGrid viewer.
    Returns a populated nav_msgs/OccupancyGrid.
    """
    grid = np.zeros((bev.dim, bev.dim), dtype=np.int8)

    if len(points_xy):
        row, col = bev.world_to_map_cell(points_xy[:, 0], points_xy[:, 1])
        valid = bev.in_bounds(row, col)
        grid[row[valid], col[valid]] = OCC_GENERIC

    for tp in trunk_points_list:
        if len(tp) == 0:
            continue
        row, col = bev.world_to_map_cell(tp[:, 0], tp[:, 1])
        valid = bev.in_bounds(row, col)
        grid[row[valid], col[valid]] = OCC_TRUNK

    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.info.map_load_time = stamp
    msg.info.resolution = bev.resolution
    msg.info.width = bev.dim
    msg.info.height = bev.dim
    msg.info.origin.position.x = -bev.half
    msg.info.origin.position.y = -bev.half
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.reshape(-1).tolist()
    return msg


def render_bev_bgr_image(bev, points_xy, trunks, show_generic_points=True,
                          show_range_rings=True, ring_spacing_m=5.0):
    """
    trunks: list of dicts with keys x, y, radius_m, confirmed, id, distance_m
        (all in the LiDAR frame / meters).
    Returns a (dim, dim, 3) uint8 BGR image, sensor at the image center.
    """
    dim = bev.dim
    img = np.full((dim, dim, 3), _BGR_BG, dtype=np.uint8)
    cx = cy = dim // 2
    res = bev.resolution

    def to_px(x, y):
        col = int(round(cx - y / res))
        row = int(round(cy - x / res))
        return col, row

    if show_range_rings:
        ring = ring_spacing_m
        while ring <= bev.half:
            radius_px = int(round(ring / res))
            cv2.circle(img, (cx, cy), radius_px, _BGR_RING, 1, lineType=cv2.LINE_AA)
            cv2.putText(img, f"{ring:.0f}m", (cx + 4, max(0, cy - radius_px + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, _BGR_RING, 1, cv2.LINE_AA)
            ring += ring_spacing_m
        cv2.line(img, (cx, 0), (cx, dim - 1), _BGR_RING, 1, cv2.LINE_AA)
        cv2.line(img, (0, cy), (dim - 1, cy), _BGR_RING, 1, cv2.LINE_AA)

    if show_generic_points and len(points_xy):
        cols = (cx - points_xy[:, 1] / res).astype(np.int64)
        rows = (cy - points_xy[:, 0] / res).astype(np.int64)
        valid = (rows >= 0) & (rows < dim) & (cols >= 0) & (cols < dim)
        img[rows[valid], cols[valid]] = _BGR_GENERIC_PT

    for t in trunks:
        px, py = to_px(t['x'], t['y'])
        if not (0 <= px < dim and 0 <= py < dim):
            continue
        radius_px = max(2, int(round(t.get('radius_m', 0.15) / res)))
        color = _BGR_CONFIRMED if t.get('confirmed') else _BGR_UNCONFIRMED
        cv2.circle(img, (px, py), radius_px, color, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), radius_px, (255, 255, 255), 1, cv2.LINE_AA)
        tag = "" if t.get('confirmed') else "?"
        label = f"ID:{t['id']}{tag} {t['distance_m']:.1f}m"
        cv2.putText(img, label, (px + radius_px + 3, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _BGR_TEXT, 1, cv2.LINE_AA)

    cv2.drawMarker(img, (cx, cy), _BGR_SENSOR, cv2.MARKER_TRIANGLE_UP, 10, 2)
    return img
