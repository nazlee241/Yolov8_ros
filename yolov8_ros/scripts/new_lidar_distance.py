#!/usr/bin/env python3
"""
new_lidar_distance.py

LiDAR-only palm-tree (trunk) detector.

Why this exists
----------------
lidar_3d_distance.py only checks that a cluster's centroid height is close
to `target_height`. Frond clumps often average out to roughly the same
height as a trunk, so that single scalar test can't tell a trunk from a
clump of fronds -- tightening the height tolerance to kill frond false
positives also starts killing real trunk detections.

This script instead verifies that a candidate cluster actually LOOKS like a
vertical cylinder (a trunk), using three independent geometric checks:

  1. Verticality  - PCA principal axis of the cluster must point ~straight
                     up. Fronds are wide/lateral -> low verticality.
  2. Radius shape - Points are projected onto the plane perpendicular to the
                     trunk axis; a real trunk has a small, CONSISTENT radius
                     (low std-dev) at that radius. Fronds fan out, giving a
                     large mean radius and/or large radius variance.
  3. Height span  - The cluster must span a minimum vertical extent, since a
                     thin horizontal slab of points isn't a trunk segment.

On top of the shape filter there are two more layers that target the
specific failure mode of "a clump of fronds next to a real trunk gets
detected as a second tree":

  - Temporal confirmation: a candidate must be matched by the tracker for
    several consecutive frames before it is drawn/published. Frond clumps
    are far less geometrically stable frame-to-frame than a rigid trunk.
  - Spacing-based NMS: palm rows are planted at a known, fixed spacing
    (~9 m). Two "trees" detected within a couple of meters of each other in
    the same frame are almost certainly one trunk + one frond clump, not
    two separate trees, so the lower-scoring one is suppressed.

All of the new thresholds are exposed as ROS params so they can be tuned
per-site without touching code.
"""
import os
import threading

import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import SetBool, SetBoolResponse

from bev_occupancy_grid import BevGrid, build_occupancy_grid_msg, render_bev_bgr_image


# 12 edges of an axis-aligned box given corners in this fixed order:
# 0-3 = bottom face (zmin), 4-7 = top face (zmax), matching pillars 0-4,1-5,2-6,3-7
_BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def compute_aabb_corners(cluster_pts):
    """Axis-aligned bounding box corners (8,3) for a set of 3D points."""
    mins = cluster_pts.min(axis=0)
    maxs = cluster_pts.max(axis=0)
    return np.array([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
    ])


def make_bbox_marker(marker_id, cluster_pts, frame_id, stamp, color, lifetime, line_width=0.03):
    """Wireframe LINE_LIST marker outlining the cluster's AABB, in the LiDAR frame."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "trunk_bbox"
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = line_width
    marker.color.r, marker.color.g, marker.color.b = color
    marker.color.a = 1.0
    marker.lifetime = rospy.Duration(lifetime)
    marker.pose.orientation.w = 1.0

    corners = compute_aabb_corners(cluster_pts)
    for a, b in _BBOX_EDGES:
        pa, pb = corners[a], corners[b]
        marker.points.append(Point(x=float(pa[0]), y=float(pa[1]), z=float(pa[2])))
        marker.points.append(Point(x=float(pb[0]), y=float(pb[1]), z=float(pb[2])))
    return marker


def make_label_marker(marker_id, position, text, frame_id, stamp, color, lifetime, text_size=0.4):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "trunk_label"
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.scale.z = text_size
    marker.color.r, marker.color.g, marker.color.b = color
    marker.color.a = 1.0
    marker.lifetime = rospy.Duration(lifetime)
    marker.pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]) + 0.3)
    marker.pose.orientation.w = 1.0
    marker.text = text
    return marker


def make_delete_marker(marker_id, ns, frame_id, stamp):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = marker_id
    marker.action = Marker.DELETE
    return marker


# ==================== Cylinder / Trunk shape analysis ====================
def analyze_cylinder_shape(cluster_pts):
    """
    Fit a vertical-axis cylinder to a set of 3D points via PCA and return
    its shape descriptors, or None if there are too few points to analyze.
    """
    n = len(cluster_pts)
    if n < 3:
        return None

    centroid = np.mean(cluster_pts, axis=0)
    centered = cluster_pts - centroid

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order
    principal_axis = eigvecs[:, -1]
    if principal_axis[2] < 0:
        principal_axis = -principal_axis
    axis_norm = np.linalg.norm(principal_axis)
    if axis_norm < 1e-9:
        return None
    principal_axis = principal_axis / axis_norm

    verticality = abs(principal_axis[2])  # 1.0 == perfectly vertical stem

    z_span = float(cluster_pts[:, 2].max() - cluster_pts[:, 2].min())

    proj_len = centered @ principal_axis
    radial_vec = centered - np.outer(proj_len, principal_axis)
    radial_dist = np.linalg.norm(radial_vec, axis=1)

    return {
        'centroid': centroid,
        'verticality': float(verticality),
        'z_span': z_span,
        'mean_radius': float(np.mean(radial_dist)),
        'radius_std': float(np.std(radial_dist)),
        'num_points': n,
    }


def trunk_score(shape, max_radius_std):
    """Higher == more trunk-like. Used to rank overlapping candidates."""
    radius_term = max(0.0, 1.0 - shape['radius_std'] / max(max_radius_std, 1e-6))
    density_term = min(1.0, shape['num_points'] / 40.0)
    return shape['verticality'] * radius_term * density_term


# ==================== Simple Tracker (with confirmation) ====================
class ObjectTracker:
    def __init__(self, max_age=5, alpha=0.7, association_dist=0.5, confirm_hits=3):
        self.max_age = max_age
        self.alpha = alpha
        self.association_dist = association_dist
        self.confirm_hits = confirm_hits
        self.objects = {}  # id -> {'centroid':, 'age':, 'hits':}
        self.next_id = 0

    def update(self, detections):
        """
        detections: list of dict with key 'centroid' (3D)
        Returns: list of dict with keys 'id', 'centroid', 'hits', 'confirmed'
        """
        if not detections:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['age'] += 1
                if self.objects[obj_id]['age'] > self.max_age:
                    del self.objects[obj_id]
            return []

        matched_ids = set()
        for det in detections:
            best_id = None
            best_dist = float('inf')
            for obj_id, obj in self.objects.items():
                if obj_id in matched_ids:
                    continue
                dist = np.linalg.norm(det['centroid'] - obj['centroid'])
                if dist < self.association_dist and dist < best_dist:
                    best_dist = dist
                    best_id = obj_id

            if best_id is not None:
                obj = self.objects[best_id]
                obj['centroid'] = self.alpha * det['centroid'] + (1 - self.alpha) * obj['centroid']
                obj['age'] = 0
                obj['hits'] = min(obj['hits'] + 1, self.confirm_hits + 5)
                matched_ids.add(best_id)
                det['id'] = best_id
            else:
                det['id'] = self.next_id
                self.next_id += 1
                self.objects[det['id']] = {
                    'centroid': det['centroid'].copy(),
                    'age': 0,
                    'hits': 1,
                }

        for obj_id, obj in self.objects.items():
            if obj_id not in matched_ids:
                obj['age'] += 1
        for obj_id in list(self.objects.keys()):
            if self.objects[obj_id]['age'] > self.max_age:
                del self.objects[obj_id]

        result = []
        for det in detections:
            obj = self.objects[det['id']]
            result.append({
                'id': det['id'],
                'centroid': obj['centroid'],
                'hits': obj['hits'],
                'confirmed': obj['hits'] >= self.confirm_hits,
            })
        return result


# ==================== Main Detector Node ====================
class LidarTrunkDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # ROS topics
        self.image_topic = rospy.get_param("~image_topic", "/zed2i/zed_node/left/image_rect_color")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/zed2i/zed_node/left/camera_info")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/ouster/points")
        self.output_topic = rospy.get_param("~output_topic", "/lidar_overlay/lidar_image")

        # Ground / range filtering
        self.ground_threshold = rospy.get_param("~ground_threshold", 0.1)
        self.max_range = rospy.get_param("~max_range", 20.0)

        # DBSCAN clustering
        self.cluster_eps = rospy.get_param("~cluster_eps", 0.35)
        self.cluster_min_samples = rospy.get_param("~cluster_min_samples", 5)

        # Coarse height gate (kept loose -- shape filter does the real work)
        self.target_height = rospy.get_param("~target_height", 1.5)
        self.height_tolerance = rospy.get_param("~height_tolerance", 1.2)

        # -------- Cylinder shape verification (the core new filter) --------
        self.min_verticality = rospy.get_param("~min_verticality", 0.80)
        self.min_trunk_radius = rospy.get_param("~min_trunk_radius", 0.04)
        self.max_trunk_radius = rospy.get_param("~max_trunk_radius", 0.45)
        self.max_radius_std = rospy.get_param("~max_radius_std", 0.12)
        self.min_height_span = rospy.get_param("~min_height_span", 0.6)
        self.min_points_for_shape = rospy.get_param("~min_points_for_shape", 8)

        # -------- Temporal confirmation --------
        self.tracker_alpha = rospy.get_param("~tracker_alpha", 0.7)
        self.association_dist = rospy.get_param("~association_dist", 0.5)
        self.confirm_hits = rospy.get_param("~confirm_hits", 3)
        self.tracker = ObjectTracker(alpha=self.tracker_alpha,
                                      association_dist=self.association_dist,
                                      confirm_hits=self.confirm_hits)

        # -------- Plantation-spacing prior (row spacing ~9 m) --------
        self.row_spacing = rospy.get_param("~row_spacing", 9.0)
        self.min_tree_spacing = rospy.get_param("~min_tree_spacing", 3.0)  # NMS radius (m)

        # Debug drawing of not-yet-confirmed candidates
        self.draw_unconfirmed = rospy.get_param("~draw_unconfirmed", False)

        # -------- Output toggles --------
        # 2D box drawn on the camera image overlay
        self.enable_2d_bbox = rospy.get_param("~enable_2d_bbox", True)
        # 3D wireframe box published as RViz markers directly in the LiDAR frame
        self.enable_3d_bbox = rospy.get_param("~enable_3d_bbox", True)
        self.show_marker_labels = rospy.get_param("~show_marker_labels", True)
        self.bbox_marker_topic = rospy.get_param("~bbox_marker_topic", "/lidar_overlay/bbox_markers")
        self.marker_lifetime = rospy.get_param("~marker_lifetime", 0.3)  # sec; re-published every frame
        self.confirmed_color = tuple(rospy.get_param("~confirmed_color_rgb", [0.0, 1.0, 0.0]))
        self.unconfirmed_color = tuple(rospy.get_param("~unconfirmed_color_rgb", [1.0, 0.65, 0.0]))

        # -------- Bird's-Eye View (BEV) occupancy grid --------
        # Top-down grid built every frame from the same ground-filtered
        # points/trunk clusters used above. Published as both a standard
        # nav_msgs/OccupancyGrid (RViz "Map" display, costmap-style tools)
        # and a colorized sensor_msgs/Image (rqt_image_view, Foxglove,
        # web_video_server, ...) for viewing without RViz.
        self.enable_bev = rospy.get_param("~enable_bev", True)
        self.bev_size_m = rospy.get_param("~bev_size_m", 30.0)          # grid side length, meters
        self.bev_resolution = rospy.get_param("~bev_resolution", 0.1)   # meters/cell
        self.bev_max_height = rospy.get_param("~bev_max_height", 3.0)   # drop points above this (canopy/wires)
        self.bev_show_generic_points = rospy.get_param("~bev_show_generic_points", True)
        self.bev_show_range_rings = rospy.get_param("~bev_show_range_rings", True)
        self.bev_ring_spacing_m = rospy.get_param("~bev_ring_spacing_m", 5.0)
        self.bev_grid_topic = rospy.get_param("~bev_grid_topic", "/lidar_overlay/bev_occupancy_grid")
        self.bev_image_topic = rospy.get_param("~bev_image_topic", "/lidar_overlay/bev_image")
        self.bev_toggle_service = rospy.get_param("~bev_toggle_service", "~toggle_bev")
        self.bev_grid = BevGrid(self.bev_size_m, self.bev_resolution)

        # Fused-image saving
        self.save_fused_image = rospy.get_param("~save_fused_image", False)
        self.save_fused_image_dir = rospy.get_param("~save_fused_image_dir", "/home/amt4/fastcalib_ws/data/lidar_distance")
        self.save_fused_image_interval_sec = rospy.get_param("~save_fused_image_interval_sec", 5.0)
        self.last_saved_fused_image_time = None
        self.fused_image_counter = 1
        if os.path.isdir(self.save_fused_image_dir):
            existing_files = [f for f in os.listdir(self.save_fused_image_dir) if f.startswith("fuse_image_") and f.endswith(".png")]
            if existing_files:
                numbers = []
                for name in existing_files:
                    stem = name[len("fuse_image_"): -4]
                    try:
                        numbers.append(int(stem))
                    except ValueError:
                        continue
                if numbers:
                    self.fused_image_counter = max(numbers) + 1

        # Extrinsic calibration (LiDAR -> Camera), only used for projection/drawing
        self.Rcl = np.array([
            [-0.080003, -0.996782, 0.005132],
            [0.014233, -0.006290, -0.999879],
            [0.996693, -0.079920, 0.014690]
        ], dtype=np.float64)
        self.tcl = np.array([0.253684, 0.103372, -0.090891], dtype=np.float64)

        self.cache_K = None
        self.cache_imsize = None

        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=5)
        self.marker_pub = rospy.Publisher(self.bbox_marker_topic, MarkerArray, queue_size=5)
        self.prev_marker_ids = set()

        self.bev_grid_pub = rospy.Publisher(self.bev_grid_topic, OccupancyGrid, queue_size=2)
        self.bev_image_pub = rospy.Publisher(self.bev_image_topic, Image, queue_size=2)
        self.bev_toggle_srv = rospy.Service(self.bev_toggle_service, SetBool, self._handle_toggle_bev)

        self.latest_image = None
        self.latest_cam_info = None
        self.latest_pts = None
        self.latest_lidar_header = None
        self.received_data = {'image': False, 'camera_info': False, 'lidar': False}

        rospy.loginfo("Subscribing to:")
        rospy.loginfo(f"  Image: {self.image_topic}")
        rospy.loginfo(f"  Camera info: {self.cam_info_topic}")
        rospy.loginfo(f"  LiDAR: {self.lidar_topic}")

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.cam_info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
        rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)

        self.processing_timer = rospy.Timer(rospy.Duration(0.1), self.process_callback)

        rospy.loginfo("=" * 60)
        rospy.loginfo("LIDAR TRUNK DETECTOR - cylinder-shape verified palm-tree detection")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Ground threshold: {self.ground_threshold} m, max range: {self.max_range} m")
        rospy.loginfo(f"Cluster eps: {self.cluster_eps}, min_samples: {self.cluster_min_samples}")
        rospy.loginfo(f"Coarse height gate: {self.target_height} +/- {self.height_tolerance} m")
        rospy.loginfo(f"Shape filter: verticality>={self.min_verticality}, "
                       f"radius in [{self.min_trunk_radius}, {self.max_trunk_radius}] m, "
                       f"radius_std<={self.max_radius_std} m, height_span>={self.min_height_span} m")
        rospy.loginfo(f"Confirmation: {self.confirm_hits} consecutive matched frames")
        rospy.loginfo(f"Row spacing prior: {self.row_spacing} m, NMS spacing: {self.min_tree_spacing} m")
        rospy.loginfo(f"2D image bbox: {self.enable_2d_bbox}, 3D RViz bbox: {self.enable_3d_bbox} ({self.bbox_marker_topic})")
        rospy.loginfo(f"BEV occupancy grid: {self.enable_bev} ({self.bev_size_m}m @ {self.bev_resolution}m/cell) "
                       f"-> {self.bev_grid_topic}, {self.bev_image_topic}")
        rospy.loginfo(f"Toggle BEV at runtime: rosservice call {self.bev_toggle_service} \"data: false\"")
        rospy.loginfo("=" * 60)

        rospy.Timer(rospy.Duration(5.0), self.status_callback)

    # --- callbacks ---
    def status_callback(self, event):
        confirmed = sum(1 for o in self.tracker.objects.values() if o['hits'] >= self.confirm_hits)
        rospy.loginfo("=== STATUS ===")
        rospy.loginfo(f"Image: {self.received_data['image']}")
        rospy.loginfo(f"Camera Info: {self.received_data['camera_info']}")
        rospy.loginfo(f"LiDAR: {self.received_data['lidar']}")
        rospy.loginfo(f"Tracked objects: {len(self.tracker.objects)} (confirmed: {confirmed})")
        rospy.loginfo("==============")

    def _handle_toggle_bev(self, req):
        """rosservice call <bev_toggle_service> "data: true/false" flips the
        Bird's-Eye View on/off at runtime -- no restart needed."""
        with self.lock:
            self.enable_bev = bool(req.data)
        state = "ENABLED" if self.enable_bev else "DISABLED"
        rospy.loginfo(f"[BEV] Bird's-Eye View {state} via service call")
        return SetBoolResponse(success=True, message=f"BEV occupancy grid {state.lower()}")

    def image_callback(self, msg):
        with self.lock:
            self.latest_image = msg
            self.received_data['image'] = True

    def cam_info_callback(self, msg):
        with self.lock:
            self.latest_cam_info = msg
            self.received_data['camera_info'] = True

    def lidar_callback(self, msg):
        try:
            pts_list = []
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                pts_list.append([p[0], p[1], p[2]])
            if pts_list:
                pts = np.array(pts_list, dtype=np.float32)
                with self.lock:
                    self.latest_pts = pts
                    self.latest_lidar_header = msg.header
                    self.received_data['lidar'] = True
        except Exception as e:
            rospy.logerr(f"LiDAR callback error: {e}")

    def save_fused_image_if_needed(self, img):
        if not self.save_fused_image:
            return
        now = rospy.Time.now().to_sec()
        if self.last_saved_fused_image_time is not None:
            elapsed = now - self.last_saved_fused_image_time
            if elapsed < self.save_fused_image_interval_sec:
                return
        self.last_saved_fused_image_time = now
        os.makedirs(self.save_fused_image_dir, exist_ok=True)
        filename = os.path.join(self.save_fused_image_dir, f"lidar_{self.fused_image_counter}.png")
        self.fused_image_counter += 1
        success = cv2.imwrite(filename, img)
        if success:
            rospy.loginfo(f"Saved fused image to {filename}")
        else:
            rospy.logwarn(f"Failed to save fused image to {filename}")

    # --- candidate filtering pipeline ---
    def _cluster_to_candidate(self, cluster_pts):
        """Apply the coarse height gate + cylinder shape filter. Returns a
        candidate dict or None if the cluster is rejected."""
        centroid = np.mean(cluster_pts, axis=0)
        if abs(centroid[2] - self.target_height) > self.height_tolerance:
            return None

        if len(cluster_pts) < self.min_points_for_shape:
            return None

        shape = analyze_cylinder_shape(cluster_pts)
        if shape is None:
            return None

        if shape['verticality'] < self.min_verticality:
            return None
        if not (self.min_trunk_radius <= shape['mean_radius'] <= self.max_trunk_radius):
            return None
        if shape['radius_std'] > self.max_radius_std:
            return None
        if shape['z_span'] < self.min_height_span:
            return None

        return {
            'centroid': centroid,
            'points': cluster_pts,
            'shape': shape,
            'score': trunk_score(shape, self.max_radius_std),
        }

    def _spacing_nms(self, candidates):
        """Suppress candidates that sit implausibly close (< min_tree_spacing,
        horizontal distance) to a higher-scoring candidate in the same frame.
        Palm rows are planted ~row_spacing apart, so two "trees" a couple of
        meters apart in one frame are almost always trunk + frond-clump of
        the same tree, not two distinct trees."""
        if not candidates:
            return candidates
        ordered = sorted(candidates, key=lambda c: c['score'], reverse=True)
        kept = []
        for cand in ordered:
            xy = cand['centroid'][:2]
            too_close = False
            for k in kept:
                if np.linalg.norm(xy - k['centroid'][:2]) < self.min_tree_spacing:
                    too_close = True
                    break
            if not too_close:
                kept.append(cand)
        return kept

    def _publish_markers(self, visible_objs, id_to_candidate, lidar_header):
        """Publish (or clear) 3D wireframe bounding boxes directly in the
        LiDAR frame, for RViz. Independent of the 2D image overlay toggle."""
        if not self.enable_3d_bbox:
            if self.prev_marker_ids:
                array = MarkerArray()
                stamp = lidar_header.stamp if lidar_header is not None else rospy.Time.now()
                frame_id = lidar_header.frame_id if lidar_header is not None else "os_sensor"
                for obj_id in self.prev_marker_ids:
                    array.markers.append(make_delete_marker(obj_id, "trunk_bbox", frame_id, stamp))
                    array.markers.append(make_delete_marker(obj_id, "trunk_label", frame_id, stamp))
                self.marker_pub.publish(array)
                self.prev_marker_ids = set()
            return

        frame_id = lidar_header.frame_id if lidar_header is not None else "os_sensor"
        stamp = lidar_header.stamp if lidar_header is not None else rospy.Time.now()

        array = MarkerArray()
        current_ids = set()
        for obj in visible_objs:
            obj_id = obj['id']
            cand = id_to_candidate.get(obj_id)
            if cand is None:
                continue
            current_ids.add(obj_id)
            color = self.confirmed_color if obj['confirmed'] else self.unconfirmed_color
            array.markers.append(make_bbox_marker(
                obj_id, cand['points'], frame_id, stamp, color, self.marker_lifetime
            ))
            if self.show_marker_labels:
                distance = np.linalg.norm(obj['centroid'])
                tag = "" if obj['confirmed'] else "?"
                text = f"ID:{obj_id}{tag} {distance:.2f}m r={cand['shape']['mean_radius']:.2f}m"
                array.markers.append(make_label_marker(
                    obj_id, cand['centroid'], text, frame_id, stamp, color, self.marker_lifetime
                ))

        # Clear markers for objects that disappeared since the last frame
        for stale_id in (self.prev_marker_ids - current_ids):
            array.markers.append(make_delete_marker(stale_id, "trunk_bbox", frame_id, stamp))
            array.markers.append(make_delete_marker(stale_id, "trunk_label", frame_id, stamp))

        self.prev_marker_ids = current_ids
        self.marker_pub.publish(array)

    def _publish_bev(self, pts_above, visible, id_to_candidate, lidar_header):
        """Build + publish the top-down occupancy grid (nav_msgs/OccupancyGrid)
        and the colorized BEV snapshot (sensor_msgs/Image). No-op whenever
        BEV is disabled (~enable_bev at startup, or ~toggle_bev at runtime),
        so it costs nothing when the feature isn't wanted."""
        if not self.enable_bev:
            return

        try:
            frame_id = lidar_header.frame_id if lidar_header is not None else "os_sensor"
            stamp = lidar_header.stamp if lidar_header is not None else rospy.Time.now()

            # Keep the grid roughly planar: drop returns well above the
            # sensor (canopy, overhead wires) so they don't clutter the top-down view.
            band = pts_above[pts_above[:, 2] <= self.bev_max_height]
            points_xy = band[:, :2]

            trunk_points_list = []
            trunk_render_list = []
            for obj in visible:
                cand = id_to_candidate.get(obj['id'])
                if cand is None:
                    continue
                trunk_points_list.append(cand['points'][:, :2])
                trunk_render_list.append({
                    'x': float(obj['centroid'][0]),
                    'y': float(obj['centroid'][1]),
                    'radius_m': cand['shape']['mean_radius'],
                    'confirmed': obj['confirmed'],
                    'id': obj['id'],
                    'distance_m': float(np.linalg.norm(obj['centroid'][:2])),
                })

            grid_msg = build_occupancy_grid_msg(self.bev_grid, points_xy, trunk_points_list, frame_id, stamp)
            self.bev_grid_pub.publish(grid_msg)

            bev_img = render_bev_bgr_image(
                self.bev_grid, points_xy, trunk_render_list,
                show_generic_points=self.bev_show_generic_points,
                show_range_rings=self.bev_show_range_rings,
                ring_spacing_m=self.bev_ring_spacing_m,
            )
            bev_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding="bgr8")
            bev_msg.header.frame_id = frame_id
            bev_msg.header.stamp = stamp
            self.bev_image_pub.publish(bev_msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"BEV publish error: {e}")

    def process_callback(self, event):
        with self.lock:
            if not all(self.received_data.values()):
                return
            img_msg = self.latest_image
            cam_info_msg = self.latest_cam_info
            pts = self.latest_pts.copy()
            lidar_header = self.latest_lidar_header

        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            H, W = img.shape[:2]
            if self.cache_K is None or self.cache_imsize != (H, W):
                K = np.array(cam_info_msg.K, dtype=np.float32).reshape(3, 3)
                self.cache_K = K
                self.cache_imsize = (H, W)
                rospy.loginfo(f"Camera matrix:\n{K}")
            else:
                K = self.cache_K
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # Range filter (LiDAR frame)
            ranges = np.linalg.norm(pts, axis=1)
            pts = pts[ranges <= self.max_range]
            if len(pts) == 0:
                return

            # Ground removal
            pts_above = pts[pts[:, 2] > self.ground_threshold]
            if len(pts_above) == 0:
                return

            # DBSCAN clustering
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(pts_above)
            labels = clustering.labels_

            # Coarse height gate + cylinder shape verification
            candidates = []
            for label in set(labels):
                if label == -1:
                    continue
                cluster_pts = pts_above[labels == label]
                cand = self._cluster_to_candidate(cluster_pts)
                if cand is not None:
                    candidates.append(cand)

            # Spacing-prior NMS (suppress frond-clump "second trees")
            candidates = self._spacing_nms(candidates)

            if not candidates:
                self._publish_markers([], {}, lidar_header)
                self._publish_bev(pts_above, [], {}, lidar_header)
                out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                out_msg.header = img_msg.header
                self.pub.publish(out_msg)
                return

            # Update tracker (temporal confirmation)
            centroid_list = [{'centroid': c['centroid']} for c in candidates]
            tracked = self.tracker.update(centroid_list)

            # Map tracked ID -> current-frame candidate (for points/shape)
            id_to_candidate = {}
            for obj in tracked:
                best_idx, best_dist = -1, float('inf')
                for i, cand in enumerate(candidates):
                    dist = np.linalg.norm(obj['centroid'] - cand['centroid'])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx >= 0:
                    id_to_candidate[obj['id']] = candidates[best_idx]

            # Figure out which tracked objects are visible this frame
            visible = [obj for obj in tracked if obj['confirmed'] or self.draw_unconfirmed]

            for obj in visible:
                obj_id = obj['id']
                centroid = obj['centroid']
                distance = np.linalg.norm(centroid)
                cand = id_to_candidate.get(obj_id)
                if cand is None:
                    continue
                shape = cand['shape']

                rospy.loginfo_throttle(
                    2.0,
                    f"Trunk ID:{obj_id} dist={distance:.2f}m vert={shape['verticality']:.2f} "
                    f"r={shape['mean_radius']:.2f}m std={shape['radius_std']:.2f}m "
                    f"zspan={shape['z_span']:.2f}m hits={obj['hits']}"
                )

                # -- 2D image overlay (camera frame) --
                if self.enable_2d_bbox:
                    cluster_pts = cand['points']
                    pts_cam = (cluster_pts @ self.Rcl.T) + self.tcl
                    Z = pts_cam[:, 2]
                    valid = Z > 0.1
                    if np.any(valid):
                        pts_cam_v = pts_cam[valid]
                        Zv = Z[valid]
                        u = (fx * pts_cam_v[:, 0] / Zv + cx).astype(np.int32)
                        v = (fy * pts_cam_v[:, 1] / Zv + cy).astype(np.int32)

                        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                        u = u[in_img]
                        v = v[in_img]
                        if len(u) > 0:
                            x1, y1 = np.min(u), np.min(v)
                            x2, y2 = np.max(u), np.max(v)
                            color = (0, 255, 0) if obj['confirmed'] else (0, 165, 255)  # green / orange
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                            centroid_cam = centroid @ self.Rcl.T + self.tcl
                            if centroid_cam[2] > 0.1:
                                uc = int(fx * centroid_cam[0] / centroid_cam[2] + cx)
                                vc = int(fy * centroid_cam[1] / centroid_cam[2] + cy)
                                tag = "" if obj['confirmed'] else "?"
                                label = f"ID:{obj_id}{tag} {distance:.2f}m r={shape['mean_radius']:.2f}"
                                cv2.putText(img, label, (uc - 30, vc - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # -- 3D wireframe markers (native LiDAR frame, for RViz) --
            self._publish_markers(visible, id_to_candidate, lidar_header)

            # -- Bird's-Eye View occupancy grid + snapshot image --
            self._publish_bev(pts_above, visible, id_to_candidate, lidar_header)

            out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            out_msg.header = img_msg.header
            self.pub.publish(out_msg)
            self.save_fused_image_if_needed(img)

        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
            import traceback
            traceback.print_exc()


def main():
    rospy.init_node("lidar_trunk_detector", anonymous=False)
    rospy.loginfo("Starting LiDAR Trunk Detector (cylinder-shape verified)")
    try:
        node = LidarTrunkDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
