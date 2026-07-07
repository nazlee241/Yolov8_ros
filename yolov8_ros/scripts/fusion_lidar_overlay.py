#!/usr/bin/env python3
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import threading
from ultralytics import YOLO
import torch
import os
# os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

# ---------- [NEW] Imports for LiDAR processing ----------
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
# ---------------------------------------------------------

class FusionLidarImageOverlay:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # --- Existing Parameters ---
        self.image_topic = rospy.get_param("~image_topic", "/zed2i/zed_node/left/image_rect_color")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/zed2i/zed_node/left/camera_info")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/ouster/points")
        self.output_topic = rospy.get_param("~output_topic", "/lidar_overlay/fused_image")
        
        # YOLO parameters
        self.yolo_model_path = rospy.get_param("~yolo_model", "/home/amt4/yolov8_ros/src/Yolov8_ros/yolov8_ros/weights/palmtree2.pt")
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.5)
        self.target_classes = rospy.get_param("~target_classes", [0])  # 0 = palmtree
        
        # Performance
        self.max_points_to_draw = rospy.get_param("~max_points", 3000)
        self.downsample_rate = rospy.get_param("~downsample_rate", 4)
        self.frame_skip = rospy.get_param("~frame_skip", 1)
        self.point_size = rospy.get_param("~point_size", 3)
        self.min_depth = rospy.get_param("~min_depth", 0.5)
        self.max_depth = rospy.get_param("~max_depth", 40.0)
        
        # ---- [NEW] Core Configuration Flags ----
        self.enable_2d_detection = rospy.get_param("~enable_2d_detection", True) #True
        rospy.loginfo(f"2D Detection (YOLO) Enabled: {self.enable_2d_detection}")
        
        # ---- [NEW] LiDAR Trunk Detection Params ----
        self.cluster_eps = rospy.get_param("~cluster_eps", 0.5)
        self.cluster_min_samples = rospy.get_param("~cluster_min_samples", 20)  #10
        self.ransac_iterations = rospy.get_param("~ransac_iterations", 200) #150
        self.cylinder_inlier_thresh = rospy.get_param("~cylinder_inlier_thresh", 0.20) #0.05
        self.min_cluster_height = rospy.get_param("~min_cluster_height", 0.5)
        
        # ---- [NEW] Row Alignment Params ----
        self.row_spacing = rospy.get_param("~row_spacing", 8.6)
        self.row_orientation = np.radians(rospy.get_param("~row_orientation_deg", 0.0))
        self.fill_gap_threshold = rospy.get_param("~fill_gap_threshold", 1.3)
        
        # ---- [NEW] Fusion Association Params ----
        self.iou_threshold = rospy.get_param("~iou_threshold", 0.1)
        self.bbox_w_ratio = rospy.get_param("~trunk_bbox_width_ratio", 0.4)
        self.bbox_h_ratio = rospy.get_param("~trunk_bbox_height_ratio", 1.5)
        
        # --- Existing Data Storage ---
        self.latest_image = None
        self.latest_cam_info = None
        self.latest_pts = None
        self.received_data = {'image': False, 'camera_info': False, 'lidar': False}
        self.frame_counter = 0
        self.processing_counter = 0
        
        # Transformation matrix (LiDAR to Camera)
        self.Rcl = np.array([
            [ -0.080003,  -0.996782,   0.005132],
            [0.014233,  -0.006290,  -0.999879],
            [0.996693,  -0.079920,   0.014690]
        ], dtype=np.float64)
        self.tcl = np.array([0.253684, 0.103372, -0.090891], dtype=np.float64)
        
        # Cache
        self.cache_K = None
        self.cache_imsize = None
        
        # --- Initialize YOLO (only if enabled) ---
        self.yolo_model = None
        if self.enable_2d_detection:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            rospy.loginfo(f"Loading YOLO model from {self.yolo_model_path} on {self.device}...")
            if not os.path.exists(self.yolo_model_path):
                rospy.logerr(f"Model not found. Using default yolov8n.pt")
                self.yolo_model_path = "yolov8n.pt"
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
                self.yolo_model.to(self.device)
                rospy.loginfo("YOLO loaded successfully")
            except Exception as e:
                rospy.logerr(f"YOLO load failed: {e}")
                self.yolo_model = None
        else:
            rospy.loginfo("YOLO disabled by user.")
        
        # --- Publishers ---
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=5)
        self.pub_debug = rospy.Publisher("/lidar_overlay/debug_image", Image, queue_size=5)
        
        # --- Subscribers ---
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.cam_info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
        rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
        
        # Timer
        self.processing_timer = rospy.Timer(rospy.Duration(0.1), self.process_callback)
        rospy.Timer(rospy.Duration(5.0), self.status_callback)
        
        rospy.loginfo("="*60)
        rospy.loginfo("ENHANCED FUSION NODE (with Cylindricity + Row-Alignment)")
        rospy.loginfo("="*60)

    # ---------- EXISTING CALLBACKS (Unchanged) ----------
    def status_callback(self, event):
        rospy.loginfo(f"=== STATUS ===")
        rospy.loginfo(f"Image: {self.received_data['image']}")
        rospy.loginfo(f"Camera Info: {self.received_data['camera_info']}")
        rospy.loginfo(f"LiDAR: {self.received_data['lidar']}")
        rospy.loginfo(f"Processing count: {self.processing_counter}")
        if self.latest_pts is not None:
            rospy.loginfo(f"LiDAR points: {len(self.latest_pts)}")
        rospy.loginfo("==============")

    def image_callback(self, msg):
        with self.lock:
            self.latest_image = msg
            self.received_data['image'] = True

    def cam_info_callback(self, msg):
        with self.lock:
            self.latest_cam_info = msg
            self.received_data['camera_info'] = True

    def lidar_callback(self, msg):  # panggil lidar data
        try:
            points_list = []
            count = 0
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                if count % self.downsample_rate == 0:
                    points_list.append([point[0], point[1], point[2]])
                count += 1
                if len(points_list) >= self.max_points_to_draw * 2:
                    break
            if points_list:
                with self.lock:
                    self.latest_pts = np.array(points_list, dtype=np.float32)
                    self.received_data['lidar'] = True
                rospy.loginfo(f"Received {len(self.latest_pts)} LiDAR points")
        except Exception as e:
            rospy.logerr(f"LiDAR error: {e}")

    # ---------- [NEW] CYLINDRICITY + RANSAC FUNCTIONS ----------
    def _circle_from_3pts(self, p1, p2, p3):
        """Compute center (cx, cy) and radius from 3 points."""
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        d = 2 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
        if abs(d) < 1e-8:
            return None, 0
        ux = ((ax**2 + ay**2)*(by - cy) + (bx**2 + by**2)*(cy - ay) + (cx**2 + cy**2)*(ay - by)) / d
        uy = ((ax**2 + ay**2)*(cx - bx) + (bx**2 + by**2)*(ax - cx) + (cx**2 + cy**2)*(bx - ax)) / d
        center = np.array([ux, uy])
        radius = np.linalg.norm(center - np.array([ax, ay]))
        return center, radius

    def _fit_cylinder_ransac(self, cluster_points):
        """
        Fit a cylinder to a point cluster.
        Returns: (center_3d, radius, height, z_min, z_max) or None
        """
        if len(cluster_points) < 15:
            return None
        
        # 1. PCA to find main axis (should be vertical)
        pca = PCA(n_components=3)
        pca.fit(cluster_points)
        axis = pca.components_[0]  # Primary eigenvector
        
        # Ensure axis points upward
        if axis[2] < 0:
            axis = -axis
        
        # 2. Project points onto plane perpendicular to axis
        # Rotation matrix to align axis with Z
        z_axis = np.array([0, 0, 1])
        v = np.cross(axis, z_axis)
        s = np.linalg.norm(v)
        if s > 1e-6:
            c = np.dot(axis, z_axis)
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
        else:
            R = np.eye(3) if np.dot(axis, z_axis) > 0 else -np.eye(3)
        
        rotated_pts = np.dot(cluster_points, R.T)
        pts_2d = rotated_pts[:, :2]  # X, Y in rotated frame
        
        # 3. RANSAC circle fitting on 2D points
        best_inliers = []
        best_center = None
        best_radius = 0
        
        for _ in range(self.ransac_iterations):
            if len(pts_2d) < 3:
                break
            idx = np.random.choice(len(pts_2d), 3, replace=False)
            res = self._circle_from_3pts(pts_2d[idx[0]], pts_2d[idx[1]], pts_2d[idx[2]])
            if res[0] is None:
                continue
            center, radius = res
            if radius < 0.05 or radius > 0.4 :  # Palm trunks typically 0.1 - 0.4m
                continue
            
            dist = np.abs(np.linalg.norm(pts_2d - center, axis=1) - radius)
            inliers = np.where(dist < self.cylinder_inlier_thresh)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_center = center
                best_radius = radius
        
        if len(best_inliers) < 10:
            return None
        
        # 4. Refit using all inliers (least squares circle)
        inlier_pts = pts_2d[best_inliers]
        if len(inlier_pts) < 3:
            return None
        # Quick refinement: take mean of inliers as center approximation (or re-run RANSAC on them)
        # For simplicity, we use the RANSAC result.
        
        # 5. Compute height (extent along axis)
        z_vals = rotated_pts[best_inliers, 2]
        z_min = np.min(z_vals)
        z_max = np.max(z_vals)
        height = z_max - z_min
        
        if height < self.min_cluster_height:
            return None
        
        # 6. Transform center back to original frame
        center_3d = np.array([best_center[0], best_center[1], (z_min + z_max)/2])
        center_3d = np.dot(center_3d, R)  # Inverse rotation
        
        return {
            'center': center_3d,
            'radius': best_radius,
            'height': height,
            'z_min': z_min,
            'z_max': z_max,
            'points': cluster_points[best_inliers]  # for visualization
        }

    def _detect_trunks_lidar(self, points):
        """Main LiDAR trunk detection pipeline: DBSCAN -> Cylinder RANSAC."""
        if len(points) < 20:
            return []
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(points)
        labels = clustering.labels_
        unique_labels = set(labels)
        
        trunks = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster_mask = (labels == label)
            cluster_pts = points[cluster_mask]
            if len(cluster_pts) < 15:
                continue
            
            result = self._fit_cylinder_ransac(cluster_pts)
            if result is not None:
                trunks.append(result)
        
        rospy.loginfo(f"LiDAR detected {len(trunks)} trunk candidates.")
        return trunks

    # ---------- [NEW] ROW ALIGNMENT ----------
    def _apply_row_alignment(self, trunks):
        """
        Align trunk centers to plantation grid and fill missing ones.
        Assumes rows are parallel to row_orientation.
        """
        if len(trunks) < 2:
            return trunks
        
        # Rotate coordinates to align rows with X-axis
        cos_a = np.cos(-self.row_orientation)
        sin_a = np.sin(-self.row_orientation)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        centroids = np.array([t['center'][:2] for t in trunks])
        rotated_centroids = np.dot(centroids, R.T)
        
        # Group by Y (row index)
        row_tolerance = self.row_spacing * 0.3
        rows = {}
        for i, (x, y) in enumerate(rotated_centroids):
            assigned = False
            for row_y in rows.keys():
                if abs(y - row_y) < row_tolerance:
                    rows[row_y].append(i)
                    assigned = True
                    break
            if not assigned:
                rows[y] = [i]
        
        # For each row, sort by X and check gaps
        new_trunks = []
        for row_y, indices in rows.items():
            row_indices = sorted(indices, key=lambda i: rotated_centroids[i, 0])
            row_x_vals = [rotated_centroids[i, 0] for i in row_indices]
            
            # Average row Y
            avg_y = np.mean([rotated_centroids[i, 1] for i in row_indices])
            
            # Snap Y to grid (optional)
            snapped_y = avg_y
            
            # Fill gaps
            for j in range(len(row_x_vals) - 1):
                x1 = row_x_vals[j]
                x2 = row_x_vals[j+1]
                gap = x2 - x1
                if gap > self.row_spacing * self.fill_gap_threshold:
                    num_missing = int(round(gap / self.row_spacing)) - 1
                    for k in range(1, num_missing + 1):
                        new_x = x1 + k * self.row_spacing
                        # Create virtual trunk
                        virtual_center = np.dot(np.array([new_x, snapped_y]), R)  # Rotate back
                        virtual_trunk = {
                            'center': np.array([virtual_center[0], virtual_center[1], 0.0]),
                            'radius': 0.2,
                            'height': 3.0,
                            'z_min': 0.0,
                            'z_max': 3.0,
                            'is_virtual': True
                        }
                        new_trunks.append(virtual_trunk)
                        rospy.loginfo(f"Filled virtual trunk at ({virtual_center[0]:.2f}, {virtual_center[1]:.2f})")
            
            # Add original trunks back with snapped Y
            for i in row_indices:
                orig = trunks[i]
                # Snap the centroid slightly
                orig_center = orig['center'].copy()
                orig_center[1] = np.dot(np.array([rotated_centroids[i, 0], snapped_y]), R)[1]
                orig['center'] = orig_center
                orig['is_virtual'] = False
                new_trunks.append(orig)
        
        return new_trunks

    # ---------- [NEW] FUSION ASSOCIATION (IoU) ----------
    def _project_trunk_to_bbox(self, trunk, K, W, H):
        """Project LiDAR trunk to 2D bbox."""
        center_3d = trunk['center']
        radius = trunk['radius']
        height = trunk['height']
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Project center
        u = int((fx * center_3d[0] / center_3d[2]) + cx)
        v = int((fy * center_3d[1] / center_3d[2]) + cy)
        
        if center_3d[2] < 0.1:
            return None
        
        # Estimate bbox size in image plane
        width_px = int((fx * radius * self.bbox_w_ratio) / center_3d[2])
        height_px = int((fy * height * self.bbox_h_ratio) / center_3d[2])
        width_px = max(10, min(width_px, 200))
        height_px = max(20, min(height_px, 300))
        
        x1 = max(0, u - width_px // 2)
        y1 = max(0, v - height_px // 2)
        x2 = min(W, u + width_px // 2)
        y2 = min(H, v + height_px // 2)
        
        return [x1, y1, x2, y2], center_3d[2]  # bbox, depth

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    # ---------- MAIN PROCESSING CALLBACK (Modified) ----------
    def process_callback(self, event):
        self.frame_counter += 1
        if self.frame_counter % (self.frame_skip + 1) != 0:
            return
        
        with self.lock:
            if self.latest_image is None or self.latest_cam_info is None or self.latest_pts is None:
                return
            img_msg = self.latest_image
            cam_info_msg = self.latest_cam_info
            pts = self.latest_pts.copy()
        
        self.processing_counter += 1
        rospy.loginfo(f"Processing #{self.processing_counter}")
        
        try:
            # --- 1. Decode Image & Camera Intrinsics ---
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            H, W = img.shape[:2]
            
            if self.cache_K is None or self.cache_imsize != (H, W):
                K = np.array(cam_info_msg.K, dtype=np.float32).reshape(3, 3)
                self.cache_K = K
                self.cache_imsize = (H, W)
            else:
                K = self.cache_K
            
            # --- 2. YOLO Detection (Conditional) ---
            yolo_boxes = []
            if self.enable_2d_detection and self.yolo_model is not None:
                results = self.yolo_model(img, conf=self.conf_threshold, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        if class_id in self.target_classes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            yolo_boxes.append({
                                'bbox': [x1, y1, x2, y2],
                                'class': class_id,
                                'confidence': float(box.conf[0])
                            })
                rospy.loginfo(f"YOLO found {len(yolo_boxes)} target boxes.")
            else:
                rospy.logdebug("YOLO disabled or not loaded.")
            
            # --- 3. LiDAR Trunk Detection (Cylindricity + RANSAC) ---
            # Filter points to region of interest (optional, speeds up clustering)
            # Limit Z to 2m height, and X/Y to within 10m
            roi_mask = (pts[:, 2] > 0.0) & (pts[:, 2] < 2.0) & (np.linalg.norm(pts[:, :2], axis=1) < 10.0)
            lidar_pts = pts[roi_mask]
            
            lidar_trunks = self._detect_trunks_lidar(lidar_pts)
            
            # --- 4. Row Alignment ---
            if len(lidar_trunks) > 1:
                lidar_trunks = self._apply_row_alignment(lidar_trunks)
            
            # --- 5. Project LiDAR trunks to 2D and Associate ---
            lidar_2d_boxes = []
            for trunk in lidar_trunks:
                proj = self._project_trunk_to_bbox(trunk, K, W, H)
                if proj is not None:
                    bbox, depth = proj
                    lidar_2d_boxes.append({
                        'bbox': bbox,
                        'depth': depth,
                        'trunk': trunk,
                        'matched': False
                    })
            
            # Match YOLO boxes with LiDAR boxes via IoU
            fused_results = []
            matched_yolo_indices = set()
            matched_lidar_indices = set()
            
            for i, lidar_item in enumerate(lidar_2d_boxes):
                best_iou = 0
                best_j = -1
                for j, yolo_item in enumerate(yolo_boxes):
                    if j in matched_yolo_indices:
                        continue
                    iou = self._compute_iou(lidar_item['bbox'], yolo_item['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                
                if best_iou > self.iou_threshold and best_j != -1:
                    # FUSED
                    matched_yolo_indices.add(best_j)
                    matched_lidar_indices.add(i)
                    fused_results.append({
                        'type': 'fused',
                        'lidar_trunk': lidar_item['trunk'],
                        'yolo_box': yolo_boxes[best_j],
                        'lidar_bbox': lidar_item['bbox'],
                        'depth': lidar_item['depth']
                    })
                    rospy.loginfo(f"FUSED: IoU={best_iou:.2f}")
            
            # LiDAR-only (missed by YOLO)
            for i, lidar_item in enumerate(lidar_2d_boxes):
                if i not in matched_lidar_indices:
                    fused_results.append({
                        'type': 'lidar_only',
                        'lidar_trunk': lidar_item['trunk'],
                        'lidar_bbox': lidar_item['bbox'],
                        'depth': lidar_item['depth']
                    })
            
            # YOLO-only (missed by LiDAR)
            for j, yolo_item in enumerate(yolo_boxes):
                if j not in matched_yolo_indices:
                    fused_results.append({
                        'type': 'yolo_only',
                        'yolo_box': yolo_item['bbox'],
                        'class': yolo_item['class'],
                        'confidence': yolo_item['confidence']
                    })
            
            # --- 6. Draw Results on Image ---
            for res in fused_results:
                if res['type'] == 'fused':
                    # Green box
                    bbox = res['lidar_bbox']
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                    label = f"FUSED {res['depth']:.2f}m"
                    cv2.putText(img, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    # Draw 3D center
                    cx = (bbox[0]+bbox[2])//2
                    cy = (bbox[1]+bbox[3])//2
                    cv2.circle(img, (cx, cy), 5, (0,255,0), -1)
                
                elif res['type'] == 'lidar_only':
                    # Blue box (LiDAR detected, YOLO missed)
                    bbox = res['lidar_bbox']
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    label = f"LIDAR {res['depth']:.2f}m"
                    cv2.putText(img, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                
                elif res['type'] == 'yolo_only':
                    # Orange box (YOLO detected, LiDAR missed)
                    bbox = res['yolo_box']
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 165, 255), 2)
                    label = f"YOLO {res['confidence']:.2f}"
                    cv2.putText(img, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
            
            # Info text
            cv2.putText(img, f"Fused: {len([r for r in fused_results if r['type']=='fused'])}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(img, f"LiDAR: {len([r for r in fused_results if r['type']=='lidar_only'])}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.putText(img, f"YOLO: {len([r for r in fused_results if r['type']=='yolo_only'])}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
            
            # Publish
            out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            out_msg.header = img_msg.header
            self.pub.publish(out_msg)
            
            rospy.loginfo(f"Published: {len(fused_results)} objects.")
            
        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
            import traceback
            traceback.print_exc()

def main():
    rospy.init_node("fusion_lidar_overlay", anonymous=False)
    node = FusionLidarImageOverlay()
    rospy.spin()

if __name__ == "__main__":
    main()