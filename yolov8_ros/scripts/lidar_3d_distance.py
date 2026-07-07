#!/usr/bin/env python3
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import threading
from sklearn.cluster import DBSCAN

# ==================== Simple Tracker ====================
class ObjectTracker:
    def __init__(self, max_age=5, alpha=0.7, association_dist=0.5):
        self.max_age = max_age
        self.alpha = alpha
        self.association_dist = association_dist
        self.objects = {}          # id -> {'centroid': np.array, 'age': int}
        self.next_id = 0

    def update(self, detections):
        """
        detections: list of dict with key 'centroid' (3D)
        Returns: list of dict with keys 'id', 'centroid'
        """
        # Remove dead objects if no detections
        if not detections:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['age'] += 1
                if self.objects[obj_id]['age'] > self.max_age:
                    del self.objects[obj_id]
            return []

        # Match detections to existing objects
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
                # Update existing object with smoothing
                obj = self.objects[best_id]
                obj['centroid'] = self.alpha * det['centroid'] + (1 - self.alpha) * obj['centroid']
                obj['age'] = 0
                matched_ids.add(best_id)
                det['id'] = best_id
            else:
                # New object
                det['id'] = self.next_id
                self.next_id += 1
                self.objects[det['id']] = {
                    'centroid': det['centroid'].copy(),
                    'age': 0
                }

        # Age unmatched objects and remove old ones
        for obj_id, obj in self.objects.items():
            if obj_id not in matched_ids:
                obj['age'] += 1
        for obj_id in list(self.objects.keys()):
            if self.objects[obj_id]['age'] > self.max_age:
                del self.objects[obj_id]

        # Return tracked objects with IDs
        result = []
        for det in detections:
            obj = self.objects[det['id']]
            result.append({
                'id': det['id'],
                'centroid': obj['centroid']   # smoothed
            })
        return result

# ==================== Main Detector Node ====================
class LidarObjectDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # ROS parameters
        self.image_topic = rospy.get_param("~image_topic", "/zed2i/zed_node/left/image_rect_color")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/zed2i/zed_node/left/camera_info")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/ouster/points")
        self.output_topic = rospy.get_param("~output_topic", "/lidar_overlay/fused_image")

        # LiDAR processing
        self.ground_threshold = rospy.get_param("~ground_threshold", 0.2)      # points above this z (m)
        self.cluster_eps = rospy.get_param("~cluster_eps", 0.8)               # DBSCAN epsilon
        self.cluster_min_samples = rospy.get_param("~cluster_min_samples", 5)
        self.target_height = rospy.get_param("~target_height", 1.5)           # trunk height above ground
        self.height_tolerance = rospy.get_param("~height_tolerance", 0.3)

        # ═══════ NEW: maximum range filter ═══════
        self.max_range = rospy.get_param("~max_range", 10.0)   # meters

        # Tracker parameters
        self.tracker_alpha = rospy.get_param("~tracker_alpha", 0.7)
        self.association_dist = rospy.get_param("~association_dist", 0.5)
        self.tracker = ObjectTracker(alpha=self.tracker_alpha, association_dist=self.association_dist)

        # Extrinsic calibration (LiDAR → Camera) – only for projection
        self.Rcl = np.array([
            [-0.080003, -0.996782,  0.005132],
            [ 0.014233, -0.006290, -0.999879],
            [ 0.996693, -0.079920,  0.014690]
        ], dtype=np.float64)
        self.tcl = np.array([0.253684, 0.103372, -0.090891], dtype=np.float64)

        # Camera intrinsic cache
        self.cache_K = None
        self.cache_imsize = None

        # Subscribers and publishers
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=5)
        self.latest_image = None
        self.latest_cam_info = None
        self.latest_pts = None
        self.received_data = {'image': False, 'camera_info': False, 'lidar': False}

        rospy.loginfo("Subscribing to:")
        rospy.loginfo(f"  Image: {self.image_topic}")
        rospy.loginfo(f"  Camera info: {self.cam_info_topic}")
        rospy.loginfo(f"  LiDAR: {self.lidar_topic}")

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.cam_info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
        rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)

        # Process timer (10 Hz)
        self.processing_timer = rospy.Timer(rospy.Duration(0.1), self.process_callback)

        rospy.loginfo("="*60)
        rospy.loginfo("LIDAR OBJECT DETECTOR – 2D Bounding Box from projected points")
        rospy.loginfo("="*60)
        rospy.loginfo(f"Ground threshold: {self.ground_threshold} m")
        rospy.loginfo(f"Max detection range: {self.max_range} m")   # <--- new
        rospy.loginfo(f"Cluster eps: {self.cluster_eps}, min_samples: {self.cluster_min_samples}")
        rospy.loginfo(f"Target height: {self.target_height} ± {self.height_tolerance} m")
        rospy.loginfo(f"Tracker alpha: {self.tracker_alpha}, association dist: {self.association_dist}")
        rospy.loginfo("="*60)

        rospy.Timer(rospy.Duration(5.0), self.status_callback)

    # --- callbacks unchanged ---
    def status_callback(self, event):
        rospy.loginfo(f"=== STATUS ===")
        rospy.loginfo(f"Image: {self.received_data['image']}")
        rospy.loginfo(f"Camera Info: {self.received_data['camera_info']}")
        rospy.loginfo(f"LiDAR: {self.received_data['lidar']}")
        rospy.loginfo(f"Tracked objects: {len(self.tracker.objects)}")
        rospy.loginfo("==============")

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
                    self.received_data['lidar'] = True
                rospy.loginfo(f"Received {len(pts)} LiDAR points")
        except Exception as e:
            rospy.logerr(f"LiDAR callback error: {e}")

    def process_callback(self, event):
        with self.lock:
            if not all(self.received_data.values()):
                return
            img_msg = self.latest_image
            cam_info_msg = self.latest_cam_info
            pts = self.latest_pts.copy()

        try:
            # -- Convert image and get camera intrinsics --
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            H, W = img.shape[:2]
            if self.cache_K is None or self.cache_imsize != (H, W):
                K = np.array(cam_info_msg.K, dtype=np.float32).reshape(3, 3)
                self.cache_K = K
                self.cache_imsize = (H, W)
                rospy.loginfo(f"Camera matrix:\n{K}")
            else:
                K = self.cache_K
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

            # ═══════ NEW: range filter (LiDAR frame) ═══════
            ranges = np.linalg.norm(pts, axis=1)  # distance from LiDAR origin
            mask_range = ranges <= self.max_range
            pts = pts[mask_range]
            if len(pts) == 0:
                rospy.logdebug("No points within max_range")
                return

            # -- 1. Ground removal --
            ground_mask = pts[:, 2] > self.ground_threshold
            pts_above = pts[ground_mask]
            if len(pts_above) == 0:
                return

            # -- 2. DBSCAN clustering --
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(pts_above)
            labels = clustering.labels_
            unique_labels = set(labels)

            # -- 3. Extract candidate clusters with height ~ target_height --
            detections = []   # each: {'centroid': 3D, 'points': Nx3}
            for label in unique_labels:
                if label == -1:
                    continue
                cluster_pts = pts_above[labels == label]
                centroid = np.mean(cluster_pts, axis=0)
                # height check: centroid z should be around target_height (trunk height)
                if abs(centroid[2] - self.target_height) > self.height_tolerance:
                    continue
                detections.append({
                    'centroid': centroid,
                    'points': cluster_pts
                })

            if not detections:
                # No objects – publish original image
                out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                out_msg.header = img_msg.header
                self.pub.publish(out_msg)
                return

            # -- 4. Update tracker (uses only centroid) --
            centroid_list = [{'centroid': d['centroid']} for d in detections]
            tracked = self.tracker.update(centroid_list)

            # Build a mapping from tracked ID to its detection (for points)
            id_to_detection = {}
            for obj in tracked:
                obj_id = obj['id']
                smoothed_centroid = obj['centroid']
                # Find closest detection
                best_idx = -1
                best_dist = float('inf')
                for i, det in enumerate(detections):
                    dist = np.linalg.norm(smoothed_centroid - det['centroid'])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx >= 0:
                    id_to_detection[obj_id] = detections[best_idx]

            # -- 5. Draw 2D bounding box for each tracked object --
            for obj in tracked:
                obj_id = obj['id']
                centroid = obj['centroid']          # smoothed LiDAR centroid
                distance = np.linalg.norm(centroid) # LiDAR range (ground truth)

                # Get the current detection points for this ID
                det = id_to_detection.get(obj_id)
                if det is None:
                    continue
                cluster_pts = det['points']         # current frame's cluster points

                # Project these points onto the image
                pts_cam = (cluster_pts @ self.Rcl.T) + self.tcl
                Z = pts_cam[:, 2]
                # Keep only points in front of camera
                valid = Z > 0.1
                if not np.any(valid):
                    continue
                pts_cam = pts_cam[valid]
                Z = Z[valid]
                u = (fx * pts_cam[:,0] / Z + cx).astype(np.int32)
                v = (fy * pts_cam[:,1] / Z + cy).astype(np.int32)

                # Clip to image boundaries
                in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                u = u[in_img]
                v = v[in_img]
                if len(u) == 0:
                    continue

                # Compute tight 2D bounding box
                x1, y1 = np.min(u), np.min(v)
                x2, y2 = np.max(u), np.max(v)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)  # cyan

                # Project smoothed centroid for label
                centroid_cam = centroid @ self.Rcl.T + self.tcl
                if centroid_cam[2] > 0.1:
                    uc = int(fx * centroid_cam[0] / centroid_cam[2] + cx)
                    vc = int(fy * centroid_cam[1] / centroid_cam[2] + cy)
                    label = f"ID:{obj_id} {distance:.2f}m"
                    cv2.putText(img, label, (uc-30, vc-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # -- Publish --
            out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            out_msg.header = img_msg.header
            self.pub.publish(out_msg)

        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
            import traceback
            traceback.print_exc()

def main():
    rospy.init_node("lidar_object_detector", anonymous=False)
    rospy.loginfo("Starting LiDAR Object Detector (2D bounding box from point cloud)")
    try:
        node = LidarObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()