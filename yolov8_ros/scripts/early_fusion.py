#!/usr/bin/env python3
"""Early (data-level) camera-LiDAR fusion: LiDAR depth is densified and blended
into the RGB image, and YOLO runs detection on that fused image instead of raw RGB."""
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


class EarlyFusionLidarImage:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # Topics
        self.image_topic = rospy.get_param("~image_topic", "/zed2i/zed_node/left/image_rect_color")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/zed2i/zed_node/left/camera_info")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/ouster/points")
        self.output_topic = rospy.get_param("~output_topic", "/early_fusion/fused_image")

        # YOLO parameters
        self.yolo_model_path = rospy.get_param("~yolo_model", "/home/amt4/yolov8_ros/src/Yolov8_ros/yolov8_ros/weights/palmtree2.pt")
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.5)
        self.target_classes = rospy.get_param("~target_classes", [0])  # 0 = palmtree

        # Performance
        self.max_points_to_draw = rospy.get_param("~max_points", 3000)
        self.downsample_rate = rospy.get_param("~downsample_rate", 4)
        self.frame_skip = rospy.get_param("~frame_skip", 1)

        # Distance outlier filtering (applied to raw points, not the densified map)
        self.bbox_shrink_factor = rospy.get_param("~bbox_shrink_factor", 0.8)
        self.sigma_rule_std = rospy.get_param("~sigma_rule_std", 1.5)
        self.use_sigma_rule = rospy.get_param("~use_sigma_rule", True)
        self.use_bbox_shrink = rospy.get_param("~use_bbox_shrink", True)

        # Depth densification (sparse LiDAR -> semi-dense depth map)
        self.depth_dilate_kernel_size = rospy.get_param("~depth_dilate_kernel_size", 9)
        self.depth_dilate_iterations = rospy.get_param("~depth_dilate_iterations", 1)

        # Early-fusion blend weight: how strongly the depth map is mixed into RGB
        # before detection. 0 = pure RGB (no fusion), 1 = depth colormap only.
        self.fusion_alpha = rospy.get_param("~fusion_alpha", 0.4)

        # Visualization
        self.point_size = rospy.get_param("~point_size", 3)
        self.min_depth = rospy.get_param("~min_depth", 0.5)
        self.max_depth = rospy.get_param("~max_depth", 40.0)

        # Fused-image saving
        self.save_fused_image = rospy.get_param("~save_fused_image", True)
        self.save_fused_image_dir = rospy.get_param("~save_fused_image_dir", "/home/amt4/fastcalib_ws/data/early_fusion_data")
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

        # Latest messages
        self.latest_image = None
        self.latest_cam_info = None
        self.latest_pts = None
        self.received_data = {'image': False, 'camera_info': False, 'lidar': False}

        self.frame_counter = 0
        self.processing_counter = 0

        # LiDAR -> camera extrinsics
        self.Rcl = np.array([
            [-0.080003, -0.996782, 0.005132],
            [0.014233, -0.006290, -0.999879],
            [0.996693, -0.079920, 0.014690]
        ], dtype=np.float64)
        self.tcl = np.array([0.253684, 0.103372, -0.090891], dtype=np.float64)

        self.cache_K = None
        self.cache_imsize = None

        # YOLO
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Loading YOLO model from {self.yolo_model_path} on {self.device}...")
        if not os.path.exists(self.yolo_model_path):
            rospy.logerr(f"YOLO model not found at {self.yolo_model_path}")
            rospy.loginfo("Using default yolov8n.pt instead")
            self.yolo_model_path = "yolov8n.pt"
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            self.yolo_model.to(self.device)
            rospy.loginfo("YOLO model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}")
            self.yolo_model = None

        # Publishers
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=5)
        self.pub_depth_debug = rospy.Publisher("/early_fusion/depth_debug", Image, queue_size=5)
        self.pub_fused_input_debug = rospy.Publisher("/early_fusion/fused_input_debug", Image, queue_size=5)

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.cam_info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
        rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)

        self.processing_timer = rospy.Timer(rospy.Duration(0.1), self.process_callback)
        rospy.Timer(rospy.Duration(1.0), self.status_callback)

        rospy.loginfo("=" * 60)
        rospy.loginfo("EARLY FUSION LiDAR-Image Node with YOLO Detection")
        rospy.loginfo(f"YOLO Model: {self.yolo_model_path}")
        rospy.loginfo(f"Target classes: {self.target_classes}")
        rospy.loginfo(f"Fusion alpha: {self.fusion_alpha}")
        rospy.loginfo("=" * 60)

    def status_callback(self, event):
        rospy.loginfo(
            f"[status] image={self.received_data['image']} "
            f"cam_info={self.received_data['camera_info']} "
            f"lidar={self.received_data['lidar']} "
            f"processed={self.processing_counter} frame={self.frame_counter}"
        )

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
            points_list = []
            count = 0
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                if count % self.downsample_rate == 0:
                    points_list.append([point[0], point[1], point[2]])
                count += 1
                if len(points_list) >= self.max_points_to_draw * 2:
                    break

            if len(points_list) > 0:
                pts = np.array(points_list, dtype=np.float32)
                with self.lock:
                    self.latest_pts = pts
                    self.received_data['lidar'] = True
            else:
                rospy.logwarn("No valid LiDAR points extracted")
        except Exception as e:
            rospy.logerr(f"LiDAR callback error: {e}")

    def shrink_bbox(self, bbox, shrink_factor):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        shrink_w = width * (1 - shrink_factor) / 2
        shrink_h = height * (1 - shrink_factor) / 2
        return [int(x1 + shrink_w), int(y1 + shrink_h), int(x2 - shrink_w), int(y2 - shrink_h)]

    def apply_sigma_filter(self, points_3d, distances, std_multiplier=1.5):
        if len(distances) < 3:
            return points_3d
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        lower_bound = mean_dist - (std_multiplier * std_dist)
        upper_bound = mean_dist + (std_multiplier * std_dist)
        mask = (distances >= lower_bound) & (distances <= upper_bound)
        return points_3d[mask]

    def build_sparse_depth_map(self, u, v, Z, H, W):
        """Project points to a per-pixel depth map, keeping the nearest point per pixel."""
        depth_map = np.zeros((H, W), dtype=np.float32)
        if len(u) == 0:
            return depth_map
        # Closest point wins on pixel collisions: sort far->near, nearer overwrites last.
        order = np.argsort(-Z)
        depth_map[v[order], u[order]] = Z[order]
        return depth_map

    def densify_depth_map(self, depth_map):
        """Grow sparse depth samples into a semi-dense map via morphological dilation.

        Depth is inverted before dilation (a max filter) so that when two nearby
        points compete for the same neighborhood, the nearer one wins rather than
        the farther one.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.depth_dilate_kernel_size, self.depth_dilate_kernel_size)
        )
        valid = depth_map > 0
        inv_depth = np.where(valid, self.max_depth - depth_map, 0).astype(np.float32)
        inv_dilated = cv2.dilate(inv_depth, kernel, iterations=self.depth_dilate_iterations)
        dense_depth = np.where(inv_dilated > 0, self.max_depth - inv_dilated, 0).astype(np.float32)
        return dense_depth

    def colorize_depth(self, depth_map):
        valid_mask = depth_map > 0
        depth_clipped = np.clip(depth_map, self.min_depth, self.max_depth)
        depth_norm = ((depth_clipped - self.min_depth) / (self.max_depth - self.min_depth) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_color[~valid_mask] = 0
        return depth_color, valid_mask

    def fuse_depth_into_image(self, img, depth_map):
        """Blend a colorized depth map into the RGB image, producing the fused
        input that gets fed to YOLO. Pixels with no depth data stay pure RGB."""
        depth_color, valid_mask = self.colorize_depth(depth_map)
        blended = cv2.addWeighted(img, 1 - self.fusion_alpha, depth_color, self.fusion_alpha, 0)
        fused = img.copy()
        fused[valid_mask] = blended[valid_mask]
        return fused, depth_color

    def save_fused_image_frame(self, img):
        if not self.save_fused_image:
            return
        now = rospy.Time.now().to_sec()
        if self.last_saved_fused_image_time is not None:
            if now - self.last_saved_fused_image_time < self.save_fused_image_interval_sec:
                return
        self.last_saved_fused_image_time = now
        os.makedirs(self.save_fused_image_dir, exist_ok=True)
        filename = os.path.join(self.save_fused_image_dir, f"fuse_image_{self.fused_image_counter}.png")
        self.fused_image_counter += 1
        if cv2.imwrite(filename, img):
            rospy.loginfo(f"Saved fused image to {filename}")
        else:
            rospy.logwarn(f"Failed to save fused image to {filename}")

    def project_lidar_to_image(self, pts, K, H, W):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        pts_cam = (pts @ self.Rcl.T.astype(np.float32)) + self.tcl.astype(np.float32)
        Z = pts_cam[:, 2]
        valid = Z > 0.1
        pts_cam = pts_cam[valid]
        Z = Z[valid]

        X = pts_cam[:, 0]
        Y = pts_cam[:, 1]
        u = (fx * X / Z + cx).astype(np.int32)
        v = (fy * Y / Z + cy).astype(np.int32)

        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return pts_cam[in_img], u[in_img], v[in_img], Z[in_img]

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

        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            H, W = img.shape[:2]

            if self.cache_K is None or self.cache_imsize != (H, W):
                K = np.array(cam_info_msg.K, dtype=np.float32).reshape(3, 3)
                self.cache_K = K
                self.cache_imsize = (H, W)
            else:
                K = self.cache_K
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            if self.yolo_model is None:
                rospy.logerr("YOLO model not loaded")
                return

            # --- Early fusion: build the depth-fused image BEFORE detection ---
            pts_cam, u, v, Z = self.project_lidar_to_image(pts, K, H, W)
            sparse_depth = self.build_sparse_depth_map(u, v, Z, H, W)
            dense_depth = self.densify_depth_map(sparse_depth)
            fused_img, depth_color = self.fuse_depth_into_image(img, dense_depth)

            self.pub_depth_debug.publish(self._to_img_msg(depth_color, img_msg.header))
            self.pub_fused_input_debug.publish(self._to_img_msg(fused_img, img_msg.header))

            # --- Detection runs on the fused image, not raw RGB ---
            results = self.yolo_model(fused_img, conf=self.conf_threshold, device=self.device, verbose=False)
            vis_img = img.copy()

            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                cv2.putText(vis_img, "No detections", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.pub.publish(self._to_img_msg(vis_img, img_msg.header))
                return

            target_boxes = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if class_id not in self.target_classes:
                    continue
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                target_boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_id,
                    'confidence': confidence,
                    'distance': None
                })

            if len(target_boxes) == 0:
                cv2.putText(vis_img, f"No target objects (looking for {self.target_classes})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                self.pub.publish(self._to_img_msg(vis_img, img_msg.header))
                return

            # --- Distance estimation from raw (non-densified) points, per box ---
            all_filtered_points = []
            for det in target_boxes:
                bbox = det['bbox']
                in_bbox = (u >= bbox[0]) & (u <= bbox[2]) & (v >= bbox[1]) & (v <= bbox[3])
                bbox_points_3d = pts_cam[in_bbox]
                bbox_points_2d = np.array([u[in_bbox], v[in_bbox]]).T
                bbox_distances = np.linalg.norm(bbox_points_3d, axis=1)

                if len(bbox_points_3d) == 0:
                    det['distance'] = None
                    continue

                filtered_3d = bbox_points_3d
                filtered_2d = bbox_points_2d
                filtered_dist = bbox_distances

                if self.use_bbox_shrink:
                    shrunk_bbox = self.shrink_bbox(bbox, self.bbox_shrink_factor)
                    in_shrunk = in_bbox & (u >= shrunk_bbox[0]) & (u <= shrunk_bbox[2]) & \
                        (v >= shrunk_bbox[1]) & (v <= shrunk_bbox[3])
                    filtered_3d = pts_cam[in_shrunk]
                    filtered_2d = np.array([u[in_shrunk], v[in_shrunk]]).T
                    filtered_dist = np.linalg.norm(filtered_3d, axis=1)
                    cv2.rectangle(vis_img, (shrunk_bbox[0], shrunk_bbox[1]),
                                  (shrunk_bbox[2], shrunk_bbox[3]), (255, 0, 0), 2)

                if self.use_sigma_rule and len(filtered_dist) >= 3:
                    filtered_3d = self.apply_sigma_filter(filtered_3d, filtered_dist, self.sigma_rule_std)
                    if len(filtered_3d) > 0:
                        fX, fY, fZ = filtered_3d[:, 0], filtered_3d[:, 1], filtered_3d[:, 2]
                        filtered_2d = np.array([
                            (fx * fX / fZ + cx).astype(np.int32),
                            (fy * fY / fZ + cy).astype(np.int32)
                        ]).T
                        filtered_dist = np.linalg.norm(filtered_3d, axis=1)

                if len(filtered_2d) > 0:
                    det['distance'] = float(np.median(filtered_dist))
                    all_filtered_points.append({'points_2d': filtered_2d, 'distances': filtered_dist})
                else:
                    det['distance'] = None

            # --- Draw LiDAR points ---
            total_points = 0
            for points_data in all_filtered_points:
                points_2d = points_data['points_2d']
                distances = points_data['distances']
                total_points += len(points_2d)
                if len(points_2d) > self.max_points_to_draw:
                    indices = np.random.choice(len(points_2d), self.max_points_to_draw, replace=False)
                    points_2d = points_2d[indices]
                    distances = distances[indices]
                for i in range(len(points_2d)):
                    depth_norm = np.clip((distances[i] - self.min_depth) / (self.max_depth - self.min_depth), 0, 1)
                    color = (int(255 * depth_norm), 0, int(255 * (1 - depth_norm)))
                    cv2.circle(vis_img, (int(points_2d[i][0]), int(points_2d[i][1])), self.point_size, color, -1)

            # --- Draw boxes and labels ---
            for det in target_boxes:
                bbox = det['bbox']
                distance = det.get('distance')
                cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                dist_str = f"{distance:.2f}m" if distance is not None else "N/A"
                label = f"Class {det['class']} {det['confidence']:.2f} {dist_str}"
                text_x = bbox[0] + (bbox[2] - bbox[0]) // 2
                text_y = bbox[1] + (bbox[3] - bbox[1]) // 2
                cv2.putText(vis_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.putText(vis_img, f"Points: {total_points}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_img, f"Detections: {len(target_boxes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.pub.publish(self._to_img_msg(vis_img, img_msg.header))
            self.save_fused_image_frame(vis_img)

        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
            import traceback
            traceback.print_exc()

    def _to_img_msg(self, img, header):
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header = header
        return msg


def main():
    rospy.init_node("early_fusion_lidar_overlay", anonymous=False)
    rospy.loginfo("Starting EARLY FUSION LiDAR-Image node with YOLO...")
    try:
        node = EarlyFusionLidarImage()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
