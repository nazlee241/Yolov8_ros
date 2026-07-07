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
import csv
from datetime import datetime, timezone

class FusionLidarImageOverlay:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # Parameters with DEBUG output
        self.image_topic = rospy.get_param("~image_topic", "/zed2i/zed_node/left/image_rect_color")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/zed2i/zed_node/left/camera_info")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/ouster/points")
        self.output_topic = rospy.get_param("~output_topic", "/lidar_overlay/fused_image")
        
        # YOLO parameters
        self.yolo_model_path = rospy.get_param("~yolo_model", "/home/amt4/yolov8_ros/src/Yolov8_ros/yolov8_ros/weights/palmtree2.pt")
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.5)
        self.target_classes = rospy.get_param("~target_classes", [0]) #0=palmtree
        
        # Performance optimizations
        self.max_points_to_draw = rospy.get_param("~max_points", 3000)
        self.downsample_rate = rospy.get_param("~downsample_rate", 4)
        self.frame_skip = rospy.get_param("~frame_skip", 1)
        
        # Outlier filtering parameters
        self.bbox_shrink_factor = rospy.get_param("~bbox_shrink_factor", 0.8)
        self.sigma_rule_std = rospy.get_param("~sigma_rule_std", 1.5)
        self.use_sigma_rule = rospy.get_param("~use_sigma_rule", True)
        self.use_bbox_shrink = rospy.get_param("~use_bbox_shrink", True)
        
        # Store latest messages
        self.latest_image = None
        self.latest_cam_info = None
        self.latest_pts = None
        self.received_data = {'image': False, 'camera_info': False, 'lidar': False}
        
        # Frame counter
        self.frame_counter = 0
        self.processing_counter = 0
        
        # Transformation matrix (LiDAR to Camera)
        self.Rcl = np.array([
            # [-0.130379, -0.991454,  0.004591],
            # [-0.076312,  0.005418, -0.997069],
            # [ 0.988523, -0.130348, -0.076366]
            [ -0.080003,  -0.996782,   0.005132],
            [0.014233,  -0.006290,  -0.999879],
            [0.996693,  -0.079920,   0.014690]
        ], dtype=np.float64)
        self.tcl = np.array([0.253684, 0.103372, -0.090891], dtype=np.float64)
        
        # Pre-allocate arrays
        self.cache_K = None
        self.cache_imsize = None
        
        # Visualization params
        self.point_size = rospy.get_param("~point_size", 3)
        self.min_depth = rospy.get_param("~min_depth", 0.5)
        self.max_depth = rospy.get_param("~max_depth", 40.0)

        # Fused-image saving params
        self.save_fused_image = rospy.get_param("~save_fused_image", True)
        self.save_fused_image_dir = rospy.get_param("~save_fused_image_dir", "/home/amt4/fastcalib_ws/data")
        self.save_fused_image_interval_sec = rospy.get_param("~save_fused_image_interval_sec", 5.0)
        self.last_saved_fused_image_time = None

        # # CSV logging params
        # self.csv_output_path = rospy.get_param("~csv_output_path", "/home/amt4/fastcalib_ws/csv/yolo_distance_detections.csv")
        # self.csv_interval_sec = rospy.get_param("~csv_interval_sec", 5.0)
        # self.latest_detection_summary = {"timestamp": None, "detection_count": 0, "distance_m": None}
        # self._ensure_csv_header()
        
        # Initialize YOLO
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Loading YOLO model from {self.yolo_model_path} on {self.device}...")
        
        # Check if model exists
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
        self.pub_debug = rospy.Publisher("/lidar_overlay/debug_image", Image, queue_size=5)
        
        # Subscribers with debug
        rospy.loginfo(f"Subscribing to:")
        rospy.loginfo(f"  Image topic: {self.image_topic}")
        rospy.loginfo(f"  Camera info: {self.cam_info_topic}")
        rospy.loginfo(f"  LiDAR topic: {self.lidar_topic}")
        
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.cam_info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
        rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
        
        # Process timer (10 Hz for debugging)
        self.processing_timer = rospy.Timer(rospy.Duration(0.1), self.process_callback)
        
        rospy.loginfo("="*60)
        rospy.loginfo("FUSION LiDAR-Image Node with YOLO Detection")
        rospy.loginfo("="*60)
        rospy.loginfo(f"YOLO Model: {self.yolo_model_path}")
        rospy.loginfo(f"Target classes: {self.target_classes}")
        rospy.loginfo(f"BBox shrink factor: {self.bbox_shrink_factor}")
        rospy.loginfo(f"Sigma rule std: {self.sigma_rule_std}")
        rospy.loginfo("="*60)
        
        # Start status timer
        rospy.Timer(rospy.Duration(5.0), self.status_callback)
        # rospy.Timer(rospy.Duration(self.csv_interval_sec), self.csv_timer_callback)

    # def _ensure_csv_header(self):
    #     output_dir = os.path.dirname(self.csv_output_path)
    #     if output_dir:
    #         os.makedirs(output_dir, exist_ok=True)

    #     if not os.path.exists(self.csv_output_path):
    #         with open(self.csv_output_path, "w", newline="") as csv_file:
    #             writer = csv.writer(csv_file)
    #             writer.writerow(["timestamp", "yolo_detections", "distance_m"])

    # def update_detection_summary(self, detection_count, distance_m):
    #     self.latest_detection_summary = {
    #         "timestamp": datetime.now(timezone.utc).isoformat(),
    #         "detection_count": int(detection_count),
    #         "distance_m": None if distance_m is None else float(distance_m),
    #     }

    # def csv_timer_callback(self, event):
    #     if self.latest_detection_summary["timestamp"] is None:
    #         return

    #     timestamp = datetime.now(timezone.utc).isoformat()
    #     distance_value = self.latest_detection_summary["distance_m"]
    #     if distance_value is None or np.isnan(distance_value):
    #         distance_str = ""
    #     else:
    #         distance_str = f"{distance_value:.3f}"

    #     with open(self.csv_output_path, "a", newline="") as csv_file:
    #         writer = csv.writer(csv_file)
    #         writer.writerow([timestamp, self.latest_detection_summary["detection_count"], distance_str])

    #     rospy.loginfo(
    #         f"CSV update: detections={self.latest_detection_summary['detection_count']}, "
    #         f"distance={distance_str if distance_str else 'N/A'}"
    #     )
    
    def status_callback(self, event):
        """Print status every 5 seconds"""
        rospy.loginfo(f"=== STATUS ===")
        rospy.loginfo(f"Image: {self.received_data['image']}")
        rospy.loginfo(f"Camera Info: {self.received_data['camera_info']}")
        rospy.loginfo(f"LiDAR: {self.received_data['lidar']}")
        rospy.loginfo(f"Processing count: {self.processing_counter}")
        rospy.loginfo(f"Frame counter: {self.frame_counter}")
        if self.latest_pts is not None:
            rospy.loginfo(f"LiDAR points: {len(self.latest_pts)}")
        rospy.loginfo("==============")
    
    def image_callback(self, msg):
        with self.lock:
            self.latest_image = msg
            self.received_data['image'] = True
            rospy.logdebug(f"Received image: {msg.width}x{msg.height}")
    
    def cam_info_callback(self, msg):
        with self.lock:
            self.latest_cam_info = msg
            self.received_data['camera_info'] = True
            rospy.logdebug("Received camera info")
    
    def lidar_callback(self, msg):
        try:
            points_list = []
            count = 0
            
            rospy.logdebug("Processing LiDAR point cloud...")
            
            # Extract points with downsampling
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
                rospy.loginfo(f"Received {len(pts)} LiDAR points (from {count} total)")
            else:
                rospy.logwarn("No valid LiDAR points extracted")
                
        except Exception as e:
            rospy.logerr(f"LiDAR callback error: {e}")
    
    def shrink_bbox(self, bbox, shrink_factor):
        """Shrink bounding box by given factor"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        shrink_w = width * (1 - shrink_factor) / 2
        shrink_h = height * (1 - shrink_factor) / 2
        
        new_x1 = int(x1 + shrink_w)
        new_y1 = int(y1 + shrink_h)
        new_x2 = int(x2 - shrink_w)
        new_y2 = int(y2 - shrink_h)
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def apply_sigma_filter(self, points_3d, distances, std_multiplier=1.5):
        """Apply sigma rule to filter outliers"""
        if len(distances) < 3:
            return points_3d
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        lower_bound = mean_dist - (std_multiplier * std_dist)
        upper_bound = mean_dist + (std_multiplier * std_dist)
        
        mask = (distances >= lower_bound) & (distances <= upper_bound)
        return points_3d[mask]
    
    def save_fused_image_frame(self, img):
        if not self.save_fused_image:
            return

        now = rospy.Time.now().to_sec()
        if self.last_saved_fused_image_time is not None:
            elapsed = now - self.last_saved_fused_image_time
            if elapsed < self.save_fused_image_interval_sec:
                return

        self.last_saved_fused_image_time = now
        os.makedirs(self.save_fused_image_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_fused_image_dir, f"fused_image_{timestamp}.png")
        success = cv2.imwrite(filename, img)
        if success:
            rospy.loginfo(f"Saved fused image to {filename}")
        else:
            rospy.logwarn(f"Failed to save fused image to {filename}")

    def process_callback(self, event):
        # Frame skipping
        self.frame_counter += 1
        if self.frame_counter % (self.frame_skip + 1) != 0:
            return
        
        with self.lock:
            if self.latest_image is None:
                rospy.logdebug_throttle(5, "Waiting for image...")
                return
            if self.latest_cam_info is None:
                rospy.logdebug_throttle(5, "Waiting for camera info...")
                return
            if self.latest_pts is None:
                rospy.logdebug_throttle(5, "Waiting for LiDAR points...")
                return
            
            img_msg = self.latest_image
            cam_info_msg = self.latest_cam_info
            pts = self.latest_pts.copy()
        
        self.processing_counter += 1
        rospy.loginfo(f"Processing frame #{self.processing_counter}")
        
        try:
            # Convert image
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            H, W = img.shape[:2]
            rospy.logdebug(f"Image size: {W}x{H}")
            
            # Get camera intrinsics
            if self.cache_K is None or self.cache_imsize != (H, W):
                K = np.array(cam_info_msg.K, dtype=np.float32).reshape(3, 3)
                self.cache_K = K
                self.cache_imsize = (H, W)
                rospy.loginfo(f"Camera matrix:\n{K}")
            else:
                K = self.cache_K
            
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # YOLO detection
            if self.yolo_model is None:
                rospy.logerr("YOLO model not loaded")
                # self.update_detection_summary(0, None)  #csv
                return
            
            rospy.logdebug("Running YOLO detection...")
            results = self.yolo_model(img, conf=self.conf_threshold, device=self.device, verbose=False)
            
            if len(results) == 0 or results[0].boxes is None:
                rospy.logwarn("No YOLO detections")
                # self.update_detection_summary(0, None) #csv
                # Publish original image with message
                cv2.putText(img, "No detections", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                out_msg.header = img_msg.header
                self.pub.publish(out_msg)
                return
            
            # Get bounding boxes
            target_boxes = []
            boxes = results[0].boxes
            rospy.loginfo(f"Found {len(boxes)} total detections")
            
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                rospy.logdebug(f"Detection: class={class_id}, conf={confidence:.2f}, bbox=[{x1},{y1},{x2},{y2}]")
                
                if class_id in self.target_classes:
                    target_boxes.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': class_id,
                        'confidence': confidence,
                        'distance': None   # will be filled later if points exist
                    })
                    rospy.loginfo(f"Added target class {class_id} with confidence {confidence:.2f}")
            
            if len(target_boxes) == 0:
                rospy.logwarn(f"No target classes detected. Looking for {self.target_classes}")
                # self.update_detection_summary(len(boxes), None) #csv
                # Publish image with message
                cv2.putText(img, f"No target objects (looking for {self.target_classes})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                out_msg.header = img_msg.header
                self.pub.publish(out_msg)
                return
            
            # Transform LiDAR points
            pts_cam = (pts @ self.Rcl.T.astype(np.float32)) + self.tcl.astype(np.float32)
            
            # Filter points
            Z = pts_cam[:, 2]
            valid = Z > 0.1
            if not valid.any():
                rospy.logwarn("No valid points after Z filtering")
                return
            
            pts_cam = pts_cam[valid]
            Z = Z[valid]
            
            # Project to image
            X = pts_cam[:, 0]
            Y = pts_cam[:, 1]
            u = (fx * X / Z + cx).astype(np.int32)
            v = (fy * Y / Z + cy).astype(np.int32)
            
            # Filter image bounds
            in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u = u[in_img]
            v = v[in_img]
            Z = Z[in_img]
            pts_cam = pts_cam[in_img]
            
            rospy.loginfo(f"Projected {len(u)} points onto image")
            
            # Process each detection and compute distance
            all_filtered_points = []
            
            for det in target_boxes:
                bbox = det['bbox']
                class_id = det['class']
                
                # Get points inside bounding box
                in_bbox = (u >= bbox[0]) & (u <= bbox[2]) & (v >= bbox[1]) & (v <= bbox[3])
                bbox_points_3d = pts_cam[in_bbox]
                bbox_points_2d = np.array([u[in_bbox], v[in_bbox]]).T
                bbox_distances = np.linalg.norm(bbox_points_3d, axis=1)
                
                rospy.logdebug(f"Class {class_id}: {len(bbox_points_3d)} points in bounding box")
                
                if len(bbox_points_3d) == 0:
                    continue
                
                filtered_3d = bbox_points_3d
                filtered_2d = bbox_points_2d
                filtered_dist = bbox_distances
                
                # Apply filtering
                if self.use_bbox_shrink:
                    shrunk_bbox = self.shrink_bbox(bbox, self.bbox_shrink_factor)
                    in_shrunk = (u >= shrunk_bbox[0]) & (u <= shrunk_bbox[2]) & \
                               (v >= shrunk_bbox[1]) & (v <= shrunk_bbox[3])
                    in_shrunk = in_shrunk & in_bbox
                    
                    filtered_3d = pts_cam[in_shrunk]
                    filtered_2d = np.array([u[in_shrunk], v[in_shrunk]]).T
                    filtered_dist = np.linalg.norm(filtered_3d, axis=1)
                    
                    cv2.rectangle(img, (shrunk_bbox[0], shrunk_bbox[1]), 
                                (shrunk_bbox[2], shrunk_bbox[3]), (255, 0, 0), 2)
                    rospy.logdebug(f"After bbox shrink: {len(filtered_3d)} points")
                
                if self.use_sigma_rule and len(filtered_dist) >= 3:
                    filtered_3d = self.apply_sigma_filter(filtered_3d, filtered_dist, self.sigma_rule_std)
                    if len(filtered_3d) > 0:
                        filtered_X = filtered_3d[:, 0]
                        filtered_Y = filtered_3d[:, 1]
                        filtered_Z = filtered_3d[:, 2]
                        filtered_u = (fx * filtered_X / filtered_Z + cx).astype(np.int32)
                        filtered_v = (fy * filtered_Y / filtered_Z + cy).astype(np.int32)
                        filtered_2d = np.array([filtered_u, filtered_v]).T
                        filtered_dist = np.linalg.norm(filtered_3d, axis=1)
                    rospy.logdebug(f"After sigma rule: {len(filtered_3d)} points")
                
                if len(filtered_2d) > 0:
                    # ---- NEW: Compute median distance for this detection ----
                    median_dist = np.median(filtered_dist)
                    det['distance'] = median_dist
                    rospy.logdebug(f"Computed median distance for class {class_id}: {median_dist:.2f} m")
                    
                    all_filtered_points.append({
                        'points_2d': filtered_2d,
                        'distances': filtered_dist,
                        'class': class_id
                    })
                else:
                    # No points after filtering, distance remains None
                    det['distance'] = None
            
            # # Update latest detection summary for CSV logging
            # valid_distances = [det['distance'] for det in target_boxes if det.get('distance') is not None]
            # if valid_distances:
            #     median_distance = float(np.median(valid_distances))
            # else:
            #     median_distance = None
            # self.update_detection_summary(len(boxes), median_distance)

            # Draw points
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
                    cv2.circle(img, (int(points_2d[i][0]), int(points_2d[i][1])), 
                              self.point_size, color, -1)
            
            # Draw bounding boxes and labels with distance
            for det in target_boxes:
                bbox = det['bbox']
                class_id = det['class']
                confidence = det['confidence']
                distance = det.get('distance', None)
                
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Build label: "Class {id} {conf:.2f} {dist:.2f}m" or "N/A"
                if distance is not None:
                    label = f"Class {class_id} {confidence:.2f} {distance:.2f}m"
                else:
                    label = f"Class {class_id} {confidence:.2f} N/A"
                
                cv2.putText(img, label, (bbox[0], bbox[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add info text
            cv2.putText(img, f"Points: {total_points}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Detections: {len(target_boxes)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Publish
            out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            out_msg.header = img_msg.header
            self.pub.publish(out_msg)
            self.save_fused_image_frame(img)
            
            rospy.loginfo(f"Published frame with {total_points} LiDAR points")
            
        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
            import traceback
            traceback.print_exc()

def main():
    rospy.init_node("fusion_lidar_overlay", anonymous=False)
    
    rospy.loginfo("Starting FUSION LiDAR-Image overlay node with YOLO...")
    
    try:
        node = FusionLidarImageOverlay()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")

if __name__ == "__main__":
    main()