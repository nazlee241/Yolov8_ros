#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import threading
from sklearn.cluster import DBSCAN

# ==================== Simple Tracker ====================
class ObjectTracker:
    def __init__(self, max_age=5, alpha=0.7, association_dist=0.5):
        self.max_age = max_age
        self.alpha = alpha
        self.association_dist = association_dist
        self.objects = {}          # id -> {'centroid': np.array, 'corners': np.array, 'age': int}
        self.next_id = 0

    def update(self, detections):
        """
        detections: list of dict with keys 'centroid' (3D) and 'corners' (8x3)
        Returns: list of dict with keys 'id', 'centroid', 'corners'
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
                obj = self.objects[best_id]
                obj['centroid'] = self.alpha * det['centroid'] + (1 - self.alpha) * obj['centroid']
                obj['corners'] = det['corners']   # update directly (could also smooth)
                obj['age'] = 0
                matched_ids.add(best_id)
                det['id'] = best_id
            else:
                det['id'] = self.next_id
                self.next_id += 1
                self.objects[det['id']] = {
                    'centroid': det['centroid'].copy(),
                    'corners': det['corners'].copy(),
                    'age': 0
                }

        # Age unmatched
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
                'centroid': obj['centroid'],
                'corners': obj['corners']
            })
        return result

# ==================== Main Node ====================
class LidarObjectDetectorRViz:
    def __init__(self):
        self.lock = threading.Lock()

        # ROS parameters
        self.lidar_topic = rospy.get_param("~lidar_topic", "/ouster/points")
        self.marker_topic = rospy.get_param("~marker_topic", "/detected_objects_markers")
        self.frame_id = rospy.get_param("~frame_id", "os_sensor")   # fallback

        # LiDAR processing
        self.ground_threshold = rospy.get_param("~ground_threshold", 0.2)
        self.cluster_eps = rospy.get_param("~cluster_eps", 0.3)
        self.cluster_min_samples = rospy.get_param("~cluster_min_samples", 5)
        self.target_height = rospy.get_param("~target_height", 1.5)
        self.height_tolerance = rospy.get_param("~height_tolerance", 0.3)

        # Tracker parameters
        self.tracker_alpha = rospy.get_param("~tracker_alpha", 0.7)
        self.association_dist = rospy.get_param("~association_dist", 0.5)
        self.tracker = ObjectTracker(alpha=self.tracker_alpha,
                                     association_dist=self.association_dist)

        # Publisher
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=5)

        # Latest data
        self.latest_pts = None
        self.latest_header = None
        self.received_lidar = False

        rospy.loginfo("Subscribing to LiDAR topic: %s", self.lidar_topic)
        rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)

        # Process timer (10 Hz)
        self.processing_timer = rospy.Timer(rospy.Duration(0.1), self.process_callback)

        rospy.loginfo("="*60)
        rospy.loginfo("LIDAR OBJECT DETECTOR – Publishing markers to RViz")
        rospy.loginfo("="*60)
        rospy.loginfo(f"Ground threshold: {self.ground_threshold} m")
        rospy.loginfo(f"Cluster eps: {self.cluster_eps}, min_samples: {self.cluster_min_samples}")
        rospy.loginfo(f"Target height: {self.target_height} ± {self.height_tolerance} m")
        rospy.loginfo(f"Tracker alpha: {self.tracker_alpha}, association dist: {self.association_dist}")
        rospy.loginfo("="*60)

        rospy.Timer(rospy.Duration(5.0), self.status_callback)

    def status_callback(self, event):
        rospy.loginfo(f"=== STATUS ===")
        rospy.loginfo(f"LiDAR received: {self.received_lidar}")
        rospy.loginfo(f"Tracked objects: {len(self.tracker.objects)}")
        rospy.loginfo("==============")

    def lidar_callback(self, msg):
        try:
            pts_list = []
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                pts_list.append([p[0], p[1], p[2]])
            if pts_list:
                pts = np.array(pts_list, dtype=np.float32)
                with self.lock:
                    self.latest_pts = pts
                    self.latest_header = msg.header
                    self.received_lidar = True
                rospy.logdebug(f"Received {len(pts)} LiDAR points")
        except Exception as e:
            rospy.logerr(f"LiDAR callback error: {e}")

    def process_callback(self, event):
        with self.lock:
            if not self.received_lidar:
                return
            pts = self.latest_pts.copy()
            header = self.latest_header

        try:
            # -- 1. Ground removal --
            ground_mask = pts[:, 2] > self.ground_threshold
            pts_above = pts[ground_mask]
            if len(pts_above) == 0:
                rospy.logwarn("No points above ground")
                # Clear markers?
                self.publish_markers([])
                return

            # -- 2. DBSCAN clustering --
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(pts_above)
            labels = clustering.labels_
            unique_labels = set(labels)

            # -- 3. Extract candidate clusters with height ~ target_height --
            detections = []
            for label in unique_labels:
                if label == -1:
                    continue
                cluster_pts = pts_above[labels == label]
                centroid = np.mean(cluster_pts, axis=0)
                if abs(centroid[2] - self.target_height) > self.height_tolerance:
                    continue

                # PCA for oriented bounding box
                mean = np.mean(cluster_pts, axis=0)
                cov = np.cov(cluster_pts, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                idx = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, idx]
                projected = cluster_pts @ eigvecs
                min_ext, max_ext = np.min(projected, axis=0), np.max(projected, axis=0)

                # 8 corners in local principal frame, then transformed to LiDAR frame
                corners_local = np.array([
                    [min_ext[0], min_ext[1], min_ext[2]],
                    [min_ext[0], min_ext[1], max_ext[2]],
                    [min_ext[0], max_ext[1], min_ext[2]],
                    [min_ext[0], max_ext[1], max_ext[2]],
                    [max_ext[0], min_ext[1], min_ext[2]],
                    [max_ext[0], min_ext[1], max_ext[2]],
                    [max_ext[0], max_ext[1], min_ext[2]],
                    [max_ext[0], max_ext[1], max_ext[2]]
                ])
                corners_lidar = mean + (corners_local @ eigvecs.T)

                detections.append({
                    'centroid': mean,
                    'corners': corners_lidar
                })

            # -- 4. Update tracker --
            tracked = self.tracker.update(detections)

            # -- 5. Build markers --
            marker_array = MarkerArray()
            for obj in tracked:
                centroid = obj['centroid']
                corners = obj['corners']
                obj_id = obj['id']
                distance = np.linalg.norm(centroid)   # LiDAR range

                # ---- Wireframe box (LINE_LIST) ----
                marker = Marker()
                marker.header = header
                marker.ns = "object_boxes"
                marker.id = obj_id
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD
                marker.scale.x = 0.05   # line width
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 1.0  # cyan

                # Define edges (12 edges of the box)
                edges = [
                    (0,1), (2,3), (4,5), (6,7),  # axis 2 (z)
                    (0,2), (1,3), (4,6), (5,7),  # axis 1 (y)
                    (0,4), (1,5), (2,6), (3,7)   # axis 0 (x)
                ]
                points = []
                for i, j in edges:
                    p1 = Point()
                    p1.x, p1.y, p1.z = corners[i]
                    p2 = Point()
                    p2.x, p2.y, p2.z = corners[j]
                    points.append(p1)
                    points.append(p2)
                marker.points = points
                marker_array.markers.append(marker)

                # ---- Distance label (TEXT_VIEW_FACING) ----
                label_marker = Marker()
                label_marker.header = header
                label_marker.ns = "object_labels"
                label_marker.id = obj_id
                label_marker.type = Marker.TEXT_VIEW_FACING
                label_marker.action = Marker.ADD
                label_marker.text = f"{distance:.2f} m"
                label_marker.pose.position.x = centroid[0]
                label_marker.pose.position.y = centroid[1]
                label_marker.pose.position.z = centroid[2] + 0.3   # offset above centroid
                label_marker.scale.z = 0.3   # font size
                label_marker.color.a = 1.0
                label_marker.color.r = 1.0
                label_marker.color.g = 1.0
                label_marker.color.b = 1.0
                marker_array.markers.append(label_marker)

            # -- 6. Publish markers --
            self.publish_markers(marker_array)

        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
            import traceback
            traceback.print_exc()

    def publish_markers(self, marker_array):
        self.marker_pub.publish(marker_array)

def main():
    rospy.init_node("lidar_object_detector_rviz", anonymous=False)
    rospy.loginfo("Starting LiDAR Object Detector for RViz...")
    try:
        node = LidarObjectDetectorRViz()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()