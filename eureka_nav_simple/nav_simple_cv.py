#!/usr/bin/env python3
"""
ROS2 обертка для Eureka Navigation Library.
Интегрирует библиотеку визуального распознавания с ROS2 для передачи
данных навигации в модуль локализации.
"""

from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

# Импорт библиотеки навигации
from eureka_nav_lib import NavigationDetector, DetectionResult

# Импорт конфигурации калибровки
from calibration_config import (
    validate_calibration_table,
    SHOW_PIXEL_WIDTH,
    SHOW_PIXEL_HEIGHT,
    SHOW_CALIBRATION_STATUS,
    TEXT_COLOR_CALIBRATED,
    TEXT_COLOR_EXTRAPOLATED,
    TEXT_COLOR_OUT_OF_RANGE,
    get_distance_from_pixels
)

# ───────────────────────── Constants ──────────────────────────
WEIGHTS_PATH = "./weights/best.pt"

# Camera-specific offsets
VERTICAL_OFFSET = 60

# Output topic names (now handles both arrows and cones)
PUB_DETECTION = "navigation_detection"
PUB_BOX_FULL = "navigation_box_full/image_raw"
PUB_BOX_CUT = "navigation_box_cut/image_raw"


# ──────────────────────  ROS2 Node class  ─────────────────────
class CVDetect(Node):
    def __init__(self):
        super().__init__("detect_navigation")

        # Validate calibration on startup
        valid, message = validate_calibration_table()
        if not valid:
            self.get_logger().warn(f"Calibration validation: {message}")
        else:
            self.get_logger().info(f"Calibration loaded: {message}")

        # Initialize Navigation Detector from library
        self.detector = NavigationDetector(WEIGHTS_PATH)
        device_info = f"CUDA ({self.detector.device})" if "cuda" in self.detector.device else "CPU"
        self.get_logger().info(f"Navigation Detector initialized on {device_info}")

        # Publishers
        self.pub_detection = self.create_publisher(JointState, PUB_DETECTION, 10)
        self.pub_full = self.create_publisher(Image, PUB_BOX_FULL, 10)
        self.pub_cut = self.create_publisher(Image, PUB_BOX_CUT, 10)

        # Subscriber
        self.sub_image = self.create_subscription(Image, "/arena_camera/images",
                                                  self.image_callback, 10)

        # Frame dispatcher
        self.timer = self.create_timer(0.0, self.process)

        # Misc
        self.bridge = CvBridge()
        self.frame: Optional[np.ndarray] = None

    def image_callback(self, msg: Image):
        self.frame = self.bridge.imgmsg_to_cv2(msg)

    def process(self):
        if self.frame is None:
            return

        frame_full = cv2.resize(self.frame, (640, 480))

        # Build central cut-out
        h_full, w_full = self.frame.shape[:2]
        cut_w = 640
        cut_h = 480
        x0 = int(w_full / 2 - cut_w / 2)
        y0 = int(h_full / 2 - cut_h / 2 - VERTICAL_OFFSET)
        frame_cut = self.frame[y0:y0 + cut_h, x0:x0 + cut_w]

        # Use library to detect all objects (arrows AND cones) on both views
        detections_full = self.detector.detect_all(frame_full)
        detections_cut = self.detector.detect_all(frame_cut)

        # IMPORTANT: Both detections are in 640x480 coordinate space
        # But frame_cut was extracted from a different region of the original image
        # So we should NOT combine them - they would create duplicates!
        # The old code properly transformed cut coordinates, but with the library approach
        # it's better to just use one or the other, or implement proper NMS across both.

        # For now, just use full frame detections (covers whole image)
        all_detections = detections_full

        # Camera centre
        cx = 320

        # Build JointState message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        any_detection = False
        for det in all_detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            # Pack JointState (compatible with existing format)
            # name: "object_type:direction" (e.g., "arrow:left", "cone:none")
            # position: distance, velocity: angle, effort: confidence
            detection_label = f"{det.object_type}:{det.direction}"
            msg.name.append(detection_label)
            msg.position.append(det.distance_m)
            msg.velocity.append(det.angle_deg)
            msg.effort.append(det.confidence)
            any_detection = True

            # Get calibration status for visualization (pass object_type)
            _, status = get_distance_from_pixels(w, h, use_width=True, object_type=det.object_type)
            color = self._get_color_for_status(status, det.confidence)

            # Visualize on frame_full
            cv2.rectangle(frame_full, (x1, y1), (x2, y2), color, 2)

            # Main label (show object type and direction/confidence)
            if det.object_type == "arrow":
                label = f"{det.object_type}:{det.direction} {det.confidence:.2f}"
            else:
                label = f"{det.object_type} {det.confidence:.2f}"
            cv2.putText(frame_full, label, (x1, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

            # Distance and angle
            info_text = f"D:{det.distance_m:.2f}m A:{det.angle_deg:.1f}deg"
            cv2.putText(frame_full, info_text, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # PIXEL MEASUREMENTS (for exhibition/calibration)
            pixel_y = y1 + 15
            if SHOW_PIXEL_WIDTH:
                text = f"W:{w}px"
                cv2.putText(frame_full, text, (x1, pixel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
                pixel_y += 18

            if SHOW_PIXEL_HEIGHT:
                text = f"H:{h}px"
                cv2.putText(frame_full, text, (x1, pixel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
                pixel_y += 18

            if SHOW_CALIBRATION_STATUS:
                status_short = status[:4] if len(status) > 4 else status
                cv2.putText(frame_full, status_short, (x1, pixel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        if not any_detection:
            msg.name.append("none")
            msg.position.append(0.0)
            msg.velocity.append(0.0)
            msg.effort.append(0.0)

        self.pub_detection.publish(msg)
        self.pub_full.publish(self.bridge.cv2_to_imgmsg(frame_full, encoding="rgb8"))
        self.pub_cut.publish(self.bridge.cv2_to_imgmsg(frame_cut, encoding="rgb8"))

    def _get_color_for_status(self, status: str, conf: float):
        """Get visualization color based on calibration status."""
        if status == "calibrated":
            return TEXT_COLOR_CALIBRATED
        elif status.startswith("extrapolated"):
            return TEXT_COLOR_EXTRAPOLATED
        else:
            return TEXT_COLOR_OUT_OF_RANGE


def main(args=None):
    rclpy.init(args=args)
    node = CVDetect()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
