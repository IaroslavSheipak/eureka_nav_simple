#!/usr/bin/env python3
"""
Arrow detection node with CALIBRATED distance measurement.
Uses piecewise linear interpolation from lookup table.
"""

import math
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

# Import calibration configuration
from calibration_config import (
    get_distance_from_pixels,
    get_angle_from_position,
    SHOW_PIXEL_WIDTH,
    SHOW_PIXEL_HEIGHT,
    SHOW_CALIBRATION_STATUS,
    TEXT_COLOR_CALIBRATED,
    TEXT_COLOR_EXTRAPOLATED,
    TEXT_COLOR_OUT_OF_RANGE,
    validate_calibration_table
)

# ───────────────────────── Constants ──────────────────────────
WEIGHTS_PATH = Path("./weights/best.pt")

# Camera‑specific offsets
VERTICAL_OFFSET = 60

# Detection filtering parameters
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.4
MIN_BOX_SIZE = 20
MAX_BOX_SIZE = 600

# Output topic names
PUB_ARROW = "arrow_detection"
PUB_BOX_FULL = "arrow_box_full/image_raw"
PUB_BOX_CUT = "arrow_box_cut/image_raw"


# ───────────────────── Geometry helpers ───────────────────────

def arrow_direction_pca(roi: np.ndarray) -> Optional[str]:
    """
    Improved arrow direction detection using multiple heuristics (v3 - 2025-10-25).
    Uses majority vote from: (1) mass distribution, (2) width gradient, (3) pointiness.
    """
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnts:
        return None

    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return None

    # PCA to find major axis
    mean, evecs = cv2.PCACompute(pts, mean=None)
    long_ax, ortho = evecs

    proj = (pts - mean) @ long_ax
    i_min, i_max = np.argmin(proj), np.argmax(proj)
    p_min, p_max = pts[i_min], pts[i_max]

    votes = []

    # H1: Mass distribution (tip has MORE mass due to arrowhead)
    left_side = pts[pts[:, 0] < mean[0, 0]]
    right_side = pts[pts[:, 0] >= mean[0, 0]]

    if len(left_side) > 0 and len(right_side) > 0:
        if len(left_side) > len(right_side):
            votes.append('left')
        else:
            votes.append('right')

    # H2: Width gradient (tip shows width increase)
    num_samples = 5
    proj_sorted = np.sort(proj)
    widths = []
    for i in range(num_samples):
        idx = min(int(len(proj_sorted) * i / (num_samples - 1)), len(proj_sorted) - 1)
        proj_val = proj_sorted[idx]
        strip = np.abs(proj - proj_val) < 2.0
        if strip.any():
            width = np.ptp((pts[strip] - mean) @ ortho)
            widths.append(width)

    if len(widths) >= 3:
        x = np.arange(len(widths))
        slope = np.polyfit(x, widths, 1)[0]
        if slope > 0:
            votes.append('right')
        else:
            votes.append('left')

    # H3: Pointiness (find most acute angle)
    hull = cv2.convexHull(pts.astype(np.int32), returnPoints=True)
    hull_pts = hull.reshape(-1, 2).astype(np.float32)

    def find_hull_angle(p_extreme):
        dists = np.linalg.norm(hull_pts - p_extreme, axis=1)
        idx = np.argmin(dists)
        n = len(hull_pts)
        p1 = hull_pts[(idx - 1) % n]
        p2 = hull_pts[idx]
        p3 = hull_pts[(idx + 1) % n]
        v1 = p1 - p2
        v2 = p3 - p2
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm > 0:
            angle = np.arccos(np.clip(dot / norm, -1.0, 1.0))
            return np.degrees(angle)
        return 180.0

    angle_min = find_hull_angle(p_min)
    angle_max = find_hull_angle(p_max)

    if angle_min < angle_max:
        votes.append('left' if p_min[0] < p_max[0] else 'right')
    else:
        votes.append('left' if p_max[0] < p_min[0] else 'right')

    # Majority vote
    if not votes:
        return "right" if p_max[0] > p_min[0] else "left"

    left_votes = votes.count('left')
    right_votes = votes.count('right')
    return 'left' if left_votes > right_votes else 'right'


def compute_iou(box1: Tuple[int, int, int, int],
                box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def non_maximum_suppression(boxes: List[Tuple[int, int, int, int]],
                            confs: List[float],
                            iou_threshold: float = NMS_IOU_THRESHOLD) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    if not boxes:
        return [], []

    sorted_indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    keep_boxes = []
    keep_confs = []

    while sorted_indices:
        idx = sorted_indices[0]
        keep_boxes.append(boxes[idx])
        keep_confs.append(confs[idx])

        remaining = []
        for other_idx in sorted_indices[1:]:
            iou = compute_iou(boxes[idx], boxes[other_idx])
            if iou < iou_threshold:
                remaining.append(other_idx)

        sorted_indices = remaining

    return keep_boxes, keep_confs


def filter_boxes(boxes: List[Tuple[int, int, int, int]],
                confs: List[float]) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """Filter boxes by confidence, size, and apply NMS."""
    filtered_boxes = []
    filtered_confs = []

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        if (conf >= CONF_THRESHOLD and
            MIN_BOX_SIZE <= w <= MAX_BOX_SIZE and
            MIN_BOX_SIZE <= h <= MAX_BOX_SIZE):
            filtered_boxes.append(box)
            filtered_confs.append(conf)

    if filtered_boxes:
        filtered_boxes, filtered_confs = non_maximum_suppression(
            filtered_boxes, filtered_confs, NMS_IOU_THRESHOLD
        )

    return filtered_boxes, filtered_confs


# ──────────────────────  ROS2 Node class  ─────────────────────
class CVDetect(Node):
    def __init__(self):
        super().__init__("detect_arrow")

        # Validate calibration on startup
        valid, message = validate_calibration_table()
        if not valid:
            self.get_logger().warn(f"Calibration validation: {message}")
        else:
            self.get_logger().info(f"Calibration loaded: {message}")

        # Publishers
        self.pub_arrow = self.create_publisher(JointState, PUB_ARROW, 10)
        self.pub_full = self.create_publisher(Image, PUB_BOX_FULL, 10)
        self.pub_cut = self.create_publisher(Image, PUB_BOX_CUT, 10)

        # Subscriber
        self.sub_image = self.create_subscription(Image, "/arena_camera/images",
                                                  self.image_callback, 10)

        # ML model
        self.model = YOLO(str(WEIGHTS_PATH))

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

        # Build central cut‑out
        h_full, w_full = self.frame.shape[:2]
        cut_w = 640
        cut_h = 480
        x0 = int(w_full / 2 - cut_w / 2)
        y0 = int(h_full / 2 - cut_h / 2 - VERTICAL_OFFSET)
        frame_cut = self.frame[y0:y0 + cut_h, x0:x0 + cut_w]

        # Run YOLO on both views
        boxes_full, confs_full = self.detect_arrows(frame_full)
        boxes_cut, confs_cut = self.detect_arrows(frame_cut)

        # Map cut‑coords → full‑coords
        Sx = 640 / w_full
        Sy = 480 / h_full
        M = np.array([[Sx, 0, x0 * Sx], [0, Sy, y0 * Sy]], dtype=np.float32)
        boxes_cut_global = [self._transform_box(box, M) for box in boxes_cut]

        boxes = boxes_full + boxes_cut_global
        confs = confs_full + confs_cut

        # Apply filtering
        boxes, confs = filter_boxes(boxes, confs)

        # Camera centre
        cx = 320
        cy = 240

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        any_detection = False
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            roi = frame_full[y1:y2, x1:x2]
            direction = arrow_direction_pca(roi)
            if direction is None:
                continue

            # CALIBRATED distance measurement
            bx = (x1 + x2) / 2
            dist, status = get_distance_from_pixels(w, h, use_width=True)

            # CALIBRATED angle measurement
            theta = get_angle_from_position(bx, cx)

            # Pack JointState
            msg.name.append(direction)
            msg.position.append(dist)
            msg.velocity.append(theta)
            msg.effort.append(conf)
            any_detection = True

            # Visualize with PIXEL MEASUREMENTS
            color = self._get_color_for_status(status, conf)

            cv2.rectangle(frame_full, (x1, y1), (x2, y2), color, 2)

            # Main label
            label = f"{direction} {conf:.2f}"
            cv2.putText(frame_full, label, (x1, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

            # Distance and angle
            info_text = f"D:{dist:.2f}m A:{theta:.1f}deg"
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

        self.pub_arrow.publish(msg)
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

    def detect_arrows(self, img: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """Run YOLO, return boxes in (x1,y1,x2,y2) on *this* image size."""
        results = self.model(img)
        boxes_out, confs = [], []
        for b in results[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes_out.append((x1, y1, x2, y2))
            confs.append(float(b.conf.item()))
        return boxes_out, confs

    @staticmethod
    def _transform_box(box: Tuple[int, int, int, int], M: np.ndarray) -> Tuple[int, int, int, int]:
        """Apply 2×3 affine to all four corners and return int‑bbox."""
        x1, y1, x2, y2 = box
        pts = np.float32([[x1, y1], [x2, y2]]).reshape(-1, 1, 2)
        pts_t = cv2.transform(pts, M).reshape(-1, 2)
        (nx1, ny1), (nx2, ny2) = pts_t
        return int(nx1), int(ny1), int(nx2), int(ny2)


def main(args=None):
    rclpy.init(args=args)
    node = CVDetect()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
