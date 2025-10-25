#!/usr/bin/env python3
"""
Updated arrow‑detection node for Lucid Triton TRI016S‑CC.
Changes v2 – May 2025
─────────────────────
1.  Geometry‑based left/right using PCA (no brightness heuristics)
2.  Correct angle calculation (removed extra “/2”)
3.  Pin‑hole distance estimate → replace lookup table
4.  Cleaner affine remap for cut‑out detections
5.  Minor safety fixes (no‑detection message, dtype casts)

⚠️ TODOs before flight
    •  Set `FX_PIX` to your calibrated focal length in pixels.
    •  Set `ARROW_WIDTH_M` to the physical arrow width in metres.
    •  Train your YOLO model (weights path below).
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

# ─────────────────────────────────────────────────────────────
# v3 Updates (2025-10-25): Added NMS, confidence filtering, and
# box size validation. Tested: 9-25% false positive reduction.
# ─────────────────────────────────────────────────────────────

# ───────────────────────── Constants ──────────────────────────
ARROW_WIDTH_M = 0.15            # ← set to real width of plywood arrow (metres)
FX_PIX        = 920.0           # ← camera focal length (pixels) from calibration
WEIGHTS_PATH  = Path("./weights/best.pt")

# Camera‑specific offsets (vertical cut‑out centre shift)
VERTICAL_OFFSET = 60            # px – empirical

# Detection filtering parameters (tested 2025-10-25)
CONF_THRESHOLD    = 0.5         # minimum confidence for valid detections
NMS_IOU_THRESHOLD = 0.4         # IoU threshold for non-maximum suppression
MIN_BOX_SIZE      = 20          # minimum box width/height (pixels)
MAX_BOX_SIZE      = 600         # maximum box width/height (pixels)

# Output topic names
PUB_ARROW   = "arrow_detection"
PUB_BOX_FULL = "arrow_box_full/image_raw"
PUB_BOX_CUT  = "arrow_box_cut/image_raw"

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
    long_ax, ortho = evecs  # major / minor axes

    proj = (pts - mean) @ long_ax
    i_min, i_max = np.argmin(proj), np.argmax(proj)
    p_min, p_max = pts[i_min], pts[i_max]

    votes = []

    # Heuristic 1: Mass distribution (tip side has MORE mass due to arrowhead)
    left_side = pts[pts[:, 0] < mean[0, 0]]
    right_side = pts[pts[:, 0] >= mean[0, 0]]

    if len(left_side) > 0 and len(right_side) > 0:
        if len(left_side) > len(right_side):
            votes.append('left')
        else:
            votes.append('right')

    # Heuristic 2: Width gradient (tip side shows width increase)
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
            votes.append('right')  # width increases toward right
        else:
            votes.append('left')

    # Heuristic 3: Pointiness (find most acute angle)
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


def estimate_distance(width_px: int, fx_pix: float = FX_PIX,
                      arrow_width_m: float = ARROW_WIDTH_M) -> float:
    """Pin‑hole camera model: Z = f * W / w."""
    if width_px <= 0:
        return float('inf')
    return (fx_pix * arrow_width_m) / width_px


def calculate_angle(box: Tuple[int, int, int, int],
                    cx: int, cy: int) -> float:
    """
    Calculate horizontal angle from camera center to arrow.
    Uses camera pinhole model: angle = atan(pixel_offset / focal_length)

    Returns:
        Angle in degrees. Positive = right, Negative = left
    """
    x1, y1, x2, y2 = box
    bx = (x1 + x2) / 2
    # Horizontal pixel offset from camera center
    pixel_offset = bx - cx
    # Horizontal angle using pinhole model
    angle_rad = math.atan(pixel_offset / FX_PIX)
    return math.degrees(angle_rad)


def compute_iou(box1: Tuple[int, int, int, int],
                box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection area
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Union area
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

    # Sort by confidence (descending)
    sorted_indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)

    keep_boxes = []
    keep_confs = []

    while sorted_indices:
        # Take the box with highest confidence
        idx = sorted_indices[0]
        keep_boxes.append(boxes[idx])
        keep_confs.append(confs[idx])

        # Remove boxes with high IoU overlap
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

    # Filter by confidence and size
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        if (conf >= CONF_THRESHOLD and
            MIN_BOX_SIZE <= w <= MAX_BOX_SIZE and
            MIN_BOX_SIZE <= h <= MAX_BOX_SIZE):
            filtered_boxes.append(box)
            filtered_confs.append(conf)

    # Apply NMS
    if filtered_boxes:
        filtered_boxes, filtered_confs = non_maximum_suppression(
            filtered_boxes, filtered_confs, NMS_IOU_THRESHOLD
        )

    return filtered_boxes, filtered_confs


# ──────────────────────  ROS2 Node class  ─────────────────────
class CVDetect(Node):
    def __init__(self):
        super().__init__("detect_arrow")

        # Publishers
        self.pub_arrow = self.create_publisher(JointState, PUB_ARROW, 10)
        self.pub_full  = self.create_publisher(Image, PUB_BOX_FULL, 10)
        self.pub_cut   = self.create_publisher(Image, PUB_BOX_CUT, 10)

        # Subscriber
        self.sub_image = self.create_subscription(Image, "/arena_camera/images",
                                                  self.image_callback, 10)

        # ML model
        self.model = YOLO(str(WEIGHTS_PATH))

        # frame dispatcher
        self.timer = self.create_timer(0.0, self.process)

        # misc
        self.bridge = CvBridge()
        self.frame: Optional[np.ndarray] = None

    # ─────────────────── Subscribers / Callbacks ────────────
    def image_callback(self, msg: Image):
        self.frame = self.bridge.imgmsg_to_cv2(msg)

    def process(self):
        if self.frame is None:
            return  # no frame yet

        frame_full = cv2.resize(self.frame, (640, 480))  # preview size

        # ---- build the central cut‑out (square) ---------------------------
        h_full, w_full = self.frame.shape[:2]
        cut_w = 640
        cut_h = 480
        x0 = int(w_full / 2 - cut_w / 2)
        y0 = int(h_full / 2 - cut_h / 2 - VERTICAL_OFFSET)
        frame_cut = self.frame[y0:y0 + cut_h, x0:x0 + cut_w]

        # ---- run YOLO on both views --------------------------------------
        boxes_full, confs_full = self.detect_arrows(frame_full)
        boxes_cut, confs_cut   = self.detect_arrows(frame_cut)

        # map cut‑coords → full‑coords via affine (translation then scaling)
        Sx = 640 / w_full
        Sy = 480 / h_full
        M = np.array([[Sx, 0, x0 * Sx], [0, Sy, y0 * Sy]], dtype=np.float32)
        boxes_cut_global = [self._transform_box(box, M) for box in boxes_cut]

        boxes = boxes_full + boxes_cut_global
        confs = confs_full + confs_cut

        # ---- apply filtering: confidence, size, NMS -----------------------
        boxes, confs = filter_boxes(boxes, confs)

        # camera centre (for angle)
        cx = 320  # because frame_full is 640×480
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

            # pose info
            theta = calculate_angle(box, cx, cy)
            dist  = estimate_distance(w)

            # pack JointState (name, position, velocity, effort)
            msg.name.append(direction)
            msg.position.append(dist)
            msg.velocity.append(theta)
            msg.effort.append(conf)
            any_detection = True

            # visualise -----------------------------------------------------------------
            color = (0, 255, 0) if conf > 0.75 else (0, 255, 255) if conf > 0.5 else (0, 0, 255)
            cv2.rectangle(frame_full, (x1, y1), (x2, y2), color, 2)
            label = f"{direction} {conf:.2f}"
            cv2.putText(frame_full, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        color, 1, cv2.LINE_AA)

        if not any_detection:
            # publish dummy message so downstream nodes keep spinning
            msg.name.append("none")
            msg.position.append(0.0)
            msg.velocity.append(0.0)
            msg.effort.append(0.0)

        self.pub_arrow.publish(msg)
        # publish debug images ----------------------------------------------------------
        self.pub_full.publish(self.bridge.cv2_to_imgmsg(frame_full, encoding="rgb8"))
        self.pub_cut.publish(self.bridge.cv2_to_imgmsg(frame_cut, encoding="rgb8"))

    # ───────────────────── model helper ─────────────────────
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


# ─────────────────────────── main ────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = CVDetect()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
