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

# ───────────────────────── Constants ──────────────────────────
ARROW_WIDTH_M = 0.15            # ← set to real width of plywood arrow (metres)
FX_PIX        = 920.0           # ← camera focal length (pixels) from calibration
WEIGHTS_PATH  = Path("./weights/best.pt")

# Camera‑specific offsets (vertical cut‑out centre shift)
VERTICAL_OFFSET = 60            # px – empirical

# Output topic names
PUB_ARROW   = "arrow_detection"
PUB_BOX_FULL = "arrow_box_full/image_raw"
PUB_BOX_CUT  = "arrow_box_cut/image_raw"

# ───────────────────── Geometry helpers ───────────────────────

def arrow_direction_pca(roi: np.ndarray) -> Optional[str]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw  = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts          = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    mean, evecs  = cv2.PCACompute(pts, mean=None)
    long_ax, ortho = evecs                # major / minor axes (unit vectors)

    proj         = (pts - mean) @ long_ax # 1-D coordinate along major axis
    i_min, i_max = np.argmin(proj), np.argmax(proj)
    p_min, p_max = pts[i_min], pts[i_max]

    # --- pick the thinner extreme as arrow head --------------------------
    def local_width(p):
        strip = np.abs(((pts - p) @ long_ax)) < 4.0
        if not strip.any():
            return 1e9               # huge width → never picked as tip
        return np.ptp((pts[strip] - p) @ ortho)

    tip  = p_min if local_width(p_min) < local_width(p_max) else p_max
    tail = p_max if tip is p_min else p_min
    return "right" if tip[0] > tail[0] else "left"


def estimate_distance(width_px: int, fx_pix: float = FX_PIX,
                      arrow_width_m: float = ARROW_WIDTH_M) -> float:
    """Pin‑hole camera model: Z = f * W / w."""
    if width_px <= 0:
        return float('inf')
    return (fx_pix * arrow_width_m) / width_px


def calculate_angle(box: Tuple[int, int, int, int],
                    cx: int, cy: int) -> float:
    x1, y1, x2, y2 = box
    bx = (x1 + x2) / 2
    by = (y1 + y2) / 2
    # angle wrt camera optical axis (positive right, degrees)
    angle_rad = math.atan2(by - cy, bx - cx)
    return math.degrees(angle_rad)


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
