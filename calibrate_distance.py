#!/usr/bin/env python3
"""
Interactive Distance Calibration Tool
Helps you collect pixel measurements at known distances
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION (adjust to match your setup)
# ═══════════════════════════════════════════════════════════════════

# Calibration distances (meters)
CALIBRATION_DISTANCES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Detection settings
WEIGHTS_PATH = "eureka_nav_simple/weights/best.pt"
CONF_THRESHOLD = 0.5

# ═══════════════════════════════════════════════════════════════════
# CALIBRATION DATA COLLECTOR
# ═══════════════════════════════════════════════════════════════════

class DistanceCalibrator:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(WEIGHTS_PATH)
        self.measurements = []
        self.current_distance_idx = 0

    def detect_arrow(self, frame):
        """Detect arrow and return box."""
        results = self.model(frame, verbose=False)

        boxes = []
        confs = []

        for b in results[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf.item())

            if conf >= CONF_THRESHOLD:
                boxes.append((x1, y1, x2, y2))
                confs.append(conf)

        if not boxes:
            return None, 0.0

        # Return highest confidence detection
        best_idx = np.argmax(confs)
        return boxes[best_idx], confs[best_idx]

    def process_frame(self, frame):
        """Process frame and show calibration interface."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Detect arrow
        box, conf = self.detect_arrow(frame)

        # Get current target distance
        if self.current_distance_idx < len(CALIBRATION_DISTANCES):
            target_distance = CALIBRATION_DISTANCES[self.current_distance_idx]
        else:
            target_distance = None

        # Draw instructions
        y_offset = 30

        # Progress bar
        progress = len(self.measurements) / len(CALIBRATION_DISTANCES)
        bar_width = 400
        bar_height = 30
        bar_x = 20
        bar_y = y_offset

        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        cv2.rectangle(display, (bar_x, bar_y),
                     (bar_x + int(bar_width * progress), bar_y + bar_height),
                     (0, 255, 0), -1)

        text = f"Progress: {len(self.measurements)}/{len(CALIBRATION_DISTANCES)}"
        cv2.putText(display, text, (bar_x + 10, bar_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_offset += 60

        if target_distance is not None:
            # Show target distance
            text = f"TARGET: Place arrow at {target_distance:.1f} meters"
            cv2.putText(display, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            y_offset += 40

            # Show measurement instructions
            text = "Use measuring tape! Press SPACE when ready"
            cv2.putText(display, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
        else:
            # Calibration complete
            text = "CALIBRATION COMPLETE!"
            cv2.putText(display, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            y_offset += 50

            text = "Press 's' to save, 'q' to quit"
            cv2.putText(display, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw detected arrow
        if box is not None:
            x1, y1, x2, y2 = box
            box_w = x2 - x1
            box_h = y2 - y1

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Show pixel measurements prominently
            text_y = y1 - 10 if y1 > 100 else y2 + 30

            # Large text for pixel measurements
            text = f"WIDTH: {box_w} px"
            cv2.putText(display, text, (x1, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            text = f"HEIGHT: {box_h} px"
            cv2.putText(display, text, (x1, text_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            text = f"Conf: {conf:.2f}"
            cv2.putText(display, text, (x1, text_y + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        else:
            # No detection
            text = "NO ARROW DETECTED"
            cv2.putText(display, text, (w//2 - 200, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Show previous measurements
        y_list = h - 30 * min(len(self.measurements), 8) - 20
        cv2.putText(display, "Measurements:", (20, y_list),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_list += 25

        for i, (dist, width, height) in enumerate(self.measurements[-8:]):
            text = f"{dist:.1f}m: {width}px × {height}px"
            cv2.putText(display, text, (20, y_list + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return display, box

    def record_measurement(self, distance, box):
        """Record a calibration measurement."""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        self.measurements.append((distance, width, height))
        print(f"✓ Recorded: {distance}m → {width}px × {height}px")

        self.current_distance_idx += 1

    def save_calibration(self, filename="calibration_data.json"):
        """Save calibration data to file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'measurements': [
                {
                    'distance_m': dist,
                    'pixel_width': width,
                    'pixel_height': height
                }
                for dist, width, height in self.measurements
            ],
            'camera': 'Lucid TRI016S-CC',
            'lens': '2.8-12mm varifocal at 2.8mm',
            'processing_resolution': '640x480'
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Calibration saved to {filename}")

        # Also generate Python code
        self.generate_python_code("calibration_config_generated.py")

    def generate_python_code(self, filename="calibration_config_generated.py"):
        """Generate Python calibration config file."""
        with open(filename, 'w') as f:
            f.write("# Auto-generated calibration configuration\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")

            f.write("CALIBRATION_TABLE = [\n")
            f.write("    # (distance_m, width_px, height_px)\n")
            for dist, width, height in self.measurements:
                f.write(f"    ({dist:.1f}, {width:3d}, {height:3d}),\n")
            f.write("]\n")

        print(f"✓ Python config saved to {filename}")
        print("\nTo use this calibration:")
        print(f"  1. Copy CALIBRATION_TABLE from {filename}")
        print(f"  2. Paste into calibration_config.py")
        print(f"  3. Run your detection system!")


def main():
    """Main calibration loop."""
    print("="*70)
    print("DISTANCE CALIBRATION TOOL")
    print("="*70)
    print("\nInstructions:")
    print("  1. Place arrow at specified distance (use measuring tape!)")
    print("  2. Ensure arrow is clearly visible to camera")
    print("  3. Press SPACE to record measurement")
    print("  4. Repeat for all distances")
    print("  5. Press 's' to save when complete")
    print("\nCalibration distances:")
    for i, dist in enumerate(CALIBRATION_DISTANCES, 1):
        print(f"  {i}. {dist:.1f} meters")
    print("\nPress any key to start...")

    # Initialize calibrator
    calibrator = DistanceCalibrator()

    # Open camera or video file
    use_camera = input("\nUse live camera? (y/n): ").lower() == 'y'

    if use_camera:
        cap = cv2.VideoCapture(0)  # Adjust camera index if needed
        if not cap.isOpened():
            print("Error: Could not open camera!")
            return
    else:
        video_path = input("Enter video file path: ")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

    print("\nCalibration started!")
    print("Controls:")
    print("  SPACE - Record measurement at current distance")
    print("  's' - Save calibration")
    print("  'q' - Quit")
    print("  'r' - Reset measurements")

    while True:
        ret, frame = cap.read()
        if not ret:
            if not use_camera:
                # Restart video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("Error: Could not read frame")
                break

        # Resize to processing resolution
        frame = cv2.resize(frame, (640, 480))

        # Process and display
        display, box = calibrator.process_frame(frame)
        cv2.imshow('Distance Calibration', display)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            # Record measurement
            if calibrator.current_distance_idx < len(CALIBRATION_DISTANCES):
                if box is not None:
                    target_dist = CALIBRATION_DISTANCES[calibrator.current_distance_idx]
                    calibrator.record_measurement(target_dist, box)
                else:
                    print("✗ No arrow detected! Cannot record measurement.")
            else:
                print("✓ All measurements complete! Press 's' to save.")

        elif key == ord('s'):
            # Save calibration
            if len(calibrator.measurements) > 0:
                calibrator.save_calibration()
                print("\n✓ Calibration complete!")
                break
            else:
                print("✗ No measurements to save!")

        elif key == ord('r'):
            # Reset
            calibrator.measurements = []
            calibrator.current_distance_idx = 0
            print("⟳ Measurements reset")

    cap.release()
    cv2.destroyAllWindows()

    print("\nCalibration session ended.")


if __name__ == "__main__":
    main()
