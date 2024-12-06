#!/usr/bin/env python3


import cv2
import math
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

class CV_detect(Node):
 
    def __init__(self):
        super().__init__('detect_arrow')
        self.publisher_ = self.create_publisher(JointState, 'arrow_detection',10)
        self.publisher_box_full = self.create_publisher(Image, 'arrow_box_full/image_raw',10)
        self.publisher_box_cut = self.create_publisher(Image, 'arrow_box_cut/image_raw',10)
        self.subscriber = self.create_subscription(Image, "hazcam/image_raw", self.image_callback, 10)
        self.model = YOLO('/home/eurekanuc/ros2_ws/src/eureka_nav_simple/eureka_nav_simple/weights/best.pt')
        self.reference_distances = [0.5, 1, 2, 4, 8]  # in meters
        self.reference_heights = [125, 94, 63, 31, 15]   # in pixels at 2m and 4m respectively
        self.original_aspect_ratio = 1.7
        self.video_input_path = '/dev/hazcam'
    #    self.cap = cv2.VideoCapture(self.video_input_path, cv2.CAP_V4L2)
        self.timer = self.create_timer(0.0, self.callback)
        self.bridge = CvBridge()
        self.image_flag = 0
    def image_callback(self, msg):
        self.image_flag = 1
        self.frame = self.bridge.imgmsg_to_cv2(msg)
        return
        

    def callback(self):
        if(self.image_flag < 1):
            return
        msg = JointState()
        frame_full = cv2.resize(self.frame, (640, 480))
        width = self.frame.shape[0]
        height = self.frame.shape[1]
        frame_cut = self.frame[ int(width/2-480/2): int(width/2 + 480/2) ,int(height/2 - 640/2):int(height/2 + 640/2)]

        processed_frame, arrow_info = self.process_frame(frame_full, frame_cut, 640/1920, 480/1080)
        for info in arrow_info:
            box, angle, distance, direction, inclination_angle, conf = info
            if direction == 'left' or direction == 'right':
                msg.name.append(str(direction))  
                msg.position.append(float(distance))    
                msg.velocity.append(float(angle))       
                msg.effort.append(float(conf)) 
        if msg is None:
            msg.name.append(str('No_detection')) 
            msg.position.append(float(0))    
            msg.velocity.append(float(0))       
            msg.effort.append(float(0)) 
        self.publisher_.publish(msg)
        
        ros_frame = self.bridge.cv2_to_imgmsg(processed_frame, encoding='rgb8')
        self.publisher_box_full.publish(ros_frame)

       


    def detect_arrows(self, img):
        results = self.model(img)
        boxes_coordinates = []
        confidences = []
        boxes = results[0].boxes  # Accessing the first (and only) result
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()  # Extract confidence score
            coordinates = (int(x1), int(y1), int(x2), int(y2))
            boxes_coordinates.append(coordinates)
            confidences.append(confidence)
        return boxes_coordinates, confidences

    def calculate_angle(self, box, cx, cy):
        x1, y1, x2, y2 = box
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2
        dx = box_center_x - cx
        dy = cy - box_center_y  # In image coordinates, y increases downward
        angle_radians = math.atan2(dx, dy)/2
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    def estimate_distance(self, height, reference_heights, reference_distances):
        if height == 0:
            return None
        distance = np.interp(height, reference_heights[::-1], reference_distances[::-1])
        return distance

    def calculate_brightness(self, roi):
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_roi)

    def determine_arrow_direction(self, arrow_roi):
        width = arrow_roi.shape[1]
        left_half = arrow_roi[:, :width // 2]
        right_half = arrow_roi[:, width // 2:]
        left_brightness = self.calculate_brightness(left_half)
        right_brightness = self.calculate_brightness(right_half)
        if left_brightness < right_brightness:
            return 'left'
        else:
            return 'right'

    def calculate_inclination_angle(self, length, height):
        detected_aspect_ratio = length / height if height != 0 else 0
        cos_theta = detected_aspect_ratio / self.original_aspect_ratio if self.original_aspect_ratio != 0 else 0
        cos_theta = max(-1.0, min(1.0, cos_theta))  # Clamp the value between -1 and 1
        theta_rad = math.acos(cos_theta)
        theta_deg = math.degrees(theta_rad)
        return theta_deg

    def process_frame(self, img, img2, ratio_hor, ratio_vert):
        boxes, confidences = self.detect_arrows(img)
        boxes2, confidences2 = self.detect_arrows(img2)
        boxes_new = []
        for c in range(len(boxes2)):
            temp = list(boxes2[c])
            if(not(temp[0] < 10 or temp[2] > 630 or temp[1] < 10 or temp [3] > 470)):
                temp[0]+=640
                temp[0]*=ratio_hor
                temp[2]+=640 
                temp[2]*=ratio_hor
                temp[1]+=300
                temp[1]*=ratio_vert
                temp[3]+=300
                temp[3]*=ratio_vert
                boxes_new.append(tuple(map(int, temp)))

        #print(confidences)
        boxes.extend(boxes_new)
        confidences.extend(confidences2)
        img_height, img_width = img.shape[:2]
        cx = img_width // 2
        cy = img_height 
        arrow_info = []
        print(boxes)
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            arrow_roi = img[y1:y2, x1:x2]
            length = x2 - x1  # Width of the bounding box
            height = y2 - y1  # Height of the bounding box
            inclination_angle = self.calculate_inclination_angle(length, height)
            angle = self.calculate_angle(box, cx, cy)
            distance = self.estimate_distance(height, self.reference_heights, self.reference_distances)
            direction = self.determine_arrow_direction(arrow_roi)
            arrow_info.append((box, angle, distance, direction, inclination_angle, conf))
            if conf > 0.75:
                color = (0, 255, 0)  # Green
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

        # Drawing the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Create label with direction and confidence
            label = f"{direction.capitalize()} Arrow {conf:.2f}"

        # Choose font and get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Calculate label position
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10

        # Draw filled rectangle as background for label
            cv2.rectangle(img, (label_x, label_y - label_height - baseline),
                      (label_x + label_width, label_y + baseline), color, cv2.FILLED)

        # Put label text above the bounding box
            cv2.putText(img, label, (label_x, label_y),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # (Optional) Draw additional annotations below the box
        # You can adjust or remove these as needed
            additional_info = (
            f"Angle: {angle:.1f}°",
            f"Incl: {inclination_angle:.1f}°",
            f"Dist: {distance:.2f}m"
        )
            for i, info in enumerate(additional_info):
                text = info
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = x1
                text_y = y2 + 20 + i * 20  # Start 20 pixels below the box
            # Ensure text stays within frame
                text_y = min(text_y, img_height - 10)
                cv2.putText(img, text, (text_x, text_y),
                        font, font_scale, color, thickness, cv2.LINE_AA)

        # (Optional) Print confidence to console
            print(f"Detected {direction} arrow with confidence {conf:.2f}")

        return img, arrow_info
    
    def merge_frames(self, processed_frame, arrow_info):
        
        return

def main(args=None):
    rclpy.init(args=args)
    cv_arrow = CV_detect()
    rclpy.spin(cv_arrow)
    cv_arrow.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()