#!/usr/bin/env python3

import cv2
import math
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class Arrow_detection(Node):
    def __init__(self):
        super().__init__('arrow_detection')
        self.model = YOLO('/home/eurekanuc/ros2_ws/src/eureka_nav_simple/eureka_nav_simple/weights/best.pt')
        self.reference_distances = [2, 4]
        self.reference_heights = [50, 25]  # known heights at 2m and 4m
        self.publish_ = self.create_publisher(JointState, 'arrow_detection',10)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )  
        self.subs= self.create_subscription(Image, '/hazcam/image_raw', self.timer_callback, qos_profile)

    def timer_callback(self,msg):
        frame = self.convert_ros_image_to_cv2(msg)
        arrow_info, height = self.detect_arrows_with_angles_and_distance_and_direction(frame)

        msg = JointState()
        msg.name = []
        msg.position = []
        msg.velocity = []
        msg.effort = []

        for info in arrow_info:
            box, angle, distance, direction = info
            if direction == 'left' or direction == 'right':
                msg.name.append(str(direction))  
                msg.position.append(float(distance))    
                msg.velocity.append(float(angle))       
                msg.effort.append(float(self.calculate_inclination_angle(distance, height)))  
        if not arrow_info: 
            msg.name.append("No_detection")  
            msg.position.append(float(0.0))    
            msg.velocity.append(float(0.0))       
            msg.effort.append(float(0.0))  
        self.publish_.publish(msg)  

    def convert_ros_image_to_cv2(self, ros_image):
        return np.frombuffer(ros_image.data, dtype=np.uint8).reshape(ros_image.height, ros_image.width, -1)    

    def calculate_inclination_angle(self, length, height):
        original_aspect_ratio = 1.5
        detected_aspect_ratio = length / height
        cos_theta = detected_aspect_ratio / original_aspect_ratio
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta_rad = math.acos(cos_theta)
        theta_deg = math.degrees(theta_rad)
        return theta_deg

    def detect_arrows(self, img):
        results = self.model(source=img, imgsz=(max(1080, img.shape[0]), max(720, img.shape[1])))
        boxes_coordinates = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                coordinates = (int(x1), int(y1), int(x2), int(y2))
                boxes_coordinates.append(coordinates)
        return boxes_coordinates

    def calculate_angle(self, img, box, cx, cy):
        x1, y1, x2, y2 = box
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2
        angle_radians = math.atan2(box_center_x - cx, cy - box_center_y)/2 
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    def estimate_distance(self, height, reference_heights, reference_distances):
        if height == 0:
            return None
        ratio = reference_heights[0] / height
        estimated_distance = reference_distances[0] * ratio
        return estimated_distance

    def calculate_blackness(self, roi):
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_roi)

    def determine_arrow_direction(self, img, box):
        x1, y1, x2, y2 = box
        arrow_roi = img[y1:y2, x1:x2]
        width = arrow_roi.shape[1]
        left_half = arrow_roi[:, :width // 2]
        right_half = arrow_roi[:, width // 2:]
        left_blackness = self.calculate_blackness(left_half)
        right_blackness = self.calculate_blackness(right_half)
        if left_blackness < right_blackness:
            return 'left'
        else:
            return 'right'

    def detect_arrows_with_angles_and_distance_and_direction(self, img):
        boxes = self.detect_arrows(img)
        img_height, img_width = img.shape[:2]
        cx = img_width // 2
        cy = img_height
        arrow_info = []
        height = 0
        for box in boxes:
            angle = self.calculate_angle(img, box, cx, cy)
            height = box[3] - box[1]
            distance = self.estimate_distance(height, self.reference_heights, self.reference_distances)
            direction = self.determine_arrow_direction(img, box)
            arrow_info.append((box, angle, distance, direction))
        return arrow_info, height

def main(args=None):
    rclpy.init(args=args)
    node = Arrow_detection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
