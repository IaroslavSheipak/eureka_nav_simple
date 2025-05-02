#!/usr/bin/env python3

#Developed by Andrei Smirnov. 2024
#MSU Rover Team. Voltbro. NIIMech 

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node
import time
import threading
import numpy as np

class arrow_class():
    def __init__(self, direction = 'left',range = 5.0, angle = 0.0, certainty = 1.0):
        self.range = range
        self.direction = direction
        self.angle = angle
        self.certainty = certainty
        self.repetition = 0

class nav_simple(Node):
    def __init__(self):
        super().__init__('nav_simple')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.sub = self.create_subscription(JointState, "arrow_detection", self.callback, 10)
        self.sub = self.create_subscription(JointState, "autonomous_commands", self.callback_2, 10)
        self.threshold_range = 1.5
        self.autonomus_mode = 0
        self.p_gain = 0.05
        self.arrow_status = 0
        self.arrow_direction = list()
        self.arrow_angle = list()
        self.arrow_range = list()
        self.arrow_certainty = list()
        self.spin_direction = 'left'
        self.maximum_range = 10.0
        self.get_logger().info("Nav_Simple Started!")
    def __del__(self):
        self.get_logger().info("Nav_Simple Killed!")
    def callback(self, data):
        self.arrow_direction = data.name
        self.arrow_range = data.position
        self.arrow_angle = data.velocity
        self.arrow_certainty = data.effort
    def callback_2(self, data):
        self.autonomus_mode = data.position[list(data.name).index('autonomous_mode')]
    def arrow_filter(self, range_min, range_max, angle_max, certainty_min):
        if(len(self.arrow_range) > 0):
            np_cert = np.array(self.arrow_certainty)
            np_range = np.array(self.arrow_range)
            np_angle = np.array(self.arrow_angle)
            np_direction = np.array(self.arrow_direction)
            np_mask = np_range > range_min
            np_mask3 = np_range < range_max
            np_mask2 = np.abs(np_angle) < angle_max
            new_mask = np.logical_and(np.logical_and(np_mask, np_mask2), np_mask3)
            np_cert_filtered = np_cert[new_mask]
            np_range_filtered = np_range[new_mask]
            np_angle_filtered = np_angle[new_mask]
            np_direction_filtered = np_direction[new_mask]
            if(len(np_cert_filtered) > 0):
                index = np.argmax(np_cert_filtered)
                if(np_cert_filtered[index] > certainty_min):
                    return (np_range_filtered[index], np_angle_filtered[index], np_cert_filtered[index], np_direction[index])
        return None
    def find_arrow(self):
        detection_ctr = 0
        message = Twist()
        direction = 1   #default direction
        detection = 0   
        # if arrow near me is present, spin in the direction it points 
        
        for c in range(len(self.arrow_range)):
            if(self.arrow_range[c] < 2.0):
                if (self.arrow_direction[c] == 'right' and self.arrow_certainty[c] > 0.5): 
                    print("Right arrow!")
                    self.spin_direction = 'right'
                break
        if(self.spin_direction == 'right'):
            direction = -1
        message.angular.z = 100.0
        message.linear.x = float(direction * 0.05)
        self.pub.publish(message)
        # spin until an arrow far away is detected and it is centered in the frame
        arrow = None
        while(detection_ctr < 10 and self.autonomus_mode == 1):
            arrow = self.arrow_filter(2, 10, 15, .6)
            if(arrow == None):
                detection_ctr = 0
            detection_ctr += 1
            self.pub.publish(message)
            time.sleep(0.1)
        #stop spinning
        if arrow is not None:
            self.maximum_range = arrow[0] + 0.5
            self.minimum_range = arrow[0] - 0.5
        message.linear.x = 0.0
        message.angular.z = 0.0
        self.pub.publish(message)
        #new target found!!!
        return  
    def approach_arrow(self):
        message = Twist()
        message.linear.x = 0.07
        arrival = 0
        error = 0
        counter = 0
        lost_ctr = 0
        self.maximum_range = 10
        self.minimum_range = 0
        while(arrival < 1 and self.autonomus_mode == 1):
            print(self.arrow_angle)
            #check if we arrived
            arrow = self.arrow_filter(1.0, 1.5, 100, .6)
            if(arrow != None and self.maximum_range < 2.0):
                self.spin_direction = arrow[3]
                print("Arrived!")
                print(arrow[0])
                time.sleep(5)
                arrival = 1
                break
            #if not arrived, update the angle error
            arrow = self.arrow_filter(self.minimum_range, self.maximum_range, 100, .6)
            if (arrow != None):
                lost_ctr = 0
                self.maximum_range = arrow[0] + 0.5
                self.minimum_range = arrow[0] - 0.5
                error = arrow[1]
            else:
                #if arrow is lost stop and start search again
                lost_ctr += 1
            if(lost_ctr > 10):
                message.linear.x = 0.0
                message.angular.z = 0.0
                self.pub.publish(message)
                print(self.maximum_range)
                print("Arrow Lost!!!")
                return
            message.angular.z = float(-error * self.p_gain)
            self.pub.publish(message)
            counter += 1
            time.sleep(0.1)
        #and stop next to arrow
        message.linear.x = 0.0
        message.angular.z = 0.0
        self.pub.publish(message)
        return
    #MAIN PIPELINE FUNCTION
    def pipeline(self):
        while (True):
 #           print(self.autonomus_mode)
            if(self.autonomus_mode == 1):
                self.find_arrow()       #find arrow
                time.sleep(2)
                self.approach_arrow()   #approach arrow
            time.sleep(5)           #wait next to arrow

def main(args=None):
    rclpy.init()
    ns = nav_simple()
    th = threading.Thread(target=ns.pipeline)
    th.daemon=True
    th.start()
    rclpy.spin(ns)
 #   ns.pipeline()

    
    ns.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()