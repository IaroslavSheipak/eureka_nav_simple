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
        self.tracked_arrows = list()
        self.maximum_range = 10.0
        self.get_logger().info("Nav_Simple Started!")
    def __del__(self):
        self.get_logger().info("Nav_Simple Killed!")
    def callback(self, data):
    #    print("callback!!!")
#        new_direction = np.array(data.name)
#        new_range = np.array(data.position)
#        new_angle = np.array(data.velocity)
#        new_certainty = np.array(data.effort)
#        np_certainty = np.array(self.arrow_certainty)
#        np_range = np.array(self.arrow_range)
#        np_angle = np.array(self.arrow_angle)
#        np_direction = np.array(self.arrow_direction)

  #      for c in range(len(self.tracked_arrows)):
  #          for c2 in range(len(data.position)):
   #             if(abs(self.tracked_arrows[c].range - data.position[c2]) < 0.2 
  #              and abs(self.tracked_arrows[c].angle - data.velocity[c2]) < 5.0 
   #             and self.tracked_arrows[c].direction == data.name[c2]):
                    #assume its the same arrow
   #                 self.tracked_arrows[c].range = data.position[ind]
   #                 self.tracked_arrows[c].range = data.position[ind]

        self.arrow_direction = data.name
        self.arrow_range = data.position
        self.arrow_angle = data.velocity
        self.arrow_certainty = data.effort
    def callback_2(self, data):
    #    print("callback2!")
        self.autonomus_mode = data.position[list(data.name).index('autonomous_mode')]
    def find_arrow(self):
        message = Twist()
        direction = 1   #default direction
        detection = 0   
        # if arrow near me is present, spin in the direction it points 
        
        for c in range(len(self.arrow_range)):
            if(self.arrow_range[c] < 2.0):
                if (self.arrow_direction[c] == 'right' and self.arrow_certainty[c] > 0.5): 
                    print("Right arrow!")
                    direction = -1
                break
        message.angular.z = 100.0
        message.linear.x = float(direction * 0.05)
        self.pub.publish(message)
        # spin until an arrow far away is detected and it is centered in the frame
        while(detection < 1 and self.autonomus_mode == 1):
            if(len(self.arrow_certainty) > 0):
                np_cert = np.array(self.arrow_certainty)
                np_range = np.array(self.arrow_range)
                np_angle = np.array(self.arrow_angle)
                np_mask = np_range > 2.0
                np_mask2 = np.abs(np_angle) < 15.0 
                np_cert_filtered = np_cert[np.logical_and(np_mask, np_mask2)]
                np_range_filtered = np_range[np.logical_and(np_mask, np_mask2)]
                np_angle_filtered = np_angle[np.logical_and(np_mask, np_mask2)]
                if(len(np_cert_filtered) > 0):
                    index = np.argmax(np_cert_filtered)
                    if(np_cert_filtered[index] > 0.6):
                        detection = 1
                        self.maximum_range = np_range_filtered[index]
            self.pub.publish(message)
            time.sleep(0.1)
        while(detection < 1 and self.autonomus_mode == 1):
            print(self.arrow_range)
            for c in range(len(self.arrow_range)):
                if(self.arrow_range[c] > 2.0 and self.arrow_certainty[c] > 0.6 and abs(self.arrow_angle[c]) < 15.0):
                    detection += 1
                    self.maximum_range = self.arrow_range[c]
                    break
            self.pub.publish(message)
            time.sleep(0.1)
        #stop spinning
        
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
        while(arrival < 1 and self.autonomus_mode == 1):
            print(self.arrow_angle)
            #check if we arrived
            for c in range(len(self.arrow_range)):
                #need to drive for at least 3 seconds before the next stop
                if(self.arrow_range[c] < 1.5 and self.arrow_range[c] > 1.0 and counter > 50 and self.arrow_certainty[c] > 0.5):
                    print("Arrived!")
                    print(self.arrow_range[c])
                    time.sleep(5)
                    arrival = 1
                    break
            #if not arrived, update the angle error
            for c in range(len(self.arrow_range)):
                if(self.arrow_range[c] > 1.5 and self.arrow_range[c] < self.maximum_range and self.arrow_certainty[c] > 0.5):
                    error = self.arrow_angle[c]
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