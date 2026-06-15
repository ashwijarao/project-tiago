#!/usr/bin/env python3
import rospy
import cv2
import os
import math
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import String

class TiagoOnboardBrain:
    def __init__(self):
        rospy.init_node('tiago_onboard_brain_node', anonymous=True)
        self.bridge = CvBridge()

        # ---- Configurations ----
        self.target_dist = 1.1       # Distance to keep from your legs (meters)
        self.laser_window = 0.45     # Leg-tracking angle window (radians)
        self.camera_fov_rad = 1.05   # TIAGo camera horizontal FOV

        # P-Controller Gains for Smooth Tracking
        self.kp_linear = 0.55
        self.kp_angular = 1.4
        self.max_linear_vel = 0.45
        self.max_angular_vel = 0.75

        # ---- Load Built-in OpenCV Face Detector ----
        cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            cascade_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        rospy.loginfo(f"Loaded built-in OpenCV Face Detector from: {cascade_path}")

        # ---- State Machine ----
        self.state = "IDLE_LOOKING_FOR_FACE"
        self.last_known_angle = 0.0
        self.lost_track_timer = None

        # ---- Publishers & Subscribers ----
        self.cmd_vel_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback)

        self.voice_sub = rospy.Subscriber("/tiago_hri/voice_command", String, self.voice_callback)

        rospy.loginfo("Onboard Tracking Brain Ready. Stand in front of TIAGo's head camera...")

    def image_callback(self, data):
        if self.state != "IDLE_LOOKING_FOR_FACE":
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            small_frame = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5)
            frame_width = small_frame.shape[1]
            gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_center_x = x + (w / 2.0)

                pixel_deviation = (frame_width / 2.0) - face_center_x
                self.last_known_angle = pixel_deviation * (self.camera_fov_rad / frame_width)

                self.state = "WAITING_FOR_VOICE_CMD"
                rospy.loginfo("==========================================================")
                rospy.loginfo("STATUS: Face detected! Handshake initiated.")
                rospy.loginfo("ACTION REQUIRED: Please give the 'follow me' command to your laptop.")
                rospy.loginfo("==========================================================")

        except Exception as e:
            rospy.logerr(f"Onboard Vision error: {e}")

    def voice_callback(self, msg):
        if self.state != "WAITING_FOR_VOICE_CMD":
            return

        spoken_text = msg.data.lower()
        rospy.loginfo(f"Network voice token arrived: '{spoken_text}'")

        if "follow me" in spoken_text or "move" in spoken_text or "tiago" in spoken_text:
            rospy.loginfo("Voice authorized! Brakes released. Tracking legs via LIDAR now...")
            self.lost_track_timer = rospy.Time.now()
            self.state = "FOLLOWING_LASER"

    def laser_callback(self, data):
        if self.state != "FOLLOWING_LASER":
           return

        twist_cmd = Twist()
        closest_range = float('inf')
        closest_angle = 0.0
        found_target = False

        for i, r in enumerate(data.ranges):
            angle = data.angle_min + (i * data.angle_increment)
            if r < 0.35 or r > 2.2 or math.isnan(r) or math.isinf(r):
                continue

            if abs(angle - self.last_known_angle) < self.laser_window:
                if r < closest_range:
                    closest_range = r
                    closest_angle = angle
                    found_target = True

        if found_target:
            self.last_known_angle = closest_angle
            self.lost_track_timer = rospy.Time.now()

            distance_error = closest_range - self.target_dist
            angular_error = closest_angle

            twist_cmd.linear.x = self.kp_linear * distance_error
            twist_cmd.angular.z = self.kp_angular * angular_error

            if abs(distance_error) < 0.08:
                twist_cmd.linear.x = 0.0

            twist_cmd.linear.x = np.clip(twist_cmd.linear.x, -0.15, self.max_linear_vel)
            twist_cmd.angular.z = np.clip(twist_cmd.angular.z, -self.max_angular_vel, self.max_angular_vel)
        else:
            if (rospy.Time.now() - self.lost_track_timer).to_sec() > 4.0:
                rospy.logwarn("Target lost. Safely resetting back to face look-out.")
                self.state = "IDLE_LOOKING_FOR_FACE"

        self.cmd_vel_pub.publish(twist_cmd)

if __name__ == '__main__':
    node = TiagoOnboardBrain()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
