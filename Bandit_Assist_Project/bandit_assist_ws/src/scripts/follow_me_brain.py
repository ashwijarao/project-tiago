#!/usr/bin/env python3
import rospy
import cv2
import os
import math
import numpy as np
import socket
import threading
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from std_msgs.msg import String

class TiagoOnboardBrain:
    def __init__(self):
        rospy.init_node('tiago_onboard_brain_node', anonymous=True)
        self.bridge = CvBridge()

        # ---- Configurations ----
        self.target_dist = 1.1       
        self.laser_window = 0.55  # Slightly wider window to handle turning radius safely   
        self.camera_fov_rad = 1.05   

        self.kp_linear = 0.6
        self.kp_angular = 1.5
        self.max_linear_vel = 0.45
        self.max_angular_vel = 0.75

        # ---- Dynamic Tracking Thresholds ----
        self.last_known_distance = self.target_dist
        self.max_jump_distance = 0.4  # Maximum meters a target can move between frames

        # ---- Load Face Detector & Custom Recognizer ----
        cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            cascade_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        model_path = os.path.expanduser('~/PAR_26/src/tiago_hri/src/scripts/me_model.xml')
        if os.path.exists(model_path):
            self.recognizer.read(model_path)
            rospy.loginfo("Loaded custom personal face recognition profile successfully.")
        else:
            rospy.logerr(f"Model file missing at {model_path}. Please run train_faces.py first!")

        # ---- State Machine & Filters ----
        self.state = "IDLE_LOOKING_FOR_FACE"
        self.last_known_angle = 0.0
        self.lost_track_timer = None
        self.consecutive_match_count = 0
        self.required_frames = 8  # STRICT FILTER: Increased from 3 to 8 frames

        # ---- Publishers & Subscribers ----
        self.cmd_vel_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback)
        self.voice_pub = rospy.Publisher("/tiago_hri/voice_command", String, queue_size=10)
        self.nav_cmd_pub = rospy.Publisher('/tiago_delivery/waypoint_command', String, queue_size=10)

        # Start Socket Server for Laptop Voice Connection
        self.server_thread = threading.Thread(target=self.run_socket_server, daemon=True)
        self.server_thread.start()

        rospy.loginfo("Onboard Tracking Brain Operational. Standing by for voice commands...")

    def run_socket_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', 55555))
        server.listen(5)
        while not rospy.is_shutdown():
            try:
                conn, _ = server.accept()
                data = conn.recv(1024).decode('utf-8').strip()
                conn.close()
                if data:
                    self.process_voice_text(data)
            except Exception:
                pass

    def process_voice_text(self, spoken_text):
        clean_text = spoken_text.lower().strip()
        rospy.loginfo(f"Network voice token arrived: '{clean_text}'")
        self.voice_pub.publish(String(data=clean_text))

        # --- UPDATED TO BE MORE FLEXIBLE ---
        # Now matches "collection point", "go to collection point", "move to collection point", etc.
        if "collection point" in clean_text:
            rospy.loginfo("Voice authorization confirmed! Sending TIAGo to Collection Point 5.")
            self.nav_cmd_pub.publish(String(data="collection_point_5"))
            return

        # 2. EMERGENCY STOP CONTROL: Brakes instantly, waits 5 seconds, goes home
        if "stop" in clean_text or "halt" in clean_text:
            rospy.logwarn("!!! EMERGENCY STOP RECEIVED !!! Locking brakes.")
            self.cmd_vel_pub.publish(Twist()) 
            self.state = "IDLE_LOOKING_FOR_FACE" 
            self.consecutive_match_count = 0
            
            rospy.loginfo("Holding position for 5 seconds...")
            rospy.sleep(5.0)
            
            rospy.loginfo("Routing back to Home base now.")
            self.nav_cmd_pub.publish(String(data="home"))
            return

        # 3. HUMAN TRACKING LOOP TRIGGER
        if self.state == "WAITING_FOR_VOICE_CMD":
            if "follow me" in clean_text or "move" in clean_text:
                rospy.loginfo("Identity verified and voice authorized! Tracking physical profile...")
                self.lost_track_timer = rospy.Time.now()
                self.last_known_distance = self.target_dist 
                self.state = "FOLLOWING_LASER"

    def image_callback(self, data):
        if self.state != "IDLE_LOOKING_FOR_FACE":
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            frame_width = cv_image.shape[1]
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            if len(faces) == 0:
                self.consecutive_match_count = max(0, self.consecutive_match_count - 1)
                return

            matched_this_frame = False

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                label, confidence = self.recognizer.predict(face_roi)
                
                # STRICT COMPARISON: Tightened threshold from 78 to 55 distance score
                if label == 1 and confidence < 55:
                    matched_this_frame = True
                    self.consecutive_match_count += 1
                    
                    rospy.loginfo_throttle(1, f"Target detected! Stability: {self.consecutive_match_count}/{self.required_frames} (Score: {confidence:.1f})")
                    
                    if self.consecutive_match_count >= self.required_frames:
                        face_center_x = x + (w / 2.0)
                        pixel_deviation = (frame_width / 2.0) - face_center_x
                        self.last_known_angle = pixel_deviation * (self.camera_fov_rad / frame_width)

                        self.state = "WAITING_FOR_VOICE_CMD"
                        rospy.loginfo("==========================================================")
                        rospy.loginfo(f"🔒 TARGET LOCK CONFIRMED. Standing by for laptop voice trigger...")
                        rospy.loginfo("==========================================================")
                    break
            
            if not matched_this_frame:
                self.consecutive_match_count = max(0, self.consecutive_match_count - 1)
        except Exception:
            pass

    def laser_callback(self, data):
        if self.state != "FOLLOWING_LASER":
            return

        twist_cmd = Twist()
        best_candidate_range = float('inf')
        best_candidate_angle = 0.0
        smallest_delta = float('inf')
        found_target = False

        for i, r in enumerate(data.ranges):
            angle = data.angle_min + (i * data.angle_increment)
            if r < 0.30 or r > 2.5 or math.isnan(r) or math.isinf(r):
                continue

            if abs(angle - self.last_known_angle) < self.laser_window:
                distance_delta = abs(r - self.last_known_distance)
                if distance_delta < self.max_jump_distance:
                    if distance_delta < smallest_delta:
                        smallest_delta = distance_delta
                        best_candidate_range = r
                        best_candidate_angle = angle
                        found_target = True

        if found_target:
            self.last_known_angle = best_candidate_angle
            self.last_known_distance = best_candidate_range
            self.lost_track_timer = rospy.Time.now()

            distance_error = best_candidate_range - self.target_dist
            angular_error = best_candidate_angle

            twist_cmd.linear.x = self.kp_linear * distance_error
            twist_cmd.angular.z = self.kp_angular * angular_error

            if abs(distance_error) < 0.07:
                twist_cmd.linear.x = 0.0

            twist_cmd.linear.x = np.clip(twist_cmd.linear.x, -0.15, self.max_linear_vel)
            twist_cmd.angular.z = np.clip(twist_cmd.angular.z, -self.max_angular_vel, self.max_angular_vel)
        else:
            if (rospy.Time.now() - self.lost_track_timer).to_sec() > 3.5:
                rospy.logwarn("Target vanished from history buffer. Resetting security perimeter...")
                self.state = "IDLE_LOOKING_FOR_FACE"
                self.consecutive_match_count = 0

        self.cmd_vel_pub.publish(twist_cmd)

if __name__ == '__main__':
    node = TiagoOnboardBrain()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

