#!/usr/bin/env python3

import rospy
import cv2
import os
import math
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge


class FollowKnownCamera:

    def __init__(self):
        rospy.init_node("follow_known_camera", anonymous=True)

        self.bridge = CvBridge()

        # ROS topics — match your launch file exactly
        self.image_topic = rospy.get_param("~image_topic", "/xtion/rgb/image_raw")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan_front_raw")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/key_vel")
        self.tts_topic = rospy.get_param("~tts_topic", "/tts_web_command")

        # Face database
        self.face_database_path = rospy.get_param(
            "~face_database_path",
            "/home/pal/Bandit_Assist/bandit_assist_ws/src/tiago_delivery/faces"
        )

        # Target person
        self.target_person = rospy.get_param("~target_person", "Viheet").strip().capitalize()

        # Recognition
        self.match_threshold = rospy.get_param("~match_threshold", 55.0)
        self.required_confirmations = rospy.get_param("~required_confirmations", 3)
        self.confirm_count = 0

        # Movement
        self.max_forward_speed = rospy.get_param("~max_forward_speed", 0.10)
        self.min_forward_speed = rospy.get_param("~min_forward_speed", 0.035)
        self.max_turn_speed = rospy.get_param("~max_turn_speed", 0.18)
        self.turn_direction = rospy.get_param("~turn_direction", 1.0)

        # Tracking
        self.center_tolerance_px = rospy.get_param("~center_tolerance_px", 200)
        self.dead_zone_px = rospy.get_param("~dead_zone_px", 35)

        # Face size distance control
        self.stop_face_height = rospy.get_param("~stop_face_height", 130)
        self.move_face_height = rospy.get_param("~move_face_height", 95)

        # Laser safety
        self.safe_stop_distance = rospy.get_param("~safe_stop_distance", 1.15)
        self.front_angle_window = rospy.get_param("~front_angle_window", 0.50)
        self.front_min_distance = 99.0

        # Lost timeout — how long to wait before going back to SEARCHING
        self.lost_timeout = rospy.get_param("~lost_timeout", 0.35)
        self.last_seen_time = rospy.Time.now()

        # Velocity smoothing
        self.linear_smoothing = rospy.get_param("~linear_smoothing", 0.50)
        self.angular_smoothing = rospy.get_param("~angular_smoothing", 0.45)
        self.target_twist = Twist()
        self.current_twist = Twist()

        # State machine
        # SEARCHING : scan all faces, confirm Viheet before locking on
        # FOLLOWING : re-verify Viheet every frame, stop if he leaves
        self.state = "SEARCHING"
        self.has_greeted = False

        # FIX: flag set True only when Viheet is confirmed in the current frame
        self.target_visible = False

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.tts_pub = rospy.Publisher(self.tts_topic, String, queue_size=1)
        self.status_pub = rospy.Publisher("/tiago_hri/status", String, queue_size=10)

        self.load_face_detector()
        self.known_faces = []
        self.known_names = []
        self.load_known_faces()

        if len(self.known_faces) == 0:
            rospy.logerr("No valid face images loaded. Check the faces folder.")
            rospy.signal_shutdown("No known faces")
            return

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Timer(rospy.Duration(0.1), self.control_timer)

        rospy.loginfo("Follow node started.")
        rospy.loginfo("Target person : {}".format(self.target_person))
        rospy.loginfo("Image topic   : {}".format(self.image_topic))
        rospy.loginfo("Velocity topic: {}".format(self.cmd_vel_topic))
        rospy.loginfo("Known names   : {}".format(self.known_names))

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def load_face_detector(self):
        possible_paths = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
        ]
        cascade_path = None
        for path in possible_paths:
            if os.path.exists(path):
                cascade_path = path
                break

        if cascade_path is None:
            rospy.logerr("Could not find Haar cascade file.")
            rospy.signal_shutdown("Missing Haar cascade")
            return

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            rospy.logerr("Could not load Haar cascade.")
            rospy.signal_shutdown("Face cascade failed")
            return

        rospy.loginfo("Loaded Haar face detector: {}".format(cascade_path))

    def preprocess_face(self, face_gray):
        face = cv2.resize(face_gray, (120, 120))
        face = cv2.equalizeHist(face)
        face = cv2.GaussianBlur(face, (3, 3), 0)
        return face

    def load_known_faces(self):
        if not os.path.exists(self.face_database_path):
            rospy.logerr("Faces folder does not exist: {}".format(self.face_database_path))
            return

        for file in sorted(os.listdir(self.face_database_path)):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(self.face_database_path, file)
            name = os.path.splitext(file)[0].split("_")[0].capitalize()

            img = cv2.imread(path)
            if img is None:
                rospy.logwarn("Could not read image: {}".format(file))
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.10, minNeighbors=3, minSize=(30, 30)
            )
            if len(faces) == 0:
                rospy.logwarn("No face detected in saved image: {}".format(file))
                continue

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = gray[y:y + h, x:x + w]
            self.known_faces.append(self.preprocess_face(face_crop))
            self.known_names.append(name)
            rospy.loginfo("Loaded face: {} from {}".format(name, file))

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def compare_face(self, detected_face_gray):
        """Return (best_name, best_score). Lower score = closer match."""
        processed = self.preprocess_face(detected_face_gray)
        best_score = 999.0
        best_name = "Unknown"
        for known_face, known_name in zip(self.known_faces, self.known_names):
            score = float(np.mean(cv2.absdiff(known_face, processed)))
            if score < best_score:
                best_score = score
                best_name = known_name
        return best_name, best_score

    def find_target_in_faces(self, faces, gray):
        """
        FIX: Check EVERY detected face, not just the largest one.
        Returns (x, y, w, h, score) of best Viheet match, or None.
        """
        best_score = 999.0
        best_box = None
        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            name, score = self.compare_face(face_crop)
            if name == self.target_person and score <= self.match_threshold:
                if score < best_score:
                    best_score = score
                    best_box = (x, y, w, h, score)
        return best_box  # None if Viheet not found

    # ------------------------------------------------------------------
    # Laser callback
    # ------------------------------------------------------------------

    def scan_callback(self, scan):
        min_dist = 99.0
        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) <= self.front_angle_window:
                if 0.05 < r < min_dist:
                    min_dist = r
        self.front_min_distance = min_dist

    # ------------------------------------------------------------------
    # Motion helpers
    # ------------------------------------------------------------------

    def hard_stop(self):
        self.target_twist = Twist()
        self.current_twist = Twist()
        zero = Twist()
        for _ in range(8):
            self.cmd_pub.publish(zero)
            rospy.sleep(0.02)

    def say_once(self):
        if self.has_greeted:
            return
        text = "Hi {}. I see you. I will follow you now.".format(self.target_person)
        rospy.loginfo("Speaking: {}".format(text))
        self.tts_pub.publish(String(data=text))
        self.status_pub.publish(String(data="tracking_{}".format(self.target_person.lower())))
        self.has_greeted = True

    def limit(self, value, lo, hi):
        return max(min(value, hi), lo)

    # ------------------------------------------------------------------
    # Control timer — 10 Hz
    # ------------------------------------------------------------------

    def control_timer(self, event):

        # FIX: If FOLLOWING but Viheet not visible this frame → stop dead.
        # Robot publishes zero every tick until Viheet reappears.
        if self.state == "FOLLOWING" and not self.target_visible:
            time_since_seen = (rospy.Time.now() - self.last_seen_time).to_sec()
            rospy.logwarn_throttle(
                1.0,
                "Viheet not visible — stopped. ({:.1f}s)".format(time_since_seen)
            )
            self.target_twist = Twist()
            self.current_twist = Twist()
            self.cmd_pub.publish(Twist())

            # After lost_timeout, reset to SEARCHING
            if time_since_seen > self.lost_timeout:
                rospy.logwarn("Lost {}. Back to SEARCHING.".format(self.target_person))
                self.state = "SEARCHING"
                self.confirm_count = 0
                self.has_greeted = False
            return

        # Laser safety — hard stop regardless of state
        if self.front_min_distance < self.safe_stop_distance:
            rospy.logwarn_throttle(
                1.0, "Obstacle at {:.2f}m. HARD STOP.".format(self.front_min_distance)
            )
            self.hard_stop()
            return

        # Smooth and publish velocity
        self.current_twist.linear.x = (
            self.linear_smoothing * self.current_twist.linear.x
            + (1.0 - self.linear_smoothing) * self.target_twist.linear.x
        )
        self.current_twist.angular.z = (
            self.angular_smoothing * self.current_twist.angular.z
            + (1.0 - self.angular_smoothing) * self.target_twist.angular.z
        )

        if abs(self.current_twist.linear.x) < 0.005:
            self.current_twist.linear.x = 0.0
        if abs(self.current_twist.angular.z) < 0.005:
            self.current_twist.angular.z = 0.0

        self.current_twist.linear.y = 0.0
        self.cmd_pub.publish(self.current_twist)

    # ------------------------------------------------------------------
    # Image callback — main logic
    # ------------------------------------------------------------------

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("Image conversion failed: {}".format(e))
            return

        image_h, image_w = frame.shape[:2]
        image_center_x = image_w / 2.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30)
        )

        # ==============================================================
        # SEARCHING STATE
        # Stay still. Check all faces. Confirm Viheet N times in a row.
        # ==============================================================
        if self.state == "SEARCHING":
            self.target_visible = False
            self.target_twist = Twist()  # Stay still while searching

            if len(faces) == 0:
                self.confirm_count = 0
                rospy.loginfo_throttle(1.0, "SEARCHING: no faces in frame.")
                return

            # FIX: check every face, not just the largest
            result = self.find_target_in_faces(faces, gray)

            if result is not None:
                self.confirm_count += 1
                rospy.loginfo("Confirming {}: {}/{}".format(
                    self.target_person, self.confirm_count, self.required_confirmations))

                if self.confirm_count >= self.required_confirmations:
                    self.state = "FOLLOWING"
                    self.target_visible = True
                    self.last_seen_time = rospy.Time.now()
                    self.say_once()
                    rospy.loginfo("Target locked. FOLLOWING started.")
            else:
                self.confirm_count = 0
                rospy.loginfo_throttle(1.0, "SEARCHING: faces found but not {}.".format(
                    self.target_person))
            return

        # ==============================================================
        # FOLLOWING STATE
        # FIX: Re-verify Viheet every single frame.
        # Stranger in frame but not Viheet → stop, do NOT follow.
        # No face at all → stop and wait (control_timer handles this).
        # ==============================================================
        if self.state == "FOLLOWING":

            if len(faces) == 0:
                self.target_visible = False
                rospy.logwarn_throttle(1.0, "FOLLOWING: no faces in frame. Stopped.")
                return

            # FIX: check every face for Viheet
            result = self.find_target_in_faces(faces, gray)

            if result is None:
                # Someone else is there but it is NOT Viheet — stop immediately
                self.target_visible = False
                self.target_twist = Twist()
                rospy.logwarn_throttle(
                    1.0,
                    "FOLLOWING: face detected but NOT {}. Stopped.".format(self.target_person)
                )
                return

            # Viheet confirmed — safe to follow
            self.target_visible = True
            self.last_seen_time = rospy.Time.now()

            x, y, w, h, matched_score = result
            face_center_x = x + (w / 2.0)
            error_x = face_center_x - image_center_x
            norm_error = error_x / image_center_x

            new_target = Twist()

            # Turn toward Viheet
            if abs(error_x) < self.dead_zone_px:
                new_target.angular.z = 0.0
            else:
                raw_turn = -norm_error * self.max_turn_speed * self.turn_direction
                new_target.angular.z = self.limit(raw_turn, -self.max_turn_speed, self.max_turn_speed)

            # Forward motion
            if self.front_min_distance < self.safe_stop_distance:
                new_target.linear.x = 0.0          # laser safety
            elif h >= self.stop_face_height:
                new_target.linear.x = 0.0          # close enough
            elif abs(error_x) > self.center_tolerance_px:
                new_target.linear.x = 0.0          # turn first, then move
            elif h < self.move_face_height:
                new_target.linear.x = self.max_forward_speed
            else:
                new_target.linear.x = self.min_forward_speed

            new_target.linear.y = 0.0
            self.target_twist = new_target

            rospy.loginfo_throttle(
                1.0,
                "FOLLOWING {}: score={:.1f} face_h={} err_x={:.0f} "
                "front={:.2f}m vx={:.3f} wz={:.3f}".format(
                    self.target_person, matched_score, h, error_x,
                    self.front_min_distance,
                    new_target.linear.x, new_target.angular.z
                )
            )


if __name__ == "__main__":
    try:
        FollowKnownCamera()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
