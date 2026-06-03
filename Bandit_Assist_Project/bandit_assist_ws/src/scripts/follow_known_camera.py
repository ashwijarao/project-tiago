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

        # ROS topics

        self.image_topic = rospy.get_param("~image_topic", "/xtion/rgb/image_raw")

        self.scan_topic = rospy.get_param("~scan_topic", "/scan_front_raw")

        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/key_vel")

        self.tts_topic = rospy.get_param("~tts_topic", "/tts_web_command")

        # Face database folder

        self.face_database_path = rospy.get_param(

            "~face_database_path",

            "/home/pal/Bandit_Assist/bandit_assist_ws/src/tiago_delivery/faces"

        )

        # Target person name from image filenames, e.g. viheet_1.jpg -> Viheet

        self.target_person = rospy.get_param("~target_person", "Viheet").strip().capitalize()

        # Recognition settings

        # Lower = stricter. Higher = easier.

        self.match_threshold = rospy.get_param("~match_threshold", 55.0)

        self.required_confirmations = rospy.get_param("~required_confirmations", 3)

        self.confirm_count = 0

        # Movement settings

        self.max_forward_speed = rospy.get_param("~max_forward_speed", 0.08)

        self.min_forward_speed = rospy.get_param("~min_forward_speed", 0.03)

        self.max_turn_speed = rospy.get_param("~max_turn_speed", 0.10)

        # If robot turns opposite direction, set to -1.0 in launch

        self.turn_direction = rospy.get_param("~turn_direction", 1.0)

        # Face tracking settings

        self.center_tolerance_px = rospy.get_param("~center_tolerance_px", 220)

        self.dead_zone_px = rospy.get_param("~dead_zone_px", 50)

        # Face size distance control

        # Lower stop_face_height = robot stops farther away.

        self.stop_face_height = rospy.get_param("~stop_face_height", 150)

        self.move_face_height = rospy.get_param("~move_face_height", 115)

        # Safety stop from laser

        self.safe_stop_distance = rospy.get_param("~safe_stop_distance", 1.00)

        self.front_angle_window = rospy.get_param("~front_angle_window", 0.45)

        self.front_min_distance = 99.0

        # Stop fast when target is lost

        self.lost_timeout = rospy.get_param("~lost_timeout", 0.25)

        self.last_seen_time = rospy.Time.now()

        # Velocity smoothing

        self.linear_smoothing = rospy.get_param("~linear_smoothing", 0.65)

        self.angular_smoothing = rospy.get_param("~angular_smoothing", 0.70)

        self.target_twist = Twist()

        self.current_twist = Twist()

        # State machine

        # SEARCHING: recognise trained person

        # FOLLOWING: follow locked face position

        self.state = "SEARCHING"

        self.has_greeted = False

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

        rospy.loginfo("Final face-follow node started.")

        rospy.loginfo("Target person: {}".format(self.target_person))

        rospy.loginfo("Image topic: {}".format(self.image_topic))

        rospy.loginfo("Velocity topic: {}".format(self.cmd_vel_topic))

        rospy.loginfo("Known names: {}".format(self.known_names))

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

        rospy.loginfo("Loaded Haar face detector.")

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

                gray,

                scaleFactor=1.10,

                minNeighbors=3,

                minSize=(30, 30)

            )

            if len(faces) == 0:

                rospy.logwarn("No face detected in saved image: {}".format(file))

                continue

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            face_crop = gray[y:y+h, x:x+w]

            face_processed = self.preprocess_face(face_crop)

            self.known_faces.append(face_processed)

            self.known_names.append(name)

            rospy.loginfo("Loaded face sample for {} from {}".format(name, file))

    def compare_face(self, detected_face):

        detected_processed = self.preprocess_face(detected_face)

        best_score = 999.0

        best_name = "Unknown"

        for known_face, known_name in zip(self.known_faces, self.known_names):

            diff = cv2.absdiff(known_face, detected_processed)

            score = float(np.mean(diff))

            if score < best_score:

                best_score = score

                best_name = known_name

        return best_name, best_score

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

    def zero_twist(self):

        return Twist()

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

    def limit(self, value, min_value, max_value):

        return max(min(value, max_value), min_value)

    def control_timer(self, event):

        # If locked target is lost, stop immediately

        if self.state == "FOLLOWING":

            time_since_seen = (rospy.Time.now() - self.last_seen_time).to_sec()

            if time_since_seen > self.lost_timeout:

                rospy.logwarn_throttle(1.0, "Target timeout. HARD STOP.")

                self.state = "SEARCHING"

                self.confirm_count = 0

                self.has_greeted = False

                self.hard_stop()

                return

        # Front laser safety

        if self.front_min_distance < self.safe_stop_distance:

            rospy.logwarn_throttle(

                1.0,

                "Too close in front: {:.2f} m. HARD STOP.".format(self.front_min_distance)

            )

            self.state = "SEARCHING"

            self.confirm_count = 0

            self.has_greeted = False

            self.hard_stop()

            return

        # Smooth command

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

        # Never publish sideways movement in this stable version

        self.current_twist.linear.y = 0.0

        self.cmd_pub.publish(self.current_twist)

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

            gray,

            scaleFactor=1.08,

            minNeighbors=3,

            minSize=(30, 30)

        )

        # CRITICAL: if face is lost while following, hard stop immediately

        if len(faces) == 0:

            if self.state == "FOLLOWING":

                rospy.logwarn_throttle(1.0, "Face lost. HARD STOP.")

                self.state = "SEARCHING"

                self.confirm_count = 0

                self.has_greeted = False

                self.hard_stop()

            else:

                self.confirm_count = 0

                self.target_twist = Twist()

                rospy.loginfo_throttle(1.0, "Searching: no face detected.")

            return

        # Use largest face

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        face_crop = gray[y:y+h, x:x+w]

        # SEARCHING: recognise target first

        if self.state == "SEARCHING":

            name, score = self.compare_face(face_crop)

            rospy.loginfo_throttle(

                1.0,

                "Recognition check: best={} score={:.1f}".format(name, score)

            )

            if name == self.target_person and score <= self.match_threshold:

                self.confirm_count += 1

                rospy.loginfo_throttle(

                    1.0,

                    "Confirming {}: {}/{}".format(

                        self.target_person,

                        self.confirm_count,

                        self.required_confirmations

                    )

                )

                if self.confirm_count >= self.required_confirmations:

                    self.state = "FOLLOWING"

                    self.last_seen_time = rospy.Time.now()

                    self.say_once()

                    rospy.loginfo("Target locked. Following started.")

            else:

                self.confirm_count = 0

                self.target_twist = Twist()

            return

        # FOLLOWING: follow face position only; stop if lost handled above

        if self.state == "FOLLOWING":

            self.last_seen_time = rospy.Time.now()

            face_center_x = x + (w / 2.0)

            error_x = face_center_x - image_center_x

            norm_error = error_x / image_center_x

            new_target = Twist()

            # Turn toward face

            if abs(error_x) < self.dead_zone_px:

                new_target.angular.z = 0.0

            else:

                raw_turn = -norm_error * self.max_turn_speed * self.turn_direction

                new_target.angular.z = self.limit(

                    raw_turn,

                    -self.max_turn_speed,

                    self.max_turn_speed

                )

            # Stop if person/object is too close

            if self.front_min_distance < self.safe_stop_distance:

                new_target.linear.x = 0.0

            # Stop when face is large enough

            elif h >= self.stop_face_height:

                new_target.linear.x = 0.0

            # Move forward only if target is reasonably centred

            else:

                if abs(error_x) > self.center_tolerance_px:

                    new_target.linear.x = 0.0

                else:

                    if h < self.move_face_height:

                        new_target.linear.x = self.max_forward_speed

                    else:

                        new_target.linear.x = self.min_forward_speed

            # No sideways movement

            new_target.linear.y = 0.0

            self.target_twist = new_target

            rospy.loginfo_throttle(

                1.0,

                "Following: face_h={}, error_x={:.1f}, front={:.2f}, target_vx={:.2f}, target_wz={:.2f}, current_vx={:.2f}, current_wz={:.2f}".format(

                    h,

                    error_x,

                    self.front_min_distance,

                    self.target_twist.linear.x,

                    self.target_twist.angular.z,

                    self.current_twist.linear.x,

                    self.current_twist.angular.z

                )

            )

if __name__ == "__main__":

    try:

        FollowKnownCamera()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass