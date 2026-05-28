#!/usr/bin/env python3
import rospy
import cv2
import face_recognition
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import actionlib
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from std_msgs.msg import String # To signal your navigation/following node

class TiagoPerceptionNode:
    def __init__(self):
        rospy.init_node('tiago_person_recognition', anonymous=True)
        self.bridge = CvBridge()
        
        # Publisher to trigger your "Follow Me" behavior node later
        self.status_pub = rospy.Publisher('/tiago_hri/status', String, queue_size=10)
        
        # Load Known Faces
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces("/path/to/your/faces/directory") # Update this path!
        
        self.greeted_people = set() 
        self.image_sub = rospy.Subscriber("/head_front_camera/rgb/image_raw", Image, self.image_callback)
        rospy.loginfo("TIAGo Face Recognition Node Initialized.")

    def load_known_faces(self, directory):
        for file in os.listdir(directory):
            if file.endswith(".jpg") or file.endswith(".png"):
                name = os.path.splitext(file)[0]
                img_path = os.path.join(directory, file)
                image = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(image)[0]
                
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name.capitalize())
        rospy.loginfo(f"Loaded {len(self.known_face_names)} authorized profiles: {self.known_face_names}")

    def say_greeting(self, name):
        rospy.loginfo(f"Greeting: Hi {name}")
        client = actionlib.SimpleActionClient('/tts', TtsAction)
        if client.wait_for_server(rospy.Duration(2.0)):
            goal = TtsGoal()
            goal.rawtext.text = f"Hi {name}, nice to meet you. I will now follow you."
            goal.rawtext.lang_id = "en_GB"
            client.send_goal(goal)
            client.wait_for_result()
        else:
            rospy.logwarn("TTS server not available!")

    def image_callback(self, data):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # Shrink image to 1/4 size for much faster processing on the robot CPU
            small_frame = cv2.resize(cv_image, (0, 0), fx=0.25, fy=0.25)
            # Convert BGR to RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all faces in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                # Compare face with our 4 authorized people
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                if name != "Unknown":
                    if name not in self.greeted_people:
                        self.say_greeting(name)
                        self.greeted_people.add(name)
                        # Publish status change to move into Phase 3 (Following)
                        self.status_pub.publish(f"start_following_{name.lower()}")
                else:
                    rospy.logthrottle(5, "Detected unauthorized individual (5th person). Ignoring.")
                    
        except Exception as e:
            rospy.logerr(f"Error in image processing: {e}")

if __name__ == '__main__':
    node = TiagoPerceptionNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down face recognition node.")