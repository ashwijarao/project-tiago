#!/usr/bin/env python3
import rospy
import cv2
import os
import re
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FaceCollector:
    def __init__(self):
        rospy.init_node('face_collector_node')
        self.bridge = CvBridge()
        
        cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            cascade_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.output_dir = os.path.expanduser('~/PAR_26/src/tiago_hri/src/scripts/face_data')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # ---- SMART INDEX CHECKING ----
        # Look at existing files to find the highest number used so far
        existing_files = os.listdir(self.output_dir)
        highest_index = 0
        for f in existing_files:
            match = re.search(r'user_(\d+)\.jpg', f)
            if match:
                index = int(match.group(1))
                if index > highest_index:
                    highest_index = index
        
        # Set our starting point based on what's already in the folder
        self.start_count = highest_index
        self.current_session_captured = 0
        self.max_per_session = 40  # Captures 40 new images *per run*
        
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.image_callback)
        rospy.loginfo("==========================================================")
        rospy.loginfo(f"📁 Existing dataset contains: {self.start_count} images.")
        rospy.loginfo(f"📸 Ready to append {self.max_per_session} new images starting at index {self.start_count + 1}!")
        rospy.loginfo("==========================================================")

    def image_callback(self, data):
        if self.current_session_captured >= self.max_per_session:
            total_now = self.start_count + self.current_session_captured
            rospy.loginfo(f"🎉 Session complete! Total images in dataset: {total_now}")
            rospy.loginfo("You can safely press Ctrl+C.")
            self.image_sub.unregister() # Stop processing frames
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                self.current_session_captured += 1
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                
                # Safe unique filename based on global tracking numbers
                global_index = self.start_count + self.current_session_captured
                file_path = os.path.join(self.output_dir, f"user_{global_index}.jpg")
                cv2.imwrite(file_path, face_img)
                
                rospy.loginfo(f"Saved: user_{global_index}.jpg (Session Progress: {self.current_session_captured}/{self.max_per_session})")
                rospy.sleep(0.1) 
                break # Process one face per camera frame
                
        except Exception as e:
            pass

if __name__ == '__main__':
    FaceCollector()
    rospy.spin()