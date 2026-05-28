#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageSaver:
    def __init__(self):
        rospy.init_node('tiago_image_saver', anonymous=True)
        self.bridge = CvBridge()
        # Note: Check your specific camera topic using 'rostopic list'
        # Usually it is /head_front_camera/rgb/image_raw or /xtion/rgb/image_raw
        self.image_sub = rospy.Subscriber("/head_front_camera/rgb/image_raw", Image, self.image_callback)
        self.current_frame = None

    def image_callback(self, data):
        self.current_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def run(self):
        print("Press 's' to save snapshot, 'q' to quit.")
        while not rospy.is_shutdown():
            if self.current_frame is not None:
                cv2.imshow("TIAGo Camera Feed", self.current_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    name = input("Enter name for this image file (e.g., viheet): ")
                    cv2.imwrite(f"{name}.jpg", self.current_frame)
                    print(f"Saved {name}.jpg successfully!")
                elif key == ord('q'):
                    break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    saver = ImageSaver()
    saver.run()