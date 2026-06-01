#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import speech_recognition as sr

def voice_bridge():
    rospy.init_node('laptop_voice_bridge', anonymous=True)
    # This publishes straight onto the shared ROS network
    pub = rospy.Publisher('/tiago_hri/voice_command', String, queue_size=10)
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    rospy.loginfo("================================================")
    rospy.loginfo("Laptop Voice Bridge Active. Using Laptop Microphone.")
    rospy.loginfo("================================================")
    
    with microphone as source:
        rospy.loginfo("Calibrating for room background noise... Stand still.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        rospy.loginfo("Calibration done! Continuous listening active.")
        
        while not rospy.is_shutdown():
            try:
                rospy.loginfo("Listening for command...")
                audio = recognizer.listen(source, phrase_time_limit=5)
                
                # Processing via Google's Web API on your laptop
                text = recognizer.recognize_google(audio).lower()
                rospy.loginfo(f"Heard locally: '{text}'")
                
                # Check for trigger phrases
                if "follow me" in text or "hey tiago" in text or "move" in text:
                    pub.publish(text)
                    rospy.loginfo(">>> Trigger match! Voice command sent to TIAGo.")
                    
            except sr.UnknownValueError:
                # Just ambient room noise/coughing, ignore and keep looping
                continue
            except sr.RequestError as e:
                rospy.logerr(f"Laptop internet or STT Service Error: {e}")
            except Exception as e:
                rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    try:
        voice_bridge()
    except rospy.ROSInterruptException:
        pass