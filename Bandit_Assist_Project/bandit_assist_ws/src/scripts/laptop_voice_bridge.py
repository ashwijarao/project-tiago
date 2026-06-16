#!/usr/bin/env python3
import speech_recognition as sr
import socket
import time

# ---- NETWORK CONFIGURATION ----
ROBOT_IP = "10.234.6.53"  # <--- REPLACE THIS WITH TIAGO'S ACTUAL IP ADDRESS
ROBOT_PORT = 55555

def laptop_voice_bridge():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("==================================================")
    print(" Lightweight Laptop Voice Bridge Operational (No-ROS Mode)")
    print(f" Target Robot IP: {ROBOT_IP}:{ROBOT_PORT}")
    print("==================================================")

    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening for command (collection point / follow me / stop)...")

            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=4)
                text = recognizer.recognize_google(audio).lower()
                print(f"Heard locally: '{text}'")

                # UPDATED LIST: Added "collection" so new transit commands pass through
                if any(k in text for k in ["follow me", "move", "tiago", "stop", "halt", "collection"]):
                    print(">> Valid token heard! Sending over network socket...")
                    
                    # Open a quick network connection to the robot, inject the text, and close it
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2.0)
                    s.connect((ROBOT_IP, ROBOT_PORT))
                    s.sendall(text.encode('utf-8'))
                    s.close()

            except sr.UnknownValueError:
                continue
            except Exception as e:
                print(f"Network/Audio Error: {e}")
                time.sleep(1)

if __name__ == '__main__':
    try:
        laptop_voice_bridge()
    except KeyboardInterrupt:
        pass