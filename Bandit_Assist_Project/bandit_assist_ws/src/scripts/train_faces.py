#!/usr/bin/env python3
import cv2
import os
import numpy as np

def train_model():
    data_dir = os.path.expanduser('~/PAR_26/src/tiago_hri/src/scripts/face_data')
    
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"❌ Error: No images found in {data_dir}. Please run capture_faces.py first!")
        return

    images = []
    labels = []

    print(f"Reading face dataset from {data_dir}...")
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(valid_extensions):
            path = os.path.join(data_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(1) # Label '1' represents you

    print(f"Loaded {len(images)} images for training.")

    # Initialize OpenCV's LBPH Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    print("Training model across all collected environments...")
    recognizer.train(images, np.array(labels))
    
    model_path = os.path.expanduser('~/PAR_26/src/tiago_hri/src/scripts/me_model.xml')
    recognizer.save(model_path)
    print(f"🎉 Model successfully trained and saved to {model_path}!")

if __name__ == '__main__':
    train_model()