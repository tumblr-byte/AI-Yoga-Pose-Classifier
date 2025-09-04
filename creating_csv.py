
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import csv
import mediapipe as mp  

dataset_path = path_list
csv_filename = 'pose_landmarks.csv'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Columns: filename + 33 landmarks (x, y, z) = 99 + class
header = ['filename']
for i in range(33):
    header += [f'x{i}', f'y{i}', f'z{i}']
header.append('class_name')

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)

        # Skip if not a directory
        if not os.path.isdir(class_folder):
            continue

        for img_file in os.listdir(class_folder):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_folder, img_file)
            image = cv2.imread(img_path)

            # Check if image loaded successfully
            if image is None:
                print(f"Could not load image: {img_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                row = [img_file]
                for landmark in results.pose_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])
                row.append(class_name)
                writer.writerow(row)
            else:
                print(f"No pose detected in: {img_path}")

print(f"CSV file '{csv_filename}' created successfully!")

