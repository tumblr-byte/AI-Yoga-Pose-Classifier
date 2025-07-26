# AI-Yoga-Pose-Classifier

![Image](https://github.com/user-attachments/assets/6e18bde4-383f-4831-a766-97e4c30963b6)

Developed an intelligent yoga pose classification system that automatically detects and identifies specific yoga poses by name. The system uses MediaPipe to extract human body keypoints and then analyzes these pose landmarks to accurately classify different yoga asanas.

https://github.com/user-attachments/assets/a9fb8711-54ff-47e2-9ebd-67e017238fbc

Key Features:
Automated yoga pose detection and classification

Identifies specific poses by name (Warrior 2, Tree Pose, Downward Dog, etc.)

33 body landmark extraction using MediaPipe

Works with images, videos, and real-time camera input

CNN-based classification model trained on yoga pose datasets

**Image/video source: Freepik**


**creating_csv.py** -- This script extracts landmarks from yoga pose images (like Warrior2, Tree, Downward Dog) and saves them in a CSV file.

**In the csv:**
33 pose keypoints detected by MediaPipe


Each keypoint has 3 coordinates (x, y, z)


Total features: 33 keypoints × 3 coordinates = 99 columns



Plus filename and class_name = 101 total columns


**main.py**: This script contains the neural network training code that trains a model on the CSV file we extracted. The model will be trained on the pose landmarks to classify different yoga pose classes.
