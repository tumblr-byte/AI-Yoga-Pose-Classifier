import cv2
import mediapipe as mp
import torch
import numpy as np
import matplotlib.pyplot as plt


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseClassifier(len(le.classes_)).to(device)
model.load_state_dict(torch.load('yoga_pose_classifier.pth'))
model.eval()

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks, dtype='float32'), results.pose_landmarks, image_rgb
    
    return None, None, image_rgb

def predict_pose(landmarks):
    landmarks_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(landmarks_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)   
    predicted_class = le.inverse_transform([predicted.item()])[0]
    confidence_score = confidence.item()
    return predicted_class, confidence_score


def visualize_pose(image_path, save_path='result.png'):
    landmarks_array, pose_landmarks, image_rgb = extract_landmarks(image_path)
    if landmarks_array is None:
        return
      
    predicted_class, confidence = predict_pose(landmarks_array)
    image_with_pose = image_rgb.copy()
    mp_drawing.draw_landmarks(
        image_with_pose,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=8),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=6))
  
    h, w, _ = image_with_pose.shape
    text = f"{predicted_class} - {confidence*100:.1f}%"
    cv2.rectangle(image_with_pose, (10, 10), (w - 10, 80), (0, 0, 0), -1)
    cv2.rectangle(image_with_pose, (10, 10), (w - 10, 80), (255, 255, 255), 3)
    cv2.putText(image_with_pose, text, (30, 55), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
  
    plt.figure(figsize=(10, 8))
    plt.imshow(image_with_pose)
    plt.axis('off')
    plt.title(f'Pose: {predicted_class} | Confidence: {confidence*100:.1f}%')
    plt.tight_layout()
    plt.show()

test_image_path = "path of yout test image"
visualize_pose(test_image_path, save_path='result.png')
