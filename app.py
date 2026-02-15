# app.py
import streamlit as st
import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import torch.nn as nn
from PIL import Image
import io

# Page config
st.set_page_config(page_title="Yoga Pose Classifier", page_icon="🧘", layout="wide")

# Load LabelEncoder
@st.cache_resource
def load_label_encoder():
    with open('label_encoder.pkl', 'rb') as f:
        return pickle.load(f)

# Define model class
class PoseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PoseClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(99, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# Load model
@st.cache_resource
def load_model():
    le = load_label_encoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseClassifier(len(le.classes_)).to(device)
    model.load_state_dict(torch.load('yoga_pose_classifier.pth', map_location=device))
    model.eval()
    return model, device, le

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Extract landmarks from uploaded image
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks, dtype='float32'), results.pose_landmarks, image_rgb
    return None, None, image_rgb

# Predict pose
def predict_pose(landmarks, model, device, le):
    landmarks_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(landmarks_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    predicted_class = le.inverse_transform([predicted.item()])[0]
    confidence_score = confidence.item()
    return predicted_class, confidence_score

# Visualize pose
def visualize_pose(image_rgb, pose_landmarks, predicted_class, confidence):
    image_with_pose = image_rgb.copy()
    
    # Draw landmarks
    mp_drawing.draw_landmarks(
        image_with_pose,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=8),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=6)
    )
    
    # Add text overlay
    h, w, _ = image_with_pose.shape
    text = f"{predicted_class} - {confidence*100:.1f}%"
    cv2.rectangle(image_with_pose, (10, 10), (w - 10, 80), (0, 0, 0), -1)
    cv2.rectangle(image_with_pose, (10, 10), (w - 10, 80), (255, 255, 255), 3)
    cv2.putText(image_with_pose, text, (30, 55), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
    
    return image_with_pose

# Main app
def main():
    st.title("🧘 Yoga Pose Classifier")
    st.markdown("Upload an image to classify the yoga pose with real-time landmark detection")
    
    # Load model
    try:
        model, device, le = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Process image
        with st.spinner('Detecting pose...'):
            landmarks_array, pose_landmarks, image_rgb = extract_landmarks(image)
            
            if landmarks_array is None:
                st.error(" No pose detected in the image. Please upload a clear image with a visible person.")
            else:
                # Predict
                predicted_class, confidence = predict_pose(landmarks_array, model, device, le)
                
                # Visualize
                result_image = visualize_pose(image_rgb, pose_landmarks, predicted_class, confidence)
                
                with col2:
                    st.subheader("Detected Pose")
                    st.image(result_image, use_container_width=True)
                
                # Display results
                st.success(f"**Detected Pose:** {predicted_class}")
                st.info(f"**Confidence:** {confidence*100:.2f}%")
                
                # Confidence meter
                st.progress(confidence)
    
    # Instructions
    with st.sidebar:
        st.header("ℹ️ Instructions")
        st.markdown("""
        1. Upload a clear image of a yoga pose
        2. Ensure the full body is visible
        3. Wait for the model to detect and classify
        4. View results with landmark visualization
        
        **Supported Poses:**
        """)
        try:
            le = load_label_encoder()
            for pose in le.classes_:
                st.markdown(f"- {pose}")
        except:
            st.markdown("Loading...")

if __name__ == "__main__":
    main()
