import streamlit as st
import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import torch.nn as nn
from PIL import Image
import tempfile
import urllib.request
import os

# Page config
st.set_page_config(page_title="Yoga Pose Classifier", page_icon="🧘", layout="wide")

# ==================== MODEL DOWNLOAD SECTION ====================
# Replace these URLs with your actual GitHub Release URLs
MODEL_URL = "https://github.com/tumblr-byte/AI-Yoga-Pose-Classifier/releases/download/v1.0/yoga_pose_classifier.pth"
ENCODER_URL = "https://github.com/tumblr-byte/AI-Yoga-Pose-Classifier/releases/download/v1.0/label_encoder.pkl"

def download_file(url, filename):
    """Download file from URL if it doesn't exist"""
    if not os.path.exists(filename):
        try:
            with st.spinner(f'📥 Downloading {filename}... (first time only)'):
                urllib.request.urlretrieve(url, filename)
                st.success(f'✅ Downloaded {filename}')
        except Exception as e:
            st.error(f'❌ Failed to download {filename}: {e}')
            st.stop()

# Download model files
download_file(MODEL_URL, 'yoga_pose_classifier.pth')
download_file(ENCODER_URL, 'label_encoder.pkl')

# ==================== MODEL ARCHITECTURE ====================
class PoseClassifier(nn.Module):
    """Neural network for yoga pose classification"""
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

# ==================== LOAD MODEL & ENCODER ====================
@st.cache_resource
def load_label_encoder():
    """Load the label encoder"""
    try:
        with open('label_encoder.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading label encoder: {e}")
        st.stop()

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Load label encoder
        le = load_label_encoder()
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        model = PoseClassifier(len(le.classes_)).to(device)
        
        # Load weights - try multiple methods
        try:
            state_dict = torch.load('yoga_pose_classifier.pth', map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
        except:
            try:
                state_dict = torch.load('yoga_pose_classifier.pth', map_location=device)
                model.load_state_dict(state_dict)
            except Exception as load_error:
                st.error(f"❌ Failed to load model weights: {load_error}")
                st.info("💡 Try re-uploading the model file to GitHub Release")
                st.stop()
        
        model.eval()
        return model, device, le
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# ==================== MEDIAPIPE SETUP ====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==================== HELPER FUNCTIONS ====================
def extract_landmarks_from_image(image):
    """Extract pose landmarks from a PIL Image"""
    # Convert PIL to numpy array
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks, dtype='float32'), results.pose_landmarks, image_rgb
    
    return None, None, image_rgb

def extract_landmarks_from_frame(frame, pose):
    """Extract pose landmarks from a video frame"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks, dtype='float32'), results.pose_landmarks
    
    return None, None

def predict_pose(landmarks, model, device, le):
    """Predict yoga pose from landmarks"""
    landmarks_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(landmarks_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = le.inverse_transform([predicted.item()])[0]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score

def visualize_pose_on_image(image_rgb, pose_landmarks, predicted_class, confidence):
    """Draw landmarks and prediction on image"""
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
    
    # Background rectangle
    cv2.rectangle(image_with_pose, (10, 10), (w - 10, 80), (0, 0, 0), -1)
    # Border
    cv2.rectangle(image_with_pose, (10, 10), (w - 10, 80), (255, 255, 255), 3)
    # Text
    cv2.putText(image_with_pose, text, (30, 55), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
    
    return image_with_pose

def process_video(video_path, model, device, le):
    """Process video file frame by frame"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Progress bar
    progress_bar = st.progress(0)
    frame_count = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks
            landmarks_array, pose_landmarks = extract_landmarks_from_frame(frame, pose)
            
            if landmarks_array is not None:
                # Predict pose
                predicted_class, confidence = predict_pose(landmarks_array, model, device, le)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3)
                )
                
                # Add prediction text
                text = f"{predicted_class} - {confidence*100:.1f}%"
                cv2.rectangle(frame, (10, 10), (width - 10, 60), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (width - 10, 60), (255, 255, 255), 2)
                cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
            
            # Update progress
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    progress_bar.empty()
    
    return output_path

# ==================== MAIN APP ====================
def main():
    st.title("🧘 Yoga Pose Classifier")
    st.markdown("Upload an **image** or **video** to classify yoga poses with real-time landmark detection")
    
    # Load model
    model, device, le = load_model()
    
    # Input type selector
    input_type = st.radio("Choose input type:", ["📷 Image", "🎥 Video"], horizontal=True)
    
    # ==================== IMAGE MODE ====================
    if input_type == "📷 Image":
        uploaded_file = st.file_uploader("Upload a yoga pose image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            
            # Create columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(image, use_container_width=True)
            
            # Process image
            with st.spinner('🔍 Detecting pose...'):
                landmarks_array, pose_landmarks, image_rgb = extract_landmarks_from_image(image)
                
                if landmarks_array is None:
                    st.error("❌ No pose detected in the image")
                    st.info("💡 Make sure the full body is visible and clearly lit")
                else:
                    # Predict
                    predicted_class, confidence = predict_pose(landmarks_array, model, device, le)
                    
                    # Visualize
                    result_image = visualize_pose_on_image(image_rgb, pose_landmarks, predicted_class, confidence)
                    
                    with col2:
                        st.subheader("🎯 Detected Pose")
                        st.image(result_image, use_container_width=True)
                    
                    # Display results
                    st.success(f"**Detected Pose:** {predicted_class}")
                    st.info(f"**Confidence:** {confidence*100:.2f}%")
                    st.progress(confidence)
    
    # ==================== VIDEO MODE ====================
    else:
        uploaded_video = st.file_uploader("Upload a yoga pose video", type=['mp4', 'mov', 'avi', 'mkv'])
        
        if uploaded_video is not None:
            # Save uploaded video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📹 Original Video")
                st.video(uploaded_video)
            
            # Process video
            with st.spinner('⚙️ Processing video... This may take a while'):
                output_path = process_video(tfile.name, model, device, le)
                
                with col2:
                    st.subheader("🎯 Processed Video")
                    st.video(output_path)
                
                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=f,
                        file_name="yoga_pose_detected.mp4",
                        mime="video/mp4"
                    )
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("ℹ️ Instructions")
        
        st.markdown("""
        ### 📷 For Images:
        1. Upload a clear yoga pose photo
        2. Ensure full body is visible
        3. Good lighting recommended
        4. View instant results
        
        ### 🎥 For Videos:
        1. Upload a video of yoga poses
        2. Processing happens frame-by-frame
        3. Download the annotated video
        4. May take time for long videos
        """)
        
        st.markdown("---")
        st.subheader("🧘 Supported Poses")
        
        try:
            for pose in le.classes_:
                st.markdown(f"✓ {pose}")
        except:
            st.markdown("Loading poses...")
        
        st.markdown("---")
        st.subheader("📊 Model Info")
        st.markdown(f"""
        - **Accuracy:** 95.4%
        - **Framework:** PyTorch
        - **Pose Detection:** MediaPipe
        - **Classes:** {len(le.classes_)}
        """)
        
        st.markdown("---")
      

if __name__ == "__main__":
    main()
