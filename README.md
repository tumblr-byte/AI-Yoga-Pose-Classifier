# AI Yoga Pose Classifier

A real-time yoga pose classification system using MediaPipe pose estimation and PyTorch neural networks. This project achieves **95.4% validation accuracy** in classifying yoga poses from images.

![Image](https://github.com/user-attachments/assets/1d3ff6ac-b69b-4fb2-ab8c-5772db219c3a)


## Project Overview

This project helps yoga practitioners get instant feedback on their poses without needing an instructor present. Built as a complete end-to-end ML pipeline, it demonstrates skills in computer vision, deep learning, and deployment.


### Why This Project?

As someone passionate about making yoga accessible, I built this to help practitioners verify their form in real-time. This project taught me:
- End-to-end ML engineering (data preprocessing → deployment)
- Production-ready computer vision pipelines
- Real-time pose estimation and classification
- Web application deployment with Streamlit

## Key Features

* **High Accuracy**: 95.4% validation accuracy on yoga pose classification
* **Real-time Detection**: Fast pose estimation using MediaPipe
* **Image & Video Support**: Works with both static images and video files
* **Visual Feedback**: Annotated pose visualization with confidence scores
* **Interactive Web App**: User-friendly Streamlit interface
* **Production-Ready**: Clean, modular code with proper train/test separation

## Architecture

The system consists of three main components:

1. **Feature Extraction**: MediaPipe extracts 33 3D landmarks (99 features total)
2. **Neural Network**: Fully connected network with dropout regularization
3. **Classification**: Multi-class softmax classifier with confidence scoring

### Model Architecture
```
Input (99 features: 33 landmarks × 3 coordinates)
    ↓
Linear(99 → 128) + ReLU + Dropout(0.3)
    ↓
Linear(128 → 64) + ReLU + Dropout(0.2)
    ↓
Linear(64 → num_classes)
    ↓
Softmax Output
```

## Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 97.2% |
| Validation Accuracy | 95.4% |
| Training Loss (final) | 0.0875 |
| Model Size | ~50KB |
| Inference Speed | Real-time |

## Live Demo

**[Try the live app here!](#)** 

Upload an image or video to see real-time pose detection and classification!

##  Installation

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/tumblr-byte/AI-Yoga-Pose-Classifier.git
cd AI-Yoga-Pose-Classifier

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the [Yoga Pose Classification Dataset](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification) from Kaggle.

**Dataset Details:**
- **5 yoga poses**: Downdog, Goddess, Plank, Tree, Warrior2
- High-quality images for training and validation
- Preprocessed using MediaPipe pose estimation

**Dataset Structure:**
```
dataset/
├── Downdog/
├── Goddess/
├── Plank/
├── Tree/
└── Warrior2/
```

**To prepare the dataset:**
1. Download from [Kaggle](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification)
2. Extract to `dataset/` folder
3. Run preprocessing script (see Usage below)

##  Usage

### 1. Prepare Dataset
```bash
# Update the dataset path in creating_csv.py
# Then run:
python creating_csv.py
```

This extracts pose landmarks and creates `pose_landmarks.csv`.

### 2. Train the Model
```bash
python train.py
```

This will:
- Train the neural network for 50 epochs
- Display training progress with loss and accuracy
- Save `yoga_pose_classifier.pth` (model weights)
- Save `label_encoder.pkl` (class encoder)

### 3. Test on Images
```bash
# Update test_image_path in test.py
python test.py
```

### 4. Run Web App
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**Web App Features:**
- Upload images or videos
- Real-time pose detection with landmarks
- Confidence scores for predictions
- Side-by-side comparison
- Download processed videos

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Pose Estimation | MediaPipe |
| Computer Vision | OpenCV |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Web Framework | Streamlit |
| ML Utils | scikit-learn |

## Project Structure
```
AI-Yoga-Pose-Classifier/
├── app.py                      # Streamlit web application
├── train.py                    # Model training script
├── test.py                     # Testing script for images
├── creating_csv.py             # Dataset preprocessing
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies
├── yoga_pose_classifier.pth    # Trained model weights
├── label_encoder.pkl           # Class label encoder
├── README.md                   # Project documentation
└── dataset/                    # Training data (not included)
    ├── Downdog/
    ├── Goddess/
    ├── Plank/
    ├── Tree/
    └── Warrior2/
```

## Challenges & Solutions

### Challenge 1: Low Initial Accuracy (78%)
**Solution**: 
- Added dropout layers (0.3 and 0.2) to prevent overfitting
- Implemented stratified train-test split for balanced data
- Result: **95.4% validation accuracy**

### Challenge 2: Real-time Performance
**Solution**:
- Used MediaPipe's optimized pose estimation
- Lightweight neural network architecture (~50KB)
- Achieved real-time inference on CPU

### Challenge 3: Video Processing
**Solution**:
- Frame-by-frame pose detection
- Efficient video encoding with OpenCV
- Added download feature for processed videos

##  Future Improvements

- [ ] Add more yoga poses (expand to 20+ poses)
- [ ] Real-time webcam support
- [ ] Mobile app deployment (Flutter/React Native)
- [ ] Pose correction suggestions
- [ ] Multi-person pose detection
- [ ] 3D pose visualization
- [ ] Export workout analytics


---

⭐ **Star this repo if you find it helpful!**


