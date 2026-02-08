# AI-Yoga-Pose-Classifier



A real-time yoga pose classification system using MediaPipe pose estimation and PyTorch neural networks. This project achieves **95.4% validation accuracy** in classifying yoga poses from images.

![Image](https://github.com/user-attachments/assets/1d3ff6ac-b69b-4fb2-ab8c-5772db219c3a)

## Key Features

- **High Accuracy**: 95.4% validation accuracy on yoga pose classification
- **Real-time Detection**: Fast pose estimation using MediaPipe
- **Robust Pipeline**: End-to-end ML pipeline from data preprocessing to inference
- **Visual Feedback**: Annotated pose visualization with confidence scores
- **Production-Ready**: Clean, modular code with proper train/test separation

## Architecture

The system consists of three main components:

1. **Feature Extraction**: MediaPipe extracts 33 3D landmarks (99 features total)
2. **Neural Network**: Fully connected network with dropout regularization
3. **Classification**: Multi-class softmax classifier with confidence scoring

### Model Architecture
```
Input (99 features) 
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

- **Training Accuracy**: 97.2%
- **Validation Accuracy**: 95.4%
- **Training Loss**: 0.0875 (final epoch)
- **Model Size**: Lightweight (~50KB)


### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yoga-pose-classifier.git
cd yoga-pose-classifier

# Install dependencies
pip install -r requirements.txt
```

### Usage

**1. Prepare Dataset**
```python
# Place your dataset in the following structure:
# dataset/
#   ├── Downward_Dog/
#   ├── Tree_Pose/
#   ├── Warrior_Pose/
#   └── ...

# Run dataset preprocessing
python creating_csv.py
```

**2. Train the Model**
```python
python train.py
```

**3. Test on New Images**
```python
python test.py
```

The output will show the detected pose with confidence score and visualized landmarks.


## Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Pose Estimation | MediaPipe |
| Computer Vision | OpenCV |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| ML Utils | scikit-learn |


