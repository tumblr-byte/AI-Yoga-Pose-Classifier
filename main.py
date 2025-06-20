from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim

# Load dataset
df = pd.read_csv("/content/pose_landmarks.csv")

# Encode class labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['class_name'])

# Split dataset with stratification for balanced splits
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

class PoseLandmarkDataset(Dataset):
    def __init__(self, dataframe):
        self.X = dataframe.iloc[:, 1:-2].values.astype('float32')  # exclude filename and class_name
        self.y = dataframe['label'].values.astype('int64')         # encoded labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

train_dataset = PoseLandmarkDataset(train_df)
valid_dataset = PoseLandmarkDataset(valid_df)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)

class PoseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PoseClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(99, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added dropout for better generalization
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseClassifier(len(le.classes_)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Reduced learning rate

# Training loop with validation tracking
train_losses = []
val_accuracies = []

epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(avg_loss)
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    val_accuracies.append(val_acc)
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")

# Save the model
torch.save(model.state_dict(), 'yoga_pose_classifier.pth')
print("Model saved successfully!")