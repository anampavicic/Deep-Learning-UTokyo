import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def extract_mel_spectrogram(file_path, sr=22050, n_mels=128):
    y, sr = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def pad_or_trim(mel, target_width=400):
    if mel.shape[1] > target_width:
        mel = mel[:, :target_width]
    elif mel.shape[1] < target_width:
        pad_width = target_width - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
    return mel

class AnimalSoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.file_paths = []
        self.labels = []
        
        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith(".wav"):
                    self.file_paths.append(os.path.join(class_path, file_name))
                    self.labels.append(idx)

        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel = extract_mel_spectrogram(self.file_paths[idx])  # [n_mels, time]
        mel = pad_or_trim(mel, target_width=400)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        label = self.labels[idx]
        return mel, label

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Use dummy input to compute the flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 400)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# Parameters
batch_size = 8
epochs = 20
lr = 0.001

# Dataset & DataLoader
path_parent_project = os.getcwd() #current walk directory
dataset_image_path = path_parent_project + '\\Dataset\\' + 'Animal-Soundprepros\\'

dataset = AnimalSoundDataset(dataset_image_path)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Val Accuracy: {acc:.2f}%")

file_path = path_parent_project + '\\Dataset\\' + 'Animal-Soundprepros\\' + 'Cat\\' + 'Cat_1.wav'
mel = extract_mel_spectrogram(file_path) 

# Plot to visually inspect
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel, sr=22050, x_axis='time', y_axis='mel')
plt.title("Mel Spectrogram")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# Prepare for model input
mel = extract_mel_spectrogram(file_path)
mel = pad_or_trim(mel, target_width=400) 
mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(mel_tensor)
    pred_idx = torch.argmax(output).item()
    print("Predicted label:", dataset.classes[pred_idx])