import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Libraries for processing sounds
import librosa
from IPython.display import Audio
import random

def extract_log_mel_spectrogram(
    filepath,
    sr=22050,
    n_fft=1024,
    hop_length=512,
    n_mels=128
):
    try:
        y, _ = librosa.load(filepath, sr=sr)

        # Check if audio is silent or invalid
        if not np.isfinite(y).all() or np.max(np.abs(y)) == 0:
            raise ValueError("Invalid or silent audio")

        # Normalize safely
        y = y / np.max(np.abs(y))

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # Convert to log scale
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Final safety check
        if not np.isfinite(log_mel).all():
            raise ValueError("NaN or inf in spectrogram")

        return log_mel

    except Exception as e:
        print(f"[WARNING] Failed to extract mel from {filepath}: {e}")
        # Return a zero tensor of expected shape
        return np.zeros((n_mels, 400), dtype=np.float32)


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
    def __init__(self, data_path, transform=None, split='train', split_ratio=0.8, seed=42):
        self.transform = transform
        self.data_path = data_path

        df = pd.read_csv(data_path)

        all_paths = df['path']
        all_labels = df['name']
        self.classes = df['name'].unique().tolist()

        # Build label-to-index map
        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}
        self.classes = list(self.class_to_idx.keys())  # e.g., ['cat', 'cow', 'dog']

        # Encode labels
        encoded_labels = [self.class_to_idx[label] for label in all_labels]

        # Shuffle and split
        combined = list(zip(all_paths, encoded_labels))
        random.seed(seed)
        random.shuffle(combined)
        split_point = int(len(combined) * split_ratio)

        if split == 'train':
            selected = combined[:split_point]
        elif split == 'val':
            selected = combined[split_point:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.file_paths, self.labels = zip(*selected) if selected else ([], [])

    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # mel = extract_mel_spectrogram(self.file_paths[idx])  # [n_mels, time]
        # mel = pad_or_trim(mel, target_width=400)
        # mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        # label = self.labels[idx]
        # return mel, label

        mel_log = extract_log_mel_spectrogram(self.file_paths[idx])  # [n_mels, time]
        mel_log = pad_or_trim(mel_log, target_width=400)
        mel_log = torch.tensor(mel_log, dtype=torch.float32).unsqueeze(0)
        label = self.labels[idx]
        return mel_log, label
  
    def get_class(self,idx):
        label = self.labels[idx]
        return self.classes[label]
    
    def visualize(self,n):
    #print(dataset[n][0].squeeze(0))
        plt.figure(figsize=(16,6))
        librosa.display.specshow(
                            self[n][0].squeeze(0).numpy(),
                            x_axis="time",
                            y_axis="mel")
        plt.colorbar()

    def play(self,n):
        path = self.file_paths[n]
        #print(path)
        x, Fs = librosa.load(path, sr=None)
        label = self.get_class(n)
        print('Class: {}'.format(label))
        return Audio(x, rate=Fs)