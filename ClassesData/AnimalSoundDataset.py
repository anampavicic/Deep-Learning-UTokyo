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

import warnings

def extract_segments_with_deltas(file_path, variant='short', silence_threshold=-80):
    """
    Extracts 2-channel (log-mel + delta) spectrogram segments from an audio file.

    Parameters:
    - file_path (str): Path to audio file.
    - variant (str): 'short' (41 frames, 50% overlap) or 'long' (101 frames, 90% overlap).
    - silence_threshold (float): dB threshold for discarding low-energy segments.

    Returns:
    - np.ndarray: Array of shape (n_segments, 2, 60, frames_per_segment)
    """
    # Config
    sr = 22050
    n_fft = 1024
    hop_length = 512
    n_mels = 60

    if variant == 'short':
        frames_per_segment = 41
        overlap = 0.5
    elif variant == 'long':
        frames_per_segment = 101
        overlap = 0.9
    else:
        raise ValueError("variant must be 'short' or 'long'")

    try:
        # Load audio in mono
        y, _ = librosa.load(file_path, sr=sr, mono=True)

        # Skip empty or very short files
        if len(y) < n_fft:
            warnings.warn(f"File too short to process: {file_path}")
            return np.empty((0, 2, 60, frames_per_segment))

        # Compute log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize safely
        mean = np.mean(log_mel_spec)
        std = np.std(log_mel_spec)
        if std == 0:
            warnings.warn(f"Zero std encountered in file: {file_path}")
            return np.empty((0, 2, 60, frames_per_segment))

        log_mel_spec = (log_mel_spec - mean) / std

        # Compute deltas
        delta_spec = librosa.feature.delta(log_mel_spec)

        # Segmenting
        step = int(frames_per_segment * (1 - overlap))
        segments = []

        for start in range(0, log_mel_spec.shape[1] - frames_per_segment + 1, step):
            seg = log_mel_spec[:, start:start + frames_per_segment]
            delta = delta_spec[:, start:start + frames_per_segment]

            # Skip silent segments
            if np.mean(seg) < silence_threshold:
                continue

            stacked = np.stack([seg, delta], axis=0)
            segments.append(stacked)

        return np.stack(segments) if segments else np.empty((0, 2, 60, frames_per_segment))

    except Exception as e:
        warnings.warn(f"Failed to process {file_path}: {e}")
        return np.empty((0, 2, 60, frames_per_segment))


class AnimalSoundDataset(Dataset):
    def __init__(self, data_path, split='train', split_ratio=0.8, seed=42, variant='short', silence_threshold=-80):
        self.variant = variant
        self.silence_threshold = silence_threshold
        self.segment_data = []

        print(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)

        all_paths = df['path']
        all_labels = df['name']
        self.classes = sorted(df['name'].unique().tolist())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

        print(f"Classes found: {self.classes}")

        # Step 1: Extract all segments and labels
        all_segments = []
        for file_path, label in zip(all_paths, all_labels):
            label_idx = self.class_to_idx[label]
            segments = extract_segments_with_deltas(file_path, variant=self.variant, silence_threshold=self.silence_threshold)
            for seg in segments:
                all_segments.append((seg, label_idx))

        print(f"Total segments extracted: {len(all_segments)}")

        # Step 2: Shuffle all segments
        random.seed(seed)
        random.shuffle(all_segments)

        # Step 3: Split segments into train/val
        split_point = int(len(all_segments) * split_ratio)
        if split == 'train':
            self.segment_data = all_segments[:split_point]
        elif split == 'val':
            self.segment_data = all_segments[split_point:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        print(f"{split} set contains {len(self.segment_data)} segments.")

    def __len__(self):
        return len(self.segment_data)

    def __getitem__(self, idx):
        segment, label = self.segment_data[idx]
        segment_tensor = torch.tensor(segment, dtype=torch.float32)  # shape: [2, 60, 41]
        return segment_tensor, label

    def get_class(self, idx):
        _, label = self.segment_data[idx]
        return self.classes[label]

    def visualize(self, idx):
        segment, _ = self.segment_data[idx]
        plt.figure(figsize=(16, 6))
        librosa.display.specshow(segment[0], x_axis="time", y_axis="mel")
        plt.colorbar()
        plt.title("Log-Mel Spectrogram (Channel 0)")
