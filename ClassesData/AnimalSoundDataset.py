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

from Utilities.Utilities import Utilities

class AnimalSoundDataset(Dataset):
    def __init__(self, dataframe, split='train', split_ratio=0.8, seed=42, variant='short', silence_threshold=-80):
        self.variant = variant
        self.silence_threshold = silence_threshold
        self.segment_data = []

        self.df = dataframe

        all_paths = self.df['path']
        all_labels = self.df['name']
        self.classes = sorted(self.df['name'].unique().tolist())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

        print(f"Classes found: {self.classes}")

        # Step 1: Extract all segments and labels
        all_segments = []
        for file_path, label in zip(all_paths, all_labels):
            label_idx = self.class_to_idx[label]
            segments = Utilities.extract_segments_with_deltas(file_path, variant=self.variant, silence_threshold=self.silence_threshold)
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
