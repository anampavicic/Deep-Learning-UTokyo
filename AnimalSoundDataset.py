# %% [markdown]
# Libraries

# %%
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

# %% [markdown]
# change the .csv so the path is correct and add row for the class

# %%
# def change_path(path, new_path):
#     train_csv = pd.read_csv(path)
#     train_csv['path']=new_path
#     train_csv['class'] = train_csv['name'].apply(lambda x: x.split('_')[0])
#     train_csv.to_csv('data.csv')
    
# path = 'C:/Users/Lorena/Documents/Uni/25 SoSe/Deep Learning/DL-mine/Animal_Sound.csv'
# new_path = 'C:/Users/Lorena/Documents/Uni/25 SoSe/Deep Learning/DL-mine/Animal-Soundprepros'
# change_path(path, new_path=new_path)

# data_path = 'C:/Users/Lorena/Documents/Uni/25 SoSe/Deep Learning/DL-mine/data.csv'
# train_csv = pd.read_csv(data_path)

# %%
# print(train_csv.head())

# %% [markdown]
# Bring into form so it can be used in ML

# %% [markdown]
# PyTorch Website:
#     https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html

# %%
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

# %% [markdown]
# Define new Datset, so we get a training and validation Dataset from .wav files

# %%
class AnimalSoundDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', split_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.transform = transform

        all_paths = []
        all_labels = []

        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith(".wav"):
                    all_paths.append(os.path.join(class_path, file_name))
                    all_labels.append(idx)

        # Shuffle and split
        combined = list(zip(all_paths, all_labels))
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
        mel = extract_mel_spectrogram(self.file_paths[idx])  # [n_mels, time]
        mel = pad_or_trim(mel, target_width=400)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        label = self.labels[idx]
        return mel, label
    
        
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



