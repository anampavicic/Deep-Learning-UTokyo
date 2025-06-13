# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Blocks import Conv2DBlock
from Preprocessing_py import AnimalSoundDataset

# Libraries for processing sounds
import librosa
from IPython.display import Audio
import random

# %%
class AudioModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.layers = nn.ModuleList()

        # Input Shape: (1, 128, target_width)
        layer = Conv2DBlock(in_channels=1, out_channels=80, kernel_size=(57,6),stride=(1,1), dropout_rate=0.5)
        self.layers.append(layer)

        layer = nn.MaxPool2d(kernel_size=(4, 3), stride=(1,3))
        self.layers.append(layer)

        layer = Conv2DBlock(in_channels=80, out_channels=80, kernel_size=(1,3),stride=(1,1))
        self.layers.append(layer)

        layer = nn.MaxPool2d(kernel_size=(1, 3), stride=(1,3))
        self.layers.append(layer)
        
        
        # Flatten
        self.layers.append(nn.Flatten())
        
        # 100 relus two times
        # First FC layer
        self.layers.append(nn.LazyLinear(100))
        
        # ReLU activation
        self.layers.append(nn.ReLU(inplace=True))
        
        # Dropout
        self.layers.append(nn.Dropout(0.5))
        
        # Second FC layer
        self.layers.append(nn.LazyLinear(13))
        
        #Relu Activation
        self.layers.append(nn.ReLU(inplace=True))
        
        # Dropout
        self.layers.append(nn.Dropout(0.5)) 
        
        # Softmax
        self.layers.append(nn.Softmax(dim=1)) 

        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):
        y_hat = self.classifier(x)
        return y_hat

# %%
path_parent_project = os.getcwd() #current walk directory
dataset_image_path = path_parent_project + '\\Animal-Soundprepros\\'

dataset_train = AnimalSoundDataset(dataset_image_path, split='train', split_ratio=0.8, seed=42)
dataset_val = AnimalSoundDataset(dataset_image_path, split='val', split_ratio=0.8, seed=42)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = DataLoader(dataset_train, batch_size=len(dataset_train))
x_train, y_train = next(iter(loader))

loader = DataLoader(dataset_val, batch_size=len(dataset_val))
x_val, y_val = next(iter(loader))

# %%
model = AudioModel().to(device)
y_hat = model(x_val)

# %%
print(y_hat.shape)





# %%
