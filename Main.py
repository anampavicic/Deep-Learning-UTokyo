# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Blocks import Conv2DBlock
from AnimalSoundDataset import AnimalSoundDataset
from AudioModel import AudioModel

# Libraries for processing sounds
import librosa
from IPython.display import Audio
import random
# %% Dataset extraction
path_parent_project = os.getcwd() #current walk directory
dataset_image_path = path_parent_project + '\\Animal-Soundprepros\\'

dataset_train = AnimalSoundDataset(dataset_image_path, split='train', split_ratio=0.8, seed=42)
dataset_val = AnimalSoundDataset(dataset_image_path, split='val', split_ratio=0.8, seed=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = DataLoader(dataset_train, batch_size=len(dataset_train))
x_train, y_train = next(iter(loader))

loader = DataLoader(dataset_val, batch_size=len(dataset_val))
x_val, y_val = next(iter(loader))

# %% Creating model loop
input_dim = 1
n_classes = int(max(y_train)+1)
print(n_classes)

hyperparameters = dict(input_dim=input_dim,
                     output_dim=n_classes,
                     hidden_layers_size=100,
                     activation='relu',
                     kernel_size_conv=[(57,6),(1,3)],
                     kernel_size_pool=[(4,3),(1,3)],
                     stride_conv=[(1,1),(1,1)],
                     stride_pool=[(1,3),(1,3)],
                     filters=[80,80],
                     batch_normalization=False,
                     dropout_rate=0.5,
                     learning_rate=0.002,
                     max_epoch=10)

model = AudioModel(hyperparameters=hyperparameters).to(device)
y_hat = model(x_val)

# %%
print(y_hat.shape)