# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from AnimalSoundDataset import AnimalSoundDataset
from ClassesML.AudioModel import AudioModel
from ClassesML.Blocks import Conv2DBlock
from ClassesML.AudioTrainer import AudioTrainer

# Libraries for processing sounds
import librosa
from IPython.display import Audio
import random
#  Dataset extraction
data_path = 'data/Animal_Sound_reduced.csv'
dataset_train = AnimalSoundDataset(data_path, split='train', split_ratio=0.8, seed=42)
dataset_val = AnimalSoundDataset(data_path, split='val', split_ratio=0.8, seed=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  Creating model loop
input_dim = 1
n_classes = len(dataset_train.classes)

hyperparameters = dict(input_dim=input_dim,
                     output_dim=n_classes,
                     hidden_layers_size=100,
                     activation='relu',
                     kernel_size_conv=[(57,6),(1,3)],
                     kernel_size_pool=[(4,3),(1,3)],
                     stride_conv=[(1,1),(1,1)],
                     stride_pool=[(1,3),(1,3)],
                     filters=[80,80],
                     batch_normalization=True,
                     batch_size = 128,
                     dropout_rate=0.5,
                     learning_rate=0.002,
                     max_epoch=10)

model = AudioModel(hyperparameters=hyperparameters).to(device)
#y_hat = model(x_val)
#print(y_hat)

trainer = AudioTrainer(model, dataset_train, dataset_val, hyperparameters, device=device)
trainer.train()