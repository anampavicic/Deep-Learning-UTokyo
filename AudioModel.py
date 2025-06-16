# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ClassesML.Blocks import Conv2DBlock
from AnimalSoundDataset import AnimalSoundDataset
from Utilities.Utilities import Utilities

# Libraries for processing sounds
import librosa
from IPython.display import Audio
import random

# %%
class AudioModel(nn.Module):

    def __init__(self, hyperparameters):
        nn.Module.__init__(self)

        self.input_dim = hyperparameters['input_dim']
        self.output_dim = hyperparameters['output_dim']
        self.hidden_layers_size = hyperparameters['hidden_layers_size']
        self.activation = hyperparameters['activation']
        self.kernel_size_conv = hyperparameters['kernel_size_conv']
        self.kernel_size_pool = hyperparameters['kernel_size_pool']
        self.stride_conv = hyperparameters['stride_conv']
        self.stride_pool = hyperparameters['stride_pool']
        self.filters = hyperparameters['filters']
        self.batch_normalization = hyperparameters['batch_normalization']
        self.dropout_rate = hyperparameters['dropout_rate']

        self.layers = nn.ModuleList()

        # Input Shape: (1, 128, target_width)
        layer = Conv2DBlock(in_channels=self.input_dim, out_channels=self.filters[0], 
                            kernel_size=self.kernel_size_conv[0],stride=self.stride_conv[0],
                            activation=Utilities.get_activation(self.activation), 
                            batch_normalization=self.batch_normalization, dropout_rate=self.dropout_rate)
        self.layers.append(layer)

        layer = nn.MaxPool2d(kernel_size=self.kernel_size_pool[0], stride=self.stride_pool[0])
        self.layers.append(layer)

        layer = Conv2DBlock(in_channels=self.filters[0], out_channels=self.filters[1],
                            kernel_size=self.kernel_size_conv[1],stride=self.stride_conv[1],
                            activation=Utilities.get_activation(self.activation),
                            batch_normalization=self.batch_normalization, dropout_rate=0.0)
        self.layers.append(layer)

        layer = nn.MaxPool2d(kernel_size=self.kernel_size_pool[1], stride=self.stride_pool[1])
        self.layers.append(layer)
        
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, self.input_dim, 128, 400)  # set correct target_width
        #     for layer in self.layers:
        #         dummy_input = layer(dummy_input)
        #     print(f"Shape after conv + pool layers: {dummy_input.shape}")
        #     self.flattened_size = dummy_input.shape[1] * dummy_input.shape[2] * dummy_input.shape[3]

        # Flatten
        self.layers.append(nn.Flatten())
        
        # Compute flattened size with dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_dim, 128, 400)
            dummy_output = dummy_input
            for layer in self.layers:
                dummy_output = layer(dummy_output)
            flattened_size = dummy_output.shape[1]

        # Now we know the size, define the actual linear layers
        self.layers.append(nn.Linear(flattened_size, self.hidden_layers_size))

        # 100 relus two times
        # First FC layer
        #self.layers.append(nn.Linear(self.flattened_size, self.hidden_layers_size))
        #self.layers.append(nn.LazyLinear(self.hidden_layers_size))
        
        # ReLU activation
        self.layers.append(Utilities.get_activation(self.activation))
        
        # Dropout
        self.layers.append(nn.Dropout(self.dropout_rate))
        
        # Second FC layer
        self.layers.append(nn.LazyLinear(self.output_dim))
        
        #Relu Activation
        self.layers.append(Utilities.get_activation(self.activation))
        
        # Dropout
        self.layers.append(nn.Dropout(self.dropout_rate)) 
        
        # Softmax
        #self.layers.append(nn.Softmax(dim=1)) 

        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):
        y_hat = self.classifier(x)
        return y_hat

