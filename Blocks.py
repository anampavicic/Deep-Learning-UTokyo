# %%
import torch
import torch.nn as nn
import numpy as np
import math

# %%
class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size,
                 stride,
                 activation=nn.ReLU(),
                 batch_normalization = False,
                 dropout_rate=0.1):
        super(Conv2DBlock, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding="same",
                                    stride=stride)
        self.activation = activation
        self.batch_norm_layer = nn.BatchNorm2d(out_channels) if batch_normalization else None
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self,x):
        x = self.linear_layer(x)
        if self.batch_norm_layer is not None:
            x = self.batch_norm_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x


