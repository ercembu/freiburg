"""Primitive modules used in the MacroGraph and Cells."""

import torch
import torch.nn as nn
import numpy as np

from typing import Tuple


class Stem(nn.Module):
    """Basic convolution with three input channels and C output channels,
       followed by batch normalization."""

    def __init__(self, C):
        super(Stem, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(C)
        )

    def forward(self, x, *args, **kwargs):
        return self.seq(x)


class ReLUConvBN(nn.Module):
    """ReLU activation, followed by Convolution, followed by batch normalization."""

    def __init__(self, in_channels, out_channels, kernel_size: Tuple[int, int], stride: Tuple[int, int] = 1):
        super(ReLUConvBN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = 0 if (stride == 1 or stride == (1, 1)) and (kernel_size == 1 or kernel_size == (1, 1)) else 1
        self.stride = stride

        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=stride,
                padding=self.padding,
                bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x):
        return self.seq(x)


class Identity(nn.Module):
    """Identity module. Doesn't perform any operation."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


class Zero(nn.Module):
    """Implementation of the zero operation. It removes the connection by multiplying its input with zero."""

    def __init__(self, **kwargs):
        super(Zero, self).__init__()

    def forward(self, x, edge_data=None):
        return x.mul(0.0)


class ResNetBasicblock(nn.Module):
    """Implementation of Residual ReLUConvBN."""

    def __init__(self, C_in, C_out, affine=True):
        super(ResNetBasicblock, self).__init__()
        self.conv_a = ReLUConvBN(C_in, C_out, kernel_size=(3, 3), stride=(2, 2))
        self.conv_b = ReLUConvBN(C_out, C_out, kernel_size=(3, 3))
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(C_in, C_out, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
        )

    def forward(self, x):
        basicblock = self.conv_a(x)
        basicblock = self.conv_b(basicblock)
        residual = self.downsample(x)
        return residual + basicblock
