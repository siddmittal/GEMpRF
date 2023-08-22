#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)

Notes:
- Code adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
- Model adapted from: Alex Krizhevsky "One weird trick for parallelizing convolutional neural networks" arxiv: 1404.5997

Adaptations:
- Made all components 1D
- Average pooling changed to global average pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):

    name = "alexnet"
    
    def __init__(self, num_in_channels=1, num_outputs=4):
        """
        args:
            num_in_channels (int): number of input channels (default: 1)
            num_ouputs (int): number of outputs (default: 4)
        """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=num_in_channels, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=192),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
        )
        self.globalaveragepool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_outputs),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.globalaveragepool(out)
        out = torch.flatten(out, 1)
        out = self.regressor(out)
        return out
        