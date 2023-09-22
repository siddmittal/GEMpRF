#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)

Notes:
- Code adapted from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
- Model adapted from: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun "Identity Mappings in Deep Residual Networks" arXiv:1603.05027

Adaptations:
- Made all components 1D
- Average pooling changed to global average pooling
- Added dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv1d(in_channels=in_planes, out_channels=self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += shortcut
        return out


class PreActBottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels=in_planes, out_channels=planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(in_channels=planes, out_channels=self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += shortcut
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_in_channels=1, num_outputs=4):
        """
        args:
            num_blocks (list): number of preact blocks
            num_in_channels (int): number of input channels (default: 1)
            num_ouputs (int): number of outputs (default: 4)
        """
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=num_in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, num_blocks[0], stride=1),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2),
        )
        self.globalaveragepool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*block.expansion, num_outputs),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.features(x)  # resnet convolutional blocks
        h = self.globalaveragepool(h)  # global average pool over time
        h = torch.flatten(h, start_dim=1)  # flatten for regressor
        y = self.regressor(h)  # dense layer
        return y


class PreActResNet18(PreActResNet):

    name = "resnet18"

    def __init__(self, num_in_channels=1, num_outputs=4):
        """
        args:
            num_in_channels (int): number of input channels (default: 1)
            num_ouputs (int): number of outputs (default: 4)
        """
        super(PreActResNet18, self).__init__(PreActBlock, [2,2,2,2], num_in_channels, num_outputs)


class PreActResNet34(PreActResNet):

    name = "resnet34"

    def __init__(self, num_in_channels=1, num_outputs=4):
        """
        args:
            num_in_channels (int): number of input channels (default: 1)
            num_ouputs (int): number of outputs (default: 4)
        """
        super(PreActResNet34, self).__init__(PreActBlock, [3,4,6,3], num_in_channels, num_outputs)


class PreActResNet50(PreActResNet):

    name = "resnet50"

    def __init__(self, num_in_channels=1, num_outputs=4):
        """
        args:
            num_in_channels (int): number of input channels (default: 1)
            num_ouputs (int): number of outputs (default: 4)
        """
        super(PreActResNet50, self).__init__(PreActBottleneck, [3,4,6,3], num_in_channels, num_outputs)


class PreActResNet101(PreActResNet):

    name = "resnet101"

    def __init__(self, num_in_channels=1, num_outputs=4):
        """
        args:
            num_in_channels (int): number of input channels (default: 1)
            num_ouputs (int): number of outputs (default: 4)
        """
        super(PreActResNet101, self).__init__(PreActBottleneck, [3,4,23,3], num_in_channels, num_outputs)


class PreActResNet152(PreActResNet):

    name = "resnet152"

    def __init__(self, num_in_channels=1, num_outputs=4):
        """
        args:
            num_in_channels (int): number of input channels (default: 1)
            num_ouputs (int): number of outputs (default: 4)
        """
        super(PreActResNet152, self).__init__(PreActBottleneck, [3,8,36,3], num_in_channels, num_outputs)
