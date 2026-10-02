# MIT License
#
# Copyright (c) 2026 Palabra.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction. The Software is provided "AS IS", without
# warranty of any kind. See the original ReDimNet2 repository for the full text.

"""Depthwise 2-D residual block used as ReDimNet's basic convolutional unit."""

import torch.nn as nn


class ResBasicBlock(nn.Module):
    """ResNet-style block with two 3x3 depthwise convs and pointwise mixing.

    Each depthwise conv (``conv1``/``conv2``) is followed by a 1x1 pointwise conv
    (``conv1pw``/``conv2pw``). Channels and resolution are preserved, so the
    residual connection is the identity.
    """

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False, groups=c)
        self.conv1pw = nn.Conv2d(c, c, 1)
        self.bn1 = nn.BatchNorm2d(c)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False, groups=c)
        self.conv2pw = nn.Conv2d(c, c, 1)
        self.bn2 = nn.BatchNorm2d(c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(self.relu(self.conv1pw(self.conv1(x))))
        out = self.bn2(self.conv2pw(self.conv2(out)))
        return self.relu(out + x)
