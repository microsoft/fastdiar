# MIT License
#
# Copyright (c) 2026 Palabra.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction. The Software is provided "AS IS", without
# warranty of any kind. See the original ReDimNet2 repository for the full text.

"""ConvNeXt-like multi-kernel depthwise 1-D block."""

import torch
import torch.nn as nn


class ConvNeXtLikeBlock(nn.Module):
    """Depthwise multi-kernel 1-D conv block with a residual connection.

    Several depthwise convolutions (one per entry in ``kernel_sizes``) are run
    in parallel, concatenated, normalized, activated and projected back to
    ``C`` channels by a pointwise conv.
    """

    def __init__(self, C, kernel_sizes):
        super().__init__()
        self.dwconvs = nn.ModuleList(
            nn.Conv1d(C, C, kernel_size=ks, padding="same", groups=C) for ks in kernel_sizes
        )
        self.norm = nn.BatchNorm1d(C * len(kernel_sizes))
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv1d(C * len(kernel_sizes), C, 1)

    def forward(self, x):
        skip = x
        x = torch.cat([dwconv(x) for dwconv in self.dwconvs], dim=1)
        x = self.act(self.norm(x))
        x = self.pwconv1(x)
        return skip + x
