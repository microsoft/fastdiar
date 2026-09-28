# MIT License
#
# Copyright (c) 2026 Palabra.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction. The Software is provided "AS IS", without
# warranty of any kind. See the original ReDimNet2 repository for the full text.

"""Channels-first LayerNorm for 1-D and 2-D feature maps."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """LayerNorm over dim 1 of a ``(batch, channels, ...)`` tensor.

    The affine parameters broadcast over the remaining (spatial/temporal)
    dimensions, so the same module works for 1-D and 2-D feature maps.
    """

    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        w, b = self.weight, self.bias
        for _ in range(x.ndim - 2):
            w = w.unsqueeze(-1)
            b = b.unsqueeze(-1)
        return w * x + b
