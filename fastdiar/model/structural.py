# MIT License
#
# Copyright (c) 2026 Palabra.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction. The Software is provided "AS IS", without
# warranty of any kind. See the original ReDimNet2 repository for the full text.

"""Structural helpers that move ReDimNet feature maps between 2-D and 1-D.

ReDimNet keeps a running list of 1-D feature maps of shape ``(B, C*F, T)`` and
reshapes them to 2-D ``(B, C, F, T)`` whenever a 2-D convolution stage runs.
``weigth1d`` is the learnable softmax aggregation that fuses all previously
produced 1-D feature maps into one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class to1d(nn.Module):
    """Reshape ``(B, C, F, T)`` -> ``(B, C*F, T)``."""

    def forward(self, x):
        bs, c, f, t = x.size()
        return x.permute((0, 2, 1, 3)).reshape((bs, c * f, t))


class to2d(nn.Module):
    """Reshape ``(B, C*F, T)`` -> ``(B, C, F, T)`` with fixed ``f``/``c``."""

    def __init__(self, f, c):
        super().__init__()
        self.f = f
        self.c = c

    def forward(self, x):
        bs, cf, t = x.size()
        return x.reshape((bs, self.f, self.c, t)).permute((0, 2, 1, 3))


class weigth1d(nn.Module):
    """Softmax-weighted sum of ``N`` feature maps of shape ``(B, C, T)``.

    The ``(1, N, C, 1)`` weight tensor is softmaxed over the ``N`` axis so the
    aggregation is a convex combination.
    """

    def __init__(self, N, C):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1, N, C, 1))

    def forward(self, xs):
        w = F.softmax(self.w, dim=1)
        xs = torch.cat([t.unsqueeze(1) for t in xs], dim=1)
        return (w * xs).sum(dim=1)
