# MIT License
#
# Copyright (c) 2026 Palabra.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction. The Software is provided "AS IS", without
# warranty of any kind. See the original ReDimNet2 repository for the full text.

"""Attentive statistics pooling used by the b6 model to turn frames into an embedding."""

import torch
import torch.nn as nn


class ASTP(nn.Module):
    """Attentive statistics pooling with global context (channel- and context-dependent).

    Pools ``(B, C, T)`` or ``(B, C, F, T)`` frame features into ``(B, 2*in_dim)``.
    """

    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.out_dim = 2 * in_dim
        # Conv1d (stride 1) instead of Linear so inputs need not be transposed.
        self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        if x.ndim == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

        context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
        context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
        x_in = torch.cat((x, context_mean, context_std), dim=1)

        # Tanh (not ReLU) keeps the attention easy to optimize.
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)
