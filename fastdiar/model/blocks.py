"""ReDimNet building blocks: 2-D conv block and 1-D time-context block."""

import torch.nn as nn

from fastdiar.model.attention import TransformerEncoderLayer
from fastdiar.model.convnext import ConvNeXtLikeBlock
from fastdiar.model.layernorm import LayerNorm
from fastdiar.model.resblocks import ResBasicBlock


class ConvBlock2d(nn.Module):
    """2-D residual conv block (ReDimNet's ``basic_resnet`` unit)."""

    def __init__(self, c):
        super().__init__()
        self.conv_block = ResBasicBlock(c)

    def forward(self, x):
        return self.conv_block(x)


class TimeContextBlock1d(nn.Module):
    """1-D temporal context block with a bottleneck and a residual connection.

    The channel dimension is reduced to ``hC`` (``red_dim_conv``), processed by
    the temporal mixer ``tcm``, then expanded back to ``C`` (``exp_dim_conv``).
    Two mixers are supported:

    * ``'fc'`` - a pointwise MLP (used by the streaming model);
    * ``'conv+att'`` - stacked large-kernel ConvNeXt convs followed by a
      transformer encoder layer (used by the b6 model).
    """

    def __init__(self, C, hC, block_type):
        super().__init__()
        self.red_dim_conv = nn.Sequential(nn.Conv1d(C, hC, 1), LayerNorm(hC, eps=1e-6))

        if block_type == "fc":
            self.tcm = nn.Sequential(
                nn.Conv1d(hC, hC * 2, 1),
                LayerNorm(hC * 2, eps=1e-6),
                nn.GELU(),
                nn.Conv1d(hC * 2, hC, 1),
            )
        elif block_type == "conv+att":
            self.tcm = nn.Sequential(
                *[ConvNeXtLikeBlock(hC, kernel_sizes=[ks]) for ks in (7, 19, 31, 59)],
                TransformerEncoderLayer(n_state=hC, n_mlp=hC, n_head=4),
            )
        else:
            raise NotImplementedError(f"Unsupported block_1d_type: {block_type!r}")

        self.exp_dim_conv = nn.Conv1d(hC, C, 1)

    def forward(self, x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x
