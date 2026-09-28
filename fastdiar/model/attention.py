# MIT License
#
# Copyright (c) 2026 Palabra.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction. The Software is provided "AS IS", without
# warranty of any kind. See the original ReDimNet2 repository for the full text.

"""Transformer self-attention encoder block (adapted from HF wav2vec2)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input/output shape: ``(batch, time, channel)``."""
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, value_states)
        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, self.embed_dim)
        )
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Position-wise feed-forward network (tanh-approximated GELU)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU(approximate="tanh")
        self.output_dense = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_act_fn(self.intermediate_dense(hidden_states))
        return self.output_dense(hidden_states)


class TransformerEncoderLayer(nn.Module):
    """Post-norm transformer encoder block (self-attention + FFN).

    Operates on ``(batch, channel, time)`` tensors.
    """

    def __init__(self, n_state: int, n_mlp: int, n_head: int, ln_eps: float = 1e-6):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim=n_state, num_heads=n_head)
        self.layer_norm = nn.LayerNorm(n_state, eps=ln_eps)
        self.feed_forward = FeedForward(hidden_size=n_state, intermediate_size=n_mlp)
        self.final_layer_norm = nn.LayerNorm(n_state, eps=ln_eps)

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = hidden_states + self.attention(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states.permute(0, 2, 1)
