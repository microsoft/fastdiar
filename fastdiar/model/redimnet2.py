# MIT License
#
# Copyright (c) 2026 Palabra.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction. The Software is provided "AS IS", without
# warranty of any kind. See the original ReDimNet2 repository for the full text.

"""ReDimNet2 speaker-embedding backbone and wrapper.

Implements the architecture from *ReDimNet2: Scaling Speaker Verification via
Time-Pooled Dimension Reshaping* (Yakovlev & Okhotnikov, 2026,
https://arxiv.org/abs/2603.11841).

The backbone keeps a shared stream of 1-D feature maps of shape ``(B, C*F, T*)``
and grows it one ``ReDimNet2Stage`` at a time. Every stage (paper Fig. 2):

1. fuses all previous 1-D maps with a learnable softmax (``stack and weight``);
2. reshapes the fused map to 2-D ``(B, C, F, T)`` (``to2d``);
3. runs a strided 2-D conv that **halves frequency while growing channels**
   (ReDimNet's volume-preserving reshape) and **pools the time axis**
   (ReDimNet2's key addition), followed by 2-D residual blocks;
4. reshapes back to 1-D ``(B, C*F, T/time_stride)`` (``to1d``) and applies a
   1-D temporal-context block;
5. nearest-neighbour upsamples back to the input length ``T*`` so the new map
   can re-enter the shared residual stream.

``ReDimNet2Wrap`` adds the log-mel front-end, a statistics-pooling head and the
embedding projection, and is the whole-utterance (non-streaming) model;
:func:`load_hub_model` loads a released one with the config it stores. The
streaming models reuse :class:`ReDimNet2` in :mod:`fastdiar.encoder`.
"""

import math

import torch
import torch.nn as nn

from fastdiar.model.blocks import ConvBlock2d, TimeContextBlock1d
from fastdiar.model.features import TFMelBanks
from fastdiar.model.layernorm import LayerNorm
from fastdiar.model.poolings import ASTP
from fastdiar.model.structural import to1d, to2d, weigth1d


class ReDimNet2Stage(nn.Module):
    """A single ReDimNet2 stage (paper Fig. 2, bottom path).

    A stage takes the list of every 1-D feature map produced so far, fuses
    them, processes the result through a 2-D conv block at a reduced
    time/frequency resolution, and returns a new 1-D map realigned to the input
    time resolution ``T*`` so it can re-enter the shared residual stream.

    The submodules below are applied in order and map one-to-one onto the
    boxes of Figure 2::

        aggregate -> to2d -> downsample -> conv_blocks -> project
                  -> to1d -> context -> upsample -> norm

    * ``aggregate`` - learnable softmax fusion (``stack and weight``) of all
      previous 1-D maps ``(B, C*F, T*)``.
    * ``to2d`` - reshape ``(B, C*F, T*)`` -> ``(B, c, f, T*)``.
    * ``downsample`` - the volume-preserving strided conv (``kernel == stride``,
      non-overlapping). The frequency stride halves ``f`` while channels grow;
      the time stride performs ReDimNet2's **time pooling**. Time is strided by
      the *cumulative* ``time_stride`` because the aggregated input always lives
      at the full resolution ``T*``.
    * ``conv_blocks`` - ``num_blocks`` 2-D residual blocks (basic ResNet).
    * ``project`` - 1x1 conv mapping the (optionally expanded) channels back to
      ``out_channels``; ``Identity`` when ``conv_exp == 1``.
    * ``to1d`` - reshape back to ``(B, C*F, T/time_stride)``.
    * ``context`` - 1-D temporal-context block (``block1d``).
    * ``upsample`` - nearest-neighbour upsampling that restores ``T*``.
    * ``norm`` - optional GroupNorm over the aggregated channels.
    """

    def __init__(
        self,
        *,
        num_prev_maps,
        cf,
        num_groups,
        in_channels,
        in_freq,
        mid_channels,
        out_channels,
        freq_stride,
        time_stride,
        num_blocks,
        conv_exp,
        att_block_red,
        block_1d_type,
        agg_gnorm,
        conv_block,
    ):
        super().__init__()
        # "Stack and weight": fuse all previous 1-D maps into one.
        self.aggregate = weigth1d(N=num_prev_maps, C=cf)
        self.to2d = to2d(f=in_freq, c=in_channels)
        # Single strided conv that does both frequency downsampling (channel
        # growth) and ReDimNet2's time pooling.
        self.downsample = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=(freq_stride, time_stride),
            stride=(freq_stride, time_stride),
            padding=0,
            groups=math.gcd(in_channels, mid_channels),
        )
        self.conv_blocks = nn.Sequential(*[conv_block(mid_channels) for _ in range(num_blocks)])
        # Project expanded channels back to `out_channels` (only when conv_exp
        # changed the channel count inside the stage).
        if conv_exp != 1:
            self.project = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding="same"),
                nn.BatchNorm2d(out_channels, eps=1e-6),
            )
        else:
            self.project = nn.Identity()
        self.to1d = to1d()
        self.context = TimeContextBlock1d(cf, hC=cf // att_block_red, block_type=block_1d_type)
        # Restore the input time resolution T* before re-entering the stream.
        self.upsample = nn.Upsample(scale_factor=time_stride, mode="nearest")
        self.norm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=cf) if agg_gnorm else nn.Identity()
        )

    def forward(self, outputs_1d):
        x = self.aggregate(outputs_1d)  # stack & weight -> (B, C*F, T*)
        x = self.to2d(x)  #               -> (B, c, f, T*)
        x = self.downsample(x)  # freq downsample + time pooling
        x = self.conv_blocks(x)  # 2-D residual blocks
        x = self.project(x)  # channels -> out_channels (optional)
        x = self.to1d(x)  #               -> (B, C*F, T/stride)
        x = self.context(x)  # 1-D temporal context
        x = self.upsample(x)  #               -> (B, C*F, T*)
        x = self.norm(x)  # optional GroupNorm
        return x


class ReDimNet2(nn.Module):
    """ReDimNet2 backbone: a stem plus a stack of :class:`ReDimNet2Stage`.

    The network maintains a shared stream of 1-D feature maps, all of shape
    ``(B, C*F, T*)``. The ``stem`` produces the first map; each stage appends a
    new one (computed from *all* previous maps); a final ``stack and weight``
    fuses every map into the backbone output, which is reshaped to 2-D and
    optionally projected to ``out_channels`` by a 1x1 conv ``head``.

    Constant-volume invariant: ReDimNet keeps ``channels * freq`` equal to
    ``C * F`` across stages by pairing every frequency-halving with a channel
    doubling, so any 2-D map ``(c, f, T)`` reshapes to the common 1-D shape
    ``(C*F, T)``. ReDimNet2 additionally pools the *time* axis inside stages and
    upsamples back before aggregation, leaving the 1-D shape unchanged.

    ``stages_setup`` is a list of ``(stride, num_blocks, conv_exp, kernel_sizes,
    att_block_red)`` tuples, one per stage:

    * ``stride = (sf, st)`` - per-stage frequency / time stride;
    * ``num_blocks`` - number of 2-D residual blocks;
    * ``conv_exp`` - channel-expansion factor inside the stage (may be
      fractional, e.g. ``0.5``);
    * ``kernel_sizes`` - kept for config compatibility (the blocks are 3x3);
    * ``att_block_red`` - bottleneck reduction of the stage's 1-D time-context
      block.

    ``conv_block`` is the class of the stages' 2-D residual blocks; the
    streaming model passes a time-causal one (see :mod:`fastdiar.encoder`).
    """

    def __init__(
        self,
        F,
        C,
        block_1d_type,
        stages_setup,
        out_channels=None,
        agg_gnorm=False,
        conv_block=ConvBlock2d,
    ):
        super().__init__()
        self.agg_gnorm = agg_gnorm

        cf = C * F  # constant 1-D feature volume (C * F)
        c, f = C, F  # running 2-D channel / frequency size
        cumulative_time_stride = 1  # product of per-stage time strides

        # Stem: 2-D conv -> LayerNorm -> reshape to the first 1-D map.
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=1, padding="same"),
            LayerNorm(c, eps=1e-6),
            to1d(),
        )
        if agg_gnorm:
            self.stem_gnorm = nn.GroupNorm(num_groups=C, num_channels=cf)

        # Build the stages, tracking the running channel / frequency size and
        # how many 1-D maps each stage will see (stem output counts as one).
        self.stages = nn.ModuleList()
        self._legacy_key_map = {}
        num_maps = 1
        for stage_ind, (stride, num_blocks, conv_exp, _kernel_sizes, att_block_red) in enumerate(
            stages_setup
        ):
            sf, st = stride
            cumulative_time_stride *= st
            assert f % sf == 0
            mid_channels = int(sf * c * conv_exp)  # expanded stage channels

            self.stages.append(
                ReDimNet2Stage(
                    num_prev_maps=num_maps,
                    cf=cf,
                    num_groups=C,
                    in_channels=c,
                    in_freq=f,
                    mid_channels=mid_channels,
                    out_channels=sf * c,
                    freq_stride=sf,
                    time_stride=cumulative_time_stride,
                    num_blocks=num_blocks,
                    conv_exp=conv_exp,
                    att_block_red=att_block_red,
                    block_1d_type=block_1d_type,
                    agg_gnorm=agg_gnorm,
                    conv_block=conv_block,
                )
            )
            self._legacy_key_map.update(self._stage_legacy_keys(stage_ind, num_blocks, conv_exp))

            c, f = sf * c, f // sf
            num_maps += 1

        # Final "stack and weight" over every 1-D map (stem + all stages).
        self.fin_wght1d = weigth1d(N=num_maps, C=cf)
        self.time_stride = cumulative_time_stride

        # Output head: back to a 2-D map, then an optional 1x1 conv.
        self.fin_to2d = to2d(f=f, c=c)
        self.head = nn.Conv2d(c, out_channels, 1) if out_channels is not None else nn.Identity()
        # Channels of the output map once flattened to 1-D.
        self.out_dim = f * (out_channels if out_channels is not None else c)

        # Allow pre-refactor checkpoints (flat ``stage{i}.{pos}.*`` Sequentials)
        # to load into the named ``stages.{i}.*`` submodules.
        self._register_load_state_dict_pre_hook(self._remap_legacy_state_dict)

    def _stage_legacy_keys(self, stage_ind, num_blocks, conv_exp):
        """Map a stage's legacy ``nn.Sequential`` indices to the new submodule
        names, so checkpoints saved before this refactor still load.

        The legacy stage was a flat ``nn.Sequential`` whose positions were:
        ``0`` aggregate, ``1`` to2d, ``2`` downsample, then ``num_blocks`` conv
        blocks, an optional projection, to1d, the context block, upsample, and
        an optional GroupNorm. Reshape-only layers carry no parameters but still
        consume an index.
        """
        old, new = f"stage{stage_ind}", f"stages.{stage_ind}"
        mapping, pos = {}, 0
        mapping[f"{old}.{pos}"] = f"{new}.aggregate"
        pos += 1  # weigth1d
        pos += 1  # to2d (no params)
        mapping[f"{old}.{pos}"] = f"{new}.downsample"
        pos += 1  # strided conv
        for j in range(num_blocks):
            mapping[f"{old}.{pos}"] = f"{new}.conv_blocks.{j}"
            pos += 1
        if conv_exp != 1:
            mapping[f"{old}.{pos}"] = f"{new}.project"
            pos += 1
        pos += 1  # to1d (no params)
        mapping[f"{old}.{pos}"] = f"{new}.context"
        pos += 1
        pos += 1  # upsample (no params)
        if self.agg_gnorm:
            mapping[f"{old}.{pos}"] = f"{new}.norm"
        return mapping

    def _remap_legacy_state_dict(self, state_dict, prefix, *args):
        """``load_state_dict`` pre-hook: rename legacy ``stage{i}.{pos}.*`` keys
        to the new ``stages.{i}.{name}.*`` layout in place."""
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            rel = key[len(prefix) :]
            for old, new in self._legacy_key_map.items():
                if rel == old or rel.startswith(old + "."):
                    state_dict[prefix + new + rel[len(old) :]] = state_dict.pop(key)
                    break

    def forward(self, inp):
        T = inp.size(-1)
        # Trim time so it divides evenly by time_stride (for clean reshapes).
        inp = inp[:, :, :, : (T // self.time_stride) * self.time_stride]
        x = self.stem(inp)
        if self.agg_gnorm:
            x = self.stem_gnorm(x)

        # Shared 1-D residual stream: each stage reads all maps, appends one.
        outputs_1d = [x]
        for stage in self.stages:
            outputs_1d.append(stage(outputs_1d))

        x = self.fin_wght1d(outputs_1d)
        x = self.fin_to2d(x)
        return self.head(x)


class ReDimNet2Wrap(nn.Module):
    """Full speaker model: log-mel front-end + backbone + pooling head.

    Wires together the four stages of a ReDimNet2 speaker extractor:

    1. ``spec`` - log-mel front-end ``(B, samples)`` -> ``(B, 1, F, T)``;
    2. ``backbone`` - :class:`ReDimNet2` -> a 2-D feature map (optionally
       projected by the head), flattened to ``(B, C*F, T)``;
    3. ``pool`` + ``bn`` - attentive statistics pooling;
    4. ``linear`` - projection to the speaker embedding.

    Instantiated from a config dict (e.g. a checkpoint's ``model_config``); config
    keys without an effect here are accepted and ignored. ``forward`` returns
    one embedding per utterance ``(B, embed_dim)``.
    """

    def __init__(
        self,
        F=72,
        C=24,
        out_channels=None,
        block_1d_type="conv+att",
        stages_setup=None,
        agg_gnorm=False,
        embed_dim=192,
        hop_length=160,
        **kwargs,
    ):
        super().__init__()
        self.backbone = ReDimNet2(
            F=F,
            C=C,
            out_channels=out_channels,
            block_1d_type=block_1d_type,
            stages_setup=stages_setup,
            agg_gnorm=agg_gnorm,
        )
        self.spec = TFMelBanks(n_mels=F, hop_length=hop_length)
        self.pool = ASTP(in_dim=self.backbone.out_dim)
        self.bn = nn.BatchNorm1d(self.pool.out_dim)
        self.linear = nn.Linear(self.pool.out_dim, embed_dim)

    def forward(self, x):
        x = self.spec(x).unsqueeze(1)
        out = self.backbone(x)
        bs, C, F, T = out.size()
        out = out.reshape(bs, C * F, T)
        return self.linear(self.bn(self.pool(out)))


def load_hub_model(url: str) -> ReDimNet2Wrap:
    """A released :class:`ReDimNet2Wrap` checkpoint, downloaded to the torch hub cache."""
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
    model = ReDimNet2Wrap(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model
