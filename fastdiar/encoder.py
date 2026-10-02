# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Streaming inference for the causal ReDimNet2 models (:data:`CHECKPOINTS`).

Processes audio incrementally in short chunks, caching intermediate states
(audio statistics, STFT and conv buffers) to avoid redundant computation.
Outputs a stream of L2-normalized per-frame speaker embeddings.

:meth:`StreamingReDimNet2.forward` runs the same model over a whole batch at
once; it is the forward used in training.

Usage:
    model = load_streaming_model("large")  # or a local checkpoint file

    streamer = StreamingInference(model)
    for chunk in audio_stream:
        embeddings = streamer.process_chunk(chunk)
        # embeddings: (1, N, embed_dim), L2-normalized
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastdiar.model.blocks import ConvBlock2d
from fastdiar.model.features import TFMelBanks
from fastdiar.model.redimnet2 import ReDimNet2
from fastdiar.model.resblocks import ResBasicBlock

RELEASE_URL = "https://github.com/microsoft/fastdiar/releases/download/v0.1.0"
# Released checkpoint of every model size; the name ends with the start of the file's SHA-256.
CHECKPOINTS = {
    "small": "fastdiar-small-90be5da5.pt",
    "medium": "fastdiar-medium-32df7509.pt",
    "large": "fastdiar-large-f6d71444.pt",
}


def _causal_conv2d(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """``conv`` over ``x``, which already starts with ``kernel_time - 1`` past frames.

    Frequency is padded symmetrically; time is not padded, so the output has
    ``kernel_time - 1`` fewer frames than ``x`` and none of them sees the future.
    """
    pad_freq = conv.kernel_size[0] // 2
    return F.conv2d(
        F.pad(x, (0, 0, pad_freq, pad_freq)),
        conv.weight,
        conv.bias,
        stride=conv.stride,
        padding=0,
        dilation=conv.dilation,
        groups=conv.groups,
    )


def _left_pad_time(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Zero past frames for :func:`_causal_conv2d` at the start of a sequence."""
    return F.pad(x, (conv.kernel_size[1] - 1, 0))


class CausalConvBlock2d(ConvBlock2d):
    """:class:`ConvBlock2d` whose 3x3 convs see past frames only (same weights)."""

    def forward(self, x):
        block = self.conv_block
        out = block.conv1pw(_causal_conv2d(block.conv1, _left_pad_time(block.conv1, x)))
        out = block.bn1(block.relu(out))
        out = block.conv2pw(_causal_conv2d(block.conv2, _left_pad_time(block.conv2, out)))
        out = block.bn2(out)
        return block.relu(out + x)


class StreamingReDimNet2(nn.Module):
    """Causal ReDimNet2 streaming model.

    Log-mel front-end, :class:`ReDimNet2` backbone with time-causal residual
    blocks and a per-frame projection head (no temporal pooling).
    :class:`StreamingInference` runs the submodules chunk by chunk;
    :meth:`forward` runs a whole batch at once (training). Instantiated from a
    config dict (``configs/*.json``, or a checkpoint's ``model_config``).
    """

    def __init__(
        self,
        F=72,
        C=24,
        out_channels=None,
        block_1d_type="fc",
        stages_setup=None,
        embed_dim=192,
        hop_length=160,
        time_upsampling=True,
    ):
        super().__init__()
        self.backbone = ReDimNet2(
            F=F,
            C=C,
            out_channels=out_channels,
            block_1d_type=block_1d_type,
            stages_setup=stages_setup,
            conv_block=CausalConvBlock2d,
        )
        self.spec = TFMelBanks(n_mels=F, hop_length=hop_length)
        self.bn = nn.BatchNorm1d(self.backbone.out_dim)
        self.linear = nn.Linear(self.backbone.out_dim, embed_dim)
        # When False the final feature map stays at the coarse time resolution
        # (downsampled by `time_stride`) instead of being upsampled back.
        self.time_upsampling = time_upsampling

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Per-frame embeddings ``(B, T, embed_dim)`` (not normalized) of ``(B, samples)`` audio.

        The training-time forward. It differs from :class:`StreamingInference`
        in two places: the audio is normalized over the whole input rather than
        by running statistics, and the stem conv sees one future frame
        (``padding="same"``) instead of two past ones.
        """
        out = self.backbone(self._causal_spec(wav))
        if not self.time_upsampling and self.backbone.time_stride > 1:
            out = out[..., :: self.backbone.time_stride]  # per-frame ops follow, so this is exact
        bs, C, Fr, T = out.size()
        return self.linear(self.bn(out.reshape(bs, C * Fr, T)).transpose(1, 2))

    def _causal_spec(self, wav: torch.Tensor) -> torch.Tensor:
        """Log-mel ``(B, 1, F, T)``: causal pre-emphasis, STFT and running mean subtraction."""
        norm_audio, preemph, stft = self.spec.torchfbank
        dtype = wav.dtype
        with torch.no_grad(), torch.amp.autocast(enabled=False, device_type=wav.device.type):
            x = norm_audio(wav.float()).unsqueeze(1)
            x = F.conv1d(F.pad(x, (1, 0)), preemph.flipped_filter)
            # Left-pad so frame i never peeks past sample (i+1)*shift - 1.
            x = F.pad(x, (stft.length - stft.shift, 0))
            real = F.conv1d(x, stft.real_kernel_pt, stride=stft.shift)
            imag = F.conv1d(x, stft.image_kernel_pt, stride=stft.shift)
            power = (real.square() + imag.square()).clip(stft.eps, 1 / stft.eps)
            mel = F.conv1d(power, stft.melbanks_pt).clip(stft.eps, 1 / stft.eps)
            x = (mel + self.spec.eps).log()
            counts = torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype)
            x = x - torch.cumsum(x, dim=-1) / counts
        return x.unsqueeze(1).to(dtype)


def load_streaming_model(
    model: str = "large", device="cpu", dtype: torch.dtype | None = None
) -> StreamingReDimNet2:
    """A streaming model: a released one by size, or a local checkpoint file.

    A size in :data:`CHECKPOINTS` is downloaded from the GitHub release to the
    torch hub cache on first use (and checked against the hash in its name). A
    checkpoint holds the model's config (``model_config``) and weights
    (``state_dict``).

    ``dtype`` is the precision of the backbone and the projection head
    (default: bf16 on a GPU, fp32 on the CPU); the log-mel front-end always
    runs in fp32. Not fp16: the LayerNorm statistics overflow it, and PyTorch
    sends fp16 depthwise convolutions to cuDNN, which spends ~0.3 s setting each
    one up for every new input length.
    """
    if model in CHECKPOINTS:
        checkpoint = torch.hub.load_state_dict_from_url(
            f"{RELEASE_URL}/{CHECKPOINTS[model]}",
            map_location="cpu",
            check_hash=True,
            weights_only=True,
        )
    else:
        checkpoint = torch.load(model, map_location="cpu", weights_only=True)
    net = StreamingReDimNet2(**checkpoint["model_config"])
    net.load_state_dict(checkpoint["state_dict"])
    net = net.to(device).eval()
    if dtype is None:
        dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    for module in (net.backbone, net.bn, net.linear):
        module.to(dtype)
    return net


@dataclass
class StreamingState:
    """Holds all mutable state for streaming inference."""

    # Audio preprocessing (the running sums stay on the model's device)
    audio_buffer: torch.Tensor | None = None
    preemph_last: torch.Tensor | None = None
    audio_sum: torch.Tensor | float = 0.0
    audio_sumsq: torch.Tensor | float = 0.0
    audio_count: int = 0

    # Spectrogram causal mean subtraction
    cms_sum: torch.Tensor | None = None
    cms_count: int = 0

    # Per-layer conv buffers: key -> Tensor
    conv_buffers: dict[str, torch.Tensor] = field(default_factory=dict)


class StreamingInference:
    """Streaming inference wrapper for the causal ReDimNet2 model.

    Processes audio in chunks of `min_chunk_samples` (or multiples thereof),
    caching all intermediate state for efficient incremental computation.

    Args:
        model: A :class:`StreamingReDimNet2` (see :func:`load_streaming_model`).
    """

    def __init__(self, model: StreamingReDimNet2):
        self.model = model
        self.model.eval()

        # Spectrogram front-end: its fixed kernels are reused causally.
        spec = model.spec
        norm_audio, preemph, stft = spec.torchfbank
        self.hop_length = stft.shift
        self._norm_eps = norm_audio.eps
        self._preemph_filter = preemph.flipped_filter
        self._stft_real_kernel = stft.real_kernel_pt
        self._stft_imag_kernel = stft.image_kernel_pt
        self._stft_melbanks = stft.melbanks_pt
        self._stft_eps = stft.eps
        self._stft_buffer_size = stft.length - stft.shift
        self._cms_eps = spec.eps

        # Backbone reference. Stage submodules are accessed by name directly
        # via `self.backbone.stages[i]` (see ReDimNet2Stage).
        self.backbone = model.backbone
        self.time_stride = self.backbone.time_stride
        self.time_upsampling = model.time_upsampling

        # Minimum chunk: time_stride frames * hop_length samples
        self.min_chunk_samples = self.time_stride * self.hop_length

        # Projection head (the streaming model has no temporal pooling)
        self.bn = model.bn
        self.linear = model.linear

        # State
        self.state = StreamingState()

    @property
    def device(self):
        """Device of the model parameters."""
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        """Compute precision of the backbone and the head (the front-end is fp32)."""
        return next(self.backbone.parameters()).dtype

    def reset(self):
        """Reset all streaming state. Call before processing a new utterance."""
        self.state = StreamingState()

    @torch.no_grad()
    def process_chunk(self, audio_chunk: torch.Tensor) -> torch.Tensor | None:
        """Process a chunk of raw audio and return new speaker embeddings.

        Args:
            audio_chunk: Raw audio samples, shape (samples,) or (1, samples).
                Must contain at least `min_chunk_samples` samples; samples past
                the last whole multiple of `min_chunk_samples` are dropped.

        Returns:
            L2-normalized speaker embeddings of shape (1, N, embed_dim) where
            N is the number of new output frames, or None if the chunk is too
            short to produce output.
        """
        audio_chunk = self._prepare_chunk(audio_chunk)
        if audio_chunk is None:
            return None

        # 1. Compute spectrogram frames (fp32) -> (1, 1, n_mels, T_new)
        spec = self._compute_spec_streaming(audio_chunk).unsqueeze(1).to(self.dtype)

        # 2. Run backbone streaming + projection head
        backbone_out = self._backbone_streaming(spec)
        return self._head(backbone_out)

    def _prepare_chunk(self, audio_chunk: torch.Tensor) -> torch.Tensor | None:
        """Normalize shape to ``(1, 1, samples)`` and clip to a whole number of
        ``min_chunk_samples`` blocks; ``None`` if the chunk is too short."""
        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk.unsqueeze(0).unsqueeze(0)
        elif audio_chunk.ndim == 2:
            audio_chunk = audio_chunk.unsqueeze(1)

        audio_chunk = audio_chunk.to(self.device)
        num_samples = audio_chunk.shape[-1]
        if num_samples < self.min_chunk_samples:
            return None
        usable = (num_samples // self.min_chunk_samples) * self.min_chunk_samples
        return audio_chunk[..., :usable]

    def _head(self, backbone_out: torch.Tensor) -> torch.Tensor:
        """Per-frame projection head -> L2-normalized per-frame embeddings.

        Each frame is projected independently, so this is inherently causal and
        block-size invariant.
        """
        bs, C, Fr, T = backbone_out.size()
        backbone_out = backbone_out.reshape(bs, C * Fr, T)
        embeddings = self.linear(self.bn(backbone_out).transpose(1, 2))
        return F.normalize(embeddings.float(), p=2, dim=-1)

    # ------------------------------------------------------------------
    #                    Spectrogram Streaming
    # ------------------------------------------------------------------

    def _compute_spec_streaming(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram frames incrementally.

        Args:
            audio: (1, 1, num_samples) raw audio chunk.

        Returns:
            (1, n_mels, T_new) log-mel spectrogram with causal CMS applied.
        """
        x = audio.float()

        # Normalize audio (causal running normalization)
        x = self._normalize_audio_streaming(x)

        # Pre-emphasis (causal, with 1-sample buffer)
        x = self._preemph_streaming(x)

        # STFT + mel filterbank (causal, with frame buffer)
        spec = self._stft_streaming(x)

        # Log
        spec = spec.clamp(min=self._cms_eps).log()

        # Causal mean subtraction
        return self._cms_streaming(spec)

    def _normalize_audio_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample causal audio normalization using cumulative statistics.

        Each sample is normalized by the running mean/std computed from all
        samples seen so far (including itself). This is chunk-size independent.
        """
        # x: (1, 1, N)
        N = x.shape[-1]

        # Cumulative stats within this chunk
        chunk_cumsum = torch.cumsum(x, dim=-1)  # (1, 1, N)
        chunk_cumsumsq = torch.cumsum(x * x, dim=-1)

        # Global cumulative stats
        global_cumsum = self.state.audio_sum + chunk_cumsum
        global_cumsumsq = self.state.audio_sumsq + chunk_cumsumsq

        counts = torch.arange(
            self.state.audio_count + 1,
            self.state.audio_count + N + 1,
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, -1)

        means = global_cumsum / counts
        vars_ = (global_cumsumsq / counts - means * means).clamp(min=1e-12)
        stds = torch.sqrt(vars_)

        # Update state with final values (tensors: no device sync)
        self.state.audio_sum = global_cumsum[..., -1:]
        self.state.audio_sumsq = global_cumsumsq[..., -1:]
        self.state.audio_count += N

        return (x - means) / (stds + self._norm_eps)

    def _preemph_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """Causal pre-emphasis with 1-sample buffer."""
        # x: (1, 1, N)
        if self.state.preemph_last is None:
            self.state.preemph_last = torch.zeros(1, 1, 1, device=x.device, dtype=x.dtype)

        # Prepend last sample from previous chunk
        x_padded = torch.cat([self.state.preemph_last, x], dim=-1)

        # Update buffer
        self.state.preemph_last = x[..., -1:]

        # Apply pre-emphasis filter
        return F.conv1d(x_padded, self._preemph_filter)

    def _stft_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """Causal STFT with frame buffer, returns mel spectrogram (linear)."""
        # x: (1, 1, N) after pre-emphasis
        # Initialize or prepend STFT buffer
        if self.state.audio_buffer is None:
            self.state.audio_buffer = torch.zeros(
                1, 1, self._stft_buffer_size, device=x.device, dtype=x.dtype
            )

        x_buffered = torch.cat([self.state.audio_buffer, x], dim=-1)

        # Update buffer: last _stft_buffer_size samples
        self.state.audio_buffer = x_buffered[..., -self._stft_buffer_size :]

        # Conv1d STFT (no padding, stride=hop_length)
        real_part = F.conv1d(x_buffered, self._stft_real_kernel, stride=self.hop_length, padding=0)
        imag_part = F.conv1d(x_buffered, self._stft_imag_kernel, stride=self.hop_length, padding=0)

        # Power spectrum
        power = real_part.square() + imag_part.square()
        power = power.clamp(min=self._stft_eps, max=1.0 / self._stft_eps)

        # Mel filterbank
        mel = F.conv1d(power, self._stft_melbanks, stride=1, padding=0)
        return mel.clamp(min=self._stft_eps, max=1.0 / self._stft_eps)

    def _cms_streaming(self, spec: torch.Tensor) -> torch.Tensor:
        """Causal cepstral mean subtraction with running accumulator."""
        # spec: (1, n_mels, T_new)
        T_new = spec.shape[-1]

        if self.state.cms_sum is None:
            self.state.cms_sum = torch.zeros(
                1, spec.shape[1], 1, device=spec.device, dtype=spec.dtype
            )

        # Cumulative sum for this chunk
        chunk_cumsum = torch.cumsum(spec, dim=-1)  # (1, F, T_new)

        # Global cumsum = past_sum + chunk_cumsum
        global_cumsum = self.state.cms_sum + chunk_cumsum

        # Frame counts
        counts = torch.arange(
            self.state.cms_count + 1,
            self.state.cms_count + T_new + 1,
            device=spec.device,
            dtype=spec.dtype,
        ).view(1, 1, -1)

        # Causal mean
        causal_mean = global_cumsum / counts

        # Update state
        self.state.cms_sum = global_cumsum[..., -1:]
        self.state.cms_count += T_new

        return spec - causal_mean

    # ------------------------------------------------------------------
    #                    Backbone Streaming
    # ------------------------------------------------------------------

    def _backbone_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone with streaming state management.

        Args:
            x: (1, 1, F, T_new) spectrogram chunk (T_new = time_stride * k).

        Returns:
            Backbone output tensor.
        """
        # Stem: Conv2d (causal) + LayerNorm + to1d
        conv, ln, to1d = self.backbone.stem
        x = to1d(ln(self._causal_conv2d(conv, x, "stem.conv")))
        outputs_1d = [x]

        # Process stages
        for stage_idx, stage in enumerate(self.backbone.stages):
            outputs_1d.append(self._stage_streaming(stage, outputs_1d, stage_idx))

        # Final weighted aggregation
        x = self.backbone.fin_wght1d(outputs_1d)

        # Optionally drop the final time upsampling: keep the aggregated map at
        # the coarse resolution (T // time_stride). Chunk lengths are multiples
        # of time_stride, so the coarse frame grid stays aligned across chunks.
        if not self.time_upsampling and self.time_stride > 1:
            x = x[..., :: self.time_stride]

        # Final reshape and head
        x = self.backbone.fin_to2d(x)
        return self.backbone.head(x)

    def _stage_streaming(
        self, stage: nn.Module, outputs_1d: list[torch.Tensor], stage_idx: int
    ) -> torch.Tensor:
        """Process one stage with streaming state.

        Mirrors :meth:`ReDimNet2Stage.forward`, but routes the 2-D residual
        blocks through the incremental (buffered) causal conv helper. Every
        other submodule is stateless along time.
        """
        # 1. Stack-and-weight aggregation
        x = stage.aggregate(outputs_1d)

        # 2. to2d reshape
        x = stage.to2d(x)

        # 3. Strided downsample conv (kernel==stride, non-overlapping: no buffer)
        x = stage.downsample(x)

        # 4. 2-D residual blocks (need causal time buffers)
        for blk_idx, conv_block in enumerate(stage.conv_blocks):
            x = self._resblock_streaming(
                conv_block.conv_block, x, f"stage{stage_idx}.block{blk_idx}"
            )

        # 5. Channel projection (1x1 conv + BN; Identity when absent)
        x = stage.project(x)

        # 6. to1d
        x = stage.to1d(x)

        # 7. 1-D temporal-context block ('fc': pointwise, so per-frame)
        x = stage.context(x)

        # 8. Upsample back to T*
        return stage.upsample(x)

    # ------------------------------------------------------------------
    #              Per-Module Streaming Helpers
    # ------------------------------------------------------------------

    def _causal_conv2d(self, conv: nn.Conv2d, x: torch.Tensor, key: str) -> torch.Tensor:
        """Apply a Conv2d causally with a time buffer.

        The time axis sees ``kernel_time - 1`` past frames (buffered across
        chunks) and no future ones; the frequency axis keeps symmetric padding.

        Args:
            conv: The Conv2d module.
            x: Input tensor (1, C, F, T_new).
            key: State dict key for this layer's buffer.
        """
        pad_time = conv.kernel_size[1] - 1
        if key not in self.state.conv_buffers:
            # Initialize with zeros
            B, C, Fr, _ = x.shape
            self.state.conv_buffers[key] = torch.zeros(
                B, C, Fr, pad_time, device=x.device, dtype=x.dtype
            )

        # Concat buffer and new input along time; keep the last pad_time frames
        x_full = torch.cat([self.state.conv_buffers[key], x], dim=-1)
        self.state.conv_buffers[key] = x_full[..., -pad_time:].clone()
        return _causal_conv2d(conv, x_full)

    def _resblock_streaming(
        self, block: ResBasicBlock, x: torch.Tensor, key_prefix: str
    ) -> torch.Tensor:
        """Process a ResBasicBlock causally with conv buffers."""
        residual = x

        # Conv1 (causal 3x3)
        out = self._causal_conv2d(block.conv1, x, f"{key_prefix}.conv1")
        out = block.conv1pw(out)
        out = block.relu(out)
        out = block.bn1(out)

        # Conv2 (causal 3x3)
        out = self._causal_conv2d(block.conv2, out, f"{key_prefix}.conv2")
        out = block.conv2pw(out)
        out = block.bn2(out)

        # Residual
        out = out + residual
        return block.relu(out)


def default_shift_sec(model: StreamingReDimNet2) -> float:
    """The fastest streaming step, in seconds, for ``model``'s device.

    The step only sets how often embeddings are produced, not their value. On
    a GPU a 320 ms step leaves it mostly idle, waiting for kernel launches,
    while 60 s blocks keep it busy (over 20x faster; longer blocks gain
    nothing and take more memory: 1.2 GB for 60 s in bf16). On the CPU long
    blocks are slower (they do not fit in cache), so the step stays 320 ms.
    """
    return 60.0 if next(model.parameters()).device.type == "cuda" else 0.32


class StreamingEncoder:
    """Incremental per-frame speaker-embedding extractor.

    Buffers incoming speech and runs it through :class:`StreamingInference` in
    blocks of ``shift_frames`` output frames.

    Args:
        model: A :class:`StreamingReDimNet2` (see :func:`load_streaming_model`).
        shift_frames: Number of output frames per processed block (latency knob).
    """

    def __init__(self, model: StreamingReDimNet2, *, shift_frames: int = 25) -> None:
        self.engine = StreamingInference(model)
        self.frame_samples = self.engine.min_chunk_samples
        self.shift_frames = max(1, int(shift_frames))
        self.block = self.shift_frames * self.frame_samples
        self.reset()

    def reset(self) -> None:
        """Reset the backbone state and the sample buffer (per file)."""
        self.engine.reset()
        self._buf = torch.zeros(0)  # speech samples not yet forming a full block

    def push(self, speech: torch.Tensor) -> list[torch.Tensor]:
        """Feed contiguous speech samples and return the new frame embeddings.

        Args:
            speech: 1-D float tensor of speech samples (silence removed).

        Returns:
            List of ``(embed_dim,)`` L2-normalized embeddings (on the CPU), one
            per new frame, in order.
        """
        self._buf = torch.cat([self._buf, speech.reshape(-1).float()])
        n_blocks = self._buf.shape[0] // self.block
        if not n_blocks:
            return []
        usable = n_blocks * self.block
        block, self._buf = self._buf[:usable], self._buf[usable:]
        return self._run(block)

    def flush(self) -> list[torch.Tensor]:
        """Process the remaining buffered speech (whole frames only)."""
        usable = (self._buf.shape[0] // self.frame_samples) * self.frame_samples
        if not usable:
            return []
        block, self._buf = self._buf[:usable], self._buf[usable:]
        return self._run(block)

    def _run(self, block: torch.Tensor) -> list[torch.Tensor]:
        """Run one block through the backbone."""
        embs = self.engine.process_chunk(block.unsqueeze(0))
        if embs is None:
            return []
        return list(embs[0].cpu())  # one device-to-host copy per block
