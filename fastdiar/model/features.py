# MIT License
#
# Copyright (c) 2026 Palabra.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction. The Software is provided "AS IS", without
# warranty of any kind. See the original ReDimNet2 repository for the full text.

"""TensorFlow-equivalent log-mel feature extractor (``feat_type='tf'``).

The STFT is implemented as a pair of fixed Conv1d kernels (cosine/sine), and
the mel projection as another fixed Conv1d. ``TFMelBanks`` is the only public
class; the rest are its building blocks. The streaming encoder reuses their
kernels to compute the same features causally.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hz2mel(hz):
    """Convert Hertz to Mels (element-wise)."""
    return 2595 * np.log10(1 + hz / 700.0)


def get_filterbanks(low_freq=20, high_freq=7600, nfilt=80, nfft=512, samplerate=16000):
    """Build a ``(nfft, nfilt)`` triangular mel filterbank matrix."""
    lowmel, highmel = hz2mel(low_freq), hz2mel(high_freq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    lower_edge_mel = melpoints[:-2].reshape(1, -1)
    center_mel = melpoints[1:-1].reshape(1, -1)
    upper_edge_mel = melpoints[2:].reshape(1, -1)

    spectrogram_bins_mel = hz2mel(np.linspace(0, samplerate // 2, nfft))[1:].reshape(-1, 1)
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
    mel_weights = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.vstack([np.zeros((1, nfilt)), mel_weights]).astype("float32")


class SpectralFeaturesTF(nn.Module):
    """Linear mel spectrogram via fixed Conv1d STFT + mel-projection kernels.

    The Hamming-windowed cosine/sine STFT kernels (``real_kernel_pt`` /
    ``image_kernel_pt``) and the mel matrix (``melbanks_pt``) are registered
    buffers, so they travel with the model checkpoint.
    """

    def __init__(
        self,
        frame_length=400,
        frame_step=160,
        fft_length=512,
        sample_rate=16000,
        eps=1e-8,
        low_freq=20,
        high_freq=7600,
        num_bins=80,
    ):
        super().__init__()
        self.length = frame_length
        self.shift = frame_step
        self.nfft = fft_length
        self.eps = eps

        win = np.hamming(self.length).astype("float32")
        freqs = np.arange(self.nfft)
        real = (
            np.asarray([np.cos(2 * np.pi * freqs * n / self.nfft) for n in range(self.nfft)])
            .astype("float32")
            .T
        )
        imag = (
            np.asarray([np.sin(2 * np.pi * freqs * n / self.nfft) for n in range(self.nfft)])
            .astype("float32")
            .T
        )
        real = (real[: self.length, : self.nfft // 2] * win[:, None])[:, None, :]
        imag = (imag[: self.length, : self.nfft // 2] * win[:, None])[:, None, :]
        self.register_buffer("real_kernel_pt", torch.from_numpy(real).permute(2, 1, 0).float())
        self.register_buffer("image_kernel_pt", torch.from_numpy(imag).permute(2, 1, 0).float())

        mel = get_filterbanks(
            nfilt=num_bins,
            nfft=self.nfft // 2,
            samplerate=sample_rate,
            low_freq=low_freq,
            high_freq=high_freq,
        )[:, :, None]
        self.register_buffer("melbanks_pt", torch.from_numpy(mel).permute(1, 0, 2).float())

    def forward(self, inputs):
        dtype = inputs.dtype
        inputs = inputs.float()
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)

        pad = self.shift // 2
        real_part = F.conv1d(inputs, self.real_kernel_pt, stride=self.shift, padding=pad)
        imag_part = F.conv1d(inputs, self.image_kernel_pt, stride=self.shift, padding=pad)

        power = (real_part.square() + imag_part.square()).clip(self.eps, 1 / self.eps)
        mel = F.conv1d(power, self.melbanks_pt, stride=1, padding=0)
        return mel.clip(self.eps, 1 / self.eps).to(dtype)


class NormalizeAudio(nn.Module):
    """Zero-mean / unit-variance normalization over the time axis."""

    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        mean = x.mean(dim=2, keepdims=True)
        std = x.std(dim=2, keepdims=True, unbiased=False)
        return ((x - mean) / (std + self.eps)).squeeze(1)


class PreEmphasis(nn.Module):
    """First-order pre-emphasis filter ``y[t] = x[t] - coef * x[t-1]``."""

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), "reflect")
        return F.conv1d(x, self.flipped_filter).squeeze(1)


class TFMelBanks(nn.Module):
    """Log-mel features with per-utterance mean subtraction.

    Pipeline: audio normalization -> pre-emphasis -> STFT mel -> log -> mean
    subtraction. The feature extraction runs in fp32 under ``no_grad`` (it has
    no learnable parameters).
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        f_min=20,
        f_max=7600,
        n_mels=80,
        eps=1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.torchfbank = nn.Sequential(
            NormalizeAudio(eps),
            PreEmphasis(),
            SpectralFeaturesTF(
                frame_length=win_length,
                frame_step=hop_length,
                fft_length=n_fft,
                sample_rate=sample_rate,
                eps=eps,
                low_freq=f_min,
                high_freq=f_max,
                num_bins=n_mels,
            ),
        )

    def forward(self, x):
        xdtype = x.dtype
        x = x.float()
        with torch.no_grad():
            with torch.amp.autocast(enabled=False, device_type="cuda"):
                x = self.torchfbank(x) + self.eps
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
        return x.to(xdtype)
