# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Training data for distillation: a weighted mix of VoxCeleb2 and LibriHeavyMix.

Every item is a fixed-length example with the same schema (see
:func:`collate_fn`):

* ``student_audio`` ``(T,)`` - the waveform the streaming student consumes;
* ``teacher_audio`` ``(S, T)`` - one clean crop per speaker slot, embedded by the
  frozen teacher; unused slots hold a copy of slot 0;
* ``seg_label`` ``(T,)`` int8 - per-sample target: ``0..S-1`` is the speaker
  slot, ``-1`` is ignored by the loss (overlap, or no speaker);
* ``speaker_valid`` ``(S,)`` bool - which speaker slots are real targets.

:class:`VoxCelebDataset` builds single- and two-speaker examples on the fly
from single-speaker utterances, with optional reverb/noise
(:class:`ReverbNoiseAugment`, adapted from wespeaker). :class:`LibriHeavyMixDataset`
reads pre-rendered meeting mixtures. Both can low-pass the student input.
"""

import io
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchaudio
from scipy import signal as scipy_signal
from scipy.io import wavfile
from torch.utils.data import Dataset, Sampler


def _lowpass_top_half(wav: torch.Tensor) -> torch.Tensor:
    """Zero the upper half of the spectrum: no energy above Nyquist/2 (~4 kHz at 16 kHz)."""
    spec = torch.fft.rfft(wav)
    spec[spec.shape[-1] // 2 :] = 0
    return torch.fft.irfft(spec, n=wav.shape[-1])


class _AudioDataset(Dataset):
    """Loading, cropping and low-pass augmentation shared by the training datasets.

    Args:
        target_duration: Length of every example, in seconds.
        sample_rate: Audio is resampled to this rate on load.
        max_speakers: Teacher speaker slots per example (the batch schema).
        lowpass_prob: Probability of low-passing the student input
            (:func:`_lowpass_top_half`); the teacher crops are never low-passed.
    """

    def __init__(
        self, target_duration: float, sample_rate: int, max_speakers: int, lowpass_prob: float
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.target_samples = round(target_duration * self.sample_rate)
        self.max_speakers = int(max_speakers)
        self.lowpass_prob = float(lowpass_prob)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Mono waveform at ``sample_rate``."""
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = self._resamplers[sr](wav)
        return wav

    def _crop(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One random ``target_samples`` window of equally long tensors (tiled if shorter)."""
        n, target = tensors[0].size(0), self.target_samples
        if n < target:
            reps = math.ceil(target / n)
            tensors = tuple(t.repeat(reps) for t in tensors)
            n *= reps
        start = random.randint(0, n - target)
        return tuple(t[start : start + target] for t in tensors)

    def _samples(self, seconds) -> int:
        return round(float(seconds) * self.sample_rate)

    def _maybe_lowpass(self, wav: torch.Tensor) -> torch.Tensor:
        if self.lowpass_prob > 0.0 and random.random() < self.lowpass_prob:
            return _lowpass_top_half(wav)
        return wav

    def _item(self, student, crops, valid, seg_label) -> dict[str, torch.Tensor]:
        """An example, with the teacher crops padded to ``max_speakers`` slots."""
        n_pad = self.max_speakers - len(crops)
        return {
            "student_audio": self._maybe_lowpass(student).contiguous().float(),
            "teacher_audio": torch.stack([*crops, *[crops[0]] * n_pad]).float(),
            "seg_label": seg_label.to(torch.int8),
            "speaker_valid": torch.tensor([*valid, *[False] * n_pad]),
        }


class VoxCelebDataset(_AudioDataset):
    """Single- and two-speaker examples built on the fly from single-speaker utterances.

    The parquet manifest has a ``path`` column (relative to ``audio_dir``), a
    ``spk_id`` column and, with ``use_vad``, a ``speech_segments`` column of
    ``[start_s, end_s]`` lists: only those regions of an utterance are kept
    (concatenated) before it is cropped.

    With probability ``multispeaker_prob`` an example is a mixture of two
    speakers' RMS-normalized crops, the second at a random signal-to-interference
    ratio in ``[sir_min_db, sir_max_db]``; otherwise it is one utterance. The
    mixture is one of two layouts, chosen uniformly (``T`` = example length):

    * overlap + change - spk1 on ``[0, T/2)``, spk2 on ``[T/2 - ov, T)``; the
      overlap is ignored by the loss. Its duration is drawn from a lognormal
      (``exp(N(overlap_log_mu, overlap_log_sigma))`` seconds, fitted to
      VoxConverse dev overlaps) clipped to ``[overlap_min_ms, overlap_max_ms]``.
    * two changes - spk1 -> spk2 -> spk1, every segment at least 1 s long.

    ``augment`` (e.g. :class:`ReverbNoiseAugment`) is applied to the student
    input only; the teacher embeds the clean crops.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        audio_dir: str | Path,
        *,
        target_duration: float,
        sample_rate: int,
        max_speakers: int,
        use_vad: bool,
        multispeaker_prob: float,
        overlap_min_ms: float,
        overlap_max_ms: float,
        overlap_log_mu: float,
        overlap_log_sigma: float,
        sir_min_db: float,
        sir_max_db: float,
        normalize_mixture: bool,
        lowpass_prob: float = 0.0,
        augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__(target_duration, sample_rate, max_speakers, lowpass_prob)
        if self.max_speakers < 2:
            raise ValueError(f"max_speakers must be >= 2; got {self.max_speakers}")
        self.audio_dir = Path(audio_dir)
        self.use_vad = bool(use_vad)
        self.multispeaker_prob = float(multispeaker_prob)
        self.overlap_min = self._samples(overlap_min_ms / 1000)
        self.overlap_max = self._samples(overlap_max_ms / 1000)
        self.overlap_log_mu = float(overlap_log_mu)
        self.overlap_log_sigma = float(overlap_log_sigma)
        self.sir_min_db = float(sir_min_db)
        self.sir_max_db = float(sir_max_db)
        self.normalize_mixture = bool(normalize_mixture)
        self.augment = augment

        df = pd.read_parquet(parquet_path)
        columns = ["path", "spk_id", *(["speech_segments"] if self.use_vad else [])]
        missing = sorted(set(columns) - set(df.columns))
        if missing:
            raise ValueError(f"{parquet_path} is missing columns {missing}")
        self.paths: list[str] = df["path"].astype(str).tolist()
        self.spk_ids: list = df["spk_id"].tolist()
        self.speech_segments = df["speech_segments"].tolist() if self.use_vad else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        if random.random() < self.multispeaker_prob:
            return self._mixture(idx)
        wav = self._load_crop(idx)
        student = self.augment(wav.clone()) if self.augment is not None else wav
        return self._item(student, [wav], [True], torch.zeros(self.target_samples))

    def _load_crop(self, idx: int) -> torch.Tensor:
        """One utterance, VAD-filtered with ``use_vad``, cropped to ``target_samples``."""
        wav = self._load_audio(self.audio_dir / self.paths[idx])
        if self.use_vad:
            segments = self.speech_segments[idx]
            pieces = [
                wav[max(0, self._samples(start)) : self._samples(end)]
                for start, end in ([] if segments is None else segments)
            ]
            pieces = [p for p in pieces if p.numel()]
            if pieces:
                wav = torch.cat(pieces)
            else:
                print(f"[VoxCelebDataset] no speech segments in {self.paths[idx]!r}; using all")
        return self._crop(wav)[0]

    def _other_speaker(self, idx: int) -> int:
        """A random utterance of a different speaker."""
        for _ in range(100):
            j = random.randrange(len(self.paths))
            if self.spk_ids[j] != self.spk_ids[idx]:
                return j
        raise RuntimeError("could not find an utterance of another speaker")

    def _mixture(self, idx: int):
        T = self.target_samples
        c1 = _rms_normalize(self._load_crop(idx))
        c2 = _rms_normalize(self._load_crop(self._other_speaker(idx)))
        g = self._sir_gain()
        mixture = torch.zeros(T)
        seg_label = torch.zeros(T, dtype=torch.int8)

        if random.random() < 0.5:
            # Overlap + change: spk2 starts `ov` samples before spk1 stops.
            half = T // 2
            start = half - self._overlap(max_len=half)
            mixture[:half] += c1[:half]
            mixture[start:] += g * c2[: T - start]
            seg_label[start:half] = -1
            seg_label[half:] = 1
        else:
            # Two changes: spk1 [0, p1), spk2 [p1, p2), spk1 [p2, T); each >= 1 s.
            one_sec = self._samples(1)
            if T - 2 * one_sec <= one_sec:
                p1, p2 = T // 3, 2 * T // 3
            else:
                p1 = random.randint(one_sec, T - 2 * one_sec)
                p2 = p1 + random.randint(one_sec, max(one_sec, T - one_sec - p1))
            mixture[:p1] = c1[:p1]
            mixture[p1:p2] = g * c2[: p2 - p1]
            mixture[p2:] = c1[p2:]
            seg_label[p1:p2] = 1

        if self.normalize_mixture:
            # The student must not tell overlap from the mixture's loudness.
            mixture = _rms_normalize(mixture)
        student = self.augment(mixture.clone()) if self.augment is not None else mixture
        return self._item(student, [c1, c2], [True, True], seg_label)

    def _sir_gain(self) -> float:
        """Interferer gain of a random signal-to-interference ratio (unit-power sources)."""
        return 10.0 ** (-random.uniform(self.sir_min_db, self.sir_max_db) / 20.0)

    def _overlap(self, max_len: int) -> int:
        """An overlap length in samples, from the lognormal prior."""
        lo, hi = max(1, self.overlap_min), min(self.overlap_max, max_len)
        seconds = math.exp(random.gauss(self.overlap_log_mu, self.overlap_log_sigma))
        return min(hi, max(lo, self._samples(seconds)))


def _rms_normalize(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return wav / torch.sqrt(wav.pow(2).mean() + eps)


class LibriHeavyMixDataset(_AudioDataset):
    """Pre-rendered LibriHeavyMix meeting mixtures (with per-speaker reverb).

    The parquet manifest has the columns:

    * ``path`` - the mixture ``<id>.flac``, at ``<mixture_dir>/<path>``;
    * ``num_speakers`` - speakers in the mixture;
    * ``segments`` - ``[offset_s, duration_s]`` of every speaker's activity; the
      clean source of speaker ``i`` is ``<source_dir>/<id>/<i>.flac``;
    * ``speech_segments`` - ``[start_s, end_s]`` VAD speech regions.

    The mixture is VAD-filtered and cropped together with its per-sample
    labels; the teacher embeds a crop of every clean source.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        mixture_dir: str | Path,
        source_dir: str | Path,
        *,
        target_duration: float,
        sample_rate: int,
        max_speakers: int,
        lowpass_prob: float = 0.0,
    ):
        super().__init__(target_duration, sample_rate, max_speakers, lowpass_prob)
        self.mixture_dir = Path(mixture_dir)
        self.source_dir = Path(source_dir)

        df = pd.read_parquet(parquet_path)
        missing = sorted({"path", "num_speakers", "segments", "speech_segments"} - set(df.columns))
        if missing:
            raise ValueError(f"{parquet_path} is missing columns {missing}")
        self.paths: list[str] = df["path"].astype(str).tolist()
        self.num_speakers: list[int] = df["num_speakers"].astype(int).tolist()
        self.segments: list = df["segments"].tolist()
        self.speech_segments: list = df["speech_segments"].tolist()
        if max(self.num_speakers) > self.max_speakers:
            raise ValueError(
                f"mixtures have up to {max(self.num_speakers)} speakers; "
                f"max_speakers={self.max_speakers} is too small"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        mixture_id = self.paths[idx].removesuffix(".flac")
        num_speakers = self.num_speakers[idx]
        mixture = self._load_audio(self.mixture_dir / self.paths[idx])
        n = mixture.size(0)

        # Per-sample labels: the speaker when exactly one is active, else -1.
        active = torch.zeros(num_speakers, n, dtype=torch.bool)
        for i, (offset, duration) in enumerate(self.segments[idx][:num_speakers]):
            active[i, self._samples(offset) : self._samples(offset + duration)] = True
        seg_label = torch.where(active.sum(0) == 1, active.to(torch.int8).argmax(0), -1)

        # Remove silence (mixtures always contain speech).
        speech = torch.zeros(n, dtype=torch.bool)
        for start, end in self.speech_segments[idx]:
            speech[self._samples(start) : self._samples(end)] = True
        if speech.any():
            mixture, seg_label = mixture[speech], seg_label[speech]

        student, seg_label = self._crop(mixture, seg_label)
        crops = [
            self._crop(self._load_audio(self.source_dir / mixture_id / f"{i}.flac"))[0]
            for i in range(num_speakers)
        ]
        return self._item(student, crops, [True] * num_speakers, seg_label)


class WeightedConcatDataset(Dataset):
    """Draws every item from a dataset picked with probability proportional to its weight.

    The index is the item's random seed (see :class:`RandomBatchSampler`): it
    seeds :mod:`random`, from which the dataset, the example and all its
    augmentation are drawn, so an item does not depend on the DataLoader worker
    that makes it. All datasets must share ``target_samples`` and
    ``max_speakers``, so their items collate into one batch.
    """

    def __init__(self, datasets: Sequence[_AudioDataset], weights: Sequence[float]):
        super().__init__()
        if not datasets or len(weights) != len(datasets):
            raise ValueError("need one weight per dataset, and at least one dataset")
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError(f"weights must be non-negative with a positive sum; got {weights}")
        for attr in ("target_samples", "max_speakers"):
            if len({getattr(d, attr) for d in datasets}) != 1:
                raise ValueError(f"all datasets must share {attr}")
        self.datasets = list(datasets)
        self.weights = [w / sum(weights) for w in weights]

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, seed: int):
        random.seed(seed)
        d = random.choices(self.datasets, weights=self.weights)[0]
        return d[random.randrange(len(d))]


class RandomBatchSampler(Sampler[list[int]]):
    """``num_batches_per_epoch`` batches of random item seeds (see :class:`WeightedConcatDataset`).

    The seeds are drawn from ``(seed, epoch, rank)``: every epoch, and every
    process of a distributed run, gets its own batches. The rank is read when an
    epoch starts, once the process group is up.
    """

    def __init__(self, batch_size: int, num_batches_per_epoch: int, seed: int):
        self.batch_size = int(batch_size)
        self.num_batches_per_epoch = int(num_batches_per_epoch)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        rng = random.Random(f"{self.seed}/{self._epoch}/{rank}")
        for _ in range(self.num_batches_per_epoch):
            yield [rng.getrandbits(63) for _ in range(self.batch_size)]

    def __len__(self) -> int:
        return self.num_batches_per_epoch


def collate_fn(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack the items: ``(B, T)``, ``(B, S, T)``, ``(B, T)`` and ``(B, S)``."""
    return {key: torch.stack([it[key] for it in items]) for key in items[0]}


# ---------------------------------------------------------------------------
# Reverb / additive-noise augmentation, adapted from wespeaker's
# `add_reverb_noise` (wespeaker/dataset/processor.py), with RIRs and noises read
# from directories instead of LMDB shards:
#   * MUSAN (additive noise, https://www.openslr.org/17/), with `noise/`,
#     `speech/` and `music/` sub-trees that select the SNR range;
#   * RIRS_NOISES (reverberation, https://www.openslr.org/28/), any tree of RIR
#     wav files (e.g. `simulated_rirs/`).
# ---------------------------------------------------------------------------

# SNR range (dB) of the added noise, by MUSAN category (wespeaker's values).
_SNR_RANGES = {"noise": (0.0, 15.0), "speech": (10.0, 30.0), "music": (5.0, 15.0)}


def _random_chunk(data: np.ndarray, chunk_len: int) -> np.ndarray:
    """A random ``chunk_len`` window of ``data``, tiled if shorter."""
    if len(data) >= chunk_len:
        start = random.randint(0, len(data) - chunk_len)
        return data[start : start + chunk_len]
    return np.tile(data, chunk_len // len(data) + 1)[:chunk_len]


class RandomWavSource:
    """Random ``.wav`` files from directory trees, as ``(key, raw bytes)``.

    ``sources`` is one directory, or a ``{prefix: directory}`` mapping whose
    keys prefix the returned keys (``"<prefix>/<relative path>"``), e.g. the
    MUSAN categories.
    """

    def __init__(self, sources: str | Path | Mapping[str, str | Path]):
        roots = sources if isinstance(sources, Mapping) else {"": sources}
        self._entries: list[tuple[str, Path]] = []
        for prefix, root in roots.items():
            for path in sorted(p for p in Path(root).glob("**/*.wav") if p.is_file()):
                key = path.relative_to(root).as_posix()
                self._entries.append((f"{prefix}/{key}" if prefix else key, path))
        if not self._entries:
            raise ValueError(f"no .wav files under {sources}")

    def __len__(self) -> int:
        return len(self._entries)

    def random_one(self) -> tuple[str, bytes]:
        key, path = random.choice(self._entries)
        return key, path.read_bytes()


class ReverbNoiseAugment:
    """With probability ``aug_prob``: reverb with a random RIR, or random additive noise.

    The two are equally likely, as in wespeaker. The output is peak-normalized
    and has the input's length.
    """

    def __init__(
        self,
        reverb_source: RandomWavSource,
        noise_source: RandomWavSource,
        sample_rate: int,
        aug_prob: float,
    ):
        self.reverb_source = reverb_source
        self.noise_source = noise_source
        self.sample_rate = int(sample_rate)
        self.aug_prob = float(aug_prob)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """Augment a 1-D waveform."""
        if random.random() >= self.aug_prob:
            return wav
        audio = wav.numpy()
        out = self._reverb(audio) if random.random() < 0.5 else self._noise(audio)
        out = out / (np.max(np.abs(out)) + 1e-4)
        return torch.from_numpy(out.astype(np.float32))

    def _read(self, source: RandomWavSource) -> tuple[str, int, np.ndarray]:
        key, data = source.random_one()
        sr, audio = wavfile.read(io.BytesIO(data))
        audio = audio.astype(np.float32)
        return key, sr, audio.mean(axis=1) if audio.ndim > 1 else audio

    def _reverb(self, audio: np.ndarray) -> np.ndarray:
        _, sr, rir = self._read(self.reverb_source)
        if sr != self.sample_rate:
            rir = scipy_signal.resample(rir, int(len(rir) / sr * self.sample_rate))
        rir = rir / (np.sqrt(np.sum(rir**2)) + 1e-12)
        return scipy_signal.convolve(audio, rir, mode="full")[: len(audio)]

    def _noise(self, audio: np.ndarray) -> np.ndarray:
        key, sr, noise = self._read(self.noise_source)
        noise = noise / (1 << 15)  # int16 -> [-1, 1]
        if sr != self.sample_rate:
            # Crop at the source rate first, so a long noise is not resampled whole.
            noise = _random_chunk(noise, int(len(audio) / self.sample_rate * sr))
            noise = scipy_signal.resample(noise, len(audio))
        else:
            noise = _random_chunk(noise, len(audio))
        snr = random.uniform(*_SNR_RANGES.get(key.split("/")[0], (0.0, 15.0)))
        audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)
        noise_db = 10 * np.log10(np.mean(noise**2) + 1e-4)
        return audio + np.sqrt(10 ** ((audio_db - noise_db - snr) / 10)) * noise
