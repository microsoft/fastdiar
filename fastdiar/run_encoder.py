# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extract per-frame speaker embeddings from audio files and save them as fp16 ``.npy``.

Usage:
    python -m fastdiar.run_encoder audio.wav                       # -> audio.npy next to it
    python -m fastdiar.run_encoder audio.wav -o out/               # -> out/audio.npy
    python -m fastdiar.run_encoder audio.wav --shift-sec 0.16
    python -m fastdiar.run_encoder data/ --ext .flac -o emb/ -m small

Each ``.npy`` holds an ``(n_frames, embed_dim)`` float16 matrix of L2-normalized
embeddings, one per 80 ms frame. The file is streamed through the causal encoder
in ``--shift-sec`` blocks, as the diarizer does: 60 s on a GPU and 320 ms on the
CPU by default, the fastest on each; the result does not depend on it, up to
rounding. The encoder runs on a GPU when one is available (``--device``).

A directory input is searched recursively for files with the given extension.
Without ``--output`` every ``.npy`` is saved next to its audio file; with it, the
files go to that directory, mirroring the input's subdirectories.
"""

import numpy as np
import torch
from tqdm import tqdm

from fastdiar.cli import SAMPLE_RATE, build_parser, load_audio, load_model, plan_jobs
from fastdiar.encoder import StreamingEncoder, StreamingReDimNet2, default_shift_sec


class FileEncoder:
    """Embeds whole files, streamed in ``shift_sec`` blocks (default: :func:`default_shift_sec`)."""

    def __init__(self, model: StreamingReDimNet2, *, shift_sec: float | None = None) -> None:
        self.embed_dim = model.linear.out_features
        shift_sec = default_shift_sec(model) if shift_sec is None else shift_sec
        # The encoder block is a whole number of backbone frames.
        frame_samples = StreamingEncoder(model, shift_frames=1).frame_samples
        shift_frames = max(1, round(shift_sec * SAMPLE_RATE / frame_samples))
        self.streamer = StreamingEncoder(model, shift_frames=shift_frames)

    def __call__(self, wav: torch.Tensor) -> np.ndarray:
        """``(n_frames, embed_dim)`` float16 embeddings of a 16 kHz waveform."""
        self.streamer.reset()
        block = self.streamer.block
        embs = []
        for start in range(0, wav.shape[0], block):
            embs += self.streamer.push(wav[start : start + block])
        embs += self.streamer.flush()
        if not embs:  # shorter than one frame
            return np.zeros((0, self.embed_dim), dtype=np.float16)
        return torch.stack(embs).numpy().astype(np.float16)


def main() -> None:
    parser = build_parser(__doc__, output_help=".npy")
    parser.add_argument(
        "--shift-sec",
        type=float,
        default=None,
        help="streaming block length in seconds; it does not change the result "
        "(default: 60 on a GPU, 0.32 on the CPU, the fastest on each)",
    )
    args = parser.parse_args()
    jobs = plan_jobs(args.input, args.output, args.ext, suffix=".npy")
    encoder = FileEncoder(load_model(args), shift_sec=args.shift_sec)

    for audio_path, npy_path in tqdm(jobs, unit="file", disable=not args.input.is_dir()):
        embeddings = encoder(load_audio(audio_path))
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, embeddings)


if __name__ == "__main__":
    main()
