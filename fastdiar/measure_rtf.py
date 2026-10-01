# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Measure the real-time factor (RTF) of the streaming diarizer.

Usage:
    python -m fastdiar.measure_rtf voxconverse/wav/dev -m small

Diarizes the first 5 files (sorted by name) of the VoxConverse dev audio
directory, each cut to its first minute, and prints the RTF of every file and
of all of them: diarization time divided by audio duration, so values below 1
are faster than real time. Model and audio loading are not timed, and one
untimed warm-up run absorbs one-off startup costs. Everything runs on a single
CPU thread.
"""

import os

# One CPU thread for every compute library. OpenMP / BLAS read these when they
# are first imported, so they must be set before numpy and torch are.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from fastdiar.cli import add_model_args, load_audio, load_model  # noqa: E402
from fastdiar.diarizer import SAMPLE_RATE, StreamingDiarizer  # noqa: E402

NUM_FILES = 5
MAX_SEC = 60.0
WARMUP_SEC = 10.0


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("audio_dir", type=Path, help="VoxConverse dev audio directory")
    add_model_args(parser)
    args = parser.parse_args()

    files = sorted(args.audio_dir.glob("*.wav"))[:NUM_FILES]
    if not files:
        raise SystemExit(f"no .wav files found in {args.audio_dir}")
    wavs = [load_audio(path)[: int(MAX_SEC * SAMPLE_RATE)] for path in files]
    diarizer = StreamingDiarizer(load_model(args))
    diarizer(wavs[0][: int(WARMUP_SEC * SAMPLE_RATE)])

    print(f"model: {args.model}, CPU threads: {torch.get_num_threads()}")
    print(f"{'file':<12}{'audio (s)':>10}{'time (s)':>10}{'RTF':>8}")
    total_audio = total_time = 0.0
    for path, wav in zip(files, wavs, strict=True):
        start = time.perf_counter()
        diarizer(wav)
        elapsed = time.perf_counter() - start
        duration = wav.shape[0] / SAMPLE_RATE
        total_audio += duration
        total_time += elapsed
        print(f"{path.stem:<12}{duration:>10.2f}{elapsed:>10.2f}{elapsed / duration:>8.3f}")
    print(f"{'total':<12}{total_audio:>10.2f}{total_time:>10.2f}{total_time / total_audio:>8.3f}")


if __name__ == "__main__":
    main()
