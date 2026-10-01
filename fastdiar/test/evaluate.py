# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluate the streaming diarizer on a diarization benchmark and print its DER.

Usage:
    python -m fastdiar.test.evaluate voxconverse \\
        --audio-dir voxconverse/wav/test --labels voxconverse/test -m small -o out/voxconverse

Every file is streamed through :class:`fastdiar.diarizer.StreamingDiarizer` and
scored with pyannote's ``DiarizationErrorRate`` (collar 0; overlapped speech is
scored unless ``--skip-overlap`` is given).
Files are diarized in parallel, one process per CPU by default (``--workers``).
When a GPU is available (``--device``) the encoder runs on it, and the files are
diarized one after the other in a single process.
With ``--output-dir`` the hypothesis of each file is saved there as
``<uri>.rttm``, and files that already have one are not diarized again. See
``fastdiar/test/README.md`` for how to obtain each dataset.
"""

import argparse
import io
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm

from fastdiar.cli import add_device_args, add_model_args, load_audio, load_model, resolve_device
from fastdiar.diarizer import StreamingDiarizer
from fastdiar.test.datasets import DATASETS

# Speaker-count split of the report, for datasets with more than this many speakers.
MAX_FEW_SPEAKERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=sorted(DATASETS), help="dataset name")
    parser.add_argument("--audio-dir", type=Path, required=True, help="dataset audio directory")
    parser.add_argument(
        "--labels", type=Path, required=True, help="dataset labels (see the README per dataset)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="save / reuse <uri>.rttm hypotheses"
    )
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        help="exclude overlapped speech regions from scoring (default: scored)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="parallel diarization processes on the CPU (default: one per CPU); 1 runs "
        "in-process, as always on a GPU",
    )
    add_model_args(parser)
    add_device_args(parser)
    return parser.parse_args()


def diarize(
    diarizer: StreamingDiarizer, audio_path: Path, uri: str, output_dir: Path | None
) -> str:
    """RTTM hypothesis of one file, also saved to ``output_dir`` when given."""
    rttm = diarizer(load_audio(audio_path), uri=uri)
    if output_dir is not None:
        # Write-then-rename, so an interrupted run never leaves a partial cache file.
        rttm_path = output_dir / f"{uri}.rttm"
        tmp_path = rttm_path.with_suffix(".rttm.tmp")
        tmp_path.write_text(rttm)
        tmp_path.replace(rttm_path)
    return rttm


# Diarizer of a worker process, built once by its initializer.
_worker_diarizer: StreamingDiarizer | None = None


def _init_worker(args: argparse.Namespace) -> None:
    global _worker_diarizer
    torch.set_num_threads(1)  # parallelism comes from the processes
    _worker_diarizer = StreamingDiarizer(load_model(args))


def _diarize_in_worker(audio_path: Path, uri: str, output_dir: Path | None) -> str:
    return diarize(_worker_diarizer, audio_path, uri, output_dir)


def diarize_all(items: list, args: argparse.Namespace) -> dict[str, str]:
    """RTTM hypothesis of every item: cached in ``--output-dir``, or diarized now."""
    rttms, todo = {}, []
    for audio_path, uri, _ in items:
        cached = None if args.output_dir is None else args.output_dir / f"{uri}.rttm"
        if cached is not None and cached.is_file():
            rttms[uri] = cached.read_text()
        else:
            todo.append((audio_path, uri))

    workers = min(args.workers, len(todo))
    with tqdm(total=len(items), initial=len(rttms), desc=args.dataset, unit="file") as progress:
        if workers == 1:
            diarizer = StreamingDiarizer(load_model(args))
            for audio_path, uri in todo:
                rttms[uri] = diarize(diarizer, audio_path, uri, args.output_dir)
                progress.update()
        elif workers > 1:
            pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(args,))
            try:
                futures = {
                    pool.submit(_diarize_in_worker, audio_path, uri, args.output_dir): uri
                    for audio_path, uri in todo
                }
                for future in as_completed(futures):
                    rttms[futures[future]] = future.result()
                    progress.update()
            finally:
                # On an error or Ctrl+C, drop the queued files instead of finishing them.
                pool.shutdown(cancel_futures=True)
    return rttms


def report(metrics: dict[str, DiarizationErrorRate]) -> None:
    """Print the accumulated (TOTAL) row of each metric's pyannote report."""
    totals = pd.DataFrame({name: m.report().loc["TOTAL"] for name, m in metrics.items()}).T
    print(
        totals.to_string(
            index=True, sparsify=False, justify="right", float_format=lambda f: f"{f:.2f}"
        )
    )


def main() -> None:
    args = parse_args()
    if not args.audio_dir.is_dir():
        raise SystemExit(f"audio directory not found: {args.audio_dir}")
    if not args.labels.exists():
        raise SystemExit(f"labels not found: {args.labels}")
    on_gpu = resolve_device(args).startswith("cuda")
    if on_gpu and args.workers not in (None, 1):
        raise SystemExit("on a GPU the files are diarized in a single process: drop --workers")
    if args.workers is None:
        args.workers = 1 if on_gpu else os.cpu_count()
    if args.workers < 1:
        raise SystemExit(f"--workers must be at least 1, got {args.workers}")
    items = DATASETS[args.dataset](args.audio_dir, args.labels)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    rttms = diarize_all(items, args)

    # No UEM is given, so pyannote scores the extent of reference + hypothesis.
    warnings.filterwarnings("ignore", message="'uem' was approximated")
    everything, few, many = (DiarizationErrorRate(skip_overlap=args.skip_overlap) for _ in range(3))
    for _, uri, reference in items:
        hypothesis = load_rttm(io.StringIO(rttms[uri])).get(uri, Annotation(uri=uri))
        everything(reference, hypothesis)
        (few if len(reference.labels()) <= MAX_FEW_SPEAKERS else many)(reference, hypothesis)

    metrics = {f"all ({len(everything.results_)} files)": everything}
    if many.results_:
        metrics = {
            f"<={MAX_FEW_SPEAKERS} speakers ({len(few.results_)} files)": few,
            f">={MAX_FEW_SPEAKERS + 1} speakers ({len(many.results_)} files)": many,
            **metrics,
        }
        metrics = {name: m for name, m in metrics.items() if m.results_}
    report(metrics)


if __name__ == "__main__":
    main()
