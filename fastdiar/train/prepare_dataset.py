"""Prepare the training protocols of ``configs/train.yaml``: VAD and filtering.

Usage:
    python -m fastdiar.train.prepare_dataset voxceleb \\
        --audio-dir voxceleb/vox2/dev/aac -o voxceleb/vox2/train_meta_vad.parquet
    python -m fastdiar.train.prepare_dataset libriheavymix \\
        --protocol LibriheavyMix/lsheavymix_cuts_medium.jsonl.gz \\
        --source-dir LibriheavyMix/src/medium_mtt -o LibriheavyMix/train_vad_protocol.parquet

Speech is detected with the repo's :class:`fastdiar.vad.StreamingVAD` (silero
with the diarizer's hysteresis), run over each whole file at once. Files are
processed in parallel, one per process (``-j``, default: one per CPU).

* ``voxceleb``: every audio file under ``--audio-dir`` (``<speaker>/<video>/<utt>``);
  the protocol has ``path``, ``spk_id`` and ``speech_segments``. Files without
  detected speech, or that fail to decode, are left out.
* ``libriheavymix``: the mixtures of the lhotse ``--protocol`` with 2 or 3
  speakers, where a single speaker is active in 20-90% of the mixture. VAD runs
  on every speaker's clean source (``<source-dir>/<id>/<i>.flac``, which starts
  at the speaker's offset in the mixture), and the mixture keeps the union; only
  mixtures with more than 5 s of speech are kept. The protocol has ``path``,
  ``num_speakers``, ``segments`` (``[offset, duration]`` per speaker) and
  ``speech_segments``.
"""

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from fastdiar.cli import load_audio
from fastdiar.diarizer import SAMPLE_RATE
from fastdiar.vad import StreamingVAD

AUDIO_EXTENSIONS = (".m4a", ".wav", ".flac")
# LibriHeavyMix filtering.
NUM_SPEAKERS = (2, 3)
SINGLE_SPEAKER_RATIO = (0.2, 0.9)  # exclusive bounds on the time share of one active speaker
MIN_SPEECH_SEC = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=("voxceleb", "libriheavymix"))
    parser.add_argument("--audio-dir", type=Path, help="voxceleb: the dataset's audio directory")
    parser.add_argument(
        "--protocol", type=Path, help="libriheavymix: lsheavymix_cuts_medium.jsonl.gz"
    )
    parser.add_argument(
        "--source-dir", type=Path, help="libriheavymix: the clean sources, <id>/<i>.flac"
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="output parquet file")
    parser.add_argument(
        "-j", "--workers", type=int, default=os.cpu_count(), help="processes (default: one per CPU)"
    )
    args = parser.parse_args()
    required = ["audio_dir"] if args.dataset == "voxceleb" else ["protocol", "source_dir"]
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"{args.dataset} requires --{name.replace('_', '-')}")
    if args.workers < 1:
        parser.error(f"--workers must be at least 1, got {args.workers}")
    return args


# VAD of a worker process, built once by its initializer.
_vad: StreamingVAD | None = None


def _init_worker() -> None:
    global _vad
    torch.set_num_threads(1)
    _vad = StreamingVAD(sr=SAMPLE_RATE)


def speech_segments(path: Path) -> list[list[float]]:
    """``[start, end]`` seconds of the speech in a file, with the VAD run over all of it."""
    _vad.reset()
    intervals = _vad(load_audio(path)) + _vad.flush()
    # A speech run is confirmed piece by piece: join the touching pieces.
    return [[start / SAMPLE_RATE, end / SAMPLE_RATE] for start, end in merge(intervals)]


def merge(intervals) -> list[list]:
    """Union of ``(start, end)`` intervals, sorted, with touching ones joined."""
    merged: list[list] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _voxceleb_file(path: Path) -> list[list[float]] | None:
    try:
        return speech_segments(path)
    except (RuntimeError, OSError) as e:  # a file that fails to decode is left out
        print(f"[skip] {path}: {e}")
        return None


def _libriheavymix_mixture(task: tuple[Path, list[list[float]]]) -> list[list[float]] | None:
    """Union of every source's speech, shifted to its offset in the mixture."""
    mixture_dir, segments = task
    try:
        return merge(
            [offset + start, offset + end]
            for i, (offset, _) in enumerate(segments)
            for start, end in speech_segments(mixture_dir / f"{i}.flac")
        )
    except (RuntimeError, OSError) as e:
        print(f"[skip] {mixture_dir}: {e}")
        return None


def run_vad(fn, tasks: list, workers: int, desc: str) -> list:
    """``fn`` over ``tasks`` in ``workers`` processes (in order)."""
    if workers == 1:
        _init_worker()
        return [fn(task) for task in tqdm(tasks, desc=desc, unit="file")]
    with ProcessPoolExecutor(workers, initializer=_init_worker) as pool:
        results = pool.map(fn, tasks, chunksize=16)
        return list(tqdm(results, total=len(tasks), desc=desc, unit="file"))


def prepare_voxceleb(audio_dir: Path, workers: int) -> pd.DataFrame:
    paths = sorted(
        Path(root) / name
        for root, _, names in os.walk(audio_dir, followlinks=True)
        for name in names
        if Path(name).suffix in AUDIO_EXTENSIONS
    )
    if not paths:
        raise SystemExit(f"no {'/'.join(AUDIO_EXTENSIONS)} files under {audio_dir}")
    segments = run_vad(_voxceleb_file, paths, workers, desc="voxceleb VAD")
    rows = [
        {"path": rel.as_posix(), "spk_id": rel.parts[0], "speech_segments": segs}
        for rel, segs in (
            (p.relative_to(audio_dir), s) for p, s in zip(paths, segments, strict=True)
        )
        if segs
    ]
    print(f"{len(rows)} of {len(paths)} files have speech")
    return pd.DataFrame(rows)


def read_mixtures(protocol: Path) -> pd.DataFrame:
    """The mixtures of a LibriHeavyMix lhotse cuts file that pass the speaker filters."""
    rows = []
    with gzip.open(protocol, "rt") as f:
        for line in tqdm(f, desc="protocol", unit="mixture"):
            cut = json.loads(line)
            segments, speakers = [], set()
            for track in cut["tracks"]:  # one track per speaker
                parts = track["cut"]["tracks"] if track["type"] == "MixedCut" else [track]
                speakers |= {
                    p["cut"]["supervisions"][0]["speaker"]
                    for p in parts
                    if p["cut"].get("supervisions")
                }
                duration = sum(p["cut"]["duration"] for p in parts)
                segments.append([round(track["offset"], 3), round(duration, 3)])
            if len(speakers) not in NUM_SPEAKERS:
                continue
            lo, hi = SINGLE_SPEAKER_RATIO
            if lo < single_speaker_ratio(segments) < hi:
                rows.append(
                    {
                        "path": f"{cut['id']}.flac",
                        "num_speakers": len(speakers),
                        "segments": segments,
                    }
                )
    return pd.DataFrame(rows)


def single_speaker_ratio(segments: list[list[float]]) -> float:
    """Share of the mixture where exactly one speaker is active (``[offset, duration]`` each)."""
    events = sorted([(o, 1) for o, _ in segments] + [(o + d, -1) for o, d in segments])
    single, active, prev = 0.0, 0, 0.0
    for time, change in events:
        if active == 1:
            single += time - prev
        active, prev = active + change, time
    return single / max(o + d for o, d in segments)


def prepare_libriheavymix(protocol: Path, source_dir: Path, workers: int) -> pd.DataFrame:
    mixtures = read_mixtures(protocol)
    print(
        f"{len(mixtures)} mixtures with {NUM_SPEAKERS} speakers and a single-speaker ratio in {SINGLE_SPEAKER_RATIO}"
    )
    tasks = [
        (source_dir / path.removesuffix(".flac"), segments)
        for path, segments in zip(mixtures["path"], mixtures["segments"], strict=True)
    ]
    mixtures["speech_segments"] = run_vad(
        _libriheavymix_mixture, tasks, workers, desc="libriheavymix VAD"
    )
    speech = mixtures["speech_segments"].map(
        lambda segs: -1.0 if segs is None else sum(end - start for start, end in segs)
    )
    mixtures = mixtures[speech > MIN_SPEECH_SEC].reset_index(drop=True)
    print(f"{len(mixtures)} mixtures with more than {MIN_SPEECH_SEC:g} s of speech")
    return mixtures


def main() -> None:
    args = parse_args()
    if args.dataset == "voxceleb":
        protocol = prepare_voxceleb(args.audio_dir, args.workers)
    else:
        protocol = prepare_libriheavymix(args.protocol, args.source_dir, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol.to_parquet(args.output, index=False)
    print(f"wrote {args.output} ({len(protocol)} rows)")


if __name__ == "__main__":
    main()
