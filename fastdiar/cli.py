"""Shared command-line helpers of the ``run_*`` scripts: arguments, audio and output paths."""

import argparse
from pathlib import Path
from urllib.error import URLError

import torch
import torchaudio

from fastdiar.diarizer import SAMPLE_RATE
from fastdiar.encoder import CHECKPOINTS, StreamingReDimNet2, load_streaming_model

MODEL_SIZES = tuple(CHECKPOINTS)


def build_parser(description: str, output_help: str) -> argparse.ArgumentParser:
    """Arguments common to every script: input, output and model."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="audio file, or directory to search recursively")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"{output_help} file or directory (a path without a suffix is a directory); "
        "defaults to next to each audio file",
    )
    parser.add_argument(
        "--ext", default=".wav", help="audio extension searched for in a directory input"
    )
    add_model_args(parser)
    return parser


def streaming_model(value: str) -> str:
    """A model size, or the path of an existing checkpoint file."""
    if value in MODEL_SIZES or Path(value).is_file():
        return value
    raise argparse.ArgumentTypeError(
        f"not a model size ({', '.join(MODEL_SIZES)}) or a checkpoint file: {value}"
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--model",
        type=streaming_model,
        default="large",
        help=f"model size ({', '.join(MODEL_SIZES)}; downloaded on first use) "
        "or a local .pt checkpoint (default: large)",
    )


def load_model(args: argparse.Namespace, model: str | None = None) -> StreamingReDimNet2:
    """Load the streaming ``model`` (default: ``args.model``)."""
    model = model or args.model
    try:
        return load_streaming_model(model)
    except URLError as e:
        raise SystemExit(
            f"could not download the {model} checkpoint: {e}\n"
            "pass a local checkpoint file with -m instead"
        ) from e


def load_audio(path: Path) -> torch.Tensor:
    """Read an audio file as a 16 kHz mono float32 waveform."""
    wav, sr = torchaudio.load(path)  # (channels, samples)
    wav = wav.mean(dim=0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def is_dir_path(path: Path) -> bool:
    """Whether an output path names a directory: an existing one, or one without a suffix."""
    return path.is_dir() or not path.suffix


def plan_jobs(
    input_path: Path, output: Path | None, ext: str, suffix: str
) -> list[tuple[Path, Path]]:
    """Pair every audio file to process with the ``suffix`` output path to write."""
    ext = ext if ext.startswith(".") else f".{ext}"
    if input_path.is_dir():
        if output is not None and not is_dir_path(output):
            raise SystemExit(f"output must be a directory when the input is one: {output}")
        files = sorted(p for p in input_path.rglob(f"*{ext}") if p.is_file())
        if not files:
            raise SystemExit(f"no *{ext} files found under {input_path}")
        if output is None:
            return [(f, f.with_suffix(suffix)) for f in files]
        return [(f, output / f.relative_to(input_path).with_suffix(suffix)) for f in files]

    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")
    if output is None:
        out_path = input_path.with_suffix(suffix)
    elif is_dir_path(output):
        out_path = output / f"{input_path.stem}{suffix}"
    else:
        out_path = output
    return [(input_path, out_path)]
