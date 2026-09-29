"""VoxCeleb1 speaker verification (EER) of the ReDimNet2 models.

Usage:
    python -m fastdiar.test.eval_vox1 \\
        --audio-dir vox1/test/wav --protocol vox1/test/veri_test2.txt -m large
    python -m fastdiar.test.eval_vox1 \\
        --audio-dir vox1/test/wav --protocol vox1/test/veri_test2.txt \\
        -m large --enroll-model b6 --cache

Models (``-m``):

* ``b6`` -- redimnet2-b6-vb2+vox2_v0-lm, the released whole-utterance model
  (weights from torch hub): one embedding per utterance, which it takes whole.
* ``small`` / ``medium`` / ``large``, or a local ``.pt`` checkpoint -- the
  streaming encoder, fed in ``--shift-sec`` blocks (by default 60 s on a GPU and
  320 ms on the CPU, the fastest on each; the embeddings do not depend on it):
  one embedding per 80 ms frame. The first ``--skip-frames`` frames (12, i.e.
  960 ms, by default) of every utterance are left out of scoring.

A trial is scored by the cosine similarity of its two utterances, averaged over
all frame pairs for a streaming side. ``--enroll-model`` embeds the enrollment
side (the protocol's second column) with another model, e.g. ``b6`` templates
against streaming trials.

The protocol's paths are relative to ``--audio-dir``. With ``--cache`` the
embeddings are saved as float16 ``.npy`` files, next to the audio as
``<file-name>_<model-name>.npy`` or under ``--output-dir`` (mirroring the
``--audio-dir`` layout), and files that already have one are not embedded again.
The models run on a GPU when one is available (``--device``), in bfloat16, in a
single process; on the CPU, files are embedded in parallel, one process per CPU by
default (``--workers``).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from fastdiar.cli import add_device_args, load_audio, load_model, resolve_device, streaming_model
from fastdiar.model.redimnet2 import load_hub_model
from fastdiar.run_encoder import FileEncoder

HUB_MODEL = "b6-vb2+vox2_v0-lm"
HUB_URL = f"https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/{HUB_MODEL}.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="VoxCeleb1 test wav directory (the protocol's paths are relative to it)",
    )
    parser.add_argument(
        "--protocol", type=Path, required=True, help="trial list, e.g. veri_test2.txt"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=model_arg,
        default="large",
        help="tested model: b6, a streaming model size or a local .pt checkpoint (default: large)",
    )
    parser.add_argument(
        "--enroll-model",
        type=model_arg,
        default=None,
        help="model of the enrollment side (default: --model)",
    )
    parser.add_argument(
        "--cache", action="store_true", help="save / reuse float16 embeddings on disk"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="cache directory (default: next to audio)",
    )
    add_device_args(parser)
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="parallel embedding processes on the CPU (default: one per CPU); 1 runs "
        "in-process, as always on a GPU",
    )
    parser.add_argument(
        "--shift-sec",
        type=float,
        default=None,
        help="audio block fed to the streaming encoder, in seconds; it does not change the "
        "embeddings (default: 60 on a GPU, 0.32 on the CPU, the fastest on each)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=12,
        help="streaming frames left out of scoring at the start of every utterance "
        "(default: 12, i.e. 960 ms, before the encoder has context)",
    )
    return parser.parse_args()


def model_arg(value: str) -> str:
    return value if value == "b6" else streaming_model(value)


def model_name(model: str) -> str:
    """Name of a model in the cache file names."""
    return HUB_MODEL if model == "b6" else f"{Path(model).stem}-stream"


class UtteranceEncoder:
    """One L2-normalized float16 embedding per utterance, from the released b6 model.

    The whole utterance goes through the model in one pass; on a GPU under
    bfloat16 autocast (the log-mel front-end stays in fp32).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = load_hub_model(HUB_URL).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, wav: torch.Tensor) -> np.ndarray:
        on_gpu = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=on_gpu):
            emb = self.model(wav[None].to(self.device))
        emb = F.normalize(emb.float(), dim=-1)[0]
        return emb.cpu().numpy().astype(np.float16)


def build_encoder(model: str, args: argparse.Namespace):
    if model == "b6":
        return UtteranceEncoder(resolve_device(args))
    return FileEncoder(load_model(args, model), shift_sec=args.shift_sec)


def read_protocol(path: Path) -> tuple[list[int], list[str], list[str]]:
    """``(labels, enrollment files, trial files)`` of a ``<label> <enroll> <trial>`` list."""
    labels, enrolls, trials = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        label, enroll, trial = line.split()
        labels.append(int(label))
        enrolls.append(enroll)
        trials.append(trial)
    return labels, enrolls, trials


def cache_path(args: argparse.Namespace, model: str, rel: str) -> Path:
    """``<file-name>_<model-name>.npy`` next to the audio, or under ``--output-dir``."""
    path = (args.output_dir or args.audio_dir) / rel
    return path.with_name(f"{path.stem}_{model_name(model)}.npy")


def embed_file(encoder, model: str, rel: str, args: argparse.Namespace) -> np.ndarray:
    """Embed one file, saving the embedding to the cache with ``--cache``."""
    emb = encoder(load_audio(args.audio_dir / rel))
    if args.cache:
        path = cache_path(args, model, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, so an interrupted run never leaves a partial file.
        tmp_path = path.with_name(f"{path.name}.tmp")
        with open(tmp_path, "wb") as f:
            np.save(f, emb)
        tmp_path.replace(path)
    return emb


# Encoder of a worker process, built once by its initializer.
_worker_encoder = None


def _init_worker(model: str, args: argparse.Namespace) -> None:
    global _worker_encoder
    torch.set_num_threads(1)  # parallelism comes from the processes
    _worker_encoder = build_encoder(model, args)


def _embed_in_worker(model: str, rel: str, args: argparse.Namespace) -> np.ndarray:
    return embed_file(_worker_encoder, model, rel, args)


def embed(model: str, files: list[str], args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Embedding of every file: read from the cache, or extracted (and cached) now."""
    embeddings, todo = {}, []
    for rel in files:
        path = cache_path(args, model, rel)
        if args.cache and path.is_file():
            embeddings[rel] = np.load(path)
        else:
            todo.append(rel)

    workers = min(args.workers, len(todo))
    desc = model_name(model)
    with tqdm(total=len(files), initial=len(embeddings), desc=desc, unit="file") as progress:
        if workers == 1:
            encoder = build_encoder(model, args)
            for rel in todo:
                embeddings[rel] = embed_file(encoder, model, rel, args)
                progress.update()
        elif workers > 1:
            build_encoder(model, args)  # download once here, not in every worker at the same time
            pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(model, args))
            try:
                futures = {pool.submit(_embed_in_worker, model, rel, args): rel for rel in todo}
                for future in as_completed(futures):
                    embeddings[futures[future]] = future.result()
                    progress.update()
            finally:
                # On an error or Ctrl+C, drop the queued files instead of finishing them.
                pool.shutdown(cancel_futures=True)
    return embeddings


def utterance_vector(emb: np.ndarray, rel: str, skip_frames: int) -> np.ndarray:
    """Mean of an utterance's scored embeddings (a streaming side drops its first frames).

    The dot product of two such means equals the mean cosine similarity over
    all frame pairs, as every embedding is L2-normalized.
    """
    emb = emb.astype(np.float32)
    if emb.ndim == 1:
        return emb
    if emb.shape[0] <= skip_frames:
        raise ValueError(f"{rel}: {emb.shape[0]} frames, too short to skip {skip_frames}")
    return emb[skip_frames:].mean(axis=0)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """``(EER, threshold)``; a higher score means a target trial is more likely."""
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    far = np.cumsum(labels_sorted == 0) / (labels == 0).sum()
    frr = 1.0 - np.cumsum(labels_sorted == 1) / (labels == 1).sum()
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2), float(scores[order][idx])


def main() -> None:
    args = parse_args()
    if args.output_dir is not None and not args.cache:
        raise SystemExit("--output-dir is the cache directory, so it requires --cache")
    on_gpu = resolve_device(args).startswith("cuda")
    if on_gpu and args.workers not in (None, 1):
        raise SystemExit("on a GPU the files are embedded in a single process: drop --workers")
    if args.workers is None:
        args.workers = 1 if on_gpu else os.cpu_count()
    if args.workers < 1:
        raise SystemExit(f"--workers must be at least 1, got {args.workers}")
    if args.shift_sec is not None and args.shift_sec <= 0:
        raise SystemExit(f"--shift-sec must be positive, got {args.shift_sec}")
    if args.skip_frames < 0:
        raise SystemExit(f"--skip-frames must be non-negative, got {args.skip_frames}")
    enroll_model = args.enroll_model or args.model
    labels, enrolls, trials = read_protocol(args.protocol)
    missing = [
        rel for rel in sorted(set(enrolls) | set(trials)) if not (args.audio_dir / rel).is_file()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} protocol files not found under {args.audio_dir}, e.g. {missing[0]}"
        )

    if enroll_model == args.model:
        enroll_embs = trial_embs = embed(args.model, sorted(set(enrolls) | set(trials)), args)
    else:
        enroll_embs = embed(enroll_model, sorted(set(enrolls)), args)
        trial_embs = embed(args.model, sorted(set(trials)), args)
    enroll_vecs = {
        rel: utterance_vector(emb, rel, args.skip_frames) for rel, emb in enroll_embs.items()
    }
    trial_vecs = {
        rel: utterance_vector(emb, rel, args.skip_frames) for rel, emb in trial_embs.items()
    }

    pairs = tqdm(
        zip(enrolls, trials, strict=True),
        total=len(labels),
        desc="scoring",
        unit="pair",
    )
    scores = np.array([enroll_vecs[e] @ trial_vecs[t] for e, t in pairs], dtype=np.float64)
    label_arr = np.asarray(labels)
    eer, threshold = compute_eer(scores, label_arr)

    print(f"Enroll      : {model_name(enroll_model)}")
    print(f"Trial       : {model_name(args.model)}")
    print(f"Pairs       : {len(labels)} ({int(label_arr.sum())} target)")
    print(f"Threshold   : {threshold:.6f}")
    print(f"EER         : {eer * 100:.4f} %")


if __name__ == "__main__":
    main()
